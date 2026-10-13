<!--
  The TinyCNN cover (redesign spec §5.6, step 0: answer first). The whole network on the learner's drawing before the
  first step: the 28×28 input, the answer with the four highest scores before softmax (the real logits on a zero line),
  and one strip from the pixels to the 10 scores with one real channel per stage and its shape. Live: every picture and
  number is the same forward pass the lesson walks through.
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { right } from '$lib/kit/icons';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import type { CnnStore } from './store.svelte';

	let { S, compact = false }: { S: CnnStore; compact?: boolean } = $props();

	const cls = (k: number) => t(S.spec.classes[k]);
	/** a softmax share is never exactly 100% (fp32 rounds 0.99999994 up to 1) */
	const pct = (p: number) => (p >= 0.9995 ? '>99.9%' : `${(p * 100).toFixed(1)}%`);
	const logits = $derived(S.trace?.t.dense.data ?? null);
	/** the four highest scores, highest first */
	const top4 = $derived.by(() => {
		const z = logits;
		if (!z) return [];
		return Array.from(z, (v, i) => ({ i, v }))
			.sort((a, b) => b.v - a.v)
			.slice(0, compact ? 3 : 4);
	});
	// one axis for the four bars: from the smaller of 0 and the lowest score to the larger of 0 and the highest
	const lo = $derived(Math.min(0, ...top4.map((s) => s.v)));
	const hi = $derived(Math.max(0, ...top4.map((s) => s.v)));
	const at = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;

	/** the strongest Conv 1-1 channel, followed through ReLU and pooling; then the strongest channel after block 2 */
	const c1 = $derived(S.trace ? S.strongestChannel('conv1_1') : 0);
	const c2 = $derived(S.trace ? S.strongestChannel('pool2') : 0);
	function slice(id: string, c: number) {
		const l = S.layer(id);
		const [, H, W] = l.outShape;
		const d = S.trace?.t[id]?.data;
		return d ? { data: d.subarray(c * H * W, (c + 1) * H * W), rows: H, cols: W, domain: S.domains?.layer[id] ?? ([-1, 1] as [number, number]) } : null;
	}
	/** C×H² (square maps) or C×H×W, from the spec */
	function shapeOf(id: string): string {
		const [C, H, W] = S.layer(id).outShape;
		return H === W ? `${C}×${H}²` : `${C}×${H}×${W}`;
	}
	const stages = $derived([
		{ id: 'conv1_1', label: { en: 'Conv 1-1', zh: '卷积 1-1' }, shape: shapeOf('conv1_1'), m: slice('conv1_1', c1), ch: c1 },
		{ id: 'relu1_1', label: { en: 'ReLU', zh: 'ReLU' }, shape: shapeOf('relu1_1'), m: slice('relu1_1', c1), ch: c1 },
		{ id: 'pool1', label: { en: 'Pool 1', zh: '池化 1' }, shape: shapeOf('pool1'), m: slice('pool1', c1), ch: c1 },
		{ id: 'pool2', label: { en: 'Block 2', zh: '第二组' }, shape: shapeOf('pool2'), m: slice('pool2', c2), ch: c2 }
	]);
	const zScale = $derived(logits ? makeScale('diverging', [Math.min(...logits), Math.max(...logits)], themeState.resolved) : null);
	const topV = $derived(top4[0]?.v ?? 0);
	const thumb = $derived(compact ? 50 : 68);
</script>

{#if S.trace && S.top >= 0}
	<figure class="cov" class:compact>
		<div class="hero">
			<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={compact ? 128 : 210} alt={t({ en: 'Your drawing as the model sees it: 28 × 28 numbers', zh: '模型看到的你的画：28 × 28 个数' })} />
			<div class="reads">
				<p class="k">{t({ en: 'It reads', zh: '它的答案' })}</p>
				<p class="a"><span class="w">{cls(S.top)}</span> <span class="num p">{pct(S.probs?.[S.top] ?? 0)}</span></p>
				<p class="k2">{t({ en: 'Scores before softmax', zh: 'softmax 之前的分数' })}</p>
				<ol class="bars" aria-label={t({ en: 'The four highest scores (logits)', zh: '最高的四个分数（logit）' })}>
					{#each top4 as s (s.i)}
						{@const a = at(Math.min(0, s.v))}
						{@const b = at(Math.max(0, s.v))}
						<li class:top={s.i === S.top}>
							<span class="bl">{cls(s.i)}</span>
							<span class="bt" aria-hidden="true">
								<span class="zero" style:left="{at(0)}%"></span>
								<span class="bf" class:neg={s.v < 0} style:left="{a}%" style:width="{Math.max(0.6, b - a)}%"></span>
							</span>
							<span class="num bv" class:neg={s.v < 0}>{fmt3(s.v)}</span>
						</li>
					{/each}
				</ol>
			</div>
		</div>
		<ol class="strip">
			<li class="st">
				<span class="pic"><MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={thumb} /></span>
				<span class="sl">{t({ en: 'Pixels', zh: '像素' })}</span>
				<span class="num ss">{S.layer('input').outShape.slice(-2).join('×')}</span>
			</li>
			{#each stages as s (s.id)}
				<li class="arr" aria-hidden="true"><Icon d={right} size={16} /></li>
				<li class="st">
					<span class="pic">{#if s.m}<MapCanvas data={s.m.data} rows={s.m.rows} cols={s.m.cols} kind="diverging" domain={s.m.domain} size={thumb} />{/if}</span>
					<span class="sl">{t(s.label)}</span>
					<span class="num ss">{s.shape}</span>
				</li>
			{/each}
			<li class="arr" aria-hidden="true"><Icon d={right} size={16} /></li>
			<li class="st">
				<span class="pic"><span class="score num" style:width="{thumb}px" style:height="{thumb}px" style:background={zScale?.css(topV)} style:color={zScale?.ink(topV)}>{fmt3(topV)}</span></span>
				<span class="sl">{t({ en: `${S.spec.classes.length} scores`, zh: `${S.spec.classes.length} 个分数` })}</span>
				<span class="num ss">{S.spec.classes.length}</span>
			</li>
		</ol>
		<figcaption class="cap">
			{t({
				en: `One real channel per stage (channel ${c1} of Conv 1-1, channel ${c2} after block 2), computed from your drawing. The lesson opens each one.`,
				zh: `每一站展示一个真实通道（卷积 1-1 的通道 ${c1}，第二组之后的通道 ${c2}），都由你的画算出。课程会逐一打开它们。`
			})}
		</figcaption>
	</figure>
{:else}
	<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
{/if}

<style>
	.cov {
		display: grid;
		gap: 22px;
		max-width: 760px;
		margin: 0 auto;
	}
	.hero {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 20px 28px;
	}
	.hero > :global(.mc) {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		overflow: hidden;
	}
	.reads {
		flex: 1 1 260px;
		min-width: 0;
	}
	.reads p {
		margin: 0;
	}
	.k,
	.k2 {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.a {
		margin: 2px 0 14px !important;
	}
	.w {
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-display);
		letter-spacing: -0.015em;
	}
	.p {
		color: var(--ink-2);
		font-size: var(--t-md);
	}
	.k2 {
		margin-bottom: 6px !important;
		font-size: var(--t-xs);
	}
	.bars {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.bars li {
		display: grid;
		grid-template-columns: minmax(64px, 110px) minmax(80px, 1fr) 56px;
		gap: 10px;
		align-items: center;
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.bars li.top {
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.bl {
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.bt {
		position: relative;
		height: 12px;
	}
	.zero {
		position: absolute;
		top: -3px;
		bottom: -3px;
		width: 1.5px;
		margin-left: -0.75px;
		background: var(--ink-2);
	}
	.bf {
		position: absolute;
		top: 0;
		bottom: 0;
		border-radius: 2px;
		background: var(--div-pos);
	}
	.bf.neg {
		background: var(--div-neg);
		opacity: 0.55;
	}
	.bv {
		text-align: right;
		color: var(--div-pos-text);
	}
	.bv.neg {
		color: var(--div-neg-text);
	}
	.strip {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: center;
		gap: 8px 6px;
		margin: 0;
		padding: 14px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		list-style: none;
	}
	.st {
		display: grid;
		justify-items: center;
		gap: 4px;
		min-width: 64px;
	}
	.pic {
		display: grid;
		place-items: center;
		border-radius: var(--r1);
		outline: 1px solid var(--rule);
		overflow: hidden;
	}
	.score {
		display: grid;
		place-items: center;
		font-size: var(--t-md);
		font-weight: var(--w-ui);
	}
	.sl {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		text-align: center;
		white-space: nowrap;
	}
	.ss {
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
	}
	.arr {
		align-self: center;
		margin-top: -30px;
		color: var(--faint);
	}
	.cap {
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		text-align: center;
	}
	/* compact (phones, small stage panes): the drawing beside the answer, three scores, a tight strip without shapes */
	.compact {
		gap: 14px;
	}
	.compact .hero {
		flex-wrap: nowrap;
		align-items: flex-start;
		gap: 14px;
	}
	.compact .reads {
		flex: 1 1 0;
	}
	.compact .w {
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.compact .bars li {
		grid-template-columns: minmax(40px, 72px) minmax(40px, 1fr) 44px;
		gap: 6px;
	}
	.compact .strip {
		gap: 6px 2px;
		padding: 10px 6px;
	}
	.compact .st {
		min-width: 50px;
	}
	.compact .arr,
	.compact .ss {
		display: none;
	}
</style>
