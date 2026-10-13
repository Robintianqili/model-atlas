<!--
  The TinyGPT cover (redesign spec §5.6, step 0: answer first). The whole model on the learner's text before the first
  step: the last tokens of the text, the next-token guess with the five most likely tokens (real probabilities at
  temperature 1), and one strip from the tokens to the logits with one real tensor per stage (rows = the tokens; the
  outlined row is the last token, where the next token is predicted). Live: the same run the lesson walks through.
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { right } from '$lib/kit/icons';
	import { fmtInt, fmtPct, visibleToken } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import MiniHeat from './MiniHeat.svelte';
	import { topIndices } from './analysis';
	import { D, NL, V } from './state.svelte';
	import type { TGPage } from './page.svelte';

	let { P, compact = false }: { P: TGPage; compact?: boolean } = $props();
	const S = $derived(P.S);
	const f = $derived(S.run?.forward ?? null);
	const T = $derived(S.T);
	const probs = $derived(P.modelProbs);
	const top = $derived(probs ? topIndices(probs, compact ? 3 : 5) : []);
	const pMax = $derived(probs && top.length ? probs[top[0]] : 1);
	/** the last tokens of the window, the last one outlined (the next token is predicted after it) */
	const shown = $derived(compact ? 6 : 10);
	const tail = $derived(f && S.tok ? Array.from(f.ids.slice(Math.max(0, T - shown)), (id) => visibleToken(S.tok!.tokenToDisplay(id))) : []);
	const thumbW = $derived(compact ? 40 : 64);
	const thumbH = $derived(compact ? 40 : 64);
	const stages = $derived(
		f
			? [
					{ id: 'embed', label: { en: 'Embedding', zh: '嵌入' }, short: { en: 'Embed', zh: '嵌入' }, data: f.t.embed.data, rows: T, cols: D },
					...Array.from({ length: NL }, (_, b) => ({ id: `L${b}.resid2`, label: { en: `Block ${b + 1}`, zh: `第 ${b + 1} 块` }, short: { en: `B${b + 1}`, zh: `第${b + 1}块` }, data: f.t[`L${b}.resid2`].data, rows: T, cols: D }))
				]
			: []
	);
	/** the last token's 1,024 logits as a 32 × 32 picture */
	const logitsRow = $derived(f ? f.t.logits.data.subarray((T - 1) * V, T * V) : null);
</script>

{#if f && probs && top.length}
	<figure class="cov" class:compact>
		<div class="hero">
			<div class="txt">
				<p class="k">{t({ en: 'Your text ends with', zh: '你的文本结尾是' })}</p>
				<p class="toks" aria-label={t({ en: 'The last tokens of your text', zh: '你的文本的最后几个词元' })}>
					{#if T > shown}<span class="more" aria-hidden="true">…</span>{/if}{#each tail as tk, i (i)}<span class="tk" class:last={i === tail.length - 1}>{tk}</span>{/each}
				</p>
			</div>
			<div class="reads">
				<p class="k">{t({ en: 'It writes next', zh: '它接下来写' })}</p>
				<p class="a"><span class="w tokw">{S.idLabel(top[0])}</span> <span class="num p">{fmtPct(probs[top[0]])}</span></p>
				<p class="k2">{t({ en: `Most likely of ${fmtInt(V)} tokens`, zh: `${fmtInt(V)} 个词元中最可能的几个` })}</p>
				<ol class="bars" aria-label={t({ en: 'The most likely next tokens', zh: '最可能的下一个词元' })}>
					{#each top as id (id)}
						<li class:top={id === top[0]}>
							<span class="bl tokw">{S.idLabel(id)}</span>
							<span class="bt" aria-hidden="true"><span class="bf" style:width="{Math.max(0.6, (probs[id] / pMax) * 100)}%"></span></span>
							<span class="num bv">{fmtPct(probs[id])}</span>
						</li>
					{/each}
				</ol>
			</div>
		</div>
		<ol class="strip">
			<li class="st">
				<span class="pic tokpic num" style:width="{thumbW}px" style:height="{thumbH}px">{T}</span>
				<span class="sl">{t({ en: 'Tokens', zh: '词元' })}</span>
				<span class="ss">{t({ en: "ids", zh: "id" })}</span>
			</li>
			{#each stages as s (s.id)}
				<li class="arr" aria-hidden="true"><Icon d={right} size={16} /></li>
				<li class="st">
					<span class="pic"><MiniHeat data={s.data} rows={s.rows} cols={s.cols} width={thumbW} height={thumbH} hlRow={T - 1} label="{s.id} {s.rows}×{s.cols}" /></span>
					<span class="sl">{t(compact ? s.short : s.label)}</span>
					<span class="num ss">{s.rows}×{s.cols}</span>
				</li>
			{/each}
			<li class="arr" aria-hidden="true"><Icon d={right} size={16} /></li>
			<li class="st">
				<span class="pic">{#if logitsRow}<MiniHeat data={logitsRow} rows={32} cols={32} width={thumbW} height={thumbH} label="logits {fmtInt(V)}" />{/if}</span>
				<span class="sl">{t({ en: 'Logits', zh: 'Logits' })}</span>
				<span class="num ss">{fmtInt(V)}</span>
			</li>
		</ol>
		<figcaption class="cap">
			{t({
				en: `One real tensor per stage: a row for each of your ${T} tokens, the last one outlined; the logits are its ${fmtInt(V)} scores. The lesson opens each one.`,
				zh: `每一站展示一个真实的张量：你的 ${T} 个词元各占一行，最后一个带框；logits 是它的 ${fmtInt(V)} 个分数。课程会逐一打开它们。`
			})}
		</figcaption>
	</figure>
{:else}
	<p class="muted">{P.loadState === 'error' ? `${t({ en: 'The model could not be loaded:', zh: '模型加载失败：' })} ${P.loadError}` : t({ en: 'Loading TinyGPT…', zh: '正在加载 TinyGPT…' })}</p>
{/if}

<style>
	.cov {
		display: grid;
		gap: 22px;
		max-width: 760px;
		margin: 0 auto;
		font-variant-ligatures: none;
	}
	.hero {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: 20px 32px;
	}
	.txt {
		flex: 1 1 220px;
		min-width: 0;
	}
	.reads {
		flex: 1 1 260px;
		min-width: 0;
	}
	.hero p {
		margin: 0;
	}
	.k,
	.k2 {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.toks {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 8px !important;
	}
	.tk,
	.more {
		padding: 2px 6px;
		border-radius: var(--r1);
		background: var(--panel);
		box-shadow: inset 0 0 0 1px var(--rule);
		font-family: var(--font-mono);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		white-space: pre;
	}
	.more {
		background: none;
		box-shadow: none;
		color: var(--muted);
	}
	.tk.last {
		box-shadow: inset 0 0 0 2px var(--ink);
		font-weight: var(--w-ui);
	}
	.a {
		margin: 2px 0 14px !important;
	}
	.w {
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-display);
	}
	.tokw {
		font-family: var(--font-mono);
		white-space: pre;
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
		grid-template-columns: minmax(56px, 110px) minmax(80px, 1fr) 56px;
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
		text-overflow: ellipsis;
	}
	.bt {
		height: 10px;
		border-radius: var(--r-pill);
		background: var(--sunken);
	}
	.bf {
		display: block;
		height: 100%;
		border-radius: var(--r-pill);
		background: var(--seq-hi);
	}
	.bv {
		text-align: right;
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
		min-width: 56px;
	}
	.pic {
		display: grid;
		place-items: center;
		border-radius: var(--r1);
		outline: 1px solid var(--rule);
		overflow: hidden;
	}
	.tokpic {
		background: var(--well);
		font-size: var(--t-lg);
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
	/* compact (phones, small stage panes) */
	.compact {
		gap: 14px;
	}
	.compact .hero {
		gap: 12px;
	}
	.compact .w {
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.compact .strip {
		gap: 6px 2px;
		padding: 10px 6px;
	}
	.compact .st {
		min-width: 40px;
	}
	.compact .arr,
	.compact .ss {
		display: none;
	}
</style>
