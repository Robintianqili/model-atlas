<!--
  Dense zoom: the logit of one class = sum over 256 inputs of weight × input, plus a bias
  (model.denseDetail, live). Shows which inputs push the class up (orange) or down (blue).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import { sym } from '$lib/kit/tex';
	import { categorical, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t, colon } from '$lib/i18n/i18n.svelte';
	import type { CnnStore, SelView } from './store.svelte';

	type Part = 'picker' | 'grids' | 'sums' | 'tops' | 'formula' | 'note';
	interface Props {
		S: CnnStore;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** blocks to show (default: all) */
		parts?: Part[];
	}
	let { S, view, parts = ['picker', 'grids', 'sums', 'tops', 'formula', 'note'] }: Props = $props();
	const V = $derived(view ?? S);
	const show = (p: Part) => parts.includes(p);

	const k = $derived(V.sel.layer === 'dense' || V.sel.layer === 'softmax' ? V.sel.ch : S.top);
	const dd = $derived(S.model && S.trace ? S.model.denseDetail(S.trace, k, 5) : null);
	const flat = $derived(S.trace?.t.flatten.data ?? null);
	const wAll = $derived(S.model?.weights['dense.weight'].data ?? null);
	const wRow = $derived(wAll ? wAll.subarray(k * 256, (k + 1) * 256) : null);
	const wDom = $derived.by<[number, number]>(() => {
		let m = 0;
		if (wAll) for (let i = 0; i < wAll.length; i++) m = Math.max(m, Math.abs(wAll[i]));
		return [-(m || 1), m || 1];
	});
	const xDom = $derived(S.domainFor('flatten', () => [-1, 1]));
	const pDom = $derived.by<[number, number]>(() => {
		let m = 0;
		if (dd) for (let i = 0; i < dd.products.length; i++) m = Math.max(m, Math.abs(dd.products[i]));
		return [-(m || 1), m || 1];
	});
	const sums = $derived.by(() => {
		let pos = 0;
		let neg = 0;
		if (dd) for (let i = 0; i < dd.products.length; i++) (dd.products[i] > 0 ? (pos += dd.products[i]) : (neg += dd.products[i]));
		return { pos, neg };
	});
	const csDom = $derived.by<[number, number]>(() => {
		let m = 0;
		if (dd) for (let i = 0; i < dd.channelSums.length; i++) m = Math.max(m, Math.abs(dd.channelSums[i]));
		return [-(m || 1), m || 1];
	});
	const csScale = $derived(makeScale('diverging', csDom, themeState.resolved));

	let hover = $state<number | null>(null);
	const hl = $derived(hover === null ? [] : [{ row: Math.floor(hover / 16), col: hover % 16, color: 'var(--accent)' }]);
	function setHover(i: number | null) {
		hover = i;
		if (i === null || !S.model) {
			S.hoverRF = null;
			return;
		}
		const p = S.model.flattenIndexToCHW(i);
		S.hoverRF = S.receptiveField('pool2', p.y, p.x);
	}
	const at = $derived(hover !== null && S.model ? S.model.flattenIndexToCHW(hover) : null);
	const name = (i: number) => t(S.spec.classes[i]);
</script>

{#if dd && flat && wRow}
	{#if show('picker')}
	<div class="pick" role="group" aria-label={t({ en: 'Class', zh: '类别' })}>
		<span class="muted">{t({ en: 'Logit of', zh: '计算以下类别的 logit' })}{colon()}</span>
		{#each S.spec.classes as c, i (c.id)}
			<button type="button" class="cb" class:on={i === k} aria-pressed={i === k} onclick={() => V.select(V.sel.layer === 'softmax' ? 'softmax' : 'dense', i)}>
				<span class="sw" style="background: {categorical(i, themeState.resolved)}"></span>{name(i)}
			</button>
		{/each}
	</div>
	{/if}

	{#if show('grids')}
	<div class="three">
		<figure>
			<figcaption>{t({ en: 'inputs x (flatten, 16 per row)', zh: '输入 x（展平，每行 16 个）' })}</figcaption>
			<TensorGrid data={flat} rows={16} cols={16} domain={xDom} maxCell={13} readout={false} highlights={hl} label="x" onhover={(e) => setHover(e ? e.row * 16 + e.col : null)} />
		</figure>
		<span class="op" aria-hidden="true">×</span>
		<figure>
			<figcaption>{t({ en: `weights for “${name(k)}”`, zh: `“${name(k)}”的权重` })}</figcaption>
			<RoleFrame role="param" label="dense.weight[{k}]">
				<TensorGrid data={wRow} rows={16} cols={16} domain={wDom} maxCell={13} readout={false} highlights={hl} label="W" onhover={(e) => setHover(e ? e.row * 16 + e.col : null)} />
			</RoleFrame>
		</figure>
		<span class="op" aria-hidden="true">=</span>
		<figure>
			<figcaption>{t({ en: 'contributions W·x', zh: '贡献 W·x' })}</figcaption>
			<TensorGrid data={dd.products} rows={16} cols={16} domain={pDom} maxCell={13} readout={false} highlights={hl} label="W x" onhover={(e) => setHover(e ? e.row * 16 + e.col : null)} />
		</figure>
	</div>
	<p class="hov">
		{#if hover !== null && at}
			<span class="num">i = {hover} ({t({ en: 'channel', zh: '通道' })} {at.c}, {t({ en: 'row', zh: '行' })} {at.y}, {t({ en: 'col', zh: '列' })} {at.x}):
			{fmt3(wRow[hover])} × {fmt3(flat[hover])} = {fmt3(dd.products[hover])}</span>
		{:else}
			<span class="muted">{view ? t({ en: 'Hover or tap a cell: weight × input = contribution.', zh: '悬停或点按格子：权重 × 输入 = 贡献。' }) : t({ en: 'Hover a cell: weight × input = contribution. The input view above outlines the 16×16 patch that number comes from.', zh: '把鼠标移到格子上：权重 × 输入 = 贡献。上方输入图会框出这个数来自的 16×16 区域。' })}</span>
		{/if}
	</p>
	{/if}

	{#if show('sums') || show('tops')}
	<div class="lower">
		{#if show('sums')}
		<div class="cs">
			<h4>{t({ en: 'Contribution per map of Max pool 2', zh: '按“最大池化 2”的每张图汇总的贡献' })}</h4>
			{#each Array.from(dd.channelSums) as v, c (c)}
				<div class="srow">
					<span class="sl num">c{c}</span>
					<span class="track"><span class="b {v < 0 ? 'neg' : 'pos'}" style="--w: {(Math.abs(v) / csDom[1]) * 50}%; background: {csScale.css(v)}"></span></span>
					<span class="sv num">{fmt3(v)}</span>
				</div>
			{/each}
		</div>
		{/if}
		{#if show('tops')}
		<div class="tops">
			<h4 class="up">{t({ en: `Pushes “${name(k)}” up most`, zh: `最推高“${name(k)}”的输入` })}</h4>
			<ul>
				{#each dd.topPositive as p (p.index)}
					{@const rf = S.receptiveField('pool2', p.y, p.x)}
					<li>
						<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={38} boxes={rf ? [{ ...rf, color: 'var(--div-pos)' }] : []} />
						<span class="num">i={p.index} (c{p.c}, {p.y},{p.x}) · {fmt3(p.weight)} × {fmt3(p.input)} = <strong class="pos">{fmt3(p.product)}</strong></span>
					</li>
				{/each}
			</ul>
			<h4 class="down">{t({ en: `Pushes “${name(k)}” down most`, zh: `最压低“${name(k)}”的输入` })}</h4>
			<ul>
				{#each dd.topNegative as p (p.index)}
					{@const rf = S.receptiveField('pool2', p.y, p.x)}
					<li>
						<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={38} boxes={rf ? [{ ...rf, color: 'var(--div-neg)' }] : []} />
						<span class="num">i={p.index} (c{p.c}, {p.y},{p.x}) · {fmt3(p.weight)} × {fmt3(p.input)} = <strong class="neg">{fmt3(p.product)}</strong></span>
					</li>
				{:else}
					<li class="muted">{t({ en: 'none', zh: '无' })}</li>
				{/each}
			</ul>
		</div>
		{/if}
	</div>
	{/if}

	{#if show('formula')}
	<FormulaCard
		tex={`\\begin{aligned} ${sym('out', 'z_{k}')} &= \\sum_{i=0}^{255} ${sym('param', 'W_{k,i}')}\\,${sym('in', 'x_i')} + ${sym('param', 'b_k')} \\\\ &= \\underbrace{{{pos}}}_{\\text{up}} + \\underbrace{({{neg}})}_{\\text{down}} + ({{b}}) \\\\ &= {{z}} \\end{aligned}`}
		values={{ pos: sums.pos, neg: sums.neg, b: dd.bias, z: dd.logit }}
		provenance="live"
		caption={t({ en: `Logit of “${name(k)}”`, zh: `“${name(k)}”的 logit` })}
	/>
	{/if}
	{#if show('note')}
	<p class="note">
		{t({
			en: `${dd.zeroInputs} of the 256 inputs are exactly 0 (switched off by ReLU), so they add nothing, whatever their weight.`,
			zh: `256 个输入中有 ${dd.zeroInputs} 个恰好为 0（被 ReLU 关掉了），无论权重多大，它们都没有贡献。`
		})}
	</p>
	{/if}
{/if}

<style>
	.pick {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		align-items: center;
		font-size: var(--fs-sm);
		margin-bottom: 10px;
	}
	.cb {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		min-height: 30px;
		padding: 0 10px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r-pill);
		background: var(--surface);
		font-size: var(--fs-sm);
	}
	.cb.on {
		border-color: var(--accent);
		background: var(--accent-soft);
		font-weight: 700;
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.three {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		flex: 1 1 150px;
		max-width: 230px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.op {
		font-size: var(--fs-lg);
		color: var(--muted);
	}
	.hov {
		margin: 6px 0 12px;
		font-size: var(--fs-xs);
		min-height: 1.4em;
	}
	.lower {
		display: grid;
		grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
		gap: 16px 28px;
	}
	@media (max-width: 760px) {
		.lower {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	h4 {
		font-size: var(--fs-sm);
		margin: 0 0 6px;
		font-family: var(--font-body);
		font-weight: 700;
	}
	.srow {
		display: grid;
		grid-template-columns: 2.6em 1fr 4.5em;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		min-height: 16px;
	}
	.sl {
		color: var(--muted);
	}
	.track {
		position: relative;
		height: 9px;
		background: linear-gradient(90deg, transparent calc(50% - 0.5px), var(--rule-strong) calc(50% - 0.5px), var(--rule-strong) calc(50% + 0.5px), transparent calc(50% + 0.5px));
	}
	.b {
		position: absolute;
		top: 1px;
		bottom: 1px;
		width: var(--w);
		border-radius: 2px;
	}
	.b.pos {
		left: 50%;
	}
	.b.neg {
		right: 50%;
	}
	.sv {
		text-align: right;
	}
	.tops ul {
		list-style: none;
		margin: 0 0 12px;
		padding: 0;
		display: grid;
		gap: 4px;
	}
	.tops li {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11.5px;
	}
	.pos {
		color: var(--div-pos-text);
	}
	.neg {
		color: var(--div-neg-text);
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>
