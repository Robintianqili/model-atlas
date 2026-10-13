<!--
  ReLU zoom: the conv map before and after max(0, x), on one diverging scale, linked by hover.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { sym } from '$lib/kit/tex';
	import { fmtPct, fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { channelDomain } from './data';
	import type { CnnStore, SelView } from './store.svelte';

	type Part = 'maps' | 'formula' | 'note';
	interface Props {
		S: CnnStore;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** blocks to show (default: all) */
		parts?: Part[];
	}
	let { S, view, parts = ['maps', 'formula', 'note'] }: Props = $props();
	const V = $derived(view ?? S);
	const show = (p: Part) => parts.includes(p);

	const L = $derived(S.layer(V.sel.layer));
	const inL = $derived(S.layer(L.in[0]));
	const H = $derived(L.outShape[1]);
	const W = $derived(L.outShape[2]);
	const hw = $derived(H * W);
	const before = $derived(S.trace ? S.trace.t[inL.id].data.subarray(V.sel.ch * hw, (V.sel.ch + 1) * hw) : null);
	const after = $derived(S.trace ? S.trace.t[L.id].data.subarray(V.sel.ch * hw, (V.sel.ch + 1) * hw) : null);
	// both maps share the conv layer's range, so a cell keeps its color unless ReLU changed it
	const dom = $derived(S.domainFor(inL.id, () => (S.trace ? channelDomain(S.trace.t[inL.id].data, V.sel.ch, hw) : [-1, 1])));
	const i = $derived(V.sel.y * W + V.sel.x);
	const v = $derived(before?.[i] ?? 0);
	const r = $derived(after?.[i] ?? 0);
	const zeroed = $derived.by(() => {
		if (!before) return 0;
		let n = 0;
		for (let k = 0; k < before.length; k++) if (before[k] < 0) n++;
		return n / before.length;
	});
	const hl = $derived([{ row: V.sel.y, col: V.sel.x, color: 'var(--accent)' }]);
	function hover(c: { row: number; col: number } | null) {
		if (!c) {
			S.hoverRF = null;
			return;
		}
		V.setCell(c.row, c.col);
		S.hoverRF = S.receptiveField(L.id, c.row, c.col);
	}
	// small plot of max(0, x) over the shared range
	const PW = 200;
	const PH = 110;
	const m = $derived(Math.max(Math.abs(dom[0]), Math.abs(dom[1])) || 1);
	const px = (x: number) => 10 + ((x + m) / (2 * m)) * (PW - 20);
	const py = (y: number) => PH - 14 - (Math.max(0, y) / m) * (PH - 28);
</script>

{#if before && after}
	{#if show('maps')}
	<div class="ba-wrap">
	<div class="ba">
		<figure>
			<figcaption>{t({ en: 'Before:', zh: '之前：' })} {t(inL.label)} · {t({ en: 'channel', zh: '通道' })} {V.sel.ch}</figcaption>
			<TensorGrid data={before} rows={H} cols={W} domain={dom} maxCell={14} highlights={hl} label={t(inL.label)} bind:cursor={() => ({ row: V.sel.y, col: V.sel.x }), (c) => { if (c) V.setCell(c.row, c.col); }} onhover={hover} />
		</figure>
		<div class="arrow" aria-hidden="true">max(0, ·) →</div>
		<figure>
			<figcaption>{t({ en: 'After:', zh: '之后：' })} {t(L.label)} · {t({ en: 'channel', zh: '通道' })} {V.sel.ch}</figcaption>
			<TensorGrid data={after} rows={H} cols={W} domain={dom} maxCell={14} highlights={hl} label={t(L.label)} bind:cursor={() => ({ row: V.sel.y, col: V.sel.x }), (c) => { if (c) V.setCell(c.row, c.col); }} onhover={hover} />
		</figure>
		<figure class="plot">
			<figcaption>ReLU</figcaption>
			<svg viewBox="0 0 {PW} {PH}" width={PW} height={PH} role="img" aria-label="ReLU: max(0, {fmt3(v)}) = {fmt3(r)}">
				<line x1="10" x2={PW - 10} y1={PH - 14} y2={PH - 14} stroke="var(--rule-strong)" />
				<line x1={px(0)} x2={px(0)} y1="8" y2={PH - 10} stroke="var(--rule-strong)" />
				<polyline points="{px(-m)},{py(-m)} {px(0)},{py(0)} {px(m)},{py(m)}" fill="none" stroke="var(--ink)" stroke-width="2" />
				<line x1={px(v)} x2={px(v)} y1={PH - 14} y2={py(v)} stroke="var(--accent)" stroke-dasharray="3 3" />
				<circle cx={px(v)} cy={py(v)} r="5" fill="var(--accent)" />
				<text x="10" y={PH - 2} class="ax">{fmt3(-m)}</text>
				<text x={PW - 10} y={PH - 2} class="ax" text-anchor="end">{fmt3(m)}</text>
			</svg>
			<ColorLegend domain={dom} width={180} />
		</figure>
	</div>
	</div>
	{/if}
	{#if show('formula')}
	<FormulaCard
		tex={`\\begin{aligned} ${sym('act', 'a_{ij}')} &= \\max(0, ${sym('out', 'y_{ij}')}) \\\\ &= \\max(0, {{v}}) = {{r}} \\end{aligned}`}
		values={{ v, r }}
		provenance="live"
		caption={t({ en: `Cell (${V.sel.y}, ${V.sel.x})`, zh: `格子 (${V.sel.y}, ${V.sel.x})` })}
	/>
	{/if}
	{#if show('note')}
	<p class="note">
		{t({
			en: `${fmtPct(zeroed)} of this map's cells were negative and became 0. Blue cells vanish; orange cells pass through unchanged. Without this bend, stacking conv layers would collapse into one big linear filter.`,
			zh: `这张图中 ${fmtPct(zeroed)} 的格子是负数，被变成了 0。蓝色格子消失，橙色格子原样通过。没有这个“折弯”，叠起来的卷积层会退化成一个大的线性滤波器。`
		})}
	</p>
	{/if}
{/if}

<style>
	/* before | arrow | after | plot; on narrow boxes the two maps stay side by side at equal size */
	.ba-wrap {
		container-type: inline-size;
	}
	.ba {
		display: grid;
		grid-template-columns: minmax(0, 360px) auto minmax(0, 360px) auto;
		gap: 12px 18px;
		align-items: center;
	}
	@container (max-width: 760px) {
		.ba {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
			gap: 10px 12px;
		}
		.arrow {
			display: none;
		}
		.plot {
			grid-column: 1 / -1;
		}
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.arrow {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.ax {
		font-family: var(--font-mono);
		font-size: var(--t-xs);
		fill: var(--muted);
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>
