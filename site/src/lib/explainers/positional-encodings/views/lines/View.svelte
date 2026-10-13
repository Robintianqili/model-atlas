<!--
  'positional-encodings/lines' (meta.ts): an SVG line chart with labelled axes, one categorical colour per line (named
  in the legend), gaps where a value does not exist, and one dashed vertical mark (the learner's p or k, or the trained
  length).
-->
<script lang="ts">
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { chartOf } from './meta.ts';

	let P: StageProps = $props();
	const c = $derived(chartOf(P));
	const [W, H] = [520, 270];
	const m = { l: 52, r: 14, t: 14, b: 42 };
	const dom = $derived.by(() => {
		if (!c) return null;
		const xs = c.lines.flatMap((l) => l.x);
		const ys = c.lines.flatMap((l) => l.y).filter(Number.isFinite);
		let [y0, y1] = ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
		if (y1 - y0 < 1e-6) [y0, y1] = [y0 - 1, y1 + 1];
		const pad = (y1 - y0) * 0.06;
		return { x0: 0, x1: Math.max(1, ...xs), y0: y0 - pad, y1: y1 + pad };
	});
	const X = (v: number) => (dom ? m.l + ((v - dom.x0) / (dom.x1 - dom.x0)) * (W - m.l - m.r) : 0);
	const Y = (v: number) => (dom ? m.t + (1 - (v - dom.y0) / (dom.y1 - dom.y0)) * (H - m.t - m.b) : 0);
	function ticks(a: number, b: number, n = 5) {
		const raw = (b - a) / n;
		const mag = 10 ** Math.floor(Math.log10(raw));
		const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
		const out: number[] = [];
		for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Number(v.toFixed(10)));
		return out;
	}
	/** a polyline path with a gap at every value that does not exist */
	function path(l: { x: number[]; y: number[] }) {
		let d = '';
		let pen = false;
		l.y.forEach((y, i) => {
			if (!Number.isFinite(y)) return void (pen = false);
			d += `${pen ? 'L' : 'M'}${X(l.x[i]).toFixed(1)} ${Y(y).toFixed(1)} `;
			pen = true;
		});
		return d;
	}
	const colors = $derived(c ? c.lines.map((_, i) => categorical(i, themeState.resolved)) : []);
	$effect(() => void (P.autoplay && P.onend()));
</script>

{#if c && dom}
	<figure class="lines">
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={`${t(c.yLabel)} / ${t(c.xLabel)}`}>
			<rect class="frame" x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} />
			{#each ticks(dom.x0, dom.x1, 6) as v (v)}
				<line class="grid" x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} />
				<text class="tick" x={X(v)} y={H - m.b + 14} text-anchor="middle">{v}</text>
			{/each}
			{#each ticks(dom.y0, dom.y1) as v (v)}
				<line class="grid" x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} />
				<text class="tick" x={m.l - 5} y={Y(v) + 4} text-anchor="end">{fmt3(v, 2)}</text>
			{/each}
			<text class="axl" x={m.l + (W - m.l - m.r) / 2} y={H - 6} text-anchor="middle">{t(c.xLabel)}</text>
			<text class="axl" transform="translate(12 {m.t + (H - m.t - m.b) / 2}) rotate(-90)" text-anchor="middle">{t(c.yLabel)}</text>
			{#if c.mark && c.mark.x >= dom.x0 && c.mark.x <= dom.x1}
				<line class="mark" x1={X(c.mark.x)} x2={X(c.mark.x)} y1={m.t} y2={H - m.b} />
				<text class="ml" x={X(c.mark.x) + 4} y={m.t + 12}>{t(c.mark.label)}</text>
			{/if}
			{#each c.lines as l, i (i)}
				<path d={path(l)} fill="none" stroke={colors[i]} stroke-width="2.2" stroke-linejoin="round" />
			{/each}
		</svg>
		<ul class="legend">
			{#each c.lines as l, i (i)}<li><i class="sw" style="background:{colors[i]}"></i>{t(l.label)}</li>{/each}
		</ul>
	</figure>
{/if}

<style>
	.lines { margin: 0; max-width: 620px; min-width: 0; }
	svg { width: 100%; height: auto; display: block; }
	.frame { fill: none; stroke: var(--rule-strong); }
	.grid { stroke: var(--rule); stroke-width: 1; }
	.tick { font-size: 10px; fill: var(--ink-2); font-family: var(--font-mono); }
	.axl { font-size: 11px; fill: var(--ink-2); }
	.mark { stroke: var(--ink-2); stroke-dasharray: 4 3; }
	.ml { font-size: 10.5px; fill: var(--ink); font-family: var(--font-mono); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
	.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 6px 0 0; padding: 0; list-style: none; font-size: var(--fs-sm); }
	.sw { display: inline-block; width: 14px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: 3px; }
</style>
