<!--
  adversarial-tinycnn Curve: one or two lines of shares (0 to 1) over x, with vertical marks (the learner's ε, the
  smallest flipping ε, the current PGD step). A small SVG; the numbers it shows are also in the view's text. Its width
  follows the box (300 to 560 units), so the 12 px labels stay 12 px on a phone; a mark in the right half has its label
  on its left, so it is never cut off at the edge.
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Line { y: ArrayLike<number>; label: Text; dashed?: boolean }
	interface Mark { x: number; label: Text }
	interface Props { xs: ArrayLike<number>; lines: Line[]; marks?: Mark[]; xLabel: Text; yLabel?: Text; label: Text; domain?: [number, number]; xTicks?: number[] }
	let { xs, lines, marks = [], xLabel, yLabel = { en: 'share', zh: '份额' }, label, domain, xTicks }: Props = $props();

	let cw = $state(0);
	const W = $derived(Math.round(Math.max(300, Math.min(560, cw || 560))));
	const H = 200, L = 40, R = 12, T = 12, B = 32;
	const lo = $derived(domain?.[0] ?? (xs.length ? xs[0] : 0)), hi = $derived(domain?.[1] ?? (xs.length ? xs[xs.length - 1] : 1));
	const sx = (x: number) => L + ((x - lo) / (hi - lo || 1)) * (W - L - R);
	const sy = (y: number) => T + (1 - y) * (H - T - B);
	const pts = (y: ArrayLike<number>) => Array.from(y, (v, i) => `${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
	const ticks = $derived(xTicks ?? [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo)));
	const right = (x: number) => sx(x) > L + 0.5 * (W - L - R);
</script>

<div class="cv" bind:clientWidth={cw}>
	<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(label)}>
		{#each [0, 0.5, 1] as v (v)}
			<line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} class="grid" /><text x={L - 6} y={sy(v) + 4} text-anchor="end">{v}</text>
		{/each}
		{#each ticks as v, i (v)}<text x={sx(v)} y={H - B + 16} text-anchor={i === ticks.length - 1 ? 'end' : 'middle'}>{Number(v.toFixed(3))}</text>{/each}
		{#each marks as m, i (i)}
			<line x1={sx(m.x)} x2={sx(m.x)} y1={T} y2={H - B} class="mark" />
			<text x={sx(m.x) + (right(m.x) ? -4 : 4)} y={T + 12 + i * 14} text-anchor={right(m.x) ? 'end' : 'start'} class="ml">{t(m.label)}</text>
		{/each}
		{#each lines as l, i (i)}<polyline points={pts(l.y)} class="ln" class:dashed={l.dashed} />{/each}
		<text x={W - R} y={H - 2} text-anchor="end" class="ml">{t(xLabel)}</text>
		<text transform="translate(11 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="ml">{t(yLabel)}</text>
	</svg>
</div>
<p class="key small">
	{#each lines as l, i (i)}<span class:dashed={l.dashed}>{l.dashed ? '- - ' : '— '}{t(l.label)}</span>{/each}
</p>

<style>
	.cv { min-width: 0; max-width: 560px; }
	svg { display: block; width: 100%; height: auto; }
	text { font-size: 12px; fill: var(--muted); }
	.grid { stroke: var(--rule); }
	.mark { stroke: var(--accent); stroke-dasharray: 3 3; }
	.ml { fill: var(--ink); }
	.ln { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.ln.dashed { stroke: var(--muted); stroke-dasharray: 6 4; }
	.key { display: flex; gap: 14px; flex-wrap: wrap; margin: 0; }
	.key .dashed { color: var(--muted); }
	.small { font-size: var(--fs-sm); }
</style>
