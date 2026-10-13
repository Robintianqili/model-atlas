<!--
  GELU curve with the real points: the 304 MLP pre-activations of the selected token (x) and
  their GELU outputs (y), as traced by the model. Hover, tap or use arrow keys to pick a unit.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { gelu } from './analysis';

	interface Props {
		xs: ArrayLike<number>;
		ys: ArrayLike<number>;
		unit?: number;
		label?: string;
	}
	let { xs, ys, unit = $bindable(0), label = '' }: Props = $props();

	const W = 520;
	const Hh = 230;
	const pad = { l: 40, r: 12, t: 12, b: 30 };
	const n = $derived(xs.length);
	const xmin = $derived(Math.min(-4, ...Array.from(xs)) - 0.2);
	const xmax = $derived(Math.max(3, ...Array.from(xs)) + 0.2);
	const ymin = -0.4;
	const ymax = $derived(Math.max(1, ...Array.from(ys)) + 0.2);
	const sx = (x: number) => pad.l + ((x - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
	const sy = (y: number) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * (Hh - pad.t - pad.b);
	const curve = $derived.by(() => {
		const pts: string[] = [];
		for (let i = 0; i <= 160; i++) {
			const x = xmin + ((xmax - xmin) * i) / 160;
			pts.push(`${sx(x).toFixed(1)},${sy(gelu(x)).toFixed(1)}`);
		}
		return pts.join(' ');
	});
	const relu = $derived(`${sx(xmin)},${sy(0)} ${sx(0)},${sy(0)} ${sx(Math.min(xmax, ymax))},${sy(Math.min(xmax, ymax))}`);
	const order = $derived(Array.from({ length: n }, (_, i) => i).sort((a, b) => xs[a] - xs[b]));
	const neg = $derived(Array.from(xs).filter((x) => x < 0).length);
	const ticksX = $derived([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6].filter((v) => v >= xmin && v <= xmax));
	const ticksY = $derived([0, 1, 2, 3, 4, 5].filter((v) => v <= ymax));
	let svg: SVGSVGElement | undefined = $state();

	function nearest(e: PointerEvent) {
		if (!svg) return;
		const r = svg.getBoundingClientRect();
		const x = ((e.clientX - r.left) / r.width) * W;
		const y = ((e.clientY - r.top) / r.height) * Hh;
		let best = 0;
		let bd = Infinity;
		for (let i = 0; i < n; i++) {
			const d = (sx(xs[i]) - x) ** 2 + (sy(ys[i]) - y) ** 2;
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		unit = best;
	}
	function onkeydown(e: KeyboardEvent) {
		const k = order.indexOf(unit);
		let j = k;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') j = Math.min(n - 1, k + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') j = Math.max(0, k - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		unit = order[j];
	}
</script>

<figure class="gp">
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (arrow keys pick a unit) -->
	<svg
		bind:this={svg}
		viewBox="0 0 {W} {Hh}"
		role="application"
		aria-roledescription="scatter plot"
		aria-label="{label} GELU: {n} {t({ en: 'points; arrow keys move along x', zh: '个点；方向键沿 x 移动' })}"
		tabindex="0"
		onpointermove={nearest}
		onpointerdown={nearest}
		{onkeydown}
	>
		<rect x={sx(xmin)} y={pad.t} width={sx(0) - sx(xmin)} height={Hh - pad.t - pad.b} class="negzone" />
		{#each ticksX as v (v)}
			<line x1={sx(v)} x2={sx(v)} y1={pad.t} y2={Hh - pad.b} class="grid" />
			<text x={sx(v)} y={Hh - pad.b + 16} class="tick" text-anchor="middle">{v}</text>
		{/each}
		{#each ticksY as v (v)}
			<line x1={pad.l} x2={W - pad.r} y1={sy(v)} y2={sy(v)} class="grid" />
			<text x={pad.l - 6} y={sy(v) + 4} class="tick" text-anchor="end">{v}</text>
		{/each}
		<line x1={pad.l} x2={W - pad.r} y1={sy(0)} y2={sy(0)} class="axis" />
		<line x1={sx(0)} x2={sx(0)} y1={pad.t} y2={Hh - pad.b} class="axis" />
		<polyline points={relu} class="relu" />
		<polyline points={curve} class="curve" />
		{#each Array.from({ length: n }, (_, i) => i) as i (i)}
			<circle cx={sx(xs[i])} cy={sy(ys[i])} r="2.6" class:neg={xs[i] < 0} class="pt" />
		{/each}
		{#if n}
			<circle cx={sx(xs[unit])} cy={sy(ys[unit])} r="6" class="sel" />
		{/if}
		<text x={W - pad.r} y={Hh - 4} class="axl" text-anchor="end">x ({t({ en: 'input', zh: '输入' })})</text>
		<text x={pad.l + 4} y={pad.t + 10} class="axl">GELU(x)</text>
	</svg>
	<figcaption class="num">
		{#if n}
			{t({ en: 'unit', zh: '单元' })} {unit}: x = {fmt3(xs[unit])} → GELU(x) = {fmt3(ys[unit])}
			<span class="muted">· {t({ en: `${neg} of ${n} inputs are negative and get squashed toward 0`, zh: `${n} 个输入中有 ${neg} 个为负，被压向 0` })}</span>
		{/if}
	</figcaption>
</figure>

<style>
	.gp {
		margin: 0;
		max-width: 560px;
	}
	svg {
		width: 100%;
		height: auto;
		touch-action: pan-y;
		cursor: crosshair;
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.negzone {
		fill: var(--surface-2);
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 0.6;
	}
	.axis {
		stroke: var(--ink-2);
		stroke-width: 1;
	}
	.tick,
	.axl {
		fill: var(--muted);
		font-size: var(--t-xs);
		font-family: var(--font-mono);
	}
	.relu {
		fill: none;
		stroke: var(--muted);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.curve {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.6;
	}
	.pt {
		fill: var(--div-pos);
		opacity: 0.7;
	}
	.pt.neg {
		fill: var(--div-neg);
	}
	.sel {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 2;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-top: 4px;
	}
</style>
