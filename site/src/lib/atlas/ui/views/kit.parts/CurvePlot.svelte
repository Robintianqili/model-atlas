<!--
  An elementwise function as a curve, with the real points of one layer on it (every input x and its output y, up to a
  sample) and the selected unit circled (as TinyGPT's GeluPlot). The curve itself is computed by the op's own forward()
  on a grid of x values, so it is the function the model ran. Hover, tap or arrow keys pick another point (onpick).
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../core/format.ts';

	interface Props {
		/** the curve: x grid and f(x) */
		cx: ArrayLike<number>;
		cy: ArrayLike<number>;
		/** points of the layer (x in, y out) and their unit index in the layer */
		px: ArrayLike<number>;
		py: ArrayLike<number>;
		/** index into px/py of the selected point */
		sel: number;
		name: string;
		label: Text;
		onpick?: (i: number) => void;
		/** a dashed reference curve (e.g. ReLU next to GELU) */
		ref?: { x: ArrayLike<number>; y: ArrayLike<number>; label: string } | null;
	}
	let { cx, cy, px, py, sel, name, label, onpick, ref = null }: Props = $props();

	const W = 520;
	const Hh = 240;
	const pad = { l: 40, r: 12, t: 12, b: 30 };
	const fin = (a: ArrayLike<number>) => Array.from(a).filter((v) => Number.isFinite(v));
	const xr = $derived.by(() => {
		const xs = fin(cx);
		return [Math.min(...xs), Math.max(...xs)] as [number, number];
	});
	const yr = $derived.by(() => {
		const ys = [...fin(cy), ...fin(py)];
		let lo = Math.min(0, ...ys);
		let hi = Math.max(0, ...ys);
		if (hi - lo < 1e-9) hi = lo + 1;
		const padY = (hi - lo) * 0.06;
		return [lo - padY, hi + padY] as [number, number];
	});
	const sx = (x: number) => pad.l + ((x - xr[0]) / (xr[1] - xr[0] || 1)) * (W - pad.l - pad.r);
	const sy = (y: number) => pad.t + (1 - (y - yr[0]) / (yr[1] - yr[0] || 1)) * (Hh - pad.t - pad.b);
	const poly = (xs: ArrayLike<number>, ys: ArrayLike<number>) => {
		const out: string[] = [];
		for (let i = 0; i < xs.length; i++) if (Number.isFinite(ys[i])) out.push(`${sx(xs[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`);
		return out.join(' ');
	};
	const curve = $derived(poly(cx, cy));
	const refLine = $derived(ref ? poly(ref.x, ref.y) : '');
	const ticks = (lo: number, hi: number) => {
		const span = hi - lo;
		const step = span > 40 ? 10 : span > 16 ? 4 : span > 8 ? 2 : span > 3 ? 1 : span > 1.2 ? 0.5 : 0.25;
		const out: number[] = [];
		for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
		return out;
	};
	const tx = $derived(ticks(xr[0], xr[1]));
	const ty = $derived(ticks(yr[0], yr[1]));
	const n = $derived(px.length);
	const order = $derived(Array.from({ length: n }, (_, i) => i).sort((a, b) => px[a] - px[b]));
	const inRange = (x: number) => x >= xr[0] && x <= xr[1];
	function nearest(e: PointerEvent) {
		if (!onpick || !n) return;
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = ((e.clientX - r.left) / r.width) * W;
		const y = ((e.clientY - r.top) / r.height) * Hh;
		let best = 0;
		let bd = Infinity;
		for (let i = 0; i < n; i++) {
			const d = (sx(Math.max(xr[0], Math.min(xr[1], px[i]))) - x) ** 2 + (sy(py[i]) - y) ** 2;
			if (d < bd) (bd = d), (best = i);
		}
		onpick(best);
	}
	function onkeydown(e: KeyboardEvent) {
		if (!onpick || !n) return;
		const k = order.indexOf(sel);
		let j = k;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') j = Math.min(n - 1, k + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') j = Math.max(0, k - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		onpick(order[j]);
	}
</script>

<figure class="cp">
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (arrow keys pick a point) -->
	<svg
		viewBox="0 0 {W} {Hh}"
		role="application"
		aria-roledescription={t({ en: 'curve with points', zh: '带数据点的曲线' })}
		aria-label="{t(label)}: {n} {t({ en: 'points; arrow keys move along x', zh: '个点；方向键沿 x 移动' })}"
		tabindex="0"
		onpointermove={nearest}
		onpointerdown={nearest}
		{onkeydown}
	>
		{#if xr[0] < 0}<rect x={sx(xr[0])} y={pad.t} width={Math.max(0, sx(Math.min(0, xr[1])) - sx(xr[0]))} height={Hh - pad.t - pad.b} class="negzone" />{/if}
		{#each tx as v (v)}
			<line x1={sx(v)} x2={sx(v)} y1={pad.t} y2={Hh - pad.b} class="grid" />
			<text x={sx(v)} y={Hh - pad.b + 16} class="tick" text-anchor="middle">{fmt3(v)}</text>
		{/each}
		{#each ty as v (v)}
			<line x1={pad.l} x2={W - pad.r} y1={sy(v)} y2={sy(v)} class="grid" />
			<text x={pad.l - 6} y={sy(v) + 4} class="tick" text-anchor="end">{fmt3(v)}</text>
		{/each}
		{#if yr[0] <= 0 && yr[1] >= 0}<line x1={pad.l} x2={W - pad.r} y1={sy(0)} y2={sy(0)} class="axis" />{/if}
		{#if xr[0] <= 0 && xr[1] >= 0}<line x1={sx(0)} x2={sx(0)} y1={pad.t} y2={Hh - pad.b} class="axis" />{/if}
		{#if refLine}<polyline points={refLine} class="ref" />{/if}
		<polyline points={curve} class="curve" />
		{#each Array.from({ length: n }, (_, i) => i) as i (i)}
			{#if inRange(px[i])}<circle cx={sx(px[i])} cy={sy(py[i])} r="2.4" class="pt" class:neg={px[i] < 0} />{/if}
		{/each}
		{#if n && sel < n}
			<circle cx={sx(Math.max(xr[0], Math.min(xr[1], px[sel])))} cy={sy(py[sel])} r="6.5" class="sel" />
		{/if}
		<text x={W - pad.r} y={Hh - 4} class="axl" text-anchor="end">x ({t({ en: 'input', zh: '输入' })})</text>
		<text x={pad.l + 4} y={pad.t + 10} class="axl">{name}(x)</text>
		{#if ref}<text x={pad.l + 4} y={pad.t + 24} class="axl refl">- - {ref.label}</text>{/if}
	</svg>
</figure>

<style>
	.cp {
		margin: 0;
		max-width: 560px;
		min-width: 0;
	}
	svg {
		display: block;
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
	.refl {
		fill: var(--muted);
	}
	.ref {
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
		opacity: 0.65;
	}
	.pt.neg {
		fill: var(--div-neg);
	}
	.sel {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 2.4;
	}
</style>
