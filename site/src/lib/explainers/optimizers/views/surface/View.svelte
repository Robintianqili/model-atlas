<!--
  'optimizers/surface' (meta.ts has the props): a canvas paints the loss of every setting of the two weights on the kit's
  sequential scale (log of the loss, so the valley floors show); an SVG on top draws the axes, the lowest points (crosses),
  each run's path (its categorical color, named in the legend) and where it is now. The glyph panel draws momentum's
  last step against the gradient-only step, or Adam's squared-gradient squares; formula: true adds the update formula.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { categorical, gradientCss, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import ViewSlot from '../../../../atlas/ui/ViewSlot.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import type { Surface } from '../../data.ts';
	import { inWindow, surfacePic } from './meta.ts';

	let stage: StageProps = $props();
	const pic = $derived(surfacePic(stage));
	const theme = $derived(themeState.resolved);
	// the surface object is a constant of data.ts, so this changes only with the landscape (not on every frame of a play)
	const surf = $derived(pic?.surface);

	const [W, H, M] = [360, 280, 32];
	const sx = (s: Surface, v: number) => M + ((v - s.xr[0]) / (s.xr[1] - s.xr[0])) * (W - 2 * M);
	const sy = (s: Surface, v: number) => H - M - ((v - s.yr[0]) / (s.yr[1] - s.yr[0])) * (H - 2 * M);
	const ticks = (r: [number, number]) => {
		const out: number[] = [];
		const step = r[1] - r[0] > 4 ? 1 : 0.5;
		for (let v = Math.ceil(r[0] / step) * step; v <= r[1] + 1e-9; v += step) out.push(Number(v.toFixed(2)));
		return out;
	};

	// the landscape: log10 of (loss + 0.001) on an 84 × 66 grid of cell centres, painted per surface and theme
	const [GX, GY] = [84, 66];
	let canvas = $state<HTMLCanvasElement | null>(null);
	$effect(() => {
		const cv = canvas;
		const s = surf;
		if (!cv || !s) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const v = new Float64Array(GX * GY);
		let [lo, hi] = [Infinity, -Infinity];
		for (let r = 0; r < GY; r++)
			for (let c = 0; c < GX; c++) {
				const x = s.xr[0] + ((c + 0.5) / GX) * (s.xr[1] - s.xr[0]);
				const y = s.yr[1] - ((r + 0.5) / GY) * (s.yr[1] - s.yr[0]);
				const f = Math.log10(s.f(x, y) + 1e-3);
				v[r * GX + c] = f;
				lo = Math.min(lo, f);
				hi = Math.max(hi, f);
			}
		const img = ctx.createImageData(GX, GY);
		const sc = makeScale('sequential', [lo, hi], theme);
		for (let i = 0; i < GX * GY; i++) sc.write(v[i], img.data, i * 4);
		ctx.putImageData(img, 0, 0);
	});
	const legend = $derived(gradientCss('sequential', theme));
	const poly = (s: Surface, pts: number[][]) => pts.map((q) => `${sx(s, q[0]).toFixed(1)},${sy(s, q[1]).toFixed(1)}`).join(' ');
	const pair = (v: number[]) => `(${fmt3(v[0])}, ${fmt3(v[1])})`;
	const id = $props.id();
	const L = {
		fig: { en: 'The loss for every setting of the two weights, with each run’s path', zh: '两个权重每种取值下的损失，以及每次运行的路径' },
		low: { en: 'low loss', zh: '损失低' },
		high: { en: 'high loss (log scale)', zh: '损失高（对数刻度）' },
		best: { en: 'lowest', zh: '最低点' },
		out: { en: 'outside the window', zh: '在窗口之外' },
		loss: { en: 'loss', zh: '损失' },
		mom: { en: 'Momentum’s last step (solid) and the step the gradient alone would take (dashed)', zh: '动量法的上一步（实线）和只用梯度时的一步（虚线）' },
		adam: { en: 'Adam’s squared-gradient squares: each side is √v̂ of one weight, so the area is v̂', zh: 'Adam 的梯度平方方块：边长是一个权重的 √v̂，所以面积是 v̂' }
	};
	// Adam's squares: the larger side is 64 px, both on one scale
	const sq = $derived.by(() => {
		const g = pic?.glyph;
		if (!g || g.kind !== 'adam') return null;
		const big = Math.max(g.b[0], g.b[1], 1e-12);
		return [0, 1].map((i) => ({ side: Math.max(1, (64 * g.b[i]) / big), sv: g.b[i], mh: g.a[i], step: g.step[i], name: i ? 'y' : 'x' }));
	});
</script>

{#if pic}
	{@const s = pic.surface}
	<div class="surf">
		<figure class="fig">
		<div class="box">
			<canvas bind:this={canvas} width={GX} height={GY} style:left="{(M / W) * 100}%" style:top="{(M / H) * 100}%" style:width="{((W - 2 * M) / W) * 100}%" style:height="{((H - 2 * M) / H) * 100}%" aria-hidden="true"></canvas>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(L.fig)}>
				<defs>
					<clipPath id="{id}-clip"><rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} /></clipPath>
					<marker id="{id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
						<path d="M0,0 L10,5 L0,10 z" class="ahead" />
					</marker>
				</defs>
				<rect class="frame" x={M} y={M} width={W - 2 * M} height={H - 2 * M} />
				{#each ticks(s.xr) as v (v)}<text class="tick" x={sx(s, v)} y={H - M + 13} text-anchor="middle">{v}</text>{/each}
				{#each ticks(s.yr) as v (v)}<text class="tick" x={M - 5} y={sy(s, v) + 4} text-anchor="end">{v}</text>{/each}
				<text class="axl" x={W - M} y={H - 3} text-anchor="end">{t(s.axes[0])} →</text>
				<text class="axl" x={4} y={M - 12}>↑ {t(s.axes[1])}</text>
				<g clip-path="url(#{id}-clip)">
					{#each s.min as q, i (i)}
						<g class="best">
							<line x1={sx(s, q[0]) - 6} x2={sx(s, q[0]) + 6} y1={sy(s, q[1]) - 6} y2={sy(s, q[1]) + 6} />
							<line x1={sx(s, q[0]) - 6} x2={sx(s, q[0]) + 6} y1={sy(s, q[1]) + 6} y2={sy(s, q[1]) - 6} />
						</g>
					{/each}
					{#each pic.runs as r (r.key)}
						{#if r.path.length > 1}
							<polyline class="halo" points={poly(s, r.path)} />
							<polyline class="path" class:dashed={r.dashed} points={poly(s, r.path)} style:stroke={categorical(r.color, theme)} />
						{/if}
					{/each}
					<circle class="start" cx={sx(s, pic.runs[0]?.path[0][0] ?? 0)} cy={sy(s, pic.runs[0]?.path[0][1] ?? 0)} r="5" />
					{#if pic.glyph}
						{@const g = pic.glyph}
						{#if g.plain}<line class="arrow plain" x1={sx(s, g.from[0])} y1={sy(s, g.from[1])} x2={sx(s, g.plain[0])} y2={sy(s, g.plain[1])} marker-end="url(#{id}-arrow)" />{/if}
						<line class="arrow" x1={sx(s, g.from[0])} y1={sy(s, g.from[1])} x2={sx(s, g.to[0])} y2={sy(s, g.to[1])} marker-end="url(#{id}-arrow)" />
					{/if}
					{#each pic.runs as r (r.key)}
						{#if r.inside}<circle class="now" cx={sx(s, r.now[0])} cy={sy(s, r.now[1])} r="5" style:fill={categorical(r.color, theme)} />{/if}
					{/each}
				</g>
				{#each s.min as q, i (i)}
					<text class="bestl" x={sx(s, q[0]) + 9} y={sy(s, q[1]) + (q[1] < 0 ? 14 : -7)}>{t(L.best)} 0</text>
				{/each}
			</svg>
		</div>
		<figcaption>
			<span class="leg"><span class="muted">{t(L.low)}</span><span class="bar" style:background={legend}></span><span class="muted">{t(L.high)}</span></span>
			<ul class="runs">
				{#each pic.runs as r (r.key)}
					<li>
						<span class="sw" class:dash={r.dashed} style:border-color={categorical(r.color, theme)}></span>
						<span>{t(r.name)}</span>
						<span class="num">{pair(r.now)} · {t(L.loss)} {fmt3(r.loss)}</span>
						{#if !r.inside}<span class="muted">({t(L.out)})</span>{/if}
					</li>
				{/each}
			</ul>
		</figcaption>
		</figure>
		{#if pic.glyph?.kind === 'momentum'}
			{@const g = pic.glyph}
			<div class="glyph">
				<p class="gcap">{t(L.mom)}</p>
				<p class="num">v = {pair(g.a)} · g = {pair(g.b)} · {t({ en: 'step', zh: '这一步' })} {pair(g.step)}</p>
			</div>
		{/if}
		{#if sq}
			<div class="glyph">
				<p class="gcap">{t(L.adam)}</p>
				<div class="squares">
					{#each sq as q (q.name)}
						<div class="sq">
							<svg width="70" height="70" viewBox="0 0 70 70" aria-hidden="true"><rect x={35 - q.side / 2} y={35 - q.side / 2} width={q.side} height={q.side} class="sqr" /></svg>
							<p class="num">{q.name}: √v̂ = {fmt3(q.sv)}<br />m̂ = {fmt3(q.mh)}<br />−η·m̂/√v̂ = {fmt3(q.step)}</p>
						</div>
					{/each}
				</div>
			</div>
		{/if}
		{#if stage.props?.formula}
			<ViewSlot view="zoom.formula" stage={{ ...stage, props: {} }} text={false} />
		{/if}
	</div>
{/if}

<style>
	.surf {
		margin: 0;
		display: grid;
		gap: 8px;
		max-width: 540px;
		min-width: 0;
	}
	.fig {
		margin: 0;
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.box {
		position: relative;
	}
	canvas {
		position: absolute;
		image-rendering: auto;
		opacity: 0.6;
	}
	svg {
		position: relative;
		width: 100%;
		height: auto;
		display: block;
	}
	.frame {
		fill: none;
		stroke: var(--rule-strong);
	}
	.tick {
		font-size: 10px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.axl {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.halo {
		fill: none;
		stroke: var(--surface);
		stroke-width: 4.5;
		stroke-linejoin: round;
	}
	.path {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.path.dashed {
		stroke-dasharray: 5 3;
	}
	.start {
		fill: var(--surface);
		stroke: var(--ink);
		stroke-width: 2;
	}
	.now {
		stroke: var(--surface);
		stroke-width: 1.5;
	}
	.best line {
		stroke: var(--ink);
		stroke-width: 2.5;
	}
	.bestl {
		font-size: 11px;
		fill: var(--ink);
		font-family: var(--font-mono);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 3px;
	}
	.arrow {
		stroke: var(--ink);
		stroke-width: 2.5;
	}
	.arrow.plain {
		stroke-dasharray: 4 3;
		stroke-width: 1.8;
	}
	.ahead {
		fill: var(--ink);
	}
	figcaption {
		display: grid;
		gap: 4px;
		font-size: var(--fs-xs);
	}
	.leg {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.bar {
		display: inline-block;
		width: 90px;
		height: 8px;
		border-radius: 4px;
		opacity: 0.6;
	}
	.runs {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.runs li {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
	}
	.sw {
		display: inline-block;
		width: 18px;
		border-top: 3px solid;
	}
	.sw.dash {
		border-top-style: dashed;
	}
	.glyph {
		border-top: 1px solid var(--rule);
		padding-top: 6px;
		font-size: var(--fs-xs);
		display: grid;
		gap: 4px;
	}
	.glyph p {
		margin: 0;
	}
	.gcap {
		color: var(--ink-2);
	}
	.squares {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
	}
	.sq {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.sq p {
		white-space: nowrap;
	}
	.sqr {
		fill: var(--div-pos);
		stroke: var(--ink);
		stroke-width: 1;
	}
</style>
