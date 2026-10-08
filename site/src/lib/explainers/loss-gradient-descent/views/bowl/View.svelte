<!--
  'loss-gradient-descent/bowl': the error landscape over (w, b) (meta.ts has the props). A canvas paints the score of
  every dial setting on the kit's sequential scale (log of the score, so the bottom of the bowl is visible); an SVG on
  top draws the axes, the lowest point (a cross), the path so far, the dials now (a ring) and the update's step (an
  arrow). Other runs' paths are dashed.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { gradientCss, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { B_RANGE, W_RANGE, scoreOf } from '../../data.ts';
	import { bowlOf } from './meta.ts';

	let { rt, at, props }: StageProps = $props();
	const pic = $derived(bowlOf({ rt, at, props }));

	const [W, H, M] = [360, 300, 30];
	const sx = (w: number) => M + ((w - W_RANGE[0]) / (W_RANGE[1] - W_RANGE[0])) * (W - 2 * M);
	const sy = (b: number) => H - M - ((b - B_RANGE[0]) / (B_RANGE[1] - B_RANGE[0])) * (H - 2 * M);
	const wt = [-2, -1, 0, 1, 2, 3, 4];
	const bt = [-2, -1, 0, 1, 2, 3];

	// the landscape: log10 of the score on a 72 × 60 grid (cell centres), painted once per theme
	const [GX, GY] = [72, 60];
	const field = (() => {
		const v = new Float64Array(GX * GY);
		let [lo, hi] = [Infinity, -Infinity];
		for (let r = 0; r < GY; r++)
			for (let c = 0; c < GX; c++) {
				const w = W_RANGE[0] + ((c + 0.5) / GX) * (W_RANGE[1] - W_RANGE[0]);
				const b = B_RANGE[1] - ((r + 0.5) / GY) * (B_RANGE[1] - B_RANGE[0]);
				const x = Math.log10(scoreOf(w, b));
				v[r * GX + c] = x;
				lo = Math.min(lo, x);
				hi = Math.max(hi, x);
			}
		return { v, lo, hi };
	})();
	let canvas = $state<HTMLCanvasElement | null>(null);
	$effect(() => {
		const cv = canvas;
		const theme = themeState.resolved;
		if (!cv) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const img = ctx.createImageData(GX, GY);
		const sc = makeScale('sequential', [field.lo, field.hi], theme);
		for (let i = 0; i < GX * GY; i++) sc.write(field.v[i], img.data, i * 4);
		ctx.putImageData(img, 0, 0);
	});
	const legend = $derived(gradientCss('sequential', themeState.resolved));
	const poly = (pts: number[][]) => pts.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ');
	const L = {
		fig: { en: 'The error score for every setting of the two dials', zh: '两个旋钮每种设置下的误差分数' },
		low: { en: 'low score', zh: '分数低' },
		high: { en: 'high score (log scale)', zh: '分数高（对数刻度）' },
		best: { en: 'lowest', zh: '最低点' },
		w: { en: 'weight w', zh: '权重 w' },
		b: { en: 'bias b', zh: '偏置 b' }
	};
</script>

{#if pic}
	<figure class="bowl">
		<div class="box">
			<canvas bind:this={canvas} width={GX} height={GY} style:left="{(M / W) * 100}%" style:top="{(M / H) * 100}%" style:width="{((W - 2 * M) / W) * 100}%" style:height="{((H - 2 * M) / H) * 100}%" aria-hidden="true"></canvas>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(L.fig)}>
				<defs>
					<clipPath id="lgd-bowl-clip"><rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} /></clipPath>
					<marker id="lgd-bowl-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">
						<path d="M0,0 L10,5 L0,10 z" class="ahead" />
					</marker>
				</defs>
				<rect class="frame" x={M} y={M} width={W - 2 * M} height={H - 2 * M} />
				{#each wt as w (w)}<text class="tick" x={sx(w)} y={H - M + 13} text-anchor="middle">{w}</text>{/each}
				{#each bt as b (b)}<text class="tick" x={M - 6} y={sy(b) + 4} text-anchor="end">{b}</text>{/each}
				<text class="axl" x={W - M} y={H - 4} text-anchor="end">{t(L.w)} →</text>
				<text class="axl" x={4} y={M - 10}>↑ {t(L.b)}</text>
				<g clip-path="url(#lgd-bowl-clip)">
					{#each pic.others as o, k (k)}
						{#if o.path.length > 1}<polyline class="other" points={poly(o.path)} />{/if}
					{/each}
					{#if pic.path.length > 1}<polyline class="path" points={poly(pic.path)} />{/if}
					{#each pic.path.slice(0, -1) as q, i (i)}<circle class="dot" cx={sx(q[0])} cy={sy(q[1])} r="2.5" />{/each}
					<g class="best">
						<line x1={sx(pic.best.w) - 6} x2={sx(pic.best.w) + 6} y1={sy(pic.best.b) - 6} y2={sy(pic.best.b) + 6} />
						<line x1={sx(pic.best.w) - 6} x2={sx(pic.best.w) + 6} y1={sy(pic.best.b) + 6} y2={sy(pic.best.b) - 6} />
					</g>
					{#if pic.arrow}
						<line class="arrow" x1={sx(pic.w)} y1={sy(pic.b)} x2={sx(pic.w + pic.arrow.dw)} y2={sy(pic.b + pic.arrow.db)} marker-end="url(#lgd-bowl-arrow)" />
					{/if}
					<circle class="now" cx={sx(pic.w)} cy={sy(pic.b)} r="6" />
				</g>
				<text class="bestl" x={sx(pic.best.w) + 9} y={sy(pic.best.b) + 4}>{t(L.best)} {fmt3(pic.best.score)}</text>
			</svg>
		</div>
		<figcaption>
			<span class="leg"><span class="muted">{t(L.low)}</span><span class="bar" style:background={legend}></span><span class="muted">{t(L.high)}</span></span>
			<span class="num">w = {fmt3(pic.w)}, b = {fmt3(pic.b)}: {fmt3(pic.score)}</span>
		</figcaption>
	</figure>
{/if}

<style>
	.bowl {
		margin: 0;
		display: grid;
		gap: 6px;
		max-width: 520px;
	}
	.box {
		position: relative;
	}
	canvas {
		position: absolute;
		image-rendering: auto;
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
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axl {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.path {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.8;
	}
	.other {
		fill: none;
		stroke: var(--div-pos);
		stroke-width: 1.8;
		stroke-dasharray: 4 3;
	}
	.dot {
		fill: var(--ink);
	}
	.now {
		fill: var(--accent);
		stroke: var(--surface);
		stroke-width: 2;
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
		stroke: var(--div-pos);
		stroke-width: 2.5;
	}
	.ahead {
		fill: var(--div-pos);
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 4px 12px;
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
	}
</style>
