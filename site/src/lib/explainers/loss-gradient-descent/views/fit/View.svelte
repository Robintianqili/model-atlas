<!--
  'loss-gradient-descent/fit': the points, the line, the errors and the score (meta.ts has the props). Left: the plot,
  each error drawn from the point to the line (orange: the line is above the point, blue: below). Right (below on a
  phone): one bar per point, its squared error, and a dashed line at their mean, the error score.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { N, XS, YS, lineText } from '../../data.ts';
	import { fitOf } from './meta.ts';

	let { rt, source, at, props, setInput }: StageProps = $props();
	const pic = $derived(fitOf({ at, props }));

	// plot window (data units) and the SVG box
	const [X0, X1, Y0, Y1] = [-2, 2.5, -2.4, 3.4];
	const [W, H, M] = [340, 280, 26];
	const sx = (x: number) => M + ((x - X0) / (X1 - X0)) * (W - 2 * M);
	const sy = (y: number) => H - M - ((y - Y0) / (Y1 - Y0)) * (H - 2 * M);
	const lineY = (x: number) => pic!.w * x + pic!.b;
	const xt = [-2, -1, 0, 1, 2];
	const yt = [-2, -1, 0, 1, 2, 3];

	// bars of squared errors
	const [BW, BH, BM] = [220, 280, 26];
	const sqMax = $derived(pic ? Math.max(...pic.sq, pic.score, 1e-9) : 1);
	const by = (v: number) => BH - BM - (v / sqMax) * (BH - 2 * BM - 8);
	const bx = (i: number) => BM + ((i + 0.5) / N) * (BW - 2 * BM);
	const barW = ((BW - 2 * BM) / N) * 0.62;

	// the gradient-descent button: the step source's first update
	let busy = $state(false);
	async function step() {
		if (!setInput) return;
		busy = true;
		try {
			const i = await rt.ensure(source, { phase: 'update', step: 0 });
			const f = rt.sources[source]?.tape?.frame(i);
			const m = f?.metrics;
			if (m && typeof m.w1 === 'number' && typeof m.b1 === 'number') setInput({ w: m.w1, b: m.b1 });
		} finally {
			busy = false;
		}
	}
	const L = {
		plot: { en: 'The points and your line', zh: '数据点和你的直线' },
		bars: { en: 'Squared error of each point', zh: '每个点的误差平方' },
		mean: { en: 'mean = score', zh: '平均 = 分数' },
		step: { en: 'Take one gradient-descent step', zh: '走一步梯度下降' }
	};
</script>

{#if pic}
	<div class="fit" class:solo={props.squares === false}>
		<figure>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(L.plot)}>
				<clipPath id="lgd-fit-clip"><rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} /></clipPath>
				{#each xt as x (x)}
					<line class="grid" x1={sx(x)} x2={sx(x)} y1={M} y2={H - M} />
					<text class="tick" x={sx(x)} y={H - M + 14} text-anchor="middle">{x}</text>
				{/each}
				{#each yt as y (y)}
					<line class="grid" x1={M} x2={W - M} y1={sy(y)} y2={sy(y)} />
					<text class="tick" x={M - 6} y={sy(y) + 4} text-anchor="end">{y}</text>
				{/each}
				<line class="axis" x1={sx(0)} x2={sx(0)} y1={M} y2={H - M} />
				<line class="axis" x1={M} x2={W - M} y1={sy(0)} y2={sy(0)} />
				<g clip-path="url(#lgd-fit-clip)">
					{#each XS as x, i (i)}
						<line class="err" class:pos={pic.errs[i] > 0} x1={sx(x)} x2={sx(x)} y1={sy(YS[i])} y2={sy(lineY(x))} />
					{/each}
					<line class="fitline" x1={sx(X0)} x2={sx(X1)} y1={sy(lineY(X0))} y2={sy(lineY(X1))} />
				</g>
				{#each XS as x, i (i)}
					<circle class="pt" cx={sx(x)} cy={sy(YS[i])} r="4.5" />
					{#if pic.mark === i}<circle class="ring" cx={sx(x)} cy={sy(YS[i])} r="9" />{/if}
				{/each}
				<text class="lbl" x={W - M} y={M - 8} text-anchor="end">y = {lineText(pic.w, pic.b)}</text>
			</svg>
			<figcaption>{t(L.plot)}</figcaption>
		</figure>
		{#if props.squares !== false}
		<figure>
			<svg viewBox="0 0 {BW} {BH}" role="img" aria-label={t(L.bars)}>
				<line class="axis" x1={BM} x2={BW - BM} y1={BH - BM} y2={BH - BM} />
				{#each pic.sq as v, i (i)}
					<rect class="bar" class:pos={pic.errs[i] > 0} class:marked={pic.mark === i} x={bx(i) - barW / 2} y={by(v)} width={barW} height={Math.max(0.5, BH - BM - by(v))} />
					<text class="tick" x={bx(i)} y={BH - BM + 14} text-anchor="middle">{i}</text>
				{/each}
				<line class="mean" x1={BM} x2={BW - BM} y1={by(pic.score)} y2={by(pic.score)} />
				<line class="mean" x1={BM} x2={BM + 18} y1={10} y2={10} />
				<text class="lbl" x={BM + 22} y={14}>{t(L.mean)} {fmt3(pic.score)}</text>
			</svg>
			<figcaption>{t(L.bars)}</figcaption>
		</figure>
		{/if}
	</div>
	{#if props.stepButton && setInput}
		<p class="act"><button type="button" class="btn primary" onclick={step} disabled={busy}>{t(L.step)}</button></p>
	{/if}
{/if}

<style>
	.fit {
		display: grid;
		grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
		gap: 12px;
		container-type: inline-size;
	}
	.fit.solo {
		grid-template-columns: minmax(0, 1fr);
		max-width: 460px;
	}
	@media (max-width: 560px) {
		.fit {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
		max-height: 320px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
		text-align: center;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 1.2;
	}
	.tick {
		font-size: 10px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.lbl {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.pt {
		fill: var(--ink);
	}
	.ring {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 2.5;
	}
	.fitline {
		stroke: var(--accent);
		stroke-width: 2.5;
	}
	.err {
		stroke: var(--div-neg);
		stroke-width: 2;
		stroke-dasharray: 3 2;
	}
	.err.pos {
		stroke: var(--div-pos);
	}
	.bar {
		fill: var(--div-neg);
		opacity: 0.85;
	}
	.bar.pos {
		fill: var(--div-pos);
	}
	.bar.marked {
		stroke: var(--highlight);
		stroke-width: 2;
	}
	.mean {
		stroke: var(--ink);
		stroke-width: 1.5;
		stroke-dasharray: 5 3;
	}
	.act {
		margin: 8px 0 0;
	}
</style>
