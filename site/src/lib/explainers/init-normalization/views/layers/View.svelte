<!--
  'init-normalization/layers': one row per hidden layer (meta.ts has the props). Each panel (activations, then the
  gradient that flows back) draws, per layer, a histogram on one x range for every row (each row scaled to its tallest
  bar) and the layer's spread as a bar on a log axis shared by the rows, with its value.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { BINS, DEPTH, stdText } from '../../data.ts';
	import { layersOf, type Panel } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(layersOf({ at, props }));

	const [W, RH, TOP] = [330, 21, 34];
	const H = TOP + DEPTH * RH + 26;
	const AY = TOP + DEPTH * RH + 14;
	const [LX, HX, HW, BX, BW] = [0, 52, 120, 182, 84];
	const minus = (s: string) => s.replace(/-/g, '−');

	function logDomain(p: Panel): [number, number] {
		const pos = p.spread.filter((v) => v > 0 && Number.isFinite(v));
		if (!pos.length) return [-1, 0];
		let lo = Math.floor(Math.log10(Math.min(...pos)));
		let hi = Math.ceil(Math.log10(Math.max(...pos)));
		if (hi - lo < 1) hi = lo + 1;
		return [lo, hi];
	}
	function bars(h: number[]) {
		const top = Math.max(...h, 1);
		return h.map((c, k) => ({ x: HX + (k / BINS) * HW, w: HW / BINS, h: (c / top) * (RH - 5) }));
	}
	const L = {
		act: { en: 'Activations', zh: '激活值' },
		grad: { en: 'Gradient dL/dz flowing back', zh: '传回来的梯度 dL/dz' },
		hist: { en: 'histogram', zh: '直方图' },
		spread: { en: 'spread (log scale)', zh: '标准差（对数刻度）' },
		layer: { en: 'layer', zh: '层' },
		zero: { en: 'all 0', zh: '全为 0' }
	};
</script>

{#snippet panel(p: Panel, title: { en: string; zh: string }, kind: string)}
	{@const dom = logDomain(p)}
	{@const sx = (v: number) => BX + ((Math.log10(v) - dom[0]) / (dom[1] - dom[0])) * BW}
	<figure class="panel {kind}">
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(title)}>
			<text class="ttl" x={LX} y="12">{t(title)}</text>
			<text class="sub" x={HX} y="28">{t(L.hist)}</text>
			<text class="sub" x={BX} y="28">{t(L.spread)}</text>
			{#each p.hist as h, l (l)}
				{@const y = TOP + l * RH}
				<text class="lab" x={LX} y={y + RH - 7}>{t(L.layer)} {l + 1}</text>
				<line class="base" x1={HX} x2={HX + HW} y1={y + RH - 3} y2={y + RH - 3} />
				{#each bars(h) as b, k (k)}
					{#if b.h > 0}<rect class="bar" x={b.x} y={y + RH - 3 - b.h} width={Math.max(b.w - 0.4, 0.6)} height={b.h} />{/if}
				{/each}
				<line class="base" x1={BX} x2={BX + BW} y1={y + RH - 3} y2={y + RH - 3} />
				{#if p.spread[l] > 0}
					<rect class="sbar" x={BX} y={y + 5} width={Math.max(sx(p.spread[l]) - BX, 1)} height={RH - 10} />
					<text class="val" x={W} y={y + RH - 7} text-anchor="end">{minus(fmt3(p.spread[l]))}</text>
				{:else}
					<text class="val" x={W} y={y + RH - 7} text-anchor="end">{t(L.zero)}</text>
				{/if}
			{/each}
			<text class="tick" x={HX} y={AY} text-anchor="start">{minus(stdText(p.lo))}</text>
			<text class="tick" x={HX + HW} y={AY} text-anchor="end">{minus(stdText(p.hi))}</text>
			{#if p.lo < 0 && p.hi > 0}<text class="tick" x={HX + HW / 2} y={AY} text-anchor="middle">0</text>{/if}
			<text class="tick" x={BX} y={AY} text-anchor="start">10<tspan dy="-4" class="exp">{minus(String(dom[0]))}</tspan></text>
			<text class="tick" x={BX + BW} y={AY} text-anchor="end">10<tspan dy="-4" class="exp">{minus(String(dom[1]))}</tspan></text>
		</svg>
	</figure>
{/snippet}

{#if pic}
	<div class="layers">
		{@render panel(pic.act, L.act, 'act')}
		{#if pic.grad}{@render panel(pic.grad, L.grad, 'grad')}{/if}
	</div>
{/if}

<style>
	.layers {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr));
		gap: 12px 20px;
		min-width: 0;
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	svg {
		width: 100%;
		max-width: 460px;
		height: auto;
		display: block;
	}
	.ttl {
		font-size: 13px;
		font-weight: 600;
		fill: var(--ink);
	}
	.sub,
	.tick,
	.lab {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.exp {
		font-size: 9px;
	}
	.val {
		font-size: 11px;
		fill: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.base {
		stroke: var(--rule);
	}
	.act .bar {
		fill: var(--seq-mid);
	}
	.act .sbar {
		fill: var(--seq-hi);
	}
	.grad .bar,
	.grad .sbar {
		fill: var(--grad);
	}
</style>
