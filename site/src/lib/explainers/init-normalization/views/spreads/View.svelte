<!--
  'init-normalization/spreads': two log-scale line plots, layer 1 to 10 across (meta.ts has the props): the spread of
  each layer's activations and of the gradient flowing back into it, one line per initialization (categorical colors,
  named in the legend; the zero initialization is all 0 and is named, not drawn).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { DEPTH, INIT_NAME } from '../../data.ts';
	import { spreadsOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(spreadsOf({ at }));
	const theme = $derived(themeState.resolved);

	const [W, H, ML, MR, MT, MB] = [320, 210, 44, 10, 24, 34];
	const minus = (s: string) => s.replace(/-/g, '−');
	function domain(rows: number[][]): [number, number] {
		const pos = rows.flat().filter((v) => v > 0 && Number.isFinite(v));
		if (!pos.length) return [-1, 0];
		let lo = Math.floor(Math.log10(Math.min(...pos)));
		let hi = Math.ceil(Math.log10(Math.max(...pos)));
		if (hi - lo < 1) hi = lo + 1;
		return [lo, hi];
	}
	const sx = (l: number) => ML + (l / (DEPTH - 1)) * (W - ML - MR);
	const ticks = (d: [number, number]) => {
		const n = d[1] - d[0];
		const step = Math.max(1, Math.ceil(n / 6));
		const out: number[] = [];
		for (let k = d[1]; k >= d[0]; k -= step) out.push(k);
		return out;
	};
	const drawn = (row: number[]) => row.some((v) => v > 0);
	const L = {
		act: { en: 'Spread of the activations', zh: '激活值的标准差' },
		grad: { en: 'Spread of the gradient flowing back', zh: '传回来的梯度的标准差' },
		layer: { en: 'layer', zh: '层' },
		zero: { en: 'all 0, not drawn on a log axis', zh: '全为 0，无法画在对数轴上' },
		legend: { en: 'Initializations', zh: '初始化方式' }
	};
</script>

{#snippet plot(rows: number[][], title: { en: string; zh: string })}
	{@const d = domain(rows)}
	{@const sy = (v: number) => MT + (1 - (Math.log10(v) - d[0]) / (d[1] - d[0])) * (H - MT - MB)}
	<figure>
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(title)}>
			<text class="ttl" x="0" y="12">{t(title)}</text>
			{#each ticks(d) as k (k)}
				<line class="grid" x1={ML} x2={W - MR} y1={sy(10 ** k)} y2={sy(10 ** k)} />
				<text class="tick" x={ML - 4} y={sy(10 ** k) + 3} text-anchor="end">10<tspan dy="-4" class="exp">{minus(String(k))}</tspan></text>
			{/each}
			{#each Array.from({ length: DEPTH }, (_, l) => l) as l (l)}
				<text class="tick" x={sx(l)} y={H - MB + 14} text-anchor="middle">{l + 1}</text>
			{/each}
			<text class="tick" x={(ML + W - MR) / 2} y={H - 4} text-anchor="middle">{t(L.layer)}</text>
			{#each rows as row, k (k)}
				{#if drawn(row)}
					<polyline class="ln" points={row.map((v, l) => (v > 0 ? `${sx(l)},${sy(v)}` : '')).filter(Boolean).join(' ')} style:stroke={categorical(k, theme)} />
					{#each row as v, l (l)}
						{#if v > 0}<circle cx={sx(l)} cy={sy(v)} r="2.6" style:fill={categorical(k, theme)} />{/if}
					{/each}
				{/if}
			{/each}
		</svg>
	</figure>
{/snippet}

{#if pic}
	<div class="spreads">
		<div class="plots">
			{@render plot(pic.act, L.act)}
			{#if props.grads !== false}{@render plot(pic.grad, L.grad)}{/if}
		</div>
		<ul class="legend" aria-label={t(L.legend)}>
			{#each pic.ids as id, k (id)}
				<li>
					<span class="sw" style:background={categorical(k, theme)}></span>{t(INIT_NAME[id])}{#if !drawn(pic.act[k])}<span class="note">({t(L.zero)})</span>{/if}
				</li>
			{/each}
		</ul>
	</div>
{/if}

<style>
	.spreads {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.plots {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
		gap: 12px 20px;
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
	.tick {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.exp {
		font-size: 9px;
	}
	.grid {
		stroke: var(--rule);
	}
	.ln {
		fill: none;
		stroke-width: 1.8;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: var(--fs-xs);
		color: var(--ink);
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.sw {
		width: 14px;
		height: 4px;
		border-radius: 2px;
	}
	.note {
		color: var(--ink-2);
	}
</style>
