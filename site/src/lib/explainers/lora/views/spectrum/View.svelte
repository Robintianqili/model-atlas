<!--
  'lora/spectrum': one bar per direction of the needed change ΔW*, its strength s_k on a linear axis from 0 (meta.ts).
  The first r bars are filled (what a rank-r adapter can learn at best); the rest are outlined and dashed.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { SPECTRUM, floorOf, pctText, rankAt, shareOf } from '../../data.ts';

	let { rt, at }: StageProps = $props();
	const r = $derived(rankAt({ rt, at }));
	const [W, H, ML, MB, MT] = [360, 200, 34, 34, 12];
	const top = 2.2;
	const n = SPECTRUM.length;
	const bw = (W - ML - 8) / n;
	const y = (v: number) => H - MB - (v / top) * (H - MB - MT);
	const ticks = [0, 0.5, 1, 1.5, 2];
	const pct = (v: unknown) => (typeof v === 'string' ? v : '…');
</script>

<div class="wrap">
	<figure>
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Strength of each direction of the needed change', zh: '所需改变量每个方向的强度' })}>
			{#each ticks as v (v)}
				<line class="grid" x1={ML} x2={W - 8} y1={y(v)} y2={y(v)} />
				<text class="tick" x={ML - 6} y={y(v) + 4} text-anchor="end">{v}</text>
			{/each}
			{#each SPECTRUM as s, k (k)}
				<rect class="bar" class:kept={k < r} x={ML + k * bw + 2} y={y(s)} width={bw - 4} height={Math.max(0.5, y(0) - y(s))} />
				<text class="tick" x={ML + (k + 0.5) * bw} y={H - MB + 14} text-anchor="middle">{k + 1}</text>
			{/each}
			<text class="axis" x={ML + (W - ML) / 2} y={H - 4} text-anchor="middle">{t({ en: 'direction (strongest first, counting from 1)', zh: '方向（从强到弱，从 1 数起）' })}</text>
			<text class="axis" transform="translate(10 {(H - MB) / 2 + 6}) rotate(-90)" text-anchor="middle">{t({ en: 'strength s', zh: '强度 s' })}</text>
		</svg>
		<figcaption>
			<span class="sw kept"></span>{t({ en: `the first ${r}: what rank ${r} can learn`, zh: `前 ${r} 个：秩 ${r} 能学到的` })}
			<span class="sw rest"></span>{t({ en: 'out of reach', zh: '够不着' })}
		</figcaption>
	</figure>
	<p class="sum">
		{t({ en: `Rank ${r} holds ${pct(pctText(shareOf(r)))} of Σ s², so its loss cannot go below `, zh: `秩 ${r} 占 Σ s² 的 ${pct(pctText(shareOf(r)))}，所以它的损失不可能低于 ` })}<strong class="num">{fmt3(floorOf(r))}</strong>{t({ en: '.', zh: '。' })}
	</p>
</div>

<style>
	.wrap {
		max-width: 520px;
	}
	figure {
		margin: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 10px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axis {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.bar {
		fill: none;
		stroke: var(--rule-strong);
		stroke-dasharray: 3 2;
	}
	.bar.kept {
		fill: var(--param);
		stroke: var(--param);
		stroke-dasharray: none;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
	}
	.sw.kept {
		background: var(--param);
	}
	.sw.rest {
		border: 1px dashed var(--rule-strong);
	}
	.sum {
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
</style>
