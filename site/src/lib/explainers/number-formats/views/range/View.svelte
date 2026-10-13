<!--
  'number-formats/range': each format's range on one log axis of sizes (meta.ts has the props), x marked or TinyGPT's
  weight sizes as a histogram, and optionally the trade-off table.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { NAMES, sup } from '../bits/meta.ts';
	import { big, rangeData, splitText } from './meta.ts';

	let { at, props }: StageProps = $props();
	const d = $derived(rangeData({ at, props }));
	const col = $derived.by(() => {
		void themeState.resolved;
		return { bar: categorical(0), hist: categorical(1) };
	});

	const [L, H] = [-26, 18];
	const W = 540;
	const ML = 84;
	const MR = 14;
	const ROW = 24;
	const HH = $derived(d.hist ? 74 : 0);
	const top = $derived(HH + 8);
	const height = $derived(top + d.rows.length * ROW + 40);
	const px = (k: number) => ML + ((Math.min(H, Math.max(L, k)) - L) / (H - L)) * (W - ML - MR);
	const lg = (v: number) => Math.log2(v);
	const ticks = [-24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16];
	const hmax = $derived(d.hist ? Math.max(1, ...d.hist) : 1);
	const xk = $derived(d.x !== null && d.x !== 0 ? lg(Math.abs(d.x)) : null);
</script>

<div class="range">
	<svg viewBox="0 0 {W} {height}" role="img" aria-label={t({ en: 'Range of each format on a log axis of sizes', zh: '各格式在对数大小轴上的范围' })}>
		{#if d.hist}
			{#each d.hist as c, i (i)}
				{#if c > 0}
					<rect x={px(d.histLo + i)} y={HH - (c / hmax) * (HH - 14)} width={Math.max(1, px(d.histLo + i + 1) - px(d.histLo + i) - 1)} height={(c / hmax) * (HH - 14)} fill={col.hist} />
				{/if}
			{/each}
			<text class="lab" x={ML} y="10">{t({ en: `TinyGPT’s weight sizes |w|: count per power of 2 (tallest bar ${fmtInt(hmax)})`, zh: `TinyGPT 权重的大小 |w|：每个 2 的幂内的个数（最高的柱子 ${fmtInt(hmax)}）` })}</text>
			<line class="base" x1={ML} x2={W - MR} y1={HH} y2={HH} />
		{/if}
		{#each ticks as k (k)}
			<line class="grid" class:one={k === 0} x1={px(k)} x2={px(k)} y1={top - 4} y2={top + d.rows.length * ROW} />
			<text class="lab" x={px(k)} y={top + d.rows.length * ROW + 14} text-anchor="middle">2{sup(k)}</text>
		{/each}
		<text class="lab" x={(ML + W - MR) / 2} y={top + d.rows.length * ROW + 32} text-anchor="middle">{t({ en: 'size |v| (log scale, one tick per 4 powers of 2; 2⁰ = 1)', zh: '大小 |v|（对数刻度，每 4 个 2 的幂一格；2⁰ = 1）' })}</text>
		{#each d.rows as r, i (r.id)}
			{@const y = top + i * ROW}
			<text class="name" x={ML - 6} y={y + ROW / 2 + 4} text-anchor="end">{t(NAMES[r.id])}</text>
			{#if r.float && r.minNormal > r.min}
				<rect class="sub" x={px(lg(r.min))} y={y + 5} width={Math.max(1, px(lg(r.minNormal)) - px(lg(r.min)))} height={ROW - 10} stroke={col.bar} fill={col.bar} fill-opacity="0.18" />
			{/if}
			<rect x={px(lg(r.minNormal))} y={y + 5} width={Math.max(2, px(lg(r.max)) - px(lg(r.minNormal)))} height={ROW - 10} fill={col.bar} />
			{#if lg(r.min) < L}<text class="arrow" x={ML + 2} y={y + ROW / 2 + 4}>◂</text>{/if}
			{#if lg(r.max) > H}<text class="arrow" x={W - MR - 10} y={y + ROW / 2 + 4}>▸</text>{/if}
		{/each}
		{#if xk !== null}
			<line class="xline" x1={px(xk)} x2={px(xk)} y1={top - 6} y2={top + d.rows.length * ROW} />
			<text class="lab xl" x={px(xk)} y={top - 8} text-anchor="middle">|x|</text>
		{/if}
	</svg>
	<p class="key">
		<span><span class="sw" style:background={col.bar}></span>{t({ en: 'normal numbers', zh: '正规数' })}</span>
		<span><span class="sw dashed" style:border-color={col.bar}></span>{t({ en: 'subnormal numbers (fewer significant bits)', zh: '次正规数（有效位更少）' })}</span>
		<span>◂ ▸ {t({ en: 'continues past the axis', zh: '超出坐标轴继续延伸' })}</span>
	</p>
	{#if props.table}
		<div class="tablewrap" use:keyScroll>
			<table>
				<thead>
					<tr>
						<th scope="col">{t({ en: 'Format', zh: '格式' })}</th>
						<th scope="col">{t({ en: 'Bits', zh: '位数' })}</th>
						<th scope="col">{t({ en: 'Largest', zh: '最大值' })}</th>
						<th scope="col">{t({ en: 'Smallest normal', zh: '最小正规数' })}</th>
						<th scope="col">{t({ en: 'Smallest', zh: '最小正数' })}</th>
						<th scope="col">{t({ en: 'Gap at 1', zh: '1 处的间隔' })}</th>
						<th scope="col">{t({ en: 'Values', zh: '可表示的值' })}</th>
					</tr>
				</thead>
				<tbody>
					{#each d.rows as r (r.id)}
						<tr>
							<th scope="row">{t(NAMES[r.id])}</th>
							<td>{t(splitText(r.id))}</td>
							<td class="num">{big(r.max)}</td>
							<td class="num">{r.float ? big(r.minNormal) : '—'}</td>
							<td class="num">{big(r.min)}</td>
							<td class="num">{big(r.gap1)}</td>
							<td class="num">{fmtInt(r.count)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.range {
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	svg {
		width: 100%;
		height: auto;
	}
	.grid {
		stroke: var(--rule);
	}
	.grid.one {
		stroke: var(--rule-strong);
	}
	.base {
		stroke: var(--rule-strong);
	}
	.sub {
		stroke-dasharray: 3 2;
		stroke-width: 1.2;
	}
	.xline {
		stroke: var(--highlight);
		stroke-width: 2;
	}
	.lab {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.lab.xl {
		fill: var(--ink);
		font-weight: 600;
	}
	.name {
		font-size: 13px;
		fill: var(--ink);
	}
	.arrow {
		font-size: 12px;
		fill: var(--ink);
	}
	.key {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.key span {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		width: 14px;
		height: 10px;
		display: inline-block;
		border-radius: 2px;
	}
	.sw.dashed {
		border: 1.5px dashed;
	}
	.tablewrap {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-xs);
		width: 100%;
	}
	th,
	td {
		padding: 3px 6px;
		border-bottom: 1px solid var(--rule);
		text-align: right;
		white-space: nowrap;
	}
	th[scope='row'],
	thead th:first-child {
		text-align: left;
	}
	thead th {
		color: var(--ink-2);
		font-weight: 500;
	}
	td.num {
		font-family: var(--font-mono);
	}
</style>
