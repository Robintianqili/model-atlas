<!--
  "Tokens in the text vs vocabulary size": one line per tokenizer, a marker at the current
  vocabulary size. Tap or click the plot to set the number of merges (within `pickMax`).
  Drawn in pixel coordinates (bind:clientWidth), so text stays readable on phones.
-->
<script lang="ts">
	import { t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import type { CurveSeries } from './types';

	interface Props {
		series: CurveSeries[];
		/** current number of merges (x = 256 + k) */
		k: number;
		/** clicks set k only up to this many merges */
		pickMax: number;
		label: Text;
		/** draw each series' name at its line end (wide charts only; the legend below names them too) */
		endLabels?: boolean;
		onpick?: (k: number) => void;
	}
	let { series, k, pickMax, label, endLabels = true, onpick }: Props = $props();

	let w = $state(640);
	const h = $derived(w < 480 ? 220 : 260);
	const m = $derived({ l: w < 480 ? 44 : 56, r: w < 480 || !endLabels ? 12 : 92, t: 28, b: 40 });
	const maxK = $derived(Math.max(1, ...series.map((s) => s.points.length - 1)));
	const maxY = $derived(Math.max(1, ...series.map((s) => s.points[0] ?? 0)));
	const x = (kk: number) => m.l + (kk / maxK) * (w - m.l - m.r);
	const y = (v: number) => m.t + (1 - v / maxY) * (h - m.t - m.b);
	let hover = $state<number | null>(null);

	function path(pts: number[]): string {
		if (!pts.length) return '';
		let d = `M${x(0).toFixed(1)},${y(pts[0]).toFixed(1)}`;
		const stride = Math.max(1, Math.floor(pts.length / 400));
		for (let i = stride; i < pts.length; i += stride) d += `L${x(i).toFixed(1)},${y(pts[i]).toFixed(1)}`;
		d += `L${x(pts.length - 1).toFixed(1)},${y(pts[pts.length - 1]).toFixed(1)}`;
		return d;
	}
	const xTicks = $derived([0, 256, 512, 767].filter((v) => v <= maxK));
	// distinct values only: a tiny or empty text gives maxY = 1, and keyed ticks must not repeat
	const yTicks = $derived([...new Set([0, Math.round(maxY / 2), maxY])]);

	function kAt(e: PointerEvent): number {
		const r = (e.currentTarget as SVGElement).getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * w;
		return Math.max(0, Math.min(maxK, Math.round(((px - m.l) / (w - m.l - m.r)) * maxK)));
	}
	const readK = $derived(hover ?? k);
	const vals = $derived(series.map((s) => ({ s, v: s.points[Math.min(readK, s.points.length - 1)], ended: readK > s.points.length - 1 })));
	const summary = $derived(
		vals.map(({ s, v }) => `${t(s.name)}: ${fmtInt(v ?? 0)}`).join('; ') + ` (${t({ en: 'vocabulary', zh: '词表' })} ${256 + readK})`
	);
</script>

<figure class="curve" bind:clientWidth={w}>
	<svg
		width={w}
		height={h}
		viewBox="0 0 {w} {h}"
		role="img"
		aria-label="{t(label)}. {summary}"
		onpointermove={(e) => (hover = kAt(e))}
		onpointerleave={() => (hover = null)}
		onpointerdown={(e) => {
			const kk = kAt(e);
			onpick?.(Math.min(kk, pickMax));
		}}
	>
		<!-- grid and axes -->
		{#each yTicks as v (v)}
			<line class="grid" x1={m.l} x2={w - m.r} y1={y(v)} y2={y(v)} />
			<text class="tick" x={m.l - 6} y={y(v) + 4} text-anchor="end">{fmtInt(v)}</text>
		{/each}
		{#each xTicks as v (v)}
			<text class="tick" x={x(v)} y={h - m.b + 16} text-anchor="middle">{256 + v}</text>
		{/each}
		<text class="axis" x={(m.l + w - m.r) / 2} y={h - 6} text-anchor="middle">{t({ en: 'vocabulary size (256 bytes + merges)', zh: '词表大小（256 个字节 + 合并数）' })}</text>
		<text class="axis" x={4} y={12} text-anchor="start">↑ {t({ en: 'tokens in the text', zh: '文本的词元数' })}</text>
		<line class="axisline" x1={m.l} x2={w - m.r} y1={y(0)} y2={y(0)} />

		<!-- series -->
		{#each series as s (s.id)}
			<path d={path(s.points)} fill="none" style:stroke={s.color} stroke-width="2.4" stroke-dasharray={s.dashed ? '6 4' : undefined} />
			{#if endLabels && w >= 480 && s.points.length}
				{@const last = s.points.length - 1}
				{@const name = t(s.name)}
				<!-- a name that does not fit after the line end (12 px bold: about 6.8 px per Latin character, 12.5 per CJK one) is right-aligned above the line instead of spilling out of the chart -->
				{@const fits = x(last) + 6 + Array.from(name).reduce((a, c) => a + ((c.codePointAt(0) ?? 0) >= 0x2e80 ? 12.5 : 6.8), 0) <= w}
				<text class="end" x={fits ? x(last) + 6 : x(last)} y={fits ? y(s.points[last]) + 4 : y(s.points[last]) - 8} text-anchor={fits ? 'start' : 'end'} style:fill={s.color}>{name}</text>
			{/if}
		{/each}

		<!-- current k -->
		<line class="cur" x1={x(k)} x2={x(k)} y1={m.t} y2={y(0)} />
		{#if hover !== null}<line class="hov" x1={x(hover)} x2={x(hover)} y1={m.t} y2={y(0)} />{/if}
		{#each vals as { s, v, ended } (s.id)}
			{#if v !== undefined && !ended}
				<circle cx={x(readK)} cy={y(v)} r="4.5" style:fill={s.color} style:stroke="var(--surface)" stroke-width="1.5" />
			{/if}
		{/each}
	</svg>
	<figcaption class="read">
		<span class="num">{t({ en: 'vocabulary', zh: '词表' })} {256 + readK}</span>
		{#each vals as { s, v, ended } (s.id)}
			<span class="item"><i style:background={s.color}></i>{t(s.name)}{colon()} <b class="num">{v === undefined ? '—' : fmtInt(v)}</b>{#if ended}<span class="muted">{t({ en: ' (no more merges)', zh: '（已无可合并）' })}</span>{/if}</span>
		{/each}
	</figcaption>
</figure>

<style>
	.curve {
		margin: 0;
		min-width: 0;
		display: grid;
		gap: 6px;
	}
	svg {
		display: block;
		touch-action: pan-y;
		cursor: crosshair;
		overflow: visible;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.axisline {
		stroke: var(--rule-strong);
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 11px;
		fill: var(--muted);
	}
	.axis {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.end {
		font-size: 12px;
		font-weight: 650;
	}
	.cur {
		stroke: var(--accent);
		stroke-width: 2;
	}
	.hov {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.read {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.item {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.item i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
	}
</style>
