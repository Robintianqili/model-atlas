<!--
  'residual-connections/bars' (meta.ts has the modes and props): one row per layer, or per depth, with its bars and
  number. No skips in the second color of the kit's categorical palette, with skips in the accent color (the same
  pair as the loss curves on this page). Log axes say so, and every tick is labelled. Follows the frame; no animation.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { AXES, NAMES, barsOf, describeBars, lengthOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(barsOf({ at, props }));
	const theme = $derived(themeState.resolved);
	const color = (skip: number) => (skip ? 'var(--accent)' : categorical(1, theme));
	const pct = (v: number) => `${(pic ? lengthOf(pic, v) : 0) * 100}%`;
	const ticks = $derived.by(() => {
		if (!pic) return [] as { at: number; label: string }[];
		if (pic.log) {
			const out: { at: number; label: string }[] = [];
			for (let e = Math.round(Math.log10(pic.lo)); e <= Math.round(Math.log10(pic.hi)); e += 2) out.push({ at: lengthOf(pic, 10 ** e), label: e === 0 ? '1' : `10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)])}` });
			return out;
		}
		const step = pic.hi > 2 ? 1 : 0.2;
		const out: { at: number; label: string }[] = [];
		for (let v = 0; v <= pic.hi + 1e-9; v += step) out.push({ at: lengthOf(pic, v), label: String(Number(v.toFixed(1))) });
		return out;
	});
</script>

<figure class="bars" aria-label={t(describeBars(pic))}>
	{#if !pic}
		<p class="muted small">{t({ en: 'Nothing has been computed yet.', zh: '还没有算出任何数值。' })}</p>
	{:else}
		<div class="legend small">
			{#each pic.series as s (s)}<span><i style:background={color(s)}></i>{t(NAMES[s])}</span>{/each}
			<span class="axis-name">{t(AXES[pic.mode])}</span>
		</div>
		{#if pic.waiting}<p class="note small">{t(pic.waiting)}</p>{/if}
		<div class="rows">
			{#each pic.rows as r, i (i)}
				<div class="row" class:mark={r.mark}>
					<span class="lab small">{t(r.label)}</span>
					<span class="track">
						{#each ticks as k (k.label)}<span class="grid" style:left="{k.at * 100}%"></span>{/each}
						{#if pic.ref}<span class="ref" style:left={pct(pic.ref.value)}></span>{/if}
						{#each r.bars as b (b.skip)}<span class="bar" class:solo={r.bars.length === 1} style:width={pct(b.value)} style:background={color(b.skip)}></span>{/each}
					</span>
					<span class="val num">{#each r.bars as b, k (b.skip)}{#if k > 0}<br />{/if}{fmt3(b.value)}{/each}{#if r.note}<br /><span class="muted">{r.note}</span>{/if}</span>
				</div>
			{/each}
		</div>
		<div class="row axis" aria-hidden="true">
			<span></span>
			<span class="ticks num">{#each ticks as k (k.label)}<span style:left="{k.at * 100}%">{k.label}</span>{/each}</span>
			<span></span>
		</div>
		{#if pic.ref}<p class="muted small"><span class="dash"></span>{t(pic.ref.label)} = {fmt3(pic.ref.value)}</p>{/if}
		{#if pic.log}<p class="muted small">{t({ en: 'Log scale: each labelled tick is 100 times the one to its left.', zh: '对数刻度：每个标注的刻度是它左边那个的 100 倍。' })}</p>{/if}
	{/if}
</figure>

<style>
	.bars {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		grid-template-columns: minmax(0, 1fr);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		align-items: center;
		color: var(--ink-2);
	}
	.legend i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 6px;
		vertical-align: -1px;
	}
	.axis-name {
		color: var(--muted);
	}
	.rows {
		display: grid;
		gap: 3px;
	}
	.row {
		display: grid;
		grid-template-columns: 4.6rem minmax(0, 1fr) 4.4rem;
		align-items: center;
		gap: 8px;
	}
	.row.mark .lab {
		font-weight: 700;
		color: var(--ink);
	}
	.row.mark .track {
		outline: 2px solid var(--ink-2);
		outline-offset: 1px;
	}
	.lab {
		color: var(--ink-2);
		white-space: nowrap;
	}
	.track {
		position: relative;
		display: grid;
		gap: 2px;
		padding: 2px 0;
		min-height: 12px;
		background: var(--surface-2);
		border-radius: 3px;
	}
	.grid {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 1px;
		background: var(--rule);
	}
	.ref {
		position: absolute;
		top: -2px;
		bottom: -2px;
		border-left: 2px dashed var(--ink-2);
	}
	.bar {
		position: relative;
		display: block;
		height: 7px;
		border-radius: 2px;
	}
	.bar.solo {
		height: 10px;
	}
	.val {
		font-size: 11px;
		line-height: 1.25;
		text-align: right;
		color: var(--ink-2);
	}
	.ticks {
		position: relative;
		height: 14px;
		font-size: 11px;
		color: var(--muted);
	}
	.ticks span {
		position: absolute;
		transform: translateX(-50%);
		white-space: nowrap;
	}
	.ticks span:first-child {
		transform: none;
	}
	.ticks span:last-child {
		transform: translateX(-100%);
	}
	.dash {
		display: inline-block;
		width: 16px;
		margin-right: 6px;
		border-top: 2px dashed var(--ink-2);
		vertical-align: 3px;
	}
	.note {
		margin: 0;
		padding: 6px 8px;
		border: 1px dashed var(--rule);
		border-radius: 4px;
		color: var(--ink-2);
	}
	p {
		margin: 0;
	}
	.small {
		font-size: var(--fs-sm);
	}
	@media (max-width: 420px) {
		.row {
			grid-template-columns: 3.6rem minmax(0, 1fr) 3.8rem;
			gap: 6px;
		}
	}
</style>
