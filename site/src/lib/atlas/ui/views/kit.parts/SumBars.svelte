<!--
  Signed bars for the parts of a sum: one row per part (per input channel, per head, …), then the bias, then the
  total. All rows share one scale, so their lengths add up by eye. A row with `pick` is a button (selects that part).
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtFull } from '../../../core/format.ts';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Row {
		label: string;
		value: number;
		/** the row the other panels show */
		on?: boolean;
		title?: Text;
	}
	interface Props {
		rows: Row[];
		bias?: number | null;
		total: number;
		/** accessible name of the group */
		label: Text;
		/** label of the total row ("= y") */
		totalLabel?: string;
		onpick?: (i: number) => void;
	}
	let { rows, bias = null, total, label, totalLabel = '= y', onpick }: Props = $props();

	const m = $derived.by(() => {
		let a = Math.abs(total);
		if (bias !== null) a = Math.max(a, Math.abs(bias));
		for (const r of rows) if (Number.isFinite(r.value)) a = Math.max(a, Math.abs(r.value));
		return Number.isFinite(a) && a > 0 ? a : 1;
	});
	const sc = $derived(makeScale('diverging', [-m, m], themeState.resolved));
	const w = (v: number) => `${Math.min(50, (Math.abs(v) / m) * 50)}%`;
</script>

<div class="sb" role="group" aria-label={t(label)}>
	{#each rows as r, i (i)}
		{#if onpick}
			<button type="button" class="row" class:on={r.on} aria-pressed={!!r.on} title={r.title ? t(r.title) : undefined} onclick={() => onpick?.(i)}>
				<span class="l num">{r.label}</span>
				<span class="track"><span class="b" class:neg={r.value < 0} style="--w: {w(r.value)}; background: {sc.css(r.value)}"></span></span>
				<span class="v num" title={fmtFull(r.value)}>{fmt3(r.value)}</span>
			</button>
		{:else}
			<div class="row" class:on={r.on}>
				<span class="l num">{r.label}</span>
				<span class="track"><span class="b" class:neg={r.value < 0} style="--w: {w(r.value)}; background: {sc.css(r.value)}"></span></span>
				<span class="v num" title={fmtFull(r.value)}>{fmt3(r.value)}</span>
			</div>
		{/if}
	{/each}
	{#if bias !== null}
		<div class="row fixed">
			<span class="l num">+ b</span>
			<span class="track"><span class="b" class:neg={bias < 0} style="--w: {w(bias)}; background: {sc.css(bias)}"></span></span>
			<span class="v num" title={fmtFull(bias)}>{fmt3(bias)}</span>
		</div>
	{/if}
	<div class="row fixed total">
		<span class="l num">{totalLabel}</span>
		<span class="track"><span class="b" class:neg={total < 0} style="--w: {w(total)}; background: {sc.css(total)}"></span></span>
		<span class="v num" title={fmtFull(total)}>{fmt3(total)}</span>
	</div>
</div>

<style>
	.sb {
		display: grid;
		gap: 1px;
		max-width: 380px;
		min-width: 0;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(3.2em, auto) minmax(60px, 1fr) 5.4em;
		align-items: center;
		gap: 8px;
		padding: 1px 4px;
		border: 0;
		border-radius: 3px;
		background: none;
		font: inherit;
		font-size: 11px;
		text-align: left;
		color: var(--ink);
		min-height: 20px;
	}
	button.row {
		cursor: pointer;
	}
	button.row:hover {
		background: var(--accent-soft);
	}
	.row.on {
		box-shadow: inset 0 0 0 1.5px var(--accent);
	}
	.row.fixed {
		border-top: 1px solid var(--rule);
	}
	.row.total {
		font-weight: 700;
	}
	.l {
		color: var(--muted);
		white-space: nowrap;
	}
	.track {
		position: relative;
		height: 10px;
		background: linear-gradient(90deg, transparent calc(50% - 0.5px), var(--rule-strong) calc(50% - 0.5px), var(--rule-strong) calc(50% + 0.5px), transparent calc(50% + 0.5px));
	}
	.b {
		position: absolute;
		top: 1px;
		bottom: 1px;
		left: 50%;
		width: var(--w);
		border-radius: 2px;
	}
	.b.neg {
		left: auto;
		right: 50%;
	}
	.v {
		text-align: right;
	}
</style>
