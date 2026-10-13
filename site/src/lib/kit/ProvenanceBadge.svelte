<!--
  Provenance label (required on every panel that shows model numbers; CONTRACTS.md rule 2). Redesign look (spec §4.4,
  ProvenanceLabel): a dot and the word, with the precision in mono; the tooltip (hover, focus or tap) explains the kind
  and gives the parity detail. Same API as before.
    live         computed in the browser right now
    precomputed  computed offline from the same model
    illustrative made-up example values (always labeled; a dashed dot)
  <ProvenanceBadge mode="live" precision="fp32" tolerance={3.1e-6} against="PyTorch fp32" />
  <ProvenanceBadge mode="illustrative" />
  The root keeps the class "badge": atlas e2e looks for `.stage .badge` on every step (AK502).
-->
<script lang="ts">
	import Tooltip from './Tooltip.svelte';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from './format';
	import type { Provenance } from './types';

	interface Props {
		mode: Provenance;
		/** fp32 / fp16 / q8 / q4 … */
		precision?: string;
		/** measured max |diff| against the reference */
		tolerance?: number;
		/** what the tolerance was measured against */
		against?: string;
		/** quantized runs are "approximate", never "exact" */
		approximate?: boolean;
		/** extra sentence for the tooltip */
		note?: Text;
		/** the word only (no precision), for tight places */
		compact?: boolean;
	}
	let { mode, precision, tolerance, against = 'PyTorch', approximate = false, note, compact = false }: Props =
		$props();

	const label = $derived(
		mode === 'live' ? ui('provLive') : mode === 'precomputed' ? ui('provPrecomputed') : ui('provIllustrative')
	);
	const help = $derived(
		mode === 'live'
			? ui('provLiveHelp')
			: mode === 'precomputed'
				? ui('provPrecomputedHelp')
				: ui('provIllustrativeHelp')
	);
</script>

<Tooltip width={300}>
	{#snippet content()}
		<strong>{label}.</strong>
		{help}
		{#if precision}<br />{ui('provPrecision')}{colon()} {precision}{#if approximate}&nbsp;· {ui('provApproximate')}{/if}{/if}
		{#if tolerance !== undefined}<br />{ui('provTolerance', { against })}: {fmt3(tolerance)}{/if}
		{#if note}<br />{t(note)}{/if}
	{/snippet}
	<span class="badge {mode}" class:compact>
		<span class="dot" aria-hidden="true"></span>
		<span class="lbl">{label}</span>
		{#if precision && !compact}<span class="prec num">{precision}{#if approximate}≈{/if}</span>{/if}
		{#if tolerance !== undefined && !compact}<span class="prec num">Δ≤{fmt3(tolerance, 2)}</span>{/if}
	</span>
</Tooltip>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		padding: 2px 0;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		white-space: nowrap;
		vertical-align: middle;
	}
	.badge.compact {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		gap: 6px;
	}
	.dot {
		flex: none;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--live);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--live) 22%, transparent);
	}
	.precomputed .dot {
		background: var(--pre);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--pre) 22%, transparent);
	}
	/* made-up values: a hollow, dashed dot (never colour alone: the word says it too) */
	.illustrative .dot {
		width: 10px;
		height: 10px;
		background: transparent;
		border: 1.5px dashed var(--illus);
		box-shadow: none;
	}
	.lbl {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.prec {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	@media (forced-colors: active) {
		.dot {
			background: CanvasText;
			forced-color-adjust: none;
		}
	}
</style>
