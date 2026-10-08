<!--
  Provenance badge (required on every panel that shows model numbers; CONTRACTS.md rule 2).
    live         computed in the browser right now
    precomputed  computed offline from the same model
    illustrative made-up example values (always labeled)
  <ProvenanceBadge mode="live" precision="fp32" tolerance={3.1e-6} against="PyTorch fp32" />
  <ProvenanceBadge mode="illustrative" />
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
		/** smaller badge without the precision text */
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
		<span class="glyph" aria-hidden="true">
			{#if mode === 'live'}
				<svg viewBox="0 0 12 12" width="12" height="12"><circle cx="6" cy="6" r="3.2" fill="currentColor" /><circle class="ring" cx="6" cy="6" r="5" fill="none" stroke="currentColor" stroke-width="1.2" /></svg>
			{:else if mode === 'precomputed'}
				<svg viewBox="0 0 12 12" width="12" height="12"><ellipse cx="6" cy="3.2" rx="4.4" ry="1.9" fill="none" stroke="currentColor" stroke-width="1.3" /><path d="M1.6 3.2v5.6c0 1 2 1.9 4.4 1.9s4.4-.9 4.4-1.9V3.2M1.6 6c0 1 2 1.9 4.4 1.9S10.4 7 10.4 6" fill="none" stroke="currentColor" stroke-width="1.3" /></svg>
			{:else}
				<svg viewBox="0 0 12 12" width="12" height="12"><path d="M2 10l1-3 5.5-5.5 2 2L5 9z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" /></svg>
			{/if}
		</span>
		<span class="lbl">{label}</span>
		{#if precision && !compact}<span class="prec num">{precision}{#if approximate}≈{/if}</span>{/if}
		{#if tolerance !== undefined && !compact}<span class="prec num">Δ≤{fmt3(tolerance, 2)}</span>{/if}
	</span>
</Tooltip>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 2px 8px 2px 6px;
		border-radius: var(--r-pill);
		border: 1px solid currentColor;
		font-size: var(--fs-xs);
		font-weight: 650;
		line-height: 1.5;
		white-space: nowrap;
		vertical-align: middle;
	}
	.badge.compact {
		padding: 1px 6px 1px 4px;
	}
	.live {
		color: var(--live);
		background: var(--live-soft);
	}
	.precomputed {
		color: var(--pre);
		background: var(--pre-soft);
	}
	.illustrative {
		color: var(--illus);
		background: repeating-linear-gradient(
			-45deg,
			var(--illus-soft) 0 5px,
			color-mix(in srgb, var(--illus-soft) 55%, transparent) 5px 10px
		);
		border-style: dashed;
	}
	.glyph {
		display: inline-flex;
	}
	.ring {
		transform-origin: 6px 6px;
		/* a few pulses, not forever: infinite SVG animations repaint every frame (costly on big pages) */
		animation: pulse 2.4s var(--ease) 3;
	}
	@keyframes pulse {
		0% {
			opacity: 0.9;
			transform: scale(0.6);
		}
		70%,
		100% {
			opacity: 0;
			transform: scale(1.15);
		}
	}
	/* full color, no opacity: the precision text keeps 4.5:1 on the badge (axe color-contrast) */
	.prec {
		font-weight: 500;
	}
</style>
