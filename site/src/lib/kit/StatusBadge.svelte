<!--
  Page status for catalog cards and page headers: Live / Precomputed / Planned, optionally with the tier. Redesign look
  (spec §6.2): live is the rule, so it is a dot and the word, not a pill; planned is the exception and reads as text with
  a dashed underline. Same API as before. <StatusBadge status="planned" tier="P1" />
-->
<script lang="ts">
	import Tooltip from './Tooltip.svelte';
	import { ui } from '$lib/i18n/i18n.svelte';
	import type { Status, Tier } from '$lib/catalog/types';

	let { status, tier, tooltip = false }: { status: Status; tier?: Tier; tooltip?: boolean } = $props();

	const label = $derived(status === 'live' ? ui('statusLive') : status === 'precomputed' ? ui('statusPrecomputed') : ui('statusPlanned'));
	const tierHelp = $derived(
		tier === 'P0' ? ui('tierP0') : tier === 'V1' ? ui('tierV1') : tier === 'P1' ? ui('tierP1') : tier === 'P2' ? ui('tierP2') : ''
	);
</script>

{#snippet badge()}
	<span class="sb {status}">
		{#if status !== 'planned'}<span class="dot" aria-hidden="true"></span>{/if}
		{label}{#if tier && status !== 'live'}<span class="tier num" aria-label="{ui('tier')} {tier}">{tier}</span>{/if}
	</span>
{/snippet}

{#if tooltip && tier}
	<Tooltip tip={tierHelp}>{@render badge()}</Tooltip>
{:else}
	{@render badge()}
{/if}

<style>
	.sb {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		color: var(--ink-2);
		white-space: nowrap;
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--live);
	}
	.precomputed .dot {
		background: var(--pre);
	}
	.planned {
		color: var(--muted);
		text-decoration: underline dashed var(--line-control) 1px;
		text-underline-offset: 3px;
	}
	.tier {
		font-weight: var(--w-body);
		color: var(--muted);
	}
</style>
