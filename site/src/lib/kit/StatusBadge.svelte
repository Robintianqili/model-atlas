<!--
  Page status badge for catalog cards and page headers: Live / Precomputed / Planned,
  optionally with the tier. <StatusBadge status="planned" tier="P1" />
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
		{#if status === 'live'}<span class="dot" aria-hidden="true"></span>{/if}
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
		gap: 5px;
		padding: 1px 8px;
		border-radius: var(--r-pill);
		font-size: var(--fs-xs);
		font-weight: 700;
		line-height: 1.6;
		white-space: nowrap;
		border: 1px solid currentColor;
	}
	.live {
		color: var(--live);
		background: var(--live-soft);
	}
	.precomputed {
		color: var(--pre);
		background: var(--pre-soft);
	}
	.planned {
		color: var(--planned);
		background: var(--planned-soft);
		border-style: dashed;
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: currentColor;
	}
	.tier {
		padding-left: 5px;
		border-left: 1px solid currentColor;
		font-weight: 600;
	}
</style>
