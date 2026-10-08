<!--
  Fact strip: a row of label/value facts with the provenance badge (brief §4.2 item 2).
  <FactStrip facts={[{ label: 'Parameters', value: '6,722', mono: true }]} provenance="live" precision="fp32" tolerance={2e-6} />
-->
<script lang="ts">
	import ProvenanceBadge from './ProvenanceBadge.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Provenance } from './types';

	interface Props {
		facts: { label: Text; value: Text; mono?: boolean }[];
		provenance?: Provenance;
		precision?: string;
		tolerance?: number;
		against?: string;
	}
	let { facts, provenance, precision, tolerance, against }: Props = $props();
</script>

<div class="fs">
	<dl>
		{#each facts as f, i (i)}
			<div class="f">
				<dt>{t(f.label)}</dt>
				<dd class:num={f.mono}>{t(f.value)}</dd>
			</div>
		{/each}
	</dl>
	{#if provenance}
		<div class="prov"><ProvenanceBadge mode={provenance} {precision} {tolerance} {against} /></div>
	{/if}
</div>

<style>
	.fs {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px 24px;
		padding: 12px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
	}
	dl {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 24px;
		margin: 0;
		flex: 1 1 auto;
	}
	.f {
		min-width: 0;
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-family: var(--font-mono);
	}
	dd {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.prov {
		flex: 0 0 auto;
	}
</style>
