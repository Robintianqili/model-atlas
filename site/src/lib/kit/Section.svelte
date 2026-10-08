<!--
  A numbered page section with an anchor, a title and an optional provenance badge.
  <Section id="output" n={7} title={{ en: 'Output', zh: '输出' }} provenance="live">…</Section>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import ProvenanceBadge from './ProvenanceBadge.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Provenance } from './types';

	interface Props {
		id: string;
		title: Text;
		n?: number;
		intro?: Text;
		provenance?: Provenance;
		precision?: string;
		tolerance?: number;
		/** full-bleed background band */
		band?: boolean;
		children: Snippet;
	}
	let { id, title, n, intro, provenance, precision, tolerance, band = false, children }: Props = $props();
</script>

<section {id} class="sec" class:band aria-labelledby="{id}-title">
	<header class="sh">
		<h2 id="{id}-title">
			{#if n !== undefined}<span class="n num" aria-hidden="true">{String(n).padStart(2, '0')}</span>{/if}
			<span>{t(title)}</span>
		</h2>
		{#if provenance}<ProvenanceBadge mode={provenance} {precision} {tolerance} />{/if}
	</header>
	{#if intro}<p class="intro">{t(intro)}</p>{/if}
	<div class="body">{@render children()}</div>
</section>

<style>
	.sec {
		padding: var(--s6) 0 var(--s4);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.band {
		background: var(--surface-2);
		border-radius: var(--r3);
		padding-inline: var(--s4);
	}
	.sh {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px 16px;
		margin-bottom: var(--s3);
	}
	h2 {
		margin: 0;
		display: flex;
		align-items: baseline;
		gap: 12px;
		font-size: var(--fs-xl);
	}
	.n {
		font-size: var(--fs-sm);
		color: var(--accent);
		font-weight: 600;
		letter-spacing: 0.04em;
	}
	.intro {
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.body {
		min-width: 0;
	}
</style>
