<!--
  A page section with an anchor, a title and an optional provenance label (redesign: sentence-case titles in the sans;
  a section number, when given, is a quiet prefix, never mono).
  <Section id="output" title={{ en: 'Output', zh: '输出' }} provenance="live">…</Section>
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
		/** a background band */
		band?: boolean;
		children: Snippet;
	}
	let { id, title, n, intro, provenance, precision, tolerance, band = false, children }: Props = $props();
</script>

<section {id} class="sec" class:band aria-labelledby="{id}-title">
	<header class="sh">
		<h2 id="{id}-title">
			{#if n !== undefined}<span class="n" aria-hidden="true">{n}</span>{/if}
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
		scroll-margin-top: calc(var(--ltop-h) + 12px);
	}
	.band {
		padding-inline: var(--s4);
		border-radius: var(--r4);
		background: var(--well);
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
		display: flex;
		align-items: baseline;
		gap: 10px;
		margin: 0;
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.n {
		color: var(--muted);
		font-size: var(--t-md);
		font-weight: var(--w-ui);
	}
	.intro {
		max-width: var(--measure);
		color: var(--ink-2);
	}
	.body {
		min-width: 0;
	}
</style>
