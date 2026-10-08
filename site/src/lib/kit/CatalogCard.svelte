<!--
  A catalog card from resolved CardData (catalog/types.ts): no catalog lookups, so a page can show cards without loading
  the catalog modules (ModelPageShell's links get their cards from page.data.catalog). Live models show the measured size
  of their shipped files (root layout liveSizes); planned models the approximate size from the brief.
  ModelCard and ConceptCard build the CardData from a catalog entry and render this.
  <CatalogCard card={modelCard(MODEL_BY_SLUG.tinycnn)} compact />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Card from './Card.svelte';
	import { UI } from '$lib/i18n/strings';
	import { fmtBytes } from './format';
	import type { CardData } from '$lib/catalog/types';

	let { card, compact = false, showArea = true }: { card: CardData; compact?: boolean; showArea?: boolean } = $props();

	const measured = $derived((page.data as { liveSizes?: Record<string, number> })?.liveSizes?.[card.slug]);
	const size = $derived(card.kind === 'model' && card.status === 'live' ? (measured ? fmtBytes(measured) : undefined) : card.size);
</script>

<Card
	kind={card.kind}
	href="{base}/{card.kind === 'model' ? 'm' : 'train'}/{card.slug}/"
	title={card.title}
	status={card.status}
	tier={card.tier}
	area={showArea ? card.area : undefined}
	family={card.family}
	inputs={card.inputs}
	{size}
	hook={card.hook ?? ''}
	note={card.v1 ? UI.inV1 : undefined}
	{compact}
/>
