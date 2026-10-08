<!--
  Card for a training concept. <ConceptCard concept={CONCEPT_BY_SLUG.tokenization} />
  A page that should not load the catalog renders CatalogCard with precomputed CardData instead.
-->
<script lang="ts">
	import CatalogCard from './CatalogCard.svelte';
	import { conceptCard } from '$lib/catalog/concepts';
	import type { ConceptEntry } from '$lib/catalog/types';

	let { concept, compact = false, showSection = false }: { concept: ConceptEntry; compact?: boolean; showSection?: boolean } =
		$props();
</script>

<CatalogCard card={conceptCard(concept, showSection)} {compact} />
