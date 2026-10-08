<!--
  Card for a catalog model. Live models show the measured size of their shipped files
  (computed at build time by the root layout from static/models/<slug>/); planned models
  show the approximate size from the brief.
  <ModelCard model={MODEL_BY_SLUG.tinycnn} />
  A page that should not load the catalog renders CatalogCard with precomputed CardData instead.
-->
<script lang="ts">
	import CatalogCard from './CatalogCard.svelte';
	import { modelCard } from '$lib/catalog/cards';
	import type { ModelEntry } from '$lib/catalog/types';

	let { model, compact = false, showArea = true }: { model: ModelEntry; compact?: boolean; showArea?: boolean } = $props();
</script>

<CatalogCard card={modelCard(model)} {compact} {showArea} />
