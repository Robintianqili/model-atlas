<!--
  ConceptPageShell (redesign spec §5, R1): a concept page is its lesson. This is the hand-built pages' entry: it looks the
  concept up in the catalog and renders the view host (kit/lesson/ConceptHost.svelte): the Lesson (the Walkthrough of the
  `lesson` snippet, with its cover) and, one tab away, Explore (hook, toy, real, compare), Pitfalls (pitfalls, post) and
  Sources (the concept's models, further). The old section snippets render inside the matching view; the goals and the
  predict-first question move onto the cover. Concept packages (AutoConceptPage) render ConceptHost with the data from
  the route's server load instead, so their JS carries no catalog module.

  <ConceptPageShell slug="tokenization" goals={[…]} prequestion={q}>
    {#snippet lesson()}<Walkthrough … />{/snippet}
    {#snippet toy()}…{/snippet}
  </ConceptPageShell>
-->
<script lang="ts">
	import type { ComponentProps } from 'svelte';
	import ConceptHost, { type ConceptPageData } from './lesson/ConceptHost.svelte';
	import { CONCEPT_BY_SLUG, SECTION_BY_SLUG } from '$lib/catalog/concepts';
	import { MODEL_BY_SLUG } from '$lib/catalog/models';
	import { modelCard } from '$lib/catalog/cards';
	import { techText } from '$lib/catalog/tech-zh';

	type Props = Omit<ComponentProps<typeof ConceptHost>, 'concept'> & { slug: string };
	let { slug, ...rest }: Props = $props();

	const concept = $derived.by((): ConceptPageData | null => {
		const c = CONCEPT_BY_SLUG[slug];
		if (!c) return null;
		const s = SECTION_BY_SLUG[c.section];
		return {
			slug: c.slug,
			title: c.title,
			status: c.status,
			interaction: c.interaction,
			engine: techText(c.engine),
			mode: c.mode,
			section: s ? { slug: c.section, n: s.n, name: s.name } : null,
			lesson: c.lesson,
			models: c.links.map((m) => MODEL_BY_SLUG[m]).filter(Boolean).map((m) => modelCard(m))
		};
	});
</script>

<ConceptHost {concept} {...rest} />
