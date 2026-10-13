import { error } from '@sveltejs/kit';
import { CONCEPTS, CONCEPT_BY_SLUG, SECTION_BY_SLUG, conceptCard } from '$lib/catalog/concepts';
import { MODEL_BY_SLUG } from '$lib/catalog/models';
import { modelCard } from '$lib/catalog/cards';
import { techText } from '$lib/catalog/tech-zh';
import { conceptPage } from '$lib/atlas/ui/concept/discover.server';
import { STATIC_CONCEPT_ROUTES } from '$lib/atlas/ui/concept/routes';
import type { CardData } from '$lib/catalog/types';
import type { ConceptPageData } from '$lib/kit/lesson/ConceptHost.svelte';
import type { EntryGenerator, PageServerLoad } from './$types';

// Concept pages (K2). Prerendered for every catalog concept except the hand-built static routes
// (src/routes/train/<slug>/, which take precedence over this route). An included concept package (explainers/<slug>/ with
// type: 'concept'; catalog status 'live', or any draft while ATLAS_INCLUDE_DRAFTS=1, which the dev server sets) renders its
// auto concept page with its plain text embedded at build time: the lesson app (chrome: 'lesson', redesign spec §6.1).
// Every other slug gets `auto: null` and a planned page (spec §6.9) whose catalog data is computed here, so the route
// ships no catalog module. Server only, so none of this reaches the Planned pages' JS.
export const entries: EntryGenerator = () => CONCEPTS.filter((c) => !STATIC_CONCEPT_ROUTES.includes(c.slug)).map((c) => ({ slug: c.slug }));

const compact = (c: CardData): CardData => {
	const { hook: _hook, ...shown } = c;
	return shown;
};

export const load: PageServerLoad = ({ params }) => {
	const concept = CONCEPT_BY_SLUG[params.slug];
	if (!concept) error(404, `Unknown concept: ${params.slug}`);
	const auto = conceptPage(params.slug);
	if (!auto && concept.status === 'live') error(404, `The live page for ${params.slug} needs src/routes/train/${params.slug}/ or a concept package (explainers/${params.slug}/, type 'concept').`);
	const section = SECTION_BY_SLUG[concept.section];
	if (auto) {
		// what the concept page shows about its concept, resolved here (the page's JS carries no catalog module)
		const conceptCatalog: ConceptPageData = {
			slug: concept.slug,
			title: concept.title,
			status: concept.status as ConceptPageData['status'],
			interaction: concept.interaction,
			engine: techText(concept.engine),
			mode: concept.mode,
			section: section ? { slug: concept.section, n: section.n, name: section.name } : null,
			...(concept.lesson ? { lesson: { summary: concept.lesson.summary, ...(concept.lesson.minutes ? { minutes: concept.lesson.minutes } : {}) } } : {}),
			models: concept.links.map((s) => MODEL_BY_SLUG[s]).filter(Boolean).map((m) => compact(modelCard(m)))
		};
		return { slug: concept.slug, auto, chrome: 'lesson' as const, planned: null, conceptCatalog };
	}
	return {
		slug: concept.slug,
		auto,
		planned: {
			title: concept.title,
			section: { slug: concept.section, n: section.n, name: section.name, blurb: section.blurb },
			interaction: concept.interaction,
			engine: techText(concept.engine),
			mode: concept.mode,
			v1: concept.v1,
			tier: concept.tier ?? null,
			models: concept.links.map((s) => MODEL_BY_SLUG[s]).filter(Boolean).map((m) => compact(modelCard(m))),
			siblings: CONCEPTS.filter((c) => c.section === concept.section && c.slug !== concept.slug).map((c) => compact(conceptCard(c)))
		}
	};
};
