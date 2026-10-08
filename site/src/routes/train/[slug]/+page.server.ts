import { error } from '@sveltejs/kit';
import { CONCEPTS, CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
import { conceptPage } from '$lib/atlas/ui/concept/discover.server';
import { STATIC_CONCEPT_ROUTES } from '$lib/atlas/ui/concept/routes';
import type { EntryGenerator, PageServerLoad } from './$types';

// Concept pages (K2). Prerendered for every catalog concept except the hand-built static routes
// (src/routes/train/<slug>/, which take precedence over this route). An included concept package (explainers/<slug>/ with
// type: 'concept'; catalog status 'live', or any draft while ATLAS_INCLUDE_DRAFTS=1, which the dev server sets) renders its
// auto concept page with its plain text embedded at build time; every other slug gets `auto: null` and renders the
// Planned page exactly as before. Server only, so none of this reaches the Planned pages' JS.
export const entries: EntryGenerator = () => CONCEPTS.filter((c) => !STATIC_CONCEPT_ROUTES.includes(c.slug)).map((c) => ({ slug: c.slug }));

export const load: PageServerLoad = ({ params }) => {
	const concept = CONCEPT_BY_SLUG[params.slug];
	if (!concept) error(404, `Unknown concept: ${params.slug}`);
	const auto = conceptPage(params.slug);
	if (!auto && concept.status === 'live') error(404, `The live page for ${params.slug} needs src/routes/train/${params.slug}/ or a concept package (explainers/${params.slug}/, type 'concept').`);
	return { slug: concept.slug, auto };
};
