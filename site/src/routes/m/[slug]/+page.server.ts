import { error } from '@sveltejs/kit';
import { autoPage } from '$lib/atlas/ui/discover.server';
import { MODELS, MODEL_BY_SLUG } from '$lib/catalog/models';
import { modelCard } from '$lib/catalog/cards';
import type { CardData } from '$lib/catalog/types';
import type { EntryGenerator, PageServerLoad } from './$types';

/** Live models with their own static route folder (src/routes/m/<slug>/), which takes precedence over [slug]. */
const STATIC_ROUTES = ['tinycnn', 'tinygpt'];

// Prerender every catalog model except the static routes: a "Planned" page, or the auto page of an included explainer
// package (DevKit design §5.1; catalog status 'live', or any draft while ATLAS_INCLUDE_DRAFTS=1, which the dev server
// sets). Server only (redesign W2): the catalog modules never reach the browser from this route. An auto page is the
// lesson app (chrome: 'lesson': no site header, spec §6.1); a planned page keeps the site chrome and gets its area's other
// models as cards (spec §6.9; the rest of what it shows is page.data.catalog from the root layout).
export const entries: EntryGenerator = () => MODELS.filter((m) => !STATIC_ROUTES.includes(m.slug)).map((m) => ({ slug: m.slug }));

export const load: PageServerLoad = ({ params }) => {
	const model = MODEL_BY_SLUG[params.slug];
	if (!model) error(404, `Unknown model: ${params.slug}`);
	const auto = autoPage(params.slug);
	if (auto) return { slug: model.slug, auto, chrome: 'lesson' as const, siblings: [] as CardData[] };
	if (model.status === 'live') error(404, `The live page for ${params.slug} needs src/routes/m/${params.slug}/ or an explainer package.`);
	const siblings = MODELS.filter((m) => m.area === model.area && m.slug !== model.slug)
		.sort((a, b) => (a.status === b.status ? 0 : a.status === 'live' ? -1 : 1))
		.slice(0, 6)
		.map((m) => {
			const { hook: _hook, ...c } = modelCard(m);
			return c;
		});
	return { slug: model.slug, auto: null, siblings };
};
