import { error } from '@sveltejs/kit';
import { MODELS, MODEL_BY_SLUG } from '$lib/catalog/models';
import type { EntryGenerator, PageLoad } from './$types';

/** Live models with their own static route folder (src/routes/m/<slug>/), which takes precedence over [slug]. */
const STATIC_ROUTES = ['tinycnn', 'tinygpt'];

// Prerender every catalog model except the static routes: a "Planned" page, or the auto page of an included explainer
// package (DevKit design §5.1).
export const entries: EntryGenerator = () => MODELS.filter((m) => !STATIC_ROUTES.includes(m.slug)).map((m) => ({ slug: m.slug }));

export const load: PageLoad = async ({ params, data }) => {
	const model = MODEL_BY_SLUG[params.slug];
	if (!model) error(404, `Unknown model: ${params.slug}`);
	if (data.auto) {
		// the auto page, its view registry and KaTeX are lazy chunks behind autoentry.ts (two import levels, so the
		// Planned pages of this route neither load nor link any of it; see autoentry.ts)
		const component = await (await import('$lib/atlas/ui/autoentry.ts')).loadAutoPage();
		return { slug: model.slug, auto: data.auto, component };
	}
	if (model.status === 'live') error(404, `The live page for ${params.slug} needs src/routes/m/${params.slug}/ or an explainer package.`);
	return { slug: model.slug, auto: null, component: null };
};
