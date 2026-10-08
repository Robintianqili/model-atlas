import { measureModelAssets, readLiveSpecs } from '$lib/server/assets';
import { MODELS } from '$lib/catalog/models';
import { modelPageCatalog, modelSlugOfRoute, siteCounts } from '$lib/catalog/slices';
import type { LayoutServerLoad } from './$types';

// Runs at build time (prerender). Measures the real size of every live model's shipped files and
// reads a few facts from their spec.json, so cards, fact strips and the input launcher show
// measured values instead of estimates. Missing folders are simply absent from the maps.
// It also hands each page the slice of the catalog it shows (catalog/slices.ts): the footer counts, and on a model page
// (/m/<slug>/, the Lesson Lab) that model's entry and cards, so those pages load none of the catalog modules.
export const load: LayoutServerLoad = ({ route, params }) => {
	const live = MODELS.filter((m) => m.status === 'live').map((m) => m.slug);
	const slug = modelSlugOfRoute(route.id, params);
	return {
		liveSizes: measureModelAssets(live),
		liveSpecs: readLiveSpecs(live),
		site: siteCounts(),
		catalog: slug ? modelPageCatalog(slug) : null
	};
};
