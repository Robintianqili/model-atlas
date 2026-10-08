import { readLicenses } from '$lib/server/assets';
import { MODELS } from '$lib/catalog/models';
import { CONCEPTS } from '$lib/catalog/concepts';
import { AREAS } from '$lib/catalog/areas';
import type { PageServerLoad } from './$types';

// Read static/licenses.json at build time; the table on /about/ is always current with the repo. The scope counts come
// from the catalog, so the page never goes stale when a lesson goes live.
export const load: PageServerLoad = () => {
	const liveModels = MODELS.filter((m) => m.status === 'live');
	return {
		licenses: readLicenses(),
		scope: {
			liveModels: liveModels.length,
			liveConcepts: CONCEPTS.filter((c) => c.status === 'live').length,
			liveAreas: new Set(liveModels.map((m) => m.area)).size,
			models: MODELS.length,
			concepts: CONCEPTS.length,
			areas: AREAS.length
		}
	};
};
