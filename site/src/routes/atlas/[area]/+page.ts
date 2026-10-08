import { error } from '@sveltejs/kit';
import { AREAS, AREA_BY_SLUG, isAreaSlug } from '$lib/catalog/areas';
import type { EntryGenerator, PageLoad } from './$types';

// Prerender all 11 area pages.
export const entries: EntryGenerator = () => AREAS.map((a) => ({ area: a.slug }));

export const load: PageLoad = ({ params }) => {
	if (!isAreaSlug(params.area)) error(404, `Unknown area: ${params.area}`);
	return { area: AREA_BY_SLUG[params.area] };
};
