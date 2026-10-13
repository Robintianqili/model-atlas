import type { PageLoad } from './$types';

// The auto page, its view registry and KaTeX are lazy chunks behind autoentry.ts (two import levels, so the Planned pages
// of this route neither load nor link any of it; see autoentry.ts). Entries, the 404s and the catalog data are in
// +page.server.ts (redesign W2: this universal load imports no catalog module, so no lesson shell carries the catalog).
export const load: PageLoad = async ({ data }) => {
	if (!data.auto) return { ...data, component: null };
	const component = await (await import('$lib/atlas/ui/autoentry.ts')).loadAutoPage();
	return { ...data, component };
};
