import type { PageLoad } from './$types';

// An included concept package: its page, the view registry and the formula views are lazy chunks behind concept/entry.ts
// (two import levels, so the Planned pages of this route neither load nor link any of it; see entry.ts). Entries, the
// 404s, the chrome flag and the planned page's data are in +page.server.ts.
export const load: PageLoad = async ({ data }) => {
	if (!data.auto) return { ...data, component: null };
	const component = await (await import('$lib/atlas/ui/concept/entry.ts')).loadConceptPage();
	return { ...data, component };
};
