import { includeDrafts, labSlugs } from '$lib/atlas/ui/discover.server';
import type { EntryGenerator, PageServerLoad } from './$types';

// The Lesson Lab (design §5.1): one page per package (and per shipped spec folder, e.g. tinycnn with ?spec=models/tinycnn).
// Production builds publish no lab page: entries exist only with ATLAS_INCLUDE_DRAFTS=1, and prerender is 'auto' so a
// dynamic route without entries is not an error ("unseen route"). The dev server always serves it.
export const prerender = 'auto';

export const entries: EntryGenerator = () => {
	if (!includeDrafts()) return [];
	const { packages, specDirs } = labSlugs();
	return [...new Set([...packages, ...specDirs])].map((slug) => ({ slug }));
};

export const load: PageServerLoad = ({ params }) => {
	const { packages, specDirs, files } = labSlugs();
	return { slug: params.slug, enabled: includeDrafts(), packages, specDirs, files };
};
