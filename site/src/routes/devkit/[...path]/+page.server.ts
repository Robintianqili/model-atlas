// The DevKit section of the site (/devkit/, /devkit/guide/<section>/, /devkit/reference/ops/, ...): ONE route, so the client
// router (which every page preloads) grows by a single entry, and every page is prerendered from data generated at build
// time out of the kit's own files (src/lib/devkit-site/data.server.ts). A stale page fails the build there.
import { error } from '@sveltejs/kit';
import { devkitPage, devkitPaths } from '$lib/devkit-site/data.server';
import type { EntryGenerator, PageServerLoad } from './$types';

export const prerender = true;

export const entries: EntryGenerator = () => devkitPaths().map((path) => ({ path }));

export const load: PageServerLoad = async ({ params }) => {
	const page = await devkitPage(params.path ?? '');
	if (!page) error(404, `There is no DevKit page /devkit/${params.path}/`);
	return { page };
};
