// Package-only builds (K2 speed pack): ATLAS_ONLY=<slug>[,<slug>...] prerenders the home page and those packages' pages
// instead of all ~220 pages. svelte.config.js, the server hook it installs for such a build (only-hooks.ts) and
// scripts/check-build.ts all read the page list from here.
//   ATLAS_ONLY        package slugs; each may have a model page (/m/<slug>/), a concept page (/train/<slug>/) and a Lesson
//                     Lab page (/dev/atlas/lab/<slug>/, drafts builds only); the ones a package does not have answer 404
//                     during prerender and are skipped quietly
//   ATLAS_ONLY_PAGES  more pages to prerender in such a build (comma-separated paths, e.g. /atlas/,/glossary/)
// Plain TypeScript without imports: svelte.config.js loads it with Node's type stripping.

export function onlySlugs(env: Record<string, string | undefined> = process.env): string[] {
	return (env.ATLAS_ONLY ?? '')
		.split(',')
		.map((s) => s.trim())
		.filter(Boolean);
}

/** The pages a package may have. */
export function candidatePages(slug: string): string[] {
	return [`/m/${slug}/`, `/train/${slug}/`, `/dev/atlas/lab/${slug}/`];
}

const asPage = (p: string) => {
	const t = p.trim();
	if (!t) return '';
	const s = t.startsWith('/') ? t : `/${t}`;
	return s.endsWith('/') ? s : `${s}/`;
};

/** Every page of a package-only build: the home page, each slug's candidate pages and ATLAS_ONLY_PAGES. Null without
 *  ATLAS_ONLY (a normal build). */
export function onlyEntries(env: Record<string, string | undefined> = process.env): string[] | null {
	const slugs = onlySlugs(env);
	if (!slugs.length) return null;
	const extra = (env.ATLAS_ONLY_PAGES ?? '').split(',').map(asPage).filter(Boolean);
	return [...new Set(['/', ...slugs.flatMap(candidatePages), ...extra])];
}

/** Is a request path one a package-only build renders? (`pathname` without the base path; a page's own __data.json and
 *  the 404.html fallback count.) */
export function onlyAllows(pathname: string, entries: string[]): boolean {
	let p = pathname;
	try {
		p = decodeURIComponent(pathname);
	} catch {
		/* keep it */
	}
	if (p === '/[fallback]') return true;
	const page = p.replace(/__data\.json$/, '').replace(/\/?$/, '/');
	return entries.includes(page);
}

/** A candidate page that a package does not have (a 404 there is expected in a package-only build). */
export function isCandidate(path: string, entries: string[]): boolean {
	const p = path.replace(/\/?$/, '/');
	return entries.some((e) => e !== '/' && p.endsWith(e));
}
