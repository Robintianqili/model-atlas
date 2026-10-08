// The site-path token of the DevKit pages' generated HTML (md.ts writes it in front of every site URL, because the HTML
// is made on the server where the page's relative base is not known); the page swaps it for $app/paths `base`.
export const BASE = '%DEVKIT_BASE%';

export const withBase = (html: string, base: string): string => html.split(BASE).join(base);

/** The site root as seen from a /devkit/ page, as a relative path ('..' from /devkit/, '../../..' from
 *  /devkit/guide/set-up/). It is the same string on the server and in the browser, so generated HTML that goes through
 *  {@html} keeps one value through hydration ($app/paths `base` is relative on the server and absolute in the browser,
 *  which made Svelte keep the server's HTML and warn "hydration_html_changed"). Works under any BASE_PATH. */
export function relBase(pathname: string): string {
	const i = pathname.lastIndexOf('/devkit/');
	const depth = (i >= 0 ? pathname.slice(i) : pathname).split('/').filter(Boolean).length;
	return Array.from({ length: Math.max(1, depth) }, () => '..').join('/');
}
