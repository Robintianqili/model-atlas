// AK507 (redesign wave F, per-package view registry): what a view's meta.ts may import. The registry loads a package's
// metas with the package (prepareViews), and the kit's metas with every lesson, so a meta.ts stays cheap: it never
// imports a Svelte component (that would pull the view into the meta's chunk) and never another package's code (that
// would pull that package into this one). Types, the kit's pure modules and its own package's helpers are fine.
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const IMPORT_RE = /^\s*import\s+(type\s+)?(?:[^'"]*?\sfrom\s+)?['"]([^'"]+)['"]/gm;
const posix = (p: string) => p.split(String.fromCharCode(92)).join('/');

/** The AK507 problems of one meta.ts (`slug` null for a kit view); `src` overrides the file's text (tests). */
export function metaImportProblems(file: string, slug: string | null, src?: string): string[] {
	const text = src ?? readFileSync(file, 'utf8');
	const out: string[] = [];
	const where = slug ? `${slug}/${posix(file).split('/views/')[1]?.replace(/\/meta\.ts$/, '') ?? '?'}` : posix(file).split('/ui/views/')[1]?.replace(/\/meta\.ts$/, '') ?? posix(file);
	for (const m of text.matchAll(IMPORT_RE)) {
		const typeOnly = !!m[1];
		const spec = m[2];
		if (typeOnly) continue;
		if (/\.svelte(\?.*)?$/.test(spec) || spec.startsWith('svelte')) {
			out.push(`view ${where}: meta.ts imports '${spec}' (a Svelte module): the view's component stays in View.svelte, loaded lazily`);
			continue;
		}
		if (!spec.startsWith('.')) continue;
		const target = posix(resolve(dirname(file), spec));
		const other = /\/src\/lib\/explainers\/([^/]+)\//.exec(target)?.[1];
		if (other && other !== slug) out.push(`view ${where}: meta.ts imports '${spec}' from another package (${other}): copy what it needs into this package, or move it to the kit`);
	}
	return out;
}
