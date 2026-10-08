// atlas build <slug>[,<slug>...] [--no-types] | build --all [--no-types]   (K2 speed pack)
//   <slugs>  a package-only build (ATLAS_ONLY=<slugs>: the home page and those packages' pages, drafts included) into
//            .atlas/build-<slug>/ (several: .atlas/build-only-<n>/), after svelte-check over those packages and
//            check:catalog, then check:build on it. The same steps as the site gate of `atlas check <slug>`.
//   --all    the full build (every page) into .atlas/build-all/ after npm run check, then check:build.
//   --no-types  skip the type check (it already passed, e.g. in a quick check).
// The output is static files (the pages under their paths, 404.html); any static file server shows them.
import { AtlasError } from '../errors.ts';
import { siteSteps } from '../lib/build.ts';
import { Issues } from '../lib/pkg.ts';
import { listPackages } from '../load.ts';
import { roots } from '../root.ts';

export const USAGE = 'atlas build <slug>[,<slug>...] [--no-types] | build --all [--no-types]';

export async function main(args: string[]): Promise<number> {
	const t0 = performance.now();
	const slugs: string[] = [];
	let all = false;
	let noTypes = false;
	let echo = false;
	for (const a of args) {
		if (a === '--all') all = true;
		else if (a === '--no-types') noTypes = true;
		else if (a === '--verbose') echo = true;
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: USAGE });
		else slugs.push(...a.split(',').map((s) => s.trim()).filter(Boolean));
	}
	if (all === !!slugs.length) throw new AtlasError('AK301', `usage: ${USAGE}`);
	const r = roots();
	if (!all) {
		const known = new Set(listPackages(r.site));
		const unknown = slugs.filter((s) => !known.has(s));
		if (unknown.length) throw new AtlasError('AK301', `no package ${unknown.join(', ')} in src/lib/explainers/`, { fix: `packages: ${[...known].join(', ')}` });
	}
	const is = new Issues();
	const name = all ? 'build-all' : slugs.length === 1 ? `build-${slugs[0]}` : `build-only-${slugs.length}`;
	const res = await siteSteps(is, { site: r.site, name, slugs: all ? null : slugs, echo, noTypes });
	console.log(`atlas build ${all ? '--all' : slugs.join(',')}: ${res.ok ? 'OK' : 'FAIL'}; ${res.summary}; output ${res.dir}/ (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
	if (is.list.length) console.error(is.format());
	return res.ok ? 0 : 1;
}
