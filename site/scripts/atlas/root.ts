// ATLAS_SITE and ATLAS_ROOT (design §3.1). The same rules as pipeline/atlas_kit/paths.py.
//
//  ATLAS_SITE  the site folder the command runs in. For the CLI that is the site this file belongs to
//              (scripts/atlas/root.ts -> ../..), so a lane's copy of the CLI works on the lane.
//              The ATLAS_SITE environment variable overrides it (tests, unusual layouts).
//  ATLAS_ROOT  the nearest ancestor of ATLAS_SITE that contains CONTRACTS.md and pipeline/. From a lane
//              (site/.work/<name>/site) the search skips site/.work/<name> and site/ (no pipeline/ there) and stops at
//              the repo root. The ATLAS_ROOT environment variable overrides it.
// The CLI passes both to Python (env ATLAS_SITE, ATLAS_ROOT, PYTHONPATH=<root>/pipeline). New scripts never use a
// '../../site' relative path.
import { existsSync, statSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AtlasError } from './errors.ts';

export interface Roots {
	site: string;
	root: string;
	siteFrom: 'env' | 'cli';
	rootFrom: 'env' | 'search';
	/** lane name for <root>/site/.work/<name>/site, null in the main tree */
	lane: string | null;
}

const isDir = (p: string) => existsSync(p) && statSync(p).isDirectory();
const isFile = (p: string) => existsSync(p) && statSync(p).isFile();

export function isSite(d: string): boolean {
	return isFile(join(d, 'svelte.config.js')) && isFile(join(d, 'package.json')) && isDir(join(d, 'src', 'lib'));
}

export function isRoot(d: string): boolean {
	return isFile(join(d, 'CONTRACTS.md')) && isDir(join(d, 'pipeline'));
}

/** Nearest ancestor of `site` (the site itself excluded) with CONTRACTS.md and pipeline/. */
export function findRoot(site: string): string | null {
	let d = resolve(site);
	for (;;) {
		const up = dirname(d);
		if (up === d) return null;
		d = up;
		if (isRoot(d)) return d;
	}
}

/** 'w1-e1' for .../site/.work/w1-e1/site; null otherwise. */
export function laneName(site: string): string | null {
	const parts = resolve(site).split(sep);
	const n = parts.length;
	return n >= 3 && parts[n - 1] === 'site' && parts[n - 3] === '.work' ? parts[n - 2] : null;
}

/** The site folder that contains this CLI (scripts/atlas/root.ts). */
export function cliSite(): string {
	return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
}

export function resolveRoots(env: NodeJS.ProcessEnv = process.env): Roots {
	let site = cliSite();
	let siteFrom: Roots['siteFrom'] = 'cli';
	if (env.ATLAS_SITE?.trim()) {
		site = resolve(env.ATLAS_SITE.trim());
		siteFrom = 'env';
		if (!isSite(site))
			throw new AtlasError('AK002', `ATLAS_SITE is not a site folder (no svelte.config.js, package.json, src/lib): ${site}`, {
				fix: 'unset ATLAS_SITE (the CLI uses the site it belongs to), or point it at site/ or site/.work/<name>/site'
			});
	}
	let root: string | null;
	let rootFrom: Roots['rootFrom'] = 'search';
	if (env.ATLAS_ROOT?.trim()) {
		root = resolve(env.ATLAS_ROOT.trim());
		rootFrom = 'env';
		if (!isRoot(root))
			throw new AtlasError('AK002', `ATLAS_ROOT has no CONTRACTS.md and pipeline/: ${root}`, {
				fix: 'point ATLAS_ROOT at the repo root (the folder with CONTRACTS.md), or unset it'
			});
	} else {
		root = findRoot(site);
		if (!root)
			throw new AtlasError('AK002', `no repo root (CONTRACTS.md + pipeline/) above ${site}`, {
				likely: 'the site was copied out of the repo',
				fix: 'run the CLI from the repo (site/ or a lane under site/.work/), or set ATLAS_ROOT'
			});
	}
	return { site, root, siteFrom, rootFrom, lane: laneName(site) };
}

let cached: Roots | null = null;
/** The roots of this process (resolved once). */
export function roots(): Roots {
	return (cached ??= resolveRoots());
}

export const sitePath = (...p: string[]) => join(roots().site, ...p);
export const rootPath = (...p: string[]) => join(roots().root, ...p);

/** Environment for Python children: ATLAS_SITE, ATLAS_ROOT, PYTHONPATH=<root>/pipeline, UTF-8 I/O. */
export function pythonEnv(r: Roots = roots(), base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
	const pp = join(r.root, 'pipeline');
	const sepChar = process.platform === 'win32' ? ';' : ':';
	return {
		...base,
		ATLAS_SITE: r.site,
		ATLAS_ROOT: r.root,
		PYTHONPATH: base.PYTHONPATH ? `${pp}${sepChar}${base.PYTHONPATH}` : pp,
		PYTHONUTF8: '1',
		PYTHONIOENCODING: 'utf-8'
	};
}

// `node scripts/atlas/root.ts` prints the resolution (handy inside a lane).
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	const r = resolveRoots();
	console.log(JSON.stringify(r, null, 2));
}
