// ATLAS_SITE and ATLAS_ROOT (design §3.1). The same rules as pipeline/atlas_kit/paths.py.
//
//  ATLAS_SITE  the site folder the command runs in. For the CLI that is the site this file belongs to
//              (scripts/atlas/root.ts -> ../..), so a lane's copy of the CLI works on the lane.
//              The ATLAS_SITE environment variable overrides it (tests, unusual layouts).
//  ATLAS_ROOT  1. the ATLAS_ROOT environment variable;
//              2. for a lane, the `root` its lane.json records (<lane folder>/lane.json, next to the lane's site/). A lane
//                 may live outside the repo (ATLAS_LANES_DIR or "lanesDir" in atlas.config(.local).json, e.g. on another
//                 drive), where no walk up can find the repo;
//              3. the nearest ancestor of ATLAS_SITE that contains CONTRACTS.md and pipeline/. From a lane in
//                 site/.work/<name>/site the search skips site/.work/<name> and site/ (no pipeline/ there) and stops at
//                 the repo root.
// The main site (the tree a lane merges into) is the `main` its lane.json records, else <root>/site.
// The CLI passes ATLAS_SITE and ATLAS_ROOT to Python (env ATLAS_SITE, ATLAS_ROOT, PYTHONPATH=<root>/pipeline), and sets
// ATLAS_ROOT for every child of a lane outside the repo (cli.ts). New scripts never use a '../../site' relative path.
import { existsSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AtlasError } from './errors.ts';

export interface Roots {
	site: string;
	root: string;
	siteFrom: 'env' | 'cli';
	/** 'lane': the root recorded in the lane's lane.json */
	rootFrom: 'env' | 'search' | 'lane';
	/** lane name (lane.json next to the site folder, or the site/.work/<name>/site layout), null in the main tree */
	lane: string | null;
	/** the lane's folder (it holds lane.json and site/), null in the main tree */
	laneDir: string | null;
	/** the main site folder: the site itself in the main tree; for a lane, its lane.json `main`, else <root>/site */
	main: string;
}

const isDir = (p: string) => existsSync(p) && statSync(p).isDirectory();
const isFile = (p: string) => existsSync(p) && statSync(p).isFile();

export function isSite(d: string): boolean {
	return isFile(join(d, 'svelte.config.js')) && isFile(join(d, 'package.json')) && isDir(join(d, 'src', 'lib'));
}

export function isRoot(d: string): boolean {
	return isFile(join(d, 'CONTRACTS.md')) && isDir(join(d, 'pipeline'));
}

/** What a lane folder's lane.json says about where the lane belongs (atlas-lane/1; `root` and `main` since lanes may
 *  live outside the repo). */
export interface LaneRef {
	name: string;
	/** the lane's folder: <lanes dir>/<name> */
	dir: string;
	/** ATLAS_ROOT of the main tree, as lane.json records it */
	root?: string;
	/** the main site folder, as lane.json records it */
	main?: string;
}

/** The lane a site folder belongs to: <site>/../lane.json (in any lanes folder), else the site/.work/<name>/site layout
 *  (a lane without lane.json); null for the main tree. */
export function laneOf(site: string): LaneRef | null {
	const s = resolve(site);
	if (basename(s) !== 'site') return null;
	const dir = dirname(s);
	const lj = join(dir, 'lane.json');
	if (isFile(lj)) {
		try {
			const j = JSON.parse(readFileSync(lj, 'utf8').replace(/^﻿/, '')) as { format?: string; name?: unknown; root?: unknown; main?: unknown };
			if (j.format === 'atlas-lane/1' && typeof j.name === 'string' && j.name)
				return { name: j.name, dir, ...(typeof j.root === 'string' && j.root ? { root: resolve(j.root) } : {}), ...(typeof j.main === 'string' && j.main ? { main: resolve(j.main) } : {}) };
		} catch {
			/* not a lane.json we can read: fall back to the folder layout */
		}
	}
	const parts = s.split(sep);
	const n = parts.length;
	return n >= 3 && parts[n - 3] === '.work' ? { name: parts[n - 2], dir } : null;
}

/** The repo root for `site`: its lane.json `root` (a lane, possibly outside the repo), else the nearest ancestor (the
 *  site itself excluded) with CONTRACTS.md and pipeline/. */
export function findRoot(site: string): string | null {
	const recorded = laneOf(site)?.root;
	if (recorded && isRoot(recorded)) return recorded;
	let d = resolve(site);
	for (;;) {
		const up = dirname(d);
		if (up === d) return null;
		d = up;
		if (isRoot(d)) return d;
	}
}

/** 'w1-e1' for .../site/.work/w1-e1/site or for any <lanes dir>/w1-e1/site with a lane.json; null otherwise. */
export function laneName(site: string): string | null {
	return laneOf(site)?.name ?? null;
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
				fix: 'unset ATLAS_SITE (the CLI uses the site it belongs to), or point it at site/ or at a lane\'s <lane>/site'
			});
	}
	const lane = laneOf(site);
	let root: string | null;
	let rootFrom: Roots['rootFrom'] = 'search';
	if (env.ATLAS_ROOT?.trim()) {
		root = resolve(env.ATLAS_ROOT.trim());
		rootFrom = 'env';
		if (!isRoot(root))
			throw new AtlasError('AK002', `ATLAS_ROOT has no CONTRACTS.md and pipeline/: ${root}`, {
				fix: 'point ATLAS_ROOT at the repo root (the folder with CONTRACTS.md), or unset it'
			});
	} else if (lane?.root && isRoot(lane.root)) {
		root = lane.root;
		rootFrom = 'lane';
	} else {
		root = findRoot(site);
		if (!root)
			throw new AtlasError('AK002', `no repo root (CONTRACTS.md + pipeline/) above ${site}${lane?.root ? ` (its lane.json records ${lane.root}, which has none)` : ''}`, {
				likely: lane ? 'a lane outside the repo whose lane.json has no usable "root" (made by an older CLI, or the repo moved)' : 'the site was copied out of the repo',
				fix: lane
					? `set ATLAS_ROOT to the repo root, or add "root" (the repo root) and "main" (its site/) to ${join(lane.dir, 'lane.json')}`
					: 'run the CLI from the repo (site/ or a lane), or set ATLAS_ROOT'
			});
	}
	const main = lane ? (lane.main && isSite(lane.main) ? lane.main : join(root, 'site')) : site;
	return { site, root, siteFrom, rootFrom, lane: lane?.name ?? null, laneDir: lane?.dir ?? null, main };
}

let cached: Roots | null = null;
/** The roots of this process (resolved once). */
export function roots(): Roots {
	return (cached ??= resolveRoots());
}

/** K4: is `p` the folder `dir` or inside it? (any drive) */
export function inside(p: string, dir: string): boolean {
	const rel = relative(resolve(dir), resolve(p));
	return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

/** K4: where `atlas e2e` writes a package's screenshots, contact sheets, lesson print and axe report (git-ignored):
 *  notes/shots/<slug>/ in the repo, but <lane folder>/shots/<slug>/ for a lane outside the repo (a lane on another drive
 *  writes nothing to the repo's drive, and `lanes --rm` deletes the shots with the lane). review reads the same folder. */
export function shotsDir(slug: string, r: Roots = roots()): string {
	return r.laneDir && !inside(r.laneDir, r.root) ? join(r.laneDir, 'shots', slug) : join(r.root, 'notes', 'shots', slug);
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
