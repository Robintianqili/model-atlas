// Lanes (design §6.4): parallel copies of the site at <lanes>/<name>/site, each owning a set of paths. The lanes folder is
// site/.work by default; ATLAS_LANES_DIR or "lanesDir" in atlas.config(.local).json moves it, for example to another
// drive (config.ts lanesChoice).
//
// <lanes>/<name>/lane.json (atlas-lane/1, written by `atlas worktree`, never inside the lane's site folder):
//   { format, name, created, synced, merged, root, main, rev, owns: [root-relative globs],
//     snapshot: { <site-relative path>: sha256 } }
// root and main: the repo root (ATLAS_ROOT) and the main site the lane merges into, as absolute POSIX paths, so a lane
// outside the repo finds them (root.ts). rev: the commit a --from-git lane was made or last synced from.
// The snapshot holds the main tree's content hash of every file at the moment it was copied (or last synced/merged),
// so `merge` and `--sync` can tell what the LANE changed from what MAIN changed since.
// Globs are root-relative ('site/src/lib/explainers/tinyvit/**', 'pipeline/tinyvit/**', 'notes/tinyvit.md'):
// '**' any depth, '*' one name segment, '{a,b}' alternatives; '[' and ']' are literal ('site/src/routes/m/[slug]/**').
// Only site/ files live in a lane copy. Root-level owned paths (pipeline/<slug>/, notes/) are edited in place in the main
// tree, so `merge` has nothing to copy for them.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { LanesChoice } from '../config.ts';

export const LANE_FORMAT = 'atlas-lane/1';

/** Never copied into a lane, never compared, never merged. vite.config.ts is lane setup (the fs.allow patch). */
export const SKIP_DIRS = ['node_modules', '.svelte-kit', 'build', '.work', '.atlas'];
export const SKIP_FILES = ['package-lock.json'];
export const LANE_SETUP = ['vite.config.ts'];
/** K2: machine-local files a lane keeps as they are: never synced, compared or merged (git-ignored): the configured
 *  Python, and this computer's strings and block list for the /devkit/ guard (src/lib/devkit-site/machine.ts) */
export const LANE_LOCAL = ['scripts/atlas/atlas.config.local.json', 'src/lib/devkit-site/machine.local.json'];

export interface LaneFile {
	format: typeof LANE_FORMAT;
	name: string;
	created: string;
	synced?: string;
	merged?: string;
	/** the repo root, absolute POSIX (since lanes may live outside the repo; older lanes have none) */
	root?: string;
	/** the main site the lane merges into, absolute POSIX */
	main?: string;
	owns: string[];
	snapshot: Record<string, string>;
}

/** Owned paths of the wave-1 kit lanes and K0 (design §8.1, §8.2, §8.3). Model lanes use modelOwns(slug). */
export const KNOWN_LANES: Record<string, string[]> = {
	'w1-a': [
		'site/src/lib/atlas/{core,ops,graph}/**',
		'site/src/lib/atlas/schema/{spec,ref,facts}.ts',
		'site/src/lib/atlas/schema/ops.json',
		'pipeline/atlas_fixtures/ops/**',
		'site/tests/atlas/ops/**',
		'notes/devkit-runtime.md'
	],
	'w1-b1': ['site/src/lib/atlas/{autograd,train,data}/**', 'pipeline/atlas_fixtures/autograd/**', 'site/tests/atlas/autograd/**', 'notes/devkit-autograd.md'],
	'w1-c': ['site/src/lib/atlas/{process,lesson,runtime}/**', 'site/src/lib/atlas/schema/tape.ts', 'notes/devkit-lesson.md'],
	'w1-e1': ['pipeline/atlas_kit/**', 'pipeline/requirements.txt', 'site/scripts/atlas/**', 'notes/devkit-tools.md'],
	'w1-b2': [
		'site/src/lib/atlas/engine/**',
		'site/src/lib/atlas/ui/train/**',
		'site/src/lib/atlas/adapters/{points2d,series}/**',
		'site/src/lib/atlas/ui/views/{plane2d,curve1d,loss-curve,train-step,network,train-controls}/**',
		'site/src/routes/dev/atlas-train/**',
		'notes/devkit-engine.md'
	],
	'w1-d1': [
		'site/src/lib/atlas/ui/*.svelte',
		'site/src/lib/atlas/ui/*.ts',
		'site/src/lib/atlas/adapters/{drawing28,text-bpe,text-char}/**',
		'site/src/routes/m/[slug]/**',
		'site/src/routes/dev/atlas/lab/**',
		'notes/devkit-ui.md'
	],
	'w1-d2': [
		'site/src/lib/atlas/ui/views/{zoom.weighted-sum,zoom.reduce,zoom.elementwise,zoom.normalize,zoom.softmax,zoom.lookup,zoom.index-map,zoom.gate,zoom.formula,layer.tensor,compare,tape.table}/**',
		'site/src/lib/atlas/ui/views/kit.*/**',
		'site/src/routes/dev/atlas/+page.svelte',
		'notes/devkit-views.md'
	],
	'w1-e2': [
		'pipeline/atlas_kit/**',
		'site/scripts/atlas/**',
		'site/DEVKIT.md',
		'CONTRIBUTING.md',
		'site/src/lib/explainers/hello-xor/**',
		'pipeline/hello-xor/**',
		'site/static/models/hello-xor/**',
		'notes/devkit-tools.md'
	],
	k0: ['site/src/lib/atlas/**', 'site/scripts/atlas/**', 'pipeline/atlas_kit/**', 'notes/devkit-k0.md']
};

/** The common ownership of a model package (design §8.3). K4: with the reviewer's packet notes/review-<slug>.md, which
 *  `atlas review` writes in the main tree (merge --dry-run lists it with the other paths edited in place). */
export function modelOwns(slug: string): string[] {
	return [`pipeline/${slug}/**`, `site/src/lib/explainers/${slug}/**`, `site/static/models/${slug}/**`, `notes/${slug}.md`, `notes/review-${slug}.md`, `notes/shots/${slug}/**`];
}

/** K4: a lane.json written before K4 owns notes/<slug>.md but not the review packet: the owned paths with it added
 *  (worktree --sync writes them back; merge and ownership read them). Other lists are returned as they are. */
export function withReview(owns: string[], name: string): string[] {
	const review = `notes/review-${name}.md`;
	return owns.includes(`notes/${name}.md`) && !owns.includes(review) ? [...owns.slice(0, owns.indexOf(`notes/${name}.md`) + 1), review, ...owns.slice(owns.indexOf(`notes/${name}.md`) + 1)] : owns;
}

function expandBraces(p: string): string[] {
	const m = /\{([^{}]*)\}/.exec(p);
	if (!m) return [p];
	return m[1].split(',').flatMap((alt) => expandBraces(p.slice(0, m.index) + alt + p.slice(m.index + m[0].length)));
}

function globRe(p: string): RegExp {
	let re = '';
	for (let i = 0; i < p.length; i++) {
		const c = p[i];
		if (c === '*' && p[i + 1] === '*') {
			// 'a/**' matches 'a' itself too; '**/' any number of folders
			if (p[i + 2] === '/') {
				re += '(?:.*/)?';
				i += 2;
			} else {
				if (re.endsWith('/')) re = re.slice(0, -1) + '(?:/.*)?';
				else re += '.*';
				i += 1;
			}
		} else if (c === '*') re += '[^/]*';
		else if (c === '?') re += '[^/]';
		else re += c.replace(/[.+^$()|[\]\\]/g, '\\$&');
	}
	return new RegExp(`^${re}$`);
}

/** A matcher over root-relative POSIX paths. */
export function matcher(globs: string[]): (rootRel: string) => boolean {
	const res = globs.flatMap(expandBraces).map(globRe);
	return (p) => res.some((r) => r.test(p));
}

export function sha256(file: string): string {
	return createHash('sha256').update(readFileSync(file)).digest('hex');
}

/** Every file of a site folder that lanes carry (site-relative POSIX paths), sorted. */
export function siteFiles(site: string, rel = ''): string[] {
	const dir = rel ? join(site, ...rel.split('/')) : site;
	if (!existsSync(dir)) return [];
	const out: string[] = [];
	// K2: the directory entries carry their type, so no stat per file (5.3 s -> 1.0 s for a lane on exFAT)
	const ents = readdirSync(dir, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
	for (const d of ents) {
		const name = d.name;
		const r = rel ? `${rel}/${name}` : name;
		const isDir = d.isSymbolicLink() ? statSync(join(dir, name)).isDirectory() : d.isDirectory();
		if (isDir) {
			if (!SKIP_DIRS.includes(name)) out.push(...siteFiles(site, r));
		} else if (!(rel === '' && SKIP_FILES.includes(name)) && !name.endsWith('.tmp')) out.push(r);
	}
	return out;
}

/** server.fs.allow of a lane in site/.work/<name>/site: the lane and the main site (its node_modules). */
export const ALLOW_IN_MAIN = ['.', '../../..'];
/** server.fs.allow of a lane outside the main site (<lanes>/<name>/site): the lane and the node_modules the lanes share
 *  (<lanes>/node_modules, `atlas lanes --deps`). Nothing else of the repo is served to the browser. */
export const ALLOW_OUTSIDE = ['.', '../../node_modules'];

// The lane's vite.config.ts (idempotent): the lane-only `allow` in server.fs (ALLOW_IN_MAIN, or ALLOW_OUTSIDE for a lane
// outside the main site), and no .work pattern in server.watch.ignored. A lane in site/.work/<name>/site matched main's
// pattern with every file, so its dev server never saw an edit (K0); a lane has no .work folder of its own.
export function patchViteConfig(text: string, allow: string[] = ALLOW_IN_MAIN): { text: string; patched: boolean } {
	const r = patchFsAllow(text, allow);
	const t = r.text.replace(/(['"])\*\*\/\.work\/\*\*\1\s*,\s*/g, '').replace(/,\s*(['"])\*\*\/\.work\/\*\*\1/g, '');
	return { text: t, patched: r.patched };
}

function patchFsAllow(text: string, allow: string[]): { text: string; patched: boolean } {
	const list = '[' + allow.map((a) => `'${a}'`).join(', ') + ']';
	const esc = (s: string) => s.replace(/[.*+?^$(){}|[\]\\/]/g, '\\$&');
	const done = new RegExp('allow:\\s*\\[\\s*' + allow.map((a) => esc(`'${a}'`)).join('\\s*,\\s*') + '\\s*\\]');
	if (done.test(text)) return { text, patched: true };
	const strict = /(\n([ \t]*)strict:\s*true)(\s*\n)/;
	let m = strict.exec(text);
	if (m) return { text: text.replace(strict, (_all, a: string, ind: string, b: string) => `${a},\n${ind}allow: ${list}${b}`), patched: true };
	m = /(\n([ \t]*)fs:\s*\{)/.exec(text);
	if (m) return { text: text.replace(/(\n([ \t]*)fs:\s*\{)/, (_all, a: string, ind: string) => `${a}\n${ind}\tallow: ${list},`), patched: true };
	return { text, patched: false };
}

// K2 speed pack: the ownership question of `atlas check` (what did the lane change, and is any of it outside its owned
// paths?) with the rules of planMerge (commands/worktree.ts) but parallel I/O: an async walk and up to 16 files read and
// hashed at a time. On the shared exFAT drive the sequential planMerge took 11-39 s for a lane and main (two trees of
// ~1,200 files); this takes a few seconds. Exact content hashes, no cache. Merge and sync keep using planMerge.
async function filesAsync(site: string, rel = ''): Promise<string[]> {
	const dir = rel ? join(site, ...rel.split('/')) : site;
	let ents;
	try {
		ents = await readdir(dir, { withFileTypes: true });
	} catch {
		return [];
	}
	const out: string[] = [];
	const subs: Promise<string[]>[] = [];
	for (const d of ents) {
		const r = rel ? `${rel}/${d.name}` : d.name;
		const isDir = d.isSymbolicLink() ? statSync(join(dir, d.name)).isDirectory() : d.isDirectory();
		if (isDir) {
			if (!SKIP_DIRS.includes(d.name)) subs.push(filesAsync(site, r));
		} else if (!(rel === '' && SKIP_FILES.includes(d.name)) && !d.name.endsWith('.tmp')) out.push(r);
	}
	for (const s of await Promise.all(subs)) out.push(...s);
	return out;
}

async function hashAll(site: string, rels: string[], limit = 16): Promise<Map<string, string>> {
	const out = new Map<string, string>();
	let i = 0;
	await Promise.all(
		Array.from({ length: limit }, async () => {
			while (i < rels.length) {
				const rel = rels[i++];
				try {
					out.set(rel, createHash('sha256').update(await readFile(join(site, ...rel.split('/')))).digest('hex'));
				} catch {
					/* gone: null */
				}
			}
		})
	);
	return out;
}

export interface LaneScan {
	copy: string[];
	remove: string[];
	outside: string[];
	both: string[];
	owns: string[];
	noSnapshot: boolean;
}

/** What the lane changed against its snapshot (planMerge's rules), with parallel I/O. `fallbackOwns` applies without a
 *  lane.json. */
export async function scanLane(main: string, lane: string, laneJson: string, fallbackOwns: string[]): Promise<LaneScan> {
	const lf = existsSync(laneJson) ? (JSON.parse(readFileSync(laneJson, 'utf8').replace(/^﻿/, '')) as LaneFile) : null;
	const ownsList = lf ? withReview(lf.owns, lf.name) : fallbackOwns;
	const owns = matcher(ownsList);
	const [mf, lfiles] = await Promise.all([filesAsync(main), filesAsync(lane)]);
	const [hm, hl] = await Promise.all([hashAll(main, mf), hashAll(lane, lfiles)]);
	const res: LaneScan = { copy: [], remove: [], outside: [], both: [], owns: ownsList, noSnapshot: !lf };
	for (const rel of [...new Set([...mf, ...lfiles])].sort()) {
		if (LANE_SETUP.includes(rel) || LANE_LOCAL.includes(rel)) continue;
		const m = hm.get(rel) ?? null;
		const l = hl.get(rel) ?? null;
		const snap = lf ? (lf.snapshot[rel] ?? null) : m;
		if (l === snap || l === m) continue;
		if (!owns(`site/${rel}`)) res.outside.push(rel);
		else if (lf && m !== snap) res.both.push(rel);
		else if (l === null) res.remove.push(rel);
		else res.copy.push(rel);
	}
	return res;
}

// ------------------------------------------------------------------ the node_modules that lanes outside the main site share
export interface DepsState {
	/** false: the lanes live inside the main site and resolve its node_modules */
	shared: boolean;
	/** the node_modules folder the lanes resolve */
	dir: string;
	ok: boolean;
	/** 'up to date', 'not installed', 'stale (...)' */
	why: string;
}

/** The state of the lanes' packages. Lanes outside the main site share <lanes>/node_modules, installed by
 *  `atlas lanes --deps` with `npm ci` from a copy of the main site's package.json and package-lock.json kept next to it:
 *  it is up to date while that copy of the lock equals the main site's lock. */
export function depsState(lanes: LanesChoice, main: string): DepsState {
	if (!lanes.outside) {
		const ok = existsSync(join(lanes.deps, 'vite', 'package.json'));
		return { shared: false, dir: lanes.deps, ok, why: ok ? "the main site's node_modules" : 'not installed (npm ci in the main site)' };
	}
	if (!existsSync(join(lanes.deps, '.package-lock.json'))) return { shared: true, dir: lanes.deps, ok: false, why: 'not installed' };
	const mine = join(lanes.dir, 'package-lock.json');
	const want = join(main, 'package-lock.json');
	if (!existsSync(want)) return { shared: true, dir: lanes.deps, ok: true, why: 'installed (the main site has no package-lock.json to compare)' };
	if (!existsSync(mine) || sha256(mine) !== sha256(want)) return { shared: true, dir: lanes.deps, ok: false, why: "stale (the main site's package-lock.json changed since the install)" };
	return { shared: true, dir: lanes.deps, ok: true, why: "up to date with the main site's package-lock.json" };
}
