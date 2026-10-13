// Node-only helpers for W1-C tests: ATLAS_SITE / ATLAS_ROOT resolution (design §3.1) and a file-system FetchLike.
// ATLAS_SITE: env ATLAS_SITE, else the site folder this file lives in (works in a lane copy).
// ATLAS_ROOT: env ATLAS_ROOT, else (K4) the `root` a lane's lane.json records (a lane outside the repo, e.g. on another
// drive, where no walk up finds the repo; tests run with plain `node --test` there too), else the nearest ancestor of
// ATLAS_SITE that contains CONTRACTS.md and pipeline/ (from site/.work/<lane>/site this skips site/.work/<lane>).
// K4: repoFile(...p) is a file of the repo for claim tests that read another package's pipeline folder
// (`repoFile('pipeline', 'tinycnn', 'runs', 'x.json')`); import it from '../../atlas/process/fixtures/paths.ts'.
// scripts/atlas/root.ts (W1-E1) is the CLI's version of the same rule; this copy keeps the tests independent of it.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { FetchLike } from '../../types/core.ts';

const here = dirname(fileURLToPath(import.meta.url));

export const ATLAS_SITE = process.env.ATLAS_SITE ? resolve(process.env.ATLAS_SITE) : resolve(here, '..', '..', '..', '..', '..');

const isRoot = (d: string) => existsSync(join(d, 'CONTRACTS.md')) && existsSync(join(d, 'pipeline'));

/** The repo root for a site folder: $ATLAS_ROOT, else its lane.json `root` (<site>/../lane.json), else the nearest
 *  ancestor with CONTRACTS.md and pipeline/; null when there is none (K4: the same order as scripts/atlas/root.ts). */
export function findRepoRoot(site: string, env: NodeJS.ProcessEnv = process.env): string | null {
	if (env.ATLAS_ROOT?.trim()) return resolve(env.ATLAS_ROOT.trim());
	const lj = join(site, '..', 'lane.json');
	if (existsSync(lj)) {
		try {
			const root = (JSON.parse(readFileSync(lj, 'utf8').replace(/^﻿/, '')) as { root?: unknown }).root;
			if (typeof root === 'string' && root && isRoot(resolve(root))) return resolve(root);
		} catch {
			/* not a lane.json we can read: walk up */
		}
	}
	let d = resolve(site);
	for (;;) {
		if (isRoot(d)) return d;
		const up = dirname(d);
		if (up === d) return null;
		d = up;
	}
}

function findRoot(start: string): string {
	const root = findRepoRoot(start);
	if (!root) throw new Error(`no ancestor of ${start} contains CONTRACTS.md and pipeline/, and no lane.json records a root (set ATLAS_ROOT)`);
	return root;
}

export const ATLAS_ROOT = findRoot(ATLAS_SITE);

/** K4: a file of the repo (ATLAS_ROOT), for claim tests: repoFile('pipeline', 'tinycnn', 'runs', 'curve.json'). */
export const repoFile = (...p: string[]): string => join(ATLAS_ROOT, ...p);

export function readJson<T = unknown>(path: string): T {
	return JSON.parse(readFileSync(path, 'utf8').replace(/^﻿/, '')) as T;
}

/** fetch() over a folder: urls are paths relative to `root` (a leading '/' is ignored). */
export function fsFetch(root: string): FetchLike {
	return async (url: string) => {
		const p = join(root, url.replace(/^\/+/, ''));
		if (!existsSync(p)) return { ok: false, status: 404, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0), text: async () => '' };
		const buf = readFileSync(p);
		return {
			ok: true,
			status: 200,
			json: async () => JSON.parse(buf.toString('utf8').replace(/^﻿/, '')),
			arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
			text: async () => buf.toString('utf8')
		};
	};
}
