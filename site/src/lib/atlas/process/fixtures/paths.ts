// Node-only helpers for W1-C tests: ATLAS_SITE / ATLAS_ROOT resolution (design §3.1) and a file-system FetchLike.
// ATLAS_SITE: env ATLAS_SITE, else the site folder this file lives in (works in a lane copy).
// ATLAS_ROOT: env ATLAS_ROOT, else the nearest ancestor of ATLAS_SITE that contains CONTRACTS.md and pipeline/
// (from site/.work/<lane>/site this skips site/.work/<lane>, which has no pipeline/).
// scripts/atlas/root.ts (W1-E1) is the CLI's version of the same rule; this copy keeps the tests independent of it.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { FetchLike } from '../../types/core.ts';

const here = dirname(fileURLToPath(import.meta.url));

export const ATLAS_SITE = process.env.ATLAS_SITE ? resolve(process.env.ATLAS_SITE) : resolve(here, '..', '..', '..', '..', '..');

function findRoot(start: string): string {
	let d = start;
	for (;;) {
		if (existsSync(join(d, 'CONTRACTS.md')) && existsSync(join(d, 'pipeline'))) return d;
		const up = dirname(d);
		if (up === d) throw new Error(`no ancestor of ${start} contains CONTRACTS.md and pipeline/ (set ATLAS_ROOT)`);
		d = up;
	}
}

export const ATLAS_ROOT = process.env.ATLAS_ROOT ? resolve(process.env.ATLAS_ROOT) : findRoot(ATLAS_SITE);

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
