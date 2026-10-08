// A gate result cache for `atlas check <slug> --quick` (K2 speed pack). A gate whose inputs did not change since it last
// PASSED (same files, sizes and modification times) is reported from .atlas/<slug>/gate-cache.json instead of being run
// again. Only --quick reads it; every other check runs the gate and refreshes the cache. Used for parity and the package's
// op tests, whose Python twins and references took most of a fast check (hello-xor parity 50-150 s; gridworld's Python
// reference test 111-190 s) and whose inputs rarely change while a lesson is being written.
import { existsSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { readJson, writeJson } from './io.ts';

export interface CachedGate {
	key: string;
	at: string;
	summary: string;
	status: 'pass' | 'warn';
	issues: { code: string; level: string; message: string; where?: string; fix?: string }[];
}

/** Folders whose files never matter for a gate's inputs (training state, caches). */
const SKIP = new Set(['node_modules', '__pycache__', '.pytest_cache', 'runs', 'logs', '.atlas', '.svelte-kit', 'build', '.work']);

/** A signature of every file under the given folders and files: relative path, size and mtime. */
export function inputsKey(paths: string[], extra = ''): string {
	const h = createHash('sha256');
	h.update(extra);
	const walk = (dir: string, rel: string) => {
		let names: string[];
		try {
			names = readdirSync(dir).sort();
		} catch {
			return;
		}
		for (const n of names) {
			const p = join(dir, n);
			let st;
			try {
				st = statSync(p);
			} catch {
				continue;
			}
			if (st.isDirectory()) {
				if (!SKIP.has(n)) walk(p, `${rel}/${n}`);
			} else h.update(`${rel}/${n}\0${st.size}\0${st.mtimeMs}\n`);
		}
	};
	for (const p of paths) {
		h.update(`#${p}\n`);
		if (!existsSync(p)) continue;
		const st = statSync(p);
		if (st.isDirectory()) walk(p, '');
		else h.update(`${st.size}\0${st.mtimeMs}\n`);
	}
	return h.digest('hex');
}

const file = (site: string, slug: string) => join(site, '.atlas', slug, 'gate-cache.json');

export function readCached(site: string, slug: string, gate: string, key: string): CachedGate | null {
	const f = file(site, slug);
	if (!existsSync(f)) return null;
	try {
		const all = readJson<Record<string, CachedGate>>(f);
		const c = all[gate];
		return c && c.key === key ? c : null;
	} catch {
		return null;
	}
}

/** Remember a gate that passed (or warned); a failed gate removes its entry. */
export function writeCached(site: string, slug: string, gate: string, entry: CachedGate | null): void {
	const f = file(site, slug);
	let all: Record<string, CachedGate> = {};
	try {
		if (existsSync(f)) all = readJson<Record<string, CachedGate>>(f);
	} catch {
		all = {};
	}
	if (entry) all[gate] = entry;
	else delete all[gate];
	writeJson(f, all, 1);
}
