/**
 * Build-time helpers (run during prerender in Node; never shipped to the browser).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const STATIC = resolve('static');

/** Files a page never loads; excluded from the "download" size: reference.* (reference.json and any sidecar)
 *  and *.ref.* (DevKit reference tapes, e.g. q.ref.tape.json and q.ref.tape.bin). */
const NOT_LOADED = /^reference\.|\.ref\./;

function dirBytes(dir: string): number {
	let total = 0;
	for (const name of readdirSync(dir)) {
		if (NOT_LOADED.test(name)) continue;
		const p = join(dir, name);
		const st = statSync(p);
		total += st.isDirectory() ? dirBytes(p) : st.size;
	}
	return total;
}

/**
 * Bytes of the shipped files under static/models/<slug>/ for each slug that has a folder.
 * Checkpoint replays and other optional files are included: this is an upper bound of what a
 * page may fetch, measured, not estimated.
 */
export function measureModelAssets(slugs: string[]): Record<string, number> {
	const out: Record<string, number> = {};
	for (const slug of slugs) {
		const dir = join(STATIC, 'models', slug);
		if (existsSync(dir) && statSync(dir).isDirectory()) out[slug] = dirBytes(dir);
	}
	return out;
}

/** The small subset of spec.json the catalog pages show. */
export interface LiveSpecFacts {
	params?: number;
	precision?: string;
	input?: { kind: string; shape: (number | string)[] };
	parity?: { against: string; max_abs_diff: number; checked: string };
	version?: string;
}

/** Read spec.json facts for each live slug that already has one. */
export function readLiveSpecs(slugs: string[]): Record<string, LiveSpecFacts> {
	const out: Record<string, LiveSpecFacts> = {};
	for (const slug of slugs) {
		const p = join(STATIC, 'models', slug, 'spec.json');
		if (!existsSync(p)) continue;
		try {
			const s = JSON.parse(readFileSync(p, 'utf8'));
			// Keep only the documented fields: this object is serialized into every prerendered page
			// (root layout data), so extra parity details would be repeated ~190 times.
			const parity = s.parity
				? { against: s.parity.against, max_abs_diff: s.parity.max_abs_diff, checked: s.parity.checked }
				: undefined;
			const input = s.input ? { kind: s.input.kind, shape: s.input.shape } : undefined;
			out[slug] = { params: s.params, precision: s.precision, input, parity, version: s.version };
		} catch {
			/* a half-written spec must not break the build */
		}
	}
	return out;
}

export interface LicenseEntry {
	name: string;
	url: string;
	license: string;
	usedFor: string;
	attribution?: string;
	/** the full licence text the repository ships for material the site redistributes (a path under LICENSES/) */
	licenseText?: string;
}

/** static/licenses.json (every third-party source); [] when the file does not exist yet. */
export function readLicenses(): LicenseEntry[] {
	const p = join(STATIC, 'licenses.json');
	if (!existsSync(p)) return [];
	try {
		const data = JSON.parse(readFileSync(p, 'utf8'));
		return Array.isArray(data) ? data : [];
	} catch {
		return [];
	}
}
