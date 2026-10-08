// Build-time package discovery for /m/[slug] and the Lesson Lab (design §5.1, server only).
//   - packages are found with a server-side glob of explainers/*/explainer.ts (their code never reaches the client here);
//   - a package is INCLUDED when its catalog status is 'live' or drafts are included (ATLAS_INCLUDE_DRAFTS=1; the dev
//     server includes them implicitly); devOnly packages are never routed under /m/;
//   - for an included package the page embeds spec.json and the measured bytes of assets.auto.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { dev } from '$app/environment';
import { MODEL_BY_SLUG } from '$lib/catalog/models';
import type { ExplainerManifest } from '../types/explainer.ts';

type ManifestModule = { default?: ExplainerManifest; manifest?: ExplainerManifest };
const modules = import.meta.glob<ManifestModule>('../../explainers/*/explainer.ts', { eager: true });

const STATIC = resolve('static');

/** ATLAS_INCLUDE_DRAFTS=1 (any value but '' and '0'), or the dev server. */
export function includeDrafts(): boolean {
	const v = process.env.ATLAS_INCLUDE_DRAFTS;
	return dev || (!!v && v !== '0');
}

/** slug -> manifest of every package (devOnly included). */
export function serverManifests(): Record<string, ExplainerManifest> {
	const out: Record<string, ExplainerManifest> = {};
	for (const [path, mod] of Object.entries(modules)) {
		const slug = /explainers\/([^/]+)\/explainer\.ts$/.exec(path)?.[1];
		const m = mod.default ?? mod.manifest;
		if (slug && m && m.slug === slug) out[slug] = m;
	}
	return out;
}

export interface AutoPageData {
	slug: string;
	/** spec.json of the package (raw; the page normalizes it) */
	spec: unknown;
	/** bytes of manifest.assets.auto (+ the presets file), measured now */
	autoBytes: number;
	dir: string;
}

function fileBytes(dir: string, name: string): number {
	const p = join(STATIC, dir, name);
	if (!existsSync(p)) return 0;
	const st = statSync(p);
	if (!st.isDirectory()) return st.size;
	return readdirSync(p).reduce((a, n) => a + fileBytes(join(dir, name), n), 0);
}

/** The auto page data for /m/<slug>, or null (no package, devOnly, or a draft while drafts are excluded). */
export function autoPage(slug: string): AutoPageData | null {
	const m = serverManifests()[slug];
	// K2: a concept package (type 'concept') renders only at /train/<slug>/ (ui/concept/discover.server.ts), never under /m/
	if (!m || m.devOnly || (m.type as string) === 'concept') return null;
	const live = MODEL_BY_SLUG[slug]?.status === 'live';
	if (!live && !includeDrafts()) return null;
	const dir = m.assets?.dir ?? `models/${slug}`;
	let spec: unknown = null;
	const specPath = join(STATIC, dir, 'spec.json');
	if (existsSync(specPath)) {
		try {
			spec = JSON.parse(readFileSync(specPath, 'utf8').replace(/^﻿/, ''));
		} catch {
			spec = null;
		}
	}
	const files = new Set(m.assets?.auto ?? ['spec.json', 'weights.bin']);
	if (m.input?.presets) files.add(m.input.presets);
	let autoBytes = 0;
	for (const f of files) autoBytes += fileBytes(dir, f);
	return { slug, spec, autoBytes, dir };
}

/** Lab entries: every package slug plus every static/models/<dir> with a spec.json (the shipped TinyCNN and TinyGPT),
 *  and the top-level file names of each such folder (so the lab knows about presets.json or tokenizer.json without
 *  probing URLs that would 404 and log console errors). */
export function labSlugs(): { packages: string[]; specDirs: string[]; files: Record<string, string[]> } {
	const packages = Object.keys(serverManifests()).sort();
	const root = join(STATIC, 'models');
	const specDirs = existsSync(root) ? readdirSync(root).filter((d) => existsSync(join(root, d, 'spec.json'))).sort() : [];
	const files: Record<string, string[]> = {};
	for (const d of specDirs) files[`models/${d}`] = readdirSync(join(root, d)).filter((n) => !statSync(join(root, d, n)).isDirectory());
	return { packages, specDirs, files };
}
