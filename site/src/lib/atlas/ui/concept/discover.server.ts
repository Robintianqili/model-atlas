// Build-time discovery for /train/[slug] (server only; the concept twin of ../discover.server.ts for /m/[slug]):
//   - a concept package is an explainers/<slug>/ folder whose manifest says type: 'concept' and whose slug is a catalog
//     concept (src/lib/catalog/concepts.ts);
//   - it is INCLUDED when the concept's catalog status is 'live' (promote flips it) or drafts are included
//     (ATLAS_INCLUDE_DRAFTS=1; the dev server includes them implicitly); devOnly packages are never routed;
//   - every other slug gets null and renders the Planned page exactly as before.
// For an included package the page data carries the page's plain text (goals, the predict-first question, the hook text,
// pitfalls, the checkpoint quiz, credits, further reading), so the prerendered HTML already shows it; the stages (lesson,
// toy, real, compare) need the runtime and render after mount.
import { existsSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
import { includeDrafts, serverManifests } from '../discover.server.ts';
import { conceptIncluded, conceptOf, hookText, panelLesson, pitfallsOf, type ConceptText } from './panels.ts';

const conceptMods = import.meta.glob<Record<string, unknown>>('../../../explainers/*/concept.ts', { eager: true });
const STATIC = resolve('static');

export interface ConceptPageData {
	slug: string;
	/** bytes of manifest.assets.auto (+ the presets file), measured at build time; 0 for a package without assets */
	autoBytes: number;
	text: ConceptText;
}

function fileBytes(dir: string, name: string): number {
	const p = join(STATIC, dir, name);
	if (!existsSync(p)) return 0;
	const st = statSync(p);
	if (!st.isDirectory()) return st.size;
	return readdirSync(p).reduce((a, n) => a + fileBytes(join(dir, name), n), 0);
}

/** The JSON-safe copy of a value (functions and undefined dropped), so page data never fails to serialize. */
const plain = <T>(v: T): T => JSON.parse(JSON.stringify(v ?? null)) as T;

/** The concept page data for /train/<slug>, or null (no concept package, devOnly, or a draft while drafts are excluded). */
export function conceptPage(slug: string): ConceptPageData | null {
	const m = serverManifests()[slug];
	if (!conceptIncluded(m, CONCEPT_BY_SLUG[slug]?.status, includeDrafts())) return null;
	const dir = m.assets?.dir ?? `models/${slug}`;
	const files = new Set(m.assets?.auto ?? []);
	if (m.input?.presets) files.add(m.input.presets);
	let autoBytes = 0;
	for (const f of files) autoBytes += fileBytes(dir, f);
	const def = conceptOf(conceptMods[`../../../explainers/${slug}/concept.ts`]);
	const text: ConceptText = plain({
		page: m.page,
		credits: m.credits,
		hook: hookText(def),
		pitfalls: pitfallsOf(def, m),
		further: def.further ?? [],
		panels: panelLesson(def, m).keys,
		input: !!m.input
	});
	return { slug, autoBytes, text };
}
