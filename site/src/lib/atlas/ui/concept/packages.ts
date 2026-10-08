// The browser loader for what only a concept package has (the rest of the package loads through ../packages.ts, the same
// loader model pages use): explainers/<slug>/concept.ts (the page around the lesson) and the concept-page section
// overrides explainers/<slug>/sections/<Name>.svelte (props { store, pkg, def }). Every glob is lazy: a concept page loads
// only its own package.
import type { Component } from 'svelte';
import { conceptOf, type ConceptDef } from './panels.ts';

const conceptMods = import.meta.glob<Record<string, unknown>>('../../../explainers/*/concept.ts');
const sectionMods = import.meta.glob<Component<any>>('../../../explainers/*/sections/*.svelte', { import: 'default' });

const pkgPath = (slug: string, file: string) => `../../../explainers/${slug}/${file}`;

export async function loadConcept(slug: string): Promise<ConceptDef> {
	const l = conceptMods[pkgPath(slug, 'concept.ts')];
	return conceptOf(l ? await l() : null);
}

/** Concept-page sections a package may replace with sections/<Name>.svelte (Lesson, Checkpoint and Credits are shared with
 *  model pages; the others are concept-only). */
export const CONCEPT_SECTIONS = ['Hook', 'Lesson', 'Toy', 'Real', 'Compare', 'Pitfalls', 'Checkpoint', 'Links'] as const;
export type ConceptSection = (typeof CONCEPT_SECTIONS)[number];

export async function conceptOverrides(slug: string): Promise<Partial<Record<ConceptSection, Component<any>>>> {
	const out: Partial<Record<ConceptSection, Component<any>>> = {};
	for (const n of CONCEPT_SECTIONS) {
		const l = sectionMods[pkgPath(slug, `sections/${n}.svelte`)];
		if (l) out[n] = await l();
	}
	return out;
}
