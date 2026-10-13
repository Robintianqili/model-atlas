// The section overrides a lazy panel needs (redesign W2): a package's explainers/<slug>/sections/<Name>.svelte, loaded by
// the panel when its view opens (not with the lesson), so a page that never opens Explore never loads its overrides.
import type { Component } from 'svelte';
import { loadSection, sectionOverrides, type SectionName } from '../packages.ts';

export type Overrides = Partial<Record<SectionName, Component<any>>>;

/** The overrides among `names` that the package has, loaded. */
export async function loadOverrides(slug: string, names: readonly SectionName[]): Promise<Overrides> {
	const have = new Set(sectionOverrides(slug));
	const out: Overrides = {};
	await Promise.all(
		names
			.filter((n) => have.has(n))
			.map(async (n) => {
				const l = loadSection(slug, n);
				if (l) out[n] = await l;
			})
	);
	return out;
}
