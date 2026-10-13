// /glossary/ (redesign spec §6.6): where each term is used, at build time. Every kit lesson's steps name their glossary
// terms (StepDef.terms); this reads the lesson modules on the server (none of their code reaches the page) and gives each
// term the lessons that use it, with the first step that does, as compact rows. The five hand-built lessons do not list
// their terms, so they are not in "Used in".
import { groups, liveLessons } from '$lib/kit/catalog/lessons';
import type { PageServerLoad } from './$types';

interface StepLike {
	id?: string;
	terms?: string[];
}
const lessonModules = import.meta.glob<{ steps?: StepLike[] }>('../../lib/explainers/*/lesson.ts', { eager: true, import: 'default' });

export const load: PageServerLoad = () => {
	const live = liveLessons();
	const bySlug = new Map(live.map((l, i) => [l.slug, i]));
	/** term id -> [lesson index, step number (1-based), step id] for the first step of each lesson that uses it */
	const used: Record<string, [number, number, string][]> = {};
	for (const [path, def] of Object.entries(lessonModules)) {
		const slug = /explainers\/([^/]+)\/lesson\.ts$/.exec(path)?.[1];
		const li = slug ? bySlug.get(slug) : undefined;
		if (li === undefined || !def?.steps) continue;
		const seen = new Set<string>();
		def.steps.forEach((s, k) => {
			for (const id of s.terms ?? []) {
				if (seen.has(id) || !s.id) continue;
				seen.add(id);
				(used[id] ??= []).push([li, k + 1, s.id]);
			}
		});
	}
	// the areas that have live lessons, in catalog order, then the training methods as one group
	const areas = groups().filter((g) => g.kind === 'model' && g.live);
	return {
		used,
		lessons: live.map((l) => ({ title: l.title, href: l.href, group: l.kind === 'model' ? l.group : 'methods' })),
		areas: areas.map((g) => ({ slug: g.slug, name: g.name }))
	};
};
