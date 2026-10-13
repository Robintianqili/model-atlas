// /atlas/<area>/ (redesign spec §6.5): one area at build time: its live lessons, its path, every planned model with what
// it will show, and the training-method sections whose lessons use its models. Catalog modules stay on the server.
import { error } from '@sveltejs/kit';
import { AREAS, isAreaSlug } from '$lib/catalog/areas';
import { MODELS } from '$lib/catalog/models';
import { CONCEPTS } from '$lib/catalog/concepts';
import { catalogIndex, cutHook, learningPaths } from '$lib/kit/catalog/lessons';
import type { EntryGenerator, PageServerLoad } from './$types';

// Prerender all 11 area pages.
export const entries: EntryGenerator = () => AREAS.map((a) => ({ area: a.slug }));

export const load: PageServerLoad = ({ params }) => {
	if (!isAreaSlug(params.area)) error(404, `Unknown area: ${params.area}`);
	const slug = params.area;
	const index = catalogIndex({ shows: true });
	const group = index.groups.find((g) => g.slug === slug)!;
	const models = new Set(MODELS.filter((m) => m.area === slug).map((m) => m.slug));
	const used = new Set<string>(CONCEPTS.filter((c) => c.links.some((l) => models.has(l))).map((c) => c.section));
	const i = AREAS.findIndex((a) => a.slug === slug);
	const near = (k: number) => {
		const a = AREAS[(k + AREAS.length) % AREAS.length];
		return { slug: a.slug, name: a.name };
	};
	return {
		group,
		lessons: index.lessons.filter((l) => l.group === slug),
		planned: index.planned
			.filter((p) => p.group === slug)
			.map((p) => ({ ...p, shows: p.shows && { en: cutHook(p.shows.en, 130, false), zh: cutHook(p.shows.zh ?? p.shows.en, 64, true) } })),
		path: group.path ? learningPaths(index.lessons).find((p) => p.id === group.path!.id) : undefined,
		sections: index.groups.filter((g) => g.kind === 'concept' && used.has(g.slug)),
		sprites: index.sprites,
		prev: near(i - 1),
		next: near(i + 1)
	};
};
