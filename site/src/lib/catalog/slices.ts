/**
 * Catalog slices: the part of the catalog one page shows, computed at build time by the root layout's server load
 * (src/routes/+layout.server.ts) and read from `page.data`. Pages about one model therefore ship none of the catalog
 * modules (models, concepts, areas, tech-zh) in their JS: the flagship shells stay the same size when catalog entries
 * are added or go live (check:build). List pages (home, /atlas/, /train/, /glossary/) still import the catalog they list.
 *
 * Server side and scripts only: importing this module from a component would bring the whole catalog back into its page.
 */
import type { CardData, ModelPageCatalog, SiteCounts } from './types.ts';
import { MODELS, MODEL_BY_SLUG } from './models.ts';
import { CONCEPTS, conceptCard, conceptsForModel } from './concepts.ts';
import { AREA_BY_SLUG } from './areas.ts';
import { techText } from './tech-zh.ts';
import { modelCard } from './cards.ts';

/** Footer counts (kit/SiteFooter.svelte). */
export function siteCounts(): SiteCounts {
	return {
		models: MODELS.length,
		concepts: CONCEPTS.length,
		live: MODELS.filter((m) => m.status === 'live').length + CONCEPTS.filter((c) => c.status === 'live').length
	};
}

/** A card as ModelPageShell shows it (compact: no hook), so the page data carries only what is shown. */
function compact(c: CardData): CardData {
	const { hook: _hook, ...shown } = c;
	return shown;
}

/** Everything ModelPageShell shows about a catalog model; null for a slug that is not in the catalog. */
export function modelPageCatalog(slug: string): ModelPageCatalog | null {
	const model = MODEL_BY_SLUG[slug];
	if (!model) return null;
	const tech: ModelPageCatalog['tech'] = { runtime: techText(model.runtime) };
	if (model.params) tech.params = techText(model.params);
	if (model.size) tech.size = techText(model.size);
	return {
		model,
		area: AREA_BY_SLUG[model.area].name,
		tech,
		usedBy: conceptsForModel(slug).map((c) => compact(conceptCard(c))),
		prereqs: model.prereqs.map((s) => MODEL_BY_SLUG[s]).filter(Boolean).map((m) => compact(modelCard(m))),
		next: MODELS.filter((m) => m.prereqs.includes(slug)).slice(0, 6).map((m) => compact(modelCard(m)))
	};
}

/**
 * The catalog model a route is about: /m/[slug] (planned and auto pages), a live model's own route folder (/m/tinycnn,
 * /m/tinygpt) and the Lesson Lab (/dev/atlas/lab/[slug]). Every other route gets no model slice.
 */
export function modelSlugOfRoute(routeId: string | null, params: Partial<Record<string, string>>): string | undefined {
	if (!routeId) return undefined;
	if (routeId === '/m/[slug]' || routeId === '/dev/atlas/lab/[slug]') return params.slug;
	return /^\/m\/([a-z0-9-]+)$/.exec(routeId)?.[1];
}
