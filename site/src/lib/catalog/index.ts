/**
 * Catalog entry point. Import from '$lib/catalog' in pages; scripts import the .ts files directly.
 * A model page imports none of this: it reads its slice from page.data (slices.ts, computed by the root layout's
 * server load).
 */
export * from './types.ts';
export { AREAS, AREA_BY_SLUG, isAreaSlug } from './areas.ts';
export { MODELS, MODEL_BY_SLUG, sortModels } from './models.ts';
export {
	SECTIONS,
	SECTION_BY_SLUG,
	CONCEPTS,
	CONCEPT_BY_SLUG,
	LINK_ALIASES,
	conceptsBySection,
	conceptsForModel,
	conceptCard
} from './concepts.ts';
export { modelCard } from './cards.ts';
export { GLOSSARY, GLOSSARY_BY_ID } from './glossary.ts';
export { PATHS } from './paths.ts';

import { MODELS } from './models.ts';
import type { InputKind, ModelEntry } from './types.ts';

/** Models whose inputs include `kind` (primary input first, live before planned). */
export function modelsAccepting(kind: InputKind): ModelEntry[] {
	const rank = { live: 0, precomputed: 1, planned: 2 } as const;
	return MODELS.filter((m) => m.inputs.includes(kind)).sort(
		(a, b) =>
			rank[a.status] - rank[b.status] ||
			Number(b.inputs[0] === kind) - Number(a.inputs[0] === kind) ||
			({ P0: 0, V1: 1, P1: 2, P2: 3 } as const)[a.tier] - ({ P0: 0, V1: 1, P1: 2, P2: 3 } as const)[b.tier]
	);
}

/** Routes for live pages that are built as static folders (not by the [slug] routes). */
export const LIVE_MODEL_SLUGS = MODELS.filter((m) => m.status === 'live').map((m) => m.slug);
