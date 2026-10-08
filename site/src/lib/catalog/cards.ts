/**
 * A model's card data (CardData in types.ts): the entry with its area's name and the bilingual size text resolved. Used by
 * kit/ModelCard.svelte on list pages and by slices.ts for the cards of a model page. Concept cards: conceptCard() in
 * concepts.ts (kept there so a page with concept cards only does not load the areas and tech-zh tables).
 */
import type { CardData, ModelEntry } from './types.ts';
import { AREA_BY_SLUG } from './areas.ts';
import { techText } from './tech-zh.ts';

export function modelCard(m: ModelEntry): CardData {
	const card: CardData = {
		kind: 'model',
		slug: m.slug,
		title: m.name,
		status: m.status,
		tier: m.tier,
		area: AREA_BY_SLUG[m.area].name,
		family: m.family,
		inputs: m.inputs,
		hook: m.hook
	};
	if (m.status !== 'live' && m.size) card.size = techText(m.size);
	return card;
}
