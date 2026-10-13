import { catalogIndex } from '$lib/kit/catalog/lessons';
import { CONCEPT_BY_SLUG, SECTION_BY_SLUG } from '$lib/catalog/concepts';
import { MODEL_BY_SLUG } from '$lib/catalog/models';
import { modelCard } from '$lib/catalog/cards';
import { techText } from '$lib/catalog/tech-zh';
import type { ConceptPageData } from '$lib/kit/lesson/ConceptHost.svelte';
import type { PageServerLoad } from './$types';

// what the page shows about its concept, resolved here as the kit route train/[slug] does: ConceptHost gets it as data,
// so the page's JS carries no catalog module (ConceptPageShell would import the catalog tables, about 62 KB gzip)
function conceptCatalog(slug: string): ConceptPageData {
	const c = CONCEPT_BY_SLUG[slug];
	const section = SECTION_BY_SLUG[c.section];
	return {
		slug: c.slug,
		title: c.title,
		status: c.status as ConceptPageData['status'],
		interaction: c.interaction,
		engine: techText(c.engine),
		mode: c.mode,
		section: section ? { slug: c.section, n: section.n, name: section.name } : null,
		...(c.lesson ? { lesson: { summary: c.lesson.summary, ...(c.lesson.minutes ? { minutes: c.lesson.minutes } : {}) } } : {}),
		models: c.links
			.map((s) => MODEL_BY_SLUG[s])
			.filter(Boolean)
			.map((m) => {
				const { hook: _hook, ...shown } = modelCard(m);
				return shown;
			})
	};
}

// Build time only. The page is the lesson app (redesign spec §6.1: no site header or footer), and its Sources view lists
// the related lessons as catalog rows: built here from the catalog index, so no catalog module ships with the page.
export const load: PageServerLoad = () => {
	const idx = catalogIndex();
	const group = (title: { en: string; zh: string }, slugs: string[]) => ({
		title,
		lessons: slugs.flatMap((s) => idx.lessons.filter((l) => l.slug === s)),
		planned: slugs.flatMap((s) => idx.planned.filter((p) => p.slug === s)).map((p) => ({ title: p.title, href: p.href }))
	});
	return {
		chrome: 'lesson' as const,
		conceptCatalog: conceptCatalog('tokenization'),
		related: {
			sprites: idx.sprites,
			groups: [
				group({ en: 'Next', zh: '下一步' }, ['next-token-prediction', 'tinygpt']),
				group({ en: 'Models that depend on a tokenizer', zh: '依赖分词器的模型' }, ['tokenizers', 'bert-base', 'tinybert', 'whisper-tiny', 'clip'])
			]
		}
	};
};
