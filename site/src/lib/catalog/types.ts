/**
 * Catalog types. Pure TS (runs under Node type stripping: no enums, no namespaces,
 * imports use explicit .ts extensions).
 */
import type { I18n } from '../i18n/types.ts';

export type { I18n };

/** The 11 application areas (brief §4.1). */
export type AreaSlug =
	| 'vision'
	| 'language'
	| 'audio'
	| 'time-series'
	| 'recommendation'
	| 'tabular'
	| 'rl'
	| 'graphs'
	| 'generative'
	| 'multimodal'
	| 'science';

/** P0 = first version, V1 = flagship that completes v1 (one per area), P1 = next, P2 = later. */
export type Tier = 'P0' | 'V1' | 'P1' | 'P2';

/** Page status. `live` pages exist; `planned` pages render a "Planned" placeholder. */
export type Status = 'live' | 'precomputed' | 'planned';

/** What the learner gives the model (brief §4.5). */
export type InputKind =
	| 'drawing'
	| 'image'
	| 'webcam'
	| 'text'
	| 'audio'
	| 'series'
	| 'tabular'
	| 'points2d'
	| 'graph'
	| 'game'
	| 'protein'
	| 'ratings'
	| 'conversation'
	| 'video'
	| 'pointcloud'
	| 'tokens'
	| 'none';

/** How numbers on the finished page are planned to be produced. */
export type Mode = 'live' | 'precomputed' | 'both';

export interface Area {
	slug: AreaSlug;
	/** 1-based display order */
	n: number;
	name: I18n;
	blurb: I18n;
	/** a few example families, shown on area cards */
	examples: I18n;
}

export interface ModelEntry {
	slug: string;
	area: AreaSlug;
	/** Display name (model names are usually the same in both languages). */
	name: I18n;
	family: I18n;
	tier: Tier;
	/** Extra tier context, e.g. "P0 for precomputed data, P1 for live". */
	tierNote?: I18n;
	status: Status;
	/** Accepted inputs, primary first. */
	inputs: InputKind[];
	/** true = a tiny model we train ourselves; false = an existing (Hub) model. */
	own: boolean;
	/** Approximate parameter count as text (design targets are marked). */
	params?: string;
	/** Approximate download as text, as given in the brief. Live pages show the measured size. */
	size?: string;
	runtime: string;
	mode: Mode;
	/** One-line hook: at most 100 English and 50 Chinese characters, written to fit two lines of a card, never cut. */
	hook: I18n;
	/** A short name (at most 24 characters) for path stations; the name is used when absent. */
	short?: I18n;
	/** What the page will show (layers or steps to animate). */
	shows: I18n;
	/** Caveats such as unverified licenses. */
	notes?: I18n;
	/** Model slugs worth reading first. */
	prereqs: string[];
	/** The page has a guided step-by-step lesson (section #lesson); the home page links to it. */
	lesson?: LessonInfo;
}

/** A page's guided lesson, for links from the home page and cards. */
export interface LessonInfo {
	steps: number;
	/** one line: what the lesson walks through */
	summary: I18n;
	/** reading time in minutes: round(words / 180 + steps x 20 s + beats x 30 s), from `atlas lesson <slug> print` */
	minutes?: number;
}

export type SectionSlug =
	| 'foundations'
	| 'pretraining'
	| 'finetuning'
	| 'peft'
	| 'alignment'
	| 'compression'
	| 'interpretability'
	| 'evaluation'
	| 'inference'
	| 'robustness';

export interface ConceptSection {
	slug: SectionSlug;
	/** brief section number, e.g. "6.1" */
	n: string;
	name: I18n;
	blurb: I18n;
}

export interface ConceptEntry {
	slug: string;
	section: SectionSlug;
	title: I18n;
	status: 'live' | 'planned';
	/** part of the ~11-page v1 freeze */
	v1: boolean;
	/** P0 = part of the first version; P2 = explicitly later */
	tier?: 'P0' | 'P2';
	/** One-line hook for cards (at most 100 English and 50 Chinese characters); the lesson summary, cut at its last full
	 *  stop before 100 characters, stands in until the content wave writes one. */
	hook?: I18n;
	/** A short title (at most 24 characters) for path stations; the title is used when absent. */
	short?: I18n;
	/** the playable interaction, summarized */
	interaction: I18n;
	engine: string;
	mode: Mode;
	/** catalog model slugs (validated by scripts/check-catalog.ts) */
	links: string[];
	/** The page has a guided step-by-step lesson (section #lesson). */
	lesson?: LessonInfo;
}

export interface GlossaryTerm {
	id: string;
	term: I18n;
	def: I18n;
	/** other glossary ids */
	related?: string[];
}

/** A step in a learning path: a model or a concept page. */
export interface PathStep {
	kind: 'model' | 'concept';
	slug: string;
}

export interface LearningPath {
	id: string;
	title: I18n;
	blurb: I18n;
	steps: PathStep[];
}

/**
 * What a catalog card shows, already resolved (area or section name, bilingual technical text), so a card renders
 * without the catalog modules. ModelCard / ConceptCard build it on list pages (catalog/cards.ts, concepts.ts); a model
 * page gets it precomputed in its ModelPageCatalog. Rendered by kit/CatalogCard.svelte.
 */
export interface CardData {
	kind: 'model' | 'concept';
	slug: string;
	title: I18n;
	status: Status;
	tier?: Tier;
	/** the model's area, or the concept's section when the card shows it */
	area?: I18n;
	family?: I18n;
	inputs?: InputKind[];
	/** a planned model's approximate download (bilingual); live models show the measured size (root layout liveSizes) */
	size?: I18n;
	/** the one-line hook; left out of the cards a ModelPageCatalog carries (compact cards do not show it) */
	hook?: I18n;
	/** a concept page of the v1 freeze (corner note) */
	v1?: boolean;
}

/**
 * The slice of the catalog one model page shows (ModelPageShell): computed at build time by the root layout's server load
 * (catalog/slices.ts) for every /m/<slug>/ route and the Lesson Lab, and read from `page.data.catalog`. Model pages
 * therefore load none of the catalog modules: their JS stays the same when other catalog entries change.
 */
export interface ModelPageCatalog {
	/** the page's own catalog entry */
	model: ModelEntry;
	/** its area's name */
	area: I18n;
	/** the entry's technical strings as bilingual text (tech-zh.ts) */
	tech: { runtime: I18n; params?: I18n; size?: I18n };
	/** concept pages that use this model (compact cards: no hook) */
	usedBy: CardData[];
	/** the entry's prerequisites (compact cards) */
	prereqs: CardData[];
	/** models that list this one as a prerequisite, at most 6 (compact cards) */
	next: CardData[];
}

/** Site-wide counts for the footer (root layout server load). */
export interface SiteCounts {
	models: number;
	concepts: number;
	/** live models + live concept pages */
	live: number;
}
