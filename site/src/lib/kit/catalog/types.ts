/**
 * The catalog pages' data (redesign spec §6): a compact index of the live lessons, the planned ones, the groups (areas
 * and training-method sections), the learning paths and the counts. It is computed at build time by the pages' server
 * loads (`lessons.ts`, server side only) and read from `page.data`, so the catalog pages ship none of the catalog
 * modules in their JS (the same pattern as `catalog/slices.ts` for model pages). Types only: components import this.
 */
import type { I18n } from '$lib/i18n/types';

/** The five kinds of input the home page and the atlas filter by (catalog input kinds grouped as a learner sees them). */
export type InputBucket = 'draw' | 'type' | 'points' | 'play' | 'graphs';

/** A tile of a thumbnail sprite (`static/thumbs/`): the CSS background size and position of that tile. */
export interface Tile {
	size: string;
	pos: string;
}

/** One live lesson: a model page (/m/<slug>/) or a training-method page (/train/<slug>/). */
export interface LessonItem {
	kind: 'model' | 'concept';
	slug: string;
	/** the page path without the base, e.g. /m/tinycnn/ */
	href: string;
	/** the area (models) or the section (training methods) */
	group: string;
	title: I18n;
	/** one line for cards and rows: the catalog hook, or for a training method without one the lesson summary cut at a
	 *  sentence or clause boundary (spec §6.2; the content wave writes the real hooks) */
	hook: I18n;
	steps: number;
	/** catalog lesson.minutes, else the spec's formula with the audited averages (≈) */
	minutes: number;
	/** the input words for the meta line (Draw, Type, …) */
	inputs: I18n[];
	buckets: InputBucket[];
	/** the rows sprite tile, and the cards sprite tile when the lesson has one */
	thumb?: Tile;
	card?: Tile;
}

/** A planned lesson (a name in a "Planned" line; the area page also says what it will show). */
export interface PlannedItem {
	kind: 'model' | 'concept';
	slug: string;
	href: string;
	group: string;
	title: I18n;
	buckets: InputBucket[];
	shows?: I18n;
}

/** An area (models) or a training-method section, with its live and planned counts. */
export interface GroupItem {
	kind: 'model' | 'concept';
	slug: string;
	name: I18n;
	blurb: I18n;
	examples?: I18n;
	live: number;
	planned: number;
	/** the learning path that mostly runs through this group */
	path?: { id: string; title: I18n };
}

export interface PathStation {
	kind: 'model' | 'concept';
	slug: string;
	live: boolean;
	/** catalog short name, else the name */
	name: I18n;
	steps?: number;
	href: string;
}

export interface PathItem {
	id: string;
	title: I18n;
	blurb: I18n;
	stations: PathStation[];
	live: number;
	coming: number;
	minutes: number;
}

export interface CatalogCounts {
	lessons: number;
	models: number;
	methods: number;
	planned: number;
	/** areas with at least one live model */
	liveAreas: number;
	areas: number;
}

/** The whole index the atlas and /train/ show. */
export interface CatalogIndex {
	lessons: LessonItem[];
	planned: PlannedItem[];
	groups: GroupItem[];
	counts: CatalogCounts;
	/** live lessons per input bucket */
	inputs: { bucket: InputBucket; n: number }[];
	/** the thumbnail sprite files (relative to the site root) */
	sprites: { rows?: { light: string; dark: string }; cards?: { light: string; dark: string } };
}

/**
 * The home page's preview of a real TinyCNN run (spec §10.5): the shipped weights on the lesson's HOUSE drawing, run at
 * build time by `src/routes/+page.server.ts`. Each map is 8-bit base64: the input as value × 255 (0..1), the signed maps
 * as (value / max + 1) / 2 × 255 over their layer's symmetric range, so HeroPreview can draw them with the site's colour
 * scales and no model code ships with the home page.
 */
export interface HeroData {
	input: string;
	conv: string;
	pool1: string;
	pool2: string;
	/** max |value| of each layer (the diverging domain) */
	max: { conv: number; pool1: number; pool2: number };
	/** the five most likely classes, most likely first */
	top: { en: string; zh: string; p: number }[];
	/** learned numbers in the model (spec.json params) */
	params: number;
}

/** One entry of the search index (search-index.json; SearchPalette loads it on first use). */
export interface SearchEntry {
	/** l lesson, p planned, a area or section, t glossary term */
	k: 'l' | 'p' | 'a' | 't';
	en: string;
	zh: string;
	href: string;
	/** a second line: the hook (lessons), the area (planned), nothing for terms */
	sub?: [en: string, zh: string];
}
