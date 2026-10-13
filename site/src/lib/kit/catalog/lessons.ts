/**
 * Build-time builder of the catalog pages' data (types.ts): the live lessons, the planned ones, the areas and
 * training-method sections, the learning paths, the counts and the search index. Server side only (the pages'
 * +page.server.ts and the search-index.json endpoint): importing it from a component would bring the whole catalog
 * back into that page's JS.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { AREAS } from '$lib/catalog/areas';
import { MODELS, sortModels } from '$lib/catalog/models';
import { CONCEPTS, SECTIONS } from '$lib/catalog/concepts';
import { PATHS } from '$lib/catalog/paths';
import { GLOSSARY_NAMES } from '$lib/catalog/glossary-names';
import type { ConceptEntry, I18n, InputKind, ModelEntry } from '$lib/catalog/types';
import type { CatalogIndex, GroupItem, InputBucket, LessonItem, PathItem, PlannedItem, SearchEntry, Tile } from './types';

// ------------------------------------------------------------------ input words and buckets

const WORD: Partial<Record<InputKind, I18n>> = {
	drawing: { en: 'Draw', zh: '画' },
	image: { en: 'Photo', zh: '照片' },
	webcam: { en: 'Camera', zh: '摄像头' },
	text: { en: 'Type', zh: '打字' },
	tokens: { en: 'Type', zh: '打字' },
	conversation: { en: 'Chat', zh: '对话' },
	series: { en: 'Series', zh: '序列' },
	points2d: { en: 'Points', zh: '点' },
	pointcloud: { en: 'Points', zh: '点' },
	tabular: { en: 'Table', zh: '表格' },
	audio: { en: 'Sound', zh: '声音' },
	game: { en: 'Play', zh: '游戏' },
	graph: { en: 'Graph', zh: '图' },
	ratings: { en: 'Ratings', zh: '评分' },
	protein: { en: 'Protein', zh: '蛋白质' },
	video: { en: 'Video', zh: '视频' }
};
const BUCKET: Partial<Record<InputKind, InputBucket>> = {
	drawing: 'draw',
	image: 'draw',
	webcam: 'draw',
	text: 'type',
	tokens: 'type',
	conversation: 'type',
	points2d: 'points',
	pointcloud: 'points',
	series: 'points',
	tabular: 'points',
	game: 'play',
	graph: 'graphs',
	ratings: 'graphs'
};
export const BUCKETS: InputBucket[] = ['draw', 'type', 'points', 'play', 'graphs'];

const uniq = <T>(xs: T[], key: (x: T) => string): T[] => {
	const seen = new Set<string>();
	return xs.filter((x) => (seen.has(key(x)) ? false : (seen.add(key(x)), true)));
};
const inputWords = (inputs: InputKind[]): I18n[] =>
	uniq(
		inputs.map((k) => WORD[k]).filter((w): w is I18n => !!w),
		(w) => w.en
	);
const bucketsOf = (inputs: InputKind[]): InputBucket[] => [...new Set(inputs.map((k) => BUCKET[k]).filter((b): b is InputBucket => !!b))];

// ------------------------------------------------------------------ hooks and minutes (spec §6.2)

/** Cut a summary to a hook of at most `max` characters, never mid-word and never inside brackets: at the last full stop;
 *  else at the last semicolon or colon (a whole clause); else at the last comma (a list stays whole only when it ends
 *  there); the cut ends with a full stop. */
export function cutHook(s: string, max: number, zh: boolean): string {
	if (s.length <= max) return s;
	const head = s.slice(0, max + 1);
	const open = (i: number) => (s.slice(0, i).match(/[(（]/g)?.length ?? 0) - (s.slice(0, i).match(/[)）]/g)?.length ?? 0);
	const last = (re: RegExp): number => {
		let best = -1;
		for (const m of head.matchAll(re)) if (m.index >= max * 0.2 && m.index < max && open(m.index) === 0) best = m.index;
		return best;
	};
	const stop = last(zh ? /[。！？]/g : /[.!?](?=\s|$)/g);
	if (stop > 0) return s.slice(0, stop + 1);
	for (const re of zh ? [/[；：]/g, /，/g] : [/[;:](?=\s)/g, /,(?=\s)/g]) {
		const i = last(re);
		if (i > 0) return s.slice(0, i) + (zh ? '。' : '.');
	}
	const sp = head.lastIndexOf(' ', max);
	return (sp > 0 ? head.slice(0, sp) : head.slice(0, max)) + '…';
}

/** The spec's reading-time formula, round(words / 180 + steps × 20 s + beats × 30 s), with the audit's averages
 *  (65 narration words a step, two beats a lesson) until `atlas lesson print` measures each lesson. */
export const minutesFor = (steps: number, measured?: number): number => measured ?? Math.round((steps * 65) / 180 + (steps * 20) / 60 + 1);

// ------------------------------------------------------------------ thumbnails (static/thumbs/index.json)

interface ThumbIndex {
	rows?: { cols: number; light: string; dark: string };
	cards?: { cols: number; light: string; dark: string };
	tiles: Record<string, { row?: number; card?: number }>;
}
let thumbsCache: ThumbIndex | null | undefined;
function thumbs(): ThumbIndex | null {
	if (thumbsCache !== undefined) return thumbsCache;
	const f = resolve('static', 'thumbs', 'index.json');
	thumbsCache = existsSync(f) ? (JSON.parse(readFileSync(f, 'utf8')) as ThumbIndex) : null;
	return thumbsCache;
}
function tileOf(i: number | undefined, cols: number, n: number): Tile | undefined {
	if (i === undefined) return undefined;
	const rows = Math.ceil(n / cols);
	const c = i % cols;
	const r = Math.floor(i / cols);
	const pct = (k: number, of: number) => (of > 1 ? +((k / (of - 1)) * 100).toFixed(4) : 0);
	return { size: `${cols * 100}% ${rows * 100}%`, pos: `${pct(c, cols)}% ${pct(r, rows)}%` };
}

// ------------------------------------------------------------------ the index

const live = (m: { status: string }) => m.status === 'live';
const modelHref = (slug: string) => `/m/${slug}/`;
const conceptHref = (slug: string) => `/train/${slug}/`;

function modelLesson(m: ModelEntry, th: ThumbIndex | null, nRows: number, nCards: number): LessonItem {
	const steps = m.lesson?.steps ?? 0;
	const tile = th?.tiles[m.slug];
	return {
		kind: 'model',
		slug: m.slug,
		href: modelHref(m.slug),
		group: m.area,
		title: m.name,
		hook: m.hook,
		steps,
		minutes: minutesFor(steps, m.lesson?.minutes),
		inputs: inputWords(m.inputs),
		buckets: bucketsOf(m.inputs),
		thumb: th?.rows ? tileOf(tile?.row, th.rows.cols, nRows) : undefined,
		card: th?.cards ? tileOf(tile?.card, th.cards.cols, nCards) : undefined
	};
}

function conceptHook(c: ConceptEntry): I18n {
	if (c.hook) return c.hook;
	const s = c.lesson?.summary ?? c.interaction;
	return { en: cutHook(s.en, 100, false), zh: cutHook(s.zh ?? s.en, 50, true) };
}

function conceptLesson(c: ConceptEntry, th: ThumbIndex | null, nRows: number, nCards: number): LessonItem {
	const steps = c.lesson?.steps ?? 0;
	const tile = th?.tiles[c.slug];
	return {
		kind: 'concept',
		slug: c.slug,
		href: conceptHref(c.slug),
		group: c.section,
		title: c.title,
		hook: conceptHook(c),
		steps,
		minutes: minutesFor(steps, c.lesson?.minutes),
		inputs: [],
		buckets: [],
		thumb: th?.rows ? tileOf(tile?.row, th.rows.cols, nRows) : undefined,
		card: th?.cards ? tileOf(tile?.card, th.cards.cols, nCards) : undefined
	};
}

/** The live lessons in catalog order: models by area (live first, then tier), then training methods by section. */
export function liveLessons(): LessonItem[] {
	const th = thumbs();
	const tiles = th ? Object.values(th.tiles) : [];
	const nRows = tiles.filter((t) => t.row !== undefined).length;
	const nCards = tiles.filter((t) => t.card !== undefined).length;
	const models = AREAS.flatMap((a) => sortModels(MODELS.filter((m) => m.area === a.slug && live(m)))).map((m) => modelLesson(m, th, nRows, nCards));
	const concepts = SECTIONS.flatMap((s) => CONCEPTS.filter((c) => c.section === s.slug && live(c))).map((c) => conceptLesson(c, th, nRows, nCards));
	return [...models, ...concepts];
}

export function plannedLessons(withShows = false): PlannedItem[] {
	const models = AREAS.flatMap((a) => sortModels(MODELS.filter((m) => m.area === a.slug && !live(m)))).map(
		(m): PlannedItem => ({ kind: 'model', slug: m.slug, href: modelHref(m.slug), group: m.area, title: m.name, buckets: bucketsOf(m.inputs), ...(withShows ? { shows: m.shows } : {}) })
	);
	const concepts = SECTIONS.flatMap((s) => CONCEPTS.filter((c) => c.section === s.slug && !live(c))).map(
		(c): PlannedItem => ({ kind: 'concept', slug: c.slug, href: conceptHref(c.slug), group: c.section, title: c.title, buckets: [], ...(withShows ? { shows: c.interaction } : {}) })
	);
	return [...models, ...concepts];
}

const nameOf = (kind: 'model' | 'concept', slug: string): I18n | undefined => {
	if (kind === 'model') {
		const m = MODELS.find((x) => x.slug === slug);
		return m ? (m.short ?? m.name) : undefined;
	}
	const c = CONCEPTS.find((x) => x.slug === slug);
	return c ? (c.short ?? c.title) : undefined;
};
const groupOf = (kind: 'model' | 'concept', slug: string): string | undefined =>
	kind === 'model' ? MODELS.find((m) => m.slug === slug)?.area : CONCEPTS.find((c) => c.slug === slug)?.section;

export function learningPaths(lessons = liveLessons()): PathItem[] {
	const bySlug = new Map(lessons.map((l) => [l.slug, l]));
	return PATHS.map((p) => {
		const stations = p.steps
			.map((s) => {
				const l = bySlug.get(s.slug);
				const name = nameOf(s.kind, s.slug);
				if (!name) return null;
				return { kind: s.kind, slug: s.slug, live: !!l, name, steps: l?.steps, href: s.kind === 'model' ? modelHref(s.slug) : conceptHref(s.slug) };
			})
			.filter((s): s is NonNullable<typeof s> => s !== null);
		const liveN = stations.filter((s) => s.live).length;
		return {
			id: p.id,
			title: p.title,
			blurb: p.blurb,
			stations,
			live: liveN,
			coming: stations.length - liveN,
			minutes: stations.reduce((a, s) => a + (s.live ? (bySlug.get(s.slug)?.minutes ?? 0) : 0), 0)
		};
	});
}

/** A group's path: the path with the most stations in the group (at least 2 and 40 % of its stations). */
function pathFor(group: string): { id: string; title: I18n } | undefined {
	let best: { id: string; title: I18n; n: number } | undefined;
	for (const p of PATHS) {
		const n = p.steps.filter((s) => groupOf(s.kind, s.slug) === group).length;
		if (n >= 2 && n / p.steps.length >= 0.4 && (!best || n > best.n)) best = { id: p.id, title: p.title, n };
	}
	return best && { id: best.id, title: best.title };
}

export function groups(): GroupItem[] {
	const areas = AREAS.map((a): GroupItem => {
		const ms = MODELS.filter((m) => m.area === a.slug);
		const n = ms.filter(live).length;
		return { kind: 'model', slug: a.slug, name: a.name, blurb: a.blurb, examples: a.examples, live: n, planned: ms.length - n, path: pathFor(a.slug) };
	});
	const sections = SECTIONS.map((s): GroupItem => {
		const cs = CONCEPTS.filter((c) => c.section === s.slug);
		const n = cs.filter(live).length;
		return { kind: 'concept', slug: s.slug, name: s.name, blurb: s.blurb, live: n, planned: cs.length - n, path: pathFor(s.slug) };
	});
	return [...areas, ...sections];
}

export function catalogIndex(o: { shows?: boolean } = {}): CatalogIndex {
	const lessons = liveLessons();
	const planned = plannedLessons(o.shows);
	const models = lessons.filter((l) => l.kind === 'model');
	const th = thumbs();
	const sprite = (s?: { light: string; dark: string }) => s && { light: `/thumbs/${s.light}`, dark: `/thumbs/${s.dark}` };
	return {
		lessons,
		planned,
		groups: groups(),
		counts: {
			lessons: lessons.length,
			models: models.length,
			methods: lessons.length - models.length,
			planned: planned.length,
			liveAreas: new Set(models.map((l) => l.group)).size,
			areas: AREAS.length
		},
		inputs: BUCKETS.map((b) => ({ bucket: b, n: lessons.filter((l) => l.buckets.includes(b)).length })),
		sprites: { rows: sprite(th?.rows), cards: sprite(th?.cards) }
	};
}

// ------------------------------------------------------------------ search index (search-index.json)

export function searchIndex(): SearchEntry[] {
	const gs = groups();
	const gname = new Map(gs.map((g) => [g.slug, g.name]));
	const out: SearchEntry[] = [];
	for (const l of liveLessons()) out.push({ k: 'l', en: l.title.en, zh: l.title.zh ?? l.title.en, href: l.href, sub: [l.hook.en, l.hook.zh ?? l.hook.en] });
	for (const g of gs)
		out.push({ k: 'a', en: g.name.en, zh: g.name.zh ?? g.name.en, href: g.kind === 'model' ? `/atlas/${g.slug}/` : `/train/#${g.slug}`, sub: [g.blurb.en, g.blurb.zh ?? g.blurb.en] });
	for (const [id, [en, zh]] of Object.entries(GLOSSARY_NAMES)) out.push({ k: 't', en, zh, href: `/glossary/#${id}` });
	for (const p of plannedLessons()) {
		const g = gname.get(p.group);
		out.push({ k: 'p', en: p.title.en, zh: p.title.zh ?? p.title.en, href: p.href, ...(g ? { sub: [g.en, g.zh ?? g.en] as [string, string] } : {}) });
	}
	return out;
}
