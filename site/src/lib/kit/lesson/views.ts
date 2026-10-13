/**
 * Hash routing of a lesson page (redesign spec §5.9): the one owner of the URL hash on /m/<slug>/ and /train/<slug>/.
 * `#<lesson id>-<step id>` (and `#<lesson id>`) belong to the Walkthrough; every other id opens a view (Explore, How it
 * works, Code, Training; Explore, Pitfalls, Sources for concepts), scrolled to a section. The old section anchors keep
 * working: `#overview` opens Explore at its overview, `#credits` the Training view, `#post` the Pitfalls view, and so on.
 * Pure (no DOM): the shells call parseHash() on load and on every hashchange.
 */

export type ViewId = 'lesson' | 'explore' | 'article' | 'code' | 'training' | 'pitfalls' | 'sources';
export type LessonKind = 'model' | 'concept';

export interface HashRoute {
	view: ViewId;
	/** a section inside the view (Explore: overview, inspector, unit, output, tryThis, checkpoint, toy, real, compare …) */
	section?: string;
	/** a lesson step id (#<lesson id>-<step id>) */
	step?: string;
	/** the plain #<lesson id> anchor: the lesson at its saved step (or step 1) */
	top?: boolean;
	/** #goals: the lesson's cover */
	cover?: boolean;
	/** #input: the input sheet */
	input?: boolean;
}

/** The views a page kind can have, in tab order. */
export const VIEW_ORDER: Record<LessonKind, ViewId[]> = {
	model: ['lesson', 'explore', 'article', 'code', 'training'],
	concept: ['lesson', 'explore', 'pitfalls', 'sources']
};

/** Old section anchors (before the redesign) and the view and section they open now. */
export const LEGACY: Record<LessonKind, Record<string, [ViewId, string?]>> = {
	model: {
		overview: ['explore', 'overview'],
		inspector: ['explore', 'inspector'],
		unit: ['explore', 'unit'],
		output: ['explore', 'output'],
		tryThis: ['explore', 'tryThis'],
		checkpoint: ['explore', 'checkpoint'],
		links: ['explore', 'links'],
		credits: ['training', 'credits'],
		facts: ['lesson'],
		hook: ['lesson']
	},
	concept: {
		toy: ['explore', 'toy'],
		real: ['explore', 'real'],
		compare: ['explore', 'compare'],
		checkpoint: ['pitfalls', 'checkpoint'],
		post: ['pitfalls', 'post'],
		links: ['sources'],
		further: ['sources'],
		credits: ['sources', 'credits'],
		hook: ['lesson']
	}
};

/**
 * The route of a hash ('#explore-zoom', 'lesson-conv', '' …). `has` lists the views this page has: a view it lacks
 * falls back to the lesson. Returns null for an id this table does not know (an anchor inside a view: leave the view as
 * it is and let the browser scroll).
 */
export function parseHash(hash: string, o: { kind: LessonKind; lessonId?: string; has?: readonly ViewId[] }): HashRoute | null {
	let h = hash.startsWith('#') ? hash.slice(1) : hash;
	try {
		h = decodeURIComponent(h);
	} catch {
		/* a malformed escape: use it as it is */
	}
	const id = o.lessonId ?? 'lesson';
	const has = (v: ViewId) => v === 'lesson' || !o.has || o.has.includes(v);
	const lesson = (r: Omit<HashRoute, 'view'> = {}): HashRoute => ({ view: 'lesson', ...r });
	if (!h) return lesson();
	if (h === id || h === 'lesson') return lesson({ top: true });
	if (h.startsWith(`${id}-`)) return lesson({ step: h.slice(id.length + 1) });
	if (h === 'goals') return lesson({ cover: true });
	if (h === 'input') return lesson({ input: true });
	const views = VIEW_ORDER[o.kind];
	for (const v of views) {
		if (v === 'lesson') continue;
		if (h === v) return has(v) ? { view: v } : lesson();
		if (h.startsWith(`${v}-`)) return has(v) ? { view: v, section: h.slice(v.length + 1) } : lesson();
	}
	// the concept views' names on a model page (and back) are not views of that kind
	const legacy = LEGACY[o.kind][h];
	if (legacy) {
		const [v, section] = legacy;
		if (!has(v)) return lesson();
		return section ? { view: v, section } : { view: v };
	}
	// try cards of Explore (#try-<id>)
	if (o.kind === 'model' && h.startsWith('try-')) return has('explore') ? { view: 'explore', section: h } : lesson();
	return null;
}

/** The hash that opens a route (the inverse of parseHash for view routes). */
export function hashOf(r: Pick<HashRoute, 'view' | 'section'>): string {
	if (r.view === 'lesson') return '#lesson';
	return r.section ? `#${r.view}-${r.section}` : `#${r.view}`;
}

/** Reading time in minutes (spec §6.2): words / 180 + 20 s per step + 30 s per beat (a predict or a task), at least 1. */
export function lessonMinutes(o: { words: number; steps: number; beats: number }): number {
	return Math.max(1, Math.round(o.words / 180 + (o.steps * 20) / 60 + (o.beats * 30) / 60));
}

/** Chapter groups of a step list: from explicit chapters ({ title, count }) or from consecutive step chapters. Steps
 *  outside every chapter form a group without a title. */
export function chapterGroups<T>(n: number, o: { chapters?: { title: T; count: number }[]; stepChapter?: (i: number) => T | undefined; same?: (a: T, b: T) => boolean }): { title?: T; from: number; to: number }[] {
	const out: { title?: T; from: number; to: number }[] = [];
	if (o.chapters?.length) {
		let k = 0;
		for (const c of o.chapters) {
			const count = Math.max(0, Math.min(c.count, n - k));
			if (count) out.push({ title: c.title, from: k, to: k + count - 1 });
			k += count;
		}
		if (k < n) out.push({ from: k, to: n - 1 });
		return out;
	}
	const same = o.same ?? ((a: T, b: T) => JSON.stringify(a) === JSON.stringify(b));
	for (let i = 0; i < n; i++) {
		const c = o.stepChapter?.(i);
		const last = out[out.length - 1];
		if (last && ((c === undefined && last.title === undefined) || (c !== undefined && last.title !== undefined && same(c, last.title)))) last.to = i;
		else out.push(c === undefined ? { from: i, to: i } : { title: c, from: i, to: i });
	}
	return out;
}

/** The layout mode of the lesson app (spec §5.1): app at 960 x 600 and up, flow below (atlas e2e AK605 checks it). */
export function layoutMode(width: number, height: number): 'app' | 'flow' {
	return width >= 960 && height >= 600 ? 'app' : 'flow';
}

/** Stage density from the stage pane's content box (spec §5.4). */
export function densityOf(width: number, height: number, phone = false): 'roomy' | 'regular' | 'compact' {
	if (phone || width < 600 || height < 480) return 'compact';
	if (width >= 760 && height >= 700) return 'roomy';
	return 'regular';
}
