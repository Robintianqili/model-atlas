/**
 * Where a learner left a lesson (redesign spec §5.6): localStorage `atlas.lesson.<slug>` = { step, started }. Every read
 * and write is in try/catch (a private window or blocked storage): without it the cover shows each visit and nothing
 * else changes. Per viewer only; nothing leaves the browser.
 */
export interface Saved {
	/** the step id the learner was on */
	step: string;
	/** the learner pressed Start (or opened a step): the cover never shows again for this lesson */
	started: boolean;
}

const key = (slug: string) => `atlas.lesson.${slug}`;

export function readSaved(slug: string): Saved | null {
	try {
		const raw = localStorage.getItem(key(slug));
		if (!raw) return null;
		const v = JSON.parse(raw) as Partial<Saved>;
		return typeof v.step === 'string' ? { step: v.step, started: v.started !== false } : null;
	} catch {
		return null;
	}
}

export function writeSaved(slug: string, step: string): void {
	try {
		localStorage.setItem(key(slug), JSON.stringify({ step, started: true }));
	} catch {
		/* storage blocked: the lesson still works, it just does not resume */
	}
}

export function clearSaved(slug: string): void {
	try {
		localStorage.removeItem(key(slug));
	} catch {
		/* ignore */
	}
}
