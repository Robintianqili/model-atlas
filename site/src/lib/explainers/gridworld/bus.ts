// gridworld: how a view changes the learner's input. rt.setInput() from a view would not re-resolve the lesson's steps,
// so the stage's Pull buttons and maze tools ask the page's Capture to apply an edit via its onchange (notes, Kit issues).
import type { GameRaw } from './world.ts';

export type Edit = (raw: GameRaw) => GameRaw;
const subs: ((e: Edit) => void)[] = [];

/** A Capture registers here while mounted; the most recent one applies edits. */
export function onEdit(fn: (e: Edit) => void): () => void {
	subs.push(fn);
	return () => void subs.splice(subs.indexOf(fn), 1);
}
/** Apply an edit through the page (false when no Capture is mounted, e.g. in the view gallery). */
export function edit(e: Edit): boolean {
	subs[subs.length - 1]?.(e);
	return subs.length > 0;
}
