/**
 * Input hand-off from the home-page launcher to a model page.
 * The launcher saves what the learner drew or typed in sessionStorage (never sent anywhere);
 * the model page reads it once on mount and uses it as its first input.
 *
 *   // model page (in onMount / $effect):
 *   const h = takeHandoff('drawing');   // { kind: 'drawing', strokes } | null
 *   if (h) strokes = h.strokes;
 */
import type { Stroke } from './draw.ts';

const KEY = 'atlas.handoff';

export type Handoff = { kind: 'drawing'; strokes: Stroke[] } | { kind: 'text'; text: string };

export function saveHandoff(h: Handoff): void {
	try {
		sessionStorage.setItem(KEY, JSON.stringify(h));
	} catch {
		/* storage blocked: the model page simply starts from its preset */
	}
}

/** Read and clear the hand-off if it matches `kind`. */
export function takeHandoff<K extends Handoff['kind']>(kind: K): Extract<Handoff, { kind: K }> | null {
	try {
		const raw = sessionStorage.getItem(KEY);
		if (!raw) return null;
		const h = JSON.parse(raw) as Handoff;
		if (h.kind !== kind) return null;
		sessionStorage.removeItem(KEY);
		return h as Extract<Handoff, { kind: K }>;
	} catch {
		return null;
	}
}
