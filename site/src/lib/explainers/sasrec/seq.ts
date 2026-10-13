// sasrec: what the lesson and the view compute from the trace of the learner's clicks (pure TS).
//   attention  the last click's row of the attention weights (layer 'att.w', [1, T, T]): how much it reads each click
//   next click the last row of 'logits' and 'probs' ([T, 31]): the scores and shares of every product as the next click;
//              the top K are the best products the history does not hold (ties: the smaller id)
//   order      the same model on the history reversed (a forward pass of the exported graph, memoized per model)
//   limits     the new arrival's place, and the clicks per product in the training logs (spec extra `clicks`)
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { ITEMS, NEW_ITEM, toClicks } from './world.ts';

type RT = ExplainerRuntime;
/** the top list's length */
export const K = 5;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const clicksOf = (rt: RT) => toClicks(rt.input).clicks;
export const dim = (rt: RT) => rt.model?.weights['item.weight']?.shape[1] ?? NaN;
export const positions = (rt: RT) => rt.model?.weights['pos.weight']?.shape[0] ?? NaN;
/** row r of a [rows, n] tensor */
export const row = (t: Tensor | undefined, r: number, n = t?.shape[t.shape.length - 1] ?? 0) => (t ? Array.from(t.data.subarray(r * n, (r + 1) * n)) : null);
/** the attention weights of click t (counting from 0) over clicks 0 … t; default the last click */
export function attn(rt: RT, t?: number): number[] | null {
	const w = T(rt, 'att.w'), n = w?.shape[2] ?? 0;
	return w ? row(w, t ?? n - 1, n)!.slice(0, (t ?? n - 1) + 1) : null;
}
/** the most-read click: the largest weight (ties: the later click); `before` leaves the last click out */
export function most(rt: RT, before = false): { j: number; w: number } | null {
	const a = attn(rt);
	if (!a || (before && a.length < 2)) return null;
	let j = before ? a.length - 2 : a.length - 1;
	for (let k = 0; k < (before ? a.length - 1 : a.length); k++) if (a[k] > a[j]) j = k;
	return { j, w: a[j] };
}
export type Scored = { i: number; s: number; p: number };
/** every product the history does not hold, best first, from a trace's last row */
export function rankedOf(t: Record<string, Tensor> | undefined, clicks: number[]): Scored[] {
	const lg = t?.logits, pr = t?.probs;
	if (!lg || !pr || !clicks.length) return [];
	const r = lg.shape[0] - 1, s = row(lg, r)!, p = row(pr, r)!;
	return ITEMS.map((_, i) => ({ i, s: s[i], p: p[i] })).filter((x) => !clicks.includes(x.i)).sort((a, b) => b.s - a.s || a.i - b.i);
}
export const ranked = (rt: RT) => rankedOf(rt.sources.fwd?.trace?.t, clicksOf(rt));
export const topK = (rt: RT) => ranked(rt).slice(0, K);

const memo = new WeakMap<GraphModel, Map<string, Record<string, Tensor>>>();
/** the exported model on other clicks (a forward pass of the same graph) */
export function runOn(rt: RT, ids: number[]): Record<string, Tensor> | null {
	const m = rt.model;
	if (!m || !ids.length) return null;
	if (!memo.has(m)) memo.set(m, new Map());
	const c = memo.get(m)!, key = ids.join(',');
	if (!c.has(key)) c.set(key, m.forward({ ids: { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' } }, { record: 'layers' }).t as Record<string, Tensor>);
	return c.get(key)!;
}
/** your history reversed, and its top list */
export function reversed(rt: RT) {
	const c = clicksOf(rt), r = [...c].reverse();
	return { clicks: r, top: rankedOf(runOn(rt, r) ?? undefined, r).slice(0, K) };
}
/** the new arrival's place among the products you have not clicked (null when you clicked it) */
export function newPlace(rt: RT) {
	const r = ranked(rt), k = r.findIndex((x) => x.i === NEW_ITEM);
	return k < 0 ? null : { place: k + 1, of: r.length, s: r[k].s, p: r[k].p };
}
/** clicks per product in the training logs (spec extra), and the 5 most clicked */
export const logClicks = (rt: RT, i: number) => ((rt.spec?.clicks ?? []) as number[])[i] ?? NaN;
export const popTop = (rt: RT) => (rt.spec?.popTop5 ?? []) as number[];
