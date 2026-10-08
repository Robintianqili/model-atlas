// tinybert: helpers shared by the lesson, the views and the claim tests (pure TS). Positions: the text token k (counting
// from 0) sits at position m = where[k] of TinyBERT's input ([CLS] first, a [SEP] after sentence A) and at position k of
// TinyGPT's input (the text alone). TinyGPT's guess for token k is its prediction after token k − 1.
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { visibleToken } from '../../atlas/core/format.ts';
import { contentOf, layout, maskOf, vocabOf, type MaskedRaw, type Spec } from './adapters/masked-text/adapter.ts';

export type RT = ExplainerRuntime;
export const T = (rt: RT, id: string, src = 'fwd'): Tensor | undefined => rt.sources[src]?.trace?.t[id];

/** The text, the hidden token and where it sits. */
export function gap(rt: RT) {
	const spec = rt.spec as Spec;
	const raw = rt.input as MaskedRaw;
	const c = contentOf(raw);
	const k = maskOf(raw, spec);
	const L = layout(c, k, spec);
	const word = vocabOf(spec);
	/** a token's text for a sentence: without its leading space (white space only: made visible) */
	const say = (id: number) => word(id).trim() || visibleToken(word(id));
	return { c, k, m: k >= 0 ? L.where[k] : -1, ids: L.ids, seg: L.seg, where: L.where, split: L.j, n: c.length, len: L.ids.length, hidden: k >= 0 ? c[k] : -1, word, say };
}

/** Row r of a [rows, V] tensor (null when missing). */
export function row(t: Tensor | undefined, r: number): ArrayLike<number> | null {
	if (!t || r < 0) return null;
	const V = t.shape[t.shape.length - 1];
	return r * V < t.data.length ? t.data.subarray(r * V, (r + 1) * V) : null;
}

/** Indices sorted by value, largest first (ties: smaller index first). */
export function ranked(p: ArrayLike<number>): number[] {
	return Array.from(p, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
}

/** One query's attention weights over every key: head h, query q, of an [H, T, T] layer. */
export function attnRow(t: Tensor | undefined, h: number, q: number): ArrayLike<number> | null {
	if (!t || q < 0) return null;
	const [, Tq, Tk] = t.shape;
	return t.data.subarray((h * Tq + q) * Tk, (h * Tq + q + 1) * Tk);
}

/** TinyBERT's guess at the gap: probabilities, top ids, the hidden token's share and rank (counting from 1). */
export function bertGuess(rt: RT) {
	const g = gap(rt);
	const p = row(T(rt, 'probs'), g.m);
	const order = p ? ranked(p) : [];
	return { p, top: order[0] ?? -1, pTop: p && order.length ? p[order[0]] : NaN, pHidden: p && g.hidden >= 0 ? p[g.hidden] : NaN, rank: order.indexOf(g.hidden) + 1 || NaN, order };
}

/** TinyGPT's guess for the hidden token from the tokens before it (null for the first token: nothing comes before it). */
export function gptGuess(rt: RT) {
	const g = gap(rt);
	const p = g.k >= 1 ? row(T(rt, 'probs', 'gpt'), g.k - 1) : null;
	const order = p ? ranked(p) : [];
	return { p, top: order[0] ?? -1, pTop: p && order.length ? p[order[0]] : NaN, pHidden: p && g.hidden >= 0 ? p[g.hidden] : NaN, rank: order.indexOf(g.hidden) + 1 || NaN, order };
}

/** Shares of the [MASK] row's attention (block b, head h) before the gap, on the gap itself and after it. */
export function sides(rt: RT, b: number, h: number) {
	const g = gap(rt);
	const w = attnRow(T(rt, `B${b}.attn`), h, g.m);
	let left = 0;
	let right = 0;
	if (w) for (let j = 0; j < w.length; j++) (j < g.m ? (left += w[j]) : j > g.m ? (right += w[j]) : 0);
	return { left: w ? left : NaN, self: w ? w[g.m] : NaN, right: w ? right : NaN, nLeft: g.m, nRight: g.len - g.m - 1 };
}

/** The feature of the [MASK] position's input to block 1 (layer embed) with the largest size: the one step 3 adds up. */
export function bigFeature(rt: RT): number {
	const e = row(T(rt, 'embed'), gap(rt).m);
	let f = 0;
	if (e) for (let c = 1; c < e.length; c++) if (Math.abs(e[c]) > Math.abs(e[f])) f = c;
	return f;
}

/** The same word as token `id` with or without a space in front: another text token whose text, trimmed, is the same (-1: none). */
export function twinOf(rt: RT, id: number): number {
	const g = gap(rt);
	const w = id >= 0 ? g.word(id).trim() : '';
	if (!w) return -1;
	for (let i = 0; i < 1024; i++) if (i !== id && g.word(i).trim() === w) return i;
	return -1;
}

/** The head of block 1 whose [MASK] row puts the most weight after the gap (ties: the lower head). */
export function rightHead(rt: RT): number {
	const H = Number((rt.spec?.layers.find((l) => l.id === 'B1.scores')?.params as { nHead?: number } | undefined)?.nHead ?? 1);
	let best = 0;
	for (let h = 1; h < H; h++) if (sides(rt, 1, h).right > sides(rt, 1, best).right) best = h;
	return best;
}
