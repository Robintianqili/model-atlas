// self-supervised-objectives: helpers for the live steps (pure TS; adapted from tinybert's bert.ts, credit: the tinybert
// package). Positions: text token k (counting from 0) sits at position where[k] of TinyBERT's input ([CLS] first, a [SEP]
// after sentence A) and at position k of TinyGPT's input (the text alone). TinyGPT's guess for token k is its prediction
// after token k − 1.
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { visibleToken } from '../../atlas/core/format.ts';
import { contentOf, layout, maskOf, vocab, type MaskedRaw } from './adapters/masked-text/adapter.ts';

export type RT = ExplainerRuntime;
export const T = (rt: RT, id: string, src = 'bert'): Tensor | undefined => rt.sources[src]?.trace?.t[id];
/** a token's text for a sentence: without its leading space (white space only: made visible) */
export const say = (id: number) => vocab(id).trim() || visibleToken(vocab(id));

/** The text, the hidden token and where it sits. */
export function gap(rt: RT) {
	const raw = rt.input as MaskedRaw;
	const c = contentOf(raw);
	const k = maskOf(raw);
	const L = layout(c, k);
	return { c, k, m: k >= 0 ? L.where[k] : -1, ids: L.ids, where: L.where, split: L.j, n: c.length, hidden: k >= 0 ? c[k] : -1, picked: raw?.picked === true };
}

/** Row r of a [rows, V] tensor (null when missing). */
export function row(t: Tensor | undefined, r: number): ArrayLike<number> | null {
	if (!t || r < 0) return null;
	const V = t.shape[t.shape.length - 1];
	return (r + 1) * V <= t.data.length ? t.data.subarray(r * V, (r + 1) * V) : null;
}

/** Indices sorted by value, largest first (ties: smaller index first). */
export function ranked(p: ArrayLike<number>): number[] {
	return Array.from(p, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
}

function guessOf(p: ArrayLike<number> | null, hidden: number) {
	const order = p ? ranked(p) : [];
	return { p, top: order[0] ?? -1, pTop: p && order.length ? p[order[0]] : NaN, pHidden: p && hidden >= 0 ? p[hidden] : NaN, rank: order.indexOf(hidden) + 1 || NaN, order };
}

/** TinyBERT's guess at the gap (the [MASK] row): probabilities, top id, the hidden token's share and rank (from 1). */
export const bertGuess = (rt: RT) => guessOf(row(T(rt, 'probs'), gap(rt).m), gap(rt).hidden);

/** TinyGPT's guess for the hidden token from the tokens before it (null for the first token: nothing comes before it). */
export function gptGuess(rt: RT) {
	const g = gap(rt);
	return guessOf(g.k >= 1 ? row(T(rt, 'probs', 'gpt'), g.k - 1) : null, g.hidden);
}

/** TinyGPT's probability of every actual token from the tokens before it: tokens 1 to n − 1 (token 0 has nothing before
 *  it), their mean −log p (the causal loss on this text, in nats) and the least expected token. */
export function nextProbs(rt: RT) {
	const g = gap(rt);
	const P = T(rt, 'probs', 'gpt');
	const ps: { i: number; id: number; p: number }[] = [];
	for (let i = 1; i < g.n; i++) {
		const r = row(P, i - 1);
		if (r) ps.push({ i, id: g.c[i], p: r[g.c[i]] });
	}
	let worst = ps[0];
	for (const q of ps) if (q.p < worst.p) worst = q;
	const loss = ps.length ? ps.reduce((s, q) => s - Math.log(q.p), 0) / ps.length : NaN;
	return { ps, targets: ps.length, loss, worst: worst ?? { i: NaN, id: -1, p: NaN } };
}
