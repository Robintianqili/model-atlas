// masked-lm: helpers for the live steps (pure TS; adapted from tinybert's bert.ts, credit: the tinybert package).
// Positions: text token k (counting from 0) sits at position where[k] of TinyBERT's input ([CLS] first, a [SEP] after
// sentence A) and at position k of TinyGPT's input (the text alone). TinyGPT's guess for token k is its prediction after
// token k − 1. The masked-LM loss is −log p of the original token at every chosen position (fate > 0: [MASK], a random
// token or kept), averaged over those positions only; PyTorch skips the others with ignore_index (label −100).
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { visibleToken } from '../../atlas/core/format.ts';
import { contentOf, fatesOf, layout, vocab, type MaskedRaw } from './adapters/masked-text/adapter.ts';

export type RT = ExplainerRuntime;
export const T = (rt: RT, id: string, src = 'bert'): Tensor | undefined => rt.sources[src]?.trace?.t[id];
/** a token's text for a sentence: without its leading space (white space only: made visible) */
export const say = (id: number) => vocab(id).trim() || visibleToken(vocab(id));

/** The text, what became of each token and where it sits in TinyBERT's input. */
export function text(rt: RT) {
	const raw = rt.input as MaskedRaw;
	const c = contentOf(raw);
	const { fate, shown } = fatesOf(raw);
	const L = layout(c, shown);
	const graded = fate.flatMap((f, k) => (f > 0 ? [k] : []));
	const f0 = raw?.focus;
	const focus = typeof f0 === 'number' && graded.includes(f0) ? f0 : (graded[0] ?? -1);
	return { c, fate, shown, ids: L.ids, where: L.where, split: L.j, n: c.length, graded, masked: fate.flatMap((f, k) => (f === 1 ? [k] : [])), focus, picked: raw?.picked === true, fromDraw: raw?.fromDraw === true };
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

export function guessOf(p: ArrayLike<number> | null, answer: number) {
	const order = p ? ranked(p) : [];
	return { p, top: order[0] ?? -1, pTop: p && order.length ? p[order[0]] : NaN, pAnswer: p && answer >= 0 ? p[answer] : NaN, rank: order.indexOf(answer) + 1 || NaN, order };
}

/** TinyBERT's guess at text token k (its row of TinyBERT's output) and TinyGPT's from the tokens before k. */
export function bertAt(rt: RT, k: number) {
	const x = text(rt);
	return guessOf(row(T(rt, 'probs'), x.where[k] ?? -1), x.c[k] ?? -1);
}
export const gptAt = (rt: RT, k: number) => guessOf(k >= 1 ? row(T(rt, 'probs', 'gpt'), k - 1) : null, text(rt).c[k] ?? -1);

function argmax(p: ArrayLike<number>) {
	let j = 0;
	for (let i = 1; i < p.length; i++) if (p[i] > p[j]) j = i;
	return j;
}
const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);

export type LossRow = { k: number; id: number; fate: number; p: number; nll: number; top: number; right: boolean };

/** The masked-LM loss on the learner's text: one row per text token (TinyBERT's p for the original token there), the
 *  mean −log p over the chosen positions (the loss) and, not graded, over the visible ones. */
export function mlmLoss(rt: RT) {
	const x = text(rt);
	const P = T(rt, 'probs');
	const rows: LossRow[] = [];
	x.c.forEach((id, k) => {
		const r = row(P, x.where[k]);
		if (r) rows.push({ k, id, fate: x.fate[k], p: r[id], nll: -Math.log(r[id]), top: argmax(r), right: argmax(r) === id });
	});
	const graded = rows.filter((r) => r.fate > 0);
	const seen = rows.filter((r) => r.fate === 0);
	const of = (f: number) => rows.filter((r) => r.fate === f);
	return {
		rows,
		graded,
		seen,
		loss: mean(graded.map((r) => r.nll)),
		right: graded.filter((r) => r.right).length,
		seenP: mean(seen.map((r) => r.p)),
		seenRight: seen.filter((r) => r.right).length,
		mask: of(1),
		random: of(2),
		kept: of(3)
	};
}

/** TinyGPT's causal loss on the same text: −log p of tokens 1 to n − 1 from the tokens before each (token 0 has nothing
 *  before it), and their mean. */
export function clmLoss(rt: RT) {
	const x = text(rt);
	const P = T(rt, 'probs', 'gpt');
	const rows: { k: number; id: number; p: number; nll: number }[] = [];
	for (let k = 1; k < x.n; k++) {
		const r = row(P, k - 1);
		if (r) rows.push({ k, id: x.c[k], p: r[x.c[k]], nll: -Math.log(r[x.c[k]]) });
	}
	return { rows, loss: mean(rows.map((r) => r.nll)) };
}

/** How many tokens are chosen and whether TinyBERT's top guesses there are right, as a sentence that reads right at 0,
 *  1 and more chosen tokens (of: say "of your n" too). */
export function chosenLine(rt: RT, of = false): { en: string; zh: string } {
	const x = text(rt);
	const k = x.graded.length;
	const right = mlmLoss(rt).right;
	if (!k) return { en: 'No token is chosen yet.', zh: '现在还没有选中任何词元。' };
	const enOf = of ? ` of your ${x.n}` : '';
	const zhChose = of ? `你在 ${x.n} 个词元中选中了 ${k} 个` : `你选中了 ${k} 个词元`;
	return k === 1
		? { en: `You have chosen 1 token${enOf}, and TinyBERT’s top guess there is ${right ? 'right' : 'wrong'}.`, zh: `${zhChose}，TinyBERT 在那里的第一猜测${right ? '是对的' : '不对'}。` }
		: { en: `You have chosen ${k} tokens${enOf}; TinyBERT’s top guess is right at ${right} of them.`, zh: `${zhChose}；TinyBERT 的第一猜测对了其中 ${right} 个。` };
}
