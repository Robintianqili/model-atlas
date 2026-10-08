// gnn-variants: the numbers behind each layer-1 rule for the member in focus, read from the fwd trace and the weights
// (pure TS; the views, the narration and claims.test.ts share them). With one-hot rows, a member's message is its own
// column of W, so every rule is "weights on the senders' columns of W", plus the bias and tanh:
//   gcn   Âᵢⱼ × (column j of W)                         (traced gcn.gc1#ahat, gcn.gc1#xw)
//   sage  own half: column i of W[:, :34]; friends' half: the average of their columns of W[:, 34:]
//   gin   (1 + ε) × column i of W₁ + the friends' columns of W₁, + b₁ (= traced gin.gc1.mlp.0), ReLU, linear 2
//   gat   αᵢⱼ × zⱼ (traced gat.gc1#alpha, gat.gc1#z)
import type { ExplainerRuntime, I18n } from '../../atlas/types/index.ts';
import { graphOf, KARATE, LABELLED, LABELS, N, neighbours, sorted, type GraphRaw, type Variant } from './graph.ts';

type RT = ExplainerRuntime;
const F = 4, memo = new WeakMap<object, { g: GraphRaw; i: number; nb: number[]; S: number[] }>();
const tr = (rt: RT, id: string) => rt.sources.fwd?.trace?.t[id]?.data;
const wt = (rt: RT, id: string) => rt.model?.weights[id]?.data;
const row = (a: ArrayLike<number> | undefined, r: number, w = F) => (a ? Array.from({ length: w }, (_, f) => a[r * w + f]) : []);
const col = (W: ArrayLike<number> | undefined, j: number, I: number) => (W ? Array.from({ length: F }, (_, o) => W[o * I + j]) : []);
const add = (a: number[], b: number[], k = 1) => a.map((v, f) => v + k * b[f]);

/** The member in focus, its friends and its senders (itself first), cached per input object: the views ask for every edge. */
export function focusOf(rt: RT) {
	const k = (rt.input && typeof rt.input === 'object' ? rt.input : KARATE) as object;
	if (!memo.has(k)) { const g = graphOf(k); const nb = neighbours(g, g.focus); memo.set(k, { g, i: g.focus, nb, S: [g.focus, ...nb] }); }
	return memo.get(k)!;
}
/** A sender's weight in each rule (sage: null for the member itself, which is kept apart) */
export function weightOf(rt: RT, v: Variant, j: number): number | null {
	const { i, nb } = focusOf(rt);
	if (v === 'gcn') return tr(rt, 'gcn.gc1#ahat')?.[i * N + j] ?? NaN;
	if (v === 'sage') return j === i ? null : 1 / nb.length;
	if (v === 'gin') return j === i ? 1 + (wt(rt, 'gin.gc1.agg.eps')?.[0] ?? NaN) : 1;
	return tr(rt, 'gat.gc1#alpha')?.[i * N + j] ?? NaN;
}
/** The friend with the largest weight in a rule (ties: the smaller member), or −1 without friends */
export function topFriend(rt: RT, v: Variant) {
	const { nb } = focusOf(rt);
	return nb.reduce((b, j) => (b < 0 || (weightOf(rt, v, j) ?? 0) > (weightOf(rt, v, b) ?? 0) ? j : b), -1);
}
/** The row after layer 1 (tanh) of a network for the member in focus */
export const rowAfter = (rt: RT, v: Variant) => row(tr(rt, `${v}.act1`), focusOf(rt).i);
/** GCN: each sender's message (column j of W), weighted by Â, their sum, the bias, z and tanh */
export function gcnRule(rt: RT) {
	const { i, S } = focusOf(rt);
	const xw = tr(rt, 'gcn.gc1#xw');
	const weighted = S.map((j) => row(xw, j).map((x) => x * (weightOf(rt, 'gcn', j) ?? NaN)));
	return { weighted, sum: weighted.reduce((a, m) => add(a, m), [0, 0, 0, 0]), b: Array.from(wt(rt, 'gcn.gc1.bias') ?? []), z: row(tr(rt, 'gcn.gc1'), i), h: rowAfter(rt, 'gcn') };
}
/** GraphSAGE: the own half of W reads column i, the friends' half the friends' columns; their average, the bias, z, tanh */
export function sageRule(rt: RT) {
	const { i, nb } = focusOf(rt);
	const W = wt(rt, 'sage.gc1.weight');
	const fr = nb.map((j) => col(W, N + j, 2 * N));
	const mean = fr.length ? fr.reduce((a, m) => add(a, m, 1 / fr.length), [0, 0, 0, 0]) : [0, 0, 0, 0];
	return { own: col(W, i, 2 * N), fr, mean, b: Array.from(wt(rt, 'sage.gc1.bias') ?? []), z: row(tr(rt, 'sage.gc1'), i), h: rowAfter(rt, 'sage') };
}
/** GIN: (1 + ε) × column i of W₁ and the friends' columns; + b₁ (traced mlp.0), ReLU, linear 2, tanh */
export function ginRule(rt: RT) {
	const { i, nb } = focusOf(rt);
	const W = wt(rt, 'gin.gc1.mlp.0.weight');
	const eps = wt(rt, 'gin.gc1.agg.eps')?.[0] ?? NaN;
	return { eps, own: col(W, i, N).map((x) => (1 + eps) * x), fr: nb.map((j) => col(W, j, N)), pre: row(tr(rt, 'gin.gc1.mlp.0'), i), relu: row(tr(rt, 'gin.gc1.mlp.1'), i), out: row(tr(rt, 'gin.gc1.mlp.2'), i), h: rowAfter(rt, 'gin') };
}
/** "If every member started with the same single number 1": the average is 1 for everyone, the sum is 1 + ε + friends */
export function sameStart(rt: RT) {
	const g = graphOf(rt.input);
	const sums = Array.from({ length: N }, (_, m) => 1 + (wt(rt, 'gin.gc1.agg.eps')?.[0] ?? NaN) + neighbours(g, m).length);
	return { sums, kinds: new Set(sums).size, focus: sums[g.focus] };
}
/** How many of the 32 unlabelled members a network puts on their own faction's side (fwd trace) */
export function rightOf(rt: RT, v: Variant) {
	const z = tr(rt, `${v}.head`);
	return z ? sorted(z) : { rest: N - LABELLED.length, right: NaN, wrong: [] as number[] };
}
/** The side (0 = Mr. Hi, 1 = the Officer) each network puts member m on */
export const sideOf = (rt: RT, v: Variant, m: number) => { const z = tr(rt, `${v}.head`); return z ? (z[m * 2 + 1] > z[m * 2] ? 1 : 0) : -1; }; // prettier-ignore
export const isRight = (rt: RT, v: Variant, m: number) => sideOf(rt, v, m) === LABELS[m];
/** "(0.12, −0.3, …)": NaN when a number is missing, so lint reports it */
export const vecText = (v: number[], f: (x: number) => string) => (v.length && v.every(Number.isFinite) ? `(${v.map(f).join(', ')})` : NaN);
export const VARIANT_TEXT: Record<Variant, I18n> = {
	gcn: { en: 'GCN: fixed weights Â from the degrees', zh: 'GCN：由度决定的固定权重 Â' },
	sage: { en: 'GraphSAGE: average of the friends, own row apart', zh: 'GraphSAGE：朋友取平均，自己的行单独放' },
	gin: { en: 'GIN: plain sum, 1 + ε for itself, then an MLP', zh: 'GIN：直接求和，自己乘 1 + ε，再接 MLP' },
	gat: { en: 'GAT: learned attention weights α', zh: 'GAT：学来的注意力权重 α' }
};
