// word2vec: plain float64 arithmetic on recorded weights (no autograd), shared by the views, the lesson and the claim
// tests: every number shown is read from, or recomputed from, the weights in the frames that ran.
import type { At, Tape, Tensor } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { corpusOf, type CorpusRaw, type Pair } from './corpus.ts';

type St = { params?: Record<string, Tensor>; prev?: Record<string, Tensor>; grads?: Record<string, Tensor>; batch?: Record<string, Tensor>; acts?: Record<string, Tensor>; lr?: number };
export const dot = (a: number[], b: number[]) => a.reduce((s, x, k) => s + x * b[k], 0);
export const cosine = (a: number[], b: number[]) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
export function softmax(u: number[]): number[] {
	const m = Math.max(...u), e = u.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
	return e.map((x) => x / s);
}
/** the first V rows of W_in ('emb') or W_out ('out') in a record of tensors */
export function rows(rec: Record<string, Tensor> | undefined, table: 'emb' | 'out', V: number): number[][] {
	const t = rec?.[`${table}.weight`];
	if (!t) return [];
	const d = t.shape[1];
	return Array.from({ length: Math.min(V, t.shape[0]) }, (_, i) => Array.from(t.data.subarray(i * d, (i + 1) * d)));
}
/** every other word by cosine similarity of the W_in rows, closest first (ties: the smaller id) */
export function neighbours(win: number[][], i: number): { j: number; cos: number }[] {
	return win.map((r, j) => ({ j, cos: cosine(win[i], r) })).filter((x) => x.j !== i).sort((a, b) => b.cos - a.cos || a.j - b.j);
}
/** the two words with the largest cosine similarity (a < b), or null for fewer than two words */
export function closestPair(vs: number[][]): { a: number; b: number; cos: number } | null {
	let best: { a: number; b: number; cos: number } | null = null;
	for (let a = 0; a < vs.length; a++) for (let b = a + 1; b < vs.length; b++) { const c = cosine(vs[a], vs[b]); if (!best || c > best.cos) best = { a, b, cos: c }; }
	return best;
}
/** the vocabulary size and the word vectors (W_in rows) at a frame of a training tape */
export function wordVectors(raw: CorpusRaw, st: unknown): number[][] {
	return rows((st as St | undefined)?.params, 'emb', corpusOf(raw).words.length);
}

/** One training step on one pair, at a fine frame of source `one` (forward / loss / backward / update): the numbers the
 *  'net' view and the narration show. h, u and p are the recorded activations; before = the weights the step started from. */
export interface Worked { phase: string; words: string[]; c: number; o: number; V: number; d: number; win: number[][]; wout: number[][]; h: number[]; u: number[]; p: number[]; loss: number; lr: number
	win1?: number[][]; wout1?: number[][]; gin?: number[][]; gout?: number[][]; u1?: number; loss1?: number }
export function worked(raw: CorpusRaw, at: At): Worked | null {
	const st = at.state as St | undefined, b = st?.batch, f = at.frame;
	if (!st || !b?.c?.data.length || !f) return null;
	const words = corpusOf(raw).words, V = b.V.data[0], c = b.c.data[0], o = b.o.data[0];
	const before = f.phase === 'update' ? st.prev : st.params;
	const win = rows(before, 'emb', V), wout = rows(before, 'out', V);
	const h = st.acts?.h ? Array.from(st.acts.h.data) : win[c];
	const u = st.acts?.scores ? Array.from(st.acts.scores.data) : wout.map((r) => dot(h, r));
	const p = st.acts?.p ? Array.from(st.acts.p.data) : softmax(u);
	const w: Worked = { phase: f.phase, words, c, o, V, d: h.length, win, wout, h, u, p, loss: -Math.log(p[o]), lr: st.lr ?? f.metrics?.lr ?? NaN };
	if (f.phase === 'backward' || f.phase === 'update') (w.gin = rows(st.grads, 'emb', V)), (w.gout = rows(st.grads, 'out', V));
	if (f.phase === 'update') {
		w.win1 = rows(st.params, 'emb', V);
		w.wout1 = rows(st.params, 'out', V);
		const u1 = w.wout1.map((r) => dot(w.win1![c], r));
		w.u1 = u1[o];
		w.loss1 = -Math.log(softmax(u1)[o]);
	}
	return w;
}

/** The update step's worked number: the weight of the center row with the largest gradient, old − lr × g = new. */
export function workedWeight(w: Worked): { k: number; old: number; g: number; new: number } | null {
	if (!w.gin || !w.win1) return null;
	const g = w.gin[w.c], k = g.reduce((b, x, i) => (Math.abs(x) > Math.abs(g[b]) ? i : b), 0);
	return { k, old: w.win[w.c][k], g: g[k], new: w.win1[w.c][k] };
}

/** The two directions in which the vectors spread most (PCA by power iteration on the d × d covariance), their mean,
 *  and the share of the total spread they keep. Signs: each axis's largest entry is positive. */
export function pca2(vs: number[][]): { mean: number[]; axes: number[][]; share: number } {
	const n = vs.length, d = vs[0]?.length ?? 0;
	const mean = Array.from({ length: d }, (_, k) => vs.reduce((s, v) => s + v[k], 0) / Math.max(1, n));
	const C = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => vs.reduce((s, v) => s + (v[a] - mean[a]) * (v[b] - mean[b]), 0) / Math.max(1, n)));
	const total = C.reduce((s, r, k) => s + r[k], 0);
	const axes: number[][] = [];
	let kept = 0;
	for (let a = 0; a < Math.min(2, d); a++) {
		let v = Array.from({ length: d }, (_, k) => 1 + 0.1 * k);
		for (let it = 0; it < 300; it++) {
			const w = C.map((r) => dot(r, v)), len = Math.sqrt(dot(w, w));
			if (!(len > 0)) break;
			v = w.map((x) => x / len);
		}
		const big = v.reduce((m, x, k) => (Math.abs(x) > Math.abs(v[m]) ? k : m), 0);
		if (v[big] < 0) v = v.map((x) => -x);
		const lambda = dot(v, C.map((r) => dot(r, v)));
		kept += lambda;
		for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] -= lambda * v[i] * v[j];
		axes.push(v);
	}
	return { mean, axes, share: total > 0 ? kept / total : NaN };
}
export const project = (vs: number[][], p: ReturnType<typeof pca2>) => vs.map((v) => p.axes.map((a) => dot(a, v.map((x, k) => x - p.mean[k]))));

/** the PCA of a training tape's last frame (the map's fixed axes while its frames play) */
export function finalPca(raw: CorpusRaw, tape: Tape | undefined) {
	const vs = tape?.length ? wordVectors(raw, tape.state(tape.length - 1)) : [];
	return vs.length > 1 ? pca2(vs) : null;
}

/** The pair of training step 0 of a run with this seed: the trainer's first draw from rng.fork('batch') (a claim test
 *  checks it equals the batch source `one` recorded), so a step can name it without running `one`. */
export function firstPair(raw: CorpusRaw, seed: number): Pair | null {
	const c = corpusOf(raw);
	return c.pairs.length ? c.pairs[createRng(seed).fork('batch').int(c.pairs.length)] : null;
}
