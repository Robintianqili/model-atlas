// transe: plain float64 arithmetic on recorded weights (no autograd), shared by the trainer's metrics, the process, the
// views, the lesson and the claim tests: every number shown is read from, or recomputed from, the weights in the
// frames that ran. A distance is summed in the order k = 0, 1, ... as pipeline/transe/reference.py sums it.
import type { At, Tape, Tensor } from '../../atlas/types/index.ts';
import { NE, type Fact } from './kg.ts';

type St = { params?: Record<string, Tensor>; prev?: Record<string, Tensor>; grads?: Record<string, Tensor>; batch?: Record<string, Tensor>; lr?: number };
export type Vecs = { E: number[][]; R: number[][] };
const rowsOf = (t: Tensor | undefined) => (t ? Array.from({ length: t.shape[0] }, (_, i) => Array.from(t.data.subarray(i * t.shape[1], (i + 1) * t.shape[1]))) : []);
/** the entity and relation rows of a record of tensors (a frame's params, prev or grads) */
export const tablesOf = (rec: Record<string, Tensor> | undefined): Vecs => ({ E: rowsOf(rec?.['ent.weight']), R: rowsOf(rec?.['rel.weight']) });
export const vecsAt = (st: unknown) => tablesOf((st as St | undefined)?.params);
export const lastVecs = (tape: Tape | null | undefined): Vecs => (tape?.length ? vecsAt(tape.state(tape.length - 1)) : { E: [], R: [] });

/** ‖h + r − t‖ from flat tables (row-major, `dim` columns) */
export function distFlat(E: ArrayLike<number>, R: ArrayLike<number>, dim: number, h: number, r: number, t: number): number {
	let s = 0;
	for (let k = 0; k < dim; k++) {
		const v = E[h * dim + k] + R[r * dim + k] - E[t * dim + k];
		s += v * v;
	}
	return Math.sqrt(s);
}
export const dist = (v: Vecs, h: number, r: number, t: number) => distFlat(v.E.flat(), v.R.flat(), v.R[0]?.length ?? 0, h, r, t);
/** every entity by its distance from h + r, closest first (ties: the smaller id) */
export function ranked(v: Vecs, h: number, r: number): { x: number; d: number }[] {
	const E = v.E.flat(), R = v.R.flat(), dim = v.R[0]?.length ?? 0;
	return Array.from({ length: NE }, (_, x) => ({ x, d: distFlat(E, R, dim, h, r, x) })).sort((a, b) => a.d - b.d || a.x - b.x);
}
/** the true tails of a question among the facts */
export const tailsOf = (fs: Fact[], h: number, r: number) => fs.filter((f) => f[0] === h && f[1] === r).map((f) => f[2]);
/** the rank (from 1) of tail t for (h, r, ?), with the other true tails of the question left out ("filtered") */
export function rankOf(v: Vecs, fs: Fact[], [h, r, t]: Fact): number {
	const other = new Set(tailsOf(fs, h, r).filter((x) => x !== t));
	return ranked(v, h, r).filter((c) => !other.has(c.x)).findIndex((c) => c.x === t) + 1;
}

/** avgLoss: max(0, γ + d(fact) − d(corrupted)) averaged over every fact and every corruption (each other entity as the
 *  head, then as the tail): the loss of one step on average; hits1: the share of facts whose tail is the closest entity
 *  to h + r, other true tails left out (NaN without facts) */
export function evalAll(E: ArrayLike<number>, R: ArrayLike<number>, dim: number, fs: Fact[], gamma: number) {
	let loss = 0, hits = 0;
	for (const [h, r, t] of fs) {
		const d = distFlat(E, R, dim, h, r, t), other = new Set(tailsOf(fs, h, r));
		let first = 1;
		for (let x = 0; x < NE; x++) {
			const dh = x === h ? Infinity : distFlat(E, R, dim, x, r, t), dt = x === t ? Infinity : distFlat(E, R, dim, h, r, x);
			loss += Math.max(0, gamma + d - dh) + Math.max(0, gamma + d - dt);
			if (!other.has(x) && dt < d) first = 0;
		}
		hits += first;
	}
	return { avgLoss: fs.length ? loss / (fs.length * 2 * (NE - 1)) : NaN, hits1: fs.length ? hits / fs.length : NaN };
}

/** One training step, read at the 'update' frame of a fine train source: the fact and its corrupted copy, the vectors
 *  before (prev) and after (params) the step, their gradients, both distances before and after, the loss, and k, the
 *  number of h whose gradient is largest (the one the narration works out). */
export interface Worked { h: number; r: number; t: number; h2: number; t2: number; tail: boolean; x: number; before: Vecs; after: Vecs; grad: Vecs; dPos: number; dNeg: number; dPos1: number; dNeg1: number; loss: number; loss1: number; gamma: number; lr: number; k: number; old: number; g: number; new: number }
export function worked(at: At, gamma: number): Worked | null {
	const st = at.state as St | undefined, b = st?.batch;
	if (!st?.params || !st.prev || !st.grads || !b?.h?.data.length) return null;
	const [h, r, t, h2, t2] = ['h', 'r', 't', 'h2', 't2'].map((k) => b[k].data[0]);
	const before = tablesOf(st.prev), after = tablesOf(st.params), grad = tablesOf(st.grads);
	const g = grad.E[h], k = g.reduce((bk, x, i) => (Math.abs(x) > Math.abs(g[bk]) ? i : bk), 0);
	const dPos = dist(before, h, r, t), dNeg = dist(before, h2, r, t2), dPos1 = dist(after, h, r, t), dNeg1 = dist(after, h2, r, t2);
	const tail = h2 === h;
	return { h, r, t, h2, t2, tail, x: tail ? t2 : h2, before, after, grad, dPos, dNeg, dPos1, dNeg1, loss: Math.max(0, gamma + dPos - dNeg), loss1: Math.max(0, gamma + dPos1 - dNeg1), gamma, lr: st.lr ?? NaN, k, old: before.E[h][k], g: g[k], new: after.E[h][k] };
}

/** The two directions in which the entity points spread most (PCA by power iteration on the dim × dim covariance), their
 *  mean, and the share of the total spread they keep; each axis's largest entry is positive (node2vec's pca2). */
const dot = (a: number[], b: number[]) => a.reduce((s, x, k) => s + x * b[k], 0);
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
export type Pca = ReturnType<typeof pca2>;
/** a point onto the map (its mean subtracted), and an arrow (no mean: h + r maps to the map of h plus the map of r) */
export const onMap = (p: Pca, v: number[]) => p.axes.map((a) => dot(a, v.map((x, k) => x - p.mean[k])));
export const arrowOnMap = (p: Pca, r: number[]) => p.axes.map((a) => dot(a, r));
/** mean distance between two points of a set of entities */
export function spread(v: Vecs, ids: number[]): number {
	let s = 0, n = 0;
	for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) (s += Math.sqrt(v.E[ids[a]].reduce((q, x, k) => q + (x - v.E[ids[b]][k]) ** 2, 0))), n++;
	return n ? s / n : NaN;
}
