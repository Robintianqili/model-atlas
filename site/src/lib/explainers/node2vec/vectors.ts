// node2vec: plain float64 arithmetic on recorded weights (no autograd), shared by the views, the lesson and the claim
// tests (copied from the word2vec package's vectors.ts and cut to members): every number shown is read from, or
// recomputed from, the weights in the frames that ran.
import type { At, Tape, Tensor } from '../../atlas/types/index.ts';
import { FACTIONS } from '../../atlas/data/karate.ts';
import { N, walksOf } from './walks.ts';

type St = { params?: Record<string, Tensor>; prev?: Record<string, Tensor>; grads?: Record<string, Tensor>; batch?: Record<string, Tensor>; acts?: Record<string, Tensor>; lr?: number };
export const dot = (a: number[], b: number[]) => a.reduce((s, x, k) => s + x * b[k], 0);
export const cosine = (a: number[], b: number[]) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
/** the rows of W_in ('emb', the member vectors) or W_out ('out') in a record of tensors */
export function rows(rec: Record<string, Tensor> | undefined, table: 'emb' | 'out' = 'emb'): number[][] {
	const t = rec?.[`${table}.weight`];
	if (!t) return [];
	const d = t.shape[1];
	return Array.from({ length: t.shape[0] }, (_, i) => Array.from(t.data.subarray(i * d, (i + 1) * d)));
}
export const vecsAt = (st: unknown) => rows((st as St | undefined)?.params);
/** the member vectors of a source's last frame */
export function lastVecs(tape: Tape | null | undefined): number[][] {
	return tape?.length ? vecsAt(tape.state(tape.length - 1)) : [];
}
/** every other member by cosine similarity, closest first (ties: the smaller member) */
export function neighbours(vs: number[][], i: number): { j: number; cos: number }[] {
	return vs.map((r, j) => ({ j, cos: cosine(vs[i], r) })).filter((x) => x.j !== i).sort((a, b) => b.cos - a.cos || a.j - b.j);
}

/** How the vectors sit by faction (FACTIONS: the side each member joined, never shown to training): `same`, members whose
 *  nearest member joined their side; `right`, the 32 others sorted by "closer (cosine) to member 0 or to member 33"; `hop`,
 *  the mean number of hops from each member to its 3 nearest members (members without friends left out). */
export function factionStats(raw: unknown, vs: number[][]) {
	if (vs.length < N) return null;
	const H = walksOf(raw).hop, nn = vs.map((_, i) => neighbours(vs, i));
	const same = nn.filter((x, i) => FACTIONS[x[0].j] === FACTIONS[i]).length;
	const side = vs.map((v) => (cosine(v, vs[33]) > cosine(v, vs[0]) ? 1 : 0));
	const wrong = side.flatMap((f, i) => (i !== 0 && i !== 33 && f !== FACTIONS[i] ? [i] : []));
	const linked = nn.flatMap((x, i) => (H[i].some((d, j) => j !== i && d > 0) ? [x.slice(0, 3).reduce((a, y) => a + Math.max(0, H[i][y.j]), 0) / 3] : []));
	return { nn, same, side, wrong, right: N - 2 - wrong.length, hop: linked.reduce((a, b) => a + b, 0) / Math.max(1, linked.length), H };
}

/** One training step on one pair, at a fine frame of source `one` (forward / loss / backward / update): h, the scores u
 *  and the shares p from the weights the step started from; gh, the gradient of h; h1, the center's row after the step;
 *  k, the number of h with the largest gradient (the one the narration works out). */
export interface Worked { c: number; o: number; win: number[][]; wout: number[][]; h: number[]; gh: number[]; h1: number[]; u: number[]; p: number[]; loss: number; lr: number; k: number; old: number; g: number; new: number; u1: number; loss1: number }
export function worked(at: At): Worked | null {
	const st = at.state as St | undefined, b = st?.batch;
	if (!st?.params || !st.prev || !st.grads || !b?.c?.data.length) return null;
	const c = b.c.data[0], o = b.o.data[0], win = rows(st.prev), wout = rows(st.prev, 'out'), w1 = rows(st.params), o1 = rows(st.params, 'out');
	const h = win[c], u = wout.map((r) => dot(h, r)), m = Math.max(...u), e = u.map((x) => Math.exp(x - m)), s = e.reduce((a, x) => a + x, 0);
	const g = rows(st.grads)[c], k = g.reduce((bk, x, i) => (Math.abs(x) > Math.abs(g[bk]) ? i : bk), 0);
	const u1 = o1.map((r) => dot(w1[c], r)), m1 = Math.max(...u1), s1 = u1.reduce((a, x) => a + Math.exp(x - m1), 0);
	return { c, o, win, wout, h, gh: g, h1: w1[c], u, p: e.map((x) => x / s), loss: m + Math.log(s) - u[o], lr: st.lr ?? NaN, k, old: h[k], g: g[k], new: w1[c][k], u1: u1[o], loss1: m1 + Math.log(s1) - u1[o] };
}

/** The two directions in which the vectors spread most (PCA by power iteration on the d × d covariance), their mean,
 *  and the share of the total spread they keep. Signs: each axis's largest entry is positive. (word2vec's pca2.) */
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
/** the best vertical cut through the map: the most members a threshold x on the first map axis puts on their own side
 *  (either side may be Mr. Hi's), and that threshold (halfway between two neighbouring dots) */
export function cutOf(xy: number[][]): { ok: number; x: number } {
	const v = xy.map((p, i) => [p[0], FACTIONS[i]]).sort((a, b) => a[0] - b[0]);
	let ok = -1, x = NaN;
	for (let c = 0; c <= v.length; c++) {
		const n = v.reduce((a, [, f], i) => a + ((i < c ? f === 0 : f === 1) ? 1 : 0), 0), at = c === 0 ? v[0][0] - 1 : c === v.length ? v[c - 1][0] + 1 : (v[c - 1][0] + v[c][0]) / 2;
		if (Math.max(n, v.length - n) > ok) (ok = Math.max(n, v.length - n)), (x = at);
	}
	return { ok, x };
}
