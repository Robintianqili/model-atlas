// gcn: the graph (pure TS). Zachary's karate club (1977) as networkx gives it, unweighted: 34 members, 78 friendships,
// the faction each member joined (0 = Mr. Hi, 1 = the Officer) and an offline layout (Kamada-Kawai) for the drawings.
// pipeline/gcn/export.py writes the same graph to static/models/gcn/karate.json; claims.test.ts checks they agree.
// The learner's raw input is a GraphRaw: the friendships (editable), the labelled members and the member in focus.
import type { I18n } from '../../atlas/types/index.ts';

export interface GraphRaw { edges: [number, number][]; labelled: number[]; focus: number }
const EDGES = '0-1 0-2 0-3 0-4 0-5 0-6 0-7 0-8 0-10 0-11 0-12 0-13 0-17 0-19 0-21 0-31 1-2 1-3 1-7 1-13 1-17 1-19 1-21 1-30 2-3 2-7 2-8 2-9 2-13 2-27 2-28 2-32 3-7 3-12 3-13 4-6 4-10 5-6 5-10 5-16 6-16 8-30 8-32 8-33 9-33 13-33 14-32 14-33 15-32 15-33 18-32 18-33 19-33 20-32 20-33 22-32 22-33 23-25 23-27 23-29 23-32 23-33 24-25 24-27 24-31 25-31 26-29 26-33 27-33 28-31 28-33 29-32 29-33 30-32 30-33 31-32 31-33 32-33';
const XY = '381,437 210,449 423,776 551,527 584,277 545,220 462,185 527,423 273,482 364,771 406,290 207,203 423,401 484,632 86,631 0,851 579,0 238,389 181,760 286,578 206,652 276,313 171,888 408,1000 666,644 794,727 85,700 537,856 405,670 281,838 87,516 500,556 242,767 265,655';
const pairs = (s: string, sep: string) => s.split(' ').map((p) => p.split(sep).map(Number) as [number, number]);
export const N = 34;
export const LABELS = [...'0000000001000011001010111111111111'].map(Number);
export const POS = pairs(XY, ',').map(([x, y]) => [x / 1000, y / 1000] as [number, number]);
export const KARATE: GraphRaw = { edges: pairs(EDGES, '-'), labelled: [0, 33], focus: 8 };
export const FACTION: I18n[] = [{ en: 'Mr. Hi', zh: 'Mr. Hi（教练）' }, { en: 'the Officer', zh: '干事' }];

const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
const ok = (i: unknown): i is number => Number.isInteger(i) && (i as number) >= 0 && (i as number) < N;
/** A valid GraphRaw from any raw input: friendships deduplicated (a < b, no self-loops), at most 2 labelled members. */
export function graphOf(raw: unknown): GraphRaw {
	const r = (raw ?? {}) as Partial<GraphRaw>;
	if (!Array.isArray(r.edges)) return KARATE;
	const seen = new Map<string, [number, number]>();
	for (const e of r.edges) if (Array.isArray(e) && ok(e[0]) && ok(e[1]) && e[0] !== e[1]) seen.set(key(e[0], e[1]), [Math.min(e[0], e[1]), Math.max(e[0], e[1])]);
	const labelled = [...new Set((r.labelled ?? []).filter(ok))].slice(-2);
	return { edges: [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]), labelled, focus: ok(r.focus) ? r.focus : KARATE.focus };
}
export const hasEdge = (g: GraphRaw, a: number, b: number) => g.edges.some(([i, j]) => key(i, j) === key(a, b));
/** The same graph with friendship a-b added or removed. */
export const toggleEdge = (g: GraphRaw, a: number, b: number): GraphRaw =>
	graphOf({ ...g, edges: hasEdge(g, a, b) ? g.edges.filter(([i, j]) => key(i, j) !== key(a, b)) : [...g.edges, [a, b]] });
/** Click to label: a member already labelled is dropped; otherwise it joins and the oldest of two leaves. */
export const toggleLabel = (g: GraphRaw, i: number): GraphRaw =>
	graphOf({ ...g, labelled: g.labelled.includes(i) ? g.labelled.filter((j) => j !== i) : [...g.labelled, i] });
export const neighbours = (g: GraphRaw, i: number) => g.edges.flatMap(([a, b]) => (a === i ? [b] : b === i ? [a] : [])).sort((a, b) => a - b);

/** A [N·N] 0/1, symmetric, zero diagonal. */
export function adjacency(g: GraphRaw): Float64Array {
	const a = new Float64Array(N * N);
	for (const [i, j] of g.edges) a[i * N + j] = a[j * N + i] = 1;
	return a;
}
/** Â = D^-1/2 (A + I) D^-1/2 as pipeline/gcn/model.py normalize(): loops = A + I, deg = row sums, r = 1/√deg. */
export function normalize(adj: ArrayLike<number>, n: number) {
	const loops = new Float64Array(n * n);
	const deg = new Float64Array(n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += loops[i * n + j] = adj[i * n + j] + (i === j ? 1 : 0);
	const r = deg.map((d) => 1 / Math.sqrt(d));
	const ahat = loops.map((v, k) => r[Math.floor(k / n)] * v * r[k % n]);
	return { loops, deg, ahat };
}
/** Steps from member i to every member (−1: unreachable), by breadth-first search. */
export function hops(g: GraphRaw, i: number): number[] {
	const d = new Array<number>(N).fill(-1);
	d[i] = 0;
	for (let q = [i]; q.length; ) {
		const next: number[] = [];
		for (const v of q) for (const w of neighbours(g, v)) if (d[w] < 0) (d[w] = d[v] + 1), next.push(w);
		q = next;
	}
	return d;
}
export type Params = Record<string, { data: ArrayLike<number>; shape: number[] }>;
/** The GCN of model.py in f64 from named parameters (gc1.weight, …): the 4-number rows h1, the 2-D embedding and the
 *  faction scores of every member (X is one-hot, so X W1ᵀ is W1ᵀ). */
export function forwardGcn(p: Params, ahat: ArrayLike<number>) {
	const layer = (x: ArrayLike<number> | null, name: string, I: number) => {
		const W = p[`${name}.weight`].data;
		const b = p[`${name}.bias`].data;
		const O = b.length;
		const xw = new Float64Array(N * O);
		for (let r = 0; r < N; r++) for (let o = 0; o < O; o++) for (let i = 0; i < I; i++) xw[r * O + o] += (x ? x[r * I + i] : r === i ? 1 : 0) * W[o * I + i];
		const y = new Float64Array(N * O);
		for (let r = 0; r < N; r++) for (let o = 0; o < O; o++) {
			let s = 0;
			for (let j = 0; j < N; j++) s += ahat[r * N + j] * xw[j * O + o];
			y[r * O + o] = Math.tanh(s + b[o]);
		}
		return y;
	};
	const h1 = layer(null, 'gc1', N);
	const emb = layer(h1, 'gc2', h1.length / N);
	const W = p['head.weight'].data;
	const b = p['head.bias'].data;
	const z = new Float64Array(N * 2);
	for (let r = 0; r < N; r++) for (let c = 0; c < 2; c++) z[r * 2 + c] = W[c * 2] * emb[r * 2] + W[c * 2 + 1] * emb[r * 2 + 1] + b[c];
	return { h1, emb, z };
}
/** Members other than the labelled ones that are sorted into their own faction, from two scores per member. */
export function sorted(z: ArrayLike<number>, labelled: number[]) {
	const rest = LABELS.map((_, i) => i).filter((i) => !labelled.includes(i));
	const wrong = rest.filter((i) => (z[i * 2 + 1] > z[i * 2] ? 1 : 0) !== LABELS[i]);
	return { rest: rest.length, right: rest.length - wrong.length, wrong, acc: rest.length ? 1 - wrong.length / rest.length : NaN };
}
