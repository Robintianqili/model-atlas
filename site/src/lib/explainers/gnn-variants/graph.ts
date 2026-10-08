// gnn-variants: the graph (pure TS). Zachary's karate club (1977) as networkx gives it, unweighted: 34 members, 78
// friendships, the faction each member joined (0 = Mr. Hi, 1 = the Officer) and the gcn package's offline layout
// (Kamada-Kawai). pipeline/gnn-variants/export.py writes the same graph to static/models/gnn-variants/karate.json;
// claims.test.ts checks they agree. The learner's raw input is a GraphRaw: the friendships and the member in focus.
import type { I18n } from '../../atlas/types/index.ts';

export interface GraphRaw { edges: [number, number][]; focus: number }
const EDGES = '0-1 0-2 0-3 0-4 0-5 0-6 0-7 0-8 0-10 0-11 0-12 0-13 0-17 0-19 0-21 0-31 1-2 1-3 1-7 1-13 1-17 1-19 1-21 1-30 2-3 2-7 2-8 2-9 2-13 2-27 2-28 2-32 3-7 3-12 3-13 4-6 4-10 5-6 5-10 5-16 6-16 8-30 8-32 8-33 9-33 13-33 14-32 14-33 15-32 15-33 18-32 18-33 19-33 20-32 20-33 22-32 22-33 23-25 23-27 23-29 23-32 23-33 24-25 24-27 24-31 25-31 26-29 26-33 27-33 28-31 28-33 29-32 29-33 30-32 30-33 31-32 31-33 32-33';
const XY = '381,437 210,449 423,776 551,527 584,277 545,220 462,185 527,423 273,482 364,771 406,290 207,203 423,401 484,632 86,631 0,851 579,0 238,389 181,760 286,578 206,652 276,313 171,888 408,1000 666,644 794,727 85,700 537,856 405,670 281,838 87,516 500,556 242,767 265,655';
const pairs = (s: string, sep: string) => s.split(' ').map((p) => p.split(sep).map(Number) as [number, number]);
export const N = 34;
export const LABELS = [...'0000000001000011001010111111111111'].map(Number);
export const POS = pairs(XY, ',').map(([x, y]) => [x / 1000, y / 1000] as [number, number]);
export const KARATE: GraphRaw = { edges: pairs(EDGES, '-'), focus: 8 };
export const ISOLATED: GraphRaw = { edges: KARATE.edges.filter((e) => !e.includes(11)), focus: 11 };
export const LABELLED = [0, 33];
export const FACTION: I18n[] = [{ en: 'Mr. Hi', zh: 'Mr. Hi（教练）' }, { en: 'the Officer', zh: '干事' }];
export const VARIANTS = ['gcn', 'sage', 'gin', 'gat'] as const;
export type Variant = (typeof VARIANTS)[number];
export const NAMES: Record<Variant, string> = { gcn: 'GCN', sage: 'GraphSAGE', gin: 'GIN', gat: 'GAT' };

const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
const ok = (i: unknown): i is number => Number.isInteger(i) && (i as number) >= 0 && (i as number) < N;
/** A valid GraphRaw from any raw input: friendships deduplicated (a < b, no self-loops), a member in focus. */
export function graphOf(raw: unknown): GraphRaw {
	const r = (raw ?? {}) as Partial<GraphRaw>;
	if (!Array.isArray(r.edges)) return KARATE;
	const seen = new Map<string, [number, number]>();
	for (const e of r.edges) if (Array.isArray(e) && ok(e[0]) && ok(e[1]) && e[0] !== e[1]) seen.set(key(e[0], e[1]), [Math.min(e[0], e[1]), Math.max(e[0], e[1])]);
	return { edges: [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]), focus: ok(r.focus) ? r.focus : KARATE.focus };
}
const hasEdge = (g: GraphRaw, a: number, b: number) => g.edges.some(([i, j]) => key(i, j) === key(a, b));
/** The same graph with friendship a-b added or removed. */
export const toggleEdge = (g: GraphRaw, a: number, b: number): GraphRaw =>
	graphOf({ ...g, edges: hasEdge(g, a, b) ? g.edges.filter(([i, j]) => key(i, j) !== key(a, b)) : [...g.edges, [a, b]] });
export const neighbours = (g: GraphRaw, i: number) => g.edges.flatMap(([a, b]) => (a === i ? [b] : b === i ? [a] : [])).sort((a, b) => a - b);
/** "0, 2 and 30" / "0、2 和 30" */
export function listOf(v: number[]): I18n {
	if (!v.length) return { en: 'none', zh: '无' };
	const h = v.slice(0, -1);
	return h.length ? { en: `${h.join(', ')} and ${v[v.length - 1]}`, zh: `${h.join('、')} 和 ${v[v.length - 1]}` } : { en: `${v[0]}`, zh: `${v[0]}` };
}

/** A [N·N] 0/1, symmetric, zero diagonal. */
export function adjacency(g: GraphRaw): Float64Array {
	const a = new Float64Array(N * N);
	for (const [i, j] of g.edges) a[i * N + j] = a[j * N + i] = 1;
	return a;
}
/** Â = D^-1/2 (A + I) D^-1/2 as model.py normalize(): deg = row sums of A + I. */
export function normalize(adj: ArrayLike<number>, n: number) {
	const deg = new Float64Array(n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += adj[i * n + j] + (i === j ? 1 : 0);
	const r = deg.map((d) => 1 / Math.sqrt(d));
	const ahat = new Float64Array(n * n).map((_, k) => r[Math.floor(k / n)] * (adj[k] + (k % (n + 1) === 0 ? 1 : 0)) * r[k % n]);
	return { deg, ahat };
}
/** The senders into member i: itself first, then its friends (row i of a 0/1 adjacency). */
export function senders(adj: ArrayLike<number>, n: number, i: number): number[] {
	const s = [i];
	for (let j = 0; j < n; j++) if (j !== i && adj[i * n + j]) s.push(j);
	return s;
}
export const leaky = (x: number, slope: number) => (x > 0 ? x : slope * x);

export type Params = Record<string, ArrayLike<number>>;
/** One GAT layer of model.py in f64: z = H Wᵀ (H null = one-hot), αᵢⱼ = softmax over the senders of LeakyReLU(a_dst·zᵢ + a_src·zⱼ),
 *  out = Σⱼ αᵢⱼ zⱼ + b. Returns z, α ([n·n], 0 off the senders) and out. */
export function gatLayer(h: ArrayLike<number> | null, I: number, p: Params, pre: string, adj: ArrayLike<number>, slope: number) {
	const W = p[`${pre}.weight`], as = p[`${pre}.att_src`], ad = p[`${pre}.att_dst`], b = p[`${pre}.bias`];
	const O = b.length;
	const z = new Float64Array(N * O);
	for (let j = 0; j < N; j++) for (let o = 0; o < O; o++) for (let k = 0; k < I; k++) z[j * O + o] += (h ? h[j * I + k] : j === k ? 1 : 0) * W[o * I + k];
	const dot = (v: ArrayLike<number>, j: number) => { let s = 0; for (let o = 0; o < O; o++) s += z[j * O + o] * v[o]; return s; }; // prettier-ignore
	const alpha = new Float64Array(N * N);
	const out = new Float64Array(N * O);
	for (let i = 0; i < N; i++) {
		const S = senders(adj, N, i);
		const e = S.map((j) => leaky(dot(ad, i) + dot(as, j), slope));
		const m = Math.max(...e);
		const ex = e.map((v) => Math.exp(v - m));
		let tot = 0;
		for (const v of ex) tot += v;
		S.forEach((j, k) => (alpha[i * N + j] = ex[k] / tot));
		for (let o = 0; o < O; o++) {
			let s = 0;
			for (const j of S) s += alpha[i * N + j] * z[j * O + o];
			out[i * O + o] = s + b[o];
		}
	}
	return { z, alpha, out };
}
/** The whole GAT network in f64: layer-1 and layer-2 α, the 2-D embedding and each member's score gap (the Officer's score − Mr. Hi's). */
export function forwardGat(p: Params, adj: ArrayLike<number>, slope: number) {
	const l1 = gatLayer(null, N, p, 'gat.gc1', adj, slope);
	const h1 = l1.out.map(Math.tanh);
	const l2 = gatLayer(h1, h1.length / N, p, 'gat.gc2', adj, slope);
	const emb = l2.out.map(Math.tanh);
	const W = p['gat.head.weight'], b = p['gat.head.bias'];
	const gap = Float64Array.from({ length: N }, (_, r) => (W[2] - W[0]) * emb[r * 2] + (W[3] - W[1]) * emb[r * 2 + 1] + (b[1] - b[0]));
	return { alpha: l1.alpha, alpha2: l2.alpha, emb, gap };
}
/** The unlabelled members a network sorts into their own faction, from two scores per member. */
export function sorted(z: ArrayLike<number>) {
	const rest = LABELS.map((_, i) => i).filter((i) => !LABELLED.includes(i));
	const wrong = rest.filter((i) => (z[i * 2 + 1] > z[i * 2] ? 1 : 0) !== LABELS[i]);
	return { rest: rest.length, right: rest.length - wrong.length, wrong };
}
