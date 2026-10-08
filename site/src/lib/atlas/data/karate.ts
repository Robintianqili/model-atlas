// Zachary's karate club (1977) for the kit (K1, rule of two: the gcn and gnn-variants packages each carried a copy):
// 34 members, 78 friendships, unweighted, exactly as pipeline/atlas_kit/data/karate.py reads it from networkx
// (scripts/atlas/tests/k1.test.ts checks that they agree), the faction each member joined (0 = Mr. Hi, 1 = the Officer)
// and an offline layout (Kamada-Kawai, [0, 1]²) for drawings. Plus the graph helpers both packages wrote: a valid graph
// from any raw input, adjacency, Â = D^-1/2 (A + I) D^-1/2, neighbours, toggling a friendship, hops by breadth-first
// search. Pure TS. Cite: W. W. Zachary, "An information flow model for conflict and fission in small groups", Journal of
// Anthropological Research 33(4), 452-473 (1977); data via NetworkX (BSD-3-Clause).
import type { I18n } from '../types/core.ts';

/** The learner's graph: friendships (a < b) and the member in focus; `labelled` for pages that label members. */
export interface GraphRaw {
	edges: [number, number][];
	focus: number;
	labelled?: number[];
}

const EDGES =
	'0-1 0-2 0-3 0-4 0-5 0-6 0-7 0-8 0-10 0-11 0-12 0-13 0-17 0-19 0-21 0-31 1-2 1-3 1-7 1-13 1-17 1-19 1-21 1-30 2-3 2-7 2-8 2-9 2-13 2-27 2-28 2-32 3-7 3-12 3-13 4-6 4-10 5-6 5-10 5-16 6-16 8-30 8-32 8-33 9-33 13-33 14-32 14-33 15-32 15-33 18-32 18-33 19-33 20-32 20-33 22-32 22-33 23-25 23-27 23-29 23-32 23-33 24-25 24-27 24-31 25-31 26-29 26-33 27-33 28-31 28-33 29-32 29-33 30-32 30-33 31-32 31-33 32-33';
const XY =
	'381,437 210,449 423,776 551,527 584,277 545,220 462,185 527,423 273,482 364,771 406,290 207,203 423,401 484,632 86,631 0,851 579,0 238,389 181,760 286,578 206,652 276,313 171,888 408,1000 666,644 794,727 85,700 537,856 405,670 281,838 87,516 500,556 242,767 265,655';
const pairs = (s: string, sep: string) => s.split(' ').map((p) => p.split(sep).map(Number) as [number, number]);

/** members */
export const N = 34;
/** the faction each member joined: 0 = Mr. Hi, 1 = the Officer (networkx's 'club' attribute) */
export const FACTIONS = [...'0000000001000011001010111111111111'].map(Number);
export const FACTION_NAMES: I18n[] = [
	{ en: 'Mr. Hi', zh: 'Mr. Hi（教练）' },
	{ en: 'the Officer', zh: '干事' }
];
/** an offline Kamada-Kawai layout in [0, 1]² (y down) */
export const POS: [number, number][] = pairs(XY, ',').map(([x, y]) => [x / 1000, y / 1000]);
/** the club as networkx gives it, every friendship once with a < b, sorted */
export const KARATE_EDGES: [number, number][] = pairs(EDGES, '-');
export const KARATE: GraphRaw = { edges: KARATE_EDGES, focus: 8 };
/** member 11, whose only friend is member 0, without that friendship: an isolated node (the edge fixture) */
export const ISOLATED: GraphRaw = { edges: KARATE_EDGES.filter((e) => !e.includes(11)), focus: 11 };

const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
const member = (i: unknown, n = N): i is number => Number.isInteger(i) && (i as number) >= 0 && (i as number) < n;

/** A valid graph from any raw input: friendships deduplicated (a < b, no self-loops), the focus a member (default 8),
 *  at most `maxLabelled` labelled members (kept when given). Not a graph: the club. */
export function graphOf(raw: unknown, o: { n?: number; maxLabelled?: number } = {}): GraphRaw {
	const n = o.n ?? N;
	const r = (raw ?? {}) as Partial<GraphRaw>;
	if (!Array.isArray(r.edges)) return { ...KARATE, edges: KARATE_EDGES.slice() };
	const seen = new Map<string, [number, number]>();
	for (const e of r.edges) if (Array.isArray(e) && member(e[0], n) && member(e[1], n) && e[0] !== e[1]) seen.set(key(e[0], e[1]), [Math.min(e[0], e[1]), Math.max(e[0], e[1])]);
	const out: GraphRaw = { edges: [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]), focus: member(r.focus, n) ? r.focus : KARATE.focus };
	if (Array.isArray(r.labelled)) out.labelled = [...new Set(r.labelled.filter((i) => member(i, n)))].slice(-(o.maxLabelled ?? n));
	return out;
}
export const hasEdge = (g: GraphRaw, a: number, b: number) => g.edges.some(([i, j]) => key(i, j) === key(a, b));
/** The same graph with friendship a-b added or removed. */
export const toggleEdge = (g: GraphRaw, a: number, b: number): GraphRaw =>
	graphOf({ ...g, edges: hasEdge(g, a, b) ? g.edges.filter(([i, j]) => key(i, j) !== key(a, b)) : [...g.edges, [a, b]] });
export const neighbours = (g: GraphRaw, i: number) => g.edges.flatMap(([a, b]) => (a === i ? [b] : b === i ? [a] : [])).sort((a, b) => a - b);

/** A as [n·n]: 0/1, symmetric, zero diagonal. */
export function adjacency(g: GraphRaw, n = N): Float64Array {
	const a = new Float64Array(n * n);
	for (const [i, j] of g.edges) a[i * n + j] = a[j * n + i] = 1;
	return a;
}
/** Â = D^-1/2 (A + I) D^-1/2 (Kipf & Welling): the self-loops A + I, the degrees D (row sums of A + I) and Â. */
export function normalizedAdjacency(adj: ArrayLike<number>, n: number): { loops: Float64Array; deg: Float64Array; ahat: Float64Array } {
	const loops = new Float64Array(n * n);
	const deg = new Float64Array(n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += loops[i * n + j] = adj[i * n + j] + (i === j ? 1 : 0);
	const r = deg.map((d) => 1 / Math.sqrt(d));
	const ahat = loops.map((v, k) => r[Math.floor(k / n)] * v * r[k % n]);
	return { loops, deg, ahat };
}
/** Steps from member i to every member (−1: unreachable), by breadth-first search. */
export function hops(g: GraphRaw, i: number, n = N): number[] {
	const d = new Array<number>(n).fill(-1);
	d[i] = 0;
	for (let q = [i]; q.length; ) {
		const next: number[] = [];
		for (const v of q) for (const w of neighbours(g, v)) if (d[w] < 0) (d[w] = d[v] + 1), next.push(w);
		q = next;
	}
	return d;
}
/** One-hot features X = I as [n·n] (every member is its own feature). */
export function oneHot(n = N): Float64Array {
	return Float64Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0));
}
