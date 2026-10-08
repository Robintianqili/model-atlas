// node2vec: the learner's input -> biased random walks (Grover & Leskovec 2016) -> (center, context) pairs. Pure and
// shared by the adapter, the walk process, the trainer, the views and the lesson. The input is the kit graph (Zachary's
// karate club, editable, atlas/data/karate.ts) plus the return parameter p and the in-out parameter q.
// A walk from v, having come from t, picks the next member x among v's friends with weight α: 1/p when x = t (back),
// 1 when x is also a friend of t, 1/q otherwise (farther from t). The first move of a walk has no t: every friend 1.
// The draw: u = rng.next(), the first x whose running sum of weights exceeds u × (the sum of all weights); friends in
// increasing order. pipeline/node2vec/reference.py repeats this with ak.Rng (bit-identical) for the parity gate.
import type { Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { FACTIONS, graphOf, hops, KARATE_EDGES, N, neighbours, type GraphRaw } from '../../atlas/data/karate.ts';

export { FACTIONS, N };
export type WalkRaw = (GraphRaw & { p?: number; q?: number }) | null | undefined;
/** the values of p and q the input offers */
export const PQ = [0.25, 0.5, 1, 2, 4];
export const START_P = 1;
export const START_Q = 0.5;
/** walks per member, members per walk, the window (members on each side) */
export const WALKS = 10;
export const LEN = 10;
export const WIN = 2;
/** the seed of the walks (the walk process's source seed: the same draws) */
export const WALK_SEED = 7;
export const START: GraphRaw & { p: number; q: number } = { edges: KARATE_EDGES, focus: 8, p: START_P, q: START_Q };

/** source 'base' trains on the learner's graph with the start p and q: the task step compares the learner's run with it */
export const BASE = { p: START_P, q: START_Q };
export const runInput = (raw: unknown, source?: string) => (source === 'base' ? { ...inputOf(raw), ...BASE } : inputOf(raw));

const pick = (v: unknown, d: number) => (PQ.includes(Number(v)) ? Number(v) : d);
/** a valid input: the kit graph (graphOf) with p and q from PQ (default 1 and 0.5) */
export function inputOf(raw: unknown): GraphRaw & { p: number; q: number } {
	const r = (raw ?? {}) as { p?: unknown; q?: unknown };
	return { ...graphOf(raw), p: pick(r.p, START_P), q: pick(r.q, START_Q) };
}

/** The candidates of one move from v (having come from t; t = −1 on a walk's first move): friends in increasing order,
 *  each one's kind (0 back to t, 1 also a friend of t, 2 farther; 1 on a first move), weight α and chance. */
export interface Move { x: number[]; kind: number[]; a: number[]; prob: number[]; total: number }
export function moveOf(nb: number[][], t: number, v: number, p: number, q: number): Move {
	const x = nb[v], kind = x.map((y) => (t < 0 ? 1 : y === t ? 0 : nb[t].includes(y) ? 1 : 2));
	const a = kind.map((k) => (k === 0 ? 1 / p : k === 1 ? 1 : 1 / q));
	let total = 0;
	for (const w of a) total += w;
	return { x, kind, a, prob: a.map((w) => w / total), total };
}

/** One event per move: walk w (round r, from member s), move k (0-based) from v to x (candidate i), u the draw. */
export interface Ev { w: number; r: number; s: number; k: number; t: number; v: number; x: number; u: number; i: number }
export function* walkEvents(raw: unknown, rng: Rng): Generator<Ev> {
	const g = inputOf(raw), nb = Array.from({ length: N }, (_, i) => neighbours(g, i));
	for (let r = 0; r < WALKS; r++)
		for (let s = 0; s < N; s++) {
			let t = -1, v = s;
			for (let k = 0; k < LEN - 1 && nb[v].length; k++) {
				const m = moveOf(nb, t, v, g.p, g.q), u = rng.next(), target = u * m.total;
				let i = 0, cum = 0;
				for (; i < m.a.length - 1; i++) if (target < (cum += m.a[i])) break;
				yield { w: r * N + s, r, s, k, t, v, x: m.x[i], u, i };
				t = v;
				v = m.x[i];
			}
		}
}

export interface Pair { c: number; o: number; w: number; i: number; j: number }
export interface Walks { p: number; q: number; nb: number[][]; walks: number[][]; pairs: Pair[]; cooc: Float64Array; far: number; distinct: number; hop: number[][] }
const cache = new Map<string, Walks>();
/** Every walk of an input (WALKS rounds × N members, in that order), its pairs and their counts (cooc[c·N + o]: how often
 *  o is a context of c); the last four are cached (two runs on other p and q may alternate). far: the mean
 *  number of hops between a walk's start and its last member; distinct: the mean number of different members per walk;
 *  hop[i][j]: hops from member i to member j in the graph (−1: unreachable). */
export function walksOf(raw: unknown): Walks {
	const g = inputOf(raw), key = JSON.stringify([g.edges, g.p, g.q]);
	const hit = cache.get(key);
	if (hit) return hit;
	const nb = Array.from({ length: N }, (_, i) => neighbours(g, i));
	const walks = Array.from({ length: WALKS * N }, (_, w) => [w % N]);
	for (const e of walkEvents(g, createRng(WALK_SEED))) walks[e.w].push(e.x);
	const pairs: Pair[] = [], cooc = new Float64Array(N * N);
	walks.forEach((ids, w) => {
		for (let i = 0; i < ids.length; i++)
			for (let j = Math.max(0, i - WIN); j <= Math.min(ids.length - 1, i + WIN); j++) if (j !== i) pairs.push({ c: ids[i], o: ids[j], w, i, j }), (cooc[ids[i] * N + ids[j]] += 1);
	});
	const hop = Array.from({ length: N }, (_, s) => hops(g, s));
	const moved = walks.filter((ids) => ids.length > 1);
	const far = moved.reduce((a, ids) => a + hop[ids[0]][ids[ids.length - 1]], 0) / Math.max(1, moved.length);
	const distinct = moved.reduce((a, ids) => a + new Set(ids).size, 0) / Math.max(1, moved.length);
	const v = { p: g.p, q: g.q, nb, walks, pairs, cooc, far, distinct, hop };
	if (cache.size >= 4) cache.delete(cache.keys().next().value!);
	cache.set(key, v);
	return v;
}

/** The lesson's walk: the member in focus's first walk, else (a member without friends) the first walk that moves; −1
 *  when no walk moves. A walk that moves at all has LEN members (it arrives at members along friendships). */
export function exampleWalk(raw: unknown): number {
	const W = walksOf(raw), f = inputOf(raw).focus;
	return W.walks[f].length > 1 ? f : W.walks.findIndex((ids) => ids.length > 1);
}
