// two-tower: what the lesson and the views compute from the trace (pure TS, f64 sums of the traced f32 numbers).
//   retrieval  the scores of the 36 movies (layer 'scores': your vector against every row of the item table), the
//              movies you did not pick ranked by score (ties: the smaller index), the shortlist of the best K
//   ranking    the ranker of export.py (spec extra `ranker`): p = σ(w_s·s + w_p·pop + b), pop = the movie's likes in the
//              training logs (spec extra `likes`) on a log scale, standardized; it reorders the shortlist only
//   new movie  your new movie's vector (layer 'item.fc2') dotted with your vector, and its place among the catalog
//   costs      multiplications per request, from the spec's layer shapes
//   batch      B pairs of the logs (viewer, the first movie they liked, all movies different) through the exported towers
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import type { BatchIn } from './process.ts';
import { MOVIES, N_ITEMS, N_USER_F, itemFeatures, logUser, toPrefs, userFeatures } from './world.ts';

type RT = ExplainerRuntime;
/** the shortlist's size and the batch's size */
export const K = 8;
export const B = 6;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>, bo = 0) => {
	let s = 0;
	for (let d = 0; d < a.length; d++) s += a[d] * b[bo + d];
	return s;
};
/** the item table [36, D]: every movie's vector, computed once at export by the item tower */
export function table(rt: RT): { V: ArrayLike<number>; D: number } | null {
	const w = rt.model?.weights['scores.weight'];
	return w ? { V: w.data, D: w.shape[1] } : null;
}
export const rowOf = (rt: RT, i: number) => {
	const t = table(rt);
	return t ? Array.from(t.V).slice(i * t.D, (i + 1) * t.D) : [];
};
export type Layers = { x: ArrayLike<number>; h: ArrayLike<number>; y: ArrayLike<number> };
const towers = new WeakMap<GraphModel, Map<number, Layers>>();
/** one tower's input, hidden layer (after the ReLU) and vector: the learner's (the fwd trace), or a catalog movie's
 *  through the item tower (a forward pass of the exported graph on the movie's features; its user input is all 0) */
export function towerOf(rt: RT, tower: 'user' | 'item', movie?: number): Layers | null {
	const t = rt.sources.fwd?.trace?.t, m = rt.model;
	if (movie === undefined) return t?.[tower] ? { x: t[tower].data, h: t[`${tower}.relu`].data, y: t[`${tower}.fc2`].data } : null;
	if (!m) return null;
	if (!towers.has(m)) towers.set(m, new Map());
	const memo = towers.get(m)!;
	if (!memo.has(movie)) {
		const r = m.forward({ user: { shape: [N_USER_F], data: new Float32Array(N_USER_F) }, item: { shape: [8], data: itemFeatures(MOVIES[movie]) } }, { record: 'layers' }).t;
		memo.set(movie, { x: r.item.data, h: r['item.relu'].data, y: r['item.fc2'].data });
	}
	return memo.get(movie)!;
}
export type Scored = { i: number; s: number };
/** the movies you did not pick, best score first */
export function ranked(rt: RT): Scored[] {
	const s = T(rt, 'scores')?.data, picks = new Set(toPrefs(rt.input).picks);
	if (!s) return [];
	return MOVIES.map((_, i) => ({ i, s: s[i] })).filter((x) => !picks.has(x.i)).sort((a, b) => b.s - a.s || a.i - b.i);
}
export const shortlist = (rt: RT) => ranked(rt).slice(0, K);

type Ranker = { w: [number, number]; b: number; popMean: number; popStd: number };
const rankerOf = (rt: RT) => (rt.spec?.ranker ?? null) as Ranker | null;
export const likesOf = (rt: RT, i: number) => ((rt.spec?.likes ?? []) as number[])[i] ?? NaN;
export type Ranked = Scored & { pop: number; z: number; p: number };
/** the ranker on one movie; pop = its likes in the logs on a log scale, standardized (a new movie has 0 likes) */
export function rankOne(rt: RT, x: Scored, likes = likesOf(rt, x.i)): Ranked {
	const r = rankerOf(rt), pop = r ? (Math.log1p(likes) - r.popMean) / r.popStd : NaN, z = r ? r.w[0] * x.s + r.w[1] * pop + r.b : NaN;
	return { ...x, pop, z, p: 1 / (1 + Math.exp(-z)) };
}
/** the shortlist reordered by the ranker (ties: the retrieval order) */
export function reranked(rt: RT): (Ranked & { k: number })[] {
	return shortlist(rt).map((x, k) => ({ ...rankOne(rt, x), k })).sort((a, b) => b.z - a.z || a.k - b.k);
}

/** your new movie: its score and its place among the movies you did not pick (1 = it would come first) */
export function newMovie(rt: RT) {
	const u = T(rt, 'user.fc2')?.data, v = T(rt, 'item.fc2')?.data;
	if (!u || !v) return null;
	const s = dot(u, v), r = ranked(rt);
	const place = 1 + r.filter((x) => x.s > s).length;
	return { s, place, of: r.length + 1, inList: place <= K };
}
/** movies whose features equal the new movie's (it then has the same vector as they do) */
export const sameAs = (rt: RT) => {
	const f = itemFeatures(toPrefs(rt.input).item);
	return MOVIES.flatMap((m, i) => (itemFeatures(m).every((x, k) => x === f[k]) ? [i] : []));
};

/** multiplications: per layer in × out of each tower, D per score, and a joint network that reads a viewer and a movie
 *  together (the user tower's layers with the item features appended to its input) once per movie */
export function costs(rt: RT, items = N_ITEMS) {
	const shape = (id: string) => rt.model?.weights[`${id}.weight`]?.shape ?? [NaN, NaN];
	const mul = (id: string) => shape(id)[0] * shape(id)[1];
	const user = mul('user.fc1') + mul('user.fc2'), item = mul('item.fc1') + mul('item.fc2'), D = shape('scores')[1];
	const joint = (shape('user.fc1')[1] + shape('item.fc1')[1]) * shape('user.fc1')[0] + mul('user.fc2') + D;
	return { user, item, D, N: items, perRequest: user + items * D, offline: items * item, joint: items * joint, jointOne: joint };
}

/** the batch's pairs: viewers 0, 1, … of the logs with the first movie each liked, skipping a viewer whose movie is
 *  already in the batch or in a chosen viewer's history, or whose history holds a chosen movie (training leaves a
 *  viewer's own history out of their row; this batch has none to leave out) */
export function batchPairs(): { u: number; genres: number[]; history: number[]; movie: number }[] {
	const out: { u: number; genres: number[]; history: number[]; movie: number }[] = [];
	for (let u = 0; out.length < B; u++) {
		const l = logUser(u), m = l.liked[0];
		if (!out.some((p) => p.movie === m || p.history.includes(m) || l.history.includes(p.movie))) out.push({ u, genres: l.genres, history: l.history, movie: m });
	}
	return out;
}
const memo = new WeakMap<GraphModel, BatchIn>();
/** inbatch input: the B viewers through the user tower and their movies through the item tower (= their table rows) */
export function batchOf(rt: RT): BatchIn | null {
	const m = rt.model;
	if (!m) return null;
	if (!memo.has(m)) {
		const vs = batchPairs().map((p) => {
			const tr = m.forward({ user: { shape: [15], data: userFeatures(p.genres, p.history) }, item: { shape: [8], data: itemFeatures(MOVIES[p.movie]) } }, { record: 'layers' });
			return [Array.from(tr.t['user.fc2'].data), Array.from(tr.t['item.fc2'].data)];
		});
		memo.set(m, { U: vs.flatMap((v) => v[0]), V: vs.flatMap((v) => v[1]), B, D: vs[0][0].length });
	}
	return memo.get(m)!;
}
