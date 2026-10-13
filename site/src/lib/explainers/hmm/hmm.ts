// hmm: the pieces every file shares (pure TS). The learner's input is an HmmRaw: the activities a friend reports, one per
// day (0 walk, 1 shop, 2 clean), and the model's three tables in whole percentages (rows sum to 100, every entry at least
// MINP): the start row pi, the transition table A (today's weather -> tomorrow's) and the emission table B (weather ->
// activity). dataOf() turns it into plain probabilities, the input of every process (and, as JSON, of the Python reference
// pipeline/hmm/reference.py); the starting tables travel with it (base) so the task can compare against them.
// The algorithms (forward, Viterbi, the joint probability of one path) are here too, so processes, views and claims run
// the very same arithmetic.
import type { I18n } from '../../atlas/types/index.ts';

export const I = (en: string, zh: string): I18n => ({ en, zh });
export const STATES = [I('Sunny', '晴天'), I('Rainy', '雨天')];
export const ACTS = [I('walk', '散步'), I('shop', '购物'), I('clean', '打扫')];
/** the activities as verbs in the past ("she walked") for the sampling step */
export const DID = [I('walked', '去散步'), I('shopped', '去购物'), I('cleaned', '在家打扫')];
export const K = STATES.length;
export const M = ACTS.length;
/** the free numbers of the three tables (each row sums to 1): K − 1 in the start row, K(K − 1) in A, K(M − 1) in B */
export const FREE = K - 1 + K * (K - 1) + K * (M - 1);
/** at most MAXT days; every table entry is at least MINP percent (so no activity is ever impossible) */
export const MAXT = 10;
export const MINP = 5;
export const PI0 = [60, 40];
export const A0 = [[75, 25], [35, 65]];
export const B0 = [[50, 35, 15], [10, 30, 60]];
/** the page's start: eight days; with the starting tables Viterbi calls day 3 (a clean day before a walk) sunny and day 6
 *  (a shop day between clean days) rainy, both against what the activity alone suggests */
export const START_OBS = [0, 0, 1, 2, 0, 2, 1, 2];
export const START: HmmRaw = { obs: START_OBS };
/** the sampling step makes up this many days */
export const SAMPLE_DAYS = 12;

export interface HmmRaw { obs?: number[]; pi?: number[]; A?: number[][]; B?: number[][] }
export interface Tables { pi: number[]; A: number[][]; B: number[][] }
export interface HmmData extends Tables { obs: number[]; base: Tables }

const fin = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
export function rawOf(raw: unknown): HmmRaw {
	return (raw && typeof raw === 'object' ? raw : {}) as HmmRaw;
}
/** a row of whole percentages that sums to 100 with every entry >= MINP, else the default row */
function rowOf(r: unknown, d: number[]): number[] {
	if (!Array.isArray(r) || r.length !== d.length || !r.every(fin)) return d;
	const x = r.map((v) => Math.round(v));
	return x.every((v) => v >= MINP) && x.reduce((a, b) => a + b, 0) === 100 ? x : d;
}
/** the learner's tables in percent (the defaults where a row is missing or invalid) */
export function percentOf(raw: unknown): Tables {
	const r = rawOf(raw);
	return { pi: rowOf(r.pi, PI0), A: A0.map((d, i) => rowOf(r.A?.[i], d)), B: B0.map((d, i) => rowOf(r.B?.[i], d)) };
}
export const obsOf = (raw: unknown): number[] => {
	const o = rawOf(raw).obs;
	return Array.isArray(o) ? o.filter((v) => fin(v) && v >= 0 && v < M).map(Math.floor).slice(0, MAXT) : START_OBS.slice();
};
const prob = (t: Tables): Tables => ({ pi: t.pi.map((v) => v / 100), A: t.A.map((r) => r.map((v) => v / 100)), B: t.B.map((r) => r.map((v) => v / 100)) });
export const BASE = prob({ pi: PI0, A: A0, B: B0 });
/** plain numbers for a process: the activities and the probabilities (percent / 100) */
export function dataOf(raw: unknown): HmmData {
	return { obs: obsOf(raw), ...prob(percentOf(raw)), base: BASE };
}

/** set entry j of a percentage row to v (clamped to [MINP, 100 − MINP·(n − 1)]) and share the rest among the other entries
 *  in proportion to what they hold above MINP (largest remainder; ties to the smaller index), so the row still sums to 100 */
export function setProb(row: number[], j: number, v: number): number[] {
	const n = row.length, out = row.slice(), others = row.map((_, i) => i).filter((i) => i !== j);
	out[j] = Math.round(Math.min(100 - MINP * (n - 1), Math.max(MINP, v)));
	const rest = 100 - out[j] - MINP * others.length, w = others.map((i) => Math.max(0, row[i] - MINP)), W = w.reduce((a, b) => a + b, 0);
	const share = w.map((x) => (W > 0 ? (rest * x) / W : rest / others.length)), fl = share.map(Math.floor);
	let left = rest - fl.reduce((a, b) => a + b, 0);
	const order = share.map((s, k) => [s - fl[k], k]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
	for (let q = 0; left > 0; q++, left--) fl[order[q][1]]++;
	others.forEach((i, k) => (out[i] = MINP + fl[k]));
	return out;
}

/** the index of the largest value (ties to the smallest index) */
export function argmax(xs: number[]): number {
	let b = 0;
	for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i;
	return b;
}
/** each day's weather from its activity alone: the state whose emission row makes the activity most likely */
export const naiveOf = (d: Tables & { obs: number[] }): number[] => d.obs.map((o) => argmax(d.B.map((r) => r[o])));
/** P(path, activities) = pi[s0] b[s0][o0] · Π a[s(t−1)][s(t)] b[s(t)][o(t)] */
export function jointOf(d: Tables & { obs: number[] }, path: number[]): number {
	let p = d.pi[path[0]] * d.B[path[0]][d.obs[0]];
	for (let t = 1; t < path.length; t++) p *= d.A[path[t - 1]][path[t]] * d.B[path[t]][d.obs[t]];
	return p;
}
/** the forward table alpha[t][j]: the probability of the activities of days 0..t, ending in state j on day t */
export function forwardOf(d: Tables & { obs: number[] }): number[][] {
	const al: number[][] = [];
	d.obs.forEach((o, t) => al.push(d.B.map((r, j) => (t ? al[t - 1].reduce((s, a, i) => s + a * d.A[i][j], 0) : d.pi[j]) * r[o])));
	return al;
}
/** Viterbi: delta[t][j] (the best single path's probability into state j on day t), psi[t][j] (its state the day before;
 *  ties to the smaller index), and the best path traced back from the best last state */
export function viterbiOf(d: Tables & { obs: number[] }): { delta: number[][]; psi: number[][]; path: number[] } {
	const delta: number[][] = [], psi: number[][] = [];
	d.obs.forEach((o, t) => {
		const row: number[] = [], from: number[] = [];
		for (let j = 0; j < K; j++) {
			const c = t ? delta[t - 1].map((v, i) => v * d.A[i][j]) : [d.pi[j]];
			const i = argmax(c);
			row.push(c[i] * d.B[j][o]);
			from.push(t ? i : -1);
		}
		delta.push(row);
		psi.push(from);
	});
	const T = d.obs.length, path = new Array<number>(T);
	if (T) path[T - 1] = argmax(delta[T - 1]);
	for (let t = T - 1; t > 0; t--) path[t - 1] = psi[t][path[t]];
	return { delta, psi, path };
}
/** the number of days on which two paths differ */
export const diffCount = (a: number[], b: number[]): number => a.reduce((s, v, t) => s + (v !== b[t] ? 1 : 0), 0);

/** the limits step's "true" weather: a made-up stretch of days in which every rainy spell lasts exactly RAIN days; the
 *  sunny spells after them run through SUNNY in turn, so the stretch ends on a sunny day and every rainy day has a next day */
export const RAIN = 3;
export const SUNNY = [4, 2, 6, 3, 5, 1, 7, 2, 5, 4];
export const SPELL_MAX = 8;
export const YEAR: number[] = SUNNY.flatMap((s) => [...Array(RAIN).fill(1), ...Array(s).fill(0)]);
