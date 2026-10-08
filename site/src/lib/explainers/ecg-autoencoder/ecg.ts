// ecg-autoencoder helpers (pure TS, shared by the views, the lesson and the claim tests):
//   the learner's beat and its rebuild from the trace, the errors point by point and the anomaly score (the root of the
//   mean squared error, as export.py computes it); the rebuild of any other beat by the same model; the threshold (the
//   learner's, kept in the raw input, set on the stage); the test set (testset.json: PyTorch scores and kinds) counted at
//   a threshold; the regions of a beat, named for the narration.
import type { ExplainerRuntime, GraphModel, I18n } from '../../atlas/types/index.ts';
import { I, KINDS, N, beatOf, type Kind } from './beats.ts';
import { rawOf } from './adapters/series/adapter.ts';

type RT = ExplainerRuntime;
export const T = (rt: RT, id: string) => rt.sources.fwd?.trace?.t[id];

/** The threshold the page starts with: just above fact thr95 (95 in 100 normal validation beats score below it). */
export const THR0 = 0.03;
export const THR_MIN = 0.01, THR_MAX = 0.07, THR_STEP = 0.0005;
export const thrOf = (raw: unknown) => {
	const v = rawOf(raw).thr;
	return typeof v === 'number' && Number.isFinite(v) ? Math.min(THR_MAX, Math.max(THR_MIN, v)) : THR0;
};

/** errors e_i = x_i − x̂_i, the score sqrt(mean e²), the sum of squares and the point with the largest |e| */
export function errorsOf(x: ArrayLike<number>, xr: ArrayLike<number>) {
	const e = Array.from(x, (v, i) => v - xr[i]);
	let ss = 0, worst = 0;
	e.forEach((v, i) => {
		ss += v * v;
		if (Math.abs(v) > Math.abs(e[worst])) worst = i;
	});
	return { e, ss, score: e.length ? Math.sqrt(ss / e.length) : NaN, worst };
}

/** the learner's beat, its rebuild and their errors (null before the trace has run) */
export function recOf(rt: RT) {
	const x = T(rt, 'input'), xr = T(rt, 'out');
	if (!x || !xr || x.data.length !== N) return null;
	return { x: x.data, xr: xr.data, code: Array.from(T(rt, 'code')?.data ?? []), ...errorsOf(x.data, xr.data) };
}

const REB = new WeakMap<GraphModel, Map<string, Float32Array>>();
/** the rebuild of a preset-style beat (kind, seed) by the page's model (memoized) */
export function rebuildOf(m: GraphModel, kind: Kind, seed: number) {
	let memo = REB.get(m);
	if (!memo) REB.set(m, (memo = new Map()));
	const key = `${kind}${seed}`, x = beatOf(kind, seed);
	let xr = memo.get(key);
	if (!xr) memo.set(key, (xr = Float32Array.from(m.forward({ input: { shape: [N], data: x } }, { record: 'layers' }).t.out.data)));
	return { x, xr, ...errorsOf(x, xr) };
}
/** the odd beat of the predict step and its normal twin (the same draws without the raised ST segment) */
export const ODD = { kind: 'stUp' as Kind, seed: 9 };

export type TestSet = { scores: number[]; kinds: number[] };
export const testsetOf = (rt: RT) => (rt as RT & { assets?: Record<string, unknown> }).assets?.['testset.json'] as TestSet | undefined;

/** the test set at a threshold: flagged = score ≥ thr. tp caught odd beats, fp false alarms, per kind flagged */
export function countAt(s: TestSet, thr: number) {
	let tp = 0, fp = 0, pos = 0;
	const per = KINDS.map(() => 0), of = KINDS.map(() => 0);
	s.scores.forEach((v, i) => {
		const k = s.kinds[i], odd = k > 0;
		of[k]++;
		if (odd) pos++;
		if (v >= thr) (odd ? tp++ : fp++), per[k]++;
	});
	const neg = s.scores.length - pos;
	return { tp, fp, pos, neg, per, of, recall: pos ? tp / pos : NaN, fpr: neg ? fp / neg : NaN, precision: tp + fp ? tp / (tp + fp) : NaN };
}
/** the task's goal (claims.test.ts: reachable on the slider, not met at THR0) */
export const GOAL = { tp: 150, fp: 15 };

/** the ROC curve of a test set (points after 0..n flagged beats, highest score first, ties by index), as the sweep runs */
export function rocOf(s: TestSet) {
	const order = s.scores.map((_, i) => i).sort((a, b) => s.scores[b] - s.scores[a] || a - b);
	const pos = s.kinds.filter((k) => k > 0).length, neg = s.kinds.length - pos;
	const pts: [number, number][] = [[0, 0]];
	let tp = 0, fp = 0;
	for (const i of order) {
		if (s.kinds[i] > 0) tp++;
		else fp++;
		pts.push([fp / neg, tp / pos]);
	}
	return pts;
}

/** the part of a beat a point lies in (the generator's average wave positions; drawn beats are named the same way) */
export function regionOf(i: number): I18n {
	return i < 20 ? I('flat start, before the P wave', 'P 波之前的平坦段') : i < 42 ? I('P wave', 'P 波') : i < 60 ? I('QRS spike', 'QRS 尖峰') : i < 76 ? I('ST segment, between the spike and the T wave', 'ST 段（尖峰和 T 波之间）') : i < 110 ? I('T wave', 'T 波') : I('flat end, after the T wave', 'T 波之后的平坦段');
}
export const REGIONS = [{ from: 20, to: 42, l: 'P' }, { from: 42, to: 60, l: 'QRS' }, { from: 60, to: 76, l: 'ST' }, { from: 76, to: 110, l: 'T' }];
