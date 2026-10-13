// fno helpers (pure TS, shared by the view, the lesson and the claim tests): the learner's curve, the trace's layers by
// channel, the solver's answer, the error of the prediction, the same model run with another number of kept modes or on
// another curve (memoized), and the learned filter (the model's gain on single waves).
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { N, M, fdGain, heightOf, maxAbsDiff, solve, spectrum } from './heat.ts';
import { K, keepOf, valuesOf } from './adapters/series/adapter.ts';

type RT = ExplainerRuntime;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
/** column c of a field [n, C] */
export const col = (t: Tensor | undefined, c: number) => (t ? Array.from({ length: t.shape[0] }, (_, j) => t.data[j * t.shape[1] + c]) : []);
/** the wave heights of channel c of a spectrum [m, C, 2] */
export const heights = (t: Tensor | undefined, c: number) => (t ? Array.from({ length: t.shape[0] }, (_, k) => heightOf(t.data[(k * t.shape[1] + c) * 2], t.data[(k * t.shape[1] + c) * 2 + 1], k)) : []);
export const argmax = (a: ArrayLike<number>) => Array.from(a).reduce((b, v, i, x) => (v > x[b] ? i : b), 0);

const memo = new WeakMap<GraphModel, Map<string, Float32Array>>();
/** the model's prediction for a curve with `keep` modes kept (memoized per model) */
export function predict(m: GraphModel, u: ArrayLike<number>, keep = K): Float32Array {
	let c = memo.get(m);
	if (!c) memo.set(m, (c = new Map()));
	const key = `${keep}|${Array.from(u).join(',')}`;
	let p = c.get(key);
	if (!p) c.set(key, (p = Float32Array.from(m.forward({ input: { shape: [N, 1], data: Float32Array.from(u) }, keep: { shape: [1], data: Float32Array.of(keep) } }, { record: 'layers' }).t.proj.data)));
	return p;
}

/** the learner's curve, the solver's answer and the network's prediction (null before the trace has run) */
export function stateOf(rt: RT) {
	const u0 = valuesOf(rt.input), p = T(rt, 'proj');
	if (u0.length !== N || !p) return null;
	const sol = solve(u0), d = maxAbsDiff(p.data, sol);
	const range = (a: ArrayLike<number>) => Math.max(...Array.from(a)) - Math.min(...Array.from(a));
	return { u0, sol, pred: p.data, keep: keepOf(rt.input), err: d.max, at: d.at, spread0: range(u0), spreadT: range(sol) };
}
/** the largest error for every number of kept modes 1..K */
export const errByKeep = (m: GraphModel, u: ArrayLike<number>) => {
	const sol = solve(u);
	return Array.from({ length: K }, (_, i) => maxAbsDiff(predict(m, u, i + 1), sol).max);
};
/** the task's bar: the fewest kept modes whose largest error stays below GOAL */
export const GOAL = 0.02;
export const fewest = (errs: number[]) => errs.findIndex((e) => e < GOAL) + 1;

const gains = new WeakMap<GraphModel, number[]>();
/** the learned filter: how much of a single wave cos(2 pi n x / N) of height 0.5 the network keeps, n = 0..N/2 */
export function learnedGain(m: GraphModel): number[] {
	let g = gains.get(m);
	if (!g) {
		const zero = spectrum(predict(m, new Array(N).fill(0)));
		g = Array.from({ length: M }, (_, n) => {
			const s = spectrum(predict(m, Array.from({ length: N }, (_, j) => 0.5 * Math.cos((2 * Math.PI * n * j) / N))));
			return heightOf(s.re[n] - zero.re[n], s.im[n] - zero.im[n], n) / 0.5;
		});
		gains.set(m, g);
	}
	return g;
}
export const solverGain = () => Array.from({ length: M }, (_, n) => fdGain(n));
/** the lift's scale and shift per channel (each channel is a * u + b) */
export const liftOf = (rt: RT) => {
	const w = rt.model?.weights['lift.weight']?.data, b = rt.model?.weights['lift.bias']?.data;
	return w && b ? Array.from(w, (a, c) => ({ a, b: b[c] })) : [];
};
/** the channel the lesson follows: the one the lift scales most */
export const CH = (rt: RT) => argmax(liftOf(rt).map((l) => Math.abs(l.a)));
/** the mode the "mix" step zooms into: the tallest kept wave (mode 1 or above) of the followed channel, or mode 0 (the
 *  average) when no kept wave has any height (a flat curve, or only mode 0 kept) */
export const mixMode = (rt: RT) => {
	const h = heights(T(rt, 'f0.keep'), CH(rt)).slice(1);
	return h.length && Math.max(...h) > 1e-6 ? 1 + argmax(h) : 0;
};
/** the learner's curve made `s` times taller: the solver's and the network's answers (the learner's kept modes) */
export function taller(rt: RT, s: number) {
	const u0 = valuesOf(rt.input);
	if (!rt.model || u0.length !== N) return null;
	const u = Array.from(u0, (v) => v * s), sol = solve(u), pred = predict(rt.model, u, keepOf(rt.input));
	return { u, sol, pred, err: maxAbsDiff(pred, sol).max };
}
