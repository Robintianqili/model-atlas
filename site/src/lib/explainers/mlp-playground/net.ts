// mlp-playground: plain float64 forward passes of the playground MLP from recorded parameters (no autograd), for the
// derived heatmaps, the accuracy metric, the lesson's numbers and the neuron-grid view: every number shown comes from
// the weights that ran. Layout as atlas/autograd/nn.ts MLP('net', …): net.0.weight [h1, 2], net.0.bias, net.2.weight …
import type { Tensor } from '../../atlas/types/index.ts';
import { pointsOf } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { fmt3, minus } from '../../atlas/core/format.ts';

/** 3 significant digits as plain decimals down to 0.00001 ("0.000415", not "4.15e−4": easier in a first lesson); a
 *  non-finite value stays NaN, so lint reports it */
export const dec = (v: number): string | number => (!Number.isFinite(v) ? NaN : v !== 0 && Math.abs(v) < 1e-3 && Math.abs(v) >= 1e-5 ? minus(v.toPrecision(3)) : fmt3(v));

export type Act = 'tanh' | 'relu';
export type Params = Record<string, Tensor>;
export type Layer = { w: Tensor; b?: Tensor };
/** the plane every view draws: x and y from −SPAN to SPAN (plane2d's default bounds); RES cells per side */
export const SPAN = 1.25;
export const RES = 40;

export function layersOf(params: Params): Layer[] {
	const out: Layer[] = [];
	for (let i = 0; params[`net.${i}.weight`]; i += 2) out.push({ w: params[`net.${i}.weight`], b: params[`net.${i}.bias`] });
	return out;
}

/** [2, h1, …, 2] */
export function sizesOf(params: Params): number[] {
	const ls = layersOf(params);
	return ls.length ? [ls[0].w.shape[1], ...ls.map((l) => l.w.shape[0])] : [];
}

/** Every layer's output for P rows of x [P·2]: hidden layers after the activation, the last one the two class scores. */
export function forwardAll(params: Params, act: Act, x: Float64Array, P: number): Float64Array[] {
	const ls = layersOf(params);
	const outs: Float64Array[] = [];
	let h = x;
	ls.forEach((l, li) => {
		const [O, I] = l.w.shape;
		const y = new Float64Array(P * O);
		for (let p = 0; p < P; p++)
			for (let o = 0; o < O; o++) {
				let s = l.b ? l.b.data[o] : 0;
				for (let i = 0; i < I; i++) s += h[p * I + i] * l.w.data[o * I + i];
				y[p * O + o] = li === ls.length - 1 ? s : act === 'relu' ? Math.max(0, s) : Math.tanh(s);
			}
		outs.push((h = y));
	});
	return outs;
}

/** P(class 1) from two scores per row (softmax). */
const p1Of = (z: Float64Array, P: number) => Float64Array.from({ length: P }, (_, p) => 1 / (1 + Math.exp(z[2 * p] - z[2 * p + 1])));

/** Cell centres of the RES × RES plane, row 0 at the top. */
export function planeXY(): Float64Array {
	const xy = new Float64Array(RES * RES * 2);
	for (let i = 0; i < RES * RES; i++) {
		xy[2 * i] = -SPAN + (((i % RES) + 0.5) / RES) * 2 * SPAN;
		xy[2 * i + 1] = SPAN - ((Math.floor(i / RES) + 0.5) / RES) * 2 * SPAN;
	}
	return xy;
}

/** Derived field `grids` [N, RES, RES]: every hidden neuron over the plane, layer by layer, then P(class 1) (`last`:
 *  only P(class 1), as [RES, RES]: the derived field `grid` plane2d draws). */
export function neuronGrids(params: Params, act: Act, last = false): Tensor {
	const P = RES * RES;
	const outs = forwardAll(params, act, planeXY(), P);
	if (!outs.length) return { shape: [0, RES, RES], data: new Float64Array(0) };
	const p1 = p1Of(outs[outs.length - 1], P);
	if (last) return { shape: [RES, RES], data: p1 };
	const parts: Float64Array[] = [];
	for (const h of outs.slice(0, -1)) {
		const O = h.length / P;
		for (let o = 0; o < O; o++) parts.push(Float64Array.from({ length: P }, (_, p) => h[p * O + o]));
	}
	parts.push(p1);
	const data = new Float64Array(parts.length * P);
	parts.forEach((g, k) => data.set(g, k * P));
	return { shape: [parts.length, RES, RES], data };
}

/** Each point's cross-entropy (−log of the share for its own class), P(class 1), their mean, the accuracy and the point
 *  with the largest loss. */
export function evalPoints(params: Params, act: Act, pts: Points2D) {
	const n = pts.n;
	const outs = forwardAll(params, act, pts.xy, n);
	const z = outs[outs.length - 1];
	const p1 = p1Of(z, n);
	const loss = new Float64Array(n);
	let sum = 0;
	let right = 0;
	let worst = 0;
	for (let i = 0; i < n; i++) {
		const m = Math.max(z[2 * i], z[2 * i + 1]); // log-sum-exp: exact for large scores too
		loss[i] = m + Math.log(Math.exp(z[2 * i] - m) + Math.exp(z[2 * i + 1] - m)) - z[2 * i + pts.label[i]];
		sum += loss[i];
		if ((p1[i] > 0.5 ? 1 : 0) === pts.label[i]) right++;
		if (loss[i] > loss[worst]) worst = i;
	}
	return { loss, p1, mean: n ? sum / n : NaN, accuracy: n ? right / n : NaN, worst };
}

// one Points2D per raw object (the worker and the views keep the same raw object)
const cache = new WeakMap<object, Points2D>();
/** The points a run trains on: the source's opt.data (a fixed set, e.g. the spiral) else the learner's points. */
export function dataOf(raw: unknown, opt?: { data?: unknown } | null): Points2D {
	const r = opt?.data ?? raw;
	if (!r || typeof r !== 'object') return pointsOf(r);
	let p = cache.get(r);
	if (!p) cache.set(r, (p = pointsOf(r)));
	return p;
}
