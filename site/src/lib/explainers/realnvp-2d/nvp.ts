// realnvp-2d: pure float64 helpers shared by the trainer, the processes, the lesson and the view.
//   LAYERS, HIDDEN   8 affine coupling layers; layer k keeps coordinate k % 2 (x for even k) and moves the other one
//   net(...)         layer k's small network on the kept coordinate u: 1 -> 24 -> 24 -> 2 with tanh between (MLP
//                    'c<k>'); s = tanh(first output) is the log of the stretch (so one layer stretches by e^−1 to e^1),
//                    t = the second output is the shift. kernels.linear and Math.tanh: the autograd forward's arithmetic,
//                    so it equals the trained network bit for bit
//   forward(...)     data -> noise, layer by layer: v <- v·e^s + t; the running log-determinant is Σ s
//   inverse(...)     noise -> data: v <- (v − t)·e^−s, the layers in reverse order
//   logp(...)        the exact log-likelihood log p(x) = log N(z; 0, I) + Σ s
import type { Tensor } from '../../atlas/types/index.ts';
import { linear } from '../../atlas/autograd/kernels.ts';
import { pointsOf } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';

export type Params = Record<string, Tensor>;
export const LAYERS = 8;
export const HIDDEN = 24;
export const SIZES = [1, HIDDEN, HIDDEN, 2];
export const LOG2PI = Math.log(2 * Math.PI);
/** the data plane the views draw (−DSPAN..DSPAN) and the noise plane (−ZSPAN..ZSPAN) */
export const DSPAN = 1.5;
export const ZSPAN = 3;
/** the side of the density grid over the data plane */
export const DG = 40;
/** grid lines: LINES lines each way, PER points on each */
export const LINES = 13;
export const PER = 49;
/** the coordinate layer k keeps (0: x, 1: y); the other one moves */
export const keepOf = (k: number) => k % 2;

const cache = new WeakMap<object, Points2D>();
/** the learner's points (raw points2d input), or { xy: number[] } (a process input, the parity input) */
export function dataOf(raw: unknown): Points2D {
	if (!raw || typeof raw !== 'object') return pointsOf(raw);
	let p = cache.get(raw);
	if (!p) {
		const xy = (raw as { xy?: unknown }).xy;
		const v = xy && typeof xy === 'object' && 'length' in xy ? Array.from(xy as ArrayLike<number>) : null;
		const n = v ? Math.floor(v.length / 2) : 0;
		p = v ? { name: 'xy', xy: Float64Array.from(v.slice(0, 2 * n)), label: new Int32Array(n), n, classes: 1 } : pointsOf(raw);
		cache.set(raw, p);
	}
	return p;
}

/** layer k's network on n kept coordinates u: the log-stretch s = tanh(out₀) and the shift t = out₁ */
export function net(params: Params, k: number, u: ArrayLike<number>, n: number): { s: Float64Array; t: Float64Array } {
	let h: Float64Array = Float64Array.from({ length: n }, (_, i) => u[i]);
	for (let l = 0; l + 1 < SIZES.length; l++) {
		const w = params[`c${k}.${2 * l}.weight`];
		const b = params[`c${k}.${2 * l}.bias`];
		const y = linear(h, Float64Array.from(w.data), Float64Array.from(b.data), n, SIZES[l], SIZES[l + 1]);
		if (l + 2 < SIZES.length) for (let i = 0; i < y.length; i++) y[i] = Math.tanh(y[i]);
		h = y;
	}
	const s = new Float64Array(n);
	const t = new Float64Array(n);
	for (let i = 0; i < n; i++) (s[i] = Math.tanh(h[2 * i])), (t[i] = h[2 * i + 1]);
	return { s, t };
}

/** one coupling layer in place on n points xy [n·2]: forward v <- v·e^s + t (adds s to ld), or its inverse */
export function couple(params: Params, k: number, xy: Float64Array, n: number, dir: 1 | -1, ld?: Float64Array) {
	const c = keepOf(k);
	const u = Float64Array.from({ length: n }, (_, i) => xy[2 * i + c]);
	const { s, t } = net(params, k, u, n);
	for (let i = 0; i < n; i++) {
		const j = 2 * i + 1 - c;
		xy[j] = dir > 0 ? xy[j] * Math.exp(s[i]) + t[i] : (xy[j] - t[i]) * Math.exp(-s[i]);
		if (ld) ld[i] += s[i];
	}
	return { s, t };
}

/** data -> noise through layers 0..upto−1: z and Σ s per point */
export function forward(params: Params, xy: ArrayLike<number>, n: number, upto = LAYERS) {
	const z = Float64Array.from({ length: 2 * n }, (_, i) => xy[i]);
	const ld = new Float64Array(n);
	for (let k = 0; k < upto; k++) couple(params, k, z, n, 1, ld);
	return { z, ld };
}
/** noise -> data: undo layers LAYERS−1 down to 0 */
export function inverse(params: Params, z: ArrayLike<number>, n: number) {
	const x = Float64Array.from({ length: 2 * n }, (_, i) => z[i]);
	for (let k = LAYERS - 1; k >= 0; k--) couple(params, k, x, n, -1);
	return x;
}
/** log N(z; 0, I) of one 2-D point */
export const logNormal = (zx: number, zy: number) => -0.5 * (zx * zx + zy * zy) - LOG2PI;
/** the exact log-likelihood of n points */
export function logp(params: Params, xy: ArrayLike<number>, n: number): Float64Array {
	const { z, ld } = forward(params, xy, n);
	return Float64Array.from({ length: n }, (_, i) => logNormal(z[2 * i], z[2 * i + 1]) + ld[i]);
}
/** the mean negative log-likelihood of the data (the loss on every point at once) */
export function nllOf(params: Params, d: Points2D): number {
	if (!d.n) return NaN;
	return -logp(params, d.xy, d.n).reduce((a, b) => a + b, 0) / d.n;
}

/** a G × G grid of places over ±span, row-major from the bottom left */
export function grid(G: number, span: number): Float64Array {
	const xy = new Float64Array(2 * G * G);
	for (let i = 0; i < G * G; i++) (xy[2 * i] = -span + ((i % G) + 0.5) * ((2 * span) / G)), (xy[2 * i + 1] = -span + (Math.floor(i / G) + 0.5) * ((2 * span) / G));
	return xy;
}
/** grid lines over ±span: LINES vertical lines (x fixed), then LINES horizontal ones, PER points each: [2·LINES·PER·2] */
export function lines(span: number): Float64Array {
	const out = new Float64Array(2 * LINES * PER * 2);
	for (let l = 0; l < 2 * LINES; l++) {
		const a = -span + ((l % LINES) * 2 * span) / (LINES - 1);
		for (let q = 0; q < PER; q++) {
			const b = -span + (q * 2 * span) / (PER - 1);
			const i = 2 * (l * PER + q);
			if (l < LINES) (out[i] = a), (out[i + 1] = b);
			else (out[i] = b), (out[i + 1] = a);
		}
	}
	return out;
}

/** mean over n points (xy [n·2]) of the distance to the nearest data point */
export function meanNearest(xy: ArrayLike<number>, n: number, data: Points2D): number {
	if (!n || !data.n) return NaN;
	let sum = 0;
	for (let i = 0; i < n; i++) {
		let best = Infinity;
		for (let j = 0; j < data.n; j++) best = Math.min(best, (xy[2 * i] - data.xy[2 * j]) ** 2 + (xy[2 * i + 1] - data.xy[2 * j + 1]) ** 2);
		sum += Math.sqrt(best);
	}
	return sum / n;
}
/** the share of n points farther than r from every data point */
export function farShare(xy: ArrayLike<number>, n: number, data: Points2D, r = 0.15): number {
	if (!n || !data.n) return NaN;
	let far = 0;
	for (let i = 0; i < n; i++) {
		let j = 0;
		while (j < data.n && (xy[2 * i] - data.xy[2 * j]) ** 2 + (xy[2 * i + 1] - data.xy[2 * j + 1]) ** 2 >= r * r) j++;
		if (j === data.n) far++;
	}
	return far / n;
}
/** the share of n points whose distance from (0, 0) lies between r0 and r1 (the gap between two rings) */
export function bandShare(xy: ArrayLike<number>, n: number, r0: number, r1: number): number {
	if (!n) return NaN;
	let k = 0;
	for (let i = 0; i < n; i++) {
		const r = Math.hypot(xy[2 * i], xy[2 * i + 1]);
		if (r > r0 && r < r1) k++;
	}
	return k / n;
}
/** the mean and standard deviation of each coordinate of n points */
export function momentsOf(xy: ArrayLike<number>, n: number) {
	let [mx, my, vx, vy] = [0, 0, 0, 0];
	for (let i = 0; i < n; i++) (mx += xy[2 * i] / n), (my += xy[2 * i + 1] / n);
	for (let i = 0; i < n; i++) (vx += (xy[2 * i] - mx) ** 2 / n), (vy += (xy[2 * i + 1] - my) ** 2 / n);
	return { mx, my, sx: Math.sqrt(vx), sy: Math.sqrt(vy) };
}
