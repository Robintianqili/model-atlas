// score-2d: pure float64 helpers shared by the trainer, the Langevin process, the lesson and the view.
//   SIGMAS            the noise levels: 1.28 halved five times (the annealing schedule; training draws σ log-uniformly
//                     between its ends)
//   dataOf(raw)       the learner's points (raw points2d input), or { xy: number[] } (a process input, the parity input)
//   drawPair(p, rng)  one training example, drawn in this order: a data point (int), u (next) for σ, noise ε (normal × 2)
//   denoise(...)      the network d(x, y, log σ) on many points from recorded params: the autograd forward's arithmetic
//                     (kernels.linear, then SiLU as x / (1 + e^−x)), so it equals the trained network bit for bit.
//                     Its target is −ε; the score is d / σ
//   exact(...)        the true score and density of "your points blurred by σ" (a bell of width σ on every point)
//   chains(...)       Langevin chains, one step per yield: x ← x + (α/2)·s(x, σ) + noise·√α·z (the process and the view)
import type { Rng, Tensor } from '../../atlas/types/index.ts';
import { linear } from '../../atlas/autograd/kernels.ts';
import { pointsOf } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';

export type Params = Record<string, Tensor>;
/** d: (x, y, log σ) -> 64 -> 64 -> 2, SiLU between: MLP('net', SIZES, 'silu') */
export const SIZES = [3, 64, 64, 2];
export const SIGMAS = [1.28, 0.64, 0.32, 0.16, 0.08, 0.04];
export const SMAX = SIGMAS[0];
export const SMIN = SIGMAS[SIGMAS.length - 1];
/** the σ of the lesson's density, score and single-level Langevin steps */
export const SHOW = 0.16;
/** the plane the views draw: −SPAN..SPAN; Langevin chains start uniformly inside it */
export const SPAN = 2.5;
export const BATCH = 256;

const cache = new WeakMap<object, Points2D>();
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

export type Pair = { j: number; s: number; ex: number; ey: number; x: number; y: number };
export function drawPair(p: Points2D, rng: Rng): Pair {
	const j = rng.int(p.n);
	const s = SMAX * (SMIN / SMAX) ** rng.next();
	const ex = rng.normal();
	const ey = rng.normal();
	return { j, s, ex, ey, x: p.xy[2 * j], y: p.xy[2 * j + 1] };
}

/** d(x, y, log σ) for n points (xy [n·2]) at one σ or one σ per point; returns [n·2] */
export function denoise(params: Params, xy: ArrayLike<number>, s: number | ArrayLike<number>, n: number): Float64Array {
	let h: Float64Array = new Float64Array(n * 3);
	for (let i = 0; i < n; i++) (h[3 * i] = xy[2 * i]), (h[3 * i + 1] = xy[2 * i + 1]), (h[3 * i + 2] = Math.log(typeof s === 'number' ? s : s[i]));
	for (let l = 0; l + 1 < SIZES.length; l++) {
		const w = params[`net.${2 * l}.weight`];
		const b = params[`net.${2 * l}.bias`];
		const y = linear(h, Float64Array.from(w.data), Float64Array.from(b.data), n, w.shape[1], w.shape[0]);
		if (l + 2 < SIZES.length) for (let i = 0; i < y.length; i++) y[i] = y[i] / (1 + Math.exp(-y[i]));
		h = y;
	}
	return h;
}
/** the learned score s = d / σ */
export function score(params: Params, xy: ArrayLike<number>, s: number, n: number): Float64Array {
	const d = denoise(params, xy, s, n);
	for (let i = 0; i < d.length; i++) d[i] /= s;
	return d;
}

/** The density p(x) of your points blurred by σ (the mean of a bell of width σ on every point), its score
 *  ∇log p = Σ w_j (x_j − x) / σ² with w_j ∝ exp(−|x − x_j|² / 2σ²), the largest weight and its point */
export function exact(data: Points2D, x: number, y: number, s: number) {
	let m = Infinity;
	for (let j = 0; j < data.n; j++) m = Math.min(m, (x - data.xy[2 * j]) ** 2 + (y - data.xy[2 * j + 1]) ** 2);
	let [z, sx, sy, wmax, jmax] = [0, 0, 0, 0, 0];
	for (let j = 0; j < data.n; j++) {
		const [dx, dy] = [data.xy[2 * j] - x, data.xy[2 * j + 1] - y];
		const w = Math.exp(-(dx * dx + dy * dy - m) / (2 * s * s));
		(z += w), (sx += w * dx), (sy += w * dy);
		if (w > wmax) (wmax = w), (jmax = j);
	}
	const density = (z * Math.exp(-m / (2 * s * s))) / (data.n * 2 * Math.PI * s * s);
	return { density, sx: sx / z / (s * s), sy: sy / z / (s * s), w: wmax / z, j: jmax };
}

/** the same density without the points farther than 4σ (each would add less than e^−8 of its bell's peak): fast
 *  enough for a heat map, and for finding the most crowded point, on the main thread */
export function densityNear(data: Points2D, x: number, y: number, s: number): number {
	const cut = 16 * s * s;
	let z = 0;
	for (let j = 0; j < data.n; j++) {
		const d2 = (x - data.xy[2 * j]) ** 2 + (y - data.xy[2 * j + 1]) ** 2;
		if (d2 < cut) z += Math.exp(-d2 / (2 * s * s));
	}
	return z / (data.n * 2 * Math.PI * s * s);
}
/** the probe point of the density and score steps: 0.3 to the right of your rightmost point, at the height of their mean */
export function probeOf(data: Points2D): [number, number] {
	let [mx, my] = [-Infinity, 0];
	for (let j = 0; j < data.n; j++) (mx = Math.max(mx, data.xy[2 * j])), (my += data.xy[2 * j + 1] / data.n);
	return [mx + 0.3, my];
}
/** a G × G grid of places over the plane, row-major from the bottom left */
export function grid(G: number, span: number): Float64Array {
	const xy = new Float64Array(2 * G * G);
	for (let i = 0; i < G * G; i++) (xy[2 * i] = -span + ((i % G) + 0.5) * ((2 * span) / G)), (xy[2 * i + 1] = -span + (Math.floor(i / G) + 0.5) * ((2 * span) / G));
	return xy;
}
/** the learned score against the true one at σ on a 30 × 30 grid, where no noisy copy at σ lands (more than 3σ from
 *  every point): the share of the plane, the mean angle between the arrows (degrees), how many times shorter the
 *  network's arrow is (median); and both lengths 1 to the right of the rightmost point (process 'limits') */
export function limitsOf(params: Params, data: Points2D, s: number) {
	const G = 30;
	const g = grid(G, SPAN);
	const net = score(params, g, s, G * G);
	const [angles, ratios] = [[] as number[], [] as number[]];
	for (let i = 0; i < G * G; i++) {
		if (meanNearest([g[2 * i], g[2 * i + 1]], 1, data) <= 3 * s) continue;
		const e = exact(data, g[2 * i], g[2 * i + 1], s);
		const [a, b] = [Math.hypot(net[2 * i], net[2 * i + 1]), Math.hypot(e.sx, e.sy)];
		angles.push((Math.acos(Math.max(-1, Math.min(1, (net[2 * i] * e.sx + net[2 * i + 1] * e.sy) / (a * b)))) * 180) / Math.PI);
		ratios.push(b / a);
	}
	ratios.sort((a, b) => a - b);
	const [px, py] = probeOf(data);
	const e1 = exact(data, px + 0.7, py, s);
	const n1 = score(params, [px + 0.7, py], s, 1);
	return { share: angles.length / (G * G), angle: angles.length ? angles.reduce((a, b) => a + b, 0) / angles.length : NaN, times: ratios[Math.floor(ratios.length / 2)] ?? NaN, exact1: Math.hypot(e1.sx, e1.sy), net1: Math.hypot(n1[0], n1[1]) };
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
/** the share of your points with a sample within r: how much of the shape the samples cover */
export function coverage(xy: ArrayLike<number>, n: number, data: Points2D, r = 0.1): number {
	if (!n || !data.n) return NaN;
	let hit = 0;
	for (let j = 0; j < data.n; j++)
		for (let i = 0; i < n; i++)
			if ((xy[2 * i] - data.xy[2 * j]) ** 2 + (xy[2 * i + 1] - data.xy[2 * j + 1]) ** 2 < r * r) {
				hit++;
				break;
			}
	return hit / data.n;
}

export type LangOpt = { sigmas: number[]; T: number; alpha: number; noise: number; chains: number };
/** chain 0's numbers for one step: where it was, the score there, the noise draw */
export type Move = { k: number; level: number; s: number; a: number; x: Float64Array; c0: { px: number; py: number; sx: number; sy: number; zx: number; zy: number } };
/** Langevin chains from uniform starts in ±SPAN (2 × chains draws of next()). Each step draws 2 × chains normals, then
 *  moves every chain: x ← x + (α/2)·s(x, σ) + noise·√α·z, with α = opt.alpha·(σ / σ_min)² at each level (the step
 *  shrinks with σ, as in annealed Langevin). Yields the starts (k −1), then the positions after every step. */
export function* chains(params: Params, opt: LangOpt, rng: Rng): Generator<Move> {
	const N = Math.max(1, Math.floor(opt.chains));
	const x = new Float64Array(2 * N);
	for (let i = 0; i < 2 * N; i++) x[i] = (2 * rng.next() - 1) * SPAN;
	yield { k: -1, level: 0, s: opt.sigmas[0], a: 0, x: x.slice(), c0: { px: NaN, py: NaN, sx: NaN, sy: NaN, zx: NaN, zy: NaN } };
	let k = 0;
	for (let l = 0; l < opt.sigmas.length; l++) {
		const s = opt.sigmas[l];
		const a = opt.alpha * (s / SMIN) ** 2;
		for (let t = 0; t < opt.T; t++, k++) {
			const z = new Float64Array(2 * N);
			for (let i = 0; i < 2 * N; i++) z[i] = rng.normal();
			const g = score(params, x, s, N);
			const c0 = { px: x[0], py: x[1], sx: g[0], sy: g[1], zx: z[0], zy: z[1] };
			for (let i = 0; i < 2 * N; i++) x[i] += (a / 2) * g[i] + opt.noise * Math.sqrt(a) * z[i];
			yield { k, level: l, s, a, x: x.slice(), c0 };
		}
	}
}
