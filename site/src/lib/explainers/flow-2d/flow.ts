// flow-2d: pure float64 helpers shared by the trainer, the processes, the adapter, the lesson and the view.
//   dataOf(raw)       the points to learn: the learner's raw points2d input, or { xy: number[] } (the parity check's input)
//   drawPair(p, rng)  one training pair, drawn in this order: a data point x1 (int), noise x0 = (normal, normal), t (next)
//   velocity(...)     the network v(x, y, t) on many points from recorded params: the autograd forward's arithmetic
//                     (kernels.linear, then SiLU as x / (1 + e^−x)), so it equals the trained network bit for bit
//   meanNearest(...)  the mean distance from points to their nearest data point (how close samples land)
import type { Rng, Tensor } from '../../atlas/types/index.ts';
import { linear } from '../../atlas/autograd/kernels.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { pointsOf } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';

export type Params = Record<string, Tensor>;
/** v: (x, y, t) -> 64 -> 64 -> 64 -> 2 (the arrow), SiLU between: MLP('net', SIZES, 'silu') */
export const SIZES = [3, 64, 64, 64, 2];
/** the plane the views draw: −SPAN..SPAN (noise is N(0, 1) per coordinate, so most of it lies inside) */
export const SPAN = 2.5;
/** pairs per training step */
export const BATCH = 256;

const cache = new WeakMap<object, Points2D>();
export function dataOf(raw: unknown): Points2D {
	if (!raw || typeof raw !== 'object') return pointsOf(raw);
	let p = cache.get(raw);
	if (!p) {
		const xy = (raw as { xy?: unknown }).xy;
		const n = Array.isArray(xy) ? Math.floor(xy.length / 2) : 0;
		p = Array.isArray(xy) ? { name: 'xy', xy: Float64Array.from(xy.slice(0, 2 * n) as number[]), label: new Int32Array(n), n, classes: 1 } : pointsOf(raw);
		cache.set(raw, p);
	}
	return p;
}

export type Pair = { j: number; x0: number; y0: number; x1: number; y1: number; t: number };
export function drawPair(p: Points2D, rng: Rng): Pair {
	const j = rng.int(p.n);
	const x0 = rng.normal();
	const y0 = rng.normal();
	const t = rng.next();
	return { j, x0, y0, x1: p.xy[2 * j], y1: p.xy[2 * j + 1], t };
}
/** The pair the lesson follows: the first pair of the first batch (seed 1, the batch stream's first draws, as trainer
 *  'fit' draws them) whose line is at least 1 long and whose noise end lies inside the plane, else pair 0. */
const shown = new WeakMap<object, { i: number; pair: Pair }>();
export function showcase(raw: unknown, seed = 1): { i: number; pair: Pair } | null {
	const p = dataOf(raw);
	if (!p.n) return null;
	const hit = raw && typeof raw === 'object' && seed === 1 ? shown.get(raw) : undefined;
	if (hit) return hit;
	const rng = createRng(seed).fork('batch');
	const pairs = Array.from({ length: BATCH }, () => drawPair(p, rng));
	const i = Math.max(0, pairs.findIndex((q) => Math.hypot(q.x1 - q.x0, q.y1 - q.y0) >= 1 && Math.max(Math.abs(q.x0), Math.abs(q.y0)) < 0.9 * SPAN));
	const out = { i, pair: pairs[i] };
	if (raw && typeof raw === 'object' && seed === 1) shown.set(raw, out);
	return out;
}
/** the point a fraction t along the straight line from a (noise) to b (data) */
export const along = (a: number, b: number, t: number) => (1 - t) * a + t * b;

/** v(x, y, t) for n points (xy row-major [n·2]) at one time or one time per point; returns [n·2] */
export function velocity(params: Params, xy: ArrayLike<number>, t: number | ArrayLike<number>, n: number): Float64Array {
	let h: Float64Array = new Float64Array(n * 3);
	for (let i = 0; i < n; i++) {
		h[3 * i] = xy[2 * i];
		h[3 * i + 1] = xy[2 * i + 1];
		h[3 * i + 2] = typeof t === 'number' ? t : t[i];
	}
	for (let l = 0; l + 1 < SIZES.length; l++) {
		const w = params[`net.${2 * l}.weight`];
		const b = params[`net.${2 * l}.bias`];
		const y = linear(h, Float64Array.from(w.data), Float64Array.from(b.data), n, w.shape[1], w.shape[0]);
		if (l + 2 < SIZES.length) for (let i = 0; i < y.length; i++) y[i] = y[i] / (1 + Math.exp(-y[i]));
		h = y;
	}
	return h;
}

/** mean over n points (xy [n·2]) of the distance to the nearest data point */
export function meanNearest(xy: ArrayLike<number>, n: number, data: Points2D): number {
	if (!n || !data.n) return NaN;
	let sum = 0;
	for (let i = 0; i < n; i++) {
		let best = Infinity;
		for (let j = 0; j < data.n; j++) {
			const dx = xy[2 * i] - data.xy[2 * j];
			const dy = xy[2 * i + 1] - data.xy[2 * j + 1];
			const d = dx * dx + dy * dy;
			if (d < best) best = d;
		}
		sum += Math.sqrt(best);
	}
	return sum / n;
}
