// Small trainers for the kind-B views: the view demos, the /dev/atlas-train/ performance page and the engine tests use
// them. Pure TS (they also run inside the performance page's workers). Each one reads the raw value of a kit adapter
// (points2d or series), so the same input object drives the adapter, the worker and the views.
//   classifierTrainer  MLP 2 → … → 2 on labelled points (cross-entropy); derived `grid` = P(class 1) on a res × res grid
//   flowTrainer        the flow-2d velocity MLP v(x, y, t) 3 → 64 → 64 → 64 → 2 (SiLU, Adam 1e-3, batch 256)
//   fitTrainer         MLP 1 → … → 1 fitting a series (MSE, time scaled to u ∈ [−1, 1]); derived `pred` = [200, 2] rows (u, ŷ)
// Derived fields are recomputed on the main thread from the recorded params (plain f64 loops, no autograd), so they
// show exactly the weights that ran.
import type { Tensor, TrainOpt, TrainerDef } from '../../types/index.ts';
import { input, type Var } from '../../autograd/var.ts';
import { crossEntropy, mse } from '../../autograd/loss.ts';
import { MLP } from '../../autograd/nn.ts';
import { sampleBatch, type Points2D } from '../../data/points2d.ts';
import { pointsOf, type PointsRaw } from '../../adapters/points2d/adapter.ts';
import { seriesOf, type SeriesRaw, type SeriesValue } from '../../adapters/series/adapter.ts';

export type Act = 'tanh' | 'relu' | 'sigmoid' | 'silu';
export interface Bounds2D {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}
export const BOUNDS: Bounds2D = { x0: -1.25, x1: 1.25, y0: -1.25, y1: 1.25 };

// ------------------------------------------------------------------ cached raw → data (a worker keeps one raw object)
const pointsCache = new WeakMap<object, Points2D>();
export function cachedPoints(raw: unknown): Points2D {
	if (raw && typeof raw === 'object') {
		let p = pointsCache.get(raw);
		if (!p) pointsCache.set(raw, (p = pointsOf(raw)));
		return p;
	}
	return pointsOf(raw);
}
const seriesCache = new WeakMap<object, SeriesValue>();
export function cachedSeries(raw: unknown): SeriesValue {
	if (raw && typeof raw === 'object') {
		let s = seriesCache.get(raw);
		if (!s) seriesCache.set(raw, (s = seriesOf(raw)));
		return s;
	}
	return seriesOf(raw);
}

// ------------------------------------------------------------------ plain forward pass from recorded params
function act(v: number, a: Act): number {
	switch (a) {
		case 'tanh':
			return Math.tanh(v);
		case 'relu':
			return v > 0 ? v : 0;
		case 'sigmoid':
			return 1 / (1 + Math.exp(-v));
		default:
			return v / (1 + Math.exp(-v));
	}
}

/** MLP forward with PyTorch layout (weight [out, in]) on P rows of x [P, in]; activation between layers, none at the end. */
export function mlpApply(params: Record<string, Tensor>, prefix: string, a: Act, x: Float64Array, P: number): { out: Float64Array; width: number } {
	const layers: { w: Tensor; b?: Tensor }[] = [];
	for (let i = 0; ; i += 2) {
		const w = params[`${prefix}.${i}.weight`];
		if (!w) break;
		layers.push({ w, b: params[`${prefix}.${i}.bias`] });
	}
	if (!layers.length) throw new Error(`mlpApply: no '${prefix}.0.weight' in params`);
	let h = x;
	let width = layers[0].w.shape[1];
	layers.forEach((l, li) => {
		const [O, I] = l.w.shape;
		if (I !== width) throw new Error(`mlpApply: layer ${li} takes ${I} inputs, got ${width}`);
		const out = new Float64Array(P * O);
		const W = l.w.data;
		const B = l.b?.data;
		for (let p = 0; p < P; p++) {
			for (let o = 0; o < O; o++) {
				let s = B ? B[o] : 0;
				for (let i = 0; i < I; i++) s += h[p * I + i] * W[o * I + i];
				out[p * O + o] = li + 1 < layers.length ? act(s, a) : s;
			}
		}
		h = out;
		width = O;
	});
	return { out: h, width };
}

/** Cell centres of a res × res grid, row 0 at the top (y1). */
export function gridXY(b: Bounds2D, res: number): Float64Array {
	const xy = new Float64Array(res * res * 2);
	for (let r = 0; r < res; r++) {
		const y = b.y1 - ((r + 0.5) / res) * (b.y1 - b.y0);
		for (let c = 0; c < res; c++) {
			xy[(r * res + c) * 2] = b.x0 + ((c + 0.5) / res) * (b.x1 - b.x0);
			xy[(r * res + c) * 2 + 1] = y;
		}
	}
	return xy;
}

/** P(class k) on a res × res grid ([res, res], row 0 at the top) from a classifier's params. */
export function probGrid(params: Record<string, Tensor>, prefix: string, a: Act, b: Bounds2D, res: number, k = 1): Tensor {
	const P = res * res;
	const { out, width } = mlpApply(params, prefix, a, gridXY(b, res), P);
	const g = new Float64Array(P);
	for (let p = 0; p < P; p++) {
		let m = -Infinity;
		for (let c = 0; c < width; c++) m = Math.max(m, out[p * width + c]);
		let z = 0;
		for (let c = 0; c < width; c++) z += Math.exp(out[p * width + c] - m);
		g[p] = Math.exp(out[p * width + k] - m) / z;
	}
	return { shape: [res, res], data: g };
}

/** The flow velocity field at time t on a res × res grid: [res·res, 4] rows (x, y, vx, vy). */
export function velocityField(params: Record<string, Tensor>, t: number, b: Bounds2D, res: number, prefix = 'v'): Tensor {
	const P = res * res;
	const xy = gridXY(b, res);
	const x = new Float64Array(P * 3);
	for (let p = 0; p < P; p++) {
		x[3 * p] = xy[2 * p];
		x[3 * p + 1] = xy[2 * p + 1];
		x[3 * p + 2] = t;
	}
	const { out } = mlpApply(params, prefix, 'silu', x, P);
	const v = new Float64Array(P * 4);
	for (let p = 0; p < P; p++) {
		v[4 * p] = xy[2 * p];
		v[4 * p + 1] = xy[2 * p + 1];
		v[4 * p + 2] = out[2 * p];
		v[4 * p + 3] = out[2 * p + 1];
	}
	return { shape: [P, 4], data: v };
}

/** Euler sampling with the flow field: n particles from N(0, I) (seeded mulberry32 noise passed in) over K steps.
 *  Returns paths [n, K + 1, 2]. */
export function eulerPaths(params: Record<string, Tensor>, start: Float64Array, K: number, prefix = 'v'): Tensor {
	const n = start.length / 2;
	const out = new Float64Array(n * (K + 1) * 2);
	const cur = start.slice();
	for (let i = 0; i < n; i++) {
		out[i * (K + 1) * 2] = cur[2 * i];
		out[i * (K + 1) * 2 + 1] = cur[2 * i + 1];
	}
	for (let k = 0; k < K; k++) {
		const t = k / K;
		const x = new Float64Array(n * 3);
		for (let i = 0; i < n; i++) {
			x[3 * i] = cur[2 * i];
			x[3 * i + 1] = cur[2 * i + 1];
			x[3 * i + 2] = t;
		}
		const { out: v } = mlpApply(params, prefix, 'silu', x, n);
		for (let i = 0; i < n; i++) {
			cur[2 * i] += v[2 * i] / K;
			cur[2 * i + 1] += v[2 * i + 1] / K;
			const o = (i * (K + 1) + k + 1) * 2;
			out[o] = cur[2 * i];
			out[o + 1] = cur[2 * i + 1];
		}
	}
	return { shape: [n, K + 1, 2], data: out };
}

// ------------------------------------------------------------------ trainers
export type ClsBatch = { x: Var; y: Int32Array };
export type ClsOpt = TrainOpt & { hidden?: number[]; act?: Act; res?: number };

/** A classifier MLP on labelled points (the MLP playground's core). Derived `grid`: P(class 1) on res × res. */
export function classifierTrainer(o: { id?: string; hidden?: number[]; act?: Act; res?: number; defaults?: Partial<ClsOpt> } = {}): TrainerDef<PointsRaw, MLP, ClsBatch, ClsOpt> {
	const hidden = o.hidden ?? [8, 8];
	const a: Act = o.act ?? 'tanh';
	const res = o.res ?? 40;
	return {
		id: o.id ?? 'cls',
		title: { en: 'A small classifier on 2-D points', zh: '二维点上的小分类器' },
		defaults: { seed: 1, lr: 0.03, batch: 32, steps: 600, snapshotEvery: 10, optimizer: 'adam', ...(o.defaults ?? {}) } as ClsOpt,
		model: (rng) => new MLP('net', [2, ...hidden, 2], a, rng),
		batch(raw, _step, rng, opt) {
			const pts = cachedPoints(raw);
			if (pts.n === 0) return { x: input({ shape: [0, 2], data: new Float64Array(0) }), y: new Int32Array(0) };
			const b = sampleBatch(pts, Math.max(1, Math.min(opt.batch, 4096)), rng);
			return { x: input(b.x), y: b.label };
		},
		loss(m, b, rec) {
			return { loss: crossEntropy(m.forward(b.x, rec), b.y) };
		},
		derived: {
			grid: {
				kind: 'tensor',
				label: { en: 'Share of class 1 everywhere', zh: '各处属于第 1 类的比例' },
				shape: [res, res],
				scale: 'sequential',
				role: 'output',
				derive: (s: { params?: Record<string, Tensor> }) => (s.params ? probGrid(s.params, 'net', a, BOUNDS, res) : { shape: [0, 0], data: new Float64Array(0) })
			}
		},
		metrics(m, raw): Record<string, number> {
			const pts = cachedPoints(raw);
			if (pts.n === 0) return {};
			const named: Record<string, Tensor> = {};
			for (const [k, v] of Object.entries(m.named())) named[k] = v.value;
			const { out } = mlpApply(named, 'net', a, pts.xy, pts.n);
			let right = 0;
			for (let i = 0; i < pts.n; i++) if ((out[2 * i + 1] > out[2 * i] ? 1 : 0) === pts.label[i]) right++;
			return { accuracy: right / pts.n };
		}
	};
}

export type FlowBatch = { inp: Var; target: Var };

/** The flow-2d velocity MLP (W1-B1's speed benchmark, here on a points2d raw value). */
export function flowTrainer(o: { id?: string; defaults?: Partial<TrainOpt> } = {}): TrainerDef<PointsRaw, MLP, FlowBatch, TrainOpt> {
	return {
		id: o.id ?? 'flow',
		title: { en: 'Flow matching (2-D)', zh: '流匹配（二维）' },
		defaults: { seed: 1, lr: 1e-3, batch: 256, steps: 3000, snapshotEvery: 50, optimizer: 'adam', ...(o.defaults ?? {}) },
		model: (rng) => new MLP('v', [3, 64, 64, 64, 2], 'silu', rng),
		batch(raw, _step, rng, opt) {
			const data = cachedPoints(raw);
			const B = opt.batch;
			const inp = new Float64Array(B * 3);
			const target = new Float64Array(B * 2);
			for (let i = 0; i < B; i++) {
				const j = data.n ? rng.int(data.n) : 0;
				const x1 = data.n ? data.xy[2 * j] : 0;
				const y1 = data.n ? data.xy[2 * j + 1] : 0;
				const x0 = rng.normal();
				const y0 = rng.normal();
				const t = rng.next();
				inp[3 * i] = (1 - t) * x0 + t * x1;
				inp[3 * i + 1] = (1 - t) * y0 + t * y1;
				inp[3 * i + 2] = t;
				target[2 * i] = x1 - x0;
				target[2 * i + 1] = y1 - y0;
			}
			return { inp: input({ shape: [B, 3], data: inp }), target: input({ shape: [B, 2], data: target }) };
		},
		loss(m, b, rec) {
			return { loss: mse(m.forward(b.inp, rec), b.target) };
		}
	};
}

export type FitBatch = { t: Var; y: Var };

/** An MLP that fits a series y(t), with time scaled to u ∈ [−1, 1] over the series. Derived `pred`: [200, 2] rows (u, ŷ)
 *  on an even grid (curve1d draws it over the data's time range with `domain: 'unit'`). */
export function fitTrainer(o: { id?: string; hidden?: number[]; act?: Act; defaults?: Partial<TrainOpt> } = {}): TrainerDef<SeriesRaw, MLP, FitBatch, TrainOpt> {
	const hidden = o.hidden ?? [16, 16];
	const a: Act = o.act ?? 'tanh';
	const scaled = (s: SeriesValue) => {
		const n = s.t.length;
		const t0 = n ? s.t[0] : 0;
		const t1 = n ? s.t[n - 1] : 1;
		const span = t1 - t0 || 1;
		return Float64Array.from(s.t, (t) => ((t - t0) / span) * 2 - 1);
	};
	return {
		id: o.id ?? 'fit',
		title: { en: 'Fit a curve', zh: '拟合一条曲线' },
		defaults: { seed: 1, lr: 0.01, batch: 32, steps: 1500, snapshotEvery: 25, optimizer: 'adam', ...(o.defaults ?? {}) },
		model: (rng) => new MLP('net', [1, ...hidden, 1], a, rng),
		batch(raw, _step, rng, opt) {
			const s = cachedSeries(raw);
			const ts = scaled(s);
			const n = s.y.length;
			const B = Math.max(1, Math.min(opt.batch, 1024));
			const t = new Float64Array(n ? B : 0);
			const y = new Float64Array(n ? B : 0);
			for (let i = 0; i < t.length; i++) {
				const j = rng.int(n);
				t[i] = ts[j];
				y[i] = s.y[j];
			}
			return { t: input({ shape: [t.length, 1], data: t }), y: input({ shape: [y.length, 1], data: y }) };
		},
		loss(m, b, rec) {
			return { loss: mse(m.forward(b.t, rec), b.y) };
		},
		derived: {
			pred: {
				kind: 'tensor',
				label: { en: 'The network’s curve', zh: '网络画出的曲线' },
				role: 'output',
				derive: (s: { params?: Record<string, Tensor> }) => {
					if (!s.params) return { shape: [0, 2], data: new Float64Array(0) };
					const N = 200;
					const u = Float64Array.from({ length: N }, (_, i) => -1 + (2 * i) / (N - 1));
					const y = mlpApply(s.params, 'net', a, u, N).out;
					const out = new Float64Array(N * 2);
					for (let i = 0; i < N; i++) {
						out[2 * i] = u[i];
						out[2 * i + 1] = y[i];
					}
					return { shape: [N, 2], data: out };
				}
			}
		}
	};
}
