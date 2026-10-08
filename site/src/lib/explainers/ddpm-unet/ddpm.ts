// ddpm-unet helpers (pure TS, shared by the adapter, the processes, the views, the lesson and the claim tests):
//   the noise schedule (the same float64 arithmetic as pipeline/ddpm-unet/model.py schedule(): np.linspace, cumprod);
//   the learner's raw input extras: `t` (the timestep of the traced noisy drawing) and `seed` (the sampling seed);
//   the U-Net alone (the spec's layers after q_sample, fed x_t and t), as the sampling processes run it;
//   reading the trace of the learner's drawing.
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import OPS from './ops.ts';

export const T = 100;
export const SIDE = 28;
export const N = SIDE * SIDE;
export const EPS_SEED = 2026;
export const T_TRACE = 10;
export const SEED0 = 0;
export const SEEDS = 12;
export const FILM = 20;
export const FEW = 10;
export const PROBE = 14 * SIDE + 14;
export interface Extra { t?: number; seed?: number }

/** β_t, α_t = 1 − β_t and ᾱ_t = α_1·…·α_t at index t − 1 (DDPM's linear schedule, 1e-4 … 0.02 scaled by 1000 / T) */
export function schedule(n = T) {
	const s = 1000 / n, lo = 1e-4 * s, hi = 0.02 * s, step = (hi - lo) / (n - 1);
	const beta = Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : i * step + lo));
	const alpha = beta.map((b) => 1 - b);
	const abar = new Float64Array(n);
	alpha.forEach((a, i) => (abar[i] = i ? abar[i - 1] * a : a));
	return { beta, alpha, abar };
}
export const SCHED = schedule();

/** the timesteps a sampling run visits: T, T − stride, …, down to stride (stride 1: every step) */
export const stepsOf = (stride: number) => Array.from({ length: Math.floor(T / stride) }, (_, i) => T - i * stride);

export const extraOf = (raw: unknown): Extra => (raw && typeof raw === 'object' && !Array.isArray(raw) && !ArrayBuffer.isView(raw) ? (raw as Extra) : {});
export const tOf = (raw: unknown) => Math.min(T, Math.max(1, Math.round(extraOf(raw).t ?? T_TRACE)));
export const seedOf = (raw: unknown) => Math.max(0, Math.round(extraOf(raw).seed ?? SEED0));

/** the noise ε of the traced drawing: N standard normal draws of createRng(EPS_SEED).fork('eps') */
export function epsNoise(): Float32Array {
	const r = createRng(EPS_SEED).fork('eps');
	return Float32Array.from({ length: N }, () => r.normal());
}

/** the learner's raw input with some extras changed (the drawing itself is kept as it is) */
export function withExtra(raw: unknown, extra: Extra): Record<string, unknown> {
	if (raw === null || raw === undefined) return { strokes: [], ...extra };
	if (ArrayBuffer.isView(raw) || (Array.isArray(raw) && typeof raw[0] === 'number')) return { ink: raw, ...extra };
	if (Array.isArray(raw)) return { strokes: raw, ...extra };
	return { ...(raw as Record<string, unknown>), ...extra };
}

const UNET = new WeakMap<object, GraphModel>();
/** the U-Net alone: the spec's layers after q_sample, fed the declared inputs xt [1, 28, 28] and t [1] */
export function unetOf(spec: unknown, weights: Record<string, Tensor>): GraphModel {
	let m = UNET.get(weights);
	if (!m) {
		const s = spec as { layers: { id: string }[]; inputs: { name: string }[] };
		const layers = s.layers.filter((l) => !['input', 'eps', 'xt'].includes(l.id));
		const inputs = [{ name: 'xt', adapter: 'drawing28', kind: 'drawing', shape: [1, SIDE, SIDE], dtype: 'f32' }, s.inputs.find((i) => i.name === 't')];
		m = fromParts({ ...s, layers, inputs, outputs: ['out'] }, weights, OPS);
		UNET.set(weights, m);
	}
	return m;
}
/** ε̂ = UNet(x_t, t) */
export const predictNoise = (m: GraphModel, x: ArrayLike<number>, t: number): Float32Array =>
	m.forward({ xt: { shape: [1, SIDE, SIDE], data: Float32Array.from(x) }, t: { shape: [1], data: Float32Array.of(t) } }, { record: 'layers' }).t.out.data as Float32Array;

/** what a sampling process needs: the spec and weights of the page's model (plain data, so it can go to the worker) */
export const modelInput = (rt: ExplainerRuntime) => (rt.model ? { seed: seedOf(rt.input), spec: rt.spec, weights: rt.model.weights } : null);

export const tr = (rt: ExplainerRuntime, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];

/** the mean of the squared differences of two pictures */
export function mse(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
	return a.length ? s / a.length : NaN;
}
