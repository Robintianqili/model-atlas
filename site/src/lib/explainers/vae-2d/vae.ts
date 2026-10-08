// vae-2d helpers (pure TS, shared by the adapter, the views, the lesson and the claim tests):
//   the learner's raw input carries three extras besides the drawing (drawing28's { strokes } / { b64 } / { ink }):
//     draw    which ε draw the model uses (0, 1, 2, …: the "draw another ε" button), ε = 2 normal draws of
//             createRng(EPS_SEED).fork('eps<k>')
//     probe   the point the learner dragged on the map, [z1, z2]
//     target  the class the interpolation walks to
//   numbers from the trace of the learner's drawing (μ, log σ², σ, ε, z, the redrawing) and the loss of this drawing;
//   decode(model, z): the decoder alone (the spec's layers from dec1 on, on the same weights and ops as the trace);
//   the map (latent.json: μ of test drawings by class, precomputed by pipeline/vae-2d/stats.py) and its neighbours.
import type { ExplainerRuntime, GraphModel, I18n, Tensor } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { drawingInput } from '../../atlas/adapters/drawing28/adapter.ts';

type RT = ExplainerRuntime;
export type XY = [number, number];
export interface Extra { draw?: number; probe?: XY; target?: string }
export interface Latent { classes: string[]; x: number[]; y: number[]; c: number[]; centres: XY[]; presets: Record<string, XY> }

export const EPS_SEED = 2026;
export const K = 10; // neighbours on the map
export const RANGE = 3; // the map shows −3 … 3 on both axes

export const extraOf = (raw: unknown): Extra => (raw && typeof raw === 'object' && !Array.isArray(raw) && !ArrayBuffer.isView(raw) ? (raw as Extra) : {});

export function epsOf(raw: unknown): XY {
	const r = createRng(EPS_SEED).fork(`eps${extraOf(raw).draw ?? 0}`);
	return [r.normal(), r.normal()];
}

/** the learner's raw input with some extras changed (the drawing itself is kept as it is) */
export function withExtra(raw: unknown, extra: Extra): Record<string, unknown> {
	if (raw === null || raw === undefined) return { strokes: [], ...extra };
	if (ArrayBuffer.isView(raw) || (Array.isArray(raw) && typeof raw[0] === 'number')) return { ink: raw, ...extra };
	if (Array.isArray(raw)) return { strokes: raw, ...extra };
	return { ...(raw as Record<string, unknown>), ...extra };
}

export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const two = (t: Tensor | undefined): XY => (t ? [t.data[0], t.data[1]] : [NaN, NaN]);

/** μ, log σ², σ, ε and z of the learner's drawing, from the trace */
export function codeOf(rt: RT) {
	return { mu: two(T(rt, 'mu')), logvar: two(T(rt, 'logvar')), sigma: two(T(rt, 'z#sigma')), eps: two(T(rt, 'eps')), z: two(T(rt, 'z')) };
}

const softplus = (s: number) => Math.max(s, 0) + Math.log1p(Math.exp(-Math.abs(s)));

/** the training loss of this drawing: binary cross-entropy summed over the pixels (per pixel too) + KL */
export function lossOf(rt: RT) {
	const x = T(rt, 'input'), s = T(rt, 'img');
	const { mu, logvar } = codeOf(rt);
	if (!x || !s) return null;
	const px = new Float64Array(x.data.length);
	let rec = 0;
	for (let i = 0; i < px.length; i++) rec += px[i] = softplus(s.data[i]) - s.data[i] * x.data[i];
	const klParts = mu.map((m, i) => 0.5 * (m * m + Math.exp(logvar[i]) - logvar[i] - 1));
	return { px, rec, klParts, kl: klParts[0] + klParts[1], total: rec + klParts[0] + klParts[1] };
}

/** does the −log σ² term of the KL, ½ Σ −log σ², outweigh the μ² term, ½ Σ μ²? (step 6's reason) */
export const klFromLog = (mu: XY, logvar: XY) => -0.5 * (logvar[0] + logvar[1]) >= 0.5 * (mu[0] ** 2 + mu[1] ** 2);

/** share of pixels between 0.2 and 0.8 among the pixels above 0.05 (stats.py grey_share) */
export function greyShare(a: ArrayLike<number>): number {
	let on = 0, grey = 0;
	for (let i = 0; i < a.length; i++) {
		if (a[i] > 0.05) on++;
		if (a[i] > 0.2 && a[i] < 0.8) grey++;
	}
	return on ? grey / on : NaN;
}

const DEC = new WeakMap<GraphModel, GraphModel>();
/** the decoder alone: the spec's layers from dec1 on, fed a declared input z */
export function decoderOf(m: GraphModel): GraphModel {
	let d = DEC.get(m);
	if (!d) {
		const layers = m.spec.layers.slice(m.spec.layers.findIndex((l) => l.id === 'dec1'));
		d = fromParts({ ...m.spec, layers, inputs: [{ name: 'z', adapter: 'latent', kind: 'vector', shape: [2], dtype: 'f32' }], outputs: ['out'] }, m.weights, m.ops);
		DEC.set(m, d);
	}
	return d;
}
export const decode = (m: GraphModel, z: XY) => decoderOf(m).forward({ z: { shape: [2], data: Float32Array.from(z) } }, { record: 'layers' }).t.out.data;

const MU = new WeakMap<GraphModel, Map<unknown, XY>>();
/** μ of any drawing (a preset's { b64 }, strokes, …) by the whole model, ε = 0 (memoized per raw object) */
export function muOf(m: GraphModel, raw: unknown): XY {
	let memo = MU.get(m);
	if (!memo) MU.set(m, (memo = new Map()));
	let v = memo.get(raw);
	if (!v) {
		const x = drawingInput(raw).input;
		v = two(m.forward({ input: { shape: [1, 28, 28], data: x }, eps: { shape: [2], data: new Float32Array(2) } }, { record: 'layers' }).t.mu);
		memo.set(raw, v);
	}
	return v;
}

export const latentOf = (rt: RT) => (rt as RT & { assets?: Record<string, unknown> }).assets?.['latent.json'] as Latent | undefined;

const NEAR = new WeakMap<Latent, Map<string, { counts: number[]; order: number[] }>>();
/** class counts among the K map points nearest to (x, y); `order` lists classes by count (ties: lower index) */
export function near(lat: Latent, p: XY, k = K) {
	let memo = NEAR.get(lat);
	if (!memo || memo.size > 500) NEAR.set(lat, (memo = new Map()));
	const key = `${p[0]},${p[1]},${k}`;
	const hit = memo.get(key);
	if (hit) return hit;
	const d = lat.x.map((x, i) => [(x - p[0]) ** 2 + (lat.y[i] - p[1]) ** 2, i]).sort((a, b) => a[0] - b[0]);
	const counts = lat.classes.map(() => 0);
	for (const [, i] of d.slice(0, k)) counts[lat.c[i]]++;
	const order = counts.map((_, c) => c).sort((a, b) => counts[b] - counts[a] || a - b);
	memo.set(key, { counts, order });
	return { counts, order };
}

/** a boundary: the two most common classes among the K nearest have at least 3 each */
export const atBoundary = (lat: Latent, p: XY) => {
	const { counts, order } = near(lat, p);
	return counts[order[1]] >= 3;
};

/** a zh name after a CJK character: a space before a Latin start ("9 幅 T 恤", "典型 T 恤") */
export const zhAfter = (s: string | undefined) => (s && /^[A-Za-z0-9]/.test(s) ? ` ${s}` : (s ?? ''));

/** bilingual name of kind c (spec.kinds; not `classes`: the kit would read the 784 pixels as class shares) */
export function className(rt: RT, c: number): I18n {
	const cl = (rt.spec?.kinds as { id: string; en: string; zh: string }[] | undefined)?.[c];
	return cl ? { en: cl.en, zh: cl.zh } : { en: '?', zh: '?' };
}
