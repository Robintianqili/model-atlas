// Data references for the kind-B views (plane2d, curve1d, network, train-step, loss-curve, train-controls).
// A view prop such as `points: 'pre.xy'` or `grid: 'grid'` names where its numbers come from:
//   a tensor object      used as is (demos, lesson props)
//   'pre.<name>'         rt.pre.tensors[name] (what the input adapter produced: points, a series)
//   '<field>'            at.state[field] (derived fields included; they are recomputed here on the main thread)
//   'state.<field>'      the same, spelled out
//   '<field>:<key>'      one entry of a record-valued field ('params:net.0.weight', 'grads:net.2.bias')
// Pure TS (Node tests and view demos use it).
import type { AnyTensor, Tensor } from '../../types/core.ts';
import type { At } from '../../types/lesson.ts';
import type { LiveTape, Tape } from '../../types/process.ts';
import type { ExplainerRuntime } from '../../types/runtime.ts';

export type Ref = string | Tensor | AnyTensor | null | undefined;

const isTensor = (v: unknown): v is AnyTensor => !!v && typeof v === 'object' && Array.isArray((v as Tensor).shape) && ArrayBuffer.isView((v as Tensor).data);

/** The tensor a reference names, or undefined when it is missing (views then draw without that layer). */
export function resolveRef(ref: Ref, rt: Pick<ExplainerRuntime, 'pre'> | null | undefined, at: Pick<At, 'state'> | null | undefined): AnyTensor | undefined {
	if (!ref) return undefined;
	if (typeof ref !== 'string') return isTensor(ref) ? ref : undefined;
	if (ref.startsWith('pre.')) {
		const t = rt?.pre?.tensors?.[ref.slice(4)];
		return isTensor(t) ? t : undefined;
	}
	const name = ref.startsWith('state.') ? ref.slice(6) : ref;
	const state = (at?.state ?? null) as Record<string, unknown> | null;
	if (!state) return undefined;
	const colon = name.indexOf(':');
	if (colon > 0) {
		const rec = state[name.slice(0, colon)] as Record<string, unknown> | undefined;
		const t = rec?.[name.slice(colon + 1)];
		return isTensor(t) ? t : undefined;
	}
	let v: unknown;
	try {
		v = state[name];
	} catch {
		return undefined; // a derived field that cannot be computed for this state
	}
	return isTensor(v) ? v : undefined;
}

/** A number from the state (a scalar field) or the frame metrics. */
export function resolveNum(name: string, at: Pick<At, 'state' | 'frame'> | null | undefined): number {
	const s = at?.state as Record<string, unknown> | undefined;
	const v = s?.[name];
	if (typeof v === 'number') return v;
	const m = at?.frame?.metrics?.[name];
	return typeof m === 'number' ? m : NaN;
}

export function tapeOf(rt: Pick<ExplainerRuntime, 'sources'> | null | undefined, source: string): Tape | LiveTape | null {
	return (rt?.sources?.[source]?.tape as Tape | LiveTape | null | undefined) ?? null;
}

export function isLive(t: Tape | LiveTape | null | undefined): t is LiveTape {
	return !!t && typeof (t as LiveTape).subscribe === 'function' && typeof (t as LiveTape).control === 'function';
}

/** Rows of a [n, 2] tensor as [x, y] pairs (empty when the shape does not fit). */
export function pairs(t: AnyTensor | undefined): number[][] {
	if (!t || t.shape.length !== 2 || t.shape[1] < 2) return [];
	const out: number[][] = [];
	const w = t.shape[1];
	for (let i = 0; i < t.shape[0]; i++) out.push([t.data[i * w], t.data[i * w + 1]]);
	return out;
}

/** Parameter tensors of an MLP state in layer order: [{ key, weight [out, in], bias? }]. Keys follow PyTorch's
 *  state_dict names ('net.0.weight', 'net.0.bias', 'net.2.weight', …); a 2-D '*.weight' starts a layer. */
export function mlpLayers(params: Record<string, Tensor> | undefined): { prefix: string; weight: Tensor; bias?: Tensor }[] {
	if (!params) return [];
	const out: { prefix: string; weight: Tensor; bias?: Tensor }[] = [];
	for (const [k, t] of Object.entries(params)) {
		if (!k.endsWith('.weight') || t.shape.length !== 2) continue;
		const prefix = k.slice(0, -'.weight'.length);
		out.push({ prefix, weight: t, bias: params[`${prefix}.bias`] });
	}
	return out;
}

/** Layer widths of a chained MLP ([in, h1, …, out]), or null when the weights do not chain. */
export function mlpSizes(layers: { weight: Tensor }[]): number[] | null {
	if (!layers.length) return null;
	const sizes = [layers[0].weight.shape[1]];
	for (const l of layers) {
		if (l.weight.shape[1] !== sizes[sizes.length - 1]) return null;
		sizes.push(l.weight.shape[0]);
	}
	return sizes;
}

/** Downsample a series to at most `max` points, keeping the min and the max of every bucket (spikes stay visible). */
export function downsample(xs: ArrayLike<number>, ys: ArrayLike<number>, max = 600): { x: number[]; y: number[] } {
	const x: number[] = [];
	const y: number[] = [];
	const n = Math.min(xs.length, ys.length);
	const idx: number[] = [];
	for (let i = 0; i < n; i++) if (Number.isFinite(ys[i]) && Number.isFinite(xs[i])) idx.push(i);
	if (idx.length <= max) {
		for (const i of idx) {
			x.push(xs[i]);
			y.push(ys[i]);
		}
		return { x, y };
	}
	const per = Math.ceil(idx.length / Math.floor(max / 2));
	for (let b = 0; b < idx.length; b += per) {
		let lo = idx[b];
		let hi = idx[b];
		for (let j = b; j < Math.min(idx.length, b + per); j++) {
			const i = idx[j];
			if (ys[i] < ys[lo]) lo = i;
			if (ys[i] > ys[hi]) hi = i;
		}
		for (const i of lo <= hi ? [lo, hi] : [hi, lo]) {
			x.push(xs[i]);
			y.push(ys[i]);
		}
	}
	return { x, y };
}
