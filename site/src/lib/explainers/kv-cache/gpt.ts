// kv-cache: TinyGPT's forward pass, written so that it runs with or without a KV cache (the kit's graph executor always
// runs every position). The architecture is static/models/tinygpt's (pre-LN GPT: learned positions, LayerNorm, causal
// multi-head attention, GELU (tanh) MLP, tied output head), read from its spec and weights (read only). Activations are
// stored as float32 (as the kit and PyTorch store them), sums run in float64.
//
// run(g, ids, cache, count) computes the tokens ids[cache.n ..] at their positions, appends their keys and values to the
// cache and returns the logits of the last position. Without a cache, a pass starts from an empty cache and computes every
// token again; with one, the cache already holds the earlier tokens and a pass computes one row. A row's arithmetic is the
// same in both cases (it reads the same keys and values), so the two modes give the same numbers bit for bit.
// count.macs adds up the multiply-adds of every matrix product as they run (LayerNorm, GELU and softmax are not counted).
import type { AtlasSpec, Tensor } from '../../atlas/types/index.ts';

export type Gpt = { spec: AtlasSpec; weights: Record<string, Tensor> };
export type Dims = { L: number; H: number; D: number; dh: number; F: number; V: number; C: number; eps: number };
export type Count = { macs: number };

export const EOT = 1023;
export const BYTES = 4; // float32

export function dimsOf(spec: AtlasSpec): Dims {
	const s = spec as unknown as Record<string, number>;
	return { L: s.nLayer, H: s.nHead, D: s.dModel, dh: s.dHead, F: s.dMLP, V: s.vocabSize, C: s.context, eps: s.lnEps ?? 1e-5 };
}

/** Bytes of a cache that holds n tokens: layers × heads × tokens × dHead × 2 (K and V) × bytes per number. */
export const cacheBytes = (d: Dims, n: number, bytes = BYTES) => d.L * d.H * n * d.dh * 2 * bytes;

/** The keys and values of every layer for the positions 0..n−1 ([C, D] per layer: all heads side by side). */
export class KVCache {
	k: Float32Array[];
	v: Float32Array[];
	n = 0;
	constructor(d: Dims) {
		this.k = Array.from({ length: d.L }, () => new Float32Array(d.C * d.D));
		this.v = Array.from({ length: d.L }, () => new Float32Array(d.C * d.D));
	}
	/** A cache holding the first n rows of another one (prefix reuse). */
	static prefixOf(d: Dims, src: KVCache, n: number): KVCache {
		const c = new KVCache(d);
		for (let l = 0; l < d.L; l++) {
			c.k[l].set(src.k[l].subarray(0, n * d.D));
			c.v[l].set(src.v[l].subarray(0, n * d.D));
		}
		c.n = n;
		return c;
	}
}

const f32 = Math.fround;

function layerNorm(x: Float32Array, T: number, D: number, g: Float32Array, b: Float32Array, eps: number): Float32Array {
	const out = new Float32Array(T * D);
	for (let r = 0; r < T; r++) {
		let m = 0;
		for (let i = 0; i < D; i++) m += x[r * D + i];
		m /= D;
		let v = 0;
		for (let i = 0; i < D; i++) v += (x[r * D + i] - m) ** 2;
		const inv = 1 / Math.sqrt(v / D + eps);
		for (let i = 0; i < D; i++) out[r * D + i] = (x[r * D + i] - m) * inv * g[i] + b[i];
	}
	return out;
}

/** out[r, o] = b[o] + Σ_i x[r, i] W[o, i] (PyTorch's Linear: W is [out, in]). */
function linear(x: Float32Array, T: number, nIn: number, W: Float32Array, b: Float32Array | null, nOut: number, count: Count, out: Float32Array = new Float32Array(T * nOut), row0 = 0): Float32Array {
	for (let r = 0; r < T; r++) {
		for (let o = 0; o < nOut; o++) {
			let s = b ? b[o] : 0;
			const wo = o * nIn;
			const xr = r * nIn;
			for (let i = 0; i < nIn; i++) s += x[xr + i] * W[wo + i];
			out[(row0 + r) * nOut + o] = s;
		}
	}
	count.macs += T * nIn * nOut;
	return out;
}

const GELU_C = Math.sqrt(2 / Math.PI);
const gelu = (v: number) => 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));

/** One pass. att, when given ([L, C, C]), receives each computed row's attention weights averaged over the heads. */
export function run(g: Gpt, ids: number[], cache: KVCache, count: Count, att?: Float32Array): Float32Array {
	const d = dimsOf(g.spec);
	const { L, H, D, dh, F, V, C, eps } = d;
	const W = (n: string) => g.weights[n].data as Float32Array;
	const from = cache.n;
	const T = ids.length - from;
	if (T <= 0 || ids.length > C) throw new Error(`kv-cache: a pass needs 1..${C} tokens and at least one new one`);
	const tok = W('tok_emb.weight');
	const pos = W('pos_emb.weight');
	let x = new Float32Array(T * D);
	for (let r = 0; r < T; r++) for (let i = 0; i < D; i++) x[r * D + i] = tok[ids[from + r] * D + i] + pos[(from + r) * D + i];
	const scale = 1 / Math.sqrt(dh);
	const p = new Float64Array(C);
	for (let l = 0; l < L; l++) {
		const P = `L${l}.`;
		const h = layerNorm(x, T, D, W(P + 'ln1.weight'), W(P + 'ln1.bias'), eps);
		const q = linear(h, T, D, W(P + 'attn.q.weight'), W(P + 'attn.q.bias'), D, count);
		linear(h, T, D, W(P + 'attn.k.weight'), W(P + 'attn.k.bias'), D, count, cache.k[l], from);
		linear(h, T, D, W(P + 'attn.v.weight'), W(P + 'attn.v.bias'), D, count, cache.v[l], from);
		const K = cache.k[l];
		const Vv = cache.v[l];
		const heads = new Float32Array(T * D);
		for (let r = 0; r < T; r++) {
			const i = from + r; // the row's position: it attends to the keys 0..i
			for (let hh = 0; hh < H; hh++) {
				const o = hh * dh;
				let m = -Infinity;
				for (let j = 0; j <= i; j++) {
					let s = 0;
					for (let c = 0; c < dh; c++) s += q[r * D + o + c] * K[j * D + o + c];
					p[j] = f32(s * scale);
					if (p[j] > m) m = p[j];
				}
				let z = 0;
				for (let j = 0; j <= i; j++) z += p[j] = Math.exp(p[j] - m);
				for (let j = 0; j <= i; j++) {
					p[j] = f32(p[j] / z);
					if (att) att[(l * C + i) * C + j] += p[j] / H;
				}
				for (let c = 0; c < dh; c++) {
					let s = 0;
					for (let j = 0; j <= i; j++) s += p[j] * Vv[j * D + o + c];
					heads[r * D + o + c] = s;
				}
				count.macs += 2 * (i + 1) * dh;
			}
		}
		const proj = linear(heads, T, D, W(P + 'attn.proj.weight'), W(P + 'attn.proj.bias'), D, count);
		const x1 = new Float32Array(T * D);
		for (let i = 0; i < T * D; i++) x1[i] = x[i] + proj[i];
		const h2 = layerNorm(x1, T, D, W(P + 'ln2.weight'), W(P + 'ln2.bias'), eps);
		const fc = linear(h2, T, D, W(P + 'mlp.fc.weight'), W(P + 'mlp.fc.bias'), F, count);
		for (let i = 0; i < fc.length; i++) fc[i] = gelu(fc[i]);
		const m2 = linear(fc, T, F, W(P + 'mlp.proj.weight'), W(P + 'mlp.proj.bias'), D, count);
		x = new Float32Array(T * D);
		for (let i = 0; i < T * D; i++) x[i] = x1[i] + m2[i];
	}
	cache.n = ids.length;
	const last = layerNorm(x.subarray((T - 1) * D), 1, D, W('ln_f.weight'), W('ln_f.bias'), eps);
	return linear(last, 1, D, tok, null, V, count);
}

export function argmax(a: ArrayLike<number>): number {
	let k = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
	return k;
}

export function softmaxAt(logits: ArrayLike<number>, i: number): number {
	let m = -Infinity;
	for (let j = 0; j < logits.length; j++) m = Math.max(m, logits[j]);
	let z = 0;
	for (let j = 0; j < logits.length; j++) z += Math.exp(logits[j] - m);
	return Math.exp(logits[i] - m) / z;
}

export function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>, n = Math.min(a.length, b.length), from = 0): number {
	let m = 0;
	for (let i = from; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
}
