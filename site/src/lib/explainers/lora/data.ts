// lora: the toy's world and the plain arithmetic of LoRA, shared by the processes, the views, the rank widget and the
// claim tests. No third-party data.
//
// The toy: one "pretrained" linear layer W0 (16 × 16, frozen) and a new task whose answers need W0 + ΔW*, where the
// needed change ΔW* = Σ s_k·u_k·v_kᵀ has a spectrum chosen here: two strong directions and a weak tail. The 64 inputs are
// four orthonormal bases scaled by 4, so XᵀX = 64·I exactly; then the loss (the mean squared error over all 64 × 16
// outputs) of any change ΔW is ‖ΔW − ΔW*‖²_F / 16, and the best a rank-r change can reach is the sum of the left-out s_k²
// over 16 (Eckart–Young). Everything is drawn from one seeded generator with uniform draws only (rng.next()) and
// + − × ÷ √, so pipeline/lora/reference.py rebuilds it bit for bit.
import type { I18n } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

/** the toy layer: d_in = d_out = D */
export const D = 16;
/** training inputs */
export const N = 64;
/** LoRA's α: the update is scaled by α / r (fixed α, as the LoRA paper does, so a new r needs no new learning rate) */
export const ALPHA = 8;
export const R_MIN = 1;
export const R_MAX = 8;
/** the task's goal: a loss below this after training */
export const GOAL = 0.02;

/** The singular values of the needed change: 2 and 1.2, then a tail 0.3, 0.3·0.75, 0.3·0.75², ... (16 in all). */
export const SPECTRUM: number[] = (() => {
	const s = [2, 1.2];
	let v = 0.3;
	while (s.length < D) {
		s.push(v);
		v *= 0.75;
	}
	return s;
})();

/** D orthonormal vectors (rows) by modified Gram–Schmidt on uniform random vectors in [−1, 1). */
function basis(next: () => number): number[][] {
	const out: number[][] = [];
	while (out.length < D) {
		const v = Array.from({ length: D }, () => 2 * next() - 1);
		for (const u of out) {
			let d = 0;
			for (let i = 0; i < D; i++) d += u[i] * v[i];
			for (let i = 0; i < D; i++) v[i] -= d * u[i];
		}
		let n = 0;
		for (let i = 0; i < D; i++) n += v[i] * v[i];
		n = Math.sqrt(n);
		for (let i = 0; i < D; i++) v[i] /= n;
		out.push(v);
	}
	return out;
}

export interface World {
	/** the frozen pretrained layer, [D, D] row-major (out, in) */
	W0: Float64Array;
	/** the change the new task needs, [D, D] */
	dWstar: Float64Array;
	/** inputs [N, D] and the new task's answers [N, D] = X (W0 + ΔW*)ᵀ */
	X: Float64Array;
	Y: Float64Array;
}

/** The toy's world (seed 7). Draw order: W0 (D·D), U (Gram–Schmidt), V, then the four input bases. */
export const WORLD: World = (() => {
	const rng = createRng(7);
	const next = () => rng.next();
	const W0 = new Float64Array(D * D);
	for (let i = 0; i < D * D; i++) W0[i] = next() - 0.5;
	const U = basis(next);
	const V = basis(next);
	const dWstar = new Float64Array(D * D);
	for (let o = 0; o < D; o++)
		for (let i = 0; i < D; i++) {
			let s = 0;
			for (let k = 0; k < D; k++) s += SPECTRUM[k] * U[k][o] * V[k][i];
			dWstar[o * D + i] = s;
		}
	const X = new Float64Array(N * D);
	for (let b = 0; b < N / D; b++) {
		const Q = basis(next);
		for (let r = 0; r < D; r++) for (let i = 0; i < D; i++) X[(b * D + r) * D + i] = 4 * Q[r][i];
	}
	const Y = new Float64Array(N * D);
	for (let n = 0; n < N; n++)
		for (let o = 0; o < D; o++) {
			let s = 0;
			for (let i = 0; i < D; i++) s += X[n * D + i] * (W0[o * D + i] + dWstar[o * D + i]);
			Y[n * D + o] = s;
		}
	return { W0, dWstar, X, Y };
})();

/** The rank the learner set: an integer in [R_MIN, R_MAX]; anything else is 2. */
export function rankOf(raw: unknown): number {
	const v = raw && typeof raw === 'object' ? (raw as { r?: unknown }).r : raw;
	const r = typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : 2;
	return Math.min(R_MAX, Math.max(R_MIN, r));
}

/** The frame's rank (its metrics.r), else the learner's: a step or a view before its source has run. */
export const rankAt = (p: { rt?: { input: unknown }; at?: { frame?: { metrics?: Record<string, number> } } }): number => p.at?.frame?.metrics?.r ?? rankOf(p.rt?.input);

/** LoRA's trainable numbers for one d_out × d_in matrix: B (d_out × r) and A (r × d_in). */
export const loraCount = (r: number, dIn: number, dOut: number) => r * (dIn + dOut);
/** The best any change of rank r can reach on the new task: Σ of the left-out s_k² / D (r = 0: the base layer). */
export function floorOf(r: number): number {
	let s = 0;
	for (let k = Math.max(0, r); k < D; k++) s += SPECTRUM[k] * SPECTRUM[k];
	return s / D;
}
/** the base layer's loss on the new task */
export const BASE_LOSS = floorOf(0);
/** The share of the needed change's "energy" (Σ s_k²) in its first r directions. */
export const shareOf = (r: number) => 1 - floorOf(r) / BASE_LOSS;
/** The smallest rank whose best possible loss is below the goal. */
export const MIN_RANK = (() => {
	for (let r = R_MIN; r <= R_MAX; r++) if (floorOf(r) < GOAL) return r;
	return R_MAX;
})();

// ---------------------------------------------------------------- real sizes (analytic)

export interface LayerSize {
	id: string;
	label: I18n;
	dIn: number;
	dOut: number;
}
/** GPT-2 small (124M): its published configuration. */
export const GPT2 = { vocab: 50257, context: 1024, d: 768, layers: 12, dFF: 3072 };
/** TinyGPT (this site's model): static/models/tinygpt/spec.json (dModel, dMLP); the claim tests read the file. */
export const TINYGPT = { d: 76, dMLP: 304 };

export const LAYERS: LayerSize[] = [
	{ id: 'toy', label: { en: 'The toy layer of this page', zh: '本页的玩具层' }, dIn: D, dOut: D },
	{ id: 'tinygpt-mlp', label: { en: 'TinyGPT, one MLP layer', zh: 'TinyGPT 的一个 MLP 层' }, dIn: TINYGPT.d, dOut: TINYGPT.dMLP },
	{ id: 'gpt2-attn', label: { en: 'GPT-2 small, one attention projection', zh: 'GPT-2 small 的一个注意力投影' }, dIn: GPT2.d, dOut: GPT2.d },
	{ id: 'gpt2-mlp', label: { en: 'GPT-2 small, one MLP layer', zh: 'GPT-2 small 的一个 MLP 层' }, dIn: GPT2.d, dOut: GPT2.dFF }
];

/** Every parameter of GPT-2 small (tied output head): embeddings, 12 blocks (2 LayerNorms, attention qkv + output,
 *  MLP up + down, with biases) and the final LayerNorm. */
export const GPT2_PARAMS = (() => {
	const { vocab, context, d, layers, dFF } = GPT2;
	const block = 2 * (2 * d) + (d * 3 * d + 3 * d) + (d * d + d) + (d * dFF + dFF) + (dFF * d + d);
	return vocab * d + context * d + layers * block + 2 * d;
})();
/** LoRA on GPT-2 small's query, key, value and output projections (768 × 768 each) and both MLP layers of every block. */
export const gpt2Lora = (r: number) => GPT2.layers * (4 * loraCount(r, GPT2.d, GPT2.d) + loraCount(r, GPT2.d, GPT2.dFF) + loraCount(r, GPT2.dFF, GPT2.d));

/** Training memory in bytes, without activations: every number a 32-bit float (4 bytes); each trainable number also
 *  needs its gradient (4) and Adam's two running averages (8). QLoRA stores the frozen weights with 4 bits (0.5 bytes;
 *  the small per-block scales are left out). */
export interface Memory {
	weights: number;
	grads: number;
	optimizer: number;
	total: number;
}
const mem = (frozen: number, frozenBytes: number, trainable: number): Memory => {
	const weights = frozen * frozenBytes + trainable * 4;
	const grads = trainable * 4;
	const optimizer = trainable * 8;
	return { weights, grads, optimizer, total: weights + grads + optimizer };
};
export function memoryOf(r: number): { full: Memory; lora: Memory; qlora: Memory } {
	const a = gpt2Lora(r);
	return { full: mem(0, 0, GPT2_PARAMS), lora: mem(GPT2_PARAMS, 4, a), qlora: mem(GPT2_PARAMS, 0.5, a) };
}
/** bytes -> megabytes (10⁶ bytes) */
export const MB = (b: number) => b / 1e6;

// ---------------------------------------------------------------- narration helpers

/** a frame's metrics */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
/** a share as the narration prints it: whole percent from 10 %, one decimal from 1 %, else 2 significant digits. A share
 *  just under 1 keeps one decimal and never prints "100%" (a run that still misses 0.3% of its change has not got it all). */
export const pctText = (v: number): string | number => {
	if (!Number.isFinite(v)) return NaN;
	const p = v * 100;
	if (p >= 99.5 && p < 100) return `${Math.min(99.9, Number(p.toFixed(1))).toFixed(1)}%`;
	return `${p >= 10 ? Math.round(p) : p >= 1 ? p.toFixed(1) : p.toPrecision(2)}%`;
};
