/**
 * TinyGPT guided lesson: small pure helpers (no Svelte, no DOM). Everything is computed from the
 * live trace or the weights, so the lesson can name real numbers in its narration.
 */
import type { ForwardResult } from '../../models/tinygpt';
import type { Lens } from './lens';

/**
 * The block-1 head that looks most at the previous token on this text: for every head, the mean
 * attention weight on token i−1 over query positions i ≥ 1. Returns head 0 (and NaN) for one token.
 */
export function prevTokenHead(f: ForwardResult, block: number, H: number): { head: number; avg: number } {
	const T = f.ids.length;
	if (T < 2) return { head: 0, avg: NaN };
	const w = f.t[`L${block}.attn.weights`].data;
	let best = 0;
	let bestAvg = -Infinity;
	for (let h = 0; h < H; h++) {
		let s = 0;
		for (let i = 1; i < T; i++) s += w[(h * T + i) * T + i - 1];
		const avg = s / (T - 1);
		if (avg > bestAvg) {
			best = h;
			bestAvg = avg;
		}
	}
	return { head: best, avg: bestAvg };
}

/** Index of the largest |value|. */
export function argmaxAbs(a: ArrayLike<number>): number {
	let b = 0;
	for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[b])) b = i;
	return b;
}

/** Mean and spread (biased std, as LayerNorm uses) of a vector. */
export function meanSd(a: ArrayLike<number>): { mean: number; sd: number } {
	let m = 0;
	for (let i = 0; i < a.length; i++) m += a[i];
	m /= a.length || 1;
	let v = 0;
	for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2;
	return { mean: m, sd: Math.sqrt(v / (a.length || 1)) };
}

/**
 * The logit-lens point (index into lens.points) from which the final answer at position t is the
 * top guess and stays the top guess until the end (as on the page's logit lens).
 */
export function decidedAt(lens: Lens, t: number): number {
	const P = lens.points.length;
	let first = P - 1;
	for (let p = P - 1; p >= 0; p--) {
		if (lens.topIds[(p * lens.T + t) * lens.K] === lens.finalTop[t]) first = p;
		else break;
	}
	return first;
}

/** Readable name of a lens point id ("embed", "L2.resid1", …). */
export function lensPointName(id: string): { en: string; zh: string } {
	if (id === 'embed') return { en: 'the embedding (before block 1)', zh: '嵌入之后（第 1 块之前）' };
	const m = /^L(\d)\.resid(\d)$/.exec(id);
	if (!m) return { en: id, zh: id };
	const b = Number(m[1]) + 1;
	return m[2] === '1' ? { en: `block ${b} after attention`, zh: `第 ${b} 块的注意力之后` } : { en: `block ${b} after the MLP`, zh: `第 ${b} 块的 MLP 之后` };
}

/** The tokens the sampler can pick, in token-id order, with their slice [lo, hi) of the running sum. */
export function slices(dist: ArrayLike<number>): { id: number; lo: number; hi: number }[] {
	const out: { id: number; lo: number; hi: number }[] = [];
	let cum = 0;
	for (let i = 0; i < dist.length; i++) {
		if (dist[i] === 0) continue;
		out.push({ id: i, lo: cum, hi: cum + dist[i] });
		cum += dist[i];
	}
	return out;
}

/** One token the lesson generated (step 12), kept for its history list. */
export type GenEntry = {
	/** 1-based number of this generated token in the lesson */
	n: number;
	/** the token the model had just read (its label) */
	after: string;
	token: number;
	/** probability under the sampling distribution, and under the plain model */
	p: number;
	pModel: number;
	/** the random number used (null = greedy), its draw number of the seeded generator, and the slice it hit */
	u: number | null;
	draw: number | null;
	lo: number;
	hi: number;
	/** most likely candidates under the sampling distribution */
	top: { id: number; p: number }[];
	/** the slices of the running sum the draw walked through (only kept for the latest entry) */
	slices: { id: number; lo: number; hi: number }[] | null;
	temperature: number;
	topK: number;
	seed: number;
};
