// positional-encodings: the arithmetic of the page (pure TS, float64). The four ways to give attention positions, as
// pipeline/positional-encodings/extrap.py trains them (claims.test.ts checks both against facts.json):
//   sinusoidal  PE[p, 2i] = sin(p·ω_i), PE[p, 2i+1] = cos(p·ω_i), ω_i = 10000^(−2i/d), added to the token embedding
//   learned     a table of rows, one per position, added to the token embedding (TinyGPT's pos_emb: 64 rows of 76)
//   RoPE        the pairs (x0, x1), (x2, x3), ... of a query or key turned by the angle p·θ_i, θ_i = 10000^(−2i/r), where r
//               is the number of turned dims (18 of a TinyGPT head's 19; the last one is not turned)
//   ALiBi       the attention score of a key at distance k gets −m·k, one slope m = 2^(−8(h+1)/H) per head h
// TinyGPT's layer-0 query and key without positions (qkOf) feed the RoPE stage; shuffleOf orders the text's tokens.
import type { Tensor } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const D = 76; // TinyGPT's width
export const H = 4; // heads
export const HD = 19; // dims per head
export const ROT = 18; // RoPE turns 18 of a head's 19 dims: 9 pairs
export const SIN_ROWS = 128; // positions of the sinusoidal table drawn on the page
export const BASE = 10000;

/** ω_i (sinusoidal, width d) or θ_i (RoPE over r turned dims): BASE^(−2i/d). */
export const freq = (i: number, d: number) => BASE ** ((-2 * i) / d);

/** The sinusoidal table [n, d]. */
export function sinTable(n = SIN_ROWS, d = D): Float64Array {
	const t = new Float64Array(n * d);
	for (let p = 0; p < n; p++)
		for (let i = 0; i < d / 2; i++) {
			const a = p * freq(i, d);
			t[p * d + 2 * i] = Math.sin(a);
			t[p * d + 2 * i + 1] = Math.cos(a);
		}
	return t;
}

/** A head vector (HD dims) with RoPE at position p: pairs turned by p·θ_i, the last dim as it is. */
export function rope(x: ArrayLike<number>, p: number): Float64Array {
	const out = Float64Array.from(x);
	for (let i = 0; i < ROT / 2; i++) {
		const a = p * freq(i, ROT);
		const [c, s] = [Math.cos(a), Math.sin(a)];
		out[2 * i] = x[2 * i] * c - x[2 * i + 1] * s;
		out[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c;
	}
	return out;
}

/** ALiBi's slope of head h of H: 2^(−8(h+1)/H) (1/4, 1/16, 1/64, 1/256 for four heads). */
export const slope = (h: number, nh = H) => 2 ** ((-8 * (h + 1)) / nh);

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>, n = a.length) => {
	let s = 0;
	for (let i = 0; i < n; i++) s += a[i] * b[i];
	return s;
};
/** Cosine similarity of rows i and j of a table with d columns. */
export function cosRows(t: ArrayLike<number>, d: number, i: number, j: number): number {
	let [s, a, b] = [0, 0, 0];
	for (let c = 0; c < d; c++) {
		const [x, y] = [t[i * d + c], t[j * d + c]];
		(s += x * y), (a += x * x), (b += y * y);
	}
	return s / Math.sqrt(a * b);
}
export function rowNorm(t: ArrayLike<number>, d: number, i: number): number {
	let a = 0;
	for (let c = 0; c < d; c++) a += t[i * d + c] ** 2;
	return Math.sqrt(a);
}

/** The order of the shuffled run: every token but the last one shuffled (seeded), the last one kept in place. Never the
 *  same order as typed when two or more tokens can move. */
export function shuffleOf(n: number, seed: number): number[] {
	const idx = Array.from({ length: n }, (_, i) => i);
	const m = n - 1;
	const rng = createRng(seed);
	for (let i = m - 1; i > 0; i--) {
		const j = rng.int(i + 1);
		[idx[i], idx[j]] = [idx[j], idx[i]];
	}
	if (m >= 2 && idx.slice(0, m).every((v, i) => v === i)) idx.unshift(...idx.splice(m - 1, 1));
	return idx;
}

type W = Record<string, Tensor>;
/** TinyGPT's layer-0 LayerNorm, then the q or k projection, of one input row x (76 numbers); returns all 76 outputs. */
export function project(w: W, x: ArrayLike<number>, which: 'q' | 'k'): Float64Array {
	let mu = 0;
	for (let c = 0; c < D; c++) mu += x[c] / D;
	let v = 0;
	for (let c = 0; c < D; c++) v += (x[c] - mu) ** 2 / D;
	const g = w['L0.ln1.weight'].data;
	const b = w['L0.ln1.bias'].data;
	const h = Float64Array.from({ length: D }, (_, c) => ((x[c] - mu) / Math.sqrt(v + 1e-5)) * g[c] + b[c]);
	const M = w[`L0.attn.${which}.weight`].data;
	const bias = w[`L0.attn.${which}.bias`].data;
	return Float64Array.from({ length: D }, (_, o) => {
		let s = bias[o];
		for (let c = 0; c < D; c++) s += M[o * D + c] * h[c];
		return s;
	});
}
/** One row of a [rows, D] table, plus optionally a row of a second table. */
export function rowOf(t: ArrayLike<number>, i: number, add?: ArrayLike<number>, j = 0): Float64Array {
	return Float64Array.from({ length: D }, (_, c) => t[i * D + c] + (add ? add[j * D + c] : 0));
}
