// fno: the heat equation on a ring and its Fourier modes (pure TS; the presets, the processes, the views, the lesson and
// the training data all come from here). A curve is N = 64 heights u_0..u_63 around a ring (the last point touches the
// first). The numerical solver is the explicit finite-difference rule
//     u_j <- u_j + r (u_{j-1} - 2 u_j + u_{j+1}),   r = nu dt / dx^2 = R,  STEPS steps,
// so a curve's mode n shrinks by (1 - 4R sin^2(pi n / N))^STEPS (fdGain), close to the exact heat equation's
// exp(-4 pi^2 nu T n^2) (exactGain, nu T = R STEPS / N^2). pipeline/fno/reference.py repeats solve() bit for bit.
// Fourier modes as torch.fft.rfft(norm="forward"): X_k = (1/N) sum_x u_x (cos - i sin)(2 pi k x / N), so X_0 is the mean
// height and a cosine wave of height a has |X_k| = a / 2 (heightOf).
import type { I18n, Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const I = (en: string, zh: string): I18n => ({ en, zh });
export const N = 64;
export const M = N / 2 + 1;
export const STEPS = 50;
export const R = 0.1;
export const DT = 0.02;
/** the largest |height| of a training curve: a shape scaled to at most 1.2, plus a mean of at most 0.3 */
export const AMP = 1.5;

/** one solver step on the ring (the same order of operations as reference.py) */
export function heatStep(u: ArrayLike<number>): Float64Array {
	const v = new Float64Array(N);
	for (let j = 0; j < N; j++) {
		const a = u[(j + N - 1) % N], b = u[j], c = u[(j + 1) % N];
		v[j] = b + R * (a - 2 * b + c);
	}
	return v;
}
export function solve(u0: ArrayLike<number>, steps = STEPS): Float64Array {
	let u: Float64Array = Float64Array.from(u0);
	for (let s = 0; s < steps; s++) u = heatStep(u);
	return u;
}
export const fdGain = (n: number) => (1 - 4 * R * Math.sin((Math.PI * n) / N) ** 2) ** STEPS;
export const exactGain = (n: number) => Math.exp(-4 * Math.PI ** 2 * ((R * STEPS) / N ** 2) * n * n);

/** the spectrum of a curve, as torch.fft.rfft(u, norm="forward"): re[k], im[k] for k = 0..N/2 */
export function spectrum(u: ArrayLike<number>) {
	const n = u.length, m = Math.floor(n / 2) + 1, re = new Float64Array(m), im = new Float64Array(m);
	for (let k = 0; k < m; k++) {
		let a = 0, b = 0;
		for (let x = 0; x < n; x++) {
			const th = (2 * Math.PI * ((k * x) % n)) / n;
			a += u[x] * Math.cos(th);
			b -= u[x] * Math.sin(th);
		}
		re[k] = a / n;
		im[k] = b / n;
	}
	return { re, im };
}
/** the height of mode k's wave in a curve (k = 0: the mean; the top mode N/2 is counted once) */
export const heightOf = (re: number, im: number, k: number, n = N) => (k === 0 || 2 * k === n ? 1 : 2) * Math.hypot(re, im);
/** the curve made of the modes below `keep` only (the inverse of spectrum(), cut) */
export function lowPass(u: ArrayLike<number>, keep: number): Float64Array {
	const { re, im } = spectrum(u), out = new Float64Array(N);
	for (let x = 0; x < N; x++) {
		let s = 0;
		for (let k = 0; k < Math.min(keep, M); k++) {
			const th = (2 * Math.PI * ((k * x) % N)) / N, c = k === 0 || 2 * k === N ? 1 : 2;
			s += c * (re[k] * Math.cos(th) - im[k] * Math.sin(th));
		}
		out[x] = s;
	}
	return out;
}
export const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
	let w = 0, at = 0;
	for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > w) (w = Math.abs(a[i] - b[i])), (at = i);
	return { max: w, at };
};

// ---- starting curves: the training data's four families (uniform draws only) and the presets
export type Family = 'waves' | 'pulses' | 'bumps' | 'lines';
export const FAMILIES: Family[] = ['waves', 'pulses', 'bumps', 'lines'];
const u = (r: Rng, lo: number, hi: number) => lo + (hi - lo) * r.next();
const ring = (d: number) => Math.min(Math.abs(d), 1 - Math.abs(d)); // distance around the ring of length 1

/** One random starting curve of a family; its largest |height| is drawn between 0.3 and 1.2. */
export function initial(f: Family, r: Rng): Float64Array {
	const y = new Float64Array(N), x = (j: number) => j / N;
	if (f === 'waves') {
		const n = 2 + r.int(9);
		for (let k = 1; k <= n; k++) {
			const a = u(r, -1, 1) / k, ph = u(r, 0, 2 * Math.PI);
			for (let j = 0; j < N; j++) y[j] += a * Math.cos(2 * Math.PI * k * x(j) + ph);
		}
	} else if (f === 'pulses') {
		for (let p = 1 + r.int(3); p > 0; p--) {
			const s = u(r, 0, 1), w = u(r, 0.05, 0.4), h = u(r, -1, 1);
			for (let j = 0; j < N; j++) if ((x(j) - s + 1) % 1 < w) y[j] += h;
		}
	} else if (f === 'bumps') {
		for (let p = 1 + r.int(3); p > 0; p--) {
			const c = u(r, 0, 1), w = u(r, 0.02, 0.12), h = u(r, -1.2, 1.2);
			for (let j = 0; j < N; j++) y[j] += h * Math.exp(-0.5 * (ring(x(j) - c) / w) ** 2);
		}
	} else {
		const n = 3 + r.int(6), at = Array.from({ length: n }, (_, i) => (i + u(r, 0, 0.9)) / n), h = at.map(() => u(r, -1, 1));
		for (let j = 0; j < N; j++) {
			let k = n - 1;
			for (let q = 0; q < n; q++) if (x(j) >= at[q]) k = q;
			const a0 = at[k], a1 = k + 1 < n ? at[k + 1] : at[0] + 1, xx = x(j) < at[0] ? x(j) + 1 : x(j);
			y[j] = h[k] + ((xx - a0) / (a1 - a0)) * (h[(k + 1) % n] - h[k]);
		}
	}
	const mean = u(r, -0.3, 0.3), amp = u(r, 0.3, 1.2);
	let top = 0;
	for (const v of y) top = Math.max(top, Math.abs(v));
	for (let j = 0; j < N; j++) y[j] = (top > 0 ? (y[j] / top) * amp : 0) + mean;
	return y;
}

const cosine = (k: number, a: number) => Array.from({ length: N }, (_, j) => a * Math.cos((2 * Math.PI * k * j) / N));
/** The input widget's curves (manifest input.presetsFromAdapter): made here, never a file. */
export const PRESETS: { id: string; en: string; zh: string; y: () => number[] }[] = [
	{ id: 'square', en: 'A square pulse', zh: '一个方波脉冲', y: () => Array.from({ length: N }, (_, j) => (j >= 20 && j < 40 ? 1 : 0)) },
	{ id: 'waves', en: 'A slow wave plus a fast wiggle', zh: '一个慢波加一个快速抖动', y: () => cosine(2, 0.6).map((v, j) => v + cosine(12, 0.35)[j]) },
	{ id: 'bump', en: 'One narrow bump', zh: '一个窄鼓包', y: () => Array.from({ length: N }, (_, j) => Math.exp(-0.5 * (ring(j / N - 0.5) / 0.05) ** 2)) },
	{ id: 'random', en: 'A random smooth curve', zh: '一条随机的平滑曲线', y: () => Array.from(initial('waves', createRng(7))) },
	{ id: 'zigzag', en: 'A zigzag', zh: '一条折线', y: () => Array.from(initial('lines', createRng(11))) }
];
export const presetY = (id: string) => (PRESETS.find((p) => p.id === id)?.y() ?? []).map((v) => Math.round(v * 1e5) / 1e5);

/** CRC-32 (IEEE, as zlib.crc32) of a float32 array's bytes: train.py checks the bytes it read against it. */
const CRC = Array.from({ length: 256 }, (_, n) => {
	let c = n;
	for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
	return c >>> 0;
});
export function crcBytes(a: Float32Array): number {
	const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
	let c = 0xffffffff;
	for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}
