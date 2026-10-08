// Seeded 1-D series (W1-B1) for the forecaster and the PINN. TS is the source of truth for synthetic data.
// Deterministic shapes take no Rng; noisy ones draw only from the Rng they are given.
import type { Rng } from '../types/index.ts';

export interface Series {
	name: string;
	t: Float64Array;
	y: Float64Array;
}

/** n evenly spaced values from a to b (inclusive; n = 1 gives [a]). */
export function linspace(a: number, b: number, n: number): Float64Array {
	const out = new Float64Array(n);
	for (let i = 0; i < n; i++) out[i] = n === 1 ? a : a + ((b - a) * i) / (n - 1);
	return out;
}

function steps(n: number): Float64Array {
	const t = new Float64Array(n);
	for (let i = 0; i < n; i++) t[i] = i;
	return t;
}

function addNoise(y: Float64Array, noise: number, rng?: Rng): Float64Array {
	if (noise > 0) {
		if (!rng) throw new Error('series: noise needs an rng');
		for (let i = 0; i < y.length; i++) y[i] += noise * rng.normal();
	}
	return y;
}

export function sine(n: number, o: { period?: number; amplitude?: number; phase?: number; noise?: number } = {}, rng?: Rng): Series {
	const period = o.period ?? 24;
	const amp = o.amplitude ?? 1;
	const phase = o.phase ?? 0;
	const t = steps(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) y[i] = amp * Math.sin((2 * Math.PI * i) / period + phase);
	return { name: 'sine', t, y: addNoise(y, o.noise ?? 0, rng) };
}

export function trend(n: number, o: { slope?: number; intercept?: number; noise?: number } = {}, rng?: Rng): Series {
	const slope = o.slope ?? 0.02;
	const b = o.intercept ?? -0.5;
	const t = steps(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) y[i] = b + slope * i;
	return { name: 'trend', t, y: addNoise(y, o.noise ?? 0, rng) };
}

/** Trend + seasonality + noise (the classic forecasting toy). */
export function seasonal(n: number, rng: Rng, o: { period?: number; amplitude?: number; slope?: number; noise?: number } = {}): Series {
	const period = o.period ?? 12;
	const amp = o.amplitude ?? 0.6;
	const slope = o.slope ?? 0.01;
	const t = steps(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) y[i] = -0.5 + slope * i + amp * Math.sin((2 * Math.PI * i) / period);
	return { name: 'seasonal', t, y: addNoise(y, o.noise ?? 0.1, rng) };
}

export function square(n: number, o: { period?: number; amplitude?: number } = {}): Series {
	const period = o.period ?? 16;
	const amp = o.amplitude ?? 0.8;
	const t = steps(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) y[i] = (i % period) < period / 2 ? amp : -amp;
	return { name: 'square', t, y };
}

/** A Gaussian random walk (the "nothing to learn" breakIt series). */
export function randomWalk(n: number, rng: Rng, o: { step?: number; start?: number } = {}): Series {
	const s = o.step ?? 0.1;
	const t = steps(n);
	const y = new Float64Array(n);
	let v = o.start ?? 0;
	for (let i = 0; i < n; i++) {
		y[i] = v;
		v += s * rng.normal();
	}
	return { name: 'randomWalk', t, y };
}

/** Sliding windows: inputs [W, L] and the next H values [W, H], for W = floor((n − L − H) / stride) + 1 windows. */
export function windows(y: ArrayLike<number>, L: number, H: number, stride = 1): { x: Float64Array; y: Float64Array; count: number; L: number; H: number } {
	const n = y.length;
	const count = n - L - H < 0 ? 0 : Math.floor((n - L - H) / stride) + 1;
	const xs = new Float64Array(count * L);
	const ys = new Float64Array(count * H);
	for (let w = 0; w < count; w++) {
		const s = w * stride;
		for (let i = 0; i < L; i++) xs[w * L + i] = y[s + i];
		for (let i = 0; i < H; i++) ys[w * H + i] = y[s + L + i];
	}
	return { x: xs, y: ys, count, L, H };
}

export interface Oscillator {
	/** damping ratio ζ (< 1: underdamped) */
	zeta: number;
	/** natural angular frequency ω */
	omega: number;
	x0?: number;
	v0?: number;
}

/**
 * The underdamped oscillator u'' + 2ζω u' + ω² u = 0 with u(0) = x0, u'(0) = v0, solved exactly:
 * u = e^{−ζωt}(x0 cos ω_d t + ((v0 + ζω x0)/ω_d) sin ω_d t), ω_d = ω√(1 − ζ²).
 */
export function dampedOscillator(t: ArrayLike<number>, o: Oscillator): Float64Array {
	const { zeta, omega } = o;
	if (!(zeta >= 0 && zeta < 1)) throw new Error('dampedOscillator: needs 0 ≤ zeta < 1 (underdamped)');
	const x0 = o.x0 ?? 1;
	const v0 = o.v0 ?? 0;
	const wd = omega * Math.sqrt(1 - zeta * zeta);
	const B = (v0 + zeta * omega * x0) / wd;
	const out = new Float64Array(t.length);
	for (let i = 0; i < t.length; i++) {
		const ti = t[i];
		out[i] = Math.exp(-zeta * omega * ti) * (x0 * Math.cos(wd * ti) + B * Math.sin(wd * ti));
	}
	return out;
}

/** n noisy observations of f at times drawn uniformly in [0, frac·tMax], sorted by time (PINN data points). */
export function observations(f: (t: ArrayLike<number>) => Float64Array, rng: Rng, o: { n?: number; tMax?: number; frac?: number; noise?: number } = {}): { t: Float64Array; u: Float64Array } {
	const n = o.n ?? 8;
	const tMax = o.tMax ?? 1;
	const frac = o.frac ?? 0.4;
	const noise = o.noise ?? 0.02;
	const t = new Float64Array(n);
	for (let i = 0; i < n; i++) t[i] = frac * tMax * rng.next();
	t.sort();
	const u = f(t);
	for (let i = 0; i < n; i++) u[i] += noise * rng.normal();
	return { t, u };
}
