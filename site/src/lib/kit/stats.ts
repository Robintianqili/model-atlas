/**
 * Tensor statistics for the inspector's Stats tab. Pure functions over typed arrays.
 */

export interface TensorStats {
	count: number;
	min: number;
	max: number;
	mean: number;
	std: number;
	absMax: number;
	/** fraction of entries with |v| <= zeroTol (ReLU outputs are often sparse) */
	zeros: number;
	nan: number;
}

export function tensorStats(data: ArrayLike<number>, zeroTol = 0): TensorStats {
	let n = 0;
	let min = Infinity;
	let max = -Infinity;
	let sum = 0;
	let sumSq = 0;
	let zeros = 0;
	let nan = 0;
	for (let i = 0; i < data.length; i++) {
		const v = data[i];
		if (Number.isNaN(v)) {
			nan++;
			continue;
		}
		n++;
		if (v < min) min = v;
		if (v > max) max = v;
		sum += v;
		sumSq += v * v;
		if (Math.abs(v) <= zeroTol) zeros++;
	}
	const mean = n ? sum / n : 0;
	const variance = n ? Math.max(0, sumSq / n - mean * mean) : 0;
	return {
		count: data.length,
		min: n ? min : 0,
		max: n ? max : 0,
		mean,
		std: Math.sqrt(variance),
		absMax: n ? Math.max(Math.abs(min), Math.abs(max)) : 0,
		zeros: n ? zeros / n : 0,
		nan
	};
}

export interface Histogram {
	edges: number[]; // bins + 1 edges
	counts: number[];
	max: number; // largest count (for scaling bars)
}

export function histogram(data: ArrayLike<number>, bins = 24, domain?: [number, number]): Histogram {
	let lo = domain?.[0];
	let hi = domain?.[1];
	if (lo === undefined || hi === undefined) {
		const s = tensorStats(data);
		lo = s.min;
		hi = s.max;
	}
	if (!(hi > lo)) {
		hi = lo + 1;
	}
	const counts = new Array<number>(bins).fill(0);
	const w = (hi - lo) / bins;
	for (let i = 0; i < data.length; i++) {
		const v = data[i];
		if (Number.isNaN(v)) continue;
		let b = Math.floor((v - lo) / w);
		if (b < 0) b = 0;
		if (b >= bins) b = bins - 1;
		counts[b]++;
	}
	const edges = Array.from({ length: bins + 1 }, (_, i) => lo! + i * w);
	return { edges, counts, max: Math.max(1, ...counts) };
}

/** Slice channel c out of a [C, H, W] row-major array (no copy: a subarray view). */
export function channel(data: Float32Array, c: number, h: number, w: number): Float32Array {
	return data.subarray(c * h * w, (c + 1) * h * w);
}

/** Top-k indices of an array, largest first. */
export function topK(data: ArrayLike<number>, k: number): number[] {
	const idx = Array.from({ length: data.length }, (_, i) => i);
	idx.sort((a, b) => data[b] - data[a]);
	return idx.slice(0, k);
}

/** Numerically stable softmax with temperature. */
export function softmax(logits: ArrayLike<number>, temperature = 1): Float32Array {
	const out = new Float32Array(logits.length);
	const tt = Math.max(1e-6, temperature);
	let m = -Infinity;
	for (let i = 0; i < logits.length; i++) m = Math.max(m, logits[i] / tt);
	let s = 0;
	for (let i = 0; i < logits.length; i++) {
		out[i] = Math.exp(logits[i] / tt - m);
		s += out[i];
	}
	for (let i = 0; i < out.length; i++) out[i] /= s;
	return out;
}

/** Deterministic pseudo-random numbers for illustrative data (mulberry32). */
export function seededRandom(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
