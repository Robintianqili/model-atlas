// Raw float64 loops behind the autograd ops (W1-B1). No graph, no Vars: plain arrays in, fresh arrays out.
// Sums run in index order (row-major), so results are reproducible run to run and across threads.
import { sameShape, sizeOf } from './var.ts';

/** NumPy/PyTorch broadcasting of two shapes. */
export function broadcastShape(a: readonly number[], b: readonly number[]): number[] {
	const r = Math.max(a.length, b.length);
	const out = new Array<number>(r);
	for (let i = 0; i < r; i++) {
		const da = a[a.length - r + i] ?? 1;
		const db = b[b.length - r + i] ?? 1;
		if (da !== db && da !== 1 && db !== 1) throw new Error(`shapes [${a}] and [${b}] cannot be broadcast together`);
		out[i] = da === 1 ? db : da;
	}
	return out;
}

const indexCache = new Map<string, Int32Array>();

/**
 * For every flat index of a tensor of shape `to`, the flat index into a tensor of shape `from` that broadcasts to it.
 * null when the shapes are equal (identity). Cached per shape pair.
 */
export function broadcastIndex(from: readonly number[], to: readonly number[]): Int32Array | null {
	if (sameShape(from, to)) return null;
	const key = `${from.join(',')}|${to.join(',')}`;
	const hit = indexCache.get(key);
	if (hit) return hit;
	const r = to.length;
	const off = r - from.length;
	if (off < 0) throw new Error(`cannot broadcast [${from}] to [${to}]`);
	const st = new Array<number>(r).fill(0);
	let s = 1;
	for (let i = from.length - 1; i >= 0; i--) {
		const d = from[i];
		const t = to[i + off];
		if (d !== 1 && d !== t) throw new Error(`cannot broadcast [${from}] to [${to}]`);
		st[i + off] = d === 1 ? 0 : s;
		s *= d;
	}
	const n = sizeOf(to);
	const idx = new Int32Array(n);
	const ctr = new Array<number>(r).fill(0);
	let cur = 0;
	for (let k = 0; k < n; k++) {
		idx[k] = cur;
		for (let d = r - 1; d >= 0; d--) {
			ctr[d]++;
			cur += st[d];
			if (ctr[d] < to[d]) break;
			cur -= st[d] * to[d];
			ctr[d] = 0;
		}
	}
	if (indexCache.size > 256) indexCache.clear();
	indexCache.set(key, idx);
	return idx;
}

/** Materialize `x` (shape from) broadcast to shape `to`. */
export function expand(x: Float64Array, from: readonly number[], to: readonly number[]): Float64Array {
	const idx = broadcastIndex(from, to);
	if (!idx) return x.slice();
	const out = new Float64Array(idx.length);
	for (let k = 0; k < idx.length; k++) out[k] = x[idx[k]];
	return out;
}

/** Sum `g` (shape from) down to a shape `to` that broadcasts to `from` (the adjoint of expand). */
export function reduceTo(g: Float64Array, from: readonly number[], to: readonly number[]): Float64Array {
	const idx = broadcastIndex(to, from);
	if (!idx) return g.slice();
	const out = new Float64Array(sizeOf(to));
	for (let k = 0; k < idx.length; k++) out[idx[k]] += g[k];
	return out;
}

export const ADD = 0;
export const SUB = 1;
export const MUL = 2;
export const DIV = 3;

/** Elementwise a (op) b with broadcasting. Returns the data and the output shape. */
export function binary(code: number, a: Float64Array, as: readonly number[], b: Float64Array, bs: readonly number[]): [Float64Array, number[]] {
	const shape = broadcastShape(as, bs);
	const n = sizeOf(shape);
	const out = new Float64Array(n);
	let x = a;
	let y = b;
	let xs = 0; // 0: full, 1: scalar
	let ys = 0;
	if (a.length === 1 && n !== 1) xs = 1;
	else if (!sameShape(as, shape)) x = expand(a, as, shape);
	if (b.length === 1 && n !== 1) ys = 1;
	else if (!sameShape(bs, shape)) y = expand(b, bs, shape);
	if (xs === 0 && ys === 0) {
		if (code === ADD) for (let i = 0; i < n; i++) out[i] = x[i] + y[i];
		else if (code === SUB) for (let i = 0; i < n; i++) out[i] = x[i] - y[i];
		else if (code === MUL) for (let i = 0; i < n; i++) out[i] = x[i] * y[i];
		else for (let i = 0; i < n; i++) out[i] = x[i] / y[i];
	} else if (ys === 1 && xs === 0) {
		const c = y[0];
		if (code === ADD) for (let i = 0; i < n; i++) out[i] = x[i] + c;
		else if (code === SUB) for (let i = 0; i < n; i++) out[i] = x[i] - c;
		else if (code === MUL) for (let i = 0; i < n; i++) out[i] = x[i] * c;
		else for (let i = 0; i < n; i++) out[i] = x[i] / c;
	} else if (xs === 1 && ys === 0) {
		const c = x[0];
		if (code === ADD) for (let i = 0; i < n; i++) out[i] = c + y[i];
		else if (code === SUB) for (let i = 0; i < n; i++) out[i] = c - y[i];
		else if (code === MUL) for (let i = 0; i < n; i++) out[i] = c * y[i];
		else for (let i = 0; i < n; i++) out[i] = c / y[i];
	} else {
		const c = x[0];
		const d = y[0];
		const v = code === ADD ? c + d : code === SUB ? c - d : code === MUL ? c * d : c / d;
		out.fill(v);
	}
	return [out, shape];
}

/**
 * C = op(A)·op(B), op = transpose when the flag is set. A is stored [m,k] (or [k,m] when ta), B is [k,n] (or [n,k] when tb).
 * Every C[i,j] is summed over p = 0..k-1 in order.
 */
export function matmul(A: Float64Array, B: Float64Array, m: number, k: number, n: number, ta: boolean, tb: boolean): Float64Array {
	const C = new Float64Array(m * n);
	if (!ta && !tb) {
		for (let i = 0; i < m; i++) {
			const ci = i * n;
			const ai = i * k;
			for (let p = 0; p < k; p++) {
				const a = A[ai + p];
				const bp = p * n;
				for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
			}
		}
	} else if (!ta && tb) {
		for (let i = 0; i < m; i++) {
			const ai = i * k;
			const ci = i * n;
			for (let j = 0; j < n; j++) {
				const bj = j * k;
				let s = 0;
				for (let p = 0; p < k; p++) s += A[ai + p] * B[bj + p];
				C[ci + j] = s;
			}
		}
	} else if (ta && !tb) {
		for (let p = 0; p < k; p++) {
			const ap = p * m;
			const bp = p * n;
			for (let i = 0; i < m; i++) {
				const a = A[ap + i];
				const ci = i * n;
				for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
			}
		}
	} else {
		for (let i = 0; i < m; i++) {
			const ci = i * n;
			for (let j = 0; j < n; j++) {
				const bj = j * k;
				let s = 0;
				for (let p = 0; p < k; p++) s += A[p * m + i] * B[bj + p];
				C[ci + j] = s;
			}
		}
	}
	return C;
}

/** Y[i,o] = Σ_p X[i,p]·W[o,p] (+ b[o] after the sum), X [M,in], W [out,in]. */
export function linear(X: Float64Array, W: Float64Array, b: Float64Array | null, M: number, inF: number, outF: number): Float64Array {
	const Y = new Float64Array(M * outF);
	for (let i = 0; i < M; i++) {
		const xi = i * inF;
		const yi = i * outF;
		for (let o = 0; o < outF; o++) {
			const wo = o * inF;
			let s = 0;
			for (let p = 0; p < inF; p++) s += X[xi + p] * W[wo + p];
			Y[yi + o] = b ? s + b[o] : s;
		}
	}
	return Y;
}

/** Column sums of G [M,n] -> [n] (bias gradient). */
export function sumRows(G: Float64Array, M: number, n: number): Float64Array {
	const out = new Float64Array(n);
	for (let i = 0; i < M; i++) {
		const gi = i * n;
		for (let j = 0; j < n; j++) out[j] += G[gi + j];
	}
	return out;
}

/** Split a shape around `axis` into (outer, n, inner). */
export function around(shape: readonly number[], axis: number): [number, number, number] {
	let outer = 1;
	let inner = 1;
	for (let i = 0; i < axis; i++) outer *= shape[i];
	for (let i = axis + 1; i < shape.length; i++) inner *= shape[i];
	return [outer, shape[axis], inner];
}

export function normAxis(axis: number, rank: number): number {
	const a = axis < 0 ? axis + rank : axis;
	if (a < 0 || a >= rank) throw new Error(`axis ${axis} out of range for rank ${rank}`);
	return a;
}

/** Softmax along an axis (max subtracted, exp, divided by the sum), as PyTorch's CPU kernel. */
export function softmax(x: Float64Array, shape: readonly number[], axis: number): Float64Array {
	const [outer, n, inner] = around(shape, axis);
	const y = new Float64Array(x.length);
	for (let o = 0; o < outer; o++) {
		for (let i = 0; i < inner; i++) {
			const base = o * n * inner + i;
			let m = -Infinity;
			for (let j = 0; j < n; j++) m = Math.max(m, x[base + j * inner]);
			let s = 0;
			for (let j = 0; j < n; j++) {
				const e = Math.exp(x[base + j * inner] - m);
				y[base + j * inner] = e;
				s += e;
			}
			for (let j = 0; j < n; j++) y[base + j * inner] /= s;
		}
	}
	return y;
}

/** log-softmax along an axis: (x − max) − log Σ exp(x − max). */
export function logSoftmax(x: Float64Array, shape: readonly number[], axis: number): Float64Array {
	const [outer, n, inner] = around(shape, axis);
	const y = new Float64Array(x.length);
	for (let o = 0; o < outer; o++) {
		for (let i = 0; i < inner; i++) {
			const base = o * n * inner + i;
			let m = -Infinity;
			for (let j = 0; j < n; j++) m = Math.max(m, x[base + j * inner]);
			if (!Number.isFinite(m)) m = 0;
			let s = 0;
			for (let j = 0; j < n; j++) s += Math.exp(x[base + j * inner] - m);
			const ls = Math.log(s);
			for (let j = 0; j < n; j++) y[base + j * inner] = x[base + j * inner] - m - ls;
		}
	}
	return y;
}

/** log Σ exp along an axis (keepdim layout: [outer, 1, inner]): log(Σ exp(x − max)) + max. */
export function logsumexp(x: Float64Array, shape: readonly number[], axis: number): Float64Array {
	const [outer, n, inner] = around(shape, axis);
	const y = new Float64Array(outer * inner);
	for (let o = 0; o < outer; o++) {
		for (let i = 0; i < inner; i++) {
			const base = o * n * inner + i;
			let m = -Infinity;
			for (let j = 0; j < n; j++) m = Math.max(m, x[base + j * inner]);
			if (!Number.isFinite(m)) m = 0;
			let s = 0;
			for (let j = 0; j < n; j++) s += Math.exp(x[base + j * inner] - m);
			y[o * inner + i] = Math.log(s) + m;
		}
	}
	return y;
}

// ------------------------------------------------------------- erf (float64)

const TWO_OVER_SQRT_PI = 1.1283791670955126; // 2/√π

/**
 * erf(x) to about 1e-16 relative. |x| < 2.5: e^{-x²}·Σ 2^n x^{2n+1} / (1·3·…·(2n+1)) (all terms positive, no cancellation).
 * |x| ≥ 2.5: 1 − erfc(x), erfc by its continued fraction (modified Lentz).
 */
export function erf(x: number): number {
	if (Number.isNaN(x)) return NaN;
	const ax = Math.abs(x);
	if (ax < 2.5) {
		const x2 = x * x;
		let term = x;
		let sum = x;
		for (let n = 0; n < 200; n++) {
			term *= (2 * x2) / (2 * n + 3);
			sum += term;
			if (Math.abs(term) <= 1e-17 * Math.abs(sum)) break;
		}
		return TWO_OVER_SQRT_PI * Math.exp(-x2) * sum;
	}
	if (ax > 6) return x > 0 ? 1 : -1;
	const r = 1 - erfcPos(ax);
	return x > 0 ? r : -r;
}

/** erfc(x) for x ≥ 2.5: e^{-x²}/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + …))))) */
function erfcPos(x: number): number {
	const tiny = 1e-300;
	let f = x;
	let C = x;
	let D = 0;
	for (let k = 1; k < 500; k++) {
		const a = k / 2;
		D = x + a * D;
		if (D === 0) D = tiny;
		C = x + a / C;
		if (C === 0) C = tiny;
		D = 1 / D;
		const delta = C * D;
		f *= delta;
		if (Math.abs(delta - 1) < 1e-16) break;
	}
	return (Math.exp(-x * x) / Math.sqrt(Math.PI)) / f;
}

// ------------------------------------------------------------- conv1d

export interface Conv1dCfg {
	N: number;
	C: number;
	T: number;
	O: number;
	K: number;
	stride: number;
	dilation: number;
	padL: number;
	Tout: number;
}

/** y[n,o,t] = Σ_c Σ_k w[o,c,k]·x[n,c,t·s + k·d − padL] (out-of-range x counts as 0). */
export function conv1d(x: Float64Array, w: Float64Array, c: Conv1dCfg): Float64Array {
	const { N, C, T, O, K, stride, dilation, padL, Tout } = c;
	const y = new Float64Array(N * O * Tout);
	for (let n = 0; n < N; n++) {
		for (let o = 0; o < O; o++) {
			const yb = (n * O + o) * Tout;
			for (let t = 0; t < Tout; t++) {
				let s = 0;
				for (let ci = 0; ci < C; ci++) {
					const xb = (n * C + ci) * T;
					const wb = (o * C + ci) * K;
					for (let k = 0; k < K; k++) {
						const ti = t * stride + k * dilation - padL;
						if (ti >= 0 && ti < T) s += w[wb + k] * x[xb + ti];
					}
				}
				y[yb + t] = s;
			}
		}
	}
	return y;
}

/** The input gradient of conv1d: gx[n,c,ti] = Σ g[n,o,t]·w[o,c,k] over t·s + k·d − padL = ti. */
export function conv1dGradInput(g: Float64Array, w: Float64Array, c: Conv1dCfg): Float64Array {
	const { N, C, T, O, K, stride, dilation, padL, Tout } = c;
	const gx = new Float64Array(N * C * T);
	for (let n = 0; n < N; n++) {
		for (let o = 0; o < O; o++) {
			const gb = (n * O + o) * Tout;
			for (let t = 0; t < Tout; t++) {
				const gv = g[gb + t];
				for (let ci = 0; ci < C; ci++) {
					const xb = (n * C + ci) * T;
					const wb = (o * C + ci) * K;
					for (let k = 0; k < K; k++) {
						const ti = t * stride + k * dilation - padL;
						if (ti >= 0 && ti < T) gx[xb + ti] += gv * w[wb + k];
					}
				}
			}
		}
	}
	return gx;
}

/** The weight gradient of conv1d: gw[o,c,k] = Σ_n Σ_t g[n,o,t]·x[n,c,t·s + k·d − padL]. */
export function conv1dGradWeight(g: Float64Array, x: Float64Array, c: Conv1dCfg): Float64Array {
	const { N, C, T, O, K, stride, dilation, padL, Tout } = c;
	const gw = new Float64Array(O * C * K);
	for (let n = 0; n < N; n++) {
		for (let o = 0; o < O; o++) {
			const gb = (n * O + o) * Tout;
			for (let ci = 0; ci < C; ci++) {
				const xb = (n * C + ci) * T;
				const wb = (o * C + ci) * K;
				for (let k = 0; k < K; k++) {
					let s = 0;
					for (let t = 0; t < Tout; t++) {
						const ti = t * stride + k * dilation - padL;
						if (ti >= 0 && ti < T) s += g[gb + t] * x[xb + ti];
					}
					gw[wb + k] += s;
				}
			}
		}
	}
	return gw;
}
