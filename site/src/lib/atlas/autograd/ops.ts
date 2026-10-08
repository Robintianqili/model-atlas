// Differentiable ops (design §4.5, W1-B1). float64, PyTorch semantics (broadcasting, [out, in] weights, CPU formulas).
// Every backward rule is written with these same ops, so with grad(..., { createGraph: true }) the gradients are
// differentiable again (double gradients). With grad mode off, unary and softmax-family rules take a fast raw path.
import { Var, _registerAdd, isGradEnabled, raw, result, sameShape, scalar, sizeOf } from './var.ts';
import * as K from './kernels.ts';

export type VarLike = Var | number;

function asVar(x: VarLike): Var {
	return typeof x === 'number' ? scalar(x) : x;
}

const SQRT1_2 = Math.SQRT1_2;
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
const GELU_PDF = TWO_OVER_SQRT_PI * SQRT1_2 * 0.5; // 1/√(2π)
const GELU_TANH_BETA = Math.SQRT2 * TWO_OVER_SQRT_PI * 0.5; // √(2/π)
const GELU_TANH_KAPPA = 0.044715;

/** A unary op: y = f(x). `fast` fills the input gradient from raw arrays (grad mode off); `slow` builds it with ops. */
function unary(op: string, x: Var, y: Float64Array, fast: (g: Float64Array, r: Float64Array) => void, slow: (g: Var, out: Var) => Var): Var {
	const out: Var = result(op, y, x.shape.slice(), [x], (g, need) => {
		if (!need[0]) return [null];
		if (!isGradEnabled()) {
			const r = new Float64Array(y.length);
			fast(g.data, r);
			return [raw(r, x.shape.slice())];
		}
		return [slow(g, out)];
	});
	return out;
}

// ------------------------------------------------------------------ broadcasting

/** Sum x down to `shape` (the adjoint of broadcastTo). Returns x itself when the shapes already match. */
export function sumTo(x: Var, shape: number[]): Var {
	if (sameShape(x.shape, shape)) return x;
	const y = K.reduceTo(x.data, x.shape, shape);
	return result('sumTo', y, shape.slice(), [x], (g) => [broadcastTo(g, x.shape)]);
}

/** Broadcast x to `shape` (PyTorch expand, materialized). Returns x itself when the shapes already match. */
export function broadcastTo(x: Var, shape: number[]): Var {
	if (sameShape(x.shape, shape)) return x;
	const y = K.expand(x.data, x.shape, shape);
	return result('broadcastTo', y, shape.slice(), [x], (g) => [sumTo(g, x.shape)]);
}

// ------------------------------------------------------------------ binary (broadcasting)

export function add(a: VarLike, b: VarLike): Var {
	const A = asVar(a);
	const B = asVar(b);
	const [y, shape] = K.binary(K.ADD, A.data, A.shape, B.data, B.shape);
	return result('add', y, shape, [A, B], (g, need) => [need[0] ? sumTo(g, A.shape) : null, need[1] ? sumTo(g, B.shape) : null]);
}

export function sub(a: VarLike, b: VarLike): Var {
	const A = asVar(a);
	const B = asVar(b);
	const [y, shape] = K.binary(K.SUB, A.data, A.shape, B.data, B.shape);
	return result('sub', y, shape, [A, B], (g, need) => [need[0] ? sumTo(g, A.shape) : null, need[1] ? neg(sumTo(g, B.shape)) : null]);
}

export function mul(a: VarLike, b: VarLike): Var {
	const A = asVar(a);
	const B = asVar(b);
	const [y, shape] = K.binary(K.MUL, A.data, A.shape, B.data, B.shape);
	return result('mul', y, shape, [A, B], (g, need) => [need[0] ? sumTo(mul(g, B), A.shape) : null, need[1] ? sumTo(mul(g, A), B.shape) : null]);
}

export function div(a: VarLike, b: VarLike): Var {
	const A = asVar(a);
	const B = asVar(b);
	const [y, shape] = K.binary(K.DIV, A.data, A.shape, B.data, B.shape);
	return result('div', y, shape, [A, B], (g, need) => [
		need[0] ? sumTo(div(g, B), A.shape) : null,
		need[1] ? sumTo(neg(div(mul(g, A), mul(B, B))), B.shape) : null
	]);
}

_registerAdd((a, b) => add(a, b));

// ------------------------------------------------------------------ unary

export function neg(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = -a[i];
	return unary(
		'neg',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = -g[i];
		},
		(g) => neg(g)
	);
}

/** k·x for a plain number k. */
export function scale(x: Var, k: number): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = k * a[i];
	return unary(
		'scale',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = k * g[i];
		},
		(g) => scale(g, k)
	);
}

function powScalar(a: number, p: number): number {
	if (p === 1) return a;
	if (p === 0) return 1;
	if (p === 2) return a * a;
	if (p === 0.5) return Math.sqrt(a);
	return Math.pow(a, p);
}

/** x^p for a plain number p. */
export function pow(x: Var, p: number): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = powScalar(a[i], p);
	return unary(
		'pow',
		x,
		y,
		(g, r) => {
			if (p === 0) return;
			for (let i = 0; i < r.length; i++) r[i] = g[i] * (p * powScalar(a[i], p - 1));
		},
		(g) => (p === 0 ? mul(g, 0) : mul(g, scale(pow(x, p - 1), p)))
	);
}

export function square(x: Var): Var {
	return pow(x, 2);
}

export function sqrt(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.sqrt(a[i]);
	return unary(
		'sqrt',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] / (2 * y[i]);
		},
		(g, out) => div(g, scale(out, 2))
	);
}

export function exp(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.exp(a[i]);
	return unary(
		'exp',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * y[i];
		},
		(g, out) => mul(g, out)
	);
}

export function log(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.log(a[i]);
	return unary(
		'log',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] / a[i];
		},
		(g) => div(g, x)
	);
}

export function tanh(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.tanh(a[i]);
	return unary(
		'tanh',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * (1 - y[i] * y[i]);
		},
		(g, out) => mul(g, sub(1, mul(out, out)))
	);
}

export function sigmoid(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = 1 / (1 + Math.exp(-a[i]));
	return unary(
		'sigmoid',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * (1 - y[i]) * y[i];
		},
		(g, out) => mul(mul(g, sub(1, out)), out)
	);
}

export function relu(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = a[i] > 0 ? a[i] : 0;
	return unary(
		'relu',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = a[i] > 0 ? g[i] : 0;
		},
		(g) => {
			const m = new Float64Array(a.length);
			for (let i = 0; i < a.length; i++) m[i] = a[i] > 0 ? 1 : 0;
			return mul(g, raw(m, x.shape.slice()));
		}
	);
}

/** x·sigmoid(x), computed as x / (1 + e^{−x}) like PyTorch's CPU kernel. */
export function silu(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = a[i] / (1 + Math.exp(-a[i]));
	return unary(
		'silu',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) {
				const s = 1 / (1 + Math.exp(-a[i]));
				r[i] = g[i] * s * (1 + a[i] * (1 - s));
			}
		},
		(g) => {
			const s = sigmoid(x);
			return mul(mul(g, s), add(1, mul(x, sub(1, s))));
		}
	);
}

/** erf(x), the Gauss error function (float64, about 1e-16). */
export function erf(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = K.erf(a[i]);
	return unary(
		'erf',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = TWO_OVER_SQRT_PI * Math.exp(-(a[i] * a[i])) * g[i];
		},
		(g) => mul(scale(exp(neg(mul(x, x))), TWO_OVER_SQRT_PI), g)
	);
}

/** GELU. approx 'none' (default): x·Φ(x) with erf. 'tanh': PyTorch's tanh approximation. */
export function gelu(x: Var, approx: 'none' | 'tanh' = 'none'): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	if (approx === 'tanh') {
		for (let i = 0; i < a.length; i++) {
			const v = a[i];
			y[i] = 0.5 * v * (1 + Math.tanh(GELU_TANH_BETA * (v + GELU_TANH_KAPPA * v * v * v)));
		}
		return unary(
			'gelu_tanh',
			x,
			y,
			(g, r) => {
				for (let i = 0; i < r.length; i++) {
					const v = a[i];
					const xsq = v * v;
					const t = Math.tanh(GELU_TANH_BETA * (v + GELU_TANH_KAPPA * xsq * v));
					const left = 0.5 * v;
					const leftD = 0.5 * (1 + t);
					const rightD = left * (1 - t * t) * (GELU_TANH_BETA * (1 + 3 * GELU_TANH_KAPPA * xsq));
					r[i] = g[i] * (leftD + rightD);
				}
			},
			(g) => {
				const t = tanh(scale(add(x, scale(pow(x, 3), GELU_TANH_KAPPA)), GELU_TANH_BETA));
				const leftD = scale(add(t, 1), 0.5);
				const rightD = mul(mul(scale(x, 0.5), sub(1, mul(t, t))), scale(add(scale(mul(x, x), 3 * GELU_TANH_KAPPA), 1), GELU_TANH_BETA));
				return mul(g, add(leftD, rightD));
			}
		);
	}
	for (let i = 0; i < a.length; i++) y[i] = a[i] * 0.5 * (1 + K.erf(a[i] * SQRT1_2));
	return unary(
		'gelu',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) {
				const v = a[i];
				const cdf = 0.5 * (1 + K.erf(v * SQRT1_2));
				const pdf = GELU_PDF * Math.exp(v * v * -0.5);
				r[i] = g[i] * (cdf + v * pdf);
			}
		},
		(g) => {
			const cdf = scale(add(erf(scale(x, SQRT1_2)), 1), 0.5);
			const pdf = scale(exp(scale(mul(x, x), -0.5)), GELU_PDF);
			return mul(g, add(cdf, mul(x, pdf)));
		}
	);
}

export function sin(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.sin(a[i]);
	return unary(
		'sin',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * Math.cos(a[i]);
		},
		(g) => mul(g, cos(x))
	);
}

export function cos(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.cos(a[i]);
	return unary(
		'cos',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * -Math.sin(a[i]);
		},
		(g) => mul(g, neg(sin(x)))
	);
}

export function abs(x: Var): Var {
	const a = x.data;
	const y = new Float64Array(a.length);
	for (let i = 0; i < a.length; i++) y[i] = Math.abs(a[i]);
	const sign = (v: number) => (v > 0 ? 1 : v < 0 ? -1 : 0);
	return unary(
		'abs',
		x,
		y,
		(g, r) => {
			for (let i = 0; i < r.length; i++) r[i] = g[i] * sign(a[i]);
		},
		(g) => {
			const s = new Float64Array(a.length);
			for (let i = 0; i < a.length; i++) s[i] = sign(a[i]);
			return mul(g, raw(s, x.shape.slice()));
		}
	);
}

// ------------------------------------------------------------------ linear algebra

/** op(a)·op(b) for 2-D a, b; op = transpose when the flag is set. Closed under differentiation. */
function mm(a: Var, b: Var, ta: boolean, tb: boolean): Var {
	if (a.shape.length !== 2 || b.shape.length !== 2) throw new Error(`matmul needs 2-D inputs, got [${a.shape}] and [${b.shape}]`);
	const m = ta ? a.shape[1] : a.shape[0];
	const k = ta ? a.shape[0] : a.shape[1];
	const k2 = tb ? b.shape[1] : b.shape[0];
	const n = tb ? b.shape[0] : b.shape[1];
	if (k !== k2) throw new Error(`matmul: inner sizes differ (${k} vs ${k2}) for [${a.shape}] and [${b.shape}]`);
	const y = K.matmul(a.data, b.data, m, k, n, ta, tb);
	return result('matmul', y, [m, n], [a, b], (g, need) => [
		need[0] ? (ta ? mm(b, g, tb, true) : mm(g, b, false, !tb)) : null,
		need[1] ? (tb ? mm(g, a, true, ta) : mm(a, g, !ta, false)) : null
	]);
}

/** Matrix product. 2-D × 2-D; a 1-D left operand is a row, a 1-D right operand is a column (both squeezed back). */
export function matmul(a: Var, b: Var): Var {
	const a1 = a.shape.length === 1;
	const b1 = b.shape.length === 1;
	const A = a1 ? reshape(a, [1, a.shape[0]]) : a;
	const B = b1 ? reshape(b, [b.shape[0], 1]) : b;
	const y = mm(A, B, false, false);
	if (a1 && b1) return reshape(y, []);
	if (a1) return reshape(y, [y.shape[1]]);
	if (b1) return reshape(y, [y.shape[0]]);
	return y;
}

/** y = x·Wᵀ + b over the last axis. x [..., in], W [out, in] (PyTorch layout), b [out] or null. */
export function linear(x: Var, w: Var, b?: Var | null): Var {
	if (w.shape.length !== 2) throw new Error(`linear: weight must be [out, in], got [${w.shape}]`);
	const [outF, inF] = w.shape;
	if (x.shape[x.shape.length - 1] !== inF) throw new Error(`linear: input [${x.shape}] does not end in ${inF}`);
	if (b && (b.shape.length !== 1 || b.shape[0] !== outF)) throw new Error(`linear: bias must be [${outF}], got [${b.shape}]`);
	const lead = x.shape.slice(0, -1);
	const M = sizeOf(lead);
	const y = K.linear(x.data, w.data, b ? b.data : null, M, inF, outF);
	const parents = b ? [x, w, b] : [x, w];
	return result('linear', y, [...lead, outF], parents, (g, need) => {
		const g2 = g.shape.length === 2 ? g : reshape(g, [M, outF]);
		const x2 = x.shape.length === 2 ? x : reshape(x, [M, inF]);
		const gx = need[0] ? mm(g2, w, false, false) : null;
		return [
			gx ? (x.shape.length === 2 ? gx : reshape(gx, x.shape)) : null,
			need[1] ? mm(g2, x2, true, false) : null,
			b && need[2] ? sum(g2, 0) : null
		];
	});
}

// ------------------------------------------------------------------ reductions and shapes

function normAxes(axis: number | number[] | null | undefined, rank: number): number[] {
	if (axis === undefined || axis === null) return Array.from({ length: rank }, (_, i) => i);
	const list = Array.isArray(axis) ? axis : [axis];
	return [...new Set(list.map((a) => K.normAxis(a, rank)))].sort((p, q) => p - q);
}

/** Sum over `axis` (a number, a list, or every axis when omitted). */
export function sum(x: Var, axis?: number | number[] | null, keepdim = false): Var {
	const axes = normAxes(axis, x.shape.length);
	const keepShape = x.shape.map((d, i) => (axes.includes(i) ? 1 : d));
	const outShape = keepdim ? keepShape : x.shape.filter((_, i) => !axes.includes(i));
	const y = K.reduceTo(x.data, x.shape, keepShape);
	return result('sum', y, outShape, [x], (g) => [broadcastTo(keepdim ? g : reshape(g, keepShape), x.shape)]);
}

/** Mean over `axis` (sum divided by the count, as PyTorch). */
export function mean(x: Var, axis?: number | number[] | null, keepdim = false): Var {
	const s = sum(x, axis, keepdim);
	const count = x.size / Math.max(1, s.size);
	return div(s, count);
}

/** Same values, new shape (one entry may be −1). The data is copied. */
export function reshape(x: Var, shape: number[]): Var {
	const n = x.size;
	const known = shape.reduce((p, d) => (d === -1 ? p : p * d), 1);
	const s = shape.map((d) => (d === -1 ? n / known : d));
	if (sizeOf(s) !== n) throw new Error(`reshape: [${x.shape}] has ${n} values, [${shape}] needs ${sizeOf(s)}`);
	return result('reshape', x.data.slice(), s, [x], (g) => [reshape(g, x.shape)]);
}

function permuteRaw(x: Float64Array, shape: number[], perm: number[]): [Float64Array, number[]] {
	const r = shape.length;
	const outShape = perm.map((p) => shape[p]);
	const inStride = new Array<number>(r);
	let s = 1;
	for (let i = r - 1; i >= 0; i--) {
		inStride[i] = s;
		s *= shape[i];
	}
	const st = perm.map((p) => inStride[p]);
	const n = x.length;
	const out = new Float64Array(n);
	const ctr = new Array<number>(r).fill(0);
	let cur = 0;
	for (let k = 0; k < n; k++) {
		out[k] = x[cur];
		for (let d = r - 1; d >= 0; d--) {
			ctr[d]++;
			cur += st[d];
			if (ctr[d] < outShape[d]) break;
			cur -= st[d] * outShape[d];
			ctr[d] = 0;
		}
	}
	return [out, outShape];
}

/** Swap two axes (default: the last two). */
export function transpose(x: Var, dim0 = -2, dim1 = -1): Var {
	const r = x.shape.length;
	const d0 = K.normAxis(dim0, r);
	const d1 = K.normAxis(dim1, r);
	if (d0 === d1) return x;
	const perm = Array.from({ length: r }, (_, i) => i);
	perm[d0] = d1;
	perm[d1] = d0;
	const [y, shape] = permuteRaw(x.data, x.shape, perm);
	return result('transpose', y, shape, [x], (g) => [transpose(g, d0, d1)]);
}

/** Join along `axis`. */
export function concat(xs: Var[], axis = 0): Var {
	if (xs.length === 0) throw new Error('concat of nothing');
	const r = xs[0].shape.length;
	const ax = K.normAxis(axis, r);
	for (const x of xs) {
		if (x.shape.length !== r || x.shape.some((d, i) => i !== ax && d !== xs[0].shape[i])) {
			throw new Error(`concat: shapes [${xs.map((v) => `[${v.shape}]`).join(', ')}] differ outside axis ${ax}`);
		}
	}
	const sizes = xs.map((x) => x.shape[ax]);
	const total = sizes.reduce((p, q) => p + q, 0);
	const shape = xs[0].shape.slice();
	shape[ax] = total;
	const [outer, , inner] = K.around(shape, ax);
	const y = new Float64Array(sizeOf(shape));
	let off = 0;
	for (let j = 0; j < xs.length; j++) {
		const d = xs[j].data;
		const len = sizes[j] * inner;
		for (let o = 0; o < outer; o++) y.set(d.subarray(o * len, (o + 1) * len), o * total * inner + off * inner);
		off += sizes[j];
	}
	return result('concat', y, shape, xs.slice(), (g, need) => {
		let start = 0;
		return xs.map((x, j) => {
			const s = start;
			start += sizes[j];
			return need[j] ? slice(g, ax, s, s + sizes[j]) : null;
		});
	});
}

/** Stack equal-shaped tensors along a new axis. */
export function stack(xs: Var[], axis = 0): Var {
	const r = xs[0].shape.length + 1;
	const ax = K.normAxis(axis, r);
	return concat(
		xs.map((x) => reshape(x, [...x.shape.slice(0, ax), 1, ...x.shape.slice(ax)])),
		ax
	);
}

/** x[..., start:end, ...] along `axis` (negative indices count from the end). */
export function slice(x: Var, axis: number, start: number, end?: number): Var {
	const ax = K.normAxis(axis, x.shape.length);
	const n = x.shape[ax];
	const s0 = start < 0 ? start + n : start;
	const e0 = end === undefined ? n : end < 0 ? end + n : end;
	const s = Math.max(0, Math.min(n, s0));
	const e = Math.max(s, Math.min(n, e0));
	const [outer, , inner] = K.around(x.shape, ax);
	const len = e - s;
	const shape = x.shape.slice();
	shape[ax] = len;
	const y = new Float64Array(outer * len * inner);
	for (let o = 0; o < outer; o++) y.set(x.data.subarray((o * n + s) * inner, (o * n + e) * inner), o * len * inner);
	return result('slice', y, shape, [x], (g) => [padSlice(g, x.shape, ax, s)]);
}

/** Zeros of `shape` with g placed at [start, start + g.shape[axis]) along axis (the adjoint of slice). */
function padSlice(g: Var, shape: number[], axis: number, start: number): Var {
	const n = shape[axis];
	const len = g.shape[axis];
	const [outer, , inner] = K.around(shape, axis);
	const y = new Float64Array(sizeOf(shape));
	for (let o = 0; o < outer; o++) y.set(g.data.subarray(o * len * inner, (o + 1) * len * inner), (o * n + start) * inner);
	return result('padSlice', y, shape.slice(), [g], (gg) => [slice(gg, axis, start, start + len)]);
}

/** Pick entries `index` along `axis` (PyTorch index_select; repeated indices are allowed). */
export function gather(x: Var, axis: number, index: ArrayLike<number>): Var {
	const ax = K.normAxis(axis, x.shape.length);
	const n = x.shape[ax];
	const idx = Int32Array.from(index as ArrayLike<number>);
	for (const i of idx) if (i < 0 || i >= n) throw new Error(`gather: index ${i} out of range for size ${n}`);
	const [outer, , inner] = K.around(x.shape, ax);
	const m = idx.length;
	const shape = x.shape.slice();
	shape[ax] = m;
	const y = new Float64Array(outer * m * inner);
	for (let o = 0; o < outer; o++) {
		for (let j = 0; j < m; j++) y.set(x.data.subarray((o * n + idx[j]) * inner, (o * n + idx[j] + 1) * inner), (o * m + j) * inner);
	}
	return result('gather', y, shape, [x], (g) => [scatterAdd(g, ax, idx, n)]);
}

/** The adjoint of gather: zeros with size n along axis, g rows added at `index`. */
export function scatterAdd(g: Var, axis: number, index: ArrayLike<number>, n: number): Var {
	const ax = K.normAxis(axis, g.shape.length);
	const idx = Int32Array.from(index as ArrayLike<number>);
	const m = idx.length;
	const [outer, , inner] = K.around(g.shape, ax);
	const shape = g.shape.slice();
	shape[ax] = n;
	const y = new Float64Array(outer * n * inner);
	for (let o = 0; o < outer; o++) {
		for (let j = 0; j < m; j++) {
			const src = (o * m + j) * inner;
			const dst = (o * n + idx[j]) * inner;
			for (let i = 0; i < inner; i++) y[dst + i] += g.data[src + i];
		}
	}
	return result('scatterAdd', y, shape, [g], (gg) => [gather(gg, ax, idx)]);
}

// ------------------------------------------------------------------ softmax family

/** log Σ exp(x) along one axis. */
export function logsumexp(x: Var, axis = -1, keepdim = false): Var {
	const ax = K.normAxis(axis, x.shape.length);
	const keepShape = x.shape.map((d, i) => (i === ax ? 1 : d));
	const outShape = keepdim ? keepShape : x.shape.filter((_, i) => i !== ax);
	const y = K.logsumexp(x.data, x.shape, ax);
	const out: Var = result('logsumexp', y, outShape, [x], (g) => {
		if (!isGradEnabled()) {
			const [outer, n, inner] = K.around(x.shape, ax);
			const r = new Float64Array(x.size);
			for (let o = 0; o < outer; o++) {
				for (let i = 0; i < inner; i++) {
					const gi = g.data[o * inner + i];
					const yi = y[o * inner + i];
					for (let j = 0; j < n; j++) {
						const k = (o * n + j) * inner + i;
						r[k] = gi * Math.exp(x.data[k] - yi);
					}
				}
			}
			return [raw(r, x.shape.slice())];
		}
		const gk = keepdim ? g : reshape(g, keepShape);
		const yk = keepdim ? out : reshape(out, keepShape);
		return [mul(broadcastTo(gk, x.shape), exp(sub(x, yk)))];
	});
	return out;
}

export function softmax(x: Var, axis = -1): Var {
	const ax = K.normAxis(axis, x.shape.length);
	const y = K.softmax(x.data, x.shape, ax);
	const out: Var = result('softmax', y, x.shape.slice(), [x], (g) => {
		if (!isGradEnabled()) {
			const [outer, n, inner] = K.around(x.shape, ax);
			const r = new Float64Array(x.size);
			for (let o = 0; o < outer; o++) {
				for (let i = 0; i < inner; i++) {
					const base = o * n * inner + i;
					let dot = 0;
					for (let j = 0; j < n; j++) dot += g.data[base + j * inner] * y[base + j * inner];
					for (let j = 0; j < n; j++) r[base + j * inner] = y[base + j * inner] * (g.data[base + j * inner] - dot);
				}
			}
			return [raw(r, x.shape.slice())];
		}
		return [mul(out, sub(g, sum(mul(g, out), ax, true)))];
	});
	return out;
}

export function logSoftmax(x: Var, axis = -1): Var {
	const ax = K.normAxis(axis, x.shape.length);
	const y = K.logSoftmax(x.data, x.shape, ax);
	const out: Var = result('logSoftmax', y, x.shape.slice(), [x], (g) => {
		if (!isGradEnabled()) {
			const [outer, n, inner] = K.around(x.shape, ax);
			const r = new Float64Array(x.size);
			for (let o = 0; o < outer; o++) {
				for (let i = 0; i < inner; i++) {
					const base = o * n * inner + i;
					let s = 0;
					for (let j = 0; j < n; j++) s += g.data[base + j * inner];
					for (let j = 0; j < n; j++) r[base + j * inner] = g.data[base + j * inner] - Math.exp(y[base + j * inner]) * s;
				}
			}
			return [raw(r, x.shape.slice())];
		}
		return [sub(g, mul(exp(out), sum(g, ax, true)))];
	});
	return out;
}

// ------------------------------------------------------------------ conv1d

export interface Conv1dOpts {
	stride?: number;
	dilation?: number;
	/** zeros on both sides (ignored when causal) */
	padding?: number;
	/** pad dilation·(k−1) zeros on the left only, so output t sees inputs ≤ t and the length is kept (TCN) */
	causal?: boolean;
}

function convCore(x: Var, w: Var, c: K.Conv1dCfg): Var {
	return result('conv1d', K.conv1d(x.data, w.data, c), [c.N, c.O, c.Tout], [x, w], (g, need) => [
		need[0] ? convGradInput(g, w, c) : null,
		need[1] ? convGradWeight(g, x, c) : null
	]);
}

function convGradInput(g: Var, w: Var, c: K.Conv1dCfg): Var {
	return result('conv1dGradInput', K.conv1dGradInput(g.data, w.data, c), [c.N, c.C, c.T], [g, w], (gu, need) => [
		need[0] ? convCore(gu, w, c) : null,
		need[1] ? convGradWeight(g, gu, c) : null
	]);
}

function convGradWeight(g: Var, x: Var, c: K.Conv1dCfg): Var {
	return result('conv1dGradWeight', K.conv1dGradWeight(g.data, x.data, c), [c.O, c.C, c.K], [g, x], (gv, need) => [
		need[0] ? convCore(x, gv, c) : null,
		need[1] ? convGradInput(g, gv, c) : null
	]);
}

/** 1-D convolution (PyTorch conv1d). x [N, C, T] or [C, T]; w [O, C, k]; b [O] or null. */
export function conv1d(x: Var, w: Var, b?: Var | null, o: Conv1dOpts = {}): Var {
	const batched = x.shape.length === 3;
	if (!batched && x.shape.length !== 2) throw new Error(`conv1d: input must be [N, C, T] or [C, T], got [${x.shape}]`);
	const X = batched ? x : reshape(x, [1, ...x.shape]);
	const [N, C, T] = X.shape;
	if (w.shape.length !== 3 || w.shape[1] !== C) throw new Error(`conv1d: weight must be [O, ${C}, k], got [${w.shape}]`);
	const [O, , Kk] = w.shape;
	const stride = o.stride ?? 1;
	const dilation = o.dilation ?? 1;
	const span = dilation * (Kk - 1);
	const padL = o.causal ? span : (o.padding ?? 0);
	const padR = o.causal ? 0 : (o.padding ?? 0);
	const Tout = Math.floor((T + padL + padR - span - 1) / stride) + 1;
	if (Tout < 1) throw new Error(`conv1d: input length ${T} is too short for kernel ${Kk} with dilation ${dilation}`);
	const cfg: K.Conv1dCfg = { N, C, T, O, K: Kk, stride, dilation, padL, Tout };
	let y = convCore(X, w, cfg);
	if (b) y = add(y, reshape(b, [O, 1]));
	return batched ? y : reshape(y, [O, Tout]);
}

// ------------------------------------------------------------------ normalization

/** LayerNorm over the last axis: (x − mean) / sqrt(var + eps) · w + b, biased variance (PyTorch). */
export function layerNorm(x: Var, w?: Var | null, b?: Var | null, eps = 1e-5): Var {
	const mu = mean(x, -1, true);
	const xc = sub(x, mu);
	const v = mean(mul(xc, xc), -1, true);
	const rstd = div(1, sqrt(add(v, eps)));
	let y = mul(xc, rstd);
	if (w) y = mul(y, w);
	if (b) y = add(y, b);
	return y;
}

/** The op namespace of design §4.5: op.add, op.matmul, … */
export const op = {
	add,
	sub,
	mul,
	div,
	neg,
	scale,
	pow,
	square,
	sqrt,
	exp,
	log,
	tanh,
	sigmoid,
	relu,
	silu,
	gelu,
	erf,
	sin,
	cos,
	abs,
	matmul,
	linear,
	sum,
	mean,
	reshape,
	transpose,
	broadcastTo,
	sumTo,
	concat,
	stack,
	slice,
	gather,
	scatterAdd,
	logsumexp,
	softmax,
	logSoftmax,
	conv1d,
	layerNorm
};
