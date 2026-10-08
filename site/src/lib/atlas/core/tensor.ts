// Tensor helpers for the DevKit runtime (design §3.1 core/). Pure TS: no DOM, no Node APIs.
// Tensors are CONTRACTS tensors: row-major `{ shape, data }`. Ops store float32 unless the
// first float input is a Float64Array (training toys, `precision: 'f64'` specs).
import type { AnyTensor, Dim, FArray, IntTensor, Tensor } from '../types/index.ts';

/** Number of elements of a numeric shape (1 for a scalar []). */
export function numel(shape: readonly number[]): number {
	let n = 1;
	for (const d of shape) n *= d;
	return n;
}

/** Row-major strides: strides([2, 3, 4]) = [12, 4, 1]. */
export function strides(shape: readonly number[]): number[] {
	const s = new Array<number>(shape.length);
	let acc = 1;
	for (let i = shape.length - 1; i >= 0; i--) {
		s[i] = acc;
		acc *= shape[i];
	}
	return s;
}

/** Flat index of a multi-index. Throws a RangeError when it is outside the shape. */
export function flatIndex(shape: readonly number[], idx: readonly number[]): number {
	if (idx.length !== shape.length) throw new RangeError(`index [${idx}] has rank ${idx.length}, tensor has rank ${shape.length} [${shape}]`);
	let f = 0;
	for (let i = 0; i < shape.length; i++) {
		const v = idx[i];
		if (!Number.isInteger(v) || v < 0 || v >= shape[i]) throw new RangeError(`index [${idx}] is outside shape [${shape}]`);
		f = f * shape[i] + v;
	}
	return f;
}

/** Multi-index of a flat index. */
export function unravel(shape: readonly number[], flat: number): number[] {
	const out = new Array<number>(shape.length);
	let r = flat;
	for (let i = shape.length - 1; i >= 0; i--) {
		const d = shape[i];
		out[i] = r % d;
		r = Math.floor(r / d);
	}
	return out;
}

/** True for an `{ dtype: 'i32' }` integer tensor (token ids). */
export function isIntTensor(t: AnyTensor): t is IntTensor {
	return (t as IntTensor).dtype === 'i32';
}

/** A new float array of length n with the same precision as `like` (Float32Array when `like` is absent or integer). */
export function alloc(n: number, like?: ArrayLike<number> | null): FArray {
	return like instanceof Float64Array ? new Float64Array(n) : new Float32Array(n);
}

/** A float tensor; copies `values` when given. */
export function tensor(shape: number[], values?: ArrayLike<number>, f64 = false): Tensor {
	const n = numel(shape);
	const data = f64 ? new Float64Array(n) : new Float32Array(n);
	if (values) {
		if (values.length !== n) throw new RangeError(`tensor: ${values.length} values for shape [${shape}] (${n})`);
		data.set(values);
	}
	return { shape: shape.slice(), data };
}

/** An int32 tensor (token ids, node ids). */
export function intTensor(shape: number[], values: ArrayLike<number>): IntTensor {
	const n = numel(shape);
	if (values.length !== n) throw new RangeError(`intTensor: ${values.length} values for shape [${shape}] (${n})`);
	return { shape: shape.slice(), data: Int32Array.from(values), dtype: 'i32' };
}

/** A float copy of any tensor (ids become exact floats). */
export function toFloat(t: AnyTensor, f64 = false): Tensor {
	return { shape: t.shape.slice(), data: f64 ? Float64Array.from(t.data) : Float32Array.from(t.data) };
}

export function cloneTensor(t: Tensor): Tensor {
	return { shape: t.shape.slice(), data: t.data.slice() as FArray };
}

export function shapeEq(a: readonly Dim[], b: readonly Dim[]): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

/** Replace symbolic dims with their values. Throws when a symbol has no value. */
export function resolveShape(shape: readonly Dim[], dims: Record<string, number>): number[] {
	return shape.map((d) => {
		if (typeof d === 'number') return d;
		const v = dims[d];
		if (typeof v !== 'number') throw new Error(`symbolic dim '${d}' has no value (known: ${Object.keys(dims).join(', ') || 'none'})`);
		return v;
	});
}

/** Normalize a possibly negative axis. */
export function normAxis(axis: number, rank: number): number {
	const a = axis < 0 ? axis + rank : axis;
	if (!Number.isInteger(a) || a < 0 || a >= rank) throw new RangeError(`axis ${axis} is outside rank ${rank}`);
	return a;
}

/** Product of a shape with symbolic dims: a number when all dims are numeric, else the joined string ("T×76"). */
export function sizeOf(shape: readonly Dim[]): number | string {
	return shape.every((d) => typeof d === 'number') ? numel(shape as number[]) : shape.join('×');
}

/** Index of the largest value (first one on ties); -1 for an empty range. */
export function argmax(a: ArrayLike<number>, from = 0, to = a.length): number {
	let best = -1;
	let m = -Infinity;
	for (let i = from; i < to; i++) {
		if (best < 0 || a[i] > m) {
			m = a[i];
			best = i;
		}
	}
	return best;
}

/** Index of the smallest value (first one on ties); -1 for an empty range. */
export function argmin(a: ArrayLike<number>, from = 0, to = a.length): number {
	let best = -1;
	let m = Infinity;
	for (let i = from; i < to; i++) {
		if (best < 0 || a[i] < m) {
			m = a[i];
			best = i;
		}
	}
	return best;
}

/** Index of the largest |value| (first one on ties); -1 for an empty range. */
export function argmaxAbs(a: ArrayLike<number>, from = 0, to = a.length): number {
	let best = -1;
	let m = -1;
	for (let i = from; i < to; i++) {
		const v = Math.abs(a[i]);
		if (best < 0 || v > m) {
			m = v;
			best = i;
		}
	}
	return best;
}

/** Read one value by multi-index. */
export function at(t: AnyTensor, idx: readonly number[]): number {
	return t.data[flatIndex(t.shape, idx)];
}
