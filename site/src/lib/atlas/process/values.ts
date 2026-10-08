// Value helpers shared by the tape layer: deep copies, applying a Change, and a stable content hash.
// Pure TS (Node, workers, browser).
import type { Change } from '../types/process.ts';

type AnyArray = ArrayLike<number> & { [i: number]: number };

export function isTypedArray(v: unknown): v is Float32Array | Float64Array | Int32Array | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array {
	return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/** Deep copy of tape values: typed arrays, arrays, plain objects, primitives. Functions are dropped. */
export function clone<T>(v: T): T {
	if (v === null || typeof v !== 'object') return (typeof v === 'function' ? undefined : v) as T;
	if (isTypedArray(v)) return (v as unknown as { slice(): unknown }).slice() as T;
	if (Array.isArray(v)) return v.map((x) => clone(x)) as T;
	const out: Record<string, unknown> = {};
	for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
		if (typeof x === 'function') continue;
		out[k] = clone(x);
	}
	return out as T;
}

/** Row-major flat index of `index` in `shape` (missing leading dims count as 0). */
export function flatIndex(shape: number[], index: number[]): number {
	let i = 0;
	const off = shape.length - index.length;
	for (let d = 0; d < shape.length; d++) i = i * shape[d] + (d >= off ? index[d - off] : 0);
	return i;
}

/** Apply one Change with a `to` value to a running state object (mutates `state`). Returns false when the target
 *  could not be found (the caller decides whether that is an error). Changes without `to` are notes and are ignored. */
export function applyChange(state: Record<string, unknown>, c: Change): boolean {
	if (c.to === undefined) return true;
	if (!(c.field in state) && c.key === undefined && !c.index) {
		state[c.field] = c.to;
		return true;
	}
	let target: unknown = state[c.field];
	if (c.key !== undefined) {
		if (target === null || typeof target !== 'object') return false;
		if (!c.index) {
			(target as Record<string, unknown>)[c.key] = c.to;
			return true;
		}
		target = (target as Record<string, unknown>)[c.key];
	}
	if (!c.index) {
		state[c.field] = c.to;
		return true;
	}
	if (target === null || typeof target !== 'object') return false;
	// a tensor { shape, data }
	const t = target as { shape?: number[]; data?: AnyArray };
	if (t.data && Array.isArray(t.shape)) {
		const i = flatIndex(t.shape, c.index);
		if (i < 0 || i >= t.data.length) return false;
		t.data[i] = c.to;
		return true;
	}
	// a typed array or a (nested) plain array
	let arr = target as unknown as AnyArray | unknown[];
	for (let d = 0; d < c.index.length - 1; d++) {
		arr = (arr as unknown[])[c.index[d]] as AnyArray | unknown[];
		if (arr == null) return false;
	}
	const last = c.index[c.index.length - 1];
	if (last < 0 || last >= (arr as AnyArray).length) return false;
	(arr as AnyArray)[last] = c.to;
	return true;
}

// ------------------------------------------------------------------ stable hash (two 32-bit FNV-1a lanes)
export class Hasher {
	private a = 0x811c9dc5;
	private b = 0x01000193 ^ 0x5bd1e995;
	private readonly f64 = new Float64Array(1);
	private readonly f64b = new Uint8Array(this.f64.buffer);

	byte(x: number): void {
		this.a = Math.imul(this.a ^ (x & 0xff), 0x01000193) >>> 0;
		this.b = Math.imul(this.b ^ (x & 0xff), 0x01000193 ^ 0x2f) >>> 0;
	}
	bytes(u: Uint8Array): void {
		for (let i = 0; i < u.length; i++) this.byte(u[i]);
	}
	tag(s: string): void {
		for (let i = 0; i < s.length; i++) {
			const c = s.charCodeAt(i);
			this.byte(c);
			this.byte(c >>> 8);
		}
		this.byte(0);
	}
	num(x: number): void {
		this.f64[0] = x;
		this.bytes(this.f64b);
	}
	value(v: unknown): void {
		if (v === undefined) return this.tag('u');
		if (v === null) return this.tag('n');
		switch (typeof v) {
			case 'number':
				this.tag('#');
				return this.num(v);
			case 'string':
				this.tag('s');
				return this.tag(v);
			case 'boolean':
				return this.tag(v ? 'T' : 'F');
			case 'function':
				return this.tag('f');
			case 'object':
				break;
			default:
				return this.tag(String(v));
		}
		if (isTypedArray(v)) {
			this.tag(`ta:${v.constructor.name}:${v.length}`);
			return this.bytes(new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
		}
		if (Array.isArray(v)) {
			this.tag(`a${v.length}`);
			for (const x of v) this.value(x);
			return;
		}
		const keys = Object.keys(v as object).sort();
		this.tag(`o${keys.length}`);
		for (const k of keys) {
			this.tag(k);
			this.value((v as Record<string, unknown>)[k]);
		}
	}
	hex(): string {
		return (this.a >>> 0).toString(16).padStart(8, '0') + (this.b >>> 0).toString(16).padStart(8, '0');
	}
}

export function hashValue(v: unknown): string {
	const h = new Hasher();
	h.value(v);
	return h.hex();
}

/** Deep equality for tape values (typed arrays compared element-wise; NaN equals NaN; -0 equals 0 is false). */
export function sameValue(a: unknown, b: unknown): boolean {
	return hashValue(a) === hashValue(b);
}
