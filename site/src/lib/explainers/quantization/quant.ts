// quantization: the pure arithmetic of the concept page (no kit state; Node and the browser run the same code).
//   formats   one number in fp32, fp16 and bf16 (their bits, rounded to nearest, ties to even) and in int8 / int4 on a
//             fixed range −1..1 (a code and its two's-complement bits)
//   ledger    eight numbers rounded to int8 or int4: absmax (symmetric, one scale) or affine (a scale and a zero point),
//             with an optional clipping range and blocks of `block` values that each get their own scale
//   weights   a weight matrix quantized the same way per tensor, per output channel (row) or per block of 32 along a row,
//             in float32 arithmetic as PyTorch does (pipeline/quantization/qlib.py), for the TinyGPT sweep
// Rounding is to the nearest whole number with ties to even (torch.round, numpy, Python round()).

export const N = 8;
/** The learner's input: one number for the format widgets, and the eight numbers of the rounding ledger. */
export type Raw = { x: number; vals: number[] };
export const X_MAX = 3.4e38;
export const V_MAX = 100;

export const START: Raw = { x: 0.1, vals: [0.31, -0.12, 0.05, -0.47, 0.22, 0.09, -0.26, 0.4] };

const finite = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** The learner's input, made safe: x finite (within fp32's range), exactly eight finite values within ±100. */
export function rawOf(raw: unknown): Raw {
	const r = (raw ?? {}) as Partial<Raw>;
	const vals = Array.isArray(r.vals) ? r.vals : START.vals;
	return {
		x: clamp(finite(r.x, START.x), -X_MAX, X_MAX),
		vals: Array.from({ length: N }, (_, i) => clamp(finite(vals[i], START.vals[i] ?? 0), -V_MAX, V_MAX))
	};
}

/** Round to the nearest whole number, ties to even (as torch.round). */
export function rhe(v: number): number {
	const f = Math.floor(v);
	const d = v - f;
	if (d > 0.5) return f + 1;
	if (d < 0.5) return f;
	return f % 2 === 0 ? f : f + 1;
}

// ---------------------------------------------------------------- number formats

export type FormatId = 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4';
export interface FormatRow {
	id: FormatId;
	/** bit groups: sign, exponent, fraction for floats; the code's bits for integers (most significant first) */
	groups: { role: 'sign' | 'exp' | 'frac' | 'int'; bits: string }[];
	total: number;
	/** the value the format stores for x */
	value: number;
	err: number;
	/** integer formats: the code q */
	code?: number;
}

const bitsOf = (v: number, n: number) => (v >>> 0).toString(2).padStart(32, '0').slice(32 - n);

/** fp32 bits of a number (after rounding to fp32). */
export function fp32Bits(v: number): number {
	const dv = new DataView(new ArrayBuffer(4));
	dv.setFloat32(0, v);
	return dv.getUint32(0);
}
export function fromFp32Bits(b: number): number {
	const dv = new DataView(new ArrayBuffer(4));
	dv.setUint32(0, b >>> 0);
	return dv.getFloat32(0);
}

/** fp16 bits of an fp32 value: 1 sign, 5 exponent, 10 fraction bits, round to nearest even, overflow to infinity. */
export function fp16Bits(f: number): number {
	if (Number.isNaN(f)) return 0x7e00;
	const sign = f < 0 || Object.is(f, -0) ? 0x8000 : 0;
	const a = Math.abs(f);
	if (a === Infinity || a >= 65520) return sign | 0x7c00;
	if (a < 2 ** -14) return sign | rhe(a / 2 ** -24); // subnormal (1024 becomes the smallest normal, 0x0400)
	let e = Math.floor(Math.log2(a));
	if (2 ** e > a) e--;
	if (2 ** (e + 1) <= a) e++;
	const bits = ((e + 15) << 10) + rhe((a / 2 ** e - 1) * 1024);
	return sign | Math.min(bits, 0x7c00);
}
export function fromFp16Bits(b: number): number {
	const s = b & 0x8000 ? -1 : 1;
	const e = (b >> 10) & 0x1f;
	const m = b & 0x3ff;
	if (e === 0) return s * m * 2 ** -24;
	if (e === 31) return m ? NaN : s * Infinity;
	return s * (1 + m / 1024) * 2 ** (e - 15);
}
/** bf16 bits of an fp32 value: the top 16 bits of fp32 (1 sign, 8 exponent, 7 fraction), round to nearest even. */
export function bf16Bits(f: number): number {
	if (Number.isNaN(f)) return 0x7fc0;
	const b = fp32Bits(f);
	return ((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff;
}
export const fromBf16Bits = (b: number) => fromFp32Bits((b & 0xffff) << 16);

/** Integer formats on the fixed range −R..R: q = clamp(round(x / s), −qmax, qmax) with s = R / qmax. */
export const INT_RANGE = 1;
export const qmaxOf = (bits: number) => 2 ** (bits - 1) - 1;

/** One number in every format of the page. */
export function formatsOf(x: number): FormatRow[] {
	const v32 = Math.fround(x);
	const float = (id: FormatId, bits: number, n: number, ne: number, value: number): FormatRow => {
		const s = bitsOf(bits, n);
		return {
			id,
			groups: [
				{ role: 'sign', bits: s.slice(0, 1) },
				{ role: 'exp', bits: s.slice(1, 1 + ne) },
				{ role: 'frac', bits: s.slice(1 + ne) }
			],
			total: n,
			value,
			err: value - x
		};
	};
	const int = (id: FormatId, bits: number): FormatRow => {
		const qmax = qmaxOf(bits);
		const s = INT_RANGE / qmax;
		const q = clamp(rhe(x / s), -qmax, qmax);
		const value = q * s;
		return { id, groups: [{ role: 'int', bits: bitsOf(q & (2 ** bits - 1), bits) }], total: bits, value, err: value - x, code: q };
	};
	const b16 = fp16Bits(v32);
	const bbf = bf16Bits(v32);
	return [float('fp32', fp32Bits(v32), 32, 8, v32), float('fp16', b16, 16, 5, fromFp16Bits(b16)), float('bf16', bbf, 16, 8, fromBf16Bits(bbf)), int('int8', 8), int('int4', 4)];
}

/** x with one bit of its fp32 form flipped (bit 0 = the sign, counting from the left). */
export function flipFp32(x: number, bit: number): number {
	const b = fp32Bits(Math.fround(x));
	return fromFp32Bits(b ^ (1 << (31 - bit)));
}

// ---------------------------------------------------------------- the rounding ledger

export type Scheme = 'absmax' | 'affine';
export interface LedgerOpt {
	bits: number;
	scheme: Scheme;
	/** absmax only: 'none', or 'second' (clip at the second-largest |x| of the block) */
	clip: 'none' | 'second';
	/** values per block, each block with its own scale (8 = one scale for all) */
	block: number;
}
export const LEDGER_DEFAULTS: LedgerOpt = { bits: 8, scheme: 'absmax', clip: 'none', block: 8 };

export interface Block {
	from: number;
	to: number;
	s: number;
	z: number;
	/** the range the codes cover (absmax: ±amax; affine: lo..hi) */
	lo: number;
	hi: number;
	/** the smallest and largest code */
	qlo: number;
	qhi: number;
}
export interface Ledger {
	opt: LedgerOpt;
	x: number[];
	/** x / s (+ z), before rounding */
	r: number[];
	q: number[];
	xh: number[];
	err: number[];
	blocks: Block[];
	blockOf: number[];
	maxErr: number;
	meanErr: number;
	/** the values whose code was clamped to the range (clipped) */
	clipped: number[];
	/** how many nonzero values get the code of 0 (they are stored as 0) */
	zeros: number;
	/** bits per value, counting each block's scale (and affine zero point) as a 32-bit float */
	bitsPerValue: number;
}

function blockOf(x: number[], o: LedgerOpt): Omit<Block, 'from' | 'to'> {
	const levels = 2 ** o.bits;
	if (o.scheme === 'affine') {
		const lo = Math.min(0, ...x);
		const hi = Math.max(0, ...x);
		const s = hi > lo ? (hi - lo) / (levels - 1) : 1;
		const z = rhe(-lo / s) + 0; // + 0: no −0
		return { s, z, lo, hi, qlo: 0, qhi: levels - 1 };
	}
	const qmax = qmaxOf(o.bits);
	const mags = x.map(Math.abs).sort((a, b) => b - a);
	const amax = o.clip === 'second' && mags.length > 1 ? mags[1] : mags[0];
	// a range of size 0 (every value 0, or clipped at 0) has one code, 0; its scale is never used, s = 1
	if (!(amax > 0)) return { s: 1, z: 0, lo: 0, hi: 0, qlo: 0, qhi: 0 };
	return { s: amax / qmax, z: 0, lo: -amax, hi: amax, qlo: -qmax, qhi: qmax };
}

export function ledger(vals: number[], opt: Partial<LedgerOpt> = {}): Ledger {
	const o = { ...LEDGER_DEFAULTS, ...opt };
	const x = vals.slice();
	const size = Math.max(1, Math.min(x.length, Math.round(o.block)));
	const blocks: Block[] = [];
	const bOf: number[] = [];
	for (let from = 0; from < x.length; from += size) {
		const to = Math.min(x.length, from + size);
		blocks.push({ from, to, ...blockOf(x.slice(from, to), o) });
		for (let i = from; i < to; i++) bOf.push(blocks.length - 1);
	}
	const r: number[] = [];
	const q: number[] = [];
	const xh: number[] = [];
	const err: number[] = [];
	const clipped: number[] = [];
	let zeros = 0;
	x.forEach((v, i) => {
		const b = blocks[bOf[i]];
		const ri = v / b.s + b.z;
		const qi = clamp(rhe(ri), b.qlo, b.qhi);
		const back = (qi - b.z) * b.s;
		r.push(ri);
		q.push(qi);
		xh.push(back);
		err.push(back - v);
		if (rhe(ri) !== qi) clipped.push(i);
		if (v !== 0 && qi === b.z) zeros++;
	});
	const abs = err.map(Math.abs);
	const extra = o.scheme === 'affine' ? 2 : 1;
	return {
		opt: o,
		x,
		r,
		q,
		xh,
		err,
		blocks,
		blockOf: bOf,
		maxErr: Math.max(...abs),
		meanErr: abs.reduce((a, b) => a + b, 0) / abs.length,
		clipped,
		zeros,
		bitsPerValue: o.bits + (32 * extra * blocks.length) / x.length
	};
}

/** The index of the value with the largest |x| (ties to the smallest index). */
export function biggest(vals: number[]): number {
	let k = 0;
	for (let i = 1; i < vals.length; i++) if (Math.abs(vals[i]) > Math.abs(vals[k])) k = i;
	return k;
}
/** The median of |x|. */
export function medianAbs(vals: number[]): number {
	const m = vals.map(Math.abs).sort((a, b) => a - b);
	return (m[(m.length - 1) >> 1] + m[m.length >> 1]) / 2;
}
/** An outlier: the largest |x| is at least OUTLIER times the median |x|. */
export const OUTLIER = 8;
export const hasOutlier = (vals: number[]) => Math.abs(vals[biggest(vals)]) >= OUTLIER * Math.max(medianAbs(vals), 1e-9);
/** The learner's values with an outlier: unchanged when they have one, else the largest value moved out to
 *  OUTLIER_MAKE times the median |x| (keeping its sign; 1 when every value is 0). */
export const OUTLIER_MAKE = 12;
export function withOutlier(vals: number[]): number[] {
	if (hasOutlier(vals)) return vals.slice();
	const k = biggest(vals);
	const out = vals.slice();
	const m = medianAbs(vals);
	out[k] = (vals[k] < 0 ? -1 : 1) * (m > 0 ? OUTLIER_MAKE * m : 1);
	return out;
}
export const absAll = (vals: number[]) => vals.map((v) => Math.abs(v));

// ---------------------------------------------------------------- weight matrices (float32 arithmetic)

export type Gran = 'tensor' | 'channel' | 'block';
export const GRANS: Gran[] = ['tensor', 'channel', 'block'];
export const WEIGHT_BLOCK = 32;

/** Groups of a [rows, cols] matrix: one for the tensor, one per row, or blocks of WEIGHT_BLOCK along each row. */
export function groupCount(rows: number, cols: number, gran: Gran): number {
	return gran === 'tensor' ? 1 : gran === 'channel' ? rows : rows * Math.ceil(cols / WEIGHT_BLOCK);
}

/** Symmetric absmax quantization of a row-major [rows, cols] float32 matrix, dequantized (q·s per weight), in float32
 *  arithmetic: s = fl(amax / qmax), q = clamp(round(fl(w / s))), back = fl(q·s). */
export function quantizeMatrix(w: ArrayLike<number>, rows: number, cols: number, bits: number, gran: Gran): Float32Array {
	const qmax = qmaxOf(bits);
	const out = new Float32Array(rows * cols);
	const group = (a: number, b: number) => {
		let amax = 0;
		for (let i = a; i < b; i++) amax = Math.max(amax, Math.abs(w[i]));
		const s = amax > 0 ? Math.fround(amax / qmax) : 1;
		for (let i = a; i < b; i++) out[i] = clamp(rhe(Math.fround(w[i] / s)), -qmax, qmax) * s;
	};
	if (gran === 'tensor') {
		group(0, rows * cols);
		return out;
	}
	const step = gran === 'channel' ? cols : WEIGHT_BLOCK;
	for (let r = 0; r < rows; r++) for (let c0 = 0; c0 < cols; c0 += step) group(r * cols + c0, r * cols + Math.min(cols, c0 + step));
	return out;
}
