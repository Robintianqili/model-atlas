// number-formats: the arithmetic of the concept page (no kit state; Node and the browser run the same code).
//   floats   fp32, fp16, bf16, fp8 e4m3 (OCP "e4m3fn": no infinity, one NaN code), fp8 e5m2 and fp4 e2m1 (no infinity,
//            no NaN): one generic minifloat with e exponent bits, m fraction bits and a bias. A number is first rounded
//            to fp32 (as PyTorch casts an fp32 tensor), then to the format: round to nearest, ties to even. Too large:
//            fp16, bf16 and e5m2 give infinity, e4m3 gives NaN (PyTorch's cast), fp4 saturates at ±6.
//   integers int8 and int4: a two's-complement code q stands for q × s, with the scale s = 1 / qmax (the range −1..1);
//            rounding is to the nearest code, clamped to ±qmax (symmetric absmax, as on the quantization page)
//   nf4      4-bit NormalFloat (QLoRA): a code is an index into 16 fixed levels between −1 and 1, quantiles of a normal
//            distribution (the table of bitsandbytes); rounding is to the nearest level
// pipeline/number-formats/reference.py checks every cast against PyTorch's own dtypes (fp16, bf16, e4m3fn, e5m2) and an
// independent enumeration (fp4, int8, int4, nf4).

export type FmtId = 'fp32' | 'fp16' | 'bf16' | 'e4m3' | 'e5m2' | 'fp4' | 'int8' | 'int4' | 'nf4';
export const FMT_IDS: FmtId[] = ['fp32', 'fp16', 'bf16', 'e4m3', 'e5m2', 'fp4', 'int8', 'int4', 'nf4'];
export const FLOAT_IDS: FmtId[] = ['fp32', 'fp16', 'bf16', 'e4m3', 'e5m2', 'fp4'];

export interface FloatFmt {
	id: FmtId;
	kind: 'float';
	e: number;
	m: number;
	bias: number;
	/** 'ieee': the all-ones exponent holds infinity and NaN; 'fn': only all-ones exponent and fraction is NaN, no
	 *  infinity (e4m3fn); 'none': every code is a finite number (fp4 e2m1) */
	special: 'ieee' | 'fn' | 'none';
}
export interface IntFmt {
	id: FmtId;
	kind: 'int';
	bits: number;
}
export interface TableFmt {
	id: FmtId;
	kind: 'table';
	bits: number;
	levels: number[];
}
export type Fmt = FloatFmt | IntFmt | TableFmt;

/** The 16 levels of NF4 (QLoRA, Dettmers et al. 2023; bitsandbytes, MIT licence), as float32 values. */
export const NF4 = [
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
	0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023,
	1.0
];

export const FMTS: Record<FmtId, Fmt> = {
	fp32: { id: 'fp32', kind: 'float', e: 8, m: 23, bias: 127, special: 'ieee' },
	fp16: { id: 'fp16', kind: 'float', e: 5, m: 10, bias: 15, special: 'ieee' },
	bf16: { id: 'bf16', kind: 'float', e: 8, m: 7, bias: 127, special: 'ieee' },
	e4m3: { id: 'e4m3', kind: 'float', e: 4, m: 3, bias: 7, special: 'fn' },
	e5m2: { id: 'e5m2', kind: 'float', e: 5, m: 2, bias: 15, special: 'ieee' },
	fp4: { id: 'fp4', kind: 'float', e: 2, m: 1, bias: 1, special: 'none' },
	int8: { id: 'int8', kind: 'int', bits: 8 },
	int4: { id: 'int4', kind: 'int', bits: 4 },
	nf4: { id: 'nf4', kind: 'table', bits: 4, levels: NF4 }
};

export const nbits = (f: Fmt) => (f.kind === 'float' ? 1 + f.e + f.m : f.bits);
export const qmaxOf = (bits: number) => 2 ** (bits - 1) - 1;
/** The integer formats' range is −INT_RANGE..INT_RANGE: the scale is s = INT_RANGE / qmax. */
export const INT_RANGE = 1;

/** The learner's input: one number. */
export type Raw = { x: number };
export const X_MAX = 3.4e38;
export const START: Raw = { x: 0.1 };
/** The fixed numbers of three steps: beyond fp16's range, a small weight, beyond e4m3's range. */
export const BIG_X = 70000;
export const TINY_X = 0.001;
export const HUGE_X = 500;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
/** The learner's input, made safe: a finite x within fp32's range (anything else becomes the start number). */
export function rawOf(raw: unknown): Raw {
	const x = typeof raw === 'number' ? raw : (raw as Partial<Raw> | null)?.x;
	return { x: typeof x === 'number' && Number.isFinite(x) ? clamp(x, -X_MAX, X_MAX) : START.x };
}

/** Round to the nearest whole number, ties to even (as torch.round). */
export function rhe(v: number): number {
	const f = Math.floor(v);
	const d = v - f;
	if (d > 0.5) return f + 1;
	if (d < 0.5) return f;
	return f % 2 === 0 ? f : f + 1;
}

// ---------------------------------------------------------------- fp32 bits

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

// ---------------------------------------------------------------- encode and decode

const allOnes = (n: number) => 2 ** n - 1;

/** The code (bit pattern) of x in a float format. */
function encodeFloat(f: FloatFmt, x: number): number {
	const v = Math.fround(x);
	if (f.id === 'fp32') return fp32Bits(v);
	const EM = allOnes(f.e);
	const MM = allOnes(f.m);
	const sign = v < 0 || Object.is(v, -0) ? 2 ** (f.e + f.m) : 0;
	if (Number.isNaN(v)) return f.special === 'ieee' ? sign + EM * 2 ** f.m + 2 ** (f.m - 1) : f.special === 'fn' ? sign + EM * 2 ** f.m + MM : sign;
	const a = Math.abs(v);
	const sub = 2 ** (1 - f.bias - f.m);
	let E: number;
	let M: number;
	if (a < 2 ** (1 - f.bias)) {
		// subnormal: a multiple of the smallest step (a round up to 2^m is the smallest normal number, E = 1, M = 0)
		const q = rhe(a / sub);
		E = Math.floor(q / 2 ** f.m);
		M = q % 2 ** f.m;
	} else {
		let e = Math.floor(Math.log2(a));
		if (2 ** e > a) e--;
		if (2 ** (e + 1) <= a) e++;
		M = a === Infinity ? 0 : rhe((a / 2 ** e - 1) * 2 ** f.m);
		if (M === 2 ** f.m) {
			M = 0;
			e++;
		}
		E = a === Infinity ? EM + 1 : e + f.bias;
	}
	if (f.special === 'ieee' && E >= EM) return sign + EM * 2 ** f.m;
	if (f.special === 'fn' && (E > EM || (E === EM && M === MM))) return sign + EM * 2 ** f.m + MM;
	if (f.special === 'none' && E > EM) return sign + EM * 2 ** f.m + MM;
	return sign + E * 2 ** f.m + M;
}

/** The value of a code in a float format. */
function decodeFloat(f: FloatFmt, b: number): number {
	if (f.id === 'fp32') return fromFp32Bits(b);
	const EM = allOnes(f.e);
	const MM = allOnes(f.m);
	const s = Math.floor(b / 2 ** (f.e + f.m)) % 2 ? -1 : 1;
	const E = Math.floor(b / 2 ** f.m) % 2 ** f.e;
	const M = b % 2 ** f.m;
	if (f.special === 'ieee' && E === EM) return M ? NaN : s * Infinity;
	if (f.special === 'fn' && E === EM && M === MM) return NaN;
	if (E === 0) return s * M * 2 ** (1 - f.bias - f.m);
	return s * (1 + M / 2 ** f.m) * 2 ** (E - f.bias);
}

/** The nearest NF4 level (the first of two equally near ones). */
export function nf4Index(v: number): number {
	let k = 0;
	for (let i = 1; i < NF4.length; i++) if (Math.abs(v - NF4[i]) < Math.abs(v - NF4[k])) k = i;
	return k;
}

/** The code of x in a format: the bit pattern as an unsigned number. */
export function encode(id: FmtId, x: number): number {
	const f = FMTS[id];
	if (f.kind === 'float') return encodeFloat(f, x);
	if (f.kind === 'int') {
		const qmax = qmaxOf(f.bits);
		const q = clamp(rhe((x / INT_RANGE) * qmax), -qmax, qmax) + 0;
		return q < 0 ? q + 2 ** f.bits : q;
	}
	return nf4Index(clamp(x / INT_RANGE, -1, 1));
}

/** The value a code stands for. */
export function decode(id: FmtId, code: number): number {
	const f = FMTS[id];
	if (f.kind === 'float') return decodeFloat(f, code);
	if (f.kind === 'int') {
		const q = code >= 2 ** (f.bits - 1) ? code - 2 ** f.bits : code;
		return (q * INT_RANGE) / qmaxOf(f.bits);
	}
	return f.levels[code] * INT_RANGE;
}

export const cast = (id: FmtId, x: number) => decode(id, encode(id, x));

/** The code with bit `bit` flipped (bit 0 is the leftmost, the sign of a float). */
export function flipBit(id: FmtId, code: number, bit: number): number {
	const n = nbits(FMTS[id]);
	const w = 2 ** (n - 1 - bit);
	return Math.floor(code / w) % 2 ? code - w : code + w;
}

// ---------------------------------------------------------------- one number in a format

export type Role = 'sign' | 'exp' | 'frac' | 'int' | 'idx';
export type Status = 'exact' | 'rounded' | 'subnormal' | 'underflow' | 'overflow' | 'nan' | 'saturated' | 'clipped';

export interface Row {
	id: FmtId;
	total: number;
	code: number;
	/** bit groups, most significant first: sign, exponent, fraction (floats); the code (int8, int4); the index (nf4) */
	groups: { role: Role; bits: string }[];
	value: number;
	err: number;
	/** the gap from the stored value to its neighbour farther from 0 (NaN when there is none: the largest, inf, NaN) */
	gap: number;
	status: Status;
	/** floats: the exponent field, the fraction field, the exponent in use (E − bias, or 1 − bias for a subnormal) and the
	 *  significand (1.f, or 0.f for a subnormal) */
	E?: number;
	M?: number;
	exp?: number;
	sig?: number;
	/** integers: the code as a signed number */
	q?: number;
}

const bitString = (code: number, n: number) => code.toString(2).padStart(n, '0');

/** The neighbour of a code farther from 0 (NaN when there is none). */
function outward(id: FmtId, code: number, value: number): number {
	const f = FMTS[id];
	if (!Number.isFinite(value)) return NaN;
	if (f.kind === 'float') {
		const n = nbits(f);
		const mag = code % 2 ** (n - 1);
		if (mag + 1 >= 2 ** (n - 1)) return NaN;
		const next = decode(id, code - mag + mag + 1);
		return Number.isFinite(next) ? Math.abs(next - value) : NaN;
	}
	if (f.kind === 'int') {
		const q = code >= 2 ** (f.bits - 1) ? code - 2 ** f.bits : code;
		const qmax = qmaxOf(f.bits);
		return Math.abs(q) >= qmax ? NaN : INT_RANGE / qmax;
	}
	const k = value >= 0 ? code + 1 : code - 1;
	return k >= 0 && k < NF4.length ? Math.abs(NF4[k] - NF4[code]) * INT_RANGE : NaN;
}

export function rowOf(id: FmtId, x: number, code = encode(id, x)): Row {
	const f = FMTS[id];
	const n = nbits(f);
	const bits = bitString(code, n);
	const value = decode(id, code);
	const max = largest(id);
	let status: Status = value === x ? 'exact' : 'rounded';
	const row: Row = { id, total: n, code, groups: [], value, err: value - x, gap: outward(id, code, value), status };
	if (f.kind === 'float') {
		row.groups = [
			{ role: 'sign', bits: bits.slice(0, 1) },
			{ role: 'exp', bits: bits.slice(1, 1 + f.e) },
			{ role: 'frac', bits: bits.slice(1 + f.e) }
		];
		const E = parseInt(bits.slice(1, 1 + f.e), 2);
		const M = parseInt(bits.slice(1 + f.e), 2);
		row.E = E;
		row.M = M;
		row.exp = E === 0 ? 1 - f.bias : E - f.bias;
		row.sig = (E === 0 ? 0 : 1) + M / 2 ** f.m;
		if (Number.isNaN(value)) status = 'nan';
		else if (!Number.isFinite(value)) status = 'overflow';
		else if (value === 0 && x !== 0) status = 'underflow';
		else if (f.special === 'none' && Math.abs(x) > max) status = 'saturated';
		else if (E === 0 && value !== 0) status = value === x ? 'exact' : 'subnormal';
	} else {
		row.groups = [{ role: f.kind === 'int' ? 'int' : 'idx', bits }];
		if (f.kind === 'int') row.q = code >= 2 ** (f.bits - 1) ? code - 2 ** f.bits : code;
		if (Math.abs(x) > INT_RANGE) status = 'clipped';
		else if (value === 0 && x !== 0) status = 'underflow';
	}
	row.status = status;
	return row;
}

export const rowsOf = (x: number, ids: FmtId[] = FMT_IDS) => ids.map((id) => rowOf(id, x));

// ---------------------------------------------------------------- range and precision of a format

/** The largest finite value. */
export function largest(id: FmtId): number {
	const f = FMTS[id];
	if (f.kind !== 'float') return INT_RANGE;
	const EM = allOnes(f.e);
	if (f.special === 'ieee') return (2 - 2 ** -f.m) * 2 ** (EM - 1 - f.bias);
	if (f.special === 'fn') return (2 - 2 ** (1 - f.m)) * 2 ** (EM - f.bias);
	return (2 - 2 ** -f.m) * 2 ** (EM - f.bias);
}
/** The smallest positive normal value (floats), or the smallest positive value (integer formats, NF4). */
export function smallestNormal(id: FmtId): number {
	const f = FMTS[id];
	if (f.kind === 'float') return 2 ** (1 - f.bias);
	if (f.kind === 'int') return INT_RANGE / qmaxOf(f.bits);
	return NF4[8] * INT_RANGE;
}
/** The smallest positive value (a subnormal for floats). */
export function smallest(id: FmtId): number {
	const f = FMTS[id];
	return f.kind === 'float' ? 2 ** (1 - f.bias - f.m) : smallestNormal(id);
}
/** The gap between 1 and the next value up (floats: 2^−m, machine epsilon); integers: the step s; NF4: NaN. */
export function gapAtOne(id: FmtId): number {
	const f = FMTS[id];
	if (f.kind === 'float') return 2 ** -f.m;
	if (f.kind === 'int') return INT_RANGE / qmaxOf(f.bits);
	return NaN;
}
/** How many different finite values the format can hold (−0 and +0 counted once). */
export function finiteCount(id: FmtId): number {
	const f = FMTS[id];
	if (f.kind !== 'float') return 2 ** f.bits; // int: −2^(bits−1)..2^(bits−1) − 1; nf4: 16 levels
	const n = nbits(f);
	// every code but the all-ones exponent (ieee: 2 · 2^m codes) or the two NaN codes (fn), minus one for −0
	return 2 ** n - (f.special === 'ieee' ? 2 * 2 ** f.m : f.special === 'fn' ? 2 : 0) - 1;
}

// ---------------------------------------------------------------- arrays of weights (float32 arithmetic)

export type Scale = 'none' | 'tensor' | 'block';
export interface CastCfg {
	id: string;
	fmt: FmtId;
	scale: Scale;
	/** values per block (scale 'block'), contiguous along a row */
	block?: number;
}
/** The largest value a scaled format maps the group's largest size to. */
export const scaleTarget = (id: FmtId) => (FMTS[id].kind === 'float' ? largest(id) : INT_RANGE);

/** Cast an array of float32 values to a format, in float32 arithmetic as PyTorch does: without a scale, out = cast(w);
 *  with one, s = fl(amax / target) per group, out = fl(cast(fl(w / s)) · s). `cols` is the row length (blocks never
 *  cross a row). */
export function castArray(w: ArrayLike<number>, cfg: CastCfg, cols = w.length): Float32Array {
	const out = new Float32Array(w.length);
	if (cfg.scale === 'none') {
		for (let i = 0; i < w.length; i++) out[i] = cast(cfg.fmt, w[i]);
		return out;
	}
	const target = scaleTarget(cfg.fmt);
	const group = (a: number, b: number) => {
		let amax = 0;
		for (let i = a; i < b; i++) amax = Math.max(amax, Math.abs(w[i]));
		const s = amax > 0 ? Math.fround(amax / target) : 1;
		for (let i = a; i < b; i++) out[i] = Math.fround(Math.fround(cast(cfg.fmt, Math.fround(w[i] / s))) * s);
	};
	if (cfg.scale === 'tensor') {
		group(0, w.length);
		return out;
	}
	const size = cfg.block ?? 32;
	for (let r = 0; r < w.length; r += cols) for (let c0 = 0; c0 < cols; c0 += size) group(r + c0, r + Math.min(cols, c0 + size));
	return out;
}

/** What happens to the values of an array in a float format without a scale: shares that stay normal, become
 *  subnormal, underflow to 0, or overflow (nonzero values only). */
export function fateOf(w: ArrayLike<number>, id: FmtId) {
	const lo = smallestNormal(id);
	const max = largest(id);
	let n = 0;
	let sub = 0;
	let zero = 0;
	let over = 0;
	for (let i = 0; i < w.length; i++) {
		if (w[i] === 0) continue;
		n++;
		const v = cast(id, w[i]);
		if (v === 0) zero++;
		else if (!Number.isFinite(v) || Math.abs(w[i]) > max) over++;
		else if (Math.abs(v) < lo) sub++;
	}
	return { n, sub: sub / n, zero: zero / n, over: over / n, normal: (n - sub - zero - over) / n };
}
