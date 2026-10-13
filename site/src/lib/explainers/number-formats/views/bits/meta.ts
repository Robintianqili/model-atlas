// View 'number-formats/bits': one number x in several formats, bit by bit (process 'formats': x is the frame's metric x,
// the codes are its state 'codes'). Each row: the format and its bit split, the bits coloured by role (sign, exponent,
// fraction; an integer code; an NF4 index), the value stored, the error, the gap to the neighbour farther from 0 and what
// happened (exact, rounded, subnormal, underflow, overflow, NaN, saturated, clipped). Props:
//   formats?: FmtId[]   the rows (default: all nine)
//   detail?: FmtId      a worked line for that format: sign, exponent and fraction (or code) turned into the value, and
//                       a ruler with the stored value, its two neighbours and x
//   flip?: boolean      the bits are buttons: a press flips that bit and sets x to the new value (needs setInput; a
//                       pattern that means infinity or NaN leaves x as it is and says so)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtInt, minus } from '../../../../atlas/core/format.ts';
import { decode, FMT_IDS, FMTS, rowOf, type FmtId, type Row, type Status } from '../../fmt.ts';

export const NAMES: Record<FmtId, I18n> = {
	fp32: { en: 'fp32', zh: 'fp32' },
	fp16: { en: 'fp16', zh: 'fp16' },
	bf16: { en: 'bf16', zh: 'bf16' },
	e4m3: { en: 'FP8 e4m3', zh: 'FP8 e4m3' },
	e5m2: { en: 'FP8 e5m2', zh: 'FP8 e5m2' },
	fp4: { en: 'FP4 e2m1', zh: 'FP4 e2m1' },
	int8: { en: 'int8 (s = 1/127)', zh: 'int8（s = 1/127）' },
	int4: { en: 'int4 (s = 1/7)', zh: 'int4（s = 1/7）' },
	nf4: { en: 'NF4 (scale 1)', zh: 'NF4（缩放 1）' }
};
/** The bit split of a format: "1 + 8 + 23", "8-bit code", "4-bit index". */
export function splitText(id: FmtId): I18n {
	const f = FMTS[id];
	if (f.kind === 'float') return { en: `1 + ${f.e} + ${f.m} bits`, zh: `1 + ${f.e} + ${f.m} 位` };
	if (f.kind === 'int') return { en: `${f.bits}-bit code`, zh: `${f.bits} 位编码` };
	return { en: '4-bit index', zh: '4 位索引' };
}
export const STATUS: Record<Status, I18n> = {
	exact: { en: 'exact', zh: '精确' },
	rounded: { en: 'rounded', zh: '舍入' },
	subnormal: { en: 'subnormal', zh: '次正规数' },
	underflow: { en: 'underflow to 0', zh: '下溢为 0' },
	overflow: { en: 'overflow to ∞', zh: '上溢为 ∞' },
	nan: { en: 'NaN (no ∞ in e4m3)', zh: 'NaN（e4m3 没有 ∞）' },
	saturated: { en: 'stuck at the largest', zh: '停在最大值' },
	clipped: { en: 'clipped to the range', zh: '截断到范围端点' }
};
/** Significant digits shown for a format's values: enough to tell its neighbours apart. */
export const DIGITS: Record<FmtId, number> = { fp32: 9, fp16: 6, bf16: 4, e4m3: 3, e5m2: 3, fp4: 2, int8: 4, int4: 3, nf4: 4 };

/** A number as text, also when it is not finite; plain decimals between 1e−4 and 1e7, else "1.23e−5". */
export function show(v: number, digits = 6): string {
	if (Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
	if (v === 0) return Object.is(v, -0) ? '−0' : '0';
	const a = Math.abs(v);
	if (a >= 1e-4 && a < 1e7) return Number.isInteger(v) && a >= 1000 ? fmtInt(v) : minus(String(Number(v.toPrecision(digits))));
	const [mant, e] = Number(v.toPrecision(digits)).toExponential().split('e');
	return minus(`${mant}e${e.replace('+', '')}`);
}

export type BitsData = { x: number; rows: Row[] };
export function dataOf(p: Pick<StageProps, 'at' | 'props'>): BitsData | null {
	const x = p.at?.frame?.metrics?.x;
	const codes = (p.at?.state as { codes?: { data: ArrayLike<number> } } | undefined)?.codes?.data;
	if (typeof x !== 'number' || !codes) return null;
	const ids: FmtId[] = Array.isArray(p.props?.formats) ? (p.props.formats as FmtId[]) : FMT_IDS;
	return { x, rows: ids.map((id) => rowOf(id, x, codes[FMT_IDS.indexOf(id)])) };
}

/** The worked line of a row: how its bits make its value. */
export function workedText(r: Row): I18n {
	const f = FMTS[r.id];
	const v = show(r.value, DIGITS[r.id]);
	if (f.kind === 'float') {
		if (Number.isNaN(r.value)) return { en: 'This pattern is NaN (not a number).', zh: '这个位模式是 NaN（非数）。' };
		if (!Number.isFinite(r.value)) return { en: `Exponent bits all 1 and fraction 0: ${v}.`, zh: `指数位全是 1、尾数为 0：${v}。` };
		const sign = r.groups[0].bits === '1' ? '−' : '+';
		const E = r.E ?? 0;
		const frac = `${r.M} / 2${sup(f.m)}`;
		const sig = show(r.sig ?? NaN, 10);
		if (E === 0)
			return {
				en: `Sign ${sign}. Exponent field 0: a subnormal, 2${sup(1 - f.bias)} × (0 + ${frac}) = 2${sup(1 - f.bias)} × ${sig} = ${v}.`,
				zh: `符号 ${sign}。指数字段为 0：次正规数，2${sup(1 - f.bias)} × (0 + ${frac}) = 2${sup(1 - f.bias)} × ${sig} = ${v}。`
			};
		return {
			en: `Sign ${sign}. Exponent field ${E}, minus the bias ${f.bias}: 2${sup(E - f.bias)}. Fraction ${frac}, after the leading 1: ${sig}. Value ${sign}2${sup(E - f.bias)} × ${sig} = ${v}.`,
			zh: `符号 ${sign}。指数字段 ${E}，减去偏置 ${f.bias}：2${sup(E - f.bias)}。尾数 ${frac}，加上开头的 1：${sig}。值 ${sign}2${sup(E - f.bias)} × ${sig} = ${v}。`
		};
	}
	if (f.kind === 'int') return { en: `Code ${minus(String(r.q))} (two’s complement) × s = ${minus(String(r.q))} / ${2 ** (f.bits - 1) - 1} = ${v}.`, zh: `编码 ${minus(String(r.q))}（二进制补码）× s = ${minus(String(r.q))} / ${2 ** (f.bits - 1) - 1} = ${v}。` };
	return { en: `Index ${r.code} of 16 levels (counting from 0): level ${v}.`, zh: `16 个刻度中的第 ${r.code} 个（从 0 数起）：刻度值 ${v}。` };
}

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
export const sup = (n: number) => String(n).replace(/./g, (c) => SUP[c] ?? c);

/** The stored value's neighbours below and above (in value), NaN where there is none. */
export function neighbours(r: Row): [number, number] {
	const f = FMTS[r.id];
	if (!Number.isFinite(r.value)) return [NaN, NaN];
	if (f.kind === 'table') return [r.code > 0 ? decode(r.id, r.code - 1) : NaN, r.code < 15 ? decode(r.id, r.code + 1) : NaN];
	const n = r.total;
	const neg = r.value < 0 || Object.is(r.value, -0);
	if (f.kind === 'int') {
		const q = r.q ?? 0;
		const qmax = 2 ** (f.bits - 1) - 1;
		return [q - 1 >= -qmax ? (q - 1) / qmax : NaN, q + 1 <= qmax ? (q + 1) / qmax : NaN];
	}
	const mag = r.code % 2 ** (n - 1);
	const sign = r.code - mag;
	const out = mag + 1 < 2 ** (n - 1) ? decode(r.id, sign + mag + 1) : NaN;
	const inw = mag > 0 ? decode(r.id, sign + mag - 1) : r.value === 0 ? -decode(r.id, 1) : NaN;
	const o = Number.isFinite(out) ? out : NaN;
	return neg ? [o, inw] : [inw, o];
}

export function describeBits(d: BitsData | null): I18n {
	if (!d) return { en: 'No number yet.', zh: '还没有数。' };
	const en = d.rows.map((r) => `${NAMES[r.id].en}: bits ${r.groups.map((g) => g.bits).join(' ')}, stores ${show(r.value, DIGITS[r.id])} (${STATUS[r.status].en}), error ${show(r.err, 3)}`).join('; ');
	const zh = d.rows.map((r) => `${NAMES[r.id].zh}：各位 ${r.groups.map((g) => g.bits).join(' ')}，存成 ${show(r.value, DIGITS[r.id])}（${STATUS[r.status].zh}），误差 ${show(r.err, 3)}`).join('；');
	return { en: `The number ${show(d.x, 9)}. ${en}.`, zh: `数 ${show(d.x, 9)}。${zh}。` };
}

const meta = {
	key: 'number-formats/bits',
	title: { en: 'One number, bit by bit, in several formats', zh: '一个数在几种格式里的各位' },
	kinds: ['process'],
	describe: (p: StageProps) => describeBits(dataOf(p)),
	demo: async () => {
		const [{ record }, { formats }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(formats, 0.1, {});
		const rt = fakeRuntime({ slug: 'number-formats', input: { x: 0.1 }, tapes: { formats: tape } });
		return { rt, source: 'formats', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { detail: 'fp32' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
