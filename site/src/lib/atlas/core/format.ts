// Number and shape formatting for narration and views (design §3.1 core/format.ts). Pure functions.
// Same output as kit/format.ts (atlas never imports the kit), plus the narration formats of design §4.6:
//   n3 (3 significant digits; default) | n2 n4 n5 n6 (that many significant digits) | int | pct | signed | shape | label | tok
//   K4: pct2 (a share with 2 decimals: 0.99983 -> "99.98%", where pct says "100.0%") | sci (scientific notation as the
//   formula cards print it: 1.71 × 10⁻⁴, not the calculator's 1.71e−4; char-rnn and crnn-ocr each wrote their own)
// Every minus sign is the typographic minus U+2212, in the mantissa and in the exponent ("−7.70e−4"); kit/format.ts
// (the legacy pages) still replaces only the first '-' (integration "kit fixes before wave 2a").
import type { Dim } from '../types/index.ts';

const MINUS = '−'; // typographic minus, as the kit

/** Every ASCII hyphen-minus of a formatted number becomes U+2212 (the sign and the exponent's sign alike). */
export const minus = (s: string): string => s.replace(/-/g, MINUS);

/** 3 significant digits (or `digits`), compact: 0.0123 -> "0.0123", 12345 -> "12300", 123456 -> "1.23e5", -0.5 -> "−0.500",
 *  -0.00077 -> "−7.70e−4". */
export function fmt3(v: number | null | undefined, digits = 3, list?: unknown): string {
	// K1: as a map callback (`.map(fmt3)`) the index arrives as digits (toPrecision(0) threw inside a view) and the array
	// third: then the default; otherwise only 1-21 digits count
	if (Array.isArray(list) || ArrayBuffer.isView(list) || !(Number.isInteger(digits) && digits >= 1 && digits <= 21)) digits = 3;
	if (v == null || Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
	if (v === 0) return '0';
	const a = Math.abs(v);
	let s: string;
	if (a >= 1e5 || a < 1e-3) {
		const [m, e] = v.toExponential(digits - 1).split('e');
		s = `${m}e${e.replace('+', '')}`;
	} else {
		s = v.toPrecision(digits);
		if (s.includes('e')) s = String(Number(s));
	}
	return minus(s);
}

/** Full precision (float32 needs ~9 significant digits to round-trip). */
export function fmtFull(v: number | null | undefined): string {
	if (v == null || Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
	return minus(String(Number(v.toPrecision(9))));
}

/** Integer with thousands separators: 6722 -> "6,722". */
export function fmtInt(n: number): string {
	if (!Number.isFinite(n)) return fmt3(n);
	return minus(Math.round(n).toLocaleString('en-US'));
}

/** K2: a count as itself ("50", "6,722"), any other number with 3 significant digits (the narration's default format):
 *  step axes and readouts, where fmt3 printed "step 50.0". */
export function fmtNum(v: number | null | undefined): string {
	return v != null && Number.isInteger(v) && Math.abs(v) < 1e15 ? fmtInt(v) : fmt3(v);
}

/** K2: a setting as the learner or the author set it (a learning rate 0.05, a noise 0.5, 3e−4): its shortest exact form up
 *  to 6 significant digits, never padded to 3 ("0.0500"), else 3 significant digits. For formula slots and narration of
 *  options (a PhaseDef.formula.values entry may be this string). */
export function fmtSetting(v: number | null | undefined): string {
	if (v == null || !Number.isFinite(v)) return fmt3(v);
	const short = Number(v.toPrecision(6));
	if (short !== v) return fmt3(v);
	const s = String(v);
	return minus(s.includes('e') ? s.replace('e+', 'e') : s);
}

/** A negative number in parentheses, for a product ("0.500 × (−0.0566)"); `f` formats it (default fmt3). K2. */
export function paren(v: number, f: (x: number) => string = fmt3): string {
	return v < 0 || Object.is(v, -0) ? `(${f(v)})` : f(v);
}

/** Percent with sensible digits: 0.8734 -> "87.3%". */
export function fmtPct(p: number, digits = 1): string {
	if (!Number.isFinite(p)) return '—';
	const v = p * 100;
	if (v !== 0 && Math.abs(v) < 0.1) return `${v < 0 ? MINUS : ''}<0.1%`;
	return `${minus(v.toFixed(digits))}%`;
}

/** K4: scientific notation as a person writes it: 0.000171 -> "1.71 × 10⁻⁴", 123456 -> "1.23 × 10⁵"; numbers from 0.001 to
 *  99,999 print as fmt3 does ("0.0123", "457"). `digits` significant digits (default 3). */
export function fmtSci(v: number | null | undefined, digits = 3): string {
	if (!(Number.isInteger(digits) && digits >= 1 && digits <= 21)) digits = 3;
	if (v == null || !Number.isFinite(v) || v === 0) return fmt3(v, digits);
	const a = Math.abs(v);
	if (a >= 1e-3 && a < 1e5) return fmt3(v, digits);
	const [m, e] = v.toExponential(digits - 1).split('e');
	const sup = (s: string) => s.replace(/^\+/, '').replace(/[-0-9]/g, (c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]);
	return `${minus(m)} × 10${sup(e)}`;
}

/** 3 significant digits with an explicit sign: "+0.250", "−1.20". */
export function fmtSigned(v: number): string {
	const s = fmt3(v);
	return v > 0 ? `+${s}` : s;
}

/** Shape label: [8, 26, 26] -> "[8×26×26]". Symbolic dims are kept: ['T', 64] -> "[T×64]". */
export function fmtShape(shape: ReadonlyArray<Dim>): string {
	return `[${shape.join('×')}]`;
}

/** Shape without brackets, for running text: "8×26×26". */
export function shapeText(shape: ReadonlyArray<Dim>): string {
	return shape.join('×');
}

/** Bytes as KB/MB with 3 significant digits (1 KB = 1000 B). */
export function fmtBytes(b: number): string {
	if (b < 1000) return `${b} B`;
	if (b < 1e6) return `${Number((b / 1e3).toPrecision(3))} KB`;
	if (b < 1e9) return `${Number((b / 1e6).toPrecision(3))} MB`;
	return `${Number((b / 1e9).toPrecision(3))} GB`;
}

/** Make whitespace visible in token labels: space -> "␣", newline -> "↵", tab -> "⇥". */
export function visibleToken(s: string): string {
	if (s === '') return '∅';
	let out = '';
	for (const ch of s) {
		const c = ch.codePointAt(0) ?? 0;
		if (ch === ' ') out += '␣';
		else if (ch === '\n') out += '↵';
		else if (ch === '\t') out += '⇥';
		else if (ch === '\r') out += '␍';
		else if (c < 32 || c === 127) out += `\\x${c.toString(16).padStart(2, '0')}`;
		else out += ch;
	}
	return out;
}

export type NarrationFmt = 'n2' | 'n3' | 'n4' | 'n5' | 'n6' | 'int' | 'pct' | 'pct2' | 'sci' | 'signed' | 'shape' | 'label' | 'tok';

/** The significant digits of an `nK` format (n2..n6), else null. */
export function sigDigits(fmt: string | undefined): number | null {
	const m = fmt ? /^n([2-6])$/.exec(fmt) : null;
	return m ? Number(m[1]) : null;
}

/** Format one narration value (design §4.6). Strings pass through (shape strings, labels); `tok` makes whitespace visible. */
export function fmtValue(v: number | string, fmt: string | undefined): string {
	if (typeof v === 'string') return fmt === 'tok' ? visibleToken(v) : v;
	const k = sigDigits(fmt);
	if (k !== null) return fmt3(v, k);
	switch (fmt) {
		case 'int':
			return fmtInt(v);
		case 'pct':
			return fmtPct(v);
		case 'pct2':
			return fmtPct(v, 2);
		case 'sci':
			return fmtSci(v);
		case 'signed':
			return fmtSigned(v);
		case undefined:
		case 'n3':
		case 'shape':
		case 'label':
		case 'tok':
		default:
			return fmt3(v);
	}
}
