/**
 * Number and shape formatting. Pure functions (safe in Node and workers).
 * Rule from the visual grammar: numbers show 3 significant digits; the full value is shown
 * on hover, tap or focus (see Num.svelte).
 */

const MINUS = '−'; // typographic minus

/** 3 significant digits, compact: 0.0123 -> "0.0123", 12345 -> "1.23e4", -0.5 -> "−0.500". */
export function fmt3(v: number | null | undefined, digits = 3, list?: unknown): string {
	// `.map(fmt3)` passes the index as digits and the array third (DevKit K1)
	if (Array.isArray(list) || ArrayBuffer.isView(list) || !(Number.isInteger(digits) && digits >= 1 && digits <= 21)) digits = 3;
	if (v == null || Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
	if (v === 0) return Object.is(v, -0) ? '0' : '0';
	const a = Math.abs(v);
	let s: string;
	if (a >= 1e5 || a < 1e-3) {
		const [m, e] = v.toExponential(digits - 1).split('e');
		s = `${m}e${e.replace('+', '')}`;
	} else {
		s = v.toPrecision(digits);
		// toPrecision may still produce exponent form for some values (e.g. 1.00e+3)
		if (s.includes('e')) s = String(Number(s));
	}
	return s.replace(/-/g, MINUS); // every minus, also in the exponent: "−7.70e−4" (DevKit K0)
}

/** Full precision (float32 needs ~9 significant digits to round-trip). */
export function fmtFull(v: number | null | undefined): string {
	if (v == null || Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
	return String(Number(v.toPrecision(9))).replace(/-/g, MINUS);
}

/** Integer with thousands separators: 6722 -> "6,722". */
export function fmtInt(n: number): string {
	return Math.round(n).toLocaleString('en-US');
}

/** Compact counts: 1234 -> "1.2k", 124441346 -> "124M". */
export function fmtCount(n: number): string {
	const a = Math.abs(n);
	if (a < 1000) return String(Math.round(n));
	if (a < 1e6) return `${trim(n / 1e3)}k`;
	if (a < 1e9) return `${trim(n / 1e6)}M`;
	return `${trim(n / 1e9)}B`;
}

function trim(x: number): string {
	return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1).replace(/\.0$/, '') : x.toFixed(2).replace(/\.?0+$/, '');
}

/** Bytes as KB/MB with 3 significant digits (1 KB = 1000 B, matching npm/HF size listings). */
export function fmtBytes(b: number): string {
	if (b < 1000) return `${b} B`;
	if (b < 1e6) return `${Number((b / 1e3).toPrecision(3))} KB`;
	if (b < 1e9) return `${Number((b / 1e6).toPrecision(3))} MB`;
	return `${Number((b / 1e9).toPrecision(3))} GB`;
}

/** Shape label: [8, 26, 26] -> "[8×26×26]". Strings are kept: ['T', 64] -> "[T×64]". */
export function fmtShape(shape: ReadonlyArray<number | string>): string {
	return `[${shape.join('×')}]`;
}

/** Percent with sensible digits: 0.8734 -> "87.3%". */
export function fmtPct(p: number, digits = 1): string {
	if (!Number.isFinite(p)) return '—';
	const v = p * 100;
	if (v !== 0 && Math.abs(v) < 0.1) return `${v < 0 ? MINUS : ''}<0.1%`;
	return `${v.toFixed(digits).replace('-', MINUS)}%`;
}

/**
 * Make whitespace and control characters visible in token chips and labels:
 * space -> "␣", newline -> "↵", tab -> "⇥", other control chars -> "\xNN".
 */
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

/** UTF-8 bytes of a string as hex: "爱" -> "e7 88 b1". */
export function utf8Hex(s: string): string {
	return Array.from(new TextEncoder().encode(s), (b) => b.toString(16).padStart(2, '0')).join(' ');
}
