// Helpers for a step's vars() (K0): values that read naturally in both languages. A vars() entry may be an I18n
// (TmplVal), printed as is in each language, so a sentence can change with the numbers without casts.
//   plural(n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' })   "1 point" / "200 points" / "200 个点"
//   num(v, 'n4')                                                      a number as {name|n4} would print it, in both languages
//   pick(cond, a, b)                                                  one of two I18n texts
// A non-finite n or v comes back as NaN, so lint still reports it (AK401) instead of printing "…" unnoticed.
import type { I18n } from '../types/core.ts';
import { formatValue } from './narrate.ts';

/** {n} in each text is n, formatted like a placeholder with `fmt` (default: an exact integer as itself). */
export function plural(n: number, t: { one: string; many: string; zh?: string }, fmt?: string): I18n | number {
	const en = formatValue(n, fmt, 'en');
	const zh = formatValue(n, fmt, 'zh');
	if (en === null || zh === null) return NaN;
	return { en: (n === 1 ? t.one : t.many).replaceAll('{n}', en), zh: (t.zh ?? (n === 1 ? t.one : t.many)).replaceAll('{n}', zh) };
}

/** A number printed exactly as {name|fmt} prints it (to build a sentence in vars()). */
export function num(v: number, fmt?: string): I18n | number {
	const en = formatValue(v, fmt, 'en');
	const zh = formatValue(v, fmt, 'zh');
	return en === null || zh === null ? NaN : { en, zh };
}

/** One of two texts. */
export function pick(cond: boolean, a: I18n, b: I18n): I18n {
	return cond ? a : b;
}
