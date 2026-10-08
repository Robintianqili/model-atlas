// tinyseq2seq: the synthetic task, in TS (the source of truth). A date written in one of eight formats becomes ISO 8601
// (2021-03-05). pipeline/tinyseq2seq/make_data.ts writes the training pairs for Python from this file, so Python trains
// on exactly these strings (their sha256 is in facts.json; claims.test.ts regenerates the set and checks it).
//   source characters: SRC (pad, unknown '?', space , . / -, digits, a-z, A-Z); target symbols: TGT (start, end, 0-9, -)
import type { I18n, Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
export const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
export const SRC = ['<pad>', '?', ' ', ',', '.', '/', '-', ...'0123456789', ...'abcdefghijklmnopqrstuvwxyz', ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
export const TGT = ['<s>', '</s>', ...'0123456789', '-'];
export const BOS = 0;
export const EOS = 1;
/** the longest input the model reads (the generator's longest date has 29 characters) */
export const T_MAX = 32;
/** decoder steps: ten characters and the end symbol */
export const STEPS = 11;

export type Ymd = { y: number; m: number; d: number };
type Opt = { ord: boolean; pad: boolean };

export const daysIn = (y: number, m: number) => (m === 2 ? (y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0) ? 29 : 28) : [4, 6, 9, 11].includes(m) ? 30 : 31);
/** day of the week, 0 = Sunday (Sakamoto's method) */
export function weekday({ y, m, d }: Ymd): number {
	const t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
	const yy = m < 3 ? y - 1 : y;
	return (yy + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) + t[m - 1] + d) % 7;
}
const ordinal = (d: number) => d + (d % 10 === 1 && d !== 11 ? 'st' : d % 10 === 2 && d !== 12 ? 'nd' : d % 10 === 3 && d !== 13 ? 'rd' : 'th');
const two = (n: number) => String(n).padStart(2, '0');
const mon = (x: Ymd) => MONTHS[x.m - 1];
const day = (x: Ymd, o: Opt) => (o.ord ? ordinal(x.d) : String(x.d));

/** The eight training formats (label: how the lesson names one). `variants` lists the option sets the generator uses. */
export const FORMATS: { id: string; label: I18n; example: string; variants: Opt[]; write: (x: Ymd, o: Opt) => string }[] = [
	{ id: 'long', label: { en: 'month day, year', zh: '月 日, 年' }, example: 'March 5, 2021', variants: [{ ord: false, pad: false }, { ord: true, pad: false }], write: (x, o) => `${mon(x)} ${day(x, o)}, ${x.y}` },
	{ id: 'short', label: { en: 'short month', zh: '缩写月份' }, example: 'Mar 5, 2021', variants: [{ ord: false, pad: false }], write: (x) => `${mon(x).slice(0, 3)} ${x.d}, ${x.y}` },
	{ id: 'dmy', label: { en: 'day month year', zh: '日 月 年' }, example: '5 March 2021', variants: [{ ord: false, pad: false }, { ord: true, pad: false }], write: (x, o) => `${day(x, o)} ${mon(x)} ${x.y}` },
	{ id: 'dmyShort', label: { en: 'day, short month', zh: '日 + 缩写月份' }, example: '5 Mar 2021', variants: [{ ord: false, pad: false }], write: (x) => `${x.d} ${mon(x).slice(0, 3)} ${x.y}` },
	{ id: 'weekday', label: { en: 'with the weekday', zh: '带星期' }, example: 'Friday, March 5, 2021', variants: [{ ord: false, pad: false }], write: (x) => `${WEEKDAYS[weekday(x)]}, ${mon(x)} ${x.d}, ${x.y}` },
	{ id: 'weekdayShort', label: { en: 'short weekday', zh: '缩写星期' }, example: 'Fri, 5 Mar 2021', variants: [{ ord: false, pad: false }], write: (x) => `${WEEKDAYS[weekday(x)].slice(0, 3)}, ${x.d} ${mon(x).slice(0, 3)} ${x.y}` },
	{ id: 'us', label: { en: 'US numbers: month/day/year', zh: '美式数字：月/日/年' }, example: '3/5/2021', variants: [{ ord: false, pad: false }, { ord: false, pad: true }], write: (x, o) => (o.pad ? `${two(x.m)}/${two(x.d)}/${x.y}` : `${x.m}/${x.d}/${x.y}`) },
	{ id: 'eu', label: { en: 'European numbers: day.month.year', zh: '欧式数字：日.月.年' }, example: '05.03.2021', variants: [{ ord: false, pad: true }, { ord: false, pad: false }], write: (x, o) => (o.pad ? `${two(x.d)}.${two(x.m)}.${x.y}` : `${x.d}.${x.m}.${x.y}`) }
];

export const iso = (x: Ymd) => `${x.y}-${two(x.m)}-${two(x.d)}`;

/** One training pair: a random date (years 1900 to 2099), a random format and variant, and a random case
 *  (75% as written, 20% lower case, 5% upper case). Draws, in order: y, m, d, format, variant, case. */
export function samplePair(rng: Rng): { src: string; tgt: string; fmt: string } {
	const y = 1900 + rng.int(200);
	const m = 1 + rng.int(12);
	const x = { y, m, d: 1 + rng.int(daysIn(y, m)) };
	const f = FORMATS[rng.int(FORMATS.length)];
	const s = f.write(x, f.variants[rng.int(f.variants.length)]);
	const c = rng.next();
	return { src: c < 0.75 ? s : c < 0.95 ? s.toLowerCase() : s.toUpperCase(), tgt: iso(x), fmt: f.id };
}

/** A data set: n pairs from one seed (training 1, validation 2, test 3). */
export const dataset = (seed: number, n: number) => {
	const rng = createRng(seed);
	return Array.from({ length: n }, () => samplePair(rng));
};

/** The date a text means, read leniently (month names or their first three letters in any case, ordinals, the days
 *  first to ninth as words (off.json's spelled-out days), numbers month/day/year, day.month.year, day-month-year,
 *  year/month/day), whether it is a real date, and which training format writes exactly this text (null: none does). */
export function readDate(text: string): { ymd: Ymd | null; valid: boolean; fmt: string | null } {
	const s = text.trim();
	const M = (w: string) => MONTHS.findIndex((n) => n.toLowerCase() === w.toLowerCase() || n.slice(0, 3).toLowerCase() === w.toLowerCase()) + 1;
	const W = '(?:[A-Za-z]+,?\\s+)?';
	let r: Ymd | null = null;
	const DW = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth'];
	let a = new RegExp(`^${W}([A-Za-z]+)\\.?\\s+(\\d{1,2}|${DW.join('|')})(?:st|nd|rd|th)?,?\\s+(\\d{4})$`, 'i').exec(s);
	if (a && M(a[1])) r = { y: +a[3], m: M(a[1]), d: /\d/.test(a[2]) ? +a[2] : DW.indexOf(a[2].toLowerCase()) + 1 };
	a = r ? null : new RegExp(`^${W}(\\d{1,2})(?:st|nd|rd|th)?\\s+([A-Za-z]+)\\.?,?\\s+(\\d{4})$`, 'i').exec(s);
	if (a && M(a[2])) r = { y: +a[3], m: M(a[2]), d: +a[1] };
	a = r ? null : /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(s);
	if (a) r = { y: +a[3], m: +a[1], d: +a[2] };
	a = r ? null : /^(\d{1,2})[.-](\d{1,2})[.-](\d{4})$/.exec(s);
	if (a) r = { y: +a[3], m: +a[2], d: +a[1] };
	a = r ? null : /^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$/.exec(s);
	if (a) r = { y: +a[1], m: +a[2], d: +a[3] };
	const valid = !!r && r.m >= 1 && r.m <= 12 && r.d >= 1 && r.d <= daysIn(r.y, r.m);
	const low = s.toLowerCase();
	const fmt = valid && s.length <= T_MAX ? (FORMATS.find((f) => f.variants.some((o) => f.write(r!, o).toLowerCase() === low))?.id ?? null) : null;
	return { ymd: r, valid, fmt };
}
