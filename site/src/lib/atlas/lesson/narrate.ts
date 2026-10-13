// The narration language (design §4.6, §5.3): `{name|fmt}` placeholders, no expressions.
// Names resolve in order: the template's vars() result; at-dependent paths (frame.*, run.length, opt.* of the step's
// source, unit.* of the step's unit); rt.fact(name) (built-in paths, fact:<key>).
// fmt: n3 (3 significant digits) | n2 n4 n5 n6 (that many significant digits: n4 shows a small change that n3 rounds
// away) | int | pct | signed | shape | label | tok. Without a fmt, exact integers print as integers and other numbers as n3.
// An unresolved or non-finite value renders "…" and is reported in `missing` (lint AK401).
// R1 (redesign W2): {name@ref} is {name} whose number names a stage target (StepDef.refs[ref]): the value prints as
// {name} does, and the placeholder record keeps the ref (a number chip rings that target).
// K2: on a step the empty input skipped (At.skipped) the sentences whose values do not exist yet are left out and the
// narration ends with EMPTY_LINE (a look hint drops them silently), so the learner never reads "…" next to the empty note.
import type { I18n } from '../types/core.ts';
import type { Focus } from '../types/process.ts';
import type { At, Narration, TmplVal } from '../types/lesson.ts';
import type { ExplainerRuntime, PackageModules } from '../types/runtime.ts';
import { atFact } from './facts.ts';
import { fmt3, fmtInt, fmtPct, fmtSci, sigDigits, visibleToken } from '../core/format.ts';

export type Lang = 'en' | 'zh';
export const FORMATS = ['n3', 'n2', 'n4', 'n5', 'n6', 'int', 'pct', 'pct2', 'sci', 'signed', 'shape', 'label', 'tok'] as const;
export const PLACEHOLDER_RE = /\{([A-Za-z_][\w.:#@-]*)(?:\|([A-Za-z0-9]+))?\}/g;
export const MISSING = '…';

/** R1: a placeholder name and the stage target its number names ({y@cell}: name 'y', ref 'cell'). Names with a ':'
 *  (fact:<key>) never carry a ref. */
export function splitRef(name: string): { name: string; ref?: string } {
	const i = name.lastIndexOf('@');
	if (i <= 0 || name.startsWith('fact:')) return { name };
	return { name: name.slice(0, i), ref: name.slice(i + 1) || undefined };
}

export function placeholders(text: string | undefined): { name: string; fmt?: string; ref?: string }[] {
	if (!text) return [];
	return [...text.matchAll(PLACEHOLDER_RE)].map((m) => {
		const r = splitRef(m[1]);
		return r.ref ? { name: r.name, fmt: m[2], ref: r.ref } : { name: r.name, fmt: m[2] };
	});
}

export function placeholderNames(text: string | undefined): string[] {
	return [...new Set(placeholders(text).map((p) => p.name))];
}

// ------------------------------------------------------------------ number formats: core/format.ts (one implementation;
// every minus sign is U+2212, also in exponents). Re-exported for the tests and older imports.
export { fmt3, fmtInt, fmtPct, visibleToken };

const isI18n = (v: unknown): v is I18n => !!v && typeof v === 'object' && typeof (v as I18n).en === 'string';

/** Format one resolved value. Returns null when the value cannot be shown (unresolved, non-finite, wrong type). */
export function formatValue(v: unknown, fmt: string | undefined, lang: Lang): string | null {
	if (v === undefined || v === null) return null;
	if (fmt !== undefined && !(FORMATS as readonly string[]).includes(fmt)) return null;
	if (isI18n(v)) return (lang === 'zh' ? v.zh : undefined) ?? v.en;
	if (Array.isArray(v)) {
		if (v.some((x) => typeof x === 'number' && !Number.isFinite(x))) return null;
		return fmt === 'shape' || v.every((x) => typeof x === 'number' || typeof x === 'string') ? `[${v.join('×')}]` : null;
	}
	if (typeof v === 'string') return fmt === 'tok' ? `“${visibleToken(v)}”` : v;
	if (typeof v === 'boolean') return lang === 'zh' ? (v ? '是' : '否') : v ? 'yes' : 'no';
	if (typeof v !== 'number' || !Number.isFinite(v)) return null;
	const k = sigDigits(fmt);
	if (k !== null) return fmt3(v, k);
	switch (fmt) {
		case 'int':
			return fmtInt(v);
		case 'pct':
			return fmtPct(v);
		// K4: two decimals of a share near 0% or 100%, and scientific notation without the calculator's e
		case 'pct2':
			return fmtPct(v, 2);
		case 'sci':
			return fmtSci(v);
		case 'signed':
			return v > 0 ? `+${fmt3(v)}` : fmt3(v);
		case 'shape':
		case 'label':
		case 'tok':
			return String(v);
		case 'n3':
			return fmt3(v);
		default:
			// no format given: an exact integer (a count, a kernel size) prints as itself ("9", "6,722"), not "9.00";
			// every other number gets 3 significant digits
			return Number.isInteger(v) && Math.abs(v) < 1e15 ? fmtInt(v) : fmt3(v);
	}
}

export interface Placeholder {
	name: string;
	fmt?: string;
	value: unknown;
	focus?: Focus;
	ok: boolean;
	/** where the value came from */
	from: 'vars' | 'at' | 'fact' | 'none';
}
export interface Rendered {
	text: string;
	lang: Lang;
	/** zh was requested but the template has no zh text: en was used */
	fallback: boolean;
	placeholders: Placeholder[];
	/** names that rendered as "…" */
	missing: string[];
	/** vars() threw */
	error?: string;
	/** K2, a skipped step: the names whose sentences were left out */
	dropped?: string[];
}

/** Resolve one name for a step (vars first, then the step's position, then rt.fact). */
export function resolveName(name: string, vars: Record<string, TmplVal> | null, rt: ExplainerRuntime, at: At | undefined): { value: unknown; fmt?: string; focus?: Focus; from: Placeholder['from'] } {
	if (vars && Object.prototype.hasOwnProperty.call(vars, name)) {
		const v = vars[name];
		if (v && typeof v === 'object' && 'v' in v) return { value: v.v, fmt: v.fmt, focus: v.focus, from: 'vars' };
		return { value: v, from: 'vars' };
	}
	const a = atFact(name, at);
	if (a !== undefined) return { value: a, from: 'at' };
	if (name.startsWith('opt.') && at?.source) {
		// K1: the step's own source (its options, then its process defaults), also before it has run (no "…" while training)
		const v = sourceOpt(rt, at.source, name.slice(4));
		if (v !== undefined) return { value: v, from: 'fact' };
	}
	let f: unknown;
	try {
		f = rt.fact(name);
	} catch {
		f = undefined;
	}
	return { value: f, from: f === undefined ? 'none' : 'fact' };
}

/** A source's option `key`: its SourceDef.opt, else the defaults of the process (or trainer) it runs. K1. */
export function sourceOpt(rt: ExplainerRuntime, source: string, key: string): number | string | undefined {
	const pkg = (rt as { pkg?: PackageModules }).pkg;
	const d = pkg?.manifest.sources?.[source];
	if (!pkg || !d || (d.kind !== 'train' && d.kind !== 'process')) return undefined;
	const p = pkg.processes[d.kind === 'train' ? `train:${d.trainer}` : d.process];
	const v = (d.opt as Record<string, unknown> | undefined)?.[key] ?? (p?.defaults as Record<string, unknown> | undefined)?.[key];
	return typeof v === 'number' || typeof v === 'string' ? v : undefined;
}

/** K2: the last sentence of a narration whose step the empty input skipped (its other sentences need numbers). */
export const EMPTY_LINE: I18n = { en: 'Its numbers appear once there is an input.', zh: '有了输入，这一步就会填上真实的数。' };

/** Marks a missing value while the sentences of a skipped step are sorted; never shown. */
const GAP = '\u0000';
/** A value printed from a raw number or a missing field in a vars() string (K2). */
const RAW_GAP = /(?<![A-Za-z])(NaN|undefined|null)(?![A-Za-z])/;

/** A rendered text cut into sentences, each with its own trailing space, so joining them gives the text back. en ends a
 *  sentence at . ! ? followed by a space (or the end; a decimal point is followed by a digit); zh at 。！？. */
export function sentences(text: string, lang: Lang): string[] {
	const re = lang === 'zh' ? /[^。！？!?]*(?:[。！？!?]+[”’」）)]*|$)/g : /[\s\S]*?(?:[.!?]+["”’)]*(?:\s+|$)|$)/g;
	return (text.match(re) ?? []).filter((x) => x.length > 0);
}

export interface RenderOpts {
	/** K2, a skipped step: end with EMPTY_LINE when a sentence was left out (default true; a look hint passes false) */
	emptyLine?: boolean;
	/** R1: wrap the value of every {name@ref} in chip marks (chipParts splits them; the lesson renders number chips) */
	chips?: boolean;
}

/** R1 number chips: private-use marks around a {name@ref} value in a chip rendering (never in plain narration). */
export const CHIP = { open: '\uE000', sep: '\uE001', close: '\uE002' } as const;
const CHIP_RE = /\uE000([^\uE001]*)\uE001([^\uE002]*)\uE002/g;

/** A chip rendering split into text and chips. */
export function chipParts(text: string): ({ text: string } | { ref: string; text: string })[] {
	const out: ({ text: string } | { ref: string; text: string })[] = [];
	let last = 0;
	for (const m of text.matchAll(CHIP_RE)) {
		if (m.index! > last) out.push({ text: text.slice(last, m.index) });
		out.push({ ref: m[1], text: m[2] });
		last = m.index! + m[0].length;
	}
	if (last < text.length) out.push({ text: text.slice(last) });
	return out;
}

/** The plain text of a chip rendering. */
export const stripChips = (text: string): string => text.replace(CHIP_RE, '$2');

/** Does a template name a stage target ({name@ref}) in either language? */
export const hasRefs = (t: { en?: string; zh?: string } | undefined): boolean => !!t && [t.en, t.zh].some((x) => placeholders(x).some((p) => p.ref));

/** K4: how many gap marks a text shows: a "…" that stands alone (next to no letter, digit or CJK character), as the
 *  resolver prints a missing value in English ("the loss is ….", "token “…”"). A literal ellipsis inside a word or a
 *  range ("是…的首都", "−1…1", "Training…") is text, not a gap (TransE and behavior-cloning had to reword them). A
 *  Chinese gap sits between characters ("损失是…。"), so the English pass and lint's missing values catch it. */
export function gapMarks(text: string): number {
	let n = 0;
	const w = /[\p{L}\p{N}]/u;
	for (let i = text.indexOf(MISSING); i >= 0; i = text.indexOf(MISSING, i + 1)) {
		const a = text[i - 1] ?? '';
		const b = text[i + 1] ?? '';
		if (!(w.test(a) || w.test(b))) n++;
	}
	return n;
}

/** How many more "…" the rendered text has than its template (K2, lint): a vars() value that prints "…" itself. K4:
 *  only gap marks count (gapMarks). */
export function extraEllipses(rendered: string, template: string | undefined): number {
	const n = gapMarks;
	return n(rendered) - n((template ?? '').replace(PLACEHOLDER_RE, ''));
}

/** Render a narration (or a look hint) for one language. A plain I18n renders as is. K2: when `at.skipped` (the empty
 *  input skipped the step), each sentence with a value that does not exist yet is left out instead of printing "…". */
export function render<R extends ExplainerRuntime>(tmpl: Narration<R> | undefined, rt: R, at: At | undefined, lang: Lang, o: RenderOpts = {}): Rendered {
	if (!tmpl) return { text: '', lang, fallback: false, placeholders: [], missing: [] };
	const fallback = lang === 'zh' && !tmpl.zh;
	const src = (lang === 'zh' ? tmpl.zh : undefined) ?? tmpl.en;
	const skipped = !!at?.skipped;
	let vars: Record<string, TmplVal> | null = null;
	let error: string | undefined;
	const fn = (tmpl as { vars?: (rt: R, at: At) => Record<string, TmplVal> }).vars;
	if (fn) {
		try {
			vars = fn(rt, at ?? { index: -1 }) ?? {};
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			vars = null;
		}
	}
	const seen: Placeholder[] = [];
	const missing: string[] = [];
	const text = src.replace(PLACEHOLDER_RE, (_m, raw: string, fmt: string | undefined) => {
		const { name, ref } = splitRef(raw);
		const r = resolveName(name, vars, rt, at);
		let s = formatValue(r.value, fmt ?? r.fmt, lang);
		// K2: on a skipped step a vars() fallback that prints "…", nothing, or a raw NaN (no frame to read yet) is a
		// missing value too
		if (skipped && s !== null && (s.includes(MISSING) || !s.trim() || RAW_GAP.test(s))) s = null;
		const ok = s !== null;
		seen.push({ name, fmt: fmt ?? r.fmt, value: r.value, focus: r.focus, ok, from: r.from });
		if (!ok && !missing.includes(name)) missing.push(name);
		if (s !== null && ref && o.chips) return `${CHIP.open}${ref}${CHIP.sep}${s}${CHIP.close}`;
		return s !== null ? s : skipped ? GAP : MISSING;
	});
	if (skipped && missing.length) {
		// a skipped step's vars() may throw or miss values (the frames do not exist): those sentences are left out
		const body = sentences(text, lang)
			.filter((x) => !x.includes(GAP))
			.join('')
			.trim();
		const line = o.emptyLine === false ? '' : lang === 'zh' ? (EMPTY_LINE.zh ?? EMPTY_LINE.en) : EMPTY_LINE.en;
		const out = body && line ? `${body}${lang === 'zh' ? '' : ' '}${line}` : body || line;
		return { text: out, lang, fallback, placeholders: seen, missing: [], dropped: missing };
	}
	return { text, lang, fallback, placeholders: seen, missing, ...(error && !skipped ? { error } : {}) };
}

/** Both languages as an I18n (zh omitted when the template has none, so the UI falls back to en). */
export function renderI18n<R extends ExplainerRuntime>(tmpl: Narration<R> | undefined, rt: R, at: At | undefined, o: RenderOpts = {}): I18n {
	const en = render(tmpl, rt, at, 'en', o).text;
	if (!tmpl?.zh) return { en };
	return { en, zh: render(tmpl, rt, at, 'zh', o).text };
}

// ------------------------------------------------------------------ static text checks (lint)
/** Literal numbers with two or more digits or a decimal point, outside placeholders and identifiers (AK402). */
export function literalNumbers(text: string | undefined): string[] {
	if (!text) return [];
	const bare = text.replace(PLACEHOLDER_RE, ' ');
	const out: string[] = [];
	for (const m of bare.matchAll(/(?<![A-Za-z_\d.#])\d+(?:[.,]\d+)*(?:\.\d+)?%?(?![A-Za-z_\d])/g)) {
		const s = m[0];
		const digits = s.replace(/[^\d]/g, '').length;
		if (digits >= 2 || s.includes('.')) out.push(s);
	}
	return out;
}

export function wordCount(text: string): number {
	return text.replace(PLACEHOLDER_RE, 'X').split(/\s+/).filter((w) => /[A-Za-z0-9…]/.test(w)).length;
}

export function sentenceCount(text: string, lang: Lang): number {
	const t = text.replace(PLACEHOLDER_RE, 'X').trim();
	if (!t) return 0;
	const parts = lang === 'zh' ? t.split(/[。！？!?]+/) : t.split(/(?<=[.!?])\s+(?=[A-Z"“(X{])/);
	return parts.filter((p) => p.trim().length > 0).length;
}
