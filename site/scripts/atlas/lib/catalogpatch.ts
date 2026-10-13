// K4: apply a promote patch to the catalog's TypeScript source as text (pure functions; promote --verify and --apply).
//
// The catalog files are hand-formatted TS, and they differ: models.ts is committed with CRLF line endings, concepts.ts,
// tech-zh.ts and glossary.ts with LF, and the last row of GLOSSARY and of TECH_ZH had no trailing comma. Scripts that
// searched for LF anchors silently matched nothing in models.ts, and rows pasted after a last row without a comma did not
// parse. These functions find what they change by scanning the source (strings, comments and brackets), keep each file's
// own line ending and BOM, add the missing comma, and say what they could not do instead of doing nothing.
//
//   setFields(src, arrayName, slug, fields)   set or add top-level fields of the entry { slug: '<slug>', ... }
//   addRows(src, constName, rows, comment?)    add rows (one line each) at the end of an array or object literal
//   removeRows(src, keys)                      remove one-line rows of TECH_ZH by their English key
//   q(s), f(t), tsValue(...)                   the TS source of a value, in the catalog's style
import type { I18n } from '../../../src/lib/atlas/types/index.ts';

export class PatchError extends Error {}

/** A single-quoted TS string, as the catalog writes them. */
export const q = (s: string): string => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\r?\n/g, '\\n')}'`;

/** f('en', 'zh') (or n('en') where the catalog uses n() for names that read the same in both languages). */
export function f(t: I18n, o: { n?: boolean } = {}): string {
	const zh = t.zh ?? t.en;
	if (o.n && zh === t.en) return `n(${q(t.en)})`;
	return `f(${q(t.en)}, ${q(zh)})`;
}

/** f() over several lines, in the style of the catalog's long texts (the file's own indentation). */
function fLong(t: I18n, indent: string): string {
	return `f(\n${indent}\t${q(t.en)},\n${indent}\t${q(t.zh ?? t.en)}\n${indent})`;
}

export type FieldValue = { kind: 'string'; v: string } | { kind: 'i18n'; v: I18n; n?: boolean } | { kind: 'list'; v: string[] } | { kind: 'lesson'; steps: number; summary: I18n } | { kind: 'raw'; v: string };

/** The source of one field value at an indentation (the entry's property indentation), wrapped like the catalog. */
export function tsValue(key: string, v: FieldValue, indent = '\t\t', width = 150): string {
	switch (v.kind) {
		case 'string':
			return q(v.v);
		case 'list':
			return `[${v.v.map(q).join(', ')}]`;
		case 'raw':
			return v.v;
		case 'i18n': {
			const one = f(v.v, { n: v.n });
			return indent.length * 4 + key.length + one.length + 3 > width ? fLong(v.v, indent) : one;
		}
		case 'lesson': {
			const one = `{ steps: ${v.steps}, summary: ${f(v.summary)} }`;
			if (indent.length * 4 + key.length + one.length + 3 <= width) return one;
			const sum = f(v.summary);
			const inner = `${indent}\t`;
			return `{\n${inner}steps: ${v.steps},\n${inner}summary: ${inner.length * 4 + sum.length + 9 > width ? fLong(v.summary, inner) : sum}\n${indent}}`;
		}
	}
}

// ------------------------------------------------------------------ a small scanner for TS object/array literals
interface Tok {
	/** index of the character */
	i: number;
	/** bracket depth before this character (strings and comments are skipped) */
	depth: number;
	c: string;
}

/** Walk `src` from `from` to `to`, calling `on` for every character outside strings and comments with the bracket depth
 *  before it. Returning true from `on` stops the walk. */
function walk(src: string, from: number, to: number, on: (t: Tok) => boolean | void): void {
	let depth = 0;
	for (let i = from; i < to; i++) {
		const c = src[i];
		if (c === '/' && src[i + 1] === '/') {
			const e = src.indexOf('\n', i);
			i = e < 0 ? to : e - 1;
			continue;
		}
		if (c === '/' && src[i + 1] === '*') {
			const e = src.indexOf('*/', i + 2);
			i = e < 0 ? to : e + 1;
			continue;
		}
		if (c === "'" || c === '"' || c === '`') {
			let j = i + 1;
			while (j < to && src[j] !== c) j += src[j] === '\\' ? 2 : 1;
			if (on({ i, depth, c }) === true) return;
			i = j;
			continue;
		}
		if (on({ i, depth, c }) === true) return;
		if (c === '{' || c === '[' || c === '(') depth++;
		else if (c === '}' || c === ']' || c === ')') depth--;
	}
}

/** The index of the bracket that closes the one at `open` (strings and comments skipped). */
export function closing(src: string, open: number): number {
	let at = -1;
	walk(src, open, src.length, (t) => {
		if (t.depth === 1 && (t.c === '}' || t.c === ']' || t.c === ')') && t.i > open) {
			at = t.i;
			return true;
		}
	});
	if (at < 0) throw new PatchError(`no closing bracket for the one at ${open}`);
	return at;
}

/** The literal that `export const <name>` (an array or object) opens: [open, close] of its brackets. */
export function literalOf(src: string, name: string): { open: number; close: number } {
	const m = new RegExp(`export const ${name}\\b[^=]*=\\s*([\\[{])`).exec(src);
	if (!m) throw new PatchError(`no 'export const ${name} = [' or '{' in the file`);
	const open = m.index + m[0].length - 1;
	return { open, close: closing(src, open) };
}

/** The entry { slug: '<slug>', ... } of an array literal: [open, close] of its braces. */
export function entryOf(src: string, arrayName: string, slug: string): { open: number; close: number } {
	const arr = literalOf(src, arrayName);
	let open = -1;
	let found: { open: number; close: number } | null = null;
	walk(src, arr.open, arr.close, (t) => {
		if (t.depth === 1 && t.c === '{') open = t.i;
		if (t.depth === 2 && open >= 0 && src.startsWith('slug:', t.i)) {
			const m = /^slug:\s*(['"])([^'"]*)\1/.exec(src.slice(t.i, t.i + 200));
			if (m && m[2] === slug) {
				found = { open, close: closing(src, open) };
				return true;
			}
		}
	});
	if (!found) throw new PatchError(`no entry with slug '${slug}' in ${arrayName}`);
	return found;
}

interface Prop {
	key: string;
	/** start of the line that holds the key */
	line: number;
	valueStart: number;
	/** the index of the comma after the value, or of the entry's closing brace */
	valueEnd: number;
	comma: boolean;
}

/** The top-level properties of an object literal (between its braces). */
export function propsOf(src: string, open: number, close: number): Prop[] {
	const out: Prop[] = [];
	let cur: Prop | null = null;
	walk(src, open + 1, close, (t) => {
		if (t.depth !== 0) return;
		if (!cur && /[A-Za-z_$]/.test(t.c) && /[\s{,]/.test(src[t.i - 1] ?? ' ')) {
			const m = /^([A-Za-z_$][\w$]*)\s*:/.exec(src.slice(t.i, t.i + 80));
			if (m) {
				let v = t.i + m[0].length;
				while (src[v] === ' ') v++;
				cur = { key: m[1], line: src.lastIndexOf('\n', t.i) + 1, valueStart: v, valueEnd: close, comma: false };
			}
			return;
		}
		if (cur && t.c === ',') {
			cur.valueEnd = t.i;
			cur.comma = true;
			out.push(cur);
			cur = null;
		}
	});
	if (cur) {
		const c = cur as Prop;
		let e = close;
		while (e > c.valueStart && /\s/.test(src[e - 1])) e--;
		c.valueEnd = e;
		out.push(c);
	}
	return out;
}

/** The line ending of a file: '\r\n' when its first line break is CRLF. */
export const eolOf = (src: string): string => (/\r\n/.test(src.slice(0, 4000)) ? '\r\n' : '\n');
const withEol = (s: string, eol: string) => (eol === '\n' ? s : s.replace(/\r?\n/g, eol));

/** Set (replace or add) top-level fields of one entry. Field order: an existing field keeps its place; new fields go
 *  before `lesson` when the entry has one (the catalog's order), else at the end. */
export function setFields(src: string, arrayName: string, slug: string, fields: [string, FieldValue][]): { text: string; set: string[]; added: string[] } {
	const eol = eolOf(src);
	let text = src;
	const set: string[] = [];
	const added: string[] = [];
	for (const [key, value] of fields) {
		const e = entryOf(text, arrayName, slug);
		const props = propsOf(text, e.open, e.close);
		const indent = props.length ? /^[ \t]*/.exec(text.slice(props[0].line))![0] : '\t\t';
		const src2 = withEol(tsValue(key, value, indent), eol);
		const p = props.find((x) => x.key === key);
		if (p) {
			text = text.slice(0, p.valueStart) + src2 + text.slice(p.valueEnd);
			set.push(key);
			continue;
		}
		const before = props.find((x) => x.key === 'lesson');
		if (before) {
			text = text.slice(0, before.line) + `${indent}${key}: ${src2},${eol}` + text.slice(before.line);
		} else {
			const last = props[props.length - 1];
			const at = last ? last.valueEnd : e.open + 1;
			const comma = last && !last.comma ? ',' : '';
			const tail = last && last.comma ? at + 1 : at;
			text = text.slice(0, tail) + `${comma}${eol}${indent}${key}: ${src2}` + text.slice(tail);
		}
		added.push(key);
	}
	return { text, set, added };
}

/** Add rows (each one line of source, without the trailing comma) at the end of an array or object literal, after an
 *  optional comment line. Adds the comma the last existing row lacks, and ends every new row with one. */
export function addRows(src: string, constName: string, rows: string[], comment?: string): string {
	if (!rows.length) return src;
	const eol = eolOf(src);
	const lit = literalOf(src, constName);
	const ws = (i: number) => /\s/.test(src[i - 1] ?? '');
	let end = lit.close;
	while (end > lit.open + 1 && ws(end)) end--;
	// the last row: before any comment lines that end the literal
	let rowEnd = end;
	for (;;) {
		const ls = src.lastIndexOf('\n', rowEnd - 1) + 1;
		if (ls <= lit.open || !src.slice(ls, rowEnd).trim().startsWith('//')) break;
		rowEnd = ls;
		while (rowEnd > lit.open + 1 && ws(rowEnd)) rowEnd--;
	}
	const indent = /\n([ \t]+)\S[^\n]*$/.exec(src.slice(lit.open, end))?.[1] ?? '\t';
	// the last row needs a comma unless it has one or the literal is empty
	const lastChar = src[rowEnd - 1];
	const comma = rowEnd > lit.open + 1 && lastChar !== ',' && lastChar !== '[' && lastChar !== '{' ? ',' : '';
	const add = [...(comment ? [`${indent}${comment}`] : []), ...rows.map((r) => `${indent}${r},`)].join(eol);
	return src.slice(0, rowEnd) + comma + src.slice(rowEnd, end) + eol + add + src.slice(end);
}

/** Remove one-line rows `'<key>': …` of an object literal (TECH_ZH) by their key. Returns the keys it removed. */
export function removeRows(src: string, constName: string, keys: string[]): { text: string; removed: string[] } {
	const lit = literalOf(src, constName);
	let body = src.slice(lit.open, lit.close);
	const removed: string[] = [];
	for (const k of keys) {
		const lines = body.split('\n');
		const i = lines.findIndex((l) => l.trimStart().startsWith(`${q(k)}:`));
		if (i < 0) continue;
		lines.splice(i, 1);
		body = lines.join('\n');
		removed.push(k);
	}
	return { text: src.slice(0, lit.open) + body + src.slice(lit.close), removed };
}

/** The bytes of a patched file: its BOM (if it had one) and its own line endings are kept. */
export function sameStyle(original: string, patched: string): string {
	const bom = original.startsWith('﻿') ? '﻿' : '';
	const eol = eolOf(original);
	const body = patched.replace(/^﻿/, '');
	return bom + (eol === '\r\n' ? body.replace(/\r?\n/g, '\r\n') : body.replace(/\r\n/g, '\n'));
}
