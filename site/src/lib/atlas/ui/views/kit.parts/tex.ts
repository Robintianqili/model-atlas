// TeX helpers for the views (pure TS, Node-safe).
import { minus } from '../../../core/format.ts';

/** Split `tex` at top-level occurrences of `sep` (outside braces and \begin…\end environments). */
function splitTop(tex: string, sep: string, ok: (rest: string) => boolean = () => true): string[] {
	const parts: string[] = [];
	let depth = 0;
	let env = 0;
	let start = 0;
	for (let i = 0; i < tex.length; i++) {
		const c = tex[i];
		if (c === '\\') {
			if (tex.startsWith('\\begin', i)) env++;
			else if (tex.startsWith('\\end', i)) env--;
			i++; // skip the escaped character
			continue;
		}
		if (c === '{') depth++;
		else if (c === '}') depth--;
		else if (c === sep && depth === 0 && env === 0 && ok(tex.slice(i + 1))) {
			parts.push(tex.slice(start, i));
			start = i + 1;
		}
	}
	parts.push(tex.slice(start));
	return parts;
}

/** Split a formula at its top-level '=' signs into aligned lines, so a long op formula such as
 *  `y = Σ x·w + b = {{s}} + {{b}} = {{y}}` fits a phone. Two formulas joined by a top-level ';' (lstm: `c = …;\quad h = …`)
 *  become consecutive aligned blocks. Formulas with fewer than `minParts` parts, or with their own alignment, are
 *  returned unchanged. */
export function alignTex(tex: string, minParts = 3): string {
	if (/\\begin\{(aligned|align|array|matrix|gathered)/.test(tex)) return tex;
	const chunks = splitTop(tex, ';', (rest) => /^\s*\\q?quad/.test(rest)).map((c) => c.replace(/^\s*\\q?quad\s*/, '').trim()).filter((c) => c !== '');
	const lines: string[] = [];
	let total = 0;
	for (const chunk of chunks) {
		const parts = splitTop(chunk, '=').map((p) => p.trim());
		if (parts.some((p) => p === '')) return tex;
		total += parts.length;
		const [first, ...rest] = parts;
		if (!rest.length) lines.push(`${first} &`);
		else lines.push(`${first} &= ${rest.map(breakSum).join(' \\\\ &= ')}`);
	}
	if (total < minParts) return tex;
	return `\\begin{aligned} ${lines.join(' \\\\[6pt] ')} \\end{aligned}`;
}

/** K1: a long right-hand side with three or more top-level terms is broken after every second '+' (a continuation line
 *  `&\quad + …`), so a line of live numbers ("0.199 · 4.42 + 1.00 · 1.00 + …") fits a 390 px phone instead of scrolling. */
export function breakSum(part: string, per = 2): string {
	const terms = splitTop(part, '+');
	if (terms.length < 3 || part.replace(/\{\{[^}]*\}\}/g, 'x.xxx').length < 36) return part;
	const rows: string[] = [];
	for (let i = 0; i < terms.length; i += per) rows.push(terms.slice(i, i + per).map((x) => x.trim()).join(' + '));
	return rows.join(' \\\\ &\\quad + ');
}

/** Values for FormulaCard: exact integers become text ("3", not "3.00"); other numbers stay numbers (3 significant
 *  digits, as the kit prints them). */
export function texValues(values: Record<string, number | string>): Record<string, number | string> {
	const out: Record<string, number | string> = {};
	for (const [k, v] of Object.entries(values)) out[k] = typeof v === 'number' && Number.isInteger(v) && Math.abs(v) < 1e6 ? minus(String(v)) : v;
	return out;
}

/** The {{name}} placeholders of a formula that `values` cannot fill. */
export function missingNames(tex: string, values: Record<string, unknown>): string[] {
	const out: string[] = [];
	for (const m of tex.matchAll(/\{\{\s*([\w.-]+)\s*\}\}/g)) if (!(m[1] in values) && !out.includes(m[1])) out.push(m[1]);
	return out;
}
