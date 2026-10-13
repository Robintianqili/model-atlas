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
 *  returned unchanged. K4: `narrow` (a card narrower than about 520 px; OpFormula and zoom.formula measure theirs) also
 *  breaks a long right-hand side of two terms ("= 0.744 × −6.27 + 0.256 × −3.74") after its '+'. */
export function alignTex(tex: string, minParts = 3, o: { narrow?: boolean } = {}): string {
	if (!Number.isInteger(minParts)) minParts = 3; // as a map callback the index would arrive here
	const narrow = !!o?.narrow;
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
		// K4: never `rest.map(breakSum)`: Array.map passes the index as `per`, and per = 0 looped forever (RangeError,
		// AK601) on a first right-hand side of 3 or more terms
		else lines.push(`${first} &= ${rest.map((p) => breakSum(p, 2, narrow)).join(' \\\\ &= ')}`);
	}
	if (total < minParts) return tex;
	return `\\begin{aligned} ${lines.join(' \\\\[6pt] ')} \\end{aligned}`;
}

/** About how many characters a TeX fragment shows: a {{name}} slot counts as a number ("−0.123"), \htmlClass wrappers and
 *  braces count as nothing, any other command (\times, \cdot, \sum) as one character. */
export function texWidth(tex: string): number {
	return tex
		.replace(/\{\{[^}]*\}\}/g, '−0.123')
		.replace(/\\htmlClass\{[^}]*\}/g, '')
		.replace(/\\[a-zA-Z]+/g, '#')
		.replace(/[{}\s]/g, '').length;
}

/** K1: a long right-hand side with three or more top-level terms is broken after every second '+' (a continuation line
 *  `&\quad + …`), so a line of live numbers ("0.199 · 4.42 + 1.00 · 1.00 + …") fits a 390 px phone instead of scrolling.
 *  K4: `per` below 1 (or not an integer) means 2; `narrow` also breaks two terms, and puts one term per line when two
 *  would still be wider than about 24 characters. */
export function breakSum(part: string, per = 2, narrow = false): string {
	if (!Number.isInteger(per) || per < 1) per = 2;
	const terms = splitTop(part, '+');
	if (narrow && terms.length >= 2 && texWidth(part) >= 22) {
		const each = texWidth(part) / terms.length;
		per = Math.max(1, Math.min(per, Math.floor(24 / Math.max(1, each))));
	} else if (terms.length < 3 || part.replace(/\{\{[^}]*\}\}/g, 'x.xxx').length < 36) return part;
	const rows: string[] = [];
	for (let i = 0; i < terms.length; i += per) rows.push(terms.slice(i, i + per).map((x) => x.trim()).join(' + '));
	return rows.join(' \\\\ &\\quad + ');
}

/** Values for FormulaCard: exact integers become text ("3", not "3.00"); K4: so does a number that is exactly a short
 *  decimal of fewer significant digits than the card prints (0.25 stays "0.25", not "0.250": 1/√16 next to a narration
 *  that says 0.25). Other numbers stay numbers (`digits` significant digits, default 3, as the kit prints them). */
export function texValues(values: Record<string, number | string>, digits = 3): Record<string, number | string> {
	if (!Number.isInteger(digits) || digits < 1) digits = 3;
	const out: Record<string, number | string> = {};
	for (const [k, v] of Object.entries(values)) {
		if (typeof v !== 'number') out[k] = v;
		else if (Number.isInteger(v) && Math.abs(v) < 1e6) out[k] = minus(String(v));
		else if (digits > 1 && Number.isFinite(v) && Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 && Number(v.toPrecision(digits - 1)) === v) out[k] = minus(String(v));
		else out[k] = v;
	}
	return out;
}

/** The {{name}} placeholders of a formula that `values` cannot fill. */
export function missingNames(tex: string, values: Record<string, unknown>): string[] {
	const out: string[] = [];
	for (const m of tex.matchAll(/\{\{\s*([\w.-]+)\s*\}\}/g)) if (!(m[1] in values) && !out.includes(m[1])) out.push(m[1]);
	return out;
}
