/**
 * KaTeX helpers. Formulas color each symbol like its tensor in the diagram (brief §4.7).
 *
 *   import { sym, renderTex } from '$lib/kit/tex';
 *   const tex = `${sym('out', 'y')} = ${sym('param', 'W')}${sym('act', 'x')} + ${sym('param', 'b')}`;
 *
 * Roles map to CSS classes .sym-<role> defined in styles/math.css:
 *   act (activation, ink) · param (weights) · grad (gradients) · in (input) · out (output)
 *   q / k / v (attention frames: blue / red / green) · prob (purple) · neg / pos (blue / orange) · accent
 *
 * Live numbers: write {{name}} in the TeX and pass values={{ name: 0.123 }} to FormulaCard;
 * the number is formatted to 3 significant digits and tagged .num-live.
 */
import katex from 'katex';
import { fmt3 } from './format.ts';

export type SymRole = 'act' | 'param' | 'grad' | 'in' | 'out' | 'q' | 'k' | 'v' | 'prob' | 'neg' | 'pos' | 'accent';

/** Wrap a TeX fragment so it takes the color of its role. */
export function sym(role: SymRole, tex: string): string {
	return `\\htmlClass{sym-${role}}{${tex}}`;
}

/** A number as TeX: 3 significant digits, exponent as ×10^k, ASCII minus. */
export function texNum(v: number | string, digits = 3): string {
	if (typeof v === 'string') return `\\text{${v.replace(/[{}\\]/g, '')}}`;
	const s = fmt3(v, digits).replace(/−/g, '-');
	const m = /^(-?[\d.]+)e(-?\d+)$/.exec(s);
	const body = m ? `${m[1]}\\times10^{${m[2]}}` : s === '∞' ? '\\infty' : s === '-∞' ? '-\\infty' : s;
	return `{${body}}`;
}

/** Replace {{name}} placeholders with live numbers. Unknown names render as "?". */
export function fillTex(tex: string, values: Record<string, number | string> = {}, digits = 3): string {
	return tex.replace(/\{\{\s*([\w.-]+)\s*\}\}/g, (_, name: string) =>
		name in values ? `\\htmlClass{num-live}{${texNum(values[name], digits)}}` : '\\text{?}'
	);
}

/** Render TeX to an HTML string (HTML + MathML for screen readers). Never throws. */
export function renderTex(tex: string, display = false): string {
	return katex.renderToString(tex, {
		displayMode: display,
		throwOnError: false,
		output: 'htmlAndMathml',
		strict: 'ignore',
		trust: (ctx) => ctx.command === '\\htmlClass'
	});
}
