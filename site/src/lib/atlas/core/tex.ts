// KaTeX colour roles for op formulas (design §4.3 OpDef.formula). Same output as kit/tex.ts sym(), without
// importing KaTeX: the UI fills {{name}} placeholders with kit fillTex() and renders with kit renderTex().
// Roles map to the CSS classes .sym-<role> in styles/math.css:
//   act (activation) · param (weights) · grad · in (input) · out (output) · q / k / v · prob · neg / pos · accent

export type SymRole = 'act' | 'param' | 'grad' | 'in' | 'out' | 'q' | 'k' | 'v' | 'prob' | 'neg' | 'pos' | 'accent';

/** Wrap a TeX fragment so it takes the colour of its role. */
export function sym(role: SymRole, tex: string): string {
	return `\\htmlClass{sym-${role}}{${tex}}`;
}

/** The {{name}} placeholders of a formula, in order of first use. */
export function texNames(tex: string): string[] {
	const out: string[] = [];
	for (const m of tex.matchAll(/\{\{\s*([\w.-]+)\s*\}\}/g)) if (!out.includes(m[1])) out.push(m[1]);
	return out;
}
