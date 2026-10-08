/** Helpers that redo one output of a layer by hand, for formula cards with live numbers. */
import { sym } from '$lib/kit/tex';

/** y[o] = Σ_i x[i] W[o, i] + b[o] — returns the first `show` terms and the total. */
export function linearTerms(x: ArrayLike<number>, W: Float32Array, b: Float32Array | null, nIn: number, o: number, show = 2) {
	let s = b ? b[o] : 0;
	for (let i = 0; i < nIn; i++) s += x[i] * W[o * nIn + i];
	const terms = Array.from({ length: show }, (_, i) => ({ x: x[i], w: W[o * nIn + i] }));
	return { terms, b: b ? b[o] : 0, y: s, nIn };
}

/**
 * TeX + values for a linear output, e.g. q_{t,5} = Σ_i x_i W_{5,i} + b_5 = (x0)(w0) + (x1)(w1) + … + b = y.
 * `outSym` / `inSym` / `wSym` are TeX snippets; the roles color them like the diagram.
 */
export function linearTex(
	out: string,
	inp: string,
	w: string,
	bias: string | null,
	r: ReturnType<typeof linearTerms>,
	outRole: Parameters<typeof sym>[0] = 'out'
): { tex: string; values: Record<string, number> } {
	const values: Record<string, number> = { y: r.y };
	const parts: string[] = [];
	r.terms.forEach((tm, i) => {
		values[`x${i}`] = tm.x;
		values[`w${i}`] = tm.w;
		parts.push(`({{x${i}}})(${'{{'}w${i}${'}}'})`);
	});
	if (bias) values.b = r.b;
	const tex =
		`\\begin{aligned}${sym(outRole, out)} &= \\textstyle\\sum_{i=0}^{${r.nIn - 1}} ${sym('in', inp)}\\,${sym('param', w)}${bias ? ` + ${sym('param', bias)}` : ''} \\\\ ` +
		`&= ${parts.join(' \\\\ &\\quad + ')} \\\\ &\\quad + \\dots${bias ? ' + ({{b}})' : ''} \\\\ &= {{y}}\\end{aligned}`;
	return { tex, values };
}
