// crnn-ocr: CTC (connectionist temporal classification, Graves et al. 2006) on the model's per-column chances, in
// float64. Used by the decode process (process.ts), the CTC view and the lesson; pipeline/crnn-ocr/reference.py checks
// the process frames against PyTorch (torch.nn.functional.ctc_loss for the total probability of a text).
//   collapse(path)      CTC's rule: merge repeated symbols, then drop the blanks (class 0)
//   forward(p, label)   log p(label) = log of the sum of the chances of every column path that collapses to label
//                       (the forward algorithm over the label with blanks around and between its symbols), and the
//                       number of such paths
import type { Tensor } from '../../atlas/types/index.ts';
import { SYMBOLS } from './glyphs.ts';

export const BLANK = 0;
export type Probs = { T: number; V: number; p: ArrayLike<number> };
export const probsOf = (t: Tensor | undefined | null): Probs | null => (t && t.shape.length === 2 ? { T: t.shape[0], V: t.shape[1], p: t.data } : null);
export const textOf = (ids: number[]): string => ids.map((k) => SYMBOLS[k]).join('');
export const idsOf = (text: string): number[] => Array.from(text, (c) => SYMBOLS.indexOf(c)).filter((k) => k > 0);

/** the top symbol of every column (ties go to the smallest class) */
export function bestPath(q: Probs): number[] {
	return Array.from({ length: q.T }, (_, t) => {
		let k = 0;
		for (let v = 1; v < q.V; v++) if (q.p[t * q.V + v] > q.p[t * q.V + k]) k = v;
		return k;
	});
}

/** merge repeats, then drop blanks */
export function collapse(path: number[]): number[] {
	const out: number[] = [];
	path.forEach((k, t) => {
		if (k !== BLANK && k !== path[t - 1]) out.push(k);
	});
	return out;
}

/** the log chance of one path: the sum of log p over its columns */
export function pathLog(q: Probs, path: number[]): number {
	let s = 0;
	path.forEach((k, t) => (s += Math.log(q.p[t * q.V + k])));
	return s;
}

const lse = (a: number, b: number) => (a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b))));

/** The extended label (blank, l1, blank, l2, …, blank) and whether state s may skip from s − 2 (a symbol that differs
 *  from the symbol two states back: between two equal symbols the blank cannot be skipped). */
function lattice(label: number[]) {
	const ext = [BLANK];
	for (const k of label) ext.push(k, BLANK);
	const skip = ext.map((k, s) => s >= 2 && k !== BLANK && k !== ext[s - 2]);
	return { ext, skip };
}

/** log p(label | columns) by the forward algorithm, and the number of paths that collapse to the label (0 when the
 *  label needs more columns than there are). */
export function forward(q: Probs, label: number[]): { logP: number; paths: number } {
	const { ext, skip } = lattice(label);
	const S = ext.length;
	let a = new Float64Array(S).fill(-Infinity);
	let n = new Float64Array(S);
	for (let s = 0; s < Math.min(2, S); s++) [a[s], n[s]] = [Math.log(q.p[ext[s]]), 1];
	for (let t = 1; t < q.T; t++) {
		const [b, m] = [new Float64Array(S).fill(-Infinity), new Float64Array(S)];
		for (let s = 0; s < S; s++) {
			let v = a[s];
			let c = n[s];
			if (s >= 1) [v, c] = [lse(v, a[s - 1]), c + n[s - 1]];
			if (skip[s]) [v, c] = [lse(v, a[s - 2]), c + n[s - 2]];
			[b[s], m[s]] = [v === -Infinity ? v : v + Math.log(q.p[t * q.V + ext[s]]), c];
		}
		[a, n] = [b, m];
	}
	const end = S >= 2 ? [S - 1, S - 2] : [S - 1];
	return { logP: end.reduce((v, s) => lse(v, a[s]), -Infinity), paths: end.reduce((c, s) => c + n[s], 0) };
}
