// loss-gradient-descent: the toy's data and the plain arithmetic of its error score, shared by the process (which
// computes the same score and gradient with the kit's autograd), the dials widget, the views and the claim tests.
// Eight points (x, y): y is 1.2·x + 0.4 plus a small fixed wobble, written out so every learner sees the same points.
// No third-party data.
import { fmt3, fmtInt } from '../../atlas/core/format.ts';

export const XS = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
export const YS = [-1.25, -1.0, -0.1, 0.3, 1.25, 1.45, 2.25, 2.7];
export const N = XS.length;

/** The dial ranges of the widget and the plotted window of the error landscape (w across, b up). */
export const W_RANGE: [number, number] = [-2, 4];
export const B_RANGE: [number, number] = [-2, 3];

/** The task's goal: an error score below this (a round number the learner can reach by hand; the best is about 0.022). */
export const TARGET = 0.1;

export interface Dials {
	w: number;
	b: number;
}

const finite = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v));

/** Any raw input -> dials inside the ranges: { w, b } objects or [w, b] pairs; anything else is the flat line w = b = 0. */
export function dialsOf(raw: unknown): Dials {
	if (Array.isArray(raw)) return { w: clamp(finite(raw[0], 0), W_RANGE), b: clamp(finite(raw[1], 0), B_RANGE) };
	const o = (raw && typeof raw === 'object' ? raw : {}) as { w?: unknown; b?: unknown };
	return { w: clamp(finite(o.w, 0), W_RANGE), b: clamp(finite(o.b, 0), B_RANGE) };
}

/** Each point's error, prediction minus answer. */
export function errors(w: number, b: number): number[] {
	return XS.map((x, i) => w * x + b - YS[i]);
}

/** The error score: the mean of the squared errors (MSE). */
export function scoreOf(w: number, b: number): number {
	let s = 0;
	for (let i = 0; i < N; i++) {
		const e = w * XS[i] + b - YS[i];
		s += e * e;
	}
	return s / N;
}

/** The slope of the score under each dial, by the formulas dL/dw = mean(2·e·x), dL/db = mean(2·e) (the process gets the
 *  same numbers from autograd; the claim tests compare the two). */
export function gradientOf(w: number, b: number): { gw: number; gb: number } {
	let gw = 0;
	let gb = 0;
	for (let i = 0; i < N; i++) {
		const e = w * XS[i] + b - YS[i];
		gw += (2 * e * XS[i]) / N;
		gb += (2 * e) / N;
	}
	return { gw, gb };
}

/** The best line (least squares) and its score: the bottom of the bowl. */
export function bestLine(): { w: number; b: number; score: number } {
	const mx = XS.reduce((a, x) => a + x, 0) / N;
	const my = YS.reduce((a, y) => a + y, 0) / N;
	let sxy = 0;
	let sxx = 0;
	for (let i = 0; i < N; i++) {
		sxy += (XS[i] - mx) * (YS[i] - my);
		sxx += (XS[i] - mx) * (XS[i] - mx);
	}
	const w = sxy / sxx;
	const b = my - w * mx;
	return { w, b, score: scoreOf(w, b) };
}

/** The point the line misses most (ties to the smallest index), and that miss. */
export function worstPoint(w: number, b: number): { i: number; e: number } {
	const es = errors(w, b);
	let i = 0;
	for (let k = 1; k < N; k++) if (Math.abs(es[k]) > Math.abs(es[i])) i = k;
	return { i, e: es[i] };
}

/** The largest step size (learning rate) at which gradient descent on this score still settles: 2 / the largest
 *  curvature. The score is a bowl with curvature 2·[[mean x², mean x], [mean x, 1]]; above this rate every step
 *  overshoots the bottom along the steepest direction by more than it gained. */
export function maxStableRate(): number {
	const mx = XS.reduce((a, x) => a + x, 0) / N;
	const mxx = XS.reduce((a, x) => a + x * x, 0) / N;
	const tr = mxx + 1;
	const det = mxx - mx * mx;
	const top = 2 * ((tr + Math.sqrt(tr * tr - 4 * det)) / 2);
	return 2 / top;
}

/** Narration helpers shared by lesson.ts and concept.ts: a frame's metrics; a learning rate as written (0.1, not
 *  0.100; NaN, shown as "…", until computed); a legend label "η = 0.1". */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
export const rate = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);
export const eta = (v: number) => ({ en: `η = ${v}`, zh: `η = ${v}` });
/** The line as the narration prints it: "1.2·x + 0.4", "3·x − 1" (NaN, shown as "…", until the dials are known). */
const show = (v: number) => (Number.isInteger(v) ? fmtInt(v) : fmt3(v)); // as a narration placeholder prints a number
export const lineText = (w: number, b: number): string | number =>
	Number.isFinite(w) && Number.isFinite(b) ? `${show(w)}·x ${b < 0 ? '−' : '+'} ${show(Math.abs(b))}` : NaN;
