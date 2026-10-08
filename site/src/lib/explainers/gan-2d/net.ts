// gan-2d: plain float64 forward passes of the generator G and the discriminator D from recorded parameters (no
// autograd), for the views, the metrics and the lesson's numbers: every number shown comes from weights that ran.
// Layout as atlas/autograd/nn.ts MLP('gen', …) and MLP('disc', …): gen.0.weight [h, 2], gen.0.bias, gen.2.weight, …;
// tanh between layers, nothing after the last one (G outputs a point, D outputs a score, the "logit").
import type { Tensor } from '../../atlas/types/index.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { createRng } from '../../atlas/core/rng.ts';

export type Params = Record<string, Tensor>;
type Layer = { w: Tensor; b?: Tensor };
/** the plane every view draws: x and y from −SPAN to SPAN; RES cells per side for D's heat map */
export const SPAN = 1.25;
export const RES = 48;

function layersOf(params: Params, net: string): Layer[] {
	const out: Layer[] = [];
	for (let i = 0; params[`${net}.${i}.weight`]; i += 2) out.push({ w: params[`${net}.${i}.weight`], b: params[`${net}.${i}.bias`] });
	return out;
}

/** P rows of x [P·k] through MLP `net`: every layer's output (hidden ones after tanh). */
function forwardAll(params: Params, net: string, x: Float64Array, P: number): Float64Array[] {
	const outs: Float64Array[] = [];
	let h = x;
	const ls = layersOf(params, net);
	ls.forEach((l, li) => {
		const [O, I] = l.w.shape;
		const y = new Float64Array(P * O);
		for (let p = 0; p < P; p++)
			for (let o = 0; o < O; o++) {
				let s = l.b ? l.b.data[o] : 0;
				for (let i = 0; i < I; i++) s += h[p * I + i] * l.w.data[o * I + i];
				y[p * O + o] = li === ls.length - 1 ? s : Math.tanh(s);
			}
		outs.push((h = y));
	});
	return outs;
}

/** 0 = a D step, 1 = a G step: dSteps D steps, then one G step */
export const turnOf = (step: number, dSteps = 1) => (step % (dSteps + 1) < dSteps ? 0 : 1);

const last = (a: Float64Array[]) => a[a.length - 1] ?? new Float64Array(0);
export const hasNets = (params: Params | undefined): params is Params => !!params?.['gen.0.weight'] && !!params['disc.0.weight'];
/** G: P noise points z [P·2] -> P generated points [P·2] */
export const gen = (params: Params, z: Float64Array, P = z.length / 2) => last(forwardAll(params, 'gen', z, P));
/** D's raw scores (logits) for P points [P·2] */
export const logits = (params: Params, x: Float64Array, P = x.length / 2) => last(forwardAll(params, 'disc', x, P));
export const sigmoid = (l: number) => 1 / (1 + Math.exp(-l));
/** D(x): the discriminator's probability that each point is real */
export const scores = (params: Params, x: Float64Array, P = x.length / 2) => logits(params, x, P).map(sigmoid);
export function mean(a: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i];
	return a.length ? s / a.length : NaN;
}

/** ∇ₓ log D(x) at each of P points [P·2]: (1 − D(x)) · ∂logit/∂x (backpropagated by hand through D). The generator's
 *  loss −log D(G(z)) falls fastest when a generated point moves along this arrow: uphill on D's map. */
export function uphill(params: Params, x: Float64Array, P = x.length / 2): Float64Array {
	const ls = layersOf(params, 'disc');
	const outs = forwardAll(params, 'disc', x, P);
	const g = new Float64Array(P * 2);
	for (let p = 0; p < P; p++) {
		let d = Float64Array.of(1 - sigmoid(outs[outs.length - 1][p])); // d log σ(l) / d l
		for (let li = ls.length - 1; li >= 0; li--) {
			const [O, I] = ls[li].w.shape;
			if (li < ls.length - 1) for (let o = 0; o < O; o++) d[o] *= 1 - outs[li][p * O + o] ** 2; // through tanh
			const back = new Float64Array(I);
			for (let o = 0; o < O; o++) for (let i = 0; i < I; i++) back[i] += d[o] * ls[li].w.data[o * I + i];
			d = back;
		}
		g[2 * p] = d[0];
		g[2 * p + 1] = d[1];
	}
	return g;
}

/** a window of the plane: [x0, x1, y0, y1] */
export type Box = [number, number, number, number];
export const PLANE: Box = [-SPAN, SPAN, -SPAN, SPAN];
/** Cell centres of a res × res grid over a window of the plane, row 0 at the top. */
export function planeXY(res = RES, [x0, x1, y0, y1]: Box = PLANE): Float64Array {
	const xy = new Float64Array(res * res * 2);
	for (let i = 0; i < res * res; i++) {
		xy[2 * i] = x0 + (((i % res) + 0.5) / res) * (x1 - x0);
		xy[2 * i + 1] = y1 - ((Math.floor(i / res) + 0.5) / res) * (y1 - y0);
	}
	return xy;
}
/** D(x) over a window of the plane [res, res] (the heat map) */
export const dGrid = (params: Params, res = RES, box: Box = PLANE): Tensor => ({ shape: [res, res], data: scores(params, planeXY(res, box)) });
/** a square window around points [P·2] (and more), with a margin: the magnified view of the slowed-down steps */
export function boxAround(...sets: Float64Array[]): Box {
	let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
	for (const a of sets) for (let j = 0; j + 1 < a.length; j += 2) (x0 = Math.min(x0, a[j])), (x1 = Math.max(x1, a[j])), (y0 = Math.min(y0, a[j + 1])), (y1 = Math.max(y1, a[j + 1]));
	if (!(x1 >= x0)) return PLANE;
	const half = Math.max(0.05, 0.65 * Math.max(x1 - x0, y1 - y0));
	const [cx, cy] = [(x0 + x1) / 2, (y0 + y1) / 2];
	return [cx - half, cx + half, cy - half, cy + half];
}

/** The noise the views show: N_SHOW points drawn uniformly from the square [−1, 1]² (seeded, the same for everyone). */
export const N_SHOW = 240;
export const Z_SHOW: Float64Array = (() => {
	const rng = createRng(2026);
	return Float64Array.from({ length: 2 * N_SHOW }, () => 2 * rng.next() - 1);
})();

/** The generator's sheet: LINES lines of the noise square (z₁ = c and z₂ = c), PTS points each, mapped by G:
 *  [2·LINES, PTS, 2]. With no params, the square itself (what G maps). */
export const LINES = 9;
const PTS = 41;
export function sheet(params: Params | null): Tensor {
	const z = new Float64Array(2 * LINES * PTS * 2);
	let k = 0;
	for (let dir = 0; dir < 2; dir++)
		for (let l = 0; l < LINES; l++)
			for (let s = 0; s < PTS; s++) {
				const c = -1 + (2 * l) / (LINES - 1);
				const u = -1 + (2 * s) / (PTS - 1);
				z[k++] = dir ? u : c;
				z[k++] = dir ? c : u;
			}
	return { shape: [2 * LINES, PTS, 2], data: params ? gen(params, z) : z };
}

/** How well P generated points [P·2] match the real ones: recall = the share of real points with a generated point
 *  within r (how much of the data the generator reaches), precision = the share of generated points within r of a real
 *  point (how many land on the data). */
export const NEAR = 0.1;
export function match(real: Points2D, fake: Float64Array, r = NEAR): { recall: number; precision: number } {
	const P = fake.length / 2;
	if (!real.n || !P) return { recall: NaN, precision: NaN };
	const share = (a: Float64Array, b: Float64Array) => Array.from({ length: a.length / 2 }, (_, i) => nearCount(a[2 * i], a[2 * i + 1], b, r) > 0).filter(Boolean).length / (a.length / 2);
	return { recall: share(real.xy, fake), precision: share(fake, real.xy) };
}

/** a blob counts as covered when at least MODE_MIN of the shown generated points lie within MODE_R of its centre */
export const MODE_R = 0.2;
export const MODE_MIN = 5;
export function nearCount(cx: number, cy: number, fake: Float64Array, r = MODE_R): number {
	let k = 0;
	for (let j = 0; j < fake.length / 2; j++) if ((fake[2 * j] - cx) ** 2 + (fake[2 * j + 1] - cy) ** 2 <= r * r) k++;
	return k;
}
/** how many of the blob centres are covered (the modes the generator reaches) */
export const modesHit = (centres: [number, number][], fake: Float64Array) => centres.filter(([x, y]) => nearCount(x, y, fake) >= MODE_MIN).length;
