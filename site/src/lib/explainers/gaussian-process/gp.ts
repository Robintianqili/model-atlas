// gaussian-process: the pieces every file shares (pure TS). The learner's input is a GpRaw: points on a 1-D plot (a
// preset or the learner's own clicks, in the order they were added), the kernel's two settings (length-scale ℓ and
// signal variance σf², edited on the stage) and the seed of the sample functions. dataOf() turns it into plain numbers,
// the input of the process (and, as JSON, of the Python reference): the points, the settings, the noise variance and
// the standard-normal draws that make the sample functions (drawn here, seeded, so Python reads the same numbers).
// The arithmetic (kernel, Cholesky, posterior, marginal likelihood, samples) is here too;
// pipeline/gaussian-process/reference.py repeats it with numpy's own linear algebra.
import type { I18n } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const I = (en: string, zh: string): I18n => ({ en, zh });
/** the plot (x across, y up), the grid the curves are drawn on, the number of sample functions, the most points */
export const XLO = -5;
export const XHI = 5;
export const YLO = -3;
export const YHI = 3;
export const G = 81;
export const NS = 3;
export const MAXP = 24;
export const GRID = Array.from({ length: G }, (_, g) => XLO + (g * (XHI - XLO)) / (G - 1));
/** the sliders: length-scale ℓ and signal variance σf²; the starting settings; the noise variance σn² (fixed); the
 *  jitter added to a covariance on the grid before sampling; the "large data set" of the cost step */
export const LS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3];
export const SF2 = [0.25, 0.5, 1, 2];
export const L0 = 1;
export const S0 = 1;
export const SN2 = 0.01;
export const JIT = 1e-6;
export const BIG = 10000;
/** the kernel-choice step: the step data with a short and a long length-scale */
export const SHORT = 0.5;
export const LONG = 2;

export type GpPreset = 'gap' | 'wiggle' | 'smooth' | 'step';
export interface Pt { x: number; y: number }
export interface GpRaw { preset?: GpPreset; points?: Pt[]; ls?: number; sf2?: number; seed?: number }
export interface GpData { x: number[]; y: number[]; ls: number; sf2: number; sn2: number; zPrior: number[]; zPost: number[] }

const r1 = (v: number) => Math.round(v * 10) / 10;
const mk = (xs: number[], f: (x: number) => number): Pt[] => xs.map((x) => ({ x, y: r1(f(x)) }));
const span = (a: number, step: number, n: number) => Array.from({ length: n }, (_, i) => r1(a + i * step));
export const PRESETS: { id: GpPreset; label: I18n; pts: Pt[] }[] = [
	{ id: 'gap', label: I('Five points, one wide gap', '五个点，中间一大段空白'), pts: mk([-1.2, 1.8, -3.6, 3.4, -2.4], (x) => 1.2 * Math.sin(0.7 * x) + 0.3) },
	{ id: 'wiggle', label: I('A fast wiggle', '快速起伏'), pts: mk(span(-4.5, 0.75, 13), (x) => 1.3 * Math.sin(2.2 * x)) },
	{ id: 'smooth', label: I('A gentle curve', '平缓的曲线'), pts: mk(span(-4.5, 1.5, 7), (x) => 0.08 * x * x + 0.25 * x - 1) },
	{ id: 'step', label: I('A sudden step', '突然的台阶'), pts: mk(span(-4.4, 0.8, 12), (x) => (x < 0 ? -1 : 1)) }
];
export const START: GpRaw = { preset: 'gap' };
export const STEP: GpRaw = { preset: 'step' };

const fin = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const rawOf = (raw: unknown): GpRaw => (raw && typeof raw === 'object' ? raw : {}) as GpRaw;
/** the points: the learner's (inside the plot, at most MAXP), else the preset's */
export function pointsOf(raw: unknown): Pt[] {
	const r = rawOf(raw);
	if (Array.isArray(r.points)) return r.points.filter((p) => fin(p?.x) && fin(p?.y)).slice(0, MAXP).map((p) => ({ x: clamp(p.x, XLO, XHI), y: clamp(p.y, YLO, YHI) }));
	return (PRESETS.find((p) => p.id === r.preset) ?? PRESETS[0]).pts;
}
const pickOf = (v: unknown, xs: number[], d: number) => (fin(v) && v > 0 ? xs.reduce((b, x) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(b / v)) ? x : b), xs[0]) : d);
export const lsOf = (raw: unknown) => pickOf(rawOf(raw).ls, LS, L0);
export const sf2Of = (raw: unknown) => pickOf(rawOf(raw).sf2, SF2, S0);
export const seedOf = (raw: unknown) => (fin(rawOf(raw).seed) ? Math.max(0, Math.floor(rawOf(raw).seed!)) : 1);

/** plain numbers for the process; `ls` overrides the learner's length-scale (the kernel-choice step) */
export function dataOf(raw: unknown, o: { ls?: number } = {}): GpData {
	const p = pointsOf(raw), rng = createRng(seedOf(raw));
	const z = () => Array.from({ length: NS * G }, () => rng.normal());
	const zPrior = z();
	return { x: p.map((q) => q.x), y: p.map((q) => q.y), ls: o.ls ?? lsOf(raw), sf2: sf2Of(raw), sn2: SN2, zPrior, zPost: z() };
}

/** the RBF (squared-exponential) kernel k(a, b) = σf² exp(−(a − b)² / (2ℓ²)) */
export const kern = (a: number, b: number, ls: number, sf2: number) => sf2 * Math.exp(-((a - b) * (a - b)) / (2 * ls * ls));
/** the n × n matrix K + σn² I (row-major) */
export function kmat(x: number[], ls: number, sf2: number, sn2: number): number[] {
	const n = x.length, K = new Array<number>(n * n);
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = kern(x[i], x[j], ls, sf2) + (i === j ? sn2 : 0);
	return K;
}
/** the Cholesky factor: lower-triangular L with A = L Lᵀ (A symmetric positive definite, row-major n × n) */
export function chol(A: number[], n: number): number[] {
	const L = new Array<number>(n * n).fill(0);
	for (let j = 0; j < n; j++) {
		let s = A[j * n + j];
		for (let k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
		const d = Math.sqrt(s);
		L[j * n + j] = d;
		for (let i = j + 1; i < n; i++) {
			let t = A[i * n + j];
			for (let k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
			L[i * n + j] = t / d;
		}
	}
	return L;
}
/** solve L v = b (forward substitution) */
export function fwd(L: number[], n: number, b: number[]): number[] {
	const v = new Array<number>(n);
	for (let i = 0; i < n; i++) {
		let s = b[i];
		for (let k = 0; k < i; k++) s -= L[i * n + k] * v[k];
		v[i] = s / L[i * n + i];
	}
	return v;
}
/** solve Lᵀ v = b (back substitution) */
export function bwd(L: number[], n: number, b: number[]): number[] {
	const v = new Array<number>(n);
	for (let i = n - 1; i >= 0; i--) {
		let s = b[i];
		for (let k = i + 1; k < n; k++) s -= L[k * n + i] * v[k];
		v[i] = s / L[i * n + i];
	}
	return v;
}

export interface Post { K: number[]; L: number[]; alpha: number[]; mean: number[]; sd: number[]; V: number[] }
/** the posterior of f at the points `at`: mean k*ᵀ α with α = (K + σn² I)⁻¹ y, variance σf² − vᵀv with v = L⁻¹ k*;
 *  V (n × at.length) keeps every v for the covariance of the samples */
export function posterior(x: number[], y: number[], ls: number, sf2: number, sn2: number, at: number[] = GRID): Post {
	const n = x.length, K = kmat(x, ls, sf2, sn2), L = chol(K, n), alpha = bwd(L, n, fwd(L, n, y));
	const mean: number[] = [], sd: number[] = [], V = new Array<number>(n * at.length);
	at.forEach((s, g) => {
		const ks = x.map((xi) => kern(s, xi, ls, sf2)), v = fwd(L, n, ks);
		let m = 0, q = 0;
		for (let i = 0; i < n; i++) (m += ks[i] * alpha[i]), (q += v[i] * v[i]), (V[i * at.length + g] = v[i]);
		mean.push(m);
		sd.push(Math.sqrt(Math.max(sf2 - q, 0)));
	});
	return { K, L, alpha, mean, sd, V };
}
/** log p(y | x, ℓ, σf²) = −½ yᵀα (data fit) − Σ log Lᵢᵢ (complexity, = −½ log det) − (n/2) log 2π */
export function evidence(x: number[], y: number[], ls: number, sf2: number, sn2: number): { fit: number; cpx: number; cst: number; lml: number } {
	const n = x.length, L = chol(kmat(x, ls, sf2, sn2), n), alpha = bwd(L, n, fwd(L, n, y));
	let fit = 0, cpx = 0;
	for (let i = 0; i < n; i++) (fit -= 0.5 * y[i] * alpha[i]), (cpx -= Math.log(L[i * n + i]));
	const cst = -0.5 * n * Math.log(2 * Math.PI);
	return { fit, cpx, cst, lml: fit + cpx + cst };
}
/** NS sample functions on the grid: mean + C z, with C the Cholesky factor of cov + JIT σf² I (cov row-major G × G) */
export function draw(mean: number[], cov: number[], z: number[], sf2: number): number[] {
	const A = cov.map((v, i) => v + (i % (G + 1) === 0 ? JIT * sf2 : 0)), C = chol(A, G), out = new Array<number>(NS * G);
	for (let s = 0; s < NS; s++)
		for (let g = 0; g < G; g++) {
			let v = mean[g];
			for (let k = 0; k <= g; k++) v += C[g * G + k] * z[s * G + k];
			out[s * G + g] = v;
		}
	return out;
}
/** the prior covariance on the grid, and the posterior one: K** − Vᵀ V */
export const priorCov = (ls: number, sf2: number) => GRID.flatMap((a) => GRID.map((b) => kern(a, b, ls, sf2)));
export function postCov(p: Post, n: number, ls: number, sf2: number): number[] {
	const C = priorCov(ls, sf2);
	for (let a = 0; a < G; a++)
		for (let b = 0; b < G; b++) {
			let s = 0;
			for (let i = 0; i < n; i++) s += p.V[i * G + a] * p.V[i * G + b];
			C[a * G + b] -= s;
		}
	return C;
}
