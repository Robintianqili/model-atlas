// contrastive-learning: the numbers views and narration share (pure TS). A batch's cosines C are N×N, row-major:
// C[i * N + j] is drawing i with caption j, and the true pairs sit on the diagonal.
import type { At, Tensor } from '../../atlas/types/index.ts';
import { TAU_RANGE } from './adapters/batch/adapter.ts';

export const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
export const diagOf = (a: ArrayLike<number>, N: number) => Array.from({ length: N }, (_, i) => a[i * N + i]);
export const offOf = (a: ArrayLike<number>, N: number) => Array.from(a).filter((_, k) => k % (N + 1) !== 0);

/** every row i against every column j: the sum of the products of their D numbers */
export function cosines(I: ArrayLike<number>, T: ArrayLike<number>, N: number, M: number, D: number): Float64Array {
	const C = new Float64Array(N * M);
	for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) {
		let s = 0;
		for (let d = 0; d < D; d++) s += I[i * D + d] * T[j * D + d];
		C[i * M + j] = s;
	}
	return C;
}

/** The largest entry off the diagonal and where it is (the first one on a tie). */
export function offMax(a: ArrayLike<number>, N: number) {
	let best = { v: -Infinity, i: 0, j: 1 };
	for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j && a[i * N + j] > best.v) best = { v: a[i * N + j], i, j };
	return best;
}

/** Rows (drawings) whose own caption has the strictly highest cosine, and columns (captions) whose own drawing has. */
export function wins(a: ArrayLike<number>, N: number) {
	let rows = 0, cols = 0;
	for (let i = 0; i < N; i++) {
		let r = true, c = true;
		for (let j = 0; j < N; j++) if (j !== i) {
			if (a[i * N + j] >= a[i * N + i]) r = false;
			if (a[j * N + i] >= a[i * N + i]) c = false;
		}
		rows += +r;
		cols += +c;
	}
	return { rows, cols };
}

/** −log of the softmax share of entry k in xs (stable) */
function nll(xs: number[], k: number): number {
	const m = Math.max(...xs);
	return Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)) + m - xs[k];
}
/** The symmetric InfoNCE loss of cosines C at temperature τ: the mean over rows and over columns of −log softmax(C/τ)
 *  of the true pair, the two averaged (CLIP's loss; the nce process computes the same with the kit's autograd). */
export function lossAt(C: ArrayLike<number>, N: number, tau: number): number {
	let r = 0, c = 0;
	for (let i = 0; i < N; i++) {
		r += nll(Array.from({ length: N }, (_, j) => C[i * N + j] / tau), i);
		c += nll(Array.from({ length: N }, (_, j) => C[j * N + i] / tau), i);
	}
	return (r / N + c / N) / 2;
}

/** The loss over the slider's whole range of τ (log-spaced), and its lowest point. */
export function tauCurve(C: ArrayLike<number>, N: number, n = 61) {
	const [a, b] = TAU_RANGE.map(Math.log10);
	const taus = Array.from({ length: n }, (_, k) => 10 ** (a + ((b - a) * k) / (n - 1)));
	const loss = taus.map((t) => lossAt(C, N, t));
	let k = 0;
	for (let i = 1; i < n; i++) if (loss[i] < loss[k]) k = i;
	return { taus, loss, best: { tau: taus[k], loss: loss[k], atEdge: k === 0 } };
}

/** The state of the nce source at the frame of a step, by field. */
export const field = (at: At, k: string): Tensor | undefined => (at.tape && at.index >= 0 ? (at.tape.state(at.index) as Record<string, Tensor | undefined>)[k] : undefined);
/** A frame's metric (NaN when missing). */
export const met = (at: At, k: string) => Number(at.frame?.metrics?.[k] ?? NaN);
