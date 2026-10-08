// Small measurement helpers for trainers and derived fields (W1-B1). All run without recording a graph.
import type { Tensor } from '../types/index.ts';
import { Var, input, noGrad } from '../autograd/var.ts';
import { softmax } from '../autograd/ops.ts';

/** Share of rows whose largest logit is the label (ties go to the smallest index). */
export function accuracy(logits: Var | Tensor, labels: ArrayLike<number>): number {
	const t = logits instanceof Var ? logits.value : logits;
	const C = t.shape[t.shape.length - 1];
	const N = t.data.length / C;
	if (N === 0) return NaN;
	let right = 0;
	for (let i = 0; i < N; i++) {
		let best = 0;
		for (let c = 1; c < C; c++) if (t.data[i * C + c] > t.data[i * C + best]) best = c;
		if (best === labels[i]) right++;
	}
	return right / N;
}

/** sqrt(Σ g²) over every parameter gradient. */
export function gradNorm(params: Var[]): number {
	let s = 0;
	for (const p of params) {
		if (!p.grad) continue;
		for (const g of p.grad.data) s += g * g;
	}
	return Math.sqrt(s);
}

export function paramCount(m: { params(): { value: Tensor }[] }): number {
	return m.params().reduce((p, v) => p + v.value.data.length, 0);
}

export interface Bounds {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}

/** Cell centres of a res×res grid over the bounds, row-major from the top row (y1) down: [res·res, 2]. */
export function gridPoints(b: Bounds, res: number): Tensor {
	const data = new Float64Array(res * res * 2);
	for (let r = 0; r < res; r++) {
		const y = b.y1 - ((r + 0.5) / res) * (b.y1 - b.y0);
		for (let c = 0; c < res; c++) {
			const x = b.x0 + ((c + 0.5) / res) * (b.x1 - b.x0);
			data[(r * res + c) * 2] = x;
			data[(r * res + c) * 2 + 1] = y;
		}
	}
	return { shape: [res * res, 2], data };
}

/**
 * The probability of `cls` on a res×res grid (for a decision-region view): forward maps [P, 2] points to [P, C] logits.
 * Returns [res, res] (row 0 = top). Use it in a TrainerDef.derived field: derive: (s) => decisionGrid(…).
 */
export function decisionGrid(forward: (x: Var) => Var, b: Bounds, res: number, cls = 1): Tensor {
	return noGrad(() => {
		const pts = gridPoints(b, res);
		const p = softmax(forward(input(pts)), -1);
		const C = p.shape[1];
		const out = new Float64Array(res * res);
		for (let i = 0; i < out.length; i++) out[i] = p.data[i * C + cls];
		return { shape: [res, res], data: out };
	});
}
