// pinn: the physics and the network math every file shares. The oscillator is the kit's (atlas/adapters/series OSCILLATOR,
// ζ = 0.15, ω = 12): a weight on a spring with friction, m·u″ + c·u′ + k·u = 0 with m = 1, c = 2ζω = 3.6, k = ω² = 144,
// let go at u = 1 from rest. u′ and u″ of the network come from the kit's autograd: grad(u, t, { createGraph: true })
// twice (the same code path the PINN trains with). Pure TS.
import type { Tensor } from '../../atlas/types/index.ts';
import type { MLP } from '../../atlas/autograd/nn.ts';
import { grad, input, noGrad, withGradMode, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { dampedOscillator, linspace } from '../../atlas/data/series.ts';
import { OSCILLATOR, seriesOf } from '../../atlas/adapters/series/adapter.ts';

/** damping c and stiffness k of the true law (mass m = 1) */
export const C = 2 * OSCILLATOR.zeta * OSCILLATOR.omega;
export const K = OSCILLATOR.omega * OSCILLATOR.omega;
/** the physics points: 200 evenly spaced times on [0, 1] (also where every curve is drawn) */
export const N = 200;
export const T = linspace(0, 1, N);
/** the true curve on T (the analytic solution; the networks never see it) */
export const TRUTH = dampedOscillator(T, OSCILLATOR);
/** the preset points lie in the first 40 % of the time; errors "after the data" are measured on t > SPLIT */
export const SPLIT = 0.4;

export type Pts = { n: number; t: Float64Array; u: Float64Array };
/** the learner's observation points (the series adapter's raw value) */
export function pointsOf(raw: unknown): Pts {
	const p = seriesOf(raw).points;
	return { n: p.length, t: Float64Array.from(p, (q) => q.t), u: Float64Array.from(p, (q) => q.y) };
}

/** u, u′ and u″ of the network at the times `ts` (Var [n, 1] each), by autograd: grad twice with createGraph. */
export function derivs(m: MLP, ts: ArrayLike<number>, graph = true): { t: Var; u: Var; du: Var; d2u: Var } {
	const t = input({ shape: [ts.length, 1], data: ts }, true);
	const u = m.forward(t);
	const du = grad(u, t, { createGraph: true });
	const d2u = grad(du, t, { createGraph: graph });
	return { t, u, du, d2u };
}

/** the residual of the law, divided by k so it reads in units of position: (u″ + c·u′ + k·u) / k */
export const residual = (d: { u: Var; du: Var; d2u: Var }, c: number, k: number): Var => op.scale(op.add(op.add(d.d2u, op.scale(d.du, c)), op.scale(d.u, k)), 1 / k);

/** u, u′, u″ and the residual on T as [4, N] (rows in that order): what the views draw. */
export function curves(m: MLP, c: number, k: number): Tensor {
	return withGradMode(true, () => {
		const d = derivs(m, T, false);
		const r = residual(d, c, k);
		const out = new Float64Array(4 * N);
		[d.u, d.du, d.d2u, r].forEach((v, i) => out.set(v.data, i * N));
		return { shape: [4, N], data: out };
	});
}

/** the network's value at a few times (no graph) */
export const valuesAt = (m: MLP, ts: ArrayLike<number>): Float64Array => noGrad(() => m.forward(input({ shape: [ts.length, 1], data: ts })).data.slice());

/** mean |u − truth| over the times of T in (a, b] */
export function gap(u: ArrayLike<number>, a: number, b: number): number {
	let s = 0;
	let n = 0;
	for (let i = 0; i < N; i++)
		if (T[i] > a && T[i] <= b) {
			s += Math.abs(u[i] - TRUTH[i]);
			n++;
		}
	return n ? s / n : NaN;
}

/** mean square of row `row` of a [4, N] curves tensor (row 3: the physics loss of those weights) */
export function meanSq(cv: Tensor, row: number): number {
	let s = 0;
	for (let i = 0; i < N; i++) s += cv.data[row * N + i] ** 2;
	return s / N;
}
