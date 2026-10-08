// matrix-factorization: process 'nudge' (kind C). One step of the trainer's stochastic gradient descent, slowed down into
// its events, on one rating: your rating with the largest size (ties: the first movie), else, when you rated nothing,
// person 0's rating with the largest size (passed as opt.fallback, so the Python reference gets it
// with the options). It starts from the trainer's own starting numbers (the same seed and initFactors on
// rng.fork('init'), so it is step 0 of 'fit' had it drawn this rating):
//   start (the two pairs of numbers) -> error (prediction u·v, error e = pred − r, loss e²) -> gradient (2·e·v for u,
//   2·e·u for v) -> update (each number − lr × its gradient, and the new prediction).
// Every number the narration or a formula needs is in that frame's metrics. pipeline/matrix-factorization/reference.py
// is an independent Python run of the same steps (atlas parity, AK204).   npm run atlas -- run matrix-factorization nudge
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { OTHERS, YOU, initFactors, toRaw } from './world.ts';
import { fit } from './trainer.ts';

export type NudgeOpt = { lr: number; init: number; fallback: [number, number, number] };
export type NudgeState = { u: Tensor; v: Tensor };
const T = (a: number, b: number): Tensor => ({ shape: [2], data: Float64Array.of(a, b) });
/** person 0's rating with the largest size (the first such): the example when you rated nothing */
const FALLBACK = OTHERS.filter((x) => x.u === 0).reduce((b, x) => (Math.abs(x.r) > Math.abs(b.r) ? x : b));

/** The rating the step uses: { user, item, r }. */
export function nudgeRating(raw: unknown, fb = nudge.defaults.fallback): { user: number; item: number; r: number } {
	const mine = toRaw(raw).ratings;
	let best = -1;
	mine.forEach((r, i) => r !== null && (best < 0 || Math.abs(r) > Math.abs(mine[best]!)) && (best = i));
	return best >= 0 ? { user: YOU, item: best, r: mine[best]! } : { user: fb[0], item: fb[1], r: fb[2] };
}

export const nudge: ProcessDef<unknown, NudgeOpt, NudgeState> = {
	id: 'nudge',
	kind: 'training',
	title: { en: 'One gradient-descent step on one rating', zh: '在一个评分上走一步梯度下降' },
	phases: {
		start: {
			label: { en: 'Starting numbers', zh: '起始的数' },
			kind: 'input',
			formula: { tex: 'u = ({{u1}},\\ {{u2}});\\quad v = ({{v1}},\\ {{v2}})', values: (f) => f.metrics! },
			narration: { what: { en: 'The person starts at ({frame.u1}, {frame.u2}) and the movie at ({frame.v1}, {frame.v2}), random numbers.', zh: '这个人起始于 ({frame.u1}, {frame.u2})，这部电影起始于 ({frame.v1}, {frame.v2})，都是随机数。' } }
		},
		error: {
			label: { en: 'Prediction and error', zh: '预测与误差' },
			kind: 'loss',
			formula: {
				// one product per line, so every filled line fits a 390 px phone
				tex: 'u_1 v_1 = {{u1}} \\times ({{v1}}) = {{p1}};\\quad u_2 v_2 = {{u2}} \\times ({{v2}}) = {{p2}};\\quad \\hat r = u_1 v_1 + u_2 v_2 = {{p1}} + ({{p2}}) = {{pred}};\\quad e = \\hat r - r = {{pred}} - ({{r}}) = {{e}};\\quad L = e^2 = {{loss}}',
				values: (f) => f.metrics!
			},
			narration: { what: { en: 'The prediction is {frame.pred} for a rating of {frame.r}: the error is {frame.e} and the loss {frame.loss}.', zh: '预测是 {frame.pred}，实际评分是 {frame.r}：误差是 {frame.e}，损失是 {frame.loss}。' } }
		},
		gradient: {
			label: { en: 'Gradients', zh: '梯度' },
			kind: 'backward',
			formula: { tex: 'g_{u_1} = 2 e\\, v_1 = 2 \\times ({{e}}) \\times ({{v1}}) = {{gu1}};\\quad g_{v_1} = 2 e\\, u_1 = 2 \\times ({{e}}) \\times ({{u1}}) = {{gv1}}', values: (f) => f.metrics! },
			narration: { what: { en: 'The gradient of the loss for the person’s first number is 2 × e × v₁ = {frame.gu1}.', zh: '损失对这个人第一个数的梯度是 2 × e × v₁ = {frame.gu1}。' } }
		},
		update: {
			label: { en: 'Update', zh: '更新' },
			kind: 'update',
			formula: { tex: 'g_{u_1} = 2 e\\, v_1 = 2 \\times ({{e}}) \\times ({{v1}}) = {{gu1}};\\quad u_1^{\\text{new}} = u_1 - \\eta\\, g_{u_1} = {{u1}} - {{lr}} \\times ({{gu1}}) = {{nu1}};\\quad \\hat r = {{pred}};\\quad \\hat r_{\\text{new}} = u^{\\text{new}} \\cdot v^{\\text{new}} = {{predAfter}}', values: (f) => f.metrics! },
			narration: { what: { en: 'Every number takes a step against its gradient; the prediction moves from {frame.pred} to {frame.predAfter}.', zh: '每个数都朝梯度的反方向走一步；预测从 {frame.pred} 变成 {frame.predAfter}。' } }
		}
	},
	schema: {
		u: { kind: 'tensor', label: { en: 'The person’s two numbers', zh: '这个人的两个数' }, shape: [2], role: 'param', tol: 1e-12 },
		v: { kind: 'tensor', label: { en: 'The movie’s two numbers', zh: '这部电影的两个数' }, shape: [2], role: 'param', tol: 1e-12 }
	},
	defaults: { lr: fit.defaults.lr, init: fit.defaults.init, fallback: [FALLBACK.u, FALLBACK.i, FALLBACK.r] },
	*run(input, ctx): ProcessRun<NudgeState, NudgeOpt> {
		const { lr, init } = ctx.opt;
		const { U, V } = initFactors(ctx.rng.fork('init'), init);
		const { user, item, r } = nudgeRating(input, ctx.opt.fallback);
		const [u1, u2, v1, v2] = [U[2 * user], U[2 * user + 1], V[2 * item], V[2 * item + 1]];
		const m: Record<string, number> = { user, item, r, u1, u2, v1, v2, lr };
		yield { phase: 'start', step: 0, state: { u: T(u1, u2), v: T(v1, v2) }, metrics: { ...m } };
		const p1 = u1 * v1, p2 = u2 * v2, pred = p1 + p2, e = pred - r;
		Object.assign(m, { p1, p2, pred, e, loss: e * e });
		yield { phase: 'error', step: 0, state: { u: T(u1, u2), v: T(v1, v2) }, metrics: { ...m } };
		const [gu1, gu2, gv1, gv2] = [2 * e * v1, 2 * e * v2, 2 * e * u1, 2 * e * u2];
		Object.assign(m, { gu1, gu2, gv1, gv2 });
		yield { phase: 'gradient', step: 0, state: { u: T(u1, u2), v: T(v1, v2) }, metrics: { ...m } };
		const [nu1, nu2, nv1, nv2] = [u1 - lr * gu1, u2 - lr * gu2, v1 - lr * gv1, v2 - lr * gv2];
		const predAfter = nu1 * nv1 + nu2 * nv2, eAfter = predAfter - r;
		Object.assign(m, { nu1, nu2, nv1, nv2, predAfter, eAfter, lossAfter: eAfter * eAfter });
		yield { phase: 'update', step: 0, state: { u: T(nu1, nu2), v: T(nv1, nv2) }, metrics: { ...m } };
	}
};
