// score-2d: process 'langevin' (kind C). Langevin chains on the trained score network: the input is the final weights
// of the train source (the sources run it with after: 'train') and the learner's points (for the metrics). Every
// source runs it with seed 1, so all start from the same uniform points. One frame per step:
//   start   the starts; metrics dist (mean distance to the nearest data point), far (share of chains farther than 0.2
//           from every data point), cover (share of data points with a chain within 0.1), spread (mean distance of
//           the chains from their mean)
//   move    x ← x + (α/2)·s(x, σ) + noise·√α·z for every chain; metrics: σ, level, α and chain 0's numbers (px, py
//           before; sx, sy its score; zx, zy its noise draw; nx, ny after); dist, far, cover and spread at the end of
//           each level
// pipeline/score-2d/reference.py repeats it in a plain Python loop with ak.Rng and the same seed (no replayed draws):
// normal() may differ from the TS one in the last bit (libm), so x and the metrics are compared at 1e-9.
import type { PhaseDef, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { chains, coverage, dataOf, limitsOf, meanNearest, SHOW, SMIN, type LangOpt, type Params } from './score.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';

export type LangState = { x: Tensor };
export type LangInput = { params: Params; xy: ArrayLike<number> } | null;
const L = (en: string, zh: string) => ({ en, zh });

/** the share of n points farther than r from every data point */
export function farShare(xy: ArrayLike<number>, n: number, data: Points2D, r = 0.2): number {
	let far = 0;
	for (let i = 0; i < n; i++) {
		let j = 0;
		while (j < data.n && (xy[2 * i] - data.xy[2 * j]) ** 2 + (xy[2 * i + 1] - data.xy[2 * j + 1]) ** 2 >= r * r) j++;
		if (j === data.n) far++;
	}
	return n ? far / n : NaN;
}
/** the mean distance of n points from their mean */
function spreadOf(xy: Float64Array, n: number): number {
	let [mx, my, s] = [0, 0, 0];
	for (let i = 0; i < n; i++) (mx += xy[2 * i] / n), (my += xy[2 * i + 1] / n);
	for (let i = 0; i < n; i++) s += Math.hypot(xy[2 * i] - mx, xy[2 * i + 1] - my) / n;
	return s;
}
const stats = (x: Float64Array, d: Points2D) => ({ dist: meanNearest(x, x.length / 2, d), far: farShare(x, x.length / 2, d), cover: coverage(x, x.length / 2, d), spread: spreadOf(x, x.length / 2) });

const PHASES: Record<string, PhaseDef> = {
	start: { label: L('Start anywhere', '从任意位置出发'), kind: 'input' },
	move: {
		label: L('One Langevin step', '朗之万法走一步'),
		kind: 'update',
		formula: { tex: 'x \\leftarrow x + \\tfrac{\\alpha}{2}\\,s_\\theta(x, \\sigma) + \\sqrt{\\alpha}\\,z,\\quad \\sigma = {{sigma}},\\ \\alpha = {{alpha}}', values: (f) => ({ sigma: f.metrics?.sigma ?? NaN, alpha: f.metrics?.alpha ?? NaN }) }
	}
};

export const langevin: ProcessDef<LangInput, LangOpt, LangState> = {
	id: 'langevin',
	kind: 'algorithm',
	title: L('Langevin chains on the learned score', '沿学到的分数走的朗之万链'),
	phases: PHASES,
	schema: { x: { kind: 'tensor', label: L('Chains', '链'), shape: ['N', 2], role: 'output', tol: 1e-9 } },
	defaults: { sigmas: [SHOW], T: 100, alpha: 2e-4, noise: 1, chains: 300 },
	metricTol: 1e-9,
	*run(input, ctx): ProcessRun<LangState, LangOpt> {
		const data = dataOf(input);
		if (!input?.params || !data.n) return;
		const T = Math.max(1, Math.floor(ctx.opt.T));
		for (const m of chains(input.params, { ...ctx.opt, T }, ctx.rng)) {
			if (ctx.signal.cancelled) return;
			const state = { x: { shape: [m.x.length / 2, 2], data: m.x } };
			if (m.k < 0) yield { phase: 'start', step: 0, state, metrics: { chains: m.x.length / 2, ...stats(m.x, data) } };
			else yield { phase: 'move', step: m.k, state, metrics: { sigma: m.s, level: m.level, alpha: m.a, ...m.c0, nx: m.x[0], ny: m.x[1], ...((m.k + 1) % T === 0 ? stats(m.x, data) : {}) } };
		}
	}
};

/** process 'limits': the learned score against the true one where no noisy copy lands (score.ts limitsOf), measured in the
 *  worker (about 1.5 million exponentials: a long task on the main thread). One frame; no Python reference. */
export const limits: ProcessDef<LangInput, { sigma: number }, Record<string, never>> = {
	id: 'limits',
	kind: 'algorithm',
	title: L('The learned score against the true one', '学到的分数和真实分数'),
	phases: { measure: { label: L('Compare the arrows', '比较箭头'), kind: 'observe' } },
	schema: {},
	defaults: { sigma: SMIN },
	*run(input, ctx): ProcessRun<Record<string, never>, { sigma: number }> {
		const data = dataOf(input);
		if (input?.params && data.n) yield { phase: 'measure', step: 0, state: {}, metrics: limitsOf(input.params, data, ctx.opt.sigma) };
	}
};
