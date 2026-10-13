// realnvp-2d: processes (kind C) on the trained flow. Input: the final weights of a train source and its points
// (explainer.ts trained(); the sources run with after: '<train source>').
//   flow    data -> noise, one frame per coupling layer: the points and a grid over the data plane pushed through layers
//           0..k; metrics: point 0's numbers at layer k (u the kept coordinate, v the moving one, s, t, e^s, v' after,
//           ld the running Σ s) and the spread of the points; a last frame 'density' with log p on a DG × DG grid and
//           point 0's log N(z), Σ s and log p
//   sample  noise -> data: n draws z ~ N(0, I) (recorded draws) and a grid over the noise plane, pulled back through
//           layers 7..0; metrics: sample 0's numbers when each layer is undone; at the end how close the samples lie
//           to the data (mean distance, the share farther than 0.15 from every point, the share inside the rings' gap)
// pipeline/realnvp-2d/reference.py repeats both in float64 numpy with the replayed draws.
import type { PhaseDef, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { bandShare, couple, dataOf, DG, DSPAN, farShare, grid, keepOf, LAYERS, lines, logNormal, logp, meanNearest, momentsOf, nllOf, ZSPAN, type Params } from './nvp.ts';

export type FlowInput = { params: Params; xy: ArrayLike<number> } | null;
export type FlowState = { pts: Tensor; grid: Tensor; logp?: Tensor };
const L = (en: string, zh: string) => ({ en, zh });
/** the gap between the two rings of the 'circles' points (radius 0.45 and 0.9, noise 0.05): more than 3 noise widths
 *  from both */
export const GAP: [number, number] = [0.6, 0.75];
const T = (xy: Float64Array) => ({ shape: [xy.length / 2, 2], data: xy.slice() });
const coords = (xy: Float64Array) => ({ px: xy[0], py: xy[1] });

const FLOW: Record<string, PhaseDef> = {
	start: { label: L('Your points', '你的点'), kind: 'input' },
	couple: {
		label: L('One coupling layer', '一个耦合层'),
		kind: 'forward',
		formula: { tex: "v' = v\\,e^{s} + t = {{v}} \\times e^{ {{s}} } + {{t}} = {{v2}}", values: (f) => ({ v: f.metrics?.v ?? NaN, s: f.metrics?.s ?? NaN, t: f.metrics?.t ?? NaN, v2: f.metrics?.v2 ?? NaN }) }
	},
	density: { label: L('The exact log-likelihood', '精确的对数似然'), kind: 'output' }
};

export const flow: ProcessDef<FlowInput, Record<string, never>, FlowState> = {
	id: 'flow',
	kind: 'algorithm',
	title: L('Your points through the coupling layers', '你的点穿过耦合层'),
	phases: FLOW,
	schema: {
		pts: { kind: 'tensor', label: L('Points', '点'), shape: ['N', 2], role: 'activation', tol: 1e-9 },
		grid: { kind: 'tensor', label: L('Grid', '网格'), shape: ['G', 2], role: 'activation', tol: 1e-9 },
		logp: { kind: 'tensor', label: L('log p on a grid', '网格上的 log p'), shape: ['M'], role: 'output', tol: 1e-9 }
	},
	defaults: {},
	metricTol: 1e-9,
	*run(input): ProcessRun<FlowState, Record<string, never>> {
		const data = dataOf(input);
		if (!input?.params || !data.n) return;
		const p = input.params;
		const n = data.n;
		const pts = Float64Array.from(data.xy);
		const g = lines(DSPAN);
		const gn = g.length / 2;
		const ld = new Float64Array(n);
		yield { phase: 'start', step: 0, state: { pts: T(pts), grid: T(g) }, metrics: { n, ...coords(pts), ...momentsOf(pts, n) } };
		for (let k = 0; k < LAYERS; k++) {
			const c = keepOf(k);
			const [u, v] = [pts[c], pts[1 - c]];
			const { s, t } = couple(p, k, pts, n, 1, ld);
			couple(p, k, g, gn, 1);
			yield { phase: 'couple', step: k, state: { pts: T(pts), grid: T(g) }, metrics: { k, keep: c, u, v, s: s[0], t: t[0], es: Math.exp(s[0]), v2: pts[1 - c], ld: ld[0], ...coords(pts), ...momentsOf(pts, n) } };
		}
		const dg = grid(DG, DSPAN);
		const lp = logp(p, dg, DG * DG);
		const ln = logNormal(pts[0], pts[1]);
		yield { phase: 'density', step: LAYERS, state: { pts: T(pts), grid: T(g), logp: { shape: [DG * DG], data: lp } }, metrics: { zx: pts[0], zy: pts[1], logN: ln, ld: ld[0], logp: ln + ld[0], nll: nllOf(p, data), lpMax: Math.max(...lp), lpMin: Math.min(...lp) } };
	}
};

export type SampleOpt = { n: number };
const SAMPLE: Record<string, PhaseDef> = {
	noise: { label: L('Draw noise', '抽取噪声'), kind: 'sample' },
	uncouple: {
		label: L('Undo one coupling layer', '撤销一个耦合层'),
		kind: 'forward',
		formula: { tex: "v = (v' - t)\\,e^{-s} = ({{v2}} - {{t}}) \\times e^{-{{s}}} = {{v}}", values: (f) => ({ v2: f.metrics?.v2 ?? NaN, t: f.metrics?.t ?? NaN, s: f.metrics?.s ?? NaN, v: f.metrics?.v ?? NaN }) }
	}
};

export const sample: ProcessDef<FlowInput, SampleOpt, { x: Tensor; grid: Tensor }> = {
	id: 'sample',
	kind: 'algorithm',
	title: L('Sampling: noise back through the layers', '采样：噪声倒着穿过各层'),
	phases: SAMPLE,
	schema: {
		x: { kind: 'tensor', label: L('Samples', '样本'), shape: ['N', 2], role: 'output', tol: 1e-9 },
		grid: { kind: 'tensor', label: L('Grid', '网格'), shape: ['G', 2], role: 'activation', tol: 1e-9 }
	},
	defaults: { n: 500 },
	recordDraws: true,
	metricTol: 1e-9,
	*run(input, ctx): ProcessRun<{ x: Tensor; grid: Tensor }, SampleOpt> {
		const data = dataOf(input);
		if (!input?.params || !data.n) return;
		const p = input.params;
		const n = Math.max(1, Math.floor(ctx.opt.n));
		const x = new Float64Array(2 * n);
		for (let i = 0; i < 2 * n; i++) x[i] = ctx.rng.normal();
		const g = lines(ZSPAN);
		const gn = g.length / 2;
		yield { phase: 'noise', step: 0, state: { x: T(x), grid: T(g) }, metrics: { n, ...coords(x), logN: logNormal(x[0], x[1]) } };
		for (let j = 0; j < LAYERS; j++) {
			const k = LAYERS - 1 - j;
			const c = keepOf(k);
			const v2 = x[1 - c];
			const { s, t } = couple(p, k, x, n, -1);
			couple(p, k, g, gn, -1);
			const end: Record<string, number> = j === LAYERS - 1 ? { dist: meanNearest(x, n, data), far: farShare(x, n, data), gap: bandShare(x, n, GAP[0], GAP[1]) } : {};
			yield { phase: 'uncouple', step: j, state: { x: T(x), grid: T(g) }, metrics: { k, keep: c, u: x[c], v2, s: s[0], t: t[0], v: x[1 - c], ...coords(x), ...end } };
		}
	}
};
