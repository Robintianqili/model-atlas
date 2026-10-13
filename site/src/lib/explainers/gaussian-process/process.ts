// gaussian-process: process 'gp' (kind C). Gaussian process regression with an RBF kernel on the learner's points, one
// frame per event; pipeline/gaussian-process/reference.py repeats it with numpy, frame by frame.
//   prior     no data yet: mean 0, band ±2σf everywhere, three functions drawn from the prior
//   kernel    the matrix K + σn² I of every pair of points, and the weights α = (K + σn² I)⁻¹ y
//   add       point i joins: the posterior mean and band from points 0..i (one frame per point, in the order added)
//   sample    three functions drawn from the posterior
//   evidence  the log marginal likelihood (data fit + complexity + constant) for every length-scale on the slider
//   done      the summary: the widest gap between neighbouring points, the cost of the solve, the overshoot
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { BIG, draw, evidence, G, GRID, I, kern, LS, NS, postCov, posterior, priorCov, type GpData } from './gp.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });
export type GpState = { x: Tensor; y: Tensor; mean: Tensor; sd: Tensor; samples: Tensor; K: Tensor; alpha: Tensor; lmls: Tensor };
const lab = (en: string, zh: string) => ({ label: I(en, zh) });
const range = (s: number[], g: number) => {
	const v = Array.from({ length: NS }, (_, k) => s[k * G + g]);
	return Math.max(...v) - Math.min(...v);
};
const near = (x: number) => GRID.reduce((b, v, g) => (Math.abs(v - x) < Math.abs(GRID[b] - x) ? g : b), 0);

export const gp: ProcessDef<GpData, Record<string, never>, GpState> = {
	id: 'gp',
	kind: 'algorithm',
	title: I('Gaussian process regression, point by point', '高斯过程回归，一个点一个点地加'),
	metricTol: 1e-9,
	phases: {
		prior: { ...lab('Before any data: the prior', '还没有数据：先验'), kind: 'input' },
		kernel: { ...lab('The kernel matrix and the weights', '核矩阵和权重'), kind: 'forward' },
		add: { ...lab('A point joins: the posterior', '加入一个点：后验'), kind: 'update' },
		sample: { ...lab('Functions drawn from the posterior', '从后验中抽取函数'), kind: 'sample' },
		evidence: { ...lab('The marginal likelihood', '边缘似然'), kind: 'loss' },
		done: { ...lab('Summary', '总结'), kind: 'output' }
	},
	schema: {
		x: { kind: 'tensor', ...lab('x of each point, in the order added', '每个点的 x（按加入的顺序）'), role: 'data', tol: 'exact' },
		y: { kind: 'tensor', ...lab('y of each point', '每个点的 y'), role: 'target', tol: 'exact' },
		mean: { kind: 'tensor', ...lab('Posterior mean on the grid', '网格上的后验均值'), tol: 1e-9 },
		sd: { kind: 'tensor', ...lab('Posterior standard deviation on the grid', '网格上的后验标准差'), tol: 1e-9 },
		samples: { kind: 'tensor', ...lab('Sample functions on the grid', '网格上的样本函数'), tol: 1e-6 },
		K: { kind: 'tensor', ...lab('Kernel matrix K + σn² I', '核矩阵 K + σn² I'), tol: 1e-12 },
		alpha: { kind: 'tensor', ...lab('Weights α = (K + σn² I)⁻¹ y', '权重 α = (K + σn² I)⁻¹ y'), tol: 1e-9 },
		lmls: { kind: 'tensor', ...lab('Log marginal likelihood for each length-scale on the slider', '滑块上每个长度尺度的对数边缘似然'), tol: 1e-9 }
	},
	defaults: {},
	*run(d): ProcessRun<GpState, Record<string, never>> {
		const { x, y, ls, sf2, sn2 } = d, n = x.length;
		if (!n) return;
		const sd0 = Math.sqrt(sf2), kd = sf2 + sn2;
		yield { phase: 'prior', step: 0, state: { x: T([n], x), y: T([n], y), mean: T([G], GRID.map(() => 0)), sd: T([G], GRID.map(() => sd0)), samples: T([NS, G], draw(GRID.map(() => 0), priorCov(ls, sf2), d.zPrior, sf2)) }, metrics: { n, ls, sf2, sn2, sd0, band0: 2 * sd0 } };
		// every pair: the closest and the farthest (ties to the smallest indices)
		const all = posterior(x, y, ls, sf2, sn2);
		const pair = { iN: 0, jN: 0, dN: Infinity, iF: 0, jF: 0, dF: -1 };
		for (let i = 0; i < n; i++)
			for (let j = i + 1; j < n; j++) {
				const dd = Math.abs(x[i] - x[j]);
				if (dd < pair.dN) Object.assign(pair, { iN: i, jN: j, dN: dd });
				if (dd > pair.dF) Object.assign(pair, { iF: i, jF: j, dF: dd });
			}
		const pm: Record<string, number> = n > 1 ? { ...pair, kN: kern(x[pair.iN], x[pair.jN], ls, sf2), kF: kern(x[pair.iF], x[pair.jF], ls, sf2) } : {};
		// the mean at x = 0 as a weighted sum, and its largest term
		const terms = x.map((xi, i) => all.alpha[i] * kern(0, xi, ls, sf2)), top = terms.reduce((b, v, i) => (Math.abs(v) > Math.abs(terms[b]) * (1 + 1e-9) ? i : b), 0);
		yield { phase: 'kernel', step: 0, state: { K: T([n, n], all.K), alpha: T([n], all.alpha) }, metrics: { n, ls, sf2, sn2, kd, ...pm, mq: terms.reduce((a, b) => a + b, 0), top, topA: all.alpha[top], topK: kern(0, x[top], ls, sf2), topX: x[top], topTerm: terms[top] } };
		let post = all;
		for (let i = 0; i < n; i++) {
			post = posterior(x.slice(0, i + 1), y.slice(0, i + 1), ls, sf2, sn2);
			const at = posterior(x.slice(0, i + 1), y.slice(0, i + 1), ls, sf2, sn2, x.slice(0, i + 1));
			let maxRes = 0;
			for (let j = 0; j <= i; j++) maxRes = Math.max(maxRes, Math.abs(at.mean[j] - y[j]));
			const avgBand = post.sd.reduce((a, b) => a + 2 * b, 0) / G;
			const first: Record<string, number> = i ? {} : (() => {
				const xp = x[0] + ls <= GRID[G - 1] ? x[0] + ls : x[0] - ls, kp = kern(xp, x[0], ls, sf2);
				return { kd, w: sf2 / kd, xp, kp, mp: (kp * y[0]) / kd, sdp: Math.sqrt(sf2 - (kp * kp) / kd) };
			})();
			yield { phase: 'add', step: i, state: { mean: T([G], post.mean), sd: T([G], post.sd) }, metrics: { i, used: i + 1, xi: x[i], yi: y[i], mAt: at.mean[i], sdAt: at.sd[i], sd0, avgBand, maxRes, ...first } };
		}
		// the grid point farthest from every data point, and the grid point nearest to point 0
		const gap = (g: number) => Math.min(...x.map((xi) => Math.abs(GRID[g] - xi)));
		const gFar = GRID.reduce((b, _, g) => (gap(g) > gap(b) ? g : b), 0), gNear = near(x[0]);
		const s = draw(post.mean, postCov(post, n, ls, sf2), d.zPost, sf2);
		yield { phase: 'sample', step: n - 1, state: { samples: T([NS, G], s) }, metrics: { n, xFar: GRID[gFar], dFar: gap(gFar), sdFar: post.sd[gFar], spreadFar: range(s, gFar), xNear: GRID[gNear], sdNear: post.sd[gNear], spreadNear: range(s, gNear), sd0 } };
		const ev = LS.map((l) => evidence(x, y, l, sf2, sn2)), cur = evidence(x, y, ls, sf2, sn2);
		const bestI = ev.reduce((b, e, i) => (e.lml > ev[b].lml ? i : b), 0);
		yield { phase: 'evidence', step: n - 1, state: { lmls: T([LS.length], ev.map((e) => e.lml)) }, metrics: { n, ls, sf2, fit: cur.fit, cpx: cur.cpx, cst: cur.cst, lml: cur.lml, bestI, bestLs: LS[bestI], bestLml: ev[bestI].lml, drop: ev[bestI].lml - cur.lml } };
		// the widest gap between neighbouring points (by x), and the posterior right at its ends and middle
		const xs = [...new Set(x)].sort((a, b) => a - b);
		let ga = 0, gb = 0;
		for (let i = 1; i < xs.length; i++) if (xs[i] - xs[i - 1] > gb - ga) (ga = xs[i - 1]), (gb = xs[i]);
		const gp3 = xs.length > 1 ? posterior(x, y, ls, sf2, sn2, [ga, (ga + gb) / 2, gb]) : null;
		const gm: Record<string, number> = gp3 ? { ga, gb, gw: gb - ga, gwl: (gb - ga) / ls, gmid: (ga + gb) / 2, sdA: gp3.sd[0], sdB: gp3.sd[2], sdMid: gp3.sd[1] } : {};
		const last = posterior(x, y, ls, sf2, sn2, x);
		const peak = Math.max(...post.mean), ymax = Math.max(...y);
		yield { phase: 'done', step: n - 1, state: {}, metrics: { n, ls, sf2, sd0, ...gm, ops: (n * n * n) / 3, big: BIG, opsBig: (BIG * BIG * BIG) / 3, mem: n * n, memBig: BIG * BIG, peak, ymax, maxRes: Math.max(...last.mean.map((m, j) => Math.abs(m - y[j]))), avgBand: post.sd.reduce((a, b) => a + 2 * b, 0) / G } };
	}
};
