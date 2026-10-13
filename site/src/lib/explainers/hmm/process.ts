// hmm: the processes (kind C), one frame per event; pipeline/hmm/reference.py repeats each in Python (numpy), frame by
// frame. Every number a step narrates is in its frame's metrics.
//   forward   guess (one weather path and its probability), init (day 0's cells), cell (one lattice cell: the sum over the
//             two ways in, times the emission), end (the probability of the activities, the last day's shares)
//   viterbi   init, cell (max instead of sum, and the back-pointer), end (the best last cell), back (one day of the trace
//             back), done (the decoded path against the activity-alone guess and against the starting tables' path)
//   sample    day (draw the weather, then the activity, from two uniform draws), done (counts)
//   spells    count (transition counts of the made-up 3-day-rain stretch), fit (the spell lengths those counts imply)
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { argmax, diffCount, forwardOf, I, jointOf, K, naiveOf, viterbiOf, type HmmData, type Tables } from './hmm.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });
const grid = (rows: number[][], n: number) => T([n, K], Array.from({ length: n * K }, (_, i) => rows[Math.floor(i / K)]?.[i % K] ?? 0));
const lab = (en: string, zh: string) => ({ label: I(en, zh) });
const exact = 'exact' as const;
/** a two-state row as metrics: { p0: x[0], p1: x[1] } */
const two = (name: string, x: number[]) => Object.fromEntries(x.map((v, i) => [`${name}${i}`, v]));
const first = (a: number[], b: number[]) => a.findIndex((v, t) => v !== b[t]);

export type FwdState = { obs: Tensor; guess: Tensor; alpha: Tensor };
export const forward: ProcessDef<HmmData, Record<string, never>, FwdState> = {
	id: 'forward',
	kind: 'algorithm',
	title: I('The forward algorithm fills the lattice', '前向算法填满网格'),
	phases: { guess: { ...lab('One guess of the weather, one product', '猜一条天气路径，算一个乘积'), kind: 'input' }, init: { ...lab('Day 0: start probability times emission', '第 0 天：初始概率乘发射概率'), kind: 'forward' }, cell: { ...lab('One cell: sum the ways in, times the emission', '一个格子：把进来的路加起来，再乘发射概率'), kind: 'forward' }, end: { ...lab('The probability of all the activities', '全部活动的概率'), kind: 'output' } },
	schema: {
		obs: { kind: 'tensor', ...lab('The activity of each day (0 walk, 1 shop, 2 clean)', '每天的活动（0 散步，1 购物，2 打扫）'), role: 'data', tol: exact },
		guess: { kind: 'tensor', ...lab('The guessed weather of each day (0 sunny, 1 rainy)', '猜测的每天天气（0 晴天，1 雨天）'), tol: exact },
		alpha: { kind: 'tensor', ...lab('Forward probabilities α[day, weather]', '前向概率 α[天, 天气]'), scale: 'sequential', tol: 1e-12 }
	},
	defaults: {},
	*run(d): ProcessRun<FwdState, Record<string, never>> {
		const n = d.obs.length;
		if (!n) return;
		const g = naiveOf(d), al = forwardOf(d), joint = jointOf(d, g), paths = K ** n;
		const head: Record<string, number> = n > 1 ? { g1: g[1], a1: d.A[g[0]][g[1]], b1: d.B[g[1]][d.obs[1]], two: jointOf(d, g.slice(0, 2)) } : {};
		yield { phase: 'guess', step: 0, state: { obs: T([n], d.obs), guess: T([n], g), alpha: grid([], n) }, metrics: { n, paths, g0: g[0], pi: d.pi[g[0]], b0: d.B[g[0]][d.obs[0]], ...head, joint } };
		for (let t = 0; t < n; t++)
			for (let j = 0; j < K; j++) {
				const o = d.obs[t], b = d.B[j][o], cells = al.slice(0, t).concat([al[t].map((v, i) => (i <= j ? v : 0))]);
				if (!t) yield { phase: 'init', step: 0, sub: j, state: { alpha: grid(cells, n) }, metrics: { t, j, o, pi: d.pi[j], b, alpha: al[t][j] } };
				else {
					const ins = al[t - 1].map((p, i) => p * d.A[i][j]);
					yield { phase: 'cell', step: t, sub: j, state: { alpha: grid(cells, n) }, metrics: { t, j, o, ...two('p', al[t - 1]), ...two('a', d.A.map((r) => r[j])), ...two('in', ins), sum: ins.reduce((a, c) => a + c, 0), b, alpha: al[t][j] } };
				}
			}
		const ev = al[n - 1].reduce((a, c) => a + c, 0);
		yield { phase: 'end', step: n - 1, state: {}, metrics: { n, paths, evidence: ev, ...two('s', al[n - 1].map((v) => v / ev)), joint, guessShare: joint / ev } };
	}
};

export type VitState = { obs: Tensor; delta: Tensor; psi: Tensor; path: Tensor; naive: Tensor; path0: Tensor };
export const viterbi: ProcessDef<HmmData, Record<string, never>, VitState> = {
	id: 'viterbi',
	kind: 'algorithm',
	title: I('Viterbi finds the single best path', '维特比算法找出最好的那一条路径'),
	phases: { init: { ...lab('Day 0: start probability times emission', '第 0 天：初始概率乘发射概率'), kind: 'forward' }, cell: { ...lab('One cell: the best way in, times the emission', '一个格子：取最好的那条进来的路，再乘发射概率'), kind: 'forward' }, end: { ...lab('The best last cell', '最后一天最好的格子'), kind: 'aggregate' }, back: { ...lab('Follow the back-pointer one day back', '沿回溯指针退回一天'), kind: 'other' }, done: { ...lab('The decoded weather', '解码出的天气'), kind: 'output' } },
	schema: {
		obs: { kind: 'tensor', ...lab('The activity of each day (0 walk, 1 shop, 2 clean)', '每天的活动（0 散步，1 购物，2 打扫）'), role: 'data', tol: exact },
		delta: { kind: 'tensor', ...lab('Best-path probabilities δ[day, weather]', '最优路径概率 δ[天, 天气]'), scale: 'sequential', tol: 1e-12 },
		psi: { kind: 'tensor', ...lab('Back-pointers: the weather the day before (−1 on day 0)', '回溯指针：前一天的天气（第 0 天为 −1）'), tol: exact },
		path: { kind: 'tensor', ...lab('The decoded weather, traced back (−1 not yet)', '回溯得到的天气（−1 表示还没到）'), tol: exact },
		naive: { kind: 'tensor', ...lab('Each day’s weather from its activity alone', '只看当天活动得出的天气'), tol: exact },
		path0: { kind: 'tensor', ...lab('The decoded weather with the starting tables', '用起始表格解码出的天气'), tol: exact }
	},
	defaults: {},
	*run(d): ProcessRun<VitState, Record<string, never>> {
		const n = d.obs.length;
		if (!n) return;
		const v = viterbiOf(d), ev = forwardOf(d)[n - 1].reduce((a, c) => a + c, 0), none = Array(n).fill(-1), g = naiveOf(d), p0 = viterbiOf({ ...d.base, obs: d.obs }).path;
		for (let t = 0; t < n; t++)
			for (let j = 0; j < K; j++) {
				const o = d.obs[t], b = d.B[j][o], cut = (rows: number[][], z: number) => rows.slice(0, t).concat([rows[t].map((x, i) => (i <= j ? x : z))]);
				const state = { delta: grid(cut(v.delta, 0), n), psi: grid(cut(v.psi, -1).concat(Array(n - t - 1).fill(Array(K).fill(-1))), n) };
				if (!t) yield { phase: 'init', step: 0, sub: j, state: j ? state : { ...state, obs: T([n], d.obs), path: T([n], none), naive: T([n], g), path0: T([n], p0) }, metrics: { t, j, o, pi: d.pi[j], b, delta: v.delta[t][j] } };
				else {
					const c = v.delta[t - 1].map((p, i) => p * d.A[i][j]);
					yield { phase: 'cell', step: t, sub: j, state, metrics: { t, j, o, ...two('p', v.delta[t - 1]), ...two('a', d.A.map((r) => r[j])), ...two('c', c), from: v.psi[t][j], best: c[v.psi[t][j]], b, delta: v.delta[t][j] } };
				}
			}
		const last = argmax(v.delta[n - 1]), pBest = v.delta[n - 1][last];
		yield { phase: 'end', step: n - 1, state: {}, metrics: { n, last, pBest, evidence: ev, share: pBest / ev, ...two('d', v.delta[n - 1]) } };
		const path = none.slice();
		for (let t = n - 1; t >= 0; t--) {
			path[t] = v.path[t];
			yield { phase: 'back', step: t, state: { path: T([n], path) }, metrics: { t, s: path[t], from: t ? v.psi[t][path[t]] : -1 } };
		}
		yield { phase: 'done', step: n - 1, state: {}, metrics: { n, pBest, share: pBest / ev, diff: diffCount(v.path, g), firstDiff: first(v.path, g), changed: diffCount(v.path, p0), firstChanged: first(v.path, p0), rainy: v.path.filter((s) => s === 1).length } };
	}
};

export type SampleOpt = { days: number };
export type SampleState = { weather: Tensor; acts: Tensor };
/** the index whose interval [lo, hi) of the cumulative row holds u, with the interval (the last one ends at 1) */
export function draw(row: number[], u: number): { i: number; lo: number; hi: number } {
	let lo = 0;
	for (let i = 0; i < row.length; i++) {
		const hi = i === row.length - 1 ? 1 : lo + row[i];
		if (u < hi) return { i, lo, hi };
		lo = hi;
	}
	return { i: row.length - 1, lo, hi: 1 };
}
export const sample: ProcessDef<Tables, SampleOpt, SampleState> = {
	id: 'sample',
	kind: 'algorithm',
	title: I('The model makes up days', '模型编出一串日子'),
	phases: { day: { ...lab('Draw the weather, then the activity', '先抽天气，再抽活动'), kind: 'sample' }, done: { ...lab('The made-up days, counted', '数一数编出来的日子'), kind: 'output' } },
	schema: {
		weather: { kind: 'tensor', ...lab('The drawn weather (0 sunny, 1 rainy; −1 not yet)', '抽到的天气（0 晴天，1 雨天；−1 还没抽）'), tol: exact },
		acts: { kind: 'tensor', ...lab('The drawn activity (0 walk, 1 shop, 2 clean; −1 not yet)', '抽到的活动（0 散步，1 购物，2 打扫；−1 还没抽）'), tol: exact }
	},
	defaults: { days: 12 },
	*run(tb, ctx): ProcessRun<SampleState, SampleOpt> {
		const n = Math.max(1, Math.floor(ctx.opt.days)), w = Array(n).fill(-1), a = Array(n).fill(-1);
		for (let t = 0; t < n; t++) {
			const u = ctx.rng.next(), s = draw(t ? tb.A[w[t - 1]] : tb.pi, u), v = ctx.rng.next(), x = draw(tb.B[s.i], v);
			(w[t] = s.i), (a[t] = x.i);
			yield { phase: 'day', step: t, state: { weather: T([n], w), acts: T([n], a) }, metrics: { t, u, s: s.i, lo: s.lo, hi: s.hi, v, act: x.i, alo: x.lo, ahi: x.hi, prev: t ? w[t - 1] : -1 } };
		}
		let stays = 0, run = 1, longest = 1;
		for (let t = 1; t < n; t++) {
			stays += w[t] === w[t - 1] ? 1 : 0;
			run = w[t] === w[t - 1] ? run + 1 : 1;
			longest = Math.max(longest, run);
		}
		yield { phase: 'done', step: n - 1, state: {}, metrics: { n, stays, switches: n - 1 - stays, rainy: w.filter((s) => s === 1).length, longest } };
	}
};

export type SpellInput = { year: number[]; maxLen: number };
export type SpellState = { year: Tensor; model: Tensor; truth: Tensor };
export const spells: ProcessDef<SpellInput, Record<string, never>, SpellState> = {
	id: 'spells',
	kind: 'algorithm',
	title: I('What a Markov chain can say about rainy spells', '马尔可夫链能描述怎样的连雨天'),
	phases: { count: { ...lab('Count the transitions', '数一数天气的转移'), kind: 'aggregate' }, fit: { ...lab('The spell lengths the counted table implies', '数出来的转移表所隐含的连雨天长度'), kind: 'output' } },
	schema: {
		year: { kind: 'tensor', ...lab('The made-up weather (0 sunny, 1 rainy)', '编出来的天气（0 晴天，1 雨天）'), role: 'data', tol: exact },
		model: { kind: 'tensor', ...lab('Share of rainy spells of each length, as the fitted chain implies', '拟合的马尔可夫链所隐含的各长度连雨天比例'), scale: 'sequential', tol: 1e-12 },
		truth: { kind: 'tensor', ...lab('Share of rainy spells of each length, as they happened', '实际各长度连雨天的比例'), scale: 'sequential', tol: 1e-12 }
	},
	defaults: {},
	*run(inp): ProcessRun<SpellState, Record<string, never>> {
		const y = inp.year, n = y.length, L = inp.maxLen, c = [[0, 0], [0, 0]], lens: number[] = [];
		for (let t = 1; t < n; t++) c[y[t - 1]][y[t]]++;
		for (let t = 0; t < n; t++) {
			if (y[t] !== 1 || (t && y[t - 1] === 1)) continue;
			let e = t;
			while (e < n && y[e] === 1) e++;
			lens.push(e - t);
		}
		const aSS = c[0][0] / (c[0][0] + c[0][1]), aRR = c[1][1] / (c[1][0] + c[1][1]);
		yield { phase: 'count', step: 0, state: { year: T([n], y), model: T([L], Array(L).fill(0)), truth: T([L], Array(L).fill(0)) }, metrics: { n, nSS: c[0][0], nSR: c[0][1], nRS: c[1][0], nRR: c[1][1], aSS, aRR, spells: lens.length, mean: lens.reduce((a, b) => a + b, 0) / lens.length } };
		const model = Array.from({ length: L }, (_, k) => aRR ** k * (1 - aRR)), truth = Array.from({ length: L }, (_, k) => lens.filter((l) => l === k + 1).length / lens.length);
		const per = (name: string, x: number[]) => Object.fromEntries(x.map((p, k) => [`${name}${k + 1}`, p]));
		yield { phase: 'fit', step: 0, state: { model: T([L], model), truth: T([L], truth) }, metrics: { aRR, meanModel: 1 / (1 - aRR), ...per('p', model), ...per('q', truth), mode: argmax(model) + 1, modeTrue: argmax(truth) + 1, tail: 1 - model.reduce((a, b) => a + b, 0) } };
	}
};
