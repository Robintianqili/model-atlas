// optimizers: the processes (kind C). Every update is made by the kit's optimizers (atlas/autograd/optim.ts: SGD,
// Momentum, Adam, AdamW, written to follow torch.optim's formulas step for step) or, for the network, by the kit's
// autograd; the surfaces' gradients are their derivatives written out (claims.test.ts checks them against finite
// differences).
//   race     the four racers from one start on one surface (the learner's, or a fixed one). Coarse: 'start', then one
//            'step' frame per update (step = updates done). Fine: per update k (counting from 0) one frame per racer
//            ('sgd', 'momentum', 'adam', 'adamw'), each with the numbers of that racer's update k and its formula.
//   league   the race on all three surfaces from their own starts, in lockstep (losses only)
//   range    the learning-rate range test: the 2-8-8-2 network trained while η grows from 0.001 to 10; an 'end' frame
//            reads off the steepest fall, the lowest loss and the blow-up
//   sched    the network at a peak η of 4 with the learner's schedule and warm-up, and in lockstep the same network at a
//            constant η of 4 without warm-up
//   cliff    the exploding-gradient toy: plain gradient descent and gradient descent with clipping, in lockstep
//   npm run atlas -- run optimizers race --frames 0..12
import type { I18n, ProcessDef, ProcessRun, Schema, Tensor } from '../../atlas/types/index.ts';
import { input, param, type Var } from '../../atlas/autograd/var.ts';
import { makeOptimizer, type Adam, type Momentum } from '../../atlas/autograd/optim.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { BETAS, CLIFF, EPS, LAMBDA, MU, RACERS, RACE_STEPS, RACE_SURFACES, RANGE, SCHED, clipByNorm, leagueKey, lrAt, rangeLr, readRange, setupOf, surfaceIndex, surfaceOf, type RacerKey, type SurfaceId } from './data.ts';
import { lossOf, netAndSgd, trainStep } from './net.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const t2 = (a: number, b: number): Tensor => ({ shape: [2], data: Float64Array.of(a, b) });
/** paths [runs, points, 2] from per-run point lists of equal length */
const pathsOf = (runs: number[][][]): Tensor => ({ shape: [runs.length, runs[0].length, 2], data: Float64Array.from(runs.flat(2)) });
const ALIGN = (...rows: string[]) => `\\begin{aligned} ${rows.join(' \\\\ ')} \\end{aligned}`;
const set = (v: number) => fmtSetting(v);

// ---------------------------------------------------------------- race

export type RaceIn = { surface: SurfaceId; x: number; y: number };
export type RaceOpt = { steps: number };
export type RaceState = { pos: Tensor; path: Tensor; mom: Tensor; am: Tensor; av: Tensor };

/** The race's input from the learner's setup: only the surface and the start (a new schedule keeps the race). */
export const raceInput = (raw: unknown): RaceIn => {
	const s = setupOf(raw);
	return { surface: s.surface, x: s.x, y: s.y };
};

const RACE_SCHEMA: Schema = {
	pos: { kind: 'tensor', label: L('Each racer now (x, y)', '每个选手现在的位置 (x, y)'), shape: [4, 2], role: 'param', tol: 1e-12 },
	path: { kind: 'tensor', label: L('Each racer’s path so far', '每个选手到目前为止的路径'), shape: [4, 'k', 2], role: 'param', tol: 1e-12 },
	mom: { kind: 'tensor', label: L('Momentum’s velocity v', '动量法的速度 v'), shape: [2], role: 'grad', tol: 1e-12 },
	am: { kind: 'tensor', label: L('Adam’s running mean of gradients m', 'Adam 的梯度滑动平均 m'), shape: [2], role: 'grad', tol: 1e-12 },
	av: { kind: 'tensor', label: L('Adam’s running mean of squared gradients v', 'Adam 的梯度平方滑动平均 v'), shape: [2], role: 'grad', tol: 1e-12 }
};

const ADAM_ROWS = [
	'm &\\leftarrow {{b1}}\\, m + (1 - {{b1}})\\, g = ({{mx1}},\\ {{my1}})',
	'v &\\leftarrow {{b2}}\\, v + (1 - {{b2}})\\, g^2 = ({{vx1}},\\ {{vy1}})',
	'\\hat m &= m / (1 - {{b1}}^{ {{t}} }) = ({{mhx}},\\ {{mhy}})',
	'\\sqrt{\\hat v} &= \\sqrt{v / (1 - {{b2}}^{ {{t}} })} = ({{shx}},\\ {{shy}})'
];
const settings = (f: { metrics?: Record<string, number> }) => {
	const m = f.metrics ?? {};
	return { ...m, lr: set(m.lr), mu: set(m.mu), b1: set(m.b1), b2: set(m.b2), wd: set(m.wd) };
};

export const race: ProcessDef<unknown, RaceOpt, RaceState> = {
	id: 'race',
	kind: 'algorithm',
	title: L('Four optimizers on one surface', '四个优化器，同一个曲面'),
	phases: {
		start: { label: L('The start', '起点'), kind: 'input', narration: { what: L('All four start at x = {frame.sgdX}, y = {frame.sgdY}.', '四个选手都从 x = {frame.sgdX}、y = {frame.sgdY} 出发。') } },
		step: { label: L('The race', '比赛'), kind: 'update', narration: { what: L('After {frame.step} updates the losses are {frame.sgdLoss}, {frame.momentumLoss}, {frame.adamLoss} and {frame.adamwLoss}.', '更新 {frame.step} 次之后，损失分别是 {frame.sgdLoss}、{frame.momentumLoss}、{frame.adamLoss} 和 {frame.adamwLoss}。') } },
		sgd: {
			label: L('SGD’s update', 'SGD 的更新'),
			kind: 'update',
			formula: { tex: ALIGN('x &\\leftarrow {{x}} - {{lr}} \\times ({{gx}}) = {{x1}}', 'y &\\leftarrow {{y}} - {{lr}} \\times ({{gy}}) = {{y1}}'), values: settings },
			narration: { what: L('SGD moves x to {frame.x1} and y to {frame.y1}.', 'SGD 把 x 移到 {frame.x1}，把 y 移到 {frame.y1}。') }
		},
		momentum: {
			label: L('Momentum’s update', '动量法的更新'),
			kind: 'update',
			formula: {
				tex: ALIGN(
					'v_x &\\leftarrow {{mu}} \\times ({{bx}}) + ({{gx}}) = {{bx1}}',
					'v_y &\\leftarrow {{mu}} \\times ({{by}}) + ({{gy}}) = {{by1}}',
					'x &\\leftarrow {{x}} - {{lr}} \\times ({{bx1}}) = {{x1}}',
					'y &\\leftarrow {{y}} - {{lr}} \\times ({{by1}}) = {{y1}}'
				),
				values: settings
			},
			narration: { what: L('The velocity becomes ({frame.bx1}, {frame.by1}).', '速度变成 ({frame.bx1}, {frame.by1})。') }
		},
		adam: {
			label: L('Adam’s update', 'Adam 的更新'),
			kind: 'update',
			formula: { tex: ALIGN(...ADAM_ROWS, '(x, y) &\\leftarrow ({{x}},\\ {{y}}) - {{lr}} \\times \\hat m / (\\sqrt{\\hat v} + \\epsilon) = ({{x1}},\\ {{y1}})'), values: settings },
			narration: { what: L('Adam moves x by {frame.dx} and y by {frame.dy}.', 'Adam 让 x 移动 {frame.dx}，让 y 移动 {frame.dy}。') }
		},
		adamw: {
			label: L('AdamW’s update', 'AdamW 的更新'),
			kind: 'update',
			formula: {
				tex: ALIGN(...ADAM_ROWS, '(x, y) &\\leftarrow (1 - {{lr}} \\times {{wd}})\\,({{x}},\\ {{y}}) - {{lr}} \\times \\hat m / (\\sqrt{\\hat v} + \\epsilon) = ({{x1}},\\ {{y1}})'),
				values: settings
			},
			narration: { what: L('AdamW moves x to {frame.x1} and y to {frame.y1}.', 'AdamW 把 x 移到 {frame.x1}，把 y 移到 {frame.y1}。') }
		}
	},
	schema: RACE_SCHEMA,
	defaults: { steps: RACE_STEPS },
	*run(raw, ctx): ProcessRun<RaceState, RaceOpt> {
		const { surface, x: x0, y: y0 } = raceInput(raw);
		const s = surfaceOf(surface);
		const surf = surfaceIndex(surface);
		const fine = ctx.detail === 'fine';
		const racers = RACERS.map((r) => {
			const p: Var = param(r.key, { shape: [2], data: [x0, y0] });
			const opt = makeOptimizer(r.kind, [p], { lr: r.lr, momentum: MU, betas: BETAS, eps: EPS, weightDecay: r.kind === 'adamw' ? LAMBDA : 0 });
			return { r, p, opt, path: [[x0, y0]] };
		});
		const mom = racers[1].opt as Momentum;
		const adam = racers[2].opt as Adam;
		const state = (): RaceState => ({
			pos: { shape: [4, 2], data: Float64Array.from(racers.flatMap((q) => [q.p.data[0], q.p.data[1]])) },
			path: pathsOf(racers.map((q) => q.path)),
			mom: t2(...((mom.buf.get('momentum') ?? [0, 0]) as [number, number])),
			am: t2(...((adam.m.get('adam') ?? [0, 0]) as [number, number])),
			av: t2(...((adam.v.get('adam') ?? [0, 0]) as [number, number]))
		});
		const board = (k: number) => {
			const m: Record<string, number> = { step: k, surf, adamT: k };
			for (const q of racers) {
				m[`${q.r.key}X`] = q.p.data[0];
				m[`${q.r.key}Y`] = q.p.data[1];
				m[`${q.r.key}Loss`] = s.f(q.p.data[0], q.p.data[1]);
			}
			return m;
		};
		if (!fine) yield { phase: 'start', step: 0, state: state(), metrics: board(0) };
		for (let k = 0; k < ctx.opt.steps; k++) {
			const before = racers.map((q) => {
				const [x, y] = [q.p.data[0], q.p.data[1]];
				const [gx, gy] = s.grad(x, y);
				return { x, y, gx, gy, loss: s.f(x, y) };
			});
			const b0 = Array.from(mom.buf.get('momentum') ?? [0, 0]);
			const m0 = Array.from(adam.m.get('adam') ?? [0, 0]);
			const v0 = Array.from(adam.v.get('adam') ?? [0, 0]);
			for (let i = 0; i < racers.length; i++) {
				const q = racers[i];
				q.p.grad = input({ shape: [2], data: [before[i].gx, before[i].gy] });
				q.opt.step();
				q.path.push([q.p.data[0], q.p.data[1]]);
			}
			if (!fine) {
				yield { phase: 'step', step: k + 1, state: state(), metrics: board(k + 1) };
				continue;
			}
			const st = state();
			const t = k + 1;
			for (let i = 0; i < racers.length; i++) {
				const q = racers[i];
				const b = before[i];
				const [x1, y1] = [q.p.data[0], q.p.data[1]];
				const m: Record<string, number> = { step: k, surf, lr: q.r.lr, x: b.x, y: b.y, gx: b.gx, gy: b.gy, x1, y1, loss: b.loss, loss1: s.f(x1, y1), dx: x1 - b.x, dy: y1 - b.y };
				if (q.r.kind === 'momentum') {
					const b1 = mom.buf.get('momentum')!;
					Object.assign(m, { mu: MU, bx: k === 0 ? 0 : b0[0], by: k === 0 ? 0 : b0[1], bx1: b1[0], by1: b1[1] });
				}
				if (q.r.kind === 'adam' || q.r.kind === 'adamw') {
					const o = q.opt as Adam;
					const mm = o.m.get(q.r.key)!;
					const vv = o.v.get(q.r.key)!;
					const c1 = 1 - Math.pow(BETAS[0], t);
					const c2 = 1 - Math.pow(BETAS[1], t);
					Object.assign(m, { t, b1: BETAS[0], b2: BETAS[1], eps: EPS, mx1: mm[0], my1: mm[1], vx1: vv[0], vy1: vv[1], mhx: mm[0] / c1, mhy: mm[1] / c1, shx: Math.sqrt(vv[0] / c2), shy: Math.sqrt(vv[1] / c2) });
					if (q.r.kind === 'adam') Object.assign(m, { mx: m0[0], my: m0[1], vx: v0[0], vy: v0[1] });
					else Object.assign(m, { wd: LAMBDA, decayX: -q.r.lr * LAMBDA * b.x, decayY: -q.r.lr * LAMBDA * b.y });
				}
				yield { phase: q.r.key, step: k, state: st, metrics: m };
			}
		}
	}
};

// ---------------------------------------------------------------- league

export type LeagueState = { losses: Tensor };

export const league: ProcessDef<unknown, RaceOpt, LeagueState> = {
	id: 'league',
	kind: 'algorithm',
	title: L('The race on all three surfaces', '三个曲面上的比赛'),
	phases: { step: { label: L('Three races', '三场比赛'), kind: 'update', narration: { what: L('After {frame.step} updates on each surface.', '在每个曲面上更新 {frame.step} 次之后。') } } },
	schema: { losses: { kind: 'tensor', label: L('Loss of each racer on each surface', '每个选手在每个曲面上的损失'), shape: [3, 4], tol: 1e-12 } },
	defaults: { steps: RACE_STEPS },
	*run(_raw, ctx): ProcessRun<LeagueState, RaceOpt> {
		const runs = RACE_SURFACES.map((id) => race.run({ surface: id, x: surfaceOf(id).start[0], y: surfaceOf(id).start[1] }, { ...ctx, detail: 'coarse' }));
		for (;;) {
			const frames = runs.map((g) => g.next());
			if (frames.some((f) => f.done)) return;
			const losses = new Float64Array(12);
			const m: Record<string, number> = {};
			frames.forEach((f, i) => {
				const fm = f.value!.metrics!;
				m.step = fm.step;
				RACERS.forEach((r, j) => {
					losses[i * 4 + j] = fm[`${r.key}Loss`];
					m[leagueKey(RACE_SURFACES[i], r.key)] = fm[`${r.key}Loss`];
				});
			});
			yield { phase: 'step', step: m.step, state: { losses: { shape: [3, 4], data: losses } }, metrics: m };
		}
	}
};

// ---------------------------------------------------------------- range test

export type RangeOpt = { steps: number };
export type RangeState = { lr: number };

export const range: ProcessDef<unknown, RangeOpt, RangeState> = {
	id: 'range',
	kind: 'algorithm',
	title: L('A learning-rate range test', '学习率区间测试'),
	phases: {
		step: { label: L('Raise η a little', '把 η 调大一点'), kind: 'update', narration: { what: L('At η = {frame.lr} the loss is {frame.loss}.', 'η = {frame.lr} 时，损失是 {frame.loss}。') } },
		end: { label: L('Read the curve', '读这条曲线'), kind: 'observe', narration: { what: L('The loss is lowest, {frame.lowLoss}, at η = {frame.lowLr}.', '损失在 η = {frame.lowLr} 时最低，为 {frame.lowLoss}。') } }
	},
	schema: { lr: { kind: 'scalar', label: L('Learning rate η', '学习率 η'), tol: 1e-12 } },
	defaults: { steps: RANGE.steps },
	*run(_raw, ctx): ProcessRun<RangeState, RangeOpt> {
		const { net, opt } = netAndSgd(ctx.rng.fork('init'), RANGE.lo);
		const lrs: number[] = [];
		const losses: number[] = [];
		for (let k = 0; k < ctx.opt.steps; k++) {
			const lr = rangeLr(k);
			const loss = trainStep(net, opt, lr);
			lrs.push(lr);
			losses.push(loss);
			yield { phase: 'step', step: k, state: { lr }, metrics: { step: k, lr, loss } };
		}
		const r = readRange(lrs, losses);
		const at = (i: number, a: number[]) => (Number.isFinite(i) ? a[i] : NaN);
		yield {
			phase: 'end',
			step: ctx.opt.steps,
			state: { lr: lrs[lrs.length - 1] },
			metrics: { step: ctx.opt.steps, startLoss: losses[0], lowLr: lrs[r.low], lowLoss: losses[r.low], steepLr: at(r.steep, lrs), steepLoss: at(r.steep, losses), blowLr: at(r.blow, lrs), blowLoss: at(r.blow, losses), endLoss: lossOf(net) }
		};
	}
};

// ---------------------------------------------------------------- schedules

export type SchedIn = { schedule: 'constant' | 'step' | 'cosine'; warmup: boolean };
export type SchedOpt = { steps: number };
export const schedInput = (raw: unknown): SchedIn => {
	const s = setupOf(raw);
	return { schedule: s.schedule, warmup: s.warmup };
};

export const sched: ProcessDef<unknown, SchedOpt, RangeState> = {
	id: 'sched',
	kind: 'algorithm',
	title: L('Your schedule against a constant rate', '你的调度对比恒定学习率'),
	phases: {
		step: { label: L('Training step', '训练步'), kind: 'update', narration: { what: L('At step {frame.step} η is {frame.lr}.', '第 {frame.step} 步时 η 是 {frame.lr}。') } },
		end: { label: L('After training', '训练之后'), kind: 'observe', narration: { what: L('Your schedule ends at loss {frame.loss}.', '你的调度最后损失为 {frame.loss}。') } }
	},
	schema: { lr: { kind: 'scalar', label: L('Learning rate η', '学习率 η'), tol: 1e-12 } },
	defaults: { steps: SCHED.steps },
	*run(raw, ctx): ProcessRun<RangeState, SchedOpt> {
		const { schedule, warmup } = schedInput(raw);
		const mine = netAndSgd(ctx.rng.fork('init'), SCHED.peak);
		const base = netAndSgd(ctx.rng.fork('init'), SCHED.peak);
		let max = 0;
		let maxBase = 0;
		for (let k = 0; k < ctx.opt.steps; k++) {
			const lr = lrAt(schedule, warmup, k);
			const loss = trainStep(mine.net, mine.opt, lr);
			const lossBase = trainStep(base.net, base.opt, SCHED.peak);
			max = Math.max(max, loss);
			maxBase = Math.max(maxBase, lossBase);
			yield { phase: 'step', step: k, state: { lr }, metrics: { step: k, lr, lrBase: SCHED.peak, loss, lossBase } };
		}
		const n = ctx.opt.steps;
		yield { phase: 'end', step: n, state: { lr: lrAt(schedule, warmup, n - 1) }, metrics: { step: n, loss: lossOf(mine.net), lossBase: lossOf(base.net), max, maxBase, warm: warmup ? 1 : 0 } };
	}
};

// ---------------------------------------------------------------- cliff

export type CliffOpt = { steps: number };
export type CliffState = { pos: Tensor; path: Tensor };

export const cliff: ProcessDef<unknown, CliffOpt, CliffState> = {
	id: 'cliff',
	kind: 'algorithm',
	title: L('Gradient descent next to a cliff', '悬崖边的梯度下降'),
	phases: {
		start: { label: L('The start', '起点'), kind: 'input', narration: { what: L('Both runs start at w = {frame.plainW}, b = {frame.plainB}.', '两次运行都从 w = {frame.plainW}、b = {frame.plainB} 出发。') } },
		step: { label: L('Gradient descent', '梯度下降'), kind: 'update', narration: { what: L('After {frame.step} updates the losses are {frame.plainLoss} and {frame.clipLoss}.', '更新 {frame.step} 次之后，损失分别是 {frame.plainLoss} 和 {frame.clipLoss}。') } }
	},
	schema: {
		pos: { kind: 'tensor', label: L('Each run now (w, b)', '每次运行现在的位置 (w, b)'), shape: [2, 2], role: 'param', tol: 1e-12 },
		path: { kind: 'tensor', label: L('Each run’s path so far', '每次运行到目前为止的路径'), shape: [2, 'k', 2], role: 'param', tol: 1e-12 }
	},
	defaults: { steps: CLIFF.steps },
	*run(raw, ctx): ProcessRun<CliffState, CliffOpt> {
		const mine = setupOf(raw).clip ? 1 : 0;
		const runs = [false, true].map((clip) => ({ clip, w: CLIFF.start[0], b: CLIFF.start[1], gone: 0, path: [[CLIFF.start[0], CLIFF.start[1]]], norm: NaN, scale: 1, len: 0 }));
		const metrics = (k: number) => {
			const m: Record<string, number> = { step: k, surf: surfaceIndex('cliff'), mine, lr: CLIFF.lr, maxNorm: CLIFF.maxNorm, T: CLIFF.T };
			for (const r of runs) {
				const p = r.clip ? 'clip' : 'plain';
				Object.assign(m, { [`${p}W`]: r.w, [`${p}B`]: r.b, [`${p}Loss`]: surfaceOf('cliff').f(r.w, r.b), [`${p}Gone`]: r.gone, [`${p}Norm`]: r.norm, [`${p}Scale`]: r.scale, [`${p}Len`]: r.len });
			}
			return m;
		};
		const state = (): CliffState => ({ pos: { shape: [2, 2], data: Float64Array.from(runs.flatMap((r) => [r.w, r.b])) }, path: pathsOf(runs.map((r) => r.path)) });
		yield { phase: 'start', step: 0, state: state(), metrics: metrics(0) };
		for (let k = 0; k < ctx.opt.steps; k++) {
			for (const r of runs) {
				if (r.gone) {
					r.path.push([r.w, r.b]);
					r.len = 0;
					continue;
				}
				const c = clipByNorm(surfaceOf('cliff').grad(r.w, r.b), r.clip ? CLIFF.maxNorm : Infinity);
				r.norm = c.norm;
				r.scale = c.scale;
				const [w1, b1] = [r.w - CLIFF.lr * c.g[0], r.b - CLIFF.lr * c.g[1]];
				r.len = Math.hypot(w1 - r.w, b1 - r.b);
				[r.w, r.b] = [w1, b1];
				r.path.push([w1, b1]);
				const loss = surfaceOf('cliff').f(w1, b1);
				// a run whose loss passes a million has blown up: a real trainer would reach infinity a step or two later
				if (!(loss < CLIFF.blowUp)) r.gone = k + 1;
			}
			yield { phase: 'step', step: k + 1, state: state(), metrics: metrics(k + 1) };
		}
	}
};
