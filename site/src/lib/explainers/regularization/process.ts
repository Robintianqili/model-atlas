// regularization: the processes (kind C). Every number is computed live by the kit's autograd (atlas/autograd): the
// 2 -> width -> width -> 2 tanh network of net.ts, Adam, the mean cross-entropy on all training points every update, plus
// λ·Σw² (the L2 penalty, on the three weight matrices) when λ > 0, dropout masks after both hidden layers when p > 0,
// and, with opt.aug = σ > 0, a fresh random nudge N(0, σ²) added to every training point at every update (data
// augmentation). Randomness (masks, nudges) only from the run's rng.fork('batch'); the first weights from
// rng.fork('init'), so runs with the same seed and width start from the same weights.
//   fit      one run on the learner's knobs (opt may override any knob). A frame every opt.every updates ('check'): both
//            losses and accuracies without dropout, Σw², the penalty, early stopping's bookkeeping, the weights, the
//            last update's dropout masks and nudged training points. Frame step = updates done so far. opt.stopEarly:
//            the tape ends where early stopping stops (FitOpt).
//   methods  six fit runs from the same seed in lockstep on the starting knobs (input ignored): plain, L2 (λ = 0.003),
//            dropout (p = 0.5), augmentation (σ = 0.1), too much L2 (λ = 0.1) and a small network (2 units): the plain
//            run's frames, with the others' numbers as metrics with the suffixes L2, Drop, Aug, Heavy and Small.
//   cnn      TinyCNN's recorded training, one frame per epoch (precomputed: cnn.ts).
//   npm run atlas -- run regularization run --frames 0..20
import type { ProcessDef, ProcessRun, Rng, Tensor } from '../../atlas/types/index.ts';
import { input, backward, noGrad, zeroGrad, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { Adam } from '../../atlas/autograd/optim.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { EVERY, LR, PATIENCE, STEPS, knobsOf, splitOf, type Knobs } from './data.ts';
import { RegNet, type Params } from './net.ts';
import { CNN } from './cnn.ts';

/** stopEarly: 1 ends the tape at the check where early stopping stops, with the full run's last test loss there as the
 *  metric `full` (the run goes on to opt.steps inside, only the frames after the stop are not recorded) */
export type FitOpt = { steps: number; every: number; lr: number; patience: number; aug: number; stopEarly?: number } & Partial<Knobs>;
export type FitState = { params?: Params; mask?: Tensor; aug?: Tensor };

/** One check of a run: the numbers of a frame. */
export interface Check {
	step: number;
	trainLoss: number;
	testLoss: number;
	trainAcc: number;
	testAcc: number;
	/** Σw² over the three weight matrices; penalty = λ·Σw²; total = trainLoss + penalty (the loss training lowers, when
	 *  nothing is dropped or nudged) */
	sumSq: number;
	penalty: number;
	total: number;
	/** early stopping: the lowest test loss so far, its step and the training loss there, checks since then, 1 at the
	 *  check where patience ran out */
	best: number;
	bestStep: number;
	bestTrain: number;
	wait: number;
	stopped: number;
	stopStep: number;
}

function evalOn(net: RegNet, p: Points2D): { loss: number; acc: number } {
	if (p.n === 0) return { loss: NaN, acc: NaN };
	return noGrad(() => {
		const z = net.forward(input({ shape: [p.n, 2], data: p.xy })).value.data;
		let right = 0;
		let loss = 0;
		for (let i = 0; i < p.n; i++) {
			const a = z[2 * i];
			const b = z[2 * i + 1];
			const m = Math.max(a, b);
			loss += m + Math.log(Math.exp(a - m) + Math.exp(b - m)) - (p.label[i] === 1 ? b : a);
			if ((b > a ? 1 : 0) === p.label[i]) right++;
		}
		return { loss: loss / p.n, acc: right / p.n };
	});
}

const sumSqOf = (net: RegNet) => net.weights().reduce((a, w) => a + w.value.data.reduce((s, v) => s + v * v, 0), 0);
const copyParams = (net: RegNet): Params =>
	Object.fromEntries(Object.entries(net.named()).map(([k, v]) => [k, { shape: v.shape.slice(), data: Float64Array.from(v.value.data) }]));

type RunOpt = Pick<FitOpt, 'steps' | 'every' | 'lr' | 'patience' | 'aug'>;

/** One training run: a Check (with weights, masks and nudged points when `keep`) at step 0 and every `every` updates. */
export function* trainRun(k: Knobs, o: RunOpt, rng: Rng, keep = false): Generator<Check & FitState> {
	const { train, test } = splitOf(k.noise, k.ratio);
	const net = new RegNet(k.width, rng.fork('init'));
	const params = Object.values(net.named());
	const optim = new Adam(net.named(), { lr: o.lr });
	const br = rng.fork('batch');
	const n = train.n;
	let best = Infinity;
	let bestStep = 0;
	let bestTrain = NaN;
	let wait = 0;
	let stopStep = NaN;
	let mask: Tensor | undefined;
	let aug: Tensor | undefined;
	const check = (step: number): Check & FitState => {
		const a = evalOn(net, train);
		const b = evalOn(net, test);
		const sumSq = sumSqOf(net);
		let stopped = 0;
		if (b.loss < best) {
			best = b.loss;
			bestStep = step;
			bestTrain = a.loss;
			wait = 0;
		} else if (step > 0) wait++;
		if (wait >= o.patience && Number.isNaN(stopStep)) {
			stopStep = step;
			stopped = 1;
		}
		const c: Check & FitState = {
			step,
			trainLoss: a.loss,
			testLoss: b.loss,
			trainAcc: a.acc,
			testAcc: b.acc,
			sumSq,
			penalty: k.l2 * sumSq,
			total: a.loss + k.l2 * sumSq,
			best,
			bestStep,
			bestTrain,
			wait,
			stopped,
			stopStep
		};
		if (keep) {
			c.params = copyParams(net);
			if (mask) c.mask = mask;
			if (aug) c.aug = aug;
		}
		return c;
	};
	yield check(0);
	if (n === 0) return;
	const kept = 1 - k.dropout;
	for (let s = 1; s <= o.steps; s++) {
		let masks: Var[] | null = null;
		if (k.dropout > 0) {
			const m = Float64Array.from({ length: 2 * k.width }, () => (br.next() < k.dropout ? 0 : 1 / kept));
			masks = [0, 1].map((i) => input({ shape: [1, k.width], data: m.slice(i * k.width, (i + 1) * k.width) }));
			mask = { shape: [2, k.width], data: m };
		}
		let xy = train.xy;
		if (o.aug > 0) {
			xy = Float64Array.from(train.xy, (v) => v + o.aug * br.normal());
			aug = { shape: [n, 2], data: xy };
		}
		zeroGrad(params);
		let loss = crossEntropy(net.forward(input({ shape: [n, 2], data: xy }), masks), train.label);
		if (k.l2 > 0) loss = op.add(loss, op.scale(net.weights().map((w) => op.sum(op.square(w))).reduce((a, b) => op.add(a, b)), k.l2));
		backward(loss);
		optim.step();
		if (s % o.every === 0 || s === o.steps) yield check(s);
	}
}

/** The learner's knobs with the source's overrides (opt.noise, opt.width, ...), on the widget's stops. */
function knobsFor(raw: unknown, o: Partial<Knobs>): Knobs {
	const k = knobsOf(raw);
	const over: Partial<Knobs> = {};
	for (const key of Object.keys(k) as (keyof Knobs)[]) if (typeof o[key] === 'number') over[key] = o[key];
	return knobsOf({ ...k, ...over });
}

/** A frame's metrics: the check's numbers, the knobs and the two set sizes. */
function metricsOfCheck(c: Check & FitState, k: Knobs, aug: number): Record<string, number> {
	const { params: _p, mask: _m, aug: _a, ...nums } = c;
	const s = splitOf(k.noise, k.ratio);
	return { ...nums, ...k, aug, nTrain: s.train.n, nTest: s.test.n };
}

/** a formula value with 5 significant digits: with 3, the rounded terms do not add up to the rounded total (0.215 +
 *  0.003 × 11.6 = 0.2498, shown as 0.249); with 5 they do (0.21466 + 0.003 × 11.587 = 0.24942) */
const p5 = (v: number): string | number => (Number.isFinite(v) ? Number(v.toPrecision(5)).toString() : v);

const PHASES: ProcessDef<unknown, FitOpt, FitState>['phases'] = {
	start: {
		label: { en: 'The first weights', zh: '最初的权重' },
		kind: 'input',
		narration: { what: { en: 'Before any update: training loss {frame.trainLoss}, test loss {frame.testLoss}.', zh: '还没有任何更新：训练损失 {frame.trainLoss}，测试损失 {frame.testLoss}。' } }
	},
	check: {
		label: { en: 'A check on both sets of points', zh: '在两组点上检查一次' },
		kind: 'observe',
		formula: {
			tex: 'L = \\underbrace{ {{data}} }_{L_{\\text{data}}} + \\underbrace{ {{lam}} }_{\\lambda} \\times \\underbrace{ {{sumSq}} }_{\\sum w^2} = {{total}}',
			values: (f) => ({ data: p5(f.metrics!.trainLoss), lam: String(f.metrics!.l2), sumSq: p5(f.metrics!.sumSq), total: p5(f.metrics!.total) })
		},
		narration: { what: { en: 'After {frame.step} updates: training loss {frame.trainLoss}, test loss {frame.testLoss}.', zh: '更新 {frame.step} 次之后：训练损失 {frame.trainLoss}，测试损失 {frame.testLoss}。' } }
	}
};

const SCHEMA: ProcessDef<unknown, FitOpt, FitState>['schema'] = {
	params: { kind: 'tensor', label: { en: 'Weights', zh: '权重' }, role: 'param', scale: 'diverging', tol: 1e-9 },
	mask: { kind: 'tensor', label: { en: 'Dropout masks of the last update (0 = switched off)', zh: '最近一次更新的 dropout 掩码（0 = 关掉）' }, shape: [2, 'width'], role: 'activation', tol: 'exact' },
	aug: { kind: 'tensor', label: { en: 'Nudged training points of the last update', zh: '最近一次更新时挪动过的训练点' }, shape: ['n', 2], role: 'data', tol: 1e-12 }
};

export const fit: ProcessDef<unknown, FitOpt, FitState> = {
	id: 'fit',
	kind: 'algorithm',
	title: { en: 'Train on the training points, check on the test points', zh: '在训练点上训练，在测试点上检查' },
	phases: PHASES,
	schema: SCHEMA,
	defaults: { steps: STEPS, every: EVERY, lr: LR, patience: PATIENCE, aug: 0 },
	*run(raw, ctx): ProcessRun<FitState, FitOpt> {
		const o = { ...fit.defaults, ...ctx.opt };
		const k = knobsFor(raw, o);
		let checks: Iterable<Check & FitState> = trainRun(k, o, ctx.rng, true);
		let full = NaN;
		if (o.stopEarly) {
			const all = [...checks];
			full = all[all.length - 1].testLoss;
			const at = all.findIndex((c) => c.stopped === 1);
			checks = at < 0 ? all : all.slice(0, at + 1);
		}
		for (const c of checks) {
			const state: FitState = { params: c.params };
			if (c.mask) state.mask = c.mask;
			if (c.aug) state.aug = c.aug;
			const metrics = metricsOfCheck(c, k, o.aug);
			if (c.stopped === 1 && o.stopEarly) metrics.full = full;
			yield { phase: c.step === 0 ? 'start' : 'check', step: c.step, state, metrics };
		}
	}
};

/** The variants of 'methods': a metric suffix and the settings that differ from the plain run. */
export const VARIANTS: { suffix: string; knobs: Partial<Knobs>; aug?: number }[] = [
	{ suffix: 'L2', knobs: { l2: 0.003 } },
	{ suffix: 'Drop', knobs: { dropout: 0.5 } },
	{ suffix: 'Aug', knobs: {}, aug: 0.1 },
	{ suffix: 'Heavy', knobs: { l2: 0.1 } },
	{ suffix: 'Small', knobs: { width: 2 } }
];
const SHARED = ['trainLoss', 'testLoss', 'trainAcc', 'testAcc', 'sumSq', 'penalty', 'total', 'best', 'bestStep'] as const;

export const methods: ProcessDef<unknown, FitOpt, FitState> = {
	...fit,
	id: 'methods',
	// the formula of a check is the L2 run's loss with its penalty term
	phases: {
		...PHASES,
		check: {
			...PHASES.check,
			formula: {
				tex: 'L = \\underbrace{ {{data}} }_{L_{\\text{data}}} + \\underbrace{ {{lam}} }_{\\lambda} \\times \\underbrace{ {{sumSq}} }_{\\sum w^2} = {{total}}',
				values: (f) => ({ data: p5(f.metrics!.trainLossL2), lam: String(f.metrics!.l2L2), sumSq: p5(f.metrics!.sumSqL2), total: p5(f.metrics!.totalL2) })
			}
		}
	},
	title: { en: 'Five ways to fight overfitting, side by side', zh: '五种对付过拟合的办法，并排比较' },
	*run(raw, ctx): ProcessRun<FitState, FitOpt> {
		const o = { ...fit.defaults, ...ctx.opt };
		const k = knobsFor(raw, o);
		const runs = VARIANTS.map((v) => {
			const kv = knobsOf({ ...k, ...v.knobs });
			return { v, k: kv, g: trainRun(kv, { ...o, aug: v.aug ?? 0 }, ctx.rng) };
		});
		for (const c of trainRun(k, o, ctx.rng, true)) {
			const m = metricsOfCheck(c, k, o.aug);
			for (const r of runs) {
				const q = r.g.next();
				if (q.done) continue;
				for (const key of SHARED) m[key + r.v.suffix] = q.value[key];
				m['l2' + r.v.suffix] = r.k.l2;
				m['dropout' + r.v.suffix] = r.k.dropout;
				m['width' + r.v.suffix] = r.k.width;
				m['aug' + r.v.suffix] = r.v.aug ?? 0;
			}
			yield { phase: c.step === 0 ? 'start' : 'check', step: c.step, state: { params: c.params }, metrics: m };
		}
	}
};

type CnnState = { epoch: number };
export const cnn: ProcessDef<unknown, Record<string, never>, CnnState> = {
	id: 'cnn',
	kind: 'algorithm',
	title: { en: "TinyCNN's recorded training", zh: 'TinyCNN 的训练记录' },
	phases: {
		epoch: {
			label: { en: 'One epoch', zh: '一轮' },
			kind: 'observe',
			narration: { what: { en: 'After epoch {frame.step}: validation accuracy {frame.valAcc|pct}.', zh: '第 {frame.step} 轮之后：验证准确率 {frame.valAcc|pct}。' } }
		}
	},
	schema: { epoch: { kind: 'int', label: { en: 'Epoch (counting from 1)', zh: '轮次（从 1 数起）' }, tol: 'exact' } },
	defaults: {},
	*run(): ProcessRun<CnnState, Record<string, never>> {
		let bestAt = 0;
		CNN.valAcc.forEach((v, i) => {
			if (v > CNN.valAcc[bestAt]) bestAt = i;
		});
		for (let i = 0; i < CNN.epochs; i++)
			yield {
				phase: 'epoch',
				step: i + 1,
				state: { epoch: i + 1 },
				metrics: { step: i + 1, trainLoss: CNN.trainLoss[i], valAcc: CNN.valAcc[i], best: i === bestAt ? 1 : 0, bestEpoch: bestAt + 1 }
			};
	}
};
