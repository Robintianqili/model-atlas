// loss-gradient-descent: the processes (kind C).
//   gd       gradient descent on ONE neuron, y = w·x + b, starting from the learner's dials. Every score and gradient is
//            computed by the kit's autograd (atlas/autograd): the forward pass builds the graph of the mean squared
//            error, backward() fills dL/dw and dL/db. Detail 'fine': per update k (counting from 0) the frames 'loss'
//            (forward + score), 'grad' (backward) and 'update' (the step), then 'end'. Detail 'coarse': 'start', then
//            one 'step' frame per update (its `step` = updates done so far, as the kit's training tapes count), the last
//            one being the end.
//   rates    three gd runs from the same dials in lockstep, at opt.lr, opt.lrSlow and opt.lrBig: the first run's coarse
//            frames, with the other two runs' scores as the metrics lossSlow and lossBig (one tape, so one plot and one
//            narration hold all three)
//   realNet  the 2-8-8-2 network of net.ts trained with plain gradient descent (trainingProcess frames, nothing computed
//            another way); its input is ignored, so a new dial setting keeps its run.
//   realPair two realNet runs in lockstep from different random starting weights (the run's seed and seed + 1): the
//            first run's frames, with the second's score as the metric loss2 and, at frames with weights, the first
//            weight of each run (net.0.weight[0, 0]) as w0 and w0b.
//   npm run atlas -- run loss-gradient-descent gd --frames 0..12
import type { ProcessDef, ProcessRun, Tensor, TrainState } from '../../atlas/types/index.ts';
import { backward, input, param } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { N, XS, YS, dialsOf } from './data.ts';
import { moonsNet } from './net.ts';

export type GdOpt = { lr: number; steps: number };
export type GdState = { w: number; b: number; gw: number; gb: number; path: Tensor };

/** The score and its gradient at (w, b), by autograd: L = mean((w·x + b − y)²); backward() gives dL/dw and dL/db. */
export function scoreAndGradient(w0: number, b0: number): { loss: number; gw: number; gb: number } {
	const w = param('w', { shape: [1], data: [w0] });
	const b = param('b', { shape: [1], data: [b0] });
	const x = input({ shape: [N], data: XS });
	const y = input({ shape: [N], data: YS });
	const loss = op.mean(op.square(op.sub(op.add(op.mul(x, w), b), y)));
	backward(loss);
	return { loss: loss.item(), gw: w.grad!.item(), gb: b.grad!.item() };
}

const pathOf = (pts: number[][]): Tensor => ({ shape: [pts.length, 2], data: Float64Array.from(pts.flat()) });

const ALIGN = (a: string, b: string) => `\\begin{aligned} ${a} \\\\ ${b} \\end{aligned}`;

export const gd: ProcessDef<unknown, GdOpt, GdState> = {
	id: 'gd',
	kind: 'algorithm',
	title: { en: 'Gradient descent on one neuron', zh: '一个神经元上的梯度下降' },
	phases: {
		start: {
			label: { en: 'Your dials', zh: '你的旋钮' },
			kind: 'input',
			narration: { what: { en: 'The run starts from your dials: w = {frame.w} and b = {frame.b}, with an error score of {frame.loss}.', zh: '这次运行从你的旋钮出发：w = {frame.w}，b = {frame.b}，误差分数是 {frame.loss}。' } }
		},
		loss: {
			label: { en: 'The error score', zh: '误差分数' },
			kind: 'loss',
			formula: { tex: 'L = \\tfrac{1}{8} \\sum_{i} (w x_i + b - y_i)^2 = {{loss}}', values: (f) => ({ loss: f.metrics!.loss }) },
			narration: { what: { en: 'With w = {frame.w} and b = {frame.b} the error score is {frame.loss}.', zh: 'w = {frame.w}、b = {frame.b} 时，误差分数是 {frame.loss}。' } }
		},
		grad: {
			label: { en: 'The gradient', zh: '梯度' },
			kind: 'backward',
			formula: {
				tex: ALIGN('\\tfrac{\\partial L}{\\partial w} &= \\tfrac{1}{8} \\sum_{i} 2 e_i x_i = {{gw}}', '\\tfrac{\\partial L}{\\partial b} &= \\tfrac{1}{8} \\sum_{i} 2 e_i = {{gb}}'),
				values: (f) => ({ gw: f.metrics!.gw, gb: f.metrics!.gb })
			},
			narration: { what: { en: 'Backpropagation gives the slope of the score under each dial: {frame.gw} for w and {frame.gb} for b.', zh: '反向传播给出分数在每个旋钮方向上的斜率：w 方向 {frame.gw}，b 方向 {frame.gb}。' } }
		},
		update: {
			label: { en: 'One step downhill', zh: '向下走一步' },
			kind: 'update',
			formula: {
				tex: ALIGN('w &\\leftarrow {{w}} - {{lr}} \\times ({{gw}}) = {{w1}}', 'b &\\leftarrow {{b}} - {{lr}} \\times ({{gb}}) = {{b1}}'),
				values: (f) => ({ w: f.metrics!.w, b: f.metrics!.b, lr: f.metrics!.lr, gw: f.metrics!.gw, gb: f.metrics!.gb, w1: f.metrics!.w1, b1: f.metrics!.b1 })
			},
			narration: { what: { en: 'Each dial moves against its own slope: w becomes {frame.w1} and b becomes {frame.b1}.', zh: '每个旋钮都逆着自己的斜率移动：w 变成 {frame.w1}，b 变成 {frame.b1}。' } }
		},
		step: {
			label: { en: 'Gradient descent', zh: '梯度下降' },
			kind: 'update',
			narration: { what: { en: 'After {frame.step} updates the score is {frame.loss}.', zh: '更新 {frame.step} 次之后，分数是 {frame.loss}。' } }
		},
		end: {
			label: { en: 'Where it stopped', zh: '停在哪里' },
			kind: 'observe',
			narration: { what: { en: 'After {frame.step} updates: w = {frame.w}, b = {frame.b}, score {frame.loss}.', zh: '更新 {frame.step} 次之后：w = {frame.w}，b = {frame.b}，分数 {frame.loss}。' } }
		}
	},
	schema: {
		w: { kind: 'scalar', label: { en: 'Weight w', zh: '权重 w' }, role: 'param', tol: 1e-12 },
		b: { kind: 'scalar', label: { en: 'Bias b', zh: '偏置 b' }, role: 'param', tol: 1e-12 },
		gw: { kind: 'scalar', label: { en: 'Slope under w (dL/dw)', zh: 'w 方向的斜率 (dL/dw)' }, role: 'grad', tol: 1e-12 },
		gb: { kind: 'scalar', label: { en: 'Slope under b (dL/db)', zh: 'b 方向的斜率 (dL/db)' }, role: 'grad', tol: 1e-12 },
		path: { kind: 'tensor', label: { en: 'The dials visited so far (w, b)', zh: '到目前为止走过的旋钮位置 (w, b)' }, shape: ['k', 2], role: 'param', tol: 1e-12 }
	},
	defaults: { lr: 0.1, steps: 20 },
	*run(raw, ctx): ProcessRun<GdState, GdOpt> {
		const lr = ctx.opt.lr;
		let { w, b } = dialsOf(raw);
		const pts: number[][] = [[w, b]];
		const fine = ctx.detail === 'fine';
		let s = scoreAndGradient(w, b);
		if (!fine) yield { phase: 'start', step: 0, state: { w, b, path: pathOf(pts) }, metrics: { step: 0, w, b, loss: s.loss, lr } };
		for (let k = 0; k < ctx.opt.steps; k++) {
			if (fine) {
				yield { phase: 'loss', step: k, state: { w, b, path: pathOf(pts) }, metrics: { step: k, w, b, loss: s.loss, lr } };
				yield { phase: 'grad', step: k, state: { w, b, gw: s.gw, gb: s.gb, path: pathOf(pts) }, metrics: { step: k, w, b, loss: s.loss, gw: s.gw, gb: s.gb, lr } };
			}
			const w1 = w - lr * s.gw;
			const b1 = b - lr * s.gb;
			const next = scoreAndGradient(w1, b1);
			pts.push([w1, b1]);
			if (fine)
				yield {
					phase: 'update',
					step: k,
					state: { w: w1, b: b1, gw: s.gw, gb: s.gb, path: pathOf(pts) },
					metrics: { step: k, w, b, loss: s.loss, gw: s.gw, gb: s.gb, lr, w1, b1, loss1: next.loss }
				};
			else yield { phase: 'step', step: k + 1, state: { w: w1, b: b1, gw: s.gw, gb: s.gb, path: pathOf(pts) }, metrics: { step: k + 1, w: w1, b: b1, loss: next.loss, gw: s.gw, gb: s.gb, lr } };
			w = w1;
			b = b1;
			s = next;
		}
		if (fine) yield { phase: 'end', step: ctx.opt.steps, state: { w, b, path: pathOf(pts) }, metrics: { step: ctx.opt.steps, w, b, loss: s.loss, lr } };
	}
};

export type RatesOpt = GdOpt & { lrSlow: number; lrBig: number };

export const rates: ProcessDef<unknown, RatesOpt, GdState> = {
	...gd,
	id: 'rates',
	title: { en: 'Gradient descent at three learning rates', zh: '三种学习率下的梯度下降' },
	defaults: { lr: 0.1, lrSlow: 0.01, lrBig: 0.7, steps: 30 },
	*run(raw, ctx): ProcessRun<GdState, RatesOpt> {
		const o = { ...rates.defaults, ...ctx.opt };
		const a = gd.run(raw, { ...ctx, detail: 'coarse', opt: { lr: o.lr, steps: o.steps } });
		const slow = gd.run(raw, { ...ctx, detail: 'coarse', opt: { lr: o.lrSlow, steps: o.steps } });
		const big = gd.run(raw, { ...ctx, detail: 'coarse', opt: { lr: o.lrBig, steps: o.steps } });
		for (let r = a.next(); !r.done; r = a.next()) {
			const s = slow.next();
			const g = big.next();
			const m = r.value.metrics ?? {};
			yield { ...r.value, metrics: { ...m, lrSlow: o.lrSlow, lossSlow: s.done ? NaN : (s.value.metrics?.loss ?? NaN), lrBig: o.lrBig, lossBig: g.done ? NaN : (g.value.metrics?.loss ?? NaN) } };
		}
	}
};

const inner = trainingProcess(moonsNet);
type NetOpt = (typeof moonsNet)['defaults'];

export const realNet: ProcessDef<unknown, NetOpt, TrainState> = {
	...inner,
	id: 'realNet'
};

export const realPair: ProcessDef<unknown, NetOpt, TrainState> = {
	...inner,
	id: 'realPair',
	title: { en: 'The same network from two random starts', zh: '同一个网络，两个随机起点' },
	*run(raw, ctx): ProcessRun<TrainState, NetOpt> {
		const a = inner.run(raw, { ...ctx, detail: 'coarse' });
		const b = inner.run(raw, { ...ctx, detail: 'coarse', rng: createRng(ctx.rng.seed + 1) });
		const first = (st: Partial<TrainState> | undefined) => st?.params?.['net.0.weight']?.data[0];
		for (let r = a.next(); !r.done; r = a.next()) {
			const q = b.next();
			const m: Record<string, number> = { ...r.value.metrics };
			const l2 = q.done ? undefined : q.value.metrics?.loss;
			if (l2 !== undefined) m.loss2 = l2;
			const wa = first(r.value.state);
			const wb = q.done ? undefined : first(q.value.state);
			if (wa !== undefined && wb !== undefined) {
				m.w0 = wa;
				m.w0b = wb;
			}
			yield { ...r.value, metrics: m };
		}
	}
};
