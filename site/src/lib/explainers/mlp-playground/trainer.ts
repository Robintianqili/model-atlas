// mlp-playground: trainer 'fit' (kind B; process 'train:fit'). An MLP, 2 inputs (x, y) -> hidden layers -> 2 class
// scores, learns to sort 2-D points into two classes. Defaults: 2 -> [4, 4] -> 2, tanh, Adam lr 0.03, all the points
// every step (batch 0), 300 steps, a snapshot every 5. Extra options: hidden, act ('tanh' | 'relu'), data (a points2d
// raw value that replaces the learner's points, e.g. the spiral). Twin: pipeline/mlp-playground/twin.py.
//   npm run atlas -- run mlp-playground run --frames 0..20
import type { Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { sampleBatch } from '../../atlas/data/points2d.ts';
import type { PointsRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { dataOf, evalPoints, neuronGrids, RES, type Act, type Params } from './net.ts';

export type FitOpt = TrainOpt & { hidden: number[]; act: Act; data?: PointsRaw | null };
export type FitBatch = { x: Var; y: Int32Array };

const EMPTY: Tensor = { shape: [0, 0], data: new Float64Array(0) };
type S = { params?: Params; act?: unknown };
const grids = (s: S, last: boolean) => (s.params ? neuronGrids(s.params, s.act === 'relu' ? 'relu' : 'tanh', last) : EMPTY);

export const fit: TrainerDef<unknown, MLP, FitBatch, FitOpt> = {
	id: 'fit',
	title: { en: 'Train the playground network', zh: '训练游乐场里的网络' },
	defaults: { seed: 1, lr: 0.03, batch: 0, steps: 300, snapshotEvery: 5, optimizer: 'adam', hidden: [4, 4], act: 'tanh' },
	model: (rng, opt) => new MLP('net', [2, ...opt.hidden, 2], opt.act, rng),
	batch(raw, _step, rng, opt) {
		const p = dataOf(raw, opt);
		if (!(opt.batch > 0) || opt.batch >= p.n) return { x: input({ shape: [p.n, 2], data: p.xy }), y: p.label };
		const b = sampleBatch(p, opt.batch, rng);
		return { x: input(b.x), y: b.label };
	},
	// mean cross-entropy over the batch; with no points there is nothing to learn (loss 0, every gradient 0)
	loss: (m, b, rec) => ({ loss: b.y.length ? crossEntropy(m.forward(b.x, rec), b.y) : op.scale(op.sum(m.layers[0].weight), 0) }),
	derived: {
		grid: { kind: 'tensor', label: { en: 'Share of class 1 everywhere', zh: '各处属于第 1 类的比例' }, shape: [RES, RES], role: 'output', derive: (s: S) => grids(s, true) },
		grids: { kind: 'tensor', label: { en: 'Every neuron over the plane', zh: '每个神经元在平面上的输出' }, role: 'activation', derive: (s: S) => grids(s, false) }
	},
	metrics(m, raw, opt): Record<string, number> {
		const p = dataOf(raw, opt);
		if (p.n === 0) return {};
		const params: Params = {};
		for (const [k, v] of Object.entries(m.named())) params[k] = v.value;
		return { accuracy: evalPoints(params, m.act === 'relu' ? 'relu' : 'tanh', p).accuracy, points: p.n };
	}
};
