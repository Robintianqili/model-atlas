// flow-2d: trainer 'fit' (kind B). The velocity network v(x, y, t), an MLP 3 -> 64 -> 64 -> 64 -> 2 with SiLU, learns
// flow matching on the learner's points: each pair joins noise x0 ~ N(0, I) to a data point x1 by a straight line, the
// network sees the point x_t = (1 − t)·x0 + t·x1 and t, and its target is the line's velocity x1 − x0 (mean squared
// error). Adam 1e-3, batch 256, 1,500 steps. Twin: pipeline/flow-2d/twin.py.
// The sources do not run 'train:fit' directly: process.ts wraps it as 'train:flow' (no frames on an empty input, and
// Euler sampling after training); see the notes (Kit issues).
import type { TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import { along, BATCH, dataOf, drawPair, SIZES } from './flow.ts';

/** particles, K, Kfew and sample are read by the 'train:flow' wrapper (Euler sampling after training) */
export type FlowOpt = TrainOpt & { particles: number; K: number; Kfew: number; sample: boolean };
export type FlowBatch = { inp: Var; target: Var };

export const fit: TrainerDef<unknown, MLP, FlowBatch, FlowOpt> = {
	id: 'fit',
	title: { en: 'Train the velocity network', zh: '训练速度网络' },
	defaults: { seed: 1, lr: 1e-3, batch: BATCH, steps: 1500, snapshotEvery: 100, optimizer: 'adam', particles: 400, K: 50, Kfew: 4, sample: true },
	model: (rng) => new MLP('net', SIZES, 'silu', rng),
	batch(raw, _step, rng, opt) {
		const p = dataOf(raw);
		const B = opt.batch;
		const inp = new Float64Array(B * 3);
		const target = new Float64Array(B * 2);
		for (let i = 0; i < B; i++) {
			const q = drawPair(p, rng);
			inp[3 * i] = along(q.x0, q.x1, q.t);
			inp[3 * i + 1] = along(q.y0, q.y1, q.t);
			inp[3 * i + 2] = q.t;
			target[2 * i] = q.x1 - q.x0;
			target[2 * i + 1] = q.y1 - q.y0;
		}
		return { inp: input({ shape: [B, 3], data: inp }), target: input({ shape: [B, 2], data: target }) };
	},
	loss: (m, b, rec) => ({ loss: mse(m.forward(b.inp, rec), b.target) })
};
