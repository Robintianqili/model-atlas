// The flow-2d training step used for the W1-B1 speed gate (design §8.1 acceptance 5): a velocity MLP
// v(x, y, t) 3→64→64→64→2 with SiLU, Adam 1e-3, flow-matching loss on batch 256 (straight paths x_t = (1 − t)·x0 + t·x1,
// target x1 − x0, x0 ~ N(0, I), t ~ U(0, 1)). Shared by bench-flow.ts and train/bench.test.ts.
import type { TrainOpt, TrainerDef } from '../../../src/lib/atlas/types/index.ts';
import { Var, input } from '../../../src/lib/atlas/autograd/var.ts';
import { mse } from '../../../src/lib/atlas/autograd/loss.ts';
import { MLP } from '../../../src/lib/atlas/autograd/nn.ts';
import type { Points2D } from '../../../src/lib/atlas/data/points2d.ts';

export type FlowBatch = { inp: Var; target: Var };

export const flowTrainer: TrainerDef<Points2D, MLP, FlowBatch, TrainOpt> = {
	id: 'flow',
	title: { en: 'Flow matching (2-D)', zh: '流匹配（二维）' },
	defaults: { seed: 1, lr: 1e-3, batch: 256, steps: 400, snapshotEvery: 50, optimizer: 'adam' },
	model: (rng) => new MLP('v', [3, 64, 64, 64, 2], 'silu', rng),
	batch(data, _step, rng, opt) {
		const B = opt.batch;
		const inp = new Float64Array(B * 3);
		const target = new Float64Array(B * 2);
		for (let i = 0; i < B; i++) {
			const j = rng.int(data.n);
			const x1 = data.xy[2 * j];
			const y1 = data.xy[2 * j + 1];
			const x0 = rng.normal();
			const y0 = rng.normal();
			const t = rng.next();
			inp[3 * i] = (1 - t) * x0 + t * x1;
			inp[3 * i + 1] = (1 - t) * y0 + t * y1;
			inp[3 * i + 2] = t;
			target[2 * i] = x1 - x0;
			target[2 * i + 1] = y1 - y0;
		}
		return { inp: input({ shape: [B, 3], data: inp }), target: input({ shape: [B, 2], data: target }) };
	},
	loss(m, b, rec) {
		return { loss: mse(m.forward(b.inp, rec), b.target) };
	}
};
