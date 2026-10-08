// score-2d: trainer 'fit' (kind B). Denoising score matching on the learner's points: take a data point x, a noise
// level σ (log-uniform from 0.04 to 1.28) and noise ε ~ N(0, I); the network d, an MLP 3 -> 64 -> 64 -> 2 with SiLU, sees
// the noisy point x̃ = x + σ·ε and log σ, and its target is −ε (mean squared error). The score is then s = d / σ, and
// −ε / σ = (x − x̃) / σ² is what denoising score matching asks the score to be (Vincent 2011). Adam 3e-3, batch 256,
// 1,000 steps. Twin: pipeline/score-2d/twin.py.
import type { TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import { BATCH, dataOf, drawPair, SIZES } from './score.ts';

export type FitBatch = { inp: Var; target: Var };

export const fit: TrainerDef<unknown, MLP, FitBatch, TrainOpt> = {
	id: 'fit',
	title: { en: 'Train the score network', zh: '训练分数网络' },
	defaults: { seed: 1, lr: 3e-3, batch: BATCH, steps: 1000, snapshotEvery: 50, optimizer: 'adam' },
	model: (rng) => new MLP('net', SIZES, 'silu', rng),
	batch(raw, _step, rng, opt) {
		const p = dataOf(raw);
		const B = opt.batch;
		const inp = new Float64Array(B * 3);
		const target = new Float64Array(B * 2);
		for (let i = 0; i < B; i++) {
			const q = drawPair(p, rng);
			inp[3 * i] = q.x + q.s * q.ex;
			inp[3 * i + 1] = q.y + q.s * q.ey;
			inp[3 * i + 2] = Math.log(q.s);
			target[2 * i] = -q.ex;
			target[2 * i + 1] = -q.ey;
		}
		return { inp: input({ shape: [B, 3], data: inp }), target: input({ shape: [B, 2], data: target }) };
	},
	loss: (m, b, rec) => ({ loss: mse(m.forward(b.inp, rec), b.target) })
};
