// loss-gradient-descent: the "real" network of the page. The same rule (score, gradient, step) trains a 2-8-8-2 tanh
// network, 114 weights and biases, to sort 120 points of the two-moons set, with plain gradient descent on its
// cross-entropy score. A TrainerDef kept out of trainer.ts on purpose: process.ts wraps it as the process 'realNet'
// (trainingProcess), so its source can ignore the learner's dials (a process whose input does not change keeps its run;
// a train source restarts on every new input).
import type { TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import { moons } from '../../atlas/data/points2d.ts';
import { createRng } from '../../atlas/core/rng.ts';

/** The two-moons points (seeded: every learner gets the same 120 points). */
export const MOONS = moons(120, createRng(11), { noise: 0.15 });
export const SIZES = [2, 8, 8, 2];
/** weights and biases of the 2-8-8-2 network */
export const PARAMS = SIZES.slice(1).reduce((a, n, i) => a + n * SIZES[i] + n, 0);

type Batch = { x: Var; y: Int32Array };

function accuracy(m: MLP): number {
	const z = m.forward(input({ shape: [MOONS.n, 2], data: MOONS.xy })).value.data;
	let right = 0;
	for (let i = 0; i < MOONS.n; i++) if ((z[2 * i + 1] > z[2 * i] ? 1 : 0) === MOONS.label[i]) right++;
	return right / MOONS.n;
}

export const moonsNet: TrainerDef<unknown, MLP, Batch, TrainOpt> = {
	id: 'moonsNet',
	title: { en: 'Gradient descent on a 2-8-8-2 network', zh: '在 2-8-8-2 网络上做梯度下降' },
	defaults: { seed: 1, lr: 0.5, batch: 0, steps: 300, snapshotEvery: 10, optimizer: 'sgd' },
	model: (rng) => new MLP('net', SIZES, 'tanh', rng),
	batch: () => ({ x: input({ shape: [MOONS.n, 2], data: MOONS.xy }), y: MOONS.label }),
	loss: (m, b, rec) => ({ loss: crossEntropy(m.forward(b.x, rec), b.y) }),
	metrics: (m) => ({ accuracy: accuracy(m) })
};
