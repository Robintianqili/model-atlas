// hello-xor: trainer 'xor' (kind B, design §4.5). The same 2-2-1 tanh network as pipeline/hello-xor/model.py, trained
// live in the browser on the four XOR points (full batch, mean squared error). The loaders wrap this TrainerDef with
// trainingProcess() and register it as process 'train:xor'.
//   npm run atlas -- run hello-xor train --frames 0..20
// Python twin (stronger badge): pipeline/hello-xor/twin.py, checked by `npm run atlas -- parity hello-xor`.
// Randomness: model() gets the run's rng.fork('init'); the source's opt.seed picks the run (default 1).
import type { Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import { input, noGrad } from '../../atlas/autograd/var.ts';
import { XS, YS } from './process.ts';

/** The four XOR points as tensors (TS is the source of truth for the data the twin reads). */
export const XOR: { x: Tensor; y: Tensor } = {
	x: { shape: [4, 2], data: Float64Array.from(XS.flat()) },
	y: { shape: [4, 1], data: Float64Array.from(YS) }
};

export type XorOpt = TrainOpt & { hidden: number };

/** How many of the four points the network gets right (output above 0.5 means "XOR is true"). */
export function correct(m: MLP): number {
	const y = noGrad(() => m.forward(input(XOR.x))).value.data;
	let n = 0;
	for (let i = 0; i < 4; i++) if ((y[i] > 0.5 ? 1 : 0) === YS[i]) n++;
	return n;
}

export const xor: TrainerDef<unknown, MLP, { x: Tensor; y: Tensor }, XorOpt> = {
	id: 'xor',
	title: { en: 'Train the XOR network', zh: '训练 XOR 网络' },
	defaults: { seed: 1, lr: 0.05, batch: 4, steps: 300, snapshotEvery: 5, optimizer: 'adam', hidden: 2 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, 1], 'tanh', rng),
	// full batch every step: the data never changes, so the learner's point (rt.input) is not used here
	batch: () => XOR,
	loss: (m, b, rec) => ({ loss: mse(m.forward(input(b.x), rec), input(b.y)) }),
	metrics: (m) => ({ correct: correct(m) })
};
