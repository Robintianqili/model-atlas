// optimizers: the "real" network of the page, copied from the loss-gradient-descent package (its net.ts): a 2-8-8-2 tanh
// network, 114 weights and biases, that sorts 120 points of the two-moons set by their cross-entropy score. The range
// test and the schedule demo train it with plain SGD on all 120 points at once (no batches, so no noise).
import { MLP } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { backward, input, noGrad, zeroGrad, type Var } from '../../atlas/autograd/var.ts';
import { SGD } from '../../atlas/autograd/optim.ts';
import { moons } from '../../atlas/data/points2d.ts';
import type { Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

/** The two-moons points (seeded: every learner gets the same 120 points). */
export const MOONS = moons(120, createRng(11), { noise: 0.15 });
export const SIZES = [2, 8, 8, 2];
/** weights and biases of the 2-8-8-2 network */
export const PARAMS = SIZES.slice(1).reduce((a, n, i) => a + n * SIZES[i] + n, 0);
const X: Var = input({ shape: [MOONS.n, 2], data: MOONS.xy });

/** A fresh network (its starting weights from rng) and an SGD optimizer on it. */
export function netAndSgd(rng: Rng, lr: number): { net: MLP; opt: SGD } {
	const net = new MLP('net', SIZES, 'tanh', rng);
	return { net, opt: new SGD(net, { lr }) };
}

/** One full-batch step: the loss of the current weights, its gradient by backpropagation, then the update at `lr`. */
export function trainStep(net: MLP, opt: SGD, lr: number): number {
	zeroGrad(net);
	const loss = crossEntropy(net.forward(X), MOONS.label);
	backward(loss);
	opt.lr = lr;
	opt.step();
	return loss.item();
}

/** The loss of the current weights on all 120 points. */
export const lossOf = (net: MLP) => noGrad(() => crossEntropy(net.forward(X), MOONS.label).item());
