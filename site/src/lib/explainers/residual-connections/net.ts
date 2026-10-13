// residual-connections: the deep MLP, with or without skip connections, on the kit's autograd, and its trainer.
//   layer 0          h0 = tanh(W0·x + b0)                          (2 coordinates -> WIDTH units)
//   layer l, 1..D−2  plain:      hl = tanh(Wl·h(l−1) + bl)
//                    with skips: hl = h(l−1) + tanh(Wl·h(l−1) + bl)  (the skip adds the layer's input to its output)
//   layer D−1        scores = W(D−1)·h(D−2) + b(D−1)                (WIDTH units -> 2 class scores)
// Parameter names follow a PyTorch nn.ModuleList called 'net' ('net.0.weight', [out, in]); the init is PyTorch's
// default, U(±1/√in), drawn from the rng in layer order. Skips add no parameters, so the plain and the skip network
// built from the same rng start from the very same weights.
import type { Module, Rec, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { Linear, namedParams } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { accuracy } from '../../atlas/train/metrics.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { LR, MAX_DEPTH, MOONS, STEPS, WIDTH } from './data.ts';

/** Every intermediate of one forward pass: z (each layer's weighted sums), a = tanh(z) (what the layer computes), h
 *  (the hidden state passed on: a, plus the layer's input when it has a skip) and the class scores. */
export interface Trace {
	z: Var[];
	a: Var[];
	h: Var[];
	scores: Var;
}

export class DeepNet implements Module {
	readonly layers: Linear[] = [];
	readonly depth: number;
	readonly skip: boolean;
	readonly width: number;
	constructor(depth: number, skip: boolean, rng: Rng, width = WIDTH) {
		this.depth = depth;
		this.skip = skip;
		this.width = width;
		for (let i = 0; i < depth; i++) this.layers.push(new Linear(`net.${i}`, i === 0 ? 2 : width, i === depth - 1 ? 2 : width, rng));
	}
	/** Has layer l a skip connection? (Only the layers whose input and output have the same size.) */
	skips(l: number): boolean {
		return this.skip && l > 0 && l < this.depth - 1;
	}
	trace(x: Var, rec?: Rec): Trace {
		const z: Var[] = [];
		const a: Var[] = [];
		const h: Var[] = [];
		let cur = x;
		for (let l = 0; l < this.depth - 1; l++) {
			z.push(this.layers[l].forward(cur));
			a.push(op.tanh(z[l]));
			cur = this.skips(l) ? op.add(cur, a[l]) : a[l];
			h.push(cur);
			rec?.(`h.${l}`, cur);
		}
		const scores = this.layers[this.depth - 1].forward(cur);
		rec?.('scores', scores);
		return { z, a, h, scores };
	}
	forward(x: Var, rec?: Rec): Var {
		return this.trace(x, rec).scores;
	}
	named(): Record<string, Var> {
		return namedParams(...this.layers);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}

export const pointsVar = (): Var => input({ shape: [MOONS.n, 2], data: MOONS.xy });
export const paramCount = (depth: number): number => (2 * WIDTH + WIDTH) + (depth - 2) * (WIDTH * WIDTH + WIDTH) + (WIDTH * 2 + 2);

export type DeepOpt = TrainOpt & { depth: number; skip: number };
type Batch = { x: Var; y: Int32Array };

/** The deep MLP trained with plain gradient descent on all 120 points, mean cross-entropy (opt.depth, opt.skip). */
export const deep: TrainerDef<unknown, DeepNet, Batch, DeepOpt> = {
	id: 'deep',
	title: { en: 'A deep MLP trained with gradient descent', zh: '用梯度下降训练的深层 MLP' },
	defaults: { seed: 1, lr: LR, batch: 0, steps: STEPS, snapshotEvery: 50, optimizer: 'sgd', depth: MAX_DEPTH, skip: 1 },
	model: (rng, o) => new DeepNet(o.depth ?? MAX_DEPTH, !!o.skip, rng),
	batch: () => ({ x: pointsVar(), y: MOONS.label }),
	loss: (m, b, rec) => ({ loss: crossEntropy(m.forward(b.x, rec), b.y) }),
	metrics: (m) => ({ accuracy: accuracy(m.forward(pointsVar()), MOONS.label) })
};
