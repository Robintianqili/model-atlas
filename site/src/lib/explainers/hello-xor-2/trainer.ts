// hello-xor-2: trainer 'xorSigmoid' (kind B). The same 2-2-1 network as pipeline/hello-xor-2/model.py (tanh hidden
// layer, SIGMOID output), trained live in the browser on the four XOR points (full batch, binary cross-entropy computed
// from the logit, like torch's binary_cross_entropy_with_logits). The loaders wrap this TrainerDef with
// trainingProcess() and register it as process 'train:xorSigmoid'.
//   npm run atlas -- run hello-xor-2 train --frames 0..20
// Python twin (stronger badge): pipeline/hello-xor-2/twin.py, checked by `npm run atlas -- parity hello-xor-2`.
// Randomness: model() gets the run's rng.fork('init'); the source's opt.seed picks the run (default 1).
import type { AtlasSpec, Module, Rec, Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { bceWithLogits } from '../../atlas/autograd/loss.ts';
import { input, noGrad, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { stackToSpec, type SpecStackItem } from '../../atlas/autograd/tospec.ts';
import { XS, YS } from './process.ts';

/** The four XOR points as tensors (TS is the source of truth for the data the twin reads). */
export const XOR: { x: Tensor; y: Tensor } = {
	x: { shape: [4, 2], data: Float64Array.from(XS.flat()) },
	y: { shape: [4, 1], data: Float64Array.from(YS) }
};

/**
 * MLP('net', [2, hidden, 1], 'tanh') followed by a sigmoid: the PyTorch twin's nn.Sequential(Linear, Tanh, Linear)
 * plus torch.sigmoid. Parameter names stay net.0.*, net.2.*; forward() records the probability as 'out'.
 */
export class SigmoidMLP implements Module {
	readonly mlp: MLP;
	constructor(hidden: number, rng: Rng) {
		this.mlp = new MLP('net', [2, hidden, 1], 'tanh', rng);
	}
	/** The output score before the sigmoid (the logit). */
	logits(x: Var, rec?: Rec): Var {
		return this.mlp.forward(x, rec);
	}
	/** The probability that XOR is true. */
	forward(x: Var, rec?: Rec): Var {
		const p = op.sigmoid(this.logits(x, rec));
		rec?.('out', p);
		return p;
	}
	named(): Record<string, Var> {
		return this.mlp.named();
	}
	params(): Var[] {
		return this.mlp.params();
	}
	/** The kind-A spec on the current weights (a 'live-model' source): net.0, net.1 (tanh), net.2, out (sigmoid). */
	toSpec(o: { slug: string; inputAxes?: string[] }): { spec: AtlasSpec; weights: Float32Array } {
		const [l0, l1] = this.mlp.layers;
		const items: SpecStackItem[] = [
			{ kind: 'linear', id: l0.name, weight: l0.weight, bias: l0.bias },
			{ kind: 'act', id: 'net.1', act: 'tanh' },
			{ kind: 'linear', id: l1.name, weight: l1.weight, bias: l1.bias },
			{ kind: 'act', id: 'out', act: 'sigmoid' }
		];
		return stackToSpec(items, { ...o, adapter: 'xor-point', inputKind: 'points2d' });
	}
}

export type XorOpt = TrainOpt & { hidden: number };

/** How many of the four points the network gets right (probability above 0.5 means "XOR is true"). */
export function correct(m: SigmoidMLP): number {
	const y = noGrad(() => m.forward(input(XOR.x))).value.data;
	let n = 0;
	for (let i = 0; i < 4; i++) if ((y[i] > 0.5 ? 1 : 0) === YS[i]) n++;
	return n;
}

export const xorSigmoid: TrainerDef<unknown, SigmoidMLP, { x: Tensor; y: Tensor }, XorOpt> = {
	id: 'xorSigmoid',
	title: { en: 'Train the XOR network (sigmoid output)', zh: '训练 XOR 网络（sigmoid 输出）' },
	defaults: { seed: 1, lr: 0.05, batch: 4, steps: 300, snapshotEvery: 5, optimizer: 'adam', hidden: 2 },
	model: (rng, opt) => new SigmoidMLP(opt.hidden, rng),
	// full batch every step: the data never changes, so the learner's point (rt.input) is not used here
	batch: () => XOR,
	loss: (m, b, rec) => {
		const z = m.logits(input(b.x), rec);
		rec('out', op.sigmoid(z));
		return { loss: bceWithLogits(z, input(b.y)) };
	},
	metrics: (m) => ({ correct: correct(m) })
};
