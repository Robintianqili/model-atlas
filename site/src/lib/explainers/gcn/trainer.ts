// gcn: trainer 'gcn' (kind B; process 'train:gcn'). The GCN of pipeline/gcn/model.py, trained live in the browser on the
// learner's graph from the members the learner labelled: 34 -> 4 -> 2 graph layers (tanh) and a 2 -> 2 head, Adam lr
// 0.01, the whole graph every step, 200 steps (the recipe of pipeline/gcn/train.py). The loss sees only the labelled
// members; the others get their faction from the graph. Twin: pipeline/gcn/twin.py (state_dict names gc1.weight, …).
//   npm run atlas -- run gcn train --frames 0..20
import type { Module, Rec, Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { Linear } from '../../atlas/autograd/nn.ts';
import { op } from '../../atlas/autograd/ops.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import { adjacency, forwardGcn, graphOf, LABELS, N, normalize, sorted, type Params } from './graph.ts';

export class Gcn implements Module {
	readonly gc1: Linear;
	readonly gc2: Linear;
	readonly head: Linear;
	constructor(rng: Rng) {
		this.gc1 = new Linear('gc1', N, 4, rng);
		this.gc2 = new Linear('gc2', 4, 2, rng);
		this.head = new Linear('head', 2, 2, rng);
	}
	/** tanh(Â (H Wᵀ) + b): the bias is added after the neighbours' rows are summed, as in model.py */
	private conv(l: Linear, h: Var, ahat: Var): Var {
		return op.tanh(op.add(op.matmul(ahat, op.linear(h, l.weight, null)), l.bias!));
	}
	forward(ahat: Var, x: Var, rec?: Rec): Var {
		const h = this.conv(this.gc1, x, ahat);
		rec?.('act1', h);
		const e = this.conv(this.gc2, h, ahat);
		rec?.('act2', e);
		return this.head.forward(e, rec);
	}
	named(): Record<string, Var> {
		return { ...this.gc1.named(), ...this.gc2.named(), ...this.head.named() };
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}

export type Batch = { ahat: Tensor; x: Tensor; y: Int32Array; labelled: Int32Array };
const ahatOf = (raw: unknown): Tensor => ({ shape: [N, N], data: normalize(adjacency(graphOf(raw)), N).ahat });
const EYE: Tensor = { shape: [N, N], data: Float64Array.from({ length: N * N }, (_, k) => (k % (N + 1) === 0 ? 1 : 0)) };
const paramsOf = (m: Gcn): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.value]));

export const gcn: TrainerDef<unknown, Gcn, Batch, TrainOpt> = {
	id: 'gcn',
	title: { en: 'Train the GCN from the labelled members', zh: '用已标注的成员训练 GCN' },
	defaults: { seed: 1, lr: 0.01, batch: 1, steps: 200, snapshotEvery: 5, optimizer: 'adam' },
	model: (rng) => new Gcn(rng),
	batch: (raw) => ({ ahat: ahatOf(raw), x: EYE, y: Int32Array.from(LABELS), labelled: Int32Array.from(graphOf(raw).labelled) }),
	loss(m, b, rec) {
		const z = m.forward(input(b.ahat), input(b.x), rec);
		// no labelled member: nothing to learn (loss 0, every gradient 0)
		if (!b.labelled.length) return { loss: op.scale(op.sum(m.head.weight), 0) };
		return { loss: crossEntropy(op.gather(z, 0, b.labelled), Array.from(b.labelled, (i) => b.y[i])) };
	},
	metrics(m, raw) {
		const s = sorted(forwardGcn(paramsOf(m), ahatOf(raw).data).z, graphOf(raw).labelled);
		return { accuracy: s.acc, right: s.right, rest: s.rest };
	}
};
