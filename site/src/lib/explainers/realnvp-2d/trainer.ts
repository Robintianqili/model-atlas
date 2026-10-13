// realnvp-2d: trainer 'fit' (kind B). A RealNVP flow of LAYERS affine coupling layers trained by maximum likelihood on
// the learner's points (or on opt.data, a points2d raw value, for the two-rings run of the limits step). Layer k keeps
// coordinate k % 2 and moves the other one: v' = v·e^s + t, with (s, t) from the small network c<k> of the kept
// coordinate (s = tanh of its first output). The loss is the mean negative log-likelihood of a batch of data points:
//   −log p(x) = ½|z|² + log 2π − Σ_k s_k,   z = f(x)
// The last layer of every small network starts at 0, so the untrained flow is the identity (z = x, Σ s = 0).
// Adam 3e-3, batch 128 points drawn with replacement, 1,200 steps. Twin: pipeline/realnvp-2d/twin.py.
//   npm run atlas -- run realnvp-2d train --frames 0..20
import type { Module, Rec, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP, namedParams } from '../../atlas/autograd/nn.ts';
import { input, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { dataOf, keepOf, LAYERS, LOG2PI, nllOf, SIZES, type Params } from './nvp.ts';

export type FitOpt = TrainOpt & { data?: unknown };
export type FitBatch = { x: Var };

export class Flow implements Module {
	readonly nets: MLP[];
	constructor(rng: Rng) {
		this.nets = Array.from({ length: LAYERS }, (_, k) => new MLP(`c${k}`, SIZES, 'tanh', rng));
		for (const m of this.nets) {
			const last = m.layers[m.layers.length - 1];
			last.weight.data.fill(0);
			last.bias!.data.fill(0);
		}
	}
	/** z = f(x) for a batch x [B, 2] and the log-determinant Σ s [B, 1] */
	forward(x: Var, rec?: Rec): { z: Var; ld: Var } {
		let h = x;
		let ld: Var | null = null;
		for (let k = 0; k < LAYERS; k++) {
			const c = keepOf(k);
			const u = op.slice(h, 1, c, c + 1);
			const v = op.slice(h, 1, 1 - c, 2 - c);
			const o = this.nets[k].forward(u, rec);
			const s = op.tanh(op.slice(o, 1, 0, 1));
			const v2 = op.add(op.mul(v, op.exp(s)), op.slice(o, 1, 1, 2));
			h = c === 0 ? op.concat([u, v2], 1) : op.concat([v2, u], 1);
			ld = ld ? op.add(ld, s) : s;
		}
		return { z: h, ld: ld! };
	}
	named(): Record<string, Var> {
		return namedParams(...this.nets);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}

const paramsOf = (m: Flow): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, { shape: v.shape.slice(), data: v.data }]));

export const fit: TrainerDef<unknown, Flow, FitBatch, FitOpt> = {
	id: 'fit',
	title: { en: 'Train the flow', zh: '训练流模型' },
	defaults: { seed: 1, lr: 3e-3, batch: 128, steps: 1200, snapshotEvery: 50, optimizer: 'adam' },
	model: (rng) => new Flow(rng),
	batch(raw, _step, rng, opt) {
		const p = dataOf(opt.data ?? raw);
		const B = opt.batch;
		const x = new Float64Array(p.n ? 2 * B : 0);
		for (let i = 0; p.n && i < B; i++) {
			const j = rng.int(p.n);
			(x[2 * i] = p.xy[2 * j]), (x[2 * i + 1] = p.xy[2 * j + 1]);
		}
		return { x: input({ shape: [x.length / 2, 2], data: x }) };
	},
	loss(m, b, rec) {
		const { z, ld } = m.forward(b.x, rec);
		const B = b.x.shape[0];
		const nll = op.sub(op.add(op.scale(op.sum(op.square(z), 1), 0.5), LOG2PI), op.reshape(ld, [B]));
		return { loss: op.mean(nll) };
	},
	/** the mean −log p over every data point (not only the batch) */
	metrics: (m, raw, opt) => ({ nllAll: nllOf(paramsOf(m), dataOf(opt?.data ?? raw)) })
};
