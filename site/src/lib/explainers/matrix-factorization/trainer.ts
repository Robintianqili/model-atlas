// matrix-factorization: trainer 'fit' (kind B; process 'train:fit'). Two numbers for every person (U, [25, 2]) and every
// movie (V, [20, 2]); a predicted rating is a dot product, U[u]·V[i]. Plain stochastic gradient descent: each step takes
// ONE known rating at random (rng.int over all of them, with replacement), and its loss is that rating's squared error
// (pred − r)². Defaults: lr 0.05, 4,000 steps, a snapshot every 50 (U and V for the map and the matrix), starting
// numbers uniform in ±0.5. opt.cold drops the learner's ratings (a new user who rated nothing). metrics() is the error
// over every known rating at the snapshots: mse (all), you (yours only), n (how many).
// Twin: pipeline/matrix-factorization/twin.py (nn.Embedding 'user' and 'item').   npm run atlas -- run matrix-factorization train --frames 0..20
import type { Module, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { mse as mseLoss } from '../../atlas/autograd/loss.ts';
import { N_ITEMS, N_USERS, initFactors, known, mse } from './world.ts';

export type FitOpt = TrainOpt & { init: number; cold: boolean };
export type FitBatch = { u: Int32Array; i: Int32Array; r: Float64Array };

export class MF implements Module {
	readonly user: Var;
	readonly item: Var;
	constructor(rng: Rng, init: number) {
		const { U, V } = initFactors(rng, init);
		this.user = param('user.weight', { shape: [N_USERS, 2], data: U });
		this.item = param('item.weight', { shape: [N_ITEMS, 2], data: V });
	}
	named(): Record<string, Var> {
		return { 'user.weight': this.user, 'item.weight': this.item };
	}
	params(): Var[] {
		return [this.user, this.item];
	}
}

const pick = (k: ReturnType<typeof known>, j: number): FitBatch => ({ u: k.u.slice(j, j + 1), i: k.i.slice(j, j + 1), r: k.r.slice(j, j + 1) });

export const fit: TrainerDef<unknown, MF, FitBatch, FitOpt> = {
	id: 'fit',
	title: { en: 'Learn two numbers per person and per movie', zh: '为每个人、每部电影学两个数' },
	defaults: { seed: 1, lr: 0.05, batch: 1, steps: 4000, snapshotEvery: 50, optimizer: 'sgd', init: 0.5, cold: false },
	model: (rng, opt) => new MF(rng, opt.init),
	batch: (raw, _step, rng, opt) => {
		const k = known(raw, opt.cold);
		return pick(k, rng.int(k.n));
	},
	loss: (m, b) => {
		const pred = op.sum(op.mul(op.gather(m.user, 0, b.u), op.gather(m.item, 0, b.i)), 1);
		return { loss: mseLoss(pred, { shape: [b.r.length], data: b.r }) };
	},
	metrics(m, raw, opt) {
		const k = known(raw, opt?.cold === true);
		const f = { U: m.user.value.data, V: m.item.value.data };
		const out: Record<string, number> = { mse: mse(f, k), n: k.n };
		if (k.yours) out.you = mse(f, k, true);
		return out;
	}
};
