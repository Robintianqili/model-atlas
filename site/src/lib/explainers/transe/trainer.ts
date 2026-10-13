// transe: trainer 'transe' (kind B; process 'train:transe'). TransE (Bordes et al. 2013): every entity is a point of
// `dim` numbers (a row of ent.weight), every relation an arrow of `dim` numbers (a row of rel.weight), and a fact
// (h, r, t) is plausible when h + r lands near t: its distance d = ‖h + r − t‖ (L2) is small. One step takes one fact
// at random and a corrupted copy (its head or its tail swapped for a random other entity), and plain SGD lowers the
// margin ranking loss max(0, γ + d(fact) − d(corrupted)). Unlike the paper, no step rescales the entities to length 1:
// the hinge stops pushing once a corrupted copy is γ farther away than its fact. Twin: pipeline/transe/twin.py.
//   npm run atlas -- run transe run --frames 0..20
import type { Module, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { factsOf, NE, NR, START_FACTS, type KgRaw } from './kg.ts';
import { evalAll } from './space.ts';

/** `start`: train on the start facts whatever the learner's are (source 'base', the task's comparison) */
export type TransEOpt = TrainOpt & { dim: number; gamma: number; start?: boolean };
export type TransEBatch = { h: Int32Array; r: Int32Array; t: Int32Array; h2: Int32Array; t2: Int32Array };

/** ent.weight [34, dim] and rel.weight [3, dim], uniform in [−0.5, 0.5) at the start */
export class TransE implements Module {
	readonly ent: Var;
	readonly rel: Var;
	constructor(rng: Rng, dim: number) {
		const u = (n: number) => Float64Array.from({ length: n * dim }, () => rng.next() - 0.5);
		this.ent = param('ent.weight', { shape: [NE, dim], data: u(NE) });
		this.rel = param('rel.weight', { shape: [NR, dim], data: u(NR) });
	}
	named(): Record<string, Var> {
		return { 'ent.weight': this.ent, 'rel.weight': this.rel };
	}
	params(): Var[] {
		return [this.ent, this.rel];
	}
}

/** the facts a run trains on */
export const runFacts = (raw: KgRaw | unknown, opt?: Partial<TransEOpt>) => (opt?.start ? START_FACTS : factsOf(raw));
const I = (...x: number[]) => Int32Array.from(x);

export const transe: TrainerDef<KgRaw, TransE, TransEBatch, TransEOpt> = {
	id: 'transe',
	title: { en: 'Train TransE on the facts', zh: '在这些事实上训练 TransE' },
	defaults: { seed: 1, lr: 0.05, batch: 1, steps: 4000, snapshotEvery: 50, optimizer: 'sgd', dim: 8, gamma: 1 },
	model: (rng, opt) => new TransE(rng, opt.dim),
	// one fact at random; a coin decides whether its head or its tail is swapped for one of the 32 entities that are
	// neither its head nor its tail (so a copy never joins an entity to itself)
	batch(raw, _step, rng, opt) {
		const fs = runFacts(raw, opt);
		if (!fs.length) return { h: I(), r: I(), t: I(), h2: I(), t2: I() };
		const [h, r, t] = fs[rng.int(fs.length)], tail = rng.next() < 0.5;
		let x = rng.int(NE - 2);
		if (x >= Math.min(h, t)) x++;
		if (x >= Math.max(h, t)) x++;
		return { h: I(h), r: I(r), t: I(t), h2: I(tail ? h : x), t2: I(tail ? x : t) };
	},
	loss(m, b, rec, opt) {
		// no fact (an empty graph): nothing to learn, every gradient 0
		if (!b.h.length) return { loss: op.scale(op.sum(m.ent), 0) };
		const r = op.gather(m.rel, 0, b.r);
		const d = (h: Int32Array, t: Int32Array) => op.sqrt(op.sum(op.square(op.sub(op.add(op.gather(m.ent, 0, h), r), op.gather(m.ent, 0, t)))));
		const dPos = d(b.h, b.t), dNeg = d(b.h2, b.t2);
		rec('dPos', dPos);
		rec('dNeg', dNeg);
		return { loss: op.relu(op.add(op.sub(dPos, dNeg), opt.gamma)), parts: { dPos, dNeg } };
	},
	// avgLoss: the loss of a step averaged over every fact and every corruption of it; hits1: the share of facts whose
	// tail is the closest entity to h + r (other true tails of the same question left out); facts: how many
	metrics(m, raw, opt) {
		const fs = runFacts(raw, opt as Partial<TransEOpt>), dim = m.ent.shape[1];
		return { ...evalAll(m.ent.value.data, m.rel.value.data, dim, fs, Number(opt?.gamma ?? transe.defaults.gamma)), facts: fs.length };
	}
};
