// node2vec: trainer 'sg' (kind B; process 'train:sg'). Skip-gram over the walks, as word2vec does over sentences (the
// word2vec package's trainer, copied): the center member picks its row h of W_in, the scores are h · (every member's row
// of W_out), a softmax over the 34 members turns them into shares, and the loss is −ln(the share of the true context
// member). Plain SGD on one random (center, context) pair of the walks per step. Twin: pipeline/node2vec/twin.py.
//   npm run atlas -- run node2vec train --frames 0..20
import type { Module, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { inputOf, N, walksOf, type WalkRaw } from './walks.ts';

/** p and q, when set, replace the input's (a run on the learner's graph with fixed p and q) */
export type SgOpt = TrainOpt & { dim: number; p?: number; q?: number };
export type SgBatch = { c: Int32Array; o: Int32Array };

/** W_in ('emb.weight', row i = member i's vector) and W_out ('out.weight', row i = member i's output vector), both
 *  [34, dim] and uniform in [−0.5, 0.5) at the start. */
export class SkipGram implements Module {
	readonly emb: Var;
	readonly out: Var;
	constructor(rng: Rng, dim: number) {
		const u = () => Float64Array.from({ length: N * dim }, () => rng.next() - 0.5);
		this.emb = param('emb.weight', { shape: [N, dim], data: u() });
		this.out = param('out.weight', { shape: [N, dim], data: u() });
	}
	named(): Record<string, Var> {
		return { 'emb.weight': this.emb, 'out.weight': this.out };
	}
	params(): Var[] {
		return [this.emb, this.out];
	}
}

/** Mean −ln p(context | center) over every pair of the walks, in plain float64 from the two tables (one softmax per
 *  center member, weighted by the pair counts). */
export function walkLoss(win: ArrayLike<number>, wout: ArrayLike<number>, dim: number, raw: WalkRaw): number {
	const { pairs, cooc } = walksOf(raw);
	if (!pairs.length) return NaN;
	const u = new Float64Array(N);
	let sum = 0;
	for (let c = 0; c < N; c++) {
		let m = -Infinity, e = 0, n = 0;
		for (let o = 0; o < N; o++) n += cooc[c * N + o];
		if (!n) continue;
		for (let j = 0; j < N; j++) {
			let s = 0;
			for (let k = 0; k < dim; k++) s += win[c * dim + k] * wout[j * dim + k];
			m = Math.max(m, (u[j] = s));
		}
		for (let j = 0; j < N; j++) e += Math.exp(u[j] - m);
		for (let o = 0; o < N; o++) if (cooc[c * N + o]) sum += cooc[c * N + o] * (m + Math.log(e) - u[o]);
	}
	return sum / pairs.length;
}

/** the input this run walks on: the learner's, with the run's own p and q when it has them */
export const walkInput = (raw: WalkRaw, opt?: Partial<SgOpt>): WalkRaw => (opt?.p || opt?.q ? { ...inputOf(raw), ...(opt.p ? { p: opt.p } : {}), ...(opt.q ? { q: opt.q } : {}) } : raw);

export const sg: TrainerDef<WalkRaw, SkipGram, SgBatch, SgOpt> = {
	id: 'sg',
	title: { en: 'Train node vectors with skip-gram', zh: '用 skip-gram 训练节点向量' },
	defaults: { seed: 1, lr: 0.1, batch: 1, steps: 6000, snapshotEvery: 50, optimizer: 'sgd', dim: 8 },
	model: (rng, opt) => new SkipGram(rng, opt.dim),
	// one (center, context) pair per step, drawn at random from all the pairs of the walks
	batch(raw, _step, rng, opt) {
		const { pairs } = walksOf(walkInput(raw, opt));
		const k = pairs.length ? rng.int(pairs.length) : -1;
		return { c: Int32Array.of(...(k < 0 ? [] : [pairs[k].c])), o: Int32Array.of(...(k < 0 ? [] : [pairs[k].o])) };
	},
	loss(m, b, rec) {
		// no pair (a graph without friendships): nothing to learn, every gradient 0
		if (!b.c.length) return { loss: op.scale(op.sum(m.emb), 0) };
		const h = op.gather(m.emb, 0, b.c); // [1, dim]: the center member's row of W_in
		rec('h', h);
		const u = op.linear(h, m.out); // [1, 34]: h · every member's output vector
		rec('scores', u);
		rec('p', op.softmax(u, -1));
		return { loss: crossEntropy(u, b.o) };
	},
	metrics(m, raw, opt) {
		const dim = Number(opt?.dim ?? sg.defaults.dim);
		const r = walkInput(raw, opt as Partial<SgOpt>);
		return { avgLoss: walkLoss(m.emb.value.data, m.out.value.data, dim, r), pairs: walksOf(r).pairs.length };
	}
};
