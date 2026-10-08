// word2vec: trainer 'sg' (kind B; process 'train:sg'). Skip-gram with a full softmax over the learner's vocabulary, as
// in wevi: the center word picks its row h of W_in, the scores are h · (every word's row of W_out), a softmax turns them
// into shares, and the loss is −ln(the share of the true context word). Plain SGD on one random (center, context) pair
// per step. Twin: pipeline/word2vec/twin.py (same names, same slicing).
//   npm run atlas -- run word2vec train --frames 0..20
import type { Module, Rng, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { corpusOf, VMAX, type CorpusRaw } from './corpus.ts';

export type SgOpt = TrainOpt & { dim: number };
export type SgBatch = { c: Int32Array; o: Int32Array; V: Int32Array };

/** W_in ('emb.weight') and W_out ('out.weight'), both [VMAX, dim] and uniform in [−0.5, 0.5) at the start. Row w of
 *  W_in is word w's input vector (its word vector); row w of W_out is its output vector (drawn as a column of W_out).
 *  Only the first V rows (the learner's vocabulary) ever enter a forward pass. */
export class SkipGram implements Module {
	readonly emb: Var;
	readonly out: Var;
	constructor(rng: Rng, dim: number) {
		const u = () => Float64Array.from({ length: VMAX * dim }, () => rng.next() - 0.5);
		this.emb = param('emb.weight', { shape: [VMAX, dim], data: u() });
		this.out = param('out.weight', { shape: [VMAX, dim], data: u() });
	}
	named(): Record<string, Var> {
		return { 'emb.weight': this.emb, 'out.weight': this.out };
	}
	params(): Var[] {
		return [this.emb, this.out];
	}
}

/** Mean −ln p(context | center) over every pair of the corpus, in plain float64 from the two tables. */
export function corpusLoss(win: ArrayLike<number>, wout: ArrayLike<number>, dim: number, raw: CorpusRaw): number {
	const c = corpusOf(raw), V = c.words.length;
	if (!c.pairs.length) return NaN;
	let sum = 0;
	for (const p of c.pairs) {
		const u = Array.from({ length: V }, (_, j) => {
			let s = 0;
			for (let k = 0; k < dim; k++) s += win[p.c * dim + k] * wout[j * dim + k];
			return s;
		});
		const m = Math.max(...u);
		sum += m + Math.log(u.reduce((a, x) => a + Math.exp(x - m), 0)) - u[p.o];
	}
	return sum / c.pairs.length;
}

export const sg: TrainerDef<CorpusRaw, SkipGram, SgBatch, SgOpt> = {
	id: 'sg',
	title: { en: 'Train skip-gram word vectors', zh: '训练 skip-gram 词向量' },
	defaults: { seed: 1, lr: 0.1, batch: 1, steps: 3000, snapshotEvery: 25, optimizer: 'sgd', dim: 5 },
	model: (rng, opt) => new SkipGram(rng, opt.dim),
	// one (center, context) pair per step, drawn at random from all the pairs of the corpus
	batch(raw, _step, rng) {
		const c = corpusOf(raw);
		const k = c.pairs.length ? rng.int(c.pairs.length) : -1;
		return { c: Int32Array.of(...(k < 0 ? [] : [c.pairs[k].c])), o: Int32Array.of(...(k < 0 ? [] : [c.pairs[k].o])), V: Int32Array.of(c.words.length) };
	},
	loss(m, b, rec) {
		// no pair (an empty corpus): nothing to learn, every gradient 0
		if (!b.c.length) return { loss: op.scale(op.sum(m.emb), 0) };
		const h = op.gather(m.emb, 0, b.c); // [1, dim]: the center word's row of W_in
		rec('h', h);
		const u = op.linear(h, op.slice(m.out, 0, 0, b.V[0])); // [1, V]: h · every word's output vector
		rec('scores', u);
		rec('p', op.softmax(u, -1));
		return { loss: crossEntropy(u, b.o) };
	},
	metrics(m, raw, opt) {
		const dim = Number(opt?.dim ?? sg.defaults.dim);
		return { avgLoss: corpusLoss(m.emb.value.data, m.out.value.data, dim, raw), pairs: corpusOf(raw).pairs.length, V: corpusOf(raw).words.length };
	}
};
