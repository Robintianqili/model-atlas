// cart-pole: trainer 'reinforce' (kind B; process 'train:reinforce'). REINFORCE, the policy-gradient idea of the
// TensorFlow.js cart-pole example (Apache-2.0), reimplemented on the Model Atlas autograd core. One training step = one
// round: play `batch` episodes with the current policy (the first from the learner's start state, the others from random
// starts near upright), give every step its discounted return G and its normalized advantage A = (G − mean) / std over
// the round, then minimize mean(A · −log π(action | state)) (Adam, learning rate 0.02): a step with A > 0 pulls its action
// up, A < 0 down.
// The episodes are played inside loss() (it is the only hook that sees the current weights) from the rng stream batch()
// hands over; the round's tensors are written into the batch object, so fine frames record them and the twin
// (pipeline/cart-pole/twin.py) trains on exactly these numbers.
//   npm run atlas -- run cart-pole train --frames 0..20
import type { Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { input, scalar } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { advantages, randomStart, returns, rollout, startOf, type Params, type State } from './sim.ts';

export type ReinOpt = TrainOpt & { hidden: number; gamma: number; maxLen: number };
/** batch() fills rng and start; loss() adds the round: states x [N, 4], left [N] (1 = pushed left), p(left) at play time,
 *  return G, advantage A, episode number ep and time t of every step, and each episode's length lens [batch]. */
export type Round = { rng: Rng; start: State; x?: Tensor; left?: Float64Array; p?: Float64Array; ret?: Float64Array; adv?: Float64Array; ep?: Float64Array; t?: Float64Array; lens?: Float64Array };

const T = (shape: number[], d: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(d) });
export const paramsOf = (m: MLP): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.value]));

/** Play one round with fixed weights: the same numbers loss() trains on (the lesson and the claims replay it). */
export function playRound(params: Params, start: State, rng: Rng, opt: Pick<ReinOpt, 'batch' | 'gamma' | 'maxLen'>): Required<Omit<Round, 'rng' | 'start'>> {
	const xs: number[] = [], left: number[] = [], p: number[] = [], G: number[] = [], ep: number[] = [], t: number[] = [], lens: number[] = [];
	for (let e = 0; e < Math.max(1, opt.batch); e++) {
		const r = rollout(params, e === 0 ? start : randomStart(rng), rng, opt.maxLen);
		r.states.forEach((s, i) => (xs.push(...s), left.push(r.left[i]), p.push(r.p[i]), ep.push(e), t.push(i)));
		G.push(...returns(r.len, opt.gamma));
		lens.push(r.len);
	}
	return { x: T([left.length, 4], xs), left: T([left.length], left).data as Float64Array, p: Float64Array.from(p), ret: Float64Array.from(G), adv: Float64Array.from(advantages(G)), ep: Float64Array.from(ep), t: Float64Array.from(t), lens: Float64Array.from(lens) };
}

export const reinforce: TrainerDef<unknown, MLP, Round, ReinOpt> = {
	id: 'reinforce',
	title: { en: 'REINFORCE on the cart-pole', zh: '用 REINFORCE 训练倒立摆' },
	defaults: { seed: 1, lr: 0.02, batch: 10, steps: 100, snapshotEvery: 2, optimizer: 'adam', hidden: 16, gamma: 0.95, maxLen: 200 },
	model: (rng, opt) => new MLP('net', [4, opt.hidden, 1], 'tanh', rng, { outAct: 'sigmoid' }),
	batch: (raw, _k, rng) => ({ rng, start: startOf(raw) }),
	loss(m, b, rec, opt) {
		Object.assign(b, playRound(paramsOf(m), b.start, b.rng, opt));
		const z = m.logits(input(b.x!), rec);
		const y = input({ shape: [b.left!.length, 1], data: b.left! });
		// −log π(action | state) from the score z: −log σ(z) for left (y = 1), −log(1 − σ(z)) for right; written stably
		const nll = op.add(op.sub(op.relu(z), op.mul(z, y)), op.log(op.add(op.exp(op.neg(op.abs(z))), 1)));
		const A = input({ shape: [b.adv!.length, 1], data: b.adv! });
		const lens = Array.from(b.lens!);
		return { loss: op.mean(op.mul(nll, A)), parts: { meanLen: scalar(lens.reduce((s, l) => s + l, 0) / lens.length), len0: scalar(lens[0]) } };
	}
};
