// ppo-reacher: trainer 'ppo' (kind B; process 'train:ppo'). PPO with GAE (Schulman et al. 2017; the setup follows
// CleanRL's ppo_continuous_action, reimplemented on the Model Atlas autograd core). One training step = one minibatch
// update. Every `epochs × minibatches` steps an iteration starts: batch() plays `episodes` episodes of T steps with the
// current policy (the first to the learner's target, the others to random targets), gives every step its GAE advantage
// (normalized over the batch) and its value target R = A + V, and fixes the order of the minibatches (a shuffle per
// epoch). The rollout is kept per model (a WeakMap) for the iteration's updates. loss() is
//   −mean(min(r·A, clip(r, 1 − ε, 1 + ε)·A)) + vcoef · mean((V(s) − R)²),   r = π(a | s) / π_old(a | s)
// with the masks of clip and min taken from the forward values (so the twin, pipeline/ppo-reacher/twin.py, has the same
// gradients). ε comes from the learner's input (the task changes it). loss() writes the ratio and the clipped flags into
// the batch, so fine frames record them.
//   npm run atlas -- run ppo-reacher train --frames 0..20
import type { Module, Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { input, param, scalar, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { epsOf, gae, meanStd, randomTarget, rollout, targetOf, type Params, type Pt } from './sim.ts';

export type PpoOpt = TrainOpt & { hidden: number; gamma: number; lambda: number; epochs: number; minibatches: number; episodes: number; T: number; logStd0: number; vcoef: number };
const LOG2PI = Math.log(2 * Math.PI);

/** The actor (network 'pi': the means of the two torques, and 'logstd': two learned numbers) and the critic ('v'). */
export class ActorCritic implements Module {
	readonly pi: MLP;
	readonly v: MLP;
	readonly logstd: Var;
	constructor(rng: Rng, o: Pick<PpoOpt, 'hidden' | 'logStd0'>) {
		this.pi = new MLP('pi', [6, o.hidden, 2], 'tanh', rng);
		this.v = new MLP('v', [6, o.hidden, 1], 'tanh', rng);
		this.logstd = param('logstd', { shape: [2], data: Float64Array.from([o.logStd0, o.logStd0]) });
	}
	named = (): Record<string, Var> => ({ ...this.pi.named(), logstd: this.logstd, ...this.v.named() });
	params = (): Var[] => Object.values(this.named());
}
export const paramsOf = (m: Module): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.value]));

/** One iteration's batch: every step of every episode, and the shuffled order of each epoch. */
export interface Rollout { obs: number[][]; act: Pt[]; logp: number[]; v: number[]; adv: number[]; advRaw: number[]; ret: number[]; targets: Pt[]; epRet: number[]; order: number[][] }
/** Play an iteration's episodes with fixed weights (episode 0 to the learner's target) and work out GAE. */
export function collect(p: Params, target: Pt, rng: Rng, o: Pick<PpoOpt, 'episodes' | 'T' | 'gamma' | 'lambda' | 'epochs'>): Rollout {
	const R: Rollout = { obs: [], act: [], logp: [], v: [], adv: [], advRaw: [], ret: [], targets: [], epRet: [], order: [] };
	for (let e = 0; e < o.episodes; e++) {
		const g = e === 0 ? target : randomTarget(rng), ep = rollout(p, g, rng, o.T), G = gae(ep.r, ep.v, ep.vLast, o.gamma, o.lambda);
		R.obs.push(...ep.obs), R.act.push(...ep.act), R.logp.push(...ep.logp), R.v.push(...ep.v), R.advRaw.push(...G.A), R.ret.push(...G.R);
		R.targets.push(g), R.epRet.push(ep.r.reduce((s, x) => s + x, 0));
	}
	const { mean, std } = meanStd(R.advRaw);
	R.adv = R.advRaw.map((a) => (std > 0 ? (a - mean) / std : 0));
	const N = R.adv.length;
	for (let k = 0; k < o.epochs; k++) {
		const idx = Array.from({ length: N }, (_, i) => i);
		for (let i = N - 1, j = 0; i > 0; i--) (j = rng.int(i + 1)), ([idx[i], idx[j]] = [idx[j], idx[i]]);
		R.order.push(idx);
	}
	return R;
}

const BUF = new WeakMap<object, Rollout>();
const T = (shape: number[], d: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(d) });
/** batch(): the minibatch of one update; obs [n, 6], act [n, 2], logpOld, adv, ret [n], eps [1]; idx = the rows of the
 *  rollout; j = the update within the iteration. loss() adds ratio and clipped [n]. */
export type Mini = { obs: Tensor; act: Tensor; logpOld: Float64Array; adv: Float64Array; ret: Float64Array; eps: number[]; idx: number[]; j: number[]; ro: Rollout; ratio?: Float64Array; clipped?: Float64Array; allAdv?: Float64Array; advMean?: number[]; advSd?: number[]; epRet?: Float64Array; targets?: Tensor };

export function minibatch(ro: Rollout, j: number, eps: number, o: Pick<PpoOpt, 'minibatches'>): Mini {
	const N = ro.adv.length, n = Math.floor(N / o.minibatches), ep = Math.floor(j / o.minibatches), m = j % o.minibatches;
	const idx = ro.order[ep].slice(m * n, (m + 1) * n);
	const pick = (v: number[]) => Float64Array.from(idx, (i) => v[i]);
	return {
		obs: T([n, 6], idx.flatMap((i) => ro.obs[i])), act: T([n, 2], idx.flatMap((i) => ro.act[i])), logpOld: pick(ro.logp), adv: pick(ro.adv), ret: pick(ro.ret),
		eps: [eps], idx, j: [j], ro,
		// the whole batch, for the lesson's batch step: on the iteration's first update only
		...(j === 0 ? { allAdv: Float64Array.from(ro.adv), advMean: [meanStd(ro.advRaw).mean], advSd: [meanStd(ro.advRaw).std], epRet: Float64Array.from(ro.epRet), targets: T([ro.targets.length, 2], ro.targets.flat()) } : {})
	};
}

/** The PPO loss on a minibatch; also the numbers the lesson reads (ratio, clipped, the parts). */
export function ppoLoss(m: ActorCritic, b: Mini, opt: Pick<PpoOpt, 'vcoef'>) {
	const n = b.adv.length, eps = b.eps[0], x = input(b.obs);
	const mu = m.pi.forward(x);
	const z = op.div(op.sub(input(b.act), mu), op.exp(m.logstd));
	const logp = op.sum(op.sub(op.sub(op.scale(op.mul(z, z), -0.5), m.logstd), 0.5 * LOG2PI), 1);
	const ratio = op.exp(op.sub(logp, input(T([n], b.logpOld))));
	const r = ratio.data, A = b.adv, lo = 1 - eps, hi = 1 + eps;
	const inside = new Float64Array(n), fixed = new Float64Array(n), pick = new Float64Array(n), clipped = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		inside[i] = r[i] >= lo && r[i] <= hi ? 1 : 0;
		fixed[i] = r[i] < lo ? lo : r[i] > hi ? hi : 0;
		const c = inside[i] ? r[i] : fixed[i];
		pick[i] = r[i] * A[i] <= c * A[i] ? 1 : 0;
		clipped[i] = !pick[i] && !inside[i] ? 1 : 0;
	}
	const Av = input(T([n], A));
	const surr1 = op.mul(ratio, Av), surr2 = op.mul(op.add(op.mul(ratio, input(T([n], inside))), input(T([n], fixed))), Av);
	const obj = op.add(op.mul(surr1, input(T([n], pick))), op.mul(surr2, input(T([n], Array.from(pick, (p) => 1 - p)))));
	const pg = op.neg(op.mean(obj));
	const V = op.reshape(m.v.forward(x), [n]);
	const vf = op.mean(op.square(op.sub(V, input(T([n], b.ret)))));
	// how far the policy has moved from the one that played the batch: KL(π_old ‖ π) estimated as mean(r − 1 − log r)
	let kl = 0;
	for (let i = 0; i < n; i++) kl += r[i] - 1 - Math.log(r[i]);
	return { loss: op.add(pg, op.scale(vf, opt.vcoef)), pg, vf, ratio: Float64Array.from(r), clipped, kl: kl / n };
}

export const ppo: TrainerDef<unknown, ActorCritic, Mini, PpoOpt> = {
	id: 'ppo',
	title: { en: 'PPO on the reacher', zh: '用 PPO 训练机械臂' },
	defaults: { seed: 1, lr: 0.005, batch: 160, steps: 1600, snapshotEvery: 16, optimizer: 'adam', hidden: 32, gamma: 0.95, lambda: 0.95, epochs: 4, minibatches: 4, episodes: 16, T: 40, logStd0: -0.5, vcoef: 0.5 },
	model: (rng, opt) => new ActorCritic(rng, opt),
	batch(raw, k, rng, opt, model) {
		const j = k % (opt.epochs * opt.minibatches);
		let ro = model ? BUF.get(model) : undefined;
		if (!ro || j === 0) {
			ro = collect(paramsOf(model!), targetOf(raw), rng, opt);
			if (model) BUF.set(model, ro);
		}
		return minibatch(ro, j, epsOf(raw), opt);
	},
	loss(m, b, _rec, opt) {
		const L = ppoLoss(m, b, opt), S = scalar;
		Object.assign(b, { ratio: L.ratio, clipped: L.clipped });
		// parts: the policy and critic losses, the KL, the batch's average episode return and σ (geometric mean of the two)
		return { loss: L.loss, parts: { pg: L.pg, vf: L.vf, kl: S(L.kl), epRet: S(meanStd(b.ro.epRet).mean), sigma: S(Math.exp(meanStd(m.logstd.value.data).mean)) } };
	}
};
