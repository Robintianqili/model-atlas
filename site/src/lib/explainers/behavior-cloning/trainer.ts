// behavior-cloning: trainer 'bc' (kind B; process 'train:bc'). Behavior cloning: a policy network MLP('net', [2, 16, 4],
// 'tanh') learns state -> move from the learner's demonstrations by supervised learning: full-batch cross-entropy on
// every (cell, move) pair, Adam. With `rounds` > 0 it is DAgger (Ross, Gordon & Bagnell 2011): after `bcSteps` updates,
// and then every `every` updates, the current policy drives `episodes` windy episodes from the start (the wind drawn from
// the trainer's batch stream), the expert labels every visited cell the dataset does not have yet with its own move, and
// training goes on, from the same weights, on the grown dataset. The DAgger run's first `bcSteps` updates are exactly the
// plain run (the same seed, data and optimizer; claims.test.ts checks it).
// The twin (pipeline/behavior-cloning/twin.py) trains on the batches recorded here, in PyTorch float64.
//   npm run atlas -- run behavior-cloning train --frames 0..20
import type { Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { input, scalar } from '../../atlas/autograd/var.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { distances, EVAL, expertMove, featuresOf, OPEN, pairsOf, playEpisode, policyAt, successRate, toRaw, windOf, worldOf, type Params } from './world.ts';

export type BcOpt = TrainOpt & { hidden: number; rounds: number; bcSteps: number; every: number; episodes: number };
/** The dataset: cells, moves, and where each pair came from (0 = your demonstrations, r = DAgger round r). */
export interface Data { s: number[]; a: number[]; from: number[] }
export const paramsOf = (m: MLP): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.value]));

/** One DAgger round: the policy (weights P) drives `episodes` episodes under a wind from rng; the expert labels each
 *  visited open cell the dataset lacks. Returns the cells added (in visiting order) and the episodes' outcomes. */
export function daggerRound(raw: unknown, P: Params, D: Data, r: number, rng: Rng, o: Pick<BcOpt, 'episodes'>) {
	const g = toRaw(raw), W = worldOf(g.map), d = distances(W), wind = windOf(rng, o.episodes, EVAL.T, g.wind), seen = new Set(D.s), added: number[] = [];
	let goals = 0;
	for (let e = 0; e < o.episodes; e++) {
		const ep = playEpisode(W, (s) => policyAt(P, featuresOf(W, s)).a, wind, e, EVAL.T);
		goals += ep.outcome === 1 ? 1 : 0;
		for (const s of ep.cells) if (W.cell[s] === OPEN && !seen.has(s) && expertMove(W, d, s) >= 0) seen.add(s), added.push(s), D.s.push(s), D.a.push(expertMove(W, d, s)), D.from.push(r);
	}
	return { added, goals };
}
/** The round that starts at update k (1, 2, …), or 0 when no round starts there. */
export const roundAt = (k: number, o: Pick<BcOpt, 'rounds' | 'bcSteps' | 'every'>) => (o.rounds > 0 && k >= o.bcSteps && (k - o.bcSteps) % o.every === 0 && (k - o.bcSteps) / o.every < o.rounds ? (k - o.bcSteps) / o.every + 1 : 0);

/** the DAgger source's options (explainer.ts): 100 cloning updates, then 4 rounds of 25 */
export const DAGGER = { seed: 1, rounds: 4, steps: 200 };

const DATA = new WeakMap<object, Data>();
const T = (shape: number[], d: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(d) });
/** batch(): the whole dataset, x [n, 2] (column and row scaled to [−1, 1]) and y [n] (the moves). */
export type Batch = { x: Tensor; y: Float64Array; round: number; added: number };

export const bc: TrainerDef<unknown, MLP, Batch, BcOpt> = {
	id: 'bc',
	title: { en: 'Behavior cloning: copy the demonstrated moves', zh: '行为克隆：照着示范的动作学' },
	defaults: { seed: 1, lr: 0.05, batch: 0, steps: 100, snapshotEvery: 5, optimizer: 'adam', hidden: 16, rounds: 0, bcSteps: 100, every: 25, episodes: 10 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, 4], 'tanh', rng),
	batch(raw, k, rng, opt, model) {
		const g = toRaw(raw), W = worldOf(g.map);
		let D = model ? DATA.get(model) : undefined;
		if (!D) {
			const p = pairsOf(g);
			D = { s: p.s, a: p.a, from: p.s.map(() => 0) };
			if (model) DATA.set(model, D);
		}
		const r = roundAt(k, opt), added = r && model ? daggerRound(g, paramsOf(model), D, r, rng, opt).added.length : 0;
		return { x: T([D.s.length, 2], D.s.flatMap((s) => featuresOf(W, s))), y: Float64Array.from(D.a), round: r, added };
	},
	loss(m, b) {
		const z = m.logits(input(b.x));
		let right = 0;
		for (let i = 0; i < b.y.length; i++) {
			const row = Array.from(z.data.subarray(i * 4, i * 4 + 4));
			right += row.indexOf(Math.max(...row)) === b.y[i] ? 1 : 0;
		}
		// parts: the dataset size, the share of its pairs the network gets right, the DAgger round that started here
		return { loss: crossEntropy(z, b.y), parts: { n: scalar(b.y.length), acc: scalar(right / b.y.length), round: scalar(b.round), added: scalar(b.added) } };
	},
	// the share of 20 windy episodes (the page's evaluation wind) that reach the goal with the current weights
	metrics(m, raw) {
		const g = toRaw(raw), W = worldOf(g.map), P = paramsOf(m);
		return { success: successRate(W, (s) => policyAt(P, featuresOf(W, s)).a, windOf(createRng(EVAL.seed), EVAL.episodes, EVAL.T, g.wind), EVAL.T) };
	}
};

/** A finished run's dataset, rebuilt from its tape: the snapshots at the round starts hold the weights each DAgger round
 *  drove with, and a fresh batch stream (seed, 'batch') gives the same gusts as the run. Also each round's added cells. */
type TapeLike = { length: number; frame(i: number): { phase: string; metrics?: Record<string, number> }; state(i: number): unknown };
export function datasetOf(raw: unknown, tape: TapeLike | null | undefined, opt: Partial<BcOpt> = {}) {
	const o = { ...bc.defaults, ...opt }, p = pairsOf(toRaw(raw)), D: Data = { s: p.s, a: p.a, from: p.s.map(() => 0) }, rounds: { k: number; added: number[]; goals: number }[] = [];
	const rng = createRng(o.seed).fork('batch');
	for (let r = 1; r <= o.rounds && tape; r++) {
		const k = o.bcSteps + (r - 1) * o.every;
		let i = tape.length - 1;
		while (i >= 0 && !(tape.frame(i).phase === 'snapshot' && tape.frame(i).metrics?.step === k)) i--;
		if (i < 0) break;
		rounds.push({ k, ...daggerRound(raw, (tape.state(i) as { params: Params }).params, D, r, rng, o) });
	}
	return { ...D, rounds };
}
