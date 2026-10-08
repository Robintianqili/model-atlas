// gan-2d: trainer 'gan' (kind B; process 'train:gan'). A generator G (an MLP, 2 noise numbers -> a point) and a
// discriminator D (an MLP, a point -> a score, the "logit"; D(x) = sigmoid(score) is its probability that x is real)
// trained against each other on the learner's points. One training step updates ONE of them:
//   D step (turn 0): loss_D = −mean log D(real) − mean log(1 − D(G(z)))   (G's points are held fixed)
//   G step (turn 1): loss_G = −mean log D(G(z))                         (the "non-saturating" generator loss)
// Steps alternate: dSteps D steps, then one G step (default 1 : 1). Each net has its own optimizer (Adam by default,
// betas 0.5 / 0.999 as in DCGAN); the optimizer object below steps the one whose turn it is. Every step records both
// losses (the other one under no-grad) and the mean scores, so the curves have a value at every step.
// Twin: pipeline/gan-2d/twin.py (npm run atlas -- parity gan-2d: SGD <= 1e-10 and Adam <= 1e-7 relative).
//   npm run atlas -- run gan-2d train --frames 0..20
import type { Module, Optimizer, Rng, Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP, namedParams } from '../../atlas/autograd/nn.ts';
import { bceWithLogits } from '../../atlas/autograd/loss.ts';
import { input, noGrad, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { makeOptimizer, type OptimizerKind } from '../../atlas/autograd/optim.ts';
import type { PointsRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { realOf } from './data.ts';
import { gen as genOf, match, mean, modesHit, scores, turnOf, Z_SHOW, type Params } from './net.ts';

export type GanOpt = TrainOpt & { hidden: number; dSteps: number; lrG?: number; data?: PointsRaw | null; centres?: [number, number][] };
export type GanBatch = { x: Tensor; z: Tensor; turn: number[] };

export class Gan implements Module {
	readonly gen: MLP;
	readonly disc: MLP;
	/** whose step it is (set by loss(), read by the optimizer) */
	turn = 0;
	constructor(rng: Rng, hidden: number) {
		this.gen = new MLP('gen', [2, hidden, hidden, 2], 'tanh', rng);
		this.disc = new MLP('disc', [2, hidden, hidden, 1], 'tanh', rng);
	}
	named(): Record<string, Var> {
		return namedParams(this.gen, this.disc);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}
type GanDef = TrainerDef<unknown, Gan, GanBatch, GanOpt>;

/** Two optimizers, one per net; step() updates the net whose turn it is. */
function alternating(m: Gan, opt: GanOpt): Optimizer & { kind: OptimizerKind } {
	const kind: OptimizerKind = opt.optimizer;
	const o = { betas: Array.isArray(opt.betas) ? (opt.betas as [number, number]) : undefined, eps: typeof opt.eps === 'number' ? opt.eps : undefined };
	const d = makeOptimizer(kind, m.disc.named(), { ...o, lr: opt.lr });
	const g = makeOptimizer(kind, m.gen.named(), { ...o, lr: opt.lrG ?? opt.lr });
	return {
		kind,
		get lr() {
			return d.lr;
		},
		step: () => (m.turn === 1 ? g : d).step(),
		set(s) {
			d.set(s);
			g.set(s);
		},
		state: () => ({ ...d.state(), ...g.state() })
	};
}

const ones = (n: number) => input({ shape: [n, 1], data: new Float64Array(n).fill(1) });
const zeros = (n: number) => input({ shape: [n, 1], data: new Float64Array(n) });
const meanSig = (l: Var) => noGrad(() => op.mean(op.sigmoid(l)));

export const gan: GanDef = {
	id: 'gan',
	title: { en: 'Train the generator and the discriminator', zh: '训练生成器和判别器' },
	defaults: { seed: 1, lr: 0.01, lrG: 0.002, batch: 64, steps: 3000, snapshotEvery: 20, optimizer: 'adam', betas: [0.5, 0.999], hidden: 16, dSteps: 1 },
	model: (rng, opt) => new Gan(rng, opt.hidden),
	optimizer: alternating,
	batch(raw, step, rng, opt) {
		const p = realOf(raw, opt);
		const B = opt.batch;
		const x = new Float64Array(p.n ? 2 * B : 0);
		for (let i = 0; p.n && i < B; i++) {
			const k = rng.int(p.n);
			x[2 * i] = p.xy[2 * k];
			x[2 * i + 1] = p.xy[2 * k + 1];
		}
		const z = Float64Array.from({ length: 2 * B }, () => 2 * rng.next() - 1);
		return { x: { shape: [x.length / 2, 2], data: x }, z: { shape: [B, 2], data: z }, turn: [turnOf(step, opt.dSteps)] };
	},
	loss(m, b, rec): { loss: Var; parts: Record<string, Var> } {
		const turn = (m.turn = b.turn[0]);
		const [B, n] = [b.z.shape[0], b.x.shape[0]];
		if (n === 0) {
			// no real points: nothing to learn (every gradient is 0)
			const zero = op.scale(op.sum(turn ? m.gen.layers[0].weight : m.disc.layers[0].weight), 0);
			return { loss: zero, parts: { dLoss: zero, gLoss: zero } };
		}
		// D's turn: G's points are held fixed (no gradient reaches G); G's turn: D only scores (the gradient flows through D
		// into G; D's own gradient is not used, its optimizer does not step)
		const mine = <T>(f: () => T, who: number) => (turn === who ? f() : noGrad(f));
		const fake = mine(() => m.gen.forward(input(b.z)), 1);
		const lf = m.disc.forward(fake);
		const lr = mine(() => m.disc.forward(input(b.x)), 0);
		rec('fake', fake);
		const gLoss = mine(() => bceWithLogits(lf, ones(B)), 1);
		const dLoss = mine(() => op.add(bceWithLogits(lr, ones(n)), bceWithLogits(lf, zeros(B))), 0);
		return { loss: turn ? gLoss : dLoss, parts: { dLoss, gLoss, dReal: meanSig(lr), dFake: meanSig(lf) } };
	},
	metrics(m, raw, opt): Record<string, number> {
		const p = realOf(raw, opt);
		const params: Params = {};
		for (const [k, v] of Object.entries(m.named())) params[k] = v.value;
		const fake = genOf(params, Z_SHOW);
		const mt = match(p, fake);
		const out: Record<string, number> = { fakeScore: mean(scores(params, fake)) };
		if (p.n) Object.assign(out, { realScore: mean(scores(params, p.xy)), recall: mt.recall, precision: mt.precision });
		if (opt?.centres) out.modes = modesHit(opt.centres, fake);
		return out;
	}
};
