// cart-pole: process 'episode' (kind C): the first episode of training round 0, one event at a time. It starts from the
// learner's start state with the fresh policy (the trainer's own weights: model(rng.fork('init'))) and draws from the
// trainer's batch stream (rng.fork('batch')), so it is exactly episode 0 of the first round the `train` and `step1`
// sources play (claims.test.ts checks it). Frames: start -> (act -> move) per step -> end -> return per step, backwards.
//   act     the policy gives p(left); u = rng.next(); push left when u < p(left)
//   move    one physics step (sim.ts): the push, the accelerations, the new state after 0.02 s
//   end     why the episode ended and its length T (one reward of +1 per step)
//   return  G_t = r_t + γ·G_(t+1), from the last step back to the first (field G, edited in place: changes)
// pipeline/cart-pole/reference.py is an independent Python run of the same frames (atlas parity).
// seeds: the same training (trainer reinforce) with seeds 1 to 5 in lockstep for 50 rounds, one frame per round with every
// run's mean episode length (len1 ... len5); run 1 is the first 50 rounds of the `train` source (claims.test.ts). 50 rounds:
// training is chaotic (rounding differences grow about 1.3 times per round), and the Python reference repeats all five runs
// exactly only for so long: at 100 rounds the first run diverged at round 75 (notes/cart-pole.md).
//   npm run atlas -- run cart-pole episode --frames 0..12
import type { Frame, ProcessDef, ProcessRun, Tensor, TrainState } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { physics, policy, startOf, TAU, type State } from './sim.ts';
import { paramsOf, reinforce, type ReinOpt } from './trainer.ts';

export type EpOpt = { hidden: number; gamma: number; maxLen: number };
export type EpState = { s: Tensor; G: Tensor };
const I = (en: string, zh: string) => ({ en, zh });
const vec = (s: State): Tensor => ({ shape: [4], data: Float64Array.from(s) });
const named = (s: State) => ({ x: s[0], xd: s[1], th: s[2], thd: s[3] });

export const episode: ProcessDef<unknown, EpOpt, EpState> = {
	id: 'episode',
	kind: 'algorithm',
	title: I('One episode of the fresh policy, then its returns', '新策略的一个回合，以及它的回报'),
	phases: {
		start: { label: I('The start state', '起始状态'), kind: 'input' },
		act: { label: I('Sample an action', '采样一个动作'), kind: 'sample', narration: { what: I('The policy says p(left) = {frame.p}; the draw u = {frame.u} decides the push.', '策略给出 p(左) = {frame.p}；随机数 u = {frame.u} 决定往哪边推。') } },
		move: {
			label: I('One physics step', '物理模拟一步'),
			kind: 'other',
			formula: { tex: "\\theta' = \\theta + \\tau\\,\\dot\\theta = {{th}} + {{tau}} \\times {{thd}} = {{th2}};\\quad \\dot\\theta' = \\dot\\theta + \\tau\\,\\ddot\\theta = {{thd}} + {{tau}} \\times {{thAcc}} = {{thd2}}", values: (f) => ({ ...f.metrics!, tau: TAU }) },
			narration: { what: I('A push of {frame.force} N: the pole’s angle goes from {frame.th} to {frame.th2} rad.', '推力 {frame.force} N：杆的角度从 {frame.th} 变成 {frame.th2} rad。') }
		},
		end: { label: I('The episode ends', '回合结束'), kind: 'observe', narration: { what: I('The episode lasted {frame.len|int} steps.', '这个回合持续了 {frame.len|int} 步。') } },
		return: {
			label: I('Discounted return', '折扣回报'),
			kind: 'aggregate',
			formula: { tex: 'G_{ {{t}} } = r + \\gamma\\,G_{ {{t1}} } = {{r}} + {{gamma}} \\times {{next}} = {{G}}', values: (f) => f.metrics! },
			narration: { what: I('Step {frame.t|int}: G = 1 + {frame.gamma} × {frame.next} = {frame.G}.', '第 {frame.t|int} 步：G = 1 + {frame.gamma} × {frame.next} = {frame.G}。') }
		}
	},
	schema: {
		s: { kind: 'tensor', label: I('State [x, ẋ, θ, θ̇]', '状态 [x, ẋ, θ, θ̇]'), shape: [4], tol: 1e-12 },
		G: { kind: 'tensor', label: I('Discounted return of each step', '每一步的折扣回报'), track: 'changes', tol: 1e-12 }
	},
	defaults: { hidden: reinforce.defaults.hidden, gamma: reinforce.defaults.gamma, maxLen: reinforce.defaults.maxLen },
	*run(input, ctx): ProcessRun<EpState, EpOpt> {
		const o = ctx.opt;
		const params = paramsOf(reinforce.model(ctx.rng.fork('init'), { ...reinforce.defaults, hidden: o.hidden }));
		const rng = ctx.rng.fork('batch');
		let s = startOf(input);
		let t = 0;
		let end = 3;
		yield { phase: 'start', state: { s: vec(s), G: { shape: [0], data: new Float64Array(0) } }, metrics: { ...named(s), maxLen: o.maxLen, gamma: o.gamma } };
		while (t < o.maxLen && !ctx.signal.cancelled) {
			const pol = policy(params, s), u = rng.next(), left = u < pol.p ? 1 : 0;
			yield { phase: 'act', step: t, state: {}, metrics: { t, z: pol.z, p: pol.p, u, left, ...named(s) } };
			const f = physics(s, left);
			yield { phase: 'move', step: t, state: { s: vec(f.s2) }, metrics: { t, left, force: f.force, temp: f.temp, thAcc: f.thAcc, xAcc: f.xAcc, ...named(s), x2: f.s2[0], xd2: f.s2[1], th2: f.s2[2], thd2: f.s2[3], done: f.done } };
			s = f.s2;
			t++;
			if (f.done) {
				end = f.done;
				break;
			}
		}
		const G: Tensor = { shape: [t], data: new Float64Array(t) };
		yield { phase: 'end', step: t, state: { G }, metrics: { len: t, end, ...named(s) } };
		let next = 0;
		for (let k = t - 1; k >= 0 && !ctx.signal.cancelled; k--) {
			const g = 1 + o.gamma * next;
			G.data[k] = g;
			yield { phase: 'return', step: k, state: { G }, changes: [{ field: 'G', index: [k], from: 0, to: g }], metrics: { t: k, t1: k + 1, r: 1, gamma: o.gamma, next, G: g, len: t } };
			next = g;
		}
	}
};

const inner = trainingProcess(reinforce);
export type SeedsOpt = ReinOpt & { seeds: number[] };
export const seeds: ProcessDef<unknown, SeedsOpt, TrainState> = {
	...inner,
	id: 'seeds',
	title: I('The same training with five seeds', '同样的训练，五个随机种子'),
	defaults: { ...reinforce.defaults, steps: 50, seeds: [1, 2, 3, 4, 5] },
	*run(input, ctx): ProcessRun<TrainState, SeedsOpt> {
		const runs = ctx.opt.seeds.map((s) => inner.run(input, { ...ctx, rng: createRng(s), detail: 'coarse' }));
		for (;;) {
			const fs = runs.map((g) => g.next());
			if (fs.some((r) => r.done) || ctx.signal.cancelled) return;
			const f = fs[0].value as Frame<TrainState>;
			const lens = Object.fromEntries(fs.map((r, i) => [`len${i + 1}`, (r.value as Frame<TrainState>).metrics?.meanLen ?? NaN]));
			yield { phase: f.phase, step: f.step, state: { step: f.state.step ?? 0 }, metrics: { step: f.metrics?.step ?? 0, ...(f.phase === 'init' ? {} : lens) } };
		}
	}
};
