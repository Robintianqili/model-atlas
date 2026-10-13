// ppo-reacher: process 'episode' (kind C): the first episode of PPO iteration 0, one event at a time, then its GAE. It
// plays the fresh policy (the trainer's own weights: model(rng.fork('init'))) to the learner's target, drawing from the
// trainer's batch stream (rng.fork('batch')), so it is exactly episode 0 of the first batch the `iter` and `train` sources
// play (claims.test.ts checks it). Frames: start -> (act -> move) per step -> end -> gae per step, backwards.
//   act   the policy's means μ and σ = e^logstd, the draws z ~ N(0, 1), the torques a = μ + σ·z, log π(a | s), V(s)
//   move  one step of the arm (sim.ts): the new angles and speeds, the fingertip, the distance, the reward
//   end   the episode's return (the sum of its rewards) and V of the last state
//   gae   δ_t = r_t + γ·V(s_(t+1)) − V(s_t), A_t = δ_t + γλ·A_(t+1), R_t = A_t + V(s_t), from the last step back
// The draws are recorded (rng.normal is not bit-exact across engines): pipeline/ppo-reacher/reference.py replays them.
//   npm run atlas -- run ppo-reacher episode --frames 0..12
import type { Frame, ProcessDef, ProcessRun, Tensor, TrainState } from '../../atlas/types/index.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { DEFAULT_EPS, epsOf, gae, logProb, obs, policy, START, step, targetOf, tip, value, type ArmState, type Pt } from './sim.ts';
import { paramsOf, ppo, type PpoOpt } from './trainer.ts';

export type EpOpt = { T: number; gamma: number; lambda: number; hidden: number; logStd0: number };
export type EpState = { s: Tensor; A: Tensor };
const I = (en: string, zh: string) => ({ en, zh });
const vec = (s: ArmState): Tensor => ({ shape: [4], data: Float64Array.from(s) });
const named = (s: ArmState) => ({ q1: s[0], q2: s[1], w1: s[2], w2: s[3] });

export const episode: ProcessDef<unknown, EpOpt, EpState> = {
	id: 'episode',
	kind: 'algorithm',
	title: I('One episode of the fresh policy, then its advantages', '新策略的一个回合，以及它的优势'),
	recordDraws: true,
	phases: {
		start: { label: I('The start', '起点'), kind: 'input' },
		act: { label: I('Sample the torques', '采样力矩'), kind: 'sample', narration: { what: I('The policy says μ = ({frame.mu1}, {frame.mu2}); the draws z give the torques a = ({frame.a1}, {frame.a2}).', '策略给出 μ = ({frame.mu1}, {frame.mu2})；抽到的 z 给出力矩 a = ({frame.a1}, {frame.a2})。') } },
		move: { label: I('One step of the arm', '机械臂走一步'), kind: 'other', narration: { what: I('The fingertip is now {frame.d} m from the target: reward {frame.r}.', '指尖现在离目标 {frame.d} m：奖励 {frame.r}。') } },
		end: { label: I('The episode ends', '回合结束'), kind: 'observe', narration: { what: I('The episode’s rewards add up to {frame.ret}.', '这个回合的奖励加起来是 {frame.ret}。') } },
		gae: {
			label: I('Advantage (GAE)', '优势（GAE）'),
			kind: 'aggregate',
			formula: { tex: 'A_{ {{t}} } = \\delta + \\gamma\\lambda\\,A_{ {{t1}} } = {{delta}} + {{gl}} \\times {{next}} = {{A}}', values: (f) => ({ ...f.metrics!, gl: fmtSetting(f.metrics!.gl) }) },
			narration: { what: I('Step {frame.t|int}: A = {frame.delta} + {frame.gl} × {frame.next} = {frame.A}.', '第 {frame.t|int} 步：A = {frame.delta} + {frame.gl} × {frame.next} = {frame.A}。') }
		}
	},
	schema: {
		s: { kind: 'tensor', label: I('Arm [q1, q2, ω1, ω2]', '机械臂 [q1, q2, ω1, ω2]'), shape: [4], tol: 1e-12 },
		A: { kind: 'tensor', label: I('Advantage of each step', '每一步的优势'), track: 'changes', tol: 1e-12 }
	},
	defaults: { T: ppo.defaults.T, gamma: ppo.defaults.gamma, lambda: ppo.defaults.lambda, hidden: ppo.defaults.hidden, logStd0: ppo.defaults.logStd0 },
	*run(input, ctx): ProcessRun<EpState, EpOpt> {
		const o = ctx.opt, g: Pt = targetOf(input);
		const p = paramsOf(ppo.model(ctx.rng.fork('init'), { ...ppo.defaults, hidden: o.hidden, logStd0: o.logStd0 }));
		const rng = ctx.rng.fork('batch'), r: number[] = [], v: number[] = [], t0 = tip(START);
		let s = START;
		yield { phase: 'start', state: { s: vec(s), A: { shape: [0], data: new Float64Array(0) } }, metrics: { ...named(s), tx: g[0], ty: g[1], px: t0[0], py: t0[1], d: Math.hypot(g[0] - t0[0], g[1] - t0[1]), T: o.T } };
		for (let t = 0; t < o.T && !ctx.signal.cancelled; t++) {
			const ob = obs(s, g), pol = policy(p, ob), z1 = rng.normal(), z2 = rng.normal();
			const a: Pt = [pol.mu[0] + pol.sigma[0] * z1, pol.mu[1] + pol.sigma[1] * z2];
			const lp = logProb(a, pol.mu, pol.logstd), V = value(p, ob);
			v.push(V);
			yield { phase: 'act', step: t, state: {}, metrics: { t, dx: ob[0], dy: ob[1], ...named(s), mu1: pol.mu[0], mu2: pol.mu[1], s1: pol.sigma[0], s2: pol.sigma[1], ls1: pol.logstd[0], ls2: pol.logstd[1], z1, z2, a1: a[0], a2: a[1], logp: lp, v: V } };
			const st = step(s, a, g), p2 = tip(st.s2);
			r.push(st.r);
			yield { phase: 'move', step: t, state: { s: vec(st.s2) }, metrics: { t, a1: a[0], a2: a[1], ...named(st.s2), px: p2[0], py: p2[1], d: st.d, cost: st.cost, r: st.r } };
			s = st.s2;
		}
		const T = r.length, vLast = value(p, obs(s, g)), G = gae(r, v, vLast, o.gamma, o.lambda), A: Tensor = { shape: [T], data: new Float64Array(T) };
		yield { phase: 'end', step: T, state: { A }, metrics: { T, ret: r.reduce((x, y) => x + y, 0), vLast, v0: v[0], d: Math.hypot(g[0] - tip(s)[0], g[1] - tip(s)[1]) } };
		for (let k = T - 1; k >= 0 && !ctx.signal.cancelled; k--) {
			A.data[k] = G.A[k];
			const v1 = k === T - 1 ? vLast : v[k + 1];
			yield { phase: 'gae', step: k, state: { A }, changes: [{ field: 'A', index: [k], from: 0, to: G.A[k] }], metrics: { t: k, t1: k + 1, r: r[k], v: v[k], v1, gamma: o.gamma, lambda: o.lambda, gl: o.gamma * o.lambda, delta: G.delta[k], next: k === T - 1 ? 0 : G.A[k + 1], A: G.A[k], R: G.R[k] } };
		}
	}
};

// pair: the task's comparison. The same training (trainer ppo, seed 1) twice in lockstep, with the learner's ε and with
// ε = 0.2, one frame per PPO iteration (step = the iteration), at its last update: the batch's average episode return
// and how far the iteration has moved the policy (KL), for each run (epRet, kl and epRet0, kl0). The first run is exactly the `train` source (claims.test.ts). No Python reference: the twin covers the trainer.
// kind 'training', so the e2e step cap applies to it as to the train sources.
const inner = trainingProcess(ppo);
export const pair: ProcessDef<unknown, PpoOpt, { step: number }> = {
	id: 'pair',
	kind: 'training',
	phases: inner.phases,
	defaults: ppo.defaults,
	title: I('The same training with your ε and with ε = 0.2', '同样的训练：你的 ε 和 ε = 0.2'),
	schema: { step: { kind: 'int', label: I('Step', '步数') } },
	*run(input, ctx): ProcessRun<{ step: number }, PpoOpt> {
		const U = ctx.opt.epochs * ctx.opt.minibatches, g = targetOf(input), runs = [epsOf(input), DEFAULT_EPS].map((eps) => inner.run({ target: g, eps }, { ...ctx, detail: 'coarse' }));
		for (;;) {
			const [a, b] = runs.map((r) => r.next());
			if (a.done || b.done || ctx.signal.cancelled) return;
			const f = a.value as Frame<TrainState>, m = f.metrics ?? {}, m0 = (b.value as Frame<TrainState>).metrics ?? {};
			if (f.phase === 'init') yield { phase: 'init', state: { step: 0 }, metrics: { step: 0 } };
			else if ((m.step + 1) % U === 0) yield { phase: 'step', step: (m.step + 1) / U - 1, state: { step: (m.step + 1) / U - 1 }, metrics: { step: (m.step + 1) / U - 1, epRet: m.epRet, kl: m.kl, epRet0: m0.epRet, kl0: m0.kl } };
		}
	}
};
