// gridworld: the processes (kind C). bandit: one player (you, greedy, eps, ucb) on four slot machines, choose (score for
// UCB) -> update per pull. race: greedy, ε-greedy, UCB and Thompson sampling side by side, one frame per pull, with regret.
//   q       Q-learning on the learner's maze: act (choose with u vs ε, move, reward) -> update (target, Q <- Q + αδ),
//           (step = the move, sub = the episode), and an end frame per episode (step = the episode). Q (one row of 4 per cell) and the visit counts N are tracked by changes.
//   vi      value iteration on the same maze: one frame per sweep (opt.gamma overrides the learner's γ).
// Randomness only from ctx.rng. Common random numbers: every player draws machine k's payouts from its own copy of
// rng.fork('arm<k>') (the j-th pull of a machine pays the same for everyone) and its own choices from rng.fork(policy),
// so the race repeats each single-player run exactly. Thompson sampling draws normals (Beta via Marsaglia-Tsang gammas):
// the race records its draws and pipeline/gridworld/reference.py replays them.
import type { ProcessDef, ProcessRun, Rng, Tensor } from '../../atlas/types/index.ts';
import { agreement, arrowOf, BEST, goalDistance, greedyPath, isEnd, OPEN, POLICIES, PROBS, step, toRaw, valueIteration, worldOf, xy, type Policy } from './world.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });
const K = PROBS.length;
const I = (en: string, zh: string) => ({ en, zh });
const argmaxAll = (v: number[]) => v.map((x, i) => (x === Math.max(...v) ? i : -1)).filter((i) => i >= 0);

// ------------------------------------------------------------------ bandits
export type BanditOpt = { policy: Policy; steps: number; eps: number; c: number };
export type BanditState = { Q: Tensor; N: Tensor; R: Tensor };
interface Player { policy: Policy; Q: number[]; N: number[]; R: number[]; rng: Rng; arms: Rng[]; regret: number }
const player = (policy: Policy, rng: Rng): Player => ({ policy, Q: [0, 0, 0, 0], N: [0, 0, 0, 0], R: [0, 0, 0, 0], rng: rng.fork(policy), arms: PROBS.map((_, k) => rng.fork(`arm${k}`)), regret: 0 });

/** Gamma(a) for a >= 1 (Marsaglia-Tsang), then Beta(a, b) = X / (X + Y). */
function gammaDraw(rng: Rng, a: number): number {
	const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
	for (;;) {
		let x = 0, v = 0;
		do (x = rng.normal()), (v = 1 + c * x);
		while (v <= 0);
		v = v * v * v;
		const u = rng.next();
		if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
	}
}
const betaDraw = (rng: Rng, a: number, b: number) => {
	const x = gammaDraw(rng, a);
	return x / (x + gammaDraw(rng, b));
};

/** One decision. greedy: the best average (ties at random); eps: first u = next(), u < ε explores (a random arm);
 *  ucb: untried arms first, then the highest Q + c·sqrt(ln t / n) (ties: smallest index); thompson: the highest Beta draw. */
function decide(p: Player, t: number, o: BanditOpt, mine?: number) {
	const best = argmaxAll(p.Q);
	const bonus = p.N.map((n) => (n ? o.c * Math.sqrt(Math.log(t) / n) : -1));
	let arm = 0, u = -1, explore = 0;
	const tie = () => (best.length > 1 ? best[p.rng.int(best.length)] : best[0]);
	if (p.policy === 'you') arm = mine ?? 0;
	else if (p.policy === 'greedy') arm = tie();
	else if (p.policy === 'eps') {
		u = p.rng.next();
		if (u < o.eps) (arm = p.rng.int(K)), (explore = 1);
		else arm = tie();
	} else if (p.policy === 'ucb') {
		const untried = p.N.indexOf(0);
		const score = p.Q.map((q, k) => q + bonus[k]);
		arm = untried >= 0 ? untried : score.indexOf(Math.max(...score));
	} else {
		const theta = p.Q.map((_, k) => betaDraw(p.rng, 1 + p.R[k], 1 + p.N[k] - p.R[k]));
		arm = theta.indexOf(Math.max(...theta));
	}
	return { arm, u, explore, isBest: best.includes(arm) ? 1 : 0, bonus };
}
function pull(p: Player, arm: number) {
	const r = p.arms[arm].next() < PROBS[arm] ? 1 : 0, old = p.Q[arm];
	p.N[arm] += 1;
	p.R[arm] += r;
	p.Q[arm] = old + (r - old) / p.N[arm];
	p.regret += PROBS[BEST] - PROBS[arm];
	return { r, old };
}
const per = (pre: string, v: number[]) => Object.fromEntries(v.map((x, k) => [`${pre}${k}`, x]));

export const bandit: ProcessDef<unknown, BanditOpt, BanditState> = {
	id: 'bandit', kind: 'algorithm', title: I('One player on four slot machines', '一个玩家玩四台老虎机'),
	phases: {
		init: { label: I('Four machines, nothing known', '四台机器，一无所知'), kind: 'other' },
		choose: { label: I('Choose a machine', '选一台机器'), kind: 'act', narration: { what: I('Pull {frame.n|int}: the player chooses a machine.', '第 {frame.n|int} 次：玩家选一台机器。') } },
		score: { label: I('Score every machine (UCB)', '给每台机器打分（UCB）'), kind: 'act', formula: { tex: 'U = Q + c\\sqrt{\\ln t / n} = {{q}} + {{c}} \\cdot \\sqrt{\\ln {{t}} / {{n}}} = {{score}}', values: (f) => ({ q: f.metrics!.q, c: f.metrics!.c, t: f.metrics!.t, n: f.metrics!.cnt, score: f.metrics!.score }) }, narration: { what: I('Pull {frame.n|int}: UCB picks the machine with the highest average plus bonus.', '第 {frame.n|int} 次：UCB 选“平均值加奖励分”最高的机器。') } },
		update: { label: I('Pull and update the average', '拉一下，更新平均值'), kind: 'update', formula: { tex: 'Q_{\\text{new}} = Q + \\frac{r - Q}{n} = {{old}} + \\frac{ {{r}} - {{old}} }{ {{n}} } = {{new}}', values: (f) => ({ old: f.metrics!.old, r: f.metrics!.r, n: f.metrics!.cnt, new: f.metrics!.new }) }, narration: { what: I('The machine paid {frame.r|int}; its average is now {frame.new}.', '这台机器给了 {frame.r|int} 枚金币；它的平均值现在是 {frame.new}。') } }
	},
	schema: {
		Q: { kind: 'tensor', label: I('Average payout of each machine', '每台机器的平均回报'), shape: [K], tol: 1e-12 },
		N: { kind: 'tensor', label: I('Pulls of each machine', '每台机器被拉的次数'), shape: [K], tol: 'exact' },
		R: { kind: 'tensor', label: I('Coins won on each machine', '每台机器赢到的金币'), shape: [K], tol: 'exact' }
	},
	defaults: { policy: 'greedy', steps: 200, eps: 0.1, c: 1 },
	*run(input, ctx): ProcessRun<BanditState, BanditOpt> {
		const o = ctx.opt;
		const mine = toRaw(input).pulls;
		const p = player(o.policy, ctx.rng);
		const steps = o.policy === 'you' ? mine.length : o.steps;
		const snap = () => ({ Q: T([K], p.Q), N: T([K], p.N), R: T([K], p.R) });
		yield { phase: 'init', state: snap(), metrics: { steps, best: BEST, pBest: PROBS[BEST], eps: o.eps, c: o.c } };
		for (let t = 0; t < steps && !ctx.signal.cancelled; t++) {
			const d = decide(p, t, o, mine[t]), k = d.arm;
			yield { phase: o.policy === 'ucb' ? 'score' : 'choose', step: t, state: {}, metrics: { t, n: t + 1, arm: k, ...per('q', p.Q), ...per('n', p.N), u: d.u, eps: o.eps, explore: d.explore, isBest: d.isBest, ...per('b', d.bonus), q: p.Q[k], c: o.c, cnt: p.N[k], score: p.Q[k] + d.bonus[k] } };
			const { r, old } = pull(p, k);
			yield { phase: 'update', step: t, state: snap(), metrics: { t, n: t + 1, arm: k, r, old, new: p.Q[k], cnt: p.N[k], regret: p.regret, wins: p.R.reduce((a, b) => a + b, 0), nBest: p.N[BEST] } };
		}
	}
};

export type RaceState = { N: Tensor; R: Tensor };
export const race: ProcessDef<unknown, BanditOpt, RaceState> = {
	id: 'race', kind: 'algorithm', title: I('Four strategies, the same machines', '四种策略，同样的机器'), recordDraws: true,
	phases: { pull: { label: I('Everyone pulls once', '每个玩家各拉一次'), kind: 'act', narration: { what: I('After {frame.n|int} pulls, greedy has regret {frame.r0}.', '拉了 {frame.n|int} 次后，贪心策略的遗憾是 {frame.r0}。') } } },
	schema: {
		N: { kind: 'tensor', label: I('Pulls per strategy and machine', '每种策略在每台机器上的拉动次数'), shape: [4, K], tol: 'exact' },
		R: { kind: 'tensor', label: I('Coins per strategy and machine', '每种策略在每台机器上赢的金币'), shape: [4, K], tol: 'exact' }
	},
	defaults: { policy: 'greedy', steps: 200, eps: 0.1, c: 1 },
	*run(_input, ctx): ProcessRun<RaceState, BanditOpt> {
		const ps = POLICIES.map((pol) => player(pol, ctx.rng));
		for (let t = 0; t < ctx.opt.steps && !ctx.signal.cancelled; t++) {
			const arms = ps.map((p) => decide(p, t, ctx.opt).arm);
			ps.forEach((p, i) => pull(p, arms[i]));
			yield { phase: 'pull', step: t, state: { N: T([4, K], ps.flatMap((p) => p.N)), R: T([4, K], ps.flatMap((p) => p.R)) }, metrics: { t, n: t + 1, ...per('r', ps.map((p) => p.regret)), ...per('a', arms) } };
		}
	}
};

// ------------------------------------------------------------------ the maze
export type QOpt = { maxSteps: number };
export type QState = { board: Tensor; Q: Tensor; N: Tensor; pos: number };
export const q: ProcessDef<unknown, QOpt, QState> = {
	id: 'q', kind: 'algorithm', title: I('Q-learning in your maze', '在你的迷宫里做 Q 学习'),
	phases: {
		init: { label: I('Q-table at zero', 'Q 表全为 0'), kind: 'other' },
		act: { label: I('Choose and move', '选择并移动'), kind: 'act', narration: { what: I('The agent chooses a move, makes it and gets r = {frame.r|int}.', '智能体选一个动作，走出去，得到 r = {frame.r|int}。') } },
		update: { label: I('Update one Q value', '更新一个 Q 值'), kind: 'update', formula: { tex: '\\text{target} = r + \\gamma \\max_{a\'} Q(s\', a\') = {{r}} + {{gamma}} \\times {{maxNext}} = {{target}};\\quad Q_{\\text{new}} = Q + \\alpha\\,(\\text{target} - Q) = {{old}} + {{alpha}} \\times ({{target}} - {{old}}) = {{new}}', values: (f) => ({ ...f.metrics! }) }, narration: { what: I('Q moves from {frame.old} toward the target {frame.target}: it becomes {frame.new}.', 'Q 从 {frame.old} 朝目标值 {frame.target} 靠近，变成 {frame.new}。') } },
		end: { label: I('End of an episode', '一个回合结束'), kind: 'observe', narration: { what: I('This episode took {frame.moves|int} moves.', '这个回合走了 {frame.moves|int} 步。') } }
	},
	schema: {
		board: { kind: 'tensor', label: I('The maze (0 open, 1 wall, 2 goal, 3 pit)', '迷宫（0 空地，1 墙，2 终点，3 陷阱）'), tol: 'exact' },
		Q: { kind: 'tensor', label: I('Q-table: one row of 4 moves per cell', 'Q 表：每个格子一行，对应 4 个动作'), track: 'changes', scale: 'diverging', tol: 1e-12 },
		N: { kind: 'tensor', label: I('Moves made from each cell', '从每个格子出发的次数'), track: 'changes', tol: 'exact' },
		pos: { kind: 'int', label: I('Where the agent is', '智能体的位置'), tol: 'exact' }
	},
	defaults: { maxSteps: 80 },
	*run(input, ctx): ProcessRun<QState, QOpt> {
		const raw = toRaw(input), W = worldOf(raw), max = ctx.opt.maxSteps, n = W.cell.length;
		const { gamma, epsilon: eps, alpha, episodes } = raw;
		const Q = T([n, 4], new Float64Array(n * 4)), N = T([n], new Float64Array(n)), V = valueIteration(W, gamma);
		const open = W.cell.filter((c) => c === OPEN).length;
		yield { phase: 'init', state: { board: T([W.h, W.w], W.cell), Q, N, pos: W.start }, metrics: { start: W.start, open, entries: open * 4, gamma, eps, alpha, episodes, maxSteps: max, dist: goalDistance(W), vStar: V[W.start] } };
		let goals = 0;
		for (let e = 0; e < episodes && !ctx.signal.cancelled; e++) {
			let s = W.start, ret = 0, how = 0, moves = 0;
			while (moves < max && !how) {
				const qs = [0, 1, 2, 3].map((a) => Q.data[s * 4 + a]), u = ctx.rng.next();
				let a = 0, explore = 0, ties = 1;
				if (u < eps) (a = ctx.rng.int(4)), (explore = 1);
				else {
					const best = argmaxAll(qs);
					ties = best.length;
					a = ties > 1 ? best[ctx.rng.int(ties)] : best[0];
				}
				const { s2, r, done } = step(W, s, a), nv = N.data[s], [x, y] = xy(W, s), [x2, y2] = xy(W, s2);
				N.data[s] = nv + 1;
				yield { phase: 'act', step: moves, sub: e, state: { N, pos: s2 }, changes: [{ field: 'N', index: [s], from: nv, to: nv + 1 }], metrics: { s, x, y, u, eps, explore, ties, a, ...Object.fromEntries(qs.map((v, i) => [`q${i}`, v])), s2, x2, y2, r, done: done ? 1 : 0, bump: s2 === s ? 1 : 0 } };
				const maxNext = done ? 0 : Math.max(...[0, 1, 2, 3].map((b) => Q.data[s2 * 4 + b]));
				const target = r + gamma * maxNext, old = Q.data[s * 4 + a], delta = target - old, nw = old + alpha * delta;
				Q.data[s * 4 + a] = nw;
				yield { phase: 'update', step: moves, sub: e, state: { Q }, changes: [{ field: 'Q', index: [s, a], from: old, to: nw }], metrics: { s, a, s2, r, done: done ? 1 : 0, gamma, alpha, maxNext, target, old, delta, new: nw } };
				ret += r;
				moves++;
				s = s2;
				how = done ? Math.sign(r) : 0;
			}
			if (how === 1) goals++;
			const path = greedyPath(W, Q.data), ag = agreement(W, Q.data, V, gamma);
			yield { phase: 'end', step: e, state: {}, metrics: { ep: e, moves, ret, how, goals, vStart: Math.max(...[0, 1, 2, 3].map((b) => Q.data[W.start * 4 + b])), arrow: arrowOf(Q.data, W.start), len: path ? path.length - 1 : -1, decided: ag.decided, agree: ag.agree, end: isEnd(W, s) ? s : -1 } };
		}
	}
};

export type ViOpt = { gamma: number };
export type ViState = { board: Tensor; V: Tensor };
export const vi: ProcessDef<unknown, ViOpt, ViState> = {
	id: 'vi', kind: 'algorithm', title: I('Value iteration: the rules known', '价值迭代：已知规则'),
	phases: {
		init: { label: I('All values at zero', '所有价值为 0'), kind: 'other' },
		sweep: { label: I('One sweep over every cell', '把每个格子扫一遍'), kind: 'update', narration: { what: I('Sweep {frame.k|int}: the largest change is {frame.delta}.', '第 {frame.k|int} 遍：最大的变化是 {frame.delta}。') } }
	},
	schema: {
		board: { kind: 'tensor', label: I('The maze (0 open, 1 wall, 2 goal, 3 pit)', '迷宫（0 空地，1 墙，2 终点，3 陷阱）'), tol: 'exact' },
		V: { kind: 'tensor', label: I('Value of each cell', '每个格子的价值'), scale: 'diverging', tol: 1e-12 }
	},
	defaults: { gamma: -1 },
	*run(input, ctx): ProcessRun<ViState, ViOpt> {
		const W = worldOf(input), gamma = ctx.opt.gamma >= 0 ? ctx.opt.gamma : toRaw(input).gamma;
		const frames: { V: number[]; k: number; delta: number }[] = [];
		valueIteration(W, gamma, (V, k, delta) => frames.push({ V, k, delta }));
		yield { phase: 'init', state: { board: T([W.h, W.w], W.cell), V: T([W.h, W.w], W.cell.map(() => 0)) }, metrics: { start: W.start, gamma, dist: goalDistance(W) } };
		for (const f of frames) yield { phase: 'sweep', step: f.k, state: { V: T([W.h, W.w], f.V) }, metrics: { k: f.k, delta: f.delta, gamma, vStart: f.V[W.start] } };
	}
};
