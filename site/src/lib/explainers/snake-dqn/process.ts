// snake-dqn: three processes (kind C) on the exported networks, in float64 from their float32 weights (qnet.ts).
//   dqn     one DQN step on the learner's board: Q(s, ·), the ε-greedy choice (a draw u against ε), the move and its
//           reward, the target y = r + γ · max Q_target(s′, ·) (y = r when the move ends the game), the loss (y − Q(s, a))²
//   replay  the agent (ε-greedy) plays `fill` moves from the start of the learner's seed into a buffer of `cap` moves
//           (the oldest are overwritten), then draws a batch of `batch` stored moves at random and works out each loss
//   agent   the trained agent (greedy) plays a whole game from the start of the learner's seed
// Inputs carry the weights as typed arrays (w: online network, wt: target network); their JSON (atlas parity's --input)
// is only the plain part, and pipeline/snake-dqn/reference.py reads the same weights itself.
import type { ProcessDef, ProcessRun } from '../../atlas/types/index.ts';
import { type QW, qvalues } from './qnet.ts';
import { type Game, HUNGER, L, argmax, encode, newGame, replay as play, step } from './rules.ts';

type In = { seed: number; moves?: number[]; draw?: number; w?: QW | null; wt?: QW | null };
const need = (w: QW | null | undefined) => w ?? (() => { throw new Error('snake-dqn: the network is not loaded'); })();
const J = (en: string, zh: string) => ({ kind: 'json' as const, label: L(en, zh), tol: 'exact' as const });
const qm = (q: number[], p = 'q') => ({ [`${p}0`]: q[0], [`${p}1`]: q[1], [`${p}2`]: q[2] });
const END: Record<string, number> = { '': 0, wall: 1, self: 2, full: 3 };
/** The process input: plain data plus the weights (typed arrays) for the run; its JSON is only `o`. */
export function withWeights<O extends object>(o: O, w: QW | null, wt: QW | null = null): O & { w: QW | null; wt: QW | null } {
	const x = { ...o, w, wt };
	Object.defineProperty(x, 'toJSON', { value: () => o, enumerable: false });
	return x;
}
const target = (wt: QW, s2: Game, r: number, gamma: number) => {
	const q = s2.over ? [0, 0, 0] : qvalues(wt, encode(s2)), b = argmax(q), max = s2.over ? 0 : q[b];
	return { q, b, max, y: r + (s2.over ? 0 : gamma * max) };
};

export type DqnOpt = { eps: number; gamma: number };
export const dqn: ProcessDef<In, DqnOpt, { game: Game; next: Game }> = {
	id: 'dqn',
	kind: 'algorithm',
	title: L('One step of DQN on your board', '在你的棋盘上走一步 DQN'),
	phases: {
		observe: { label: L('Look at the board', '看棋盘'), kind: 'observe' },
		act: { label: L('Choose a move (ε-greedy)', '选一步（ε-贪心）'), kind: 'act' },
		reward: { label: L('Move and get the reward', '走一步，得到奖励'), kind: 'reward' },
		target: { label: L('The target', '目标值'), kind: 'loss' },
		loss: { label: L('The loss', '损失'), kind: 'loss' }
	},
	schema: { game: J('The board s', '棋盘 s'), next: J('The board after the move, s′', '走完这一步后的棋盘 s′') },
	defaults: { eps: 0.1, gamma: 0.95 },
	*run(input, ctx): ProcessRun<{ game: Game; next: Game }, DqnOpt> {
		const w = need(input.w), wt = need(input.wt), { eps, gamma } = ctx.opt, g = play(input.seed, input.moves ?? []);
		if (g.over) return;
		const q = qvalues(w, encode(g)), best = argmax(q);
		yield { phase: 'observe', step: 0, state: { game: g }, metrics: { ...qm(q), best, len: g.snake.length, score: g.score } };
		const rng = ctx.rng.fork(`draw${input.draw ?? 0}`), u = rng.next(), explore = u < eps ? 1 : 0, a = explore ? rng.int(3) : best;
		yield { phase: 'act', step: 0, state: {}, metrics: { ...qm(q), u, eps, explore, a, best, qa: q[a] } };
		const s = step(g, a);
		yield { phase: 'reward', step: 0, state: { next: s.g }, metrics: { a, r: s.r, ate: s.ate ? 1 : 0, end: END[s.g.over], score: s.g.score } };
		const t = target(wt, s.g, s.r, gamma), o = s.g.over ? [0, 0, 0] : qvalues(w, encode(s.g));
		yield { phase: 'target', step: 0, state: {}, metrics: { r: s.r, gamma, terminal: s.g.over ? 1 : 0, ...qm(t.q, 't'), tmax: t.max, tbest: t.b, ...qm(o, 'o'), omax: Math.max(...o), y: t.y } };
		const d = t.y - q[a];
		yield { phase: 'loss', step: 0, state: {}, metrics: { a, qa: q[a], y: t.y, delta: d, loss: d * d } };
	}
};

export type Stored = { t: number; s: Game; a: number; r: number; done: number; s2: Game };
export type Row = { slot: number; t: number; a: number; r: number; done: number; q: number; tmax: number; y: number; loss: number };
export type ReplayOpt = { eps: number; gamma: number; fill: number; cap: number; batch: number };
export const replay: ProcessDef<In, ReplayOpt, { buf: Stored[]; game: Game; pick: number[]; rows: Row[] }> = {
	id: 'replay',
	kind: 'algorithm',
	title: L('A replay buffer', '经验回放缓冲区'),
	phases: {
		store: { label: L('Play a move and store it', '走一步并存起来'), kind: 'act' },
		sample: { label: L('Draw a batch at random', '随机抽一批'), kind: 'sample' },
		learn: { label: L('Targets and losses of the batch', '这一批的目标值和损失'), kind: 'loss' }
	},
	schema: { buf: J('The buffer: stored moves (s, a, r, s′)', '缓冲区：存下的步（s, a, r, s′）'), game: J('The game being played', '正在玩的这一局'), pick: J('The slots drawn for the batch', '抽中的格位'), rows: J('The batch: target and loss of each move', '这一批：每一步的目标值和损失') },
	defaults: { eps: 0.1, gamma: 0.95, fill: 20, cap: 12, batch: 4 },
	*run(input, ctx): ProcessRun<{ buf: Stored[]; game: Game; pick: number[]; rows: Row[] }, ReplayOpt> {
		const w = need(input.w), wt = need(input.wt), { eps, gamma, fill, cap, batch } = ctx.opt;
		const pr = ctx.rng.fork('play'), sr = ctx.rng.fork('sample'), buf: Stored[] = [];
		let g = newGame(input.seed), ep = 0;
		for (let t = 0; t < fill; t++) {
			const q = qvalues(w, encode(g)), u = pr.next(), explore = u < eps ? 1 : 0, a = explore ? pr.int(3) : argmax(q), s = step(g, a), slot = t % cap;
			buf[slot] = { t, s: g, a, r: s.r, done: s.g.over ? 1 : 0, s2: s.g };
			yield { phase: 'store', step: t, state: { buf: buf.slice(), game: s.g }, metrics: { t, a, r: s.r, slot, size: buf.length, done: s.g.over ? 1 : 0, explore } };
			g = s.g.over || s.g.hunger >= HUNGER ? newGame(input.seed + 1000 * ++ep) : s.g;
		}
		const pick = Array.from({ length: batch }, () => sr.int(buf.length));
		yield { phase: 'sample', step: fill, state: { pick }, metrics: { size: buf.length, batch, oldest: buf.reduce((m, x) => Math.min(m, x.t), fill) } };
		const rows = pick.map((slot): Row => {
			const x = buf[slot], q = qvalues(w, encode(x.s))[x.a], t = target(wt, x.s2, x.r, gamma);
			const d = t.y - q;
			return { slot, t: x.t, a: x.a, r: x.r, done: x.done, q, tmax: t.max, y: t.y, loss: d * d };
		});
		yield { phase: 'learn', step: fill, state: { rows }, metrics: { mean: rows.reduce((s, x) => s + x.loss, 0) / rows.length, batch } };
	}
};

export type AgentOpt = { cap: number };
export const agent: ProcessDef<In, AgentOpt, { game: Game }> = {
	id: 'agent',
	kind: 'algorithm',
	title: L('The trained agent plays a game', '训练好的智能体玩一局'),
	phases: {
		start: { label: L('A new game', '新的一盘'), kind: 'input' },
		move: { label: L('A move', '一步'), kind: 'act' },
		end: { label: L('Game over', '终局'), kind: 'output' }
	},
	schema: { game: J('The board', '棋盘') },
	defaults: { cap: 1500 },
	*run(input, ctx): ProcessRun<{ game: Game }, AgentOpt> {
		const w = need(input.w);
		let g = newGame(input.seed), q = qvalues(w, encode(g));
		yield { phase: 'start', step: 0, state: { game: g }, metrics: { ...qm(q), score: 0, len: g.snake.length } };
		for (;;) {
			const a = argmax(q), s = step(g, a);
			g = s.g;
			q = g.over ? [0, 0, 0] : qvalues(w, encode(g));
			yield { phase: 'move', step: g.moves, state: { game: g }, metrics: { a, r: s.r, ate: s.ate ? 1 : 0, score: g.score, len: g.snake.length, ...qm(q) } };
			const why = g.over ? END[g.over] : g.hunger >= HUNGER ? 4 : g.moves >= ctx.opt.cap ? 5 : 0;
			if (why) {
				yield { phase: 'end', step: g.moves, state: {}, metrics: { score: g.score, moves: g.moves, why } };
				return;
			}
		}
	}
};
