// behavior-cloning: process 'roll' (kind C): a trained policy drives on its own, 20 windy episodes from the start, one
// frame per move, and the expert drives the same 20 episodes under the same wind (the gusts are drawn first, so both
// meet exactly the same gusts). The wind comes from the run's seed (EVAL.seed, the same wind the trainer's success metric
// uses). Frames: start -> (move per step -> end) per episode -> done.
//   move  the cell s, the policy's move a, whether a gust blew it down instead, the new cell; whether s is in the
//         dataset the policy was trained on (seen), and whether a brings it closer to the goal (good)
//   end   the episode's outcome (1 goal, −1 pit, 0 out of moves), its length, its moves on unseen cells; the expert's
//         outcome on the same wind
//   done  the success rates and the moves on seen and unseen cells, and how many of each were good
// The input is { raw, params, cells }: the learner's input, the weights (the last snapshot of the train source named in
// the manifest) and the dataset's cells. pipeline/behavior-cloning/reference.py repeats it in plain Python.
//   npm run atlas -- run behavior-cloning roll --frames 0..40
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { distances, EVAL, expertMove, featuresOf, GOAL, GUST, isEnd, PIT, playEpisode, policyAt, step, toRaw, windOf, worldOf, xy, type Params } from './world.ts';

export type RollInput = { raw: unknown; params: Params; cells: number[] };
export type RollOpt = { episodes: number; T: number };
export type RollState = { pos: number; visits: Tensor };
const I = (en: string, zh: string) => ({ en, zh });

export const roll: ProcessDef<RollInput, RollOpt, RollState> = {
	id: 'roll',
	kind: 'algorithm',
	title: I('The policy drives on its own, in the wind', '策略自己在风中行驶'),
	phases: {
		start: { label: I('The start', '起点'), kind: 'input' },
		move: { label: I('One move', '走一步'), kind: 'act', narration: { what: I('The policy moves from column {frame.x}, row {frame.y} (counting from 0).', '策略从第 {frame.x} 列、第 {frame.y} 行（从 0 数起）走一步。') } },
		end: { label: I('The episode ends', '回合结束'), kind: 'observe', narration: { what: I('The episode ends after {frame.len} moves.', '这个回合走了 {frame.len} 步后结束。') } },
		done: { label: I('All episodes', '全部回合'), kind: 'aggregate', narration: { what: I('The policy reached the goal in {frame.goals} of {frame.episodes} episodes.', '策略在 {frame.episodes} 个回合中有 {frame.goals} 个到达终点。') } }
	},
	schema: {
		pos: { kind: 'int', label: I('The agent’s cell', '智能体所在的格子'), tol: 'exact' },
		visits: { kind: 'tensor', label: I('Visits per cell', '每个格子被到访的次数'), track: 'changes', tol: 'exact' }
	},
	defaults: { episodes: EVAL.episodes, T: EVAL.T },
	*run(input, ctx): ProcessRun<RollState, RollOpt> {
		if (!input?.params?.['net.0.weight']) return; // no demonstrations: no trained policy, no frames (the steps show the empty note)
		const g = toRaw(input.raw), W = worldOf(g.map), d = distances(W), o = ctx.opt, P = input.params, data = new Set(input.cells);
		const wind = windOf(ctx.rng, o.episodes, o.T, g.wind);
		const visits: Tensor = { shape: [W.h, W.w], data: new Float64Array(W.w * W.h) };
		let goals = 0, exGoals = 0, pits = 0;
		const n = { onMoves: 0, onGood: 0, offMoves: 0, offGood: 0 };
		yield { phase: 'start', state: { pos: W.start, visits }, metrics: { episodes: o.episodes, T: o.T, wind: g.wind, start: W.start, cells: data.size } };
		for (let e = 0; e < o.episodes && !ctx.signal.cancelled; e++) {
			let s = W.start, t = 0, off = 0, firstOff = -1, gusts = 0;
			for (; t < o.T && !isEnd(W, s); t++) {
				const pol = policyAt(P, featuresOf(W, s)), a = pol.a, gust = wind[e][t], s2 = step(W, s, gust ? GUST : a);
				const seen = data.has(s) ? 1 : 0, good = d[s] >= 0 && d[step(W, s, a)] === d[s] - 1 ? 1 : 0, [x, y] = xy(W, s), [x2, y2] = xy(W, s2);
				if (seen) (n.onMoves++, (n.onGood += good));
				else (n.offMoves++, (n.offGood += good), off++, firstOff < 0 && (firstOff = t));
				gusts += gust ? 1 : 0;
				const old = visits.data[s2];
				visits.data[s2] = old + 1;
				yield { phase: 'move', step: t, sub: e, state: { pos: s2, visits }, changes: [{ field: 'visits', index: [y2, x2], from: old, to: old + 1 }], metrics: { e, t, s, x, y, a, gust: gust ? 1 : 0, s2, x2, y2, seen, good, ex: expertMove(W, d, s), p0: pol.p[0], p1: pol.p[1], p2: pol.p[2], p3: pol.p[3] } };
				s = s2;
			}
			const outcome = W.cell[s] === GOAL ? 1 : W.cell[s] === PIT ? -1 : 0;
			const ex = playEpisode(W, (c) => expertMove(W, d, c), wind, e, o.T).outcome;
			goals += outcome === 1 ? 1 : 0, pits += outcome === -1 ? 1 : 0, exGoals += ex === 1 ? 1 : 0;
			yield { phase: 'end', step: e, sub: e, state: { pos: s }, metrics: { e, outcome, len: t, off, firstOff, gusts, last: s, exOutcome: ex, goals, exGoals, pits } };
		}
		yield { phase: 'done', state: { pos: W.start }, metrics: { episodes: o.episodes, goals, exGoals, pits, success: goals / o.episodes, exSuccess: exGoals / o.episodes, ...n } };
	}
};
