// snake-dqn: what the views and the lesson read (pure TS): the networks' Q-values from the traces, a frame's state, the
// learner's own game (step 10), the training log in the spec, and the screen-reader texts of the views.
import type { At, ExplainerRuntime, I18n, StageProps } from '../../atlas/types/index.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { posOf, rawOf } from './adapters/snake-board/adapter.ts';
import { ACTION, type Game, HUNGER, L, argmax, newGame, step } from './rules.ts';
import type { Row, Stored } from './process.ts';

type RT = Pick<ExplainerRuntime, 'input' | 'sources' | 'spec'>;
/** Q(s, ·) of a trace source on the learner's board (fwd: the online network, tgt: the target copy). */
export function qOf(rt: Pick<RT, 'sources'>, src = 'fwd'): number[] | null {
	const t = rt.sources[src]?.trace?.t.out;
	return t ? Array.from(t.data) : null;
}
export const best = (q: number[] | null) => (q ? argmax(q) : -1);
export type St = { game?: Game; next?: Game; buf?: Stored[]; pick?: number[]; rows?: Row[] };
export const stateOf = (at: Pick<At, 'tape' | 'index'>): St => (at.tape && at.index >= 0 ? (at.tape.state(at.index) as St) : {});
export const m = (at: Pick<At, 'frame'>, k: string) => at.frame?.metrics?.[k] ?? NaN;
/** 3 significant digits, as the narration prints them, but a small number as a plain decimal (0.000176, not 1.76e−4);
 *  the same rule as matrix-factorization's picture.ts. */
export const num = (v: number) => (v !== 0 && Math.abs(v) < 1e-3 && Math.abs(v) >= 1e-6 ? String(Number(v.toPrecision(3))).replace('-', '−') : fmt3(v));
/** A body-plane number as the input view's grid prints it (two decimals: 0.67, 0.09). */
export const dec2 = (v: number) => String(+v.toFixed(2));

/** The learner's own game (step 10): their moves from the start of the seed, until a crash or HUNGER moves without food. */
export function yourGame(input: unknown) {
	const r = rawOf(input);
	let g = newGame(r.seed), used = 0;
	for (const a of r.game) {
		if (g.over || g.hunger >= HUNGER) break;
		g = step(g, a).g;
		used++;
	}
	return { g, used, done: !!g.over || g.hunger >= HUNGER };
}
/** The training log the export put into the spec (pipeline/snake-dqn/export.py). */
export type Log = { moves: number[]; score: number[]; trainMoves: number[]; train: number[]; probe: { updates: number[]; online: number[]; target: number[] } };
export const logOf = (rt: Pick<RT, 'spec'>) => (rt.spec as unknown as { dqnlog?: Log } | null)?.dqnlog ?? null;

const both = (f: (l: 'en' | 'zh') => string) => ({ en: f('en'), zh: f('zh') });
const qs = (q: number[], l: 'en' | 'zh') => q.map((v, a) => `${ACTION[a][l]} ${fmt3(v)}`).join(l === 'en' ? ', ' : '，');
/** Screen-reader text of the step view. */
export function describeStep(p: StageProps): I18n {
	const mode = String(p.props.mode ?? 'input'), g = stateOf(p.at).game ?? posOf(p.rt.input), q = qOf(p.rt);
	const head = both((l) => (l === 'en' ? `The snake has length ${g.snake.length} and has eaten ${g.score}. ` : `蛇长 ${g.snake.length}，吃了 ${g.score} 个。`));
	const f = p.at.frame?.metrics ?? {};
	const tail =
		mode === 'q' && q ? both((l) => (l === 'en' ? 'Q-values: ' : 'Q 值：') + qs(q, l))
		: mode === 'act' ? both((l) => (l === 'en' ? `Draw u = ${fmt3(f.u)} against ε = ${f.eps}: ${f.explore ? 'a random move' : 'the best move'}, ${ACTION[f.a]?.en}.` : `抽到 u = ${fmt3(f.u)}，和 ε = ${f.eps} 比：${f.explore ? '随机走一步' : '走最好的一步'}，${ACTION[f.a]?.zh}。`))
		: mode === 'reward' ? both((l) => (l === 'en' ? `The move gives the reward ${f.r}.` : `这一步得到奖励 ${f.r}。`))
		: mode === 'target' ? both((l) => (l === 'en' ? `The target y = ${fmt3(f.y)}.` : `目标值 y = ${fmt3(f.y)}。`))
		: mode === 'loss' ? both((l) => (l === 'en' ? `Q(s, a) = ${fmt3(f.qa)}, target ${fmt3(f.y)}, loss ${fmt3(f.loss)}.` : `Q(s, a) = ${fmt3(f.qa)}，目标值 ${fmt3(f.y)}，损失 ${fmt3(f.loss)}。`))
		: L('The board as three planes of 9 × 9 numbers: head, body, food.', '棋盘是三层 9 × 9 的数：蛇头、身体、食物。');
	return { en: head.en + tail.en, zh: head.zh + tail.zh };
}
/** Screen-reader text of the replay view. */
export function describeReplay(p: StageProps): I18n {
	const s = stateOf(p.at), rows = p.at.frame?.phase === 'learn' ? s.rows ?? [] : [];
	if (rows.length) return both((l) => (l === 'en' ? 'The batch: ' : '这一批：') + rows.map((x) => `#${x.t + 1} y ${fmt3(x.y)} Q ${fmt3(x.q)}`).join('; '));
	return both((l) => (l === 'en' ? `The buffer holds ${s.buf?.length ?? 0} moves.` : `缓冲区里存了 ${s.buf?.length ?? 0} 步。`));
}
/** Screen-reader text of the watch view. */
export function describeWatch(p: StageProps): I18n {
	const g = stateOf(p.at).game, y = yourGame(p.rt.input);
	return both((l) => (l === 'en' ? `The agent: ${g?.score ?? 0} eaten after ${g?.moves ?? 0} moves. Your game: ${y.g.score} eaten after ${y.g.moves} moves.` : `智能体：走了 ${g?.moves ?? 0} 步，吃了 ${g?.score ?? 0} 个。你的这盘：走了 ${y.g.moves} 步，吃了 ${y.g.score} 个。`));
}
