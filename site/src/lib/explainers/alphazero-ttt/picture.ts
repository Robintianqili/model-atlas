// alphazero-ttt: what the views and the lesson read (pure TS): the network's answer from the trace, the search tree
// from a frame of the `search` process, and the game from the `game` process.
import type { At, ExplainerRuntime, I18n, StageProps, Tensor } from '../../atlas/types/index.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { rawOf } from './adapters/ttt-board/adapter.ts';
import { type Board, CELL, L, RESULT, boardOf, legal, markOf, play, solve, toMove } from './rules.ts';
import type { Weights } from './search.ts';

export const posOf = (rt: Pick<ExplainerRuntime, 'input'>) => boardOf(rawOf(rt.input).moves);
const argmax = (xs: ArrayLike<number>, ok: (i: number) => boolean) => { let b = -1; for (let i = 0; i < xs.length; i++) if (ok(i) && (b < 0 || xs[i] > xs[b])) b = i; return b; };

/** The network's answer on the learner's position, from the kind-A trace: the 9 priors and the value. */
export function netOf(rt: Pick<ExplainerRuntime, 'input' | 'sources' | 'pre'>) {
	const tr = rt.sources.fwd?.trace, b = posOf(rt);
	if (!tr?.t.prior || !tr.t.value) return null;
	const p = Array.from(tr.t.prior.data);
	return { b, p, v: tr.t.value.data[0], top: argmax(p, (i) => b[i] === 0), board: Array.from(rt.pre?.tensors.board?.data ?? []), mask: Array.from(rt.pre?.tensors.mask?.data ?? []) };
}

export type TreeState = { N: Tensor; W: Tensor; P: Tensor; parent: Tensor; move: Tensor; path: number[]; count: number };
export const stateOf = (at: Pick<At, 'tape' | 'index'>) => (at.tape && at.index >= 0 ? (at.tape.state(at.index) as TreeState) : null);
export function kidsOf(s: TreeState, node: number): number[] {
	const out: number[] = [];
	for (let k = 1; k < s.count; k++) if (s.parent.data[k] === node) out.push(k);
	return out;
}
/** The board at a node: the root's position plus the moves along the path. */
export function boardAt(s: TreeState, root: Board, node: number): Board {
	const moves: number[] = [];
	for (let x = node; x > 0; x = s.parent.data[x]) moves.unshift(s.move.data[x]);
	return moves.reduce(play, root);
}
/** Per cell at the root (9 entries, NaN for a taken cell): visits n, prior p, mean value q; and their totals. */
export function rootOf(s: TreeState | null) {
	const n = new Array(9).fill(NaN), p = new Array(9).fill(NaN), q = new Array(9).fill(NaN);
	if (!s) return null;
	for (const k of kidsOf(s, 0)) {
		const c = s.move.data[k], nk = s.N.data[k];
		[n[c], p[c], q[c]] = [nk, s.P.data[k], nk > 0 ? s.W.data[k] / nk : 0];
	}
	const sum = n.reduce((a, x) => a + (x || 0), 0);
	return { n, p, q, sum, most: argmax(n.map((x) => x || -1), (i) => n[i] >= 0), top: argmax(p.map((x) => x || -1), (i) => p[i] >= 0) };
}
/** The game against the search (process `game`), at its latest frame. */
export function gameOf(rt: Pick<ExplainerRuntime, 'sources'>) {
	const tp = rt.sources.game?.tape;
	if (!tp || !tp.length) return null;
	const i = tp.length - 1, f = tp.frame(i), st = tp.state(i) as { board?: Tensor };
	const b = Array.from(st.board?.data ?? new Array(9).fill(0));
	const last = [...Array(i + 1).keys()].reverse().map((j) => tp.frame(j)).find((x) => x.phase === 'ai');
	return { b, phase: f.phase, m: f.metrics ?? {}, you: f.metrics?.you ?? 1, last: last?.metrics ?? null, over: f.phase === 'end' };
}
/** Weights of a network with all-zero numbers (uniform priors, value 0): the illustrative demos of the views. */
export function zeroWeights(H = 4): Weights {
	const z = (...shape: number[]) => ({ shape, data: new Float32Array(shape.reduce((a, x) => a * x, 1)) });
	return { W1: z(H, 9), b1: z(H), W2: z(H, H), b2: z(H), Wp: z(9, H), bp: z(9), Wv: z(1, H), bv: z(1), H };
}
export const cellName = (c: number) => CELL[c] ?? L('no cell', '没有格子');
export const emptyCells = (b: Board) => legal(b).length;

const join = (xs: [I18n, string][], lang: 'en' | 'zh') => xs.map(([c, v]) => `${c[lang]} ${v}`).join(lang === 'en' ? ', ' : '，');
const both = (f: (lang: 'en' | 'zh') => string): I18n => ({ en: f('en'), zh: f('zh') });
/** Screen-reader text of the board view. */
export function describeBoard(p: StageProps): I18n {
	const mode = String(p.props.mode ?? 'input'), b = posOf(p.rt), net = netOf(p.rt), root = rootOf(stateOf(p.at)), me = markOf(toMove(b));
	const free = legal(b), guess = rawOf(p.rt.input).guess;
	if ((mode === 'visits' || mode === 'pick') && root) return both((l) => (l === 'en' ? `Visits after the search, with ${me} to move: ` : `搜索之后的访问次数，轮到 ${me}：`) + join(free.map((c) => [CELL[c], String(root.n[c])]), l));
	if (!net) return L('The board: the network has not run yet.', '棋盘：网络还没有运行。');
	if (mode === 'prior' && (guess === null || guess === undefined)) return L(`The board with ${me} to move. Tap the cell you think the network likes most.`, `棋盘，轮到 ${me}。点一个你觉得网络最喜欢的格子。`);
	if (mode === 'prior') return both((l) => (l === 'en' ? 'The network’s priors: ' : '网络的先验：') + join(free.map((c) => [CELL[c], fmt3(net.p[c])]), l));
	if (mode === 'value') return L(`The value head says v = ${fmt3(net.v)} for ${me}; perfect play from here gives ${RESULT[solve(b) + 1].en}.`, `价值头说，对 ${me} 来说 v = ${fmt3(net.v)}；从这里起双方都下得完美的话，结果是${RESULT[solve(b) + 1].zh}。`);
	return L(`The board as nine numbers from ${me}’s side: ${net.board.join(', ')}.`, `从 ${me} 这一方看，棋盘是九个数：${net.board.join('，')}。`);
}
/** Screen-reader text of the tree view. */
export function describeTree(p: StageProps): I18n {
	const s = stateOf(p.at), f = p.at.frame, m = f?.metrics ?? {};
	if (!s || !f) return L('The search tree: nothing recorded yet.', '搜索树：还没有记录。');
	const head = L(`The search tree after ${s.N.data[0]} visits: ${s.count} positions.`, `访问 ${s.N.data[0]} 次之后的搜索树：${s.count} 个局面。`);
	const tail =
		f.phase === 'select' ? L(` Simulation ${m.sim} picks ${CELL[m.cell].en}: U = ${fmt3(m.q)} + ${m.c} × ${fmt3(m.p)} × √${m.N} / (1 + ${m.n}) = ${fmt3(m.u)}.`, ` 第 ${m.sim} 次模拟选${CELL[m.cell].zh}：U = ${fmt3(m.q)} + ${m.c} × ${fmt3(m.p)} × √${m.N} / (1 + ${m.n}) = ${fmt3(m.u)}。`)
		: f.phase === 'expand' ? L(` A new position is evaluated: v = ${fmt3(m.v)}.`, ` 评估一个新局面：v = ${fmt3(m.v)}。`)
		: f.phase === 'backup' ? L(` The value ${fmt3(m.v)} goes back up ${m.depth} level(s).`, ` 价值 ${fmt3(m.v)} 往上回传 ${m.depth} 层。`)
		: L('', '');
	return { en: head.en + tail.en, zh: head.zh + tail.zh };
}
/** Screen-reader text of the play view. */
export function describePlay(p: StageProps): I18n {
	const g = gameOf(p.rt);
	if (!g) return L('The game against the search: loading.', '和搜索的对局：正在载入。');
	const cells = (s: number) => legal(g.b.map((v) => (v === s ? 0 : 1))).map((c) => CELL[c]);
	const list = (s: number, l: 'en' | 'zh') => cells(s).map((c) => c[l]).join(l === 'en' ? ', ' : '、') || (l === 'en' ? 'none' : '无');
	const end = g.over ? L(` Game over: ${g.m.result > 0 ? 'you won' : g.m.result < 0 ? 'the search won' : 'a draw'}.`, ` 终局：${g.m.result > 0 ? '你赢了' : g.m.result < 0 ? '搜索赢了' : '平局'}。`) : L('', '');
	return { en: `Your game: X on ${list(1, 'en')}; O on ${list(-1, 'en')}. You play ${markOf(g.you)}.${end.en}`, zh: `你的对局：X 在${list(1, 'zh')}；O 在${list(-1, 'zh')}。你执 ${markOf(g.you)}。${end.zh}` };
}
