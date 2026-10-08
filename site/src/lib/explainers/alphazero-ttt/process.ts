// alphazero-ttt: two processes (kind C) on the exported network, both in float64 from its float32 weights (search.ts).
//   search  Monte Carlo tree search from the learner's position, one frame per event: start, root (the network on the
//           board), then per simulation select (one frame per level, sub = depth), expand, backup; finally choose.
//   game    a game against the search: the learner's moves (input.you) and the search's replies (sims simulations each).
// Inputs carry the weights as typed arrays (search.ts weightsOf); JSON (atlas parity's --input) carries only the moves,
// and pipeline/alphazero-ttt/reference.py loads the same weights itself. No randomness: the seed changes nothing.
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { EMPTY, type Board, L, boardOf, filled, isOver, play, toMove, winner } from './rules.ts';
import { type Weights, bestMove, newTree, score, search as events } from './search.ts';

export type SearchOpt = { sims: number; c: number };
export type SearchState = { N: Tensor; W: Tensor; P: Tensor; parent: Tensor; move: Tensor; path: number[]; count: number };
export type SearchIn = { moves: number[]; w?: Weights | null };
export type GameIn = { you: number[]; aiFirst?: boolean; w?: Weights | null };
const need = (w: Weights | null | undefined) => w ?? (() => { throw new Error('alphazero-ttt: the network is not loaded'); })();
const T = (label: [string, string], extra: object = {}) => ({ kind: 'tensor' as const, label: L(...label), track: 'changes' as const, ...extra });
/** The process input: the moves, plus the weights (typed arrays) for the run; its JSON is only `o`. */
export function withWeights<O extends object>(o: O, w: Weights | null): O & { w: Weights | null } {
	const x = { ...o, w };
	Object.defineProperty(x, 'toJSON', { value: () => o, enumerable: false });
	return x;
}
function pathTo(parent: ArrayLike<number>, node: number): number[] {
	const p = [node];
	while (parent[p[0]] >= 0) p.unshift(parent[p[0]]);
	return p;
}

export const search: ProcessDef<SearchIn, SearchOpt, SearchState> = {
	id: 'search',
	kind: 'algorithm',
	title: L('Monte Carlo tree search with PUCT', '带 PUCT 的蒙特卡洛树搜索'),
	phases: {
		start: { label: L('The position', '局面'), kind: 'input' },
		root: { label: L('The network looks at the board', '网络看棋盘'), kind: 'forward' },
		select: {
			label: L('Select', '选择'),
			kind: 'act',
			formula: { tex: 'U = Q + c\\,P\\,\\frac{\\sqrt{N}}{1+n} = {{q}} + {{c}} \\times {{p}} \\times \\frac{\\sqrt{ {{N}} }}{1 + {{n}}} = {{u}}', values: (f) => ({ q: f.metrics!.q, c: f.metrics!.c, p: f.metrics!.p, N: f.metrics!.N, n: f.metrics!.n, u: f.metrics!.u }) }
		},
		expand: { label: L('Expand and evaluate', '展开并评估'), kind: 'forward' },
		backup: { label: L('Back up', '回传'), kind: 'update' },
		choose: { label: L('Pick a move', '选一步棋'), kind: 'output' }
	},
	schema: {
		N: T(['Visits N of each node', '每个节点的访问次数 N'], { tol: 'exact' }),
		W: T(['Total value W of each node', '每个节点的总价值 W'], { tol: 1e-12, scale: 'diverging' }),
		P: T(['Prior P of each node', '每个节点的先验 P'], { tol: 1e-12, scale: 'sequential' }),
		parent: T(['Parent of each node', '每个节点的父节点'], { tol: 'exact' }),
		move: T(['The move into each node (cell 0 to 8)', '走到每个节点的那一步（格子 0 到 8）'], { tol: 'exact' }),
		path: { kind: 'json', label: L('This simulation’s path (node ids)', '这次模拟走过的路径（节点编号）'), tol: 'exact' },
		count: { kind: 'int', label: L('Nodes in the tree', '树里的节点数'), tol: 'exact' }
	},
	defaults: { sims: 50, c: 1.5 },
	*run(input, ctx): ProcessRun<SearchState, SearchOpt> {
		const b0 = boardOf(input.moves), { sims, c } = ctx.opt;
		if (isOver(b0)) return;
		const w = need(input.w), t = newTree(b0, sims), len = [t.N.length];
		const ts = { N: { shape: len, data: t.N }, W: { shape: len, data: t.W }, P: { shape: len, data: t.P }, parent: { shape: len, data: t.parent }, move: { shape: len, data: t.move } };
		// the tree's tensors go in the first frame only: later frames carry their edits in `changes`, and the recorder anchors
		// the live tensors at keyframes (sending them in every frame made the main thread hash 18 KB per frame: AK603)
		const st = (path: number[]) => ({ path, count: t.count });
		const born = (kids: number[]) => kids.flatMap((k) => [{ field: 'P', index: [k], from: 0, to: t.P[k] }, { field: 'parent', index: [k], from: -1, to: t.parent[k] }, { field: 'move', index: [k], from: -1, to: t.move[k] }]);
		const top = (kids: number[]) => kids.reduce((a, k) => (t.P[k] > t.P[a] ? k : a), kids[0]);
		const sc = (node: number, k: number) => score(t.N, t.W, t.P, node, k, c);
		yield { phase: 'start', step: 0, state: { ...ts, ...st([0]) }, metrics: { sims, c, mover: toMove(b0), empties: 9 - filled(b0) } };
		let nets = 0;
		for (const e of events(t, w, sims, c)) {
			if (e.kind === 'root') {
				nets++;
				const k = top(e.kids);
				yield { phase: 'root', step: 0, state: st([0]), changes: [{ field: 'N', index: [0], from: 0, to: 1 }, { field: 'W', index: [0], from: 0, to: t.W[0] }, ...born(e.kids)], metrics: { v: e.v, kids: e.kids.length, top: t.move[k], topP: t.P[k], sims, c } };
			} else if (e.kind === 'select') {
				const s = sc(e.node, e.pick);
				const alt = t.kids[e.node].reduce((a, k) => (k !== e.pick && (a < 0 || sc(e.node, k).u > sc(e.node, a).u) ? k : a), -1);
				const a = alt >= 0 ? sc(e.node, alt) : { q: 0, bonus: 0, u: 0, n: 0, p: 0 };
				yield { phase: 'select', step: e.sim, sub: e.depth, state: st(pathTo(t.parent, e.pick)), metrics: { sim: e.sim, depth: e.depth, node: e.node, cell: t.move[e.pick], q: s.q, p: s.p, n: s.n, N: t.N[e.node], bonus: s.bonus, u: s.u, c, kids: t.kids[e.node].length, alt: alt >= 0 ? t.move[alt] : -1, altQ: a.q, altP: a.p, altN: a.n, altBonus: a.bonus, altU: a.u } };
			} else if (e.kind === 'expand') {
				if (!e.terminal) nets++;
				const k = e.kids.length ? top(e.kids) : -1;
				yield { phase: 'expand', step: e.sim, state: st(pathTo(t.parent, e.node)), changes: born(e.kids), metrics: { sim: e.sim, node: e.node, cell: t.move[e.node], depth: e.depth, v: e.v, terminal: e.terminal ? 1 : 0, kids: e.kids.length, top: k >= 0 ? t.move[k] : -1, topP: k >= 0 ? t.P[k] : 0, mover: toMove(t.board[e.node]) } };
			} else if (e.kind === 'backup') {
				const f = e.path[1] ?? 0, [n0, w0] = e.from[1] ?? e.from[0];
				yield { phase: 'backup', step: e.sim, state: st(e.path), changes: e.path.flatMap((x, i) => [{ field: 'N', index: [x], from: e.from[i][0], to: t.N[x] }, { field: 'W', index: [x], from: e.from[i][1], to: t.W[x] }]), metrics: { sim: e.sim, v: e.v, depth: e.path.length - 1, leaf: e.path[e.path.length - 1], first: t.move[f], firstN: t.N[f], firstQ: t.W[f] / t.N[f], firstQ0: n0 > 0 ? w0 / n0 : 0, rootN: t.N[0] } };
			} else {
				const k = e.pick, tp = top(t.kids[0]);
				yield { phase: 'choose', step: e.sim, state: st([0, k]), metrics: { cell: t.move[k], n: t.N[k], share: t.N[k] / (t.N[0] - 1), q: t.W[k] / t.N[k], p: t.P[k], top: t.move[tp], topP: t.P[tp], topN: t.N[tp], agree: k === tp ? 1 : 0, sims, nodes: t.count, nets } };
			}
		}
	}
};

export type GameOpt = { sims: number; c: number };
export type GameState = { board: Tensor };
export const game: ProcessDef<GameIn, GameOpt, GameState> = {
	id: 'game',
	kind: 'algorithm',
	title: L('A game against the search', '和搜索下一盘棋'),
	phases: { start: { label: L('New game', '新的一盘'), kind: 'input' }, you: { label: L('Your move', '你走一步'), kind: 'act' }, ai: { label: L('The search’s move', '搜索走一步'), kind: 'act' }, end: { label: L('Game over', '终局'), kind: 'output' } },
	schema: { board: { kind: 'tensor', label: L('The board (1 = X, −1 = O)', '棋盘（1 = X，−1 = O）'), shape: [9], tol: 'exact' } },
	defaults: { sims: 50, c: 1.5 },
	*run(input, ctx): ProcessRun<GameState, GameOpt> {
		const w = need(input.w), you = input.aiFirst ? -1 : 1, mine = Array.isArray(input.you) ? input.you : [];
		const snap = (b: Board) => ({ board: { shape: [9], data: Float64Array.from(b) } });
		let b = EMPTY, i = 0;
		yield { phase: 'start', step: 0, state: snap(b), metrics: { you, sims: ctx.opt.sims } };
		while (!isOver(b)) {
			if (toMove(b) === you) {
				while (i < mine.length && !(Number.isInteger(mine[i]) && mine[i] >= 0 && mine[i] < 9 && b[mine[i]] === 0)) i++;
				if (i >= mine.length) return;
				const cell = mine[i++];
				b = play(b, cell);
				yield { phase: 'you', step: filled(b), state: snap(b), metrics: { cell, you } };
			} else {
				const { cell, t } = bestMove(w, b, ctx.opt.sims, ctx.opt.c), k = t.kids[0].find((x) => t.move[x] === cell)!;
				b = play(b, cell);
				yield { phase: 'ai', step: filled(b), state: snap(b), metrics: { cell, n: t.N[k], sims: ctx.opt.sims, you } };
			}
		}
		yield { phase: 'end', step: filled(b), state: snap(b), metrics: { winner: winner(b), you, result: winner(b) * you } };
	}
};
