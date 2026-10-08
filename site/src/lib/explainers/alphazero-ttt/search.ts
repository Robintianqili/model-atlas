// alphazero-ttt: the network in float64 from the exported float32 weights, and Monte Carlo tree search with PUCT.
// pipeline/alphazero-ttt/mcts.py is the same search in Python (reference.py records it frame by frame for atlas parity).
// Every node stores N (visits), W (total value) and P (prior), all from the side of the player who made the move INTO
// the node, so Q = W / N is how good that move looks to the player choosing it. One simulation:
//   select  from the root, while the node has children, take the child with the largest U = Q + c·P·√N_parent / (1 + N)
//           (Q = 0 for a child never visited; ties: the lowest cell)
//   expand  a finished game gives its result; otherwise the network gives the leaf's children their priors and a value v
//   backup  from the leaf up: N += 1, W += −v at the leaf, +v one level up, −v above, …
// The root is expanded first (its N = 1); after `sims` simulations the move played is the most visited (ties: lowest cell).
import type { GraphModel, Tensor } from '../../atlas/types/index.ts';
import { type Board, isOver, legal, play, resultForMover, view } from './rules.ts';

export type Weights = Record<'W1' | 'b1' | 'W2' | 'b2' | 'Wp' | 'bp' | 'Wv' | 'bv', Tensor> & { H: number };
const memo = new WeakMap<object, Weights>();
export function weightsOf(m: GraphModel): Weights {
	let w = memo.get(m);
	if (!w) {
		const g = (n: string) => m.weights[n] ?? (() => { throw new Error(`alphazero-ttt: no tensor ${n} (run npm run atlas -- export alphazero-ttt)`); })();
		w = { W1: g('fc1.weight'), b1: g('fc1.bias'), W2: g('fc2.weight'), b2: g('fc2.bias'), Wp: g('pol.weight'), bp: g('pol.bias'), Wv: g('val.weight'), bv: g('val.bias'), H: g('fc1.bias').data.length };
		memo.set(m, w);
	}
	return w;
}
function dense(W: Tensor, b: Tensor, x: ArrayLike<number>, relu: boolean): Float64Array {
	const n = b.data.length, k = x.length, y = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let a = 0;
		for (let j = 0; j < k; j++) a += W.data[i * k + j] * x[j];
		a += b.data[i];
		y[i] = relu && a < 0 ? 0 : a;
	}
	return y;
}
/** The network on a board that is not finished: the prior of each cell (0 on taken cells) and the value v. */
export function evaluate(w: Weights, b: Board): { p: Float64Array; v: number } {
	const h = dense(w.W2, w.b2, dense(w.W1, w.b1, view(b), true), true);
	const s = dense(w.Wp, w.bp, h, false), leg = legal(b), p = new Float64Array(9);
	const m = Math.max(...leg.map((i) => s[i]));
	let tot = 0;
	for (const i of leg) tot += p[i] = Math.exp(s[i] - m);
	for (const i of leg) p[i] /= tot;
	return { p, v: Math.tanh(dense(w.Wv, w.bv, h, false)[0]) };
}

export interface Tree { count: number; N: Float64Array; W: Float64Array; P: Float64Array; parent: Float64Array; move: Float64Array; kids: number[][]; board: Board[] }
export const capacity = (sims: number) => 1 + 9 * (sims + 1);
export function newTree(b: Board, sims: number): Tree {
	const n = capacity(sims);
	return { count: 1, N: new Float64Array(n), W: new Float64Array(n), P: new Float64Array(n), parent: new Float64Array(n).fill(-1), move: new Float64Array(n).fill(-1), kids: [[]], board: [b] };
}
/** The PUCT score of child k of node: Q, the exploration bonus c·P·√N/(1 + n), and U = Q + bonus. */
export function score(N: ArrayLike<number>, W: ArrayLike<number>, P: ArrayLike<number>, node: number, k: number, c: number) {
	const n = N[k], q = n > 0 ? W[k] / n : 0, bonus = (c * P[k] * Math.sqrt(N[node])) / (1 + n);
	return { q, bonus, u: q + bonus, n, p: P[k] };
}
type Evaluated = { sim: number; node: number; v: number; terminal: boolean; kids: number[]; depth: number };
export type Ev =
	| ({ kind: 'root' } & Evaluated)
	| ({ kind: 'expand' } & Evaluated)
	| { kind: 'select'; sim: number; node: number; pick: number; depth: number }
	| { kind: 'backup'; sim: number; path: number[]; v: number; from: [number, number][] }
	| { kind: 'choose'; sim: number; pick: number };

function expand(t: Tree, node: number, w: Weights): { v: number; terminal: boolean; kids: number[] } {
	const b = t.board[node];
	if (isOver(b)) return { v: resultForMover(b), terminal: true, kids: [] };
	const { p, v } = evaluate(w, b);
	for (const cell of legal(b)) {
		const k = t.count++;
		t.P[k] = p[cell];
		t.parent[k] = node;
		t.move[k] = cell;
		t.board[k] = play(b, cell);
		t.kids[k] = [];
		t.kids[node].push(k);
	}
	return { v, terminal: false, kids: t.kids[node] };
}
function backup(t: Tree, path: number[], v: number): [number, number][] {
	const from: [number, number][] = path.map((x) => [t.N[x], t.W[x]]);
	for (let i = path.length - 1, s = -1; i >= 0; i--, s = -s) {
		t.N[path[i]] += 1;
		t.W[path[i]] += s * v;
	}
	return from;
}
export function mostVisited(t: Tree, node = 0): number {
	let best = -1;
	for (const k of t.kids[node]) if (best < 0 || t.N[k] > t.N[best]) best = k;
	return best;
}
/** The search as events; the tree is changed in place (the process turns each event into a frame). */
export function* search(t: Tree, w: Weights, sims: number, c: number): Generator<Ev, void> {
	const r = expand(t, 0, w);
	backup(t, [0], r.v);
	yield { kind: 'root', sim: 0, node: 0, depth: 0, ...r };
	for (let sim = 1; sim <= sims; sim++) {
		const path = [0];
		while (t.kids[path[path.length - 1]].length) {
			const node = path[path.length - 1];
			let pick = -1, best = -Infinity;
			for (const k of t.kids[node]) {
				const u = score(t.N, t.W, t.P, node, k, c).u;
				if (u > best) [pick, best] = [k, u];
			}
			yield { kind: 'select', sim, node, pick, depth: path.length - 1 };
			path.push(pick);
		}
		const leaf = path[path.length - 1];
		const e = expand(t, leaf, w);
		yield { kind: 'expand', sim, node: leaf, depth: path.length - 1, ...e };
		yield { kind: 'backup', sim, path, v: e.v, from: backup(t, path, e.v) };
	}
	yield { kind: 'choose', sim: sims, pick: mostVisited(t) };
}
/** The move the search plays from b (no frames): its cell, and the tree. */
export function bestMove(w: Weights, b: Board, sims: number, c: number): { cell: number; t: Tree } {
	const t = newTree(b, sims);
	let pick = -1;
	for (const e of search(t, w, sims, c)) if (e.kind === 'choose') pick = e.pick;
	return { cell: t.move[pick], t };
}
