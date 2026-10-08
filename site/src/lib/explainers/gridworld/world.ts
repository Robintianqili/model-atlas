// gridworld: the shared game model (pure TS). The learner's raw input, the maze and its rules, the slot machines, value
// iteration and the small helpers the processes, the lesson and the views all use, so every number has one source.
// pipeline/gridworld/reference.py mirrors the rules (independent Python code; atlas parity compares the tapes).
import type { I18n } from '../../atlas/types/index.ts';

/** The learner's input: the maze rows ('.' open, '#' wall, 'G' goal +1, 'P' pit −1, 'S' start), their own pulls of
 *  the slot machines (arm indices, in order) and the Q-learning settings. */
export interface GameRaw { rows: string[]; pulls: number[]; gamma: number; epsilon: number; alpha: number; episodes: number }
export const BOARDS: Record<string, string[]> = {
	classic: ['.....G', '.###..', '...#P.', '.#....', 'S#.#..'],
	cliff: ['.......', '.......', '.......', 'SPPPPPG'],
	maze: ['S.#...', '..#.#.', '.##.#.', '....#.', '.####.', '.....G'],
	walled: ['....#G', '.###.#', '...#P.', '.#....', 'S#.#..']
};
export const DEFAULTS: GameRaw = { rows: BOARDS.classic, pulls: [], gamma: 0.9, epsilon: 0.1, alpha: 0.5, episodes: 50 };
export const MAX_PULLS = 40;
/** the two discounts the lesson compares (sources vi05 and vi95) */
export const GAMMAS = [0.5, 0.95];
const clamp = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);

/** Any value -> a valid GameRaw: 2-8 rows and columns, exactly one start, settings clamped. */
export function toRaw(raw: unknown): GameRaw {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<GameRaw>;
	let rows = Array.isArray(r.rows) && r.rows.length >= 2 ? r.rows.slice(0, 8).map((s) => String(s).slice(0, 8)) : DEFAULTS.rows;
	const w = Math.max(2, ...rows.map((s) => s.length));
	let seen = false;
	rows = rows.map((s) => [...s.padEnd(w, '.')].map((c) => (c === 'S' ? (seen ? '.' : ((seen = true), 'S')) : '.#GP'.includes(c) ? c : '.')).join(''));
	if (!seen) rows[rows.length - 1] = 'S' + rows[rows.length - 1].slice(1);
	const pulls = (Array.isArray(r.pulls) ? r.pulls : []).filter((a) => Number.isInteger(a) && a >= 0 && a < PROBS.length).slice(0, MAX_PULLS);
	return { rows, pulls, gamma: clamp(r.gamma, 0, 0.99, DEFAULTS.gamma), epsilon: clamp(r.epsilon, 0, 1, DEFAULTS.epsilon), alpha: clamp(r.alpha, 0.05, 1, DEFAULTS.alpha), episodes: Math.round(clamp(r.episodes, 1, 200, DEFAULTS.episodes)) };
}

// ------------------------------------------------------------------ the maze
export const OPEN = 0, WALL = 1, GOAL = 2, PIT = 3;
/** up, right, down, left as (dx, dy); y grows downward (row 0 is the top row) */
export const DIRS: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const DIR_NAMES: I18n[] = [{ en: 'up', zh: '上' }, { en: 'right', zh: '右' }, { en: 'down', zh: '下' }, { en: 'left', zh: '左' }];
export interface World { w: number; h: number; cell: number[]; start: number }

export function worldOf(raw: unknown): World {
	const { rows } = toRaw(raw);
	const cell: number[] = [];
	let start = 0;
	rows.forEach((s, y) => [...s].forEach((c, x) => {
		if (c === 'S') start = y * s.length + x;
		cell.push(c === '#' ? WALL : c === 'G' ? GOAL : c === 'P' ? PIT : OPEN);
	}));
	return { w: rows[0].length, h: rows.length, cell, start };
}
export const isEnd = (W: World, s: number) => W.cell[s] === GOAL || W.cell[s] === PIT;
export const xy = (W: World, s: number): [number, number] => [s % W.w, Math.floor(s / W.w)];

/** One move: walls and the edge leave the agent where it is; the goal pays +1, the pit −1 (both end the episode). */
export function step(W: World, s: number, a: number): { s2: number; r: number; done: boolean } {
	const [x, y] = xy(W, s), nx = x + DIRS[a][0], ny = y + DIRS[a][1];
	const s2 = nx < 0 || ny < 0 || nx >= W.w || ny >= W.h || W.cell[ny * W.w + nx] === WALL ? s : ny * W.w + nx;
	return { s2, r: W.cell[s2] === GOAL ? 1 : W.cell[s2] === PIT ? -1 : 0, done: isEnd(W, s2) };
}

/** Fewest moves from the start to a goal (breadth-first search), or −1 when no goal can be reached. */
export function goalDistance(W: World): number {
	const d = new Map<number, number>([[W.start, 0]]);
	for (const q = [W.start]; q.length; ) {
		const s = q.shift()!;
		if (W.cell[s] === GOAL) return d.get(s)!;
		if (isEnd(W, s)) continue;
		for (const { s2 } of [0, 1, 2, 3].map((a) => step(W, s, a))) if (!d.has(s2)) q.push(s2), d.set(s2, d.get(s)! + 1);
	}
	return -1;
}

/** Value iteration (the rules known): sweep V(s) <- max_a [r + γ V(s')] over every open cell until the largest change
 *  is below 1e-10 (at most 1,000 sweeps). End cells keep V = 0: entering them pays the reward and ends the episode. */
export function valueIteration(W: World, gamma: number, onSweep?: (V: number[], k: number, delta: number) => void): number[] {
	let V = W.cell.map(() => 0);
	for (let k = 0; k < 1000; k++) {
		const next = V.map((_, s) => (W.cell[s] !== OPEN ? 0 : bestBackup(W, V, s, gamma)));
		const delta = Math.max(...next.map((v, s) => Math.abs(v - V[s])));
		V = next;
		onSweep?.(V, k, delta);
		if (delta < 1e-10) break;
	}
	return V;
}
/** r + γ V(s') for move a from s (V(s') counts as 0 when the move ends the episode), and the best of the four */
export function backup(W: World, V: ArrayLike<number>, s: number, a: number, gamma: number): number {
	const { s2, r, done } = step(W, s, a);
	return r + (done ? 0 : gamma * V[s2]);
}
export const bestBackup = (W: World, V: ArrayLike<number>, s: number, gamma: number) => Math.max(...[0, 1, 2, 3].map((a) => backup(W, V, s, a, gamma)));

/** The arrow of a cell: the one move with the highest Q, or −1 when two or more moves share the highest Q. */
export function arrowOf(q: ArrayLike<number>, s: number): number {
	const v = [0, 1, 2, 3].map((a) => q[s * 4 + a]), mx = Math.max(...v);
	return v.filter((x) => x === mx).length === 1 ? v.indexOf(mx) : -1;
}
/** Follow the arrows from the start: the cells visited until a goal, or null (a loop, a pit or an undecided cell). */
export function greedyPath(W: World, q: ArrayLike<number>): number[] | null {
	const path = [W.start];
	for (let s = W.start; path.length <= W.cell.length; ) {
		const a = arrowOf(q, s);
		if (a < 0) return null;
		s = step(W, s, a).s2;
		if (path.includes(s) || W.cell[s] === PIT) return null;
		path.push(s);
		if (W.cell[s] === GOAL) return path;
	}
	return null;
}
/** Cells with an arrow, and how many of their arrows are a best move by value iteration's V (within 1e-9). */
export function agreement(W: World, q: ArrayLike<number>, V: ArrayLike<number>, gamma: number): { decided: number; agree: number; wrong: number[] } {
	const decided = W.cell.map((c, s) => (c === OPEN && arrowOf(q, s) >= 0 ? s : -1)).filter((s) => s >= 0);
	const wrong = decided.filter((s) => backup(W, V, s, arrowOf(q, s), gamma) < bestBackup(W, V, s, gamma) - 1e-9);
	return { decided: decided.length, agree: decided.length - wrong.length, wrong };
}

// ------------------------------------------------------------------ the slot machines
/** The hidden chance that each machine (arm) pays 1 coin per pull. The lesson reveals them only at the end. */
export const PROBS = [0.3, 0.55, 0.45, 0.7];
export const ARM_NAMES = ['A', 'B', 'C', 'D'];
export const BEST = PROBS.indexOf(Math.max(...PROBS));
/** the seed of every bandit source (explainer.ts): their machines pay the same to every player */
export const BANDIT_SEED = 22;
export const POLICIES = ['greedy', 'eps', 'ucb', 'thompson'] as const;
export type Policy = (typeof POLICIES)[number] | 'you';
