// behavior-cloning: the shared world (pure TS). The learner's raw input (a map, their demonstrations and the wind
// chance), the grid rules, the expert (the shortest way to the goal), the dataset made from the demonstrations, the
// policy network's forward pass from its weights and the windy rollouts, so every number on the page has one source.
// pipeline/behavior-cloning/reference.py mirrors the rules and the rollouts (independent Python code; atlas parity
// compares the tapes frame by frame).
import type { I18n, Rng } from '../../atlas/types/index.ts';

/** A demonstration: a start cell (index y·w + x) and the moves made from it (0 up, 1 right, 2 down, 3 left). */
export interface Demo { start: number; moves: number[] }
/** The learner's input: the map, their demonstrations, the chance of a gust per move, and the probed cell. */
export interface BcRaw { map: string; demos: Demo[]; wind: number; probe: number }

/** '.' open, '#' wall, 'G' goal, 'P' pit, 'S' start */
export const MAPS: Record<string, { rows: string[]; label: I18n }> = {
	field: { rows: ['........G', '.........', '.........', 'S........', '.........', 'PPPPPPPPP'], label: { en: 'Windy field', zh: '刮风的空地' } },
	corner: { rows: ['S.......', '........', '........', '.......G', '........', 'PPPPPPPP'], label: { en: 'Corner', zh: '拐角' } }
};
export const MAP_IDS = Object.keys(MAPS);
export const MAX_DEMOS = 8, MAX_MOVES = 40;
export const DIRS: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const DIR_NAMES: I18n[] = [{ en: 'up', zh: '上' }, { en: 'right', zh: '右' }, { en: 'down', zh: '下' }, { en: 'left', zh: '左' }];
export const ARROWS = ['↑', '→', '↓', '←'];
export const OPEN = 0, WALL = 1, GOAL = 2, PIT = 3;

export interface World { w: number; h: number; cell: number[]; start: number }
export function worldOf(map: string): World {
	const rows = (MAPS[map] ?? MAPS[MAP_IDS[0]]).rows, cell: number[] = [];
	let start = 0;
	rows.forEach((s, y) => [...s].forEach((c, x) => {
		if (c === 'S') start = y * s.length + x;
		cell.push(c === '#' ? WALL : c === 'G' ? GOAL : c === 'P' ? PIT : OPEN);
	}));
	return { w: rows[0].length, h: rows.length, cell, start };
}
export const isEnd = (W: World, s: number) => W.cell[s] === GOAL || W.cell[s] === PIT;
export const xy = (W: World, s: number): [number, number] => [s % W.w, Math.floor(s / W.w)];

/** One move: a wall or the edge leaves the agent where it is; a goal or a pit ends the episode. */
export function step(W: World, s: number, a: number): number {
	const [x, y] = xy(W, s), nx = x + DIRS[a][0], ny = y + DIRS[a][1];
	return nx < 0 || ny < 0 || nx >= W.w || ny >= W.h || W.cell[ny * W.w + nx] === WALL ? s : ny * W.w + nx;
}

/** Moves to the goal from every cell (breadth-first from the goal; pits and walls are never passed): −1 = no way. */
export function distances(W: World): number[] {
	const d: number[] = W.cell.map((c) => (c === GOAL ? 0 : -1)), q = d.map((v, s) => (v === 0 ? s : -1)).filter((s) => s >= 0);
	while (q.length) {
		const s = q.shift()!;
		for (let a = 0; a < 4; a++) {
			const [x, y] = xy(W, s), nx = x + DIRS[a][0], ny = y + DIRS[a][1], n = ny * W.w + nx;
			if (nx < 0 || ny < 0 || nx >= W.w || ny >= W.h || W.cell[n] !== OPEN || d[n] >= 0) continue;
			d[n] = d[s] + 1;
			q.push(n);
		}
	}
	return d;
}
/** The expert's move: the first of up, right, down, left that brings the agent one move closer to the goal (−1 at an
 *  end cell or where no way leads to the goal). */
export function expertMove(W: World, d: number[], s: number): number {
	if (W.cell[s] !== OPEN || d[s] < 0) return -1;
	for (let a = 0; a < 4; a++) if (d[step(W, s, a)] === d[s] - 1) return a;
	return -1;
}
/** The expert's whole demonstration from a cell (at most MAX_MOVES moves). */
export function expertDemo(W: World, start: number): Demo {
	const d = distances(W), moves: number[] = [];
	for (let s = start; W.cell[s] === OPEN && moves.length < MAX_MOVES; ) {
		const a = expertMove(W, d, s);
		if (a < 0) break;
		moves.push(a), (s = step(W, s, a));
	}
	return { start, moves };
}

const clamp = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);
/** Any value -> a valid BcRaw: a known map, demonstrations that start on an open cell and stop at an end cell. */
export function toRaw(raw: unknown): BcRaw {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<BcRaw>;
	const map = typeof r.map === 'string' && MAPS[r.map] ? r.map : MAP_IDS[0], W = worldOf(map);
	const demos = (Array.isArray(r.demos) ? r.demos : []).slice(0, MAX_DEMOS).flatMap((dm): Demo[] => {
		if (!dm || !Number.isInteger(dm.start) || W.cell[dm.start] !== OPEN) return [];
		const moves: number[] = [];
		for (let s = dm.start, i = 0; i < Math.min(MAX_MOVES, dm.moves?.length ?? 0) && !isEnd(W, s); i++) {
			const a = dm.moves[i];
			if (!Number.isInteger(a) || a < 0 || a > 3) break;
			moves.push(a), (s = step(W, s, a));
		}
		return [{ start: dm.start, moves }];
	});
	const probe = Number.isInteger(r.probe) && W.cell[r.probe as number] === OPEN ? (r.probe as number) : W.start;
	return { map, demos, wind: clamp(r.wind, 0, 0.5, 0.2), probe };
}
export const DEFAULTS: BcRaw = toRaw({ map: 'field', demos: [expertDemo(worldOf('field'), worldOf('field').start)], wind: 0.2 });

/** The network's input for a cell: its column and row scaled to [−1, 1]. */
export const featuresOf = (W: World, s: number): [number, number] => [(2 * (s % W.w)) / (W.w - 1) - 1, (2 * Math.floor(s / W.w)) / (W.h - 1) - 1];

/** The (state, action) pairs of the demonstrations, in order: every cell a demonstration was in, with the move made there. */
export function pairsOf(raw: BcRaw): { s: number[]; a: number[] } {
	const W = worldOf(raw.map), s: number[] = [], a: number[] = [];
	for (const dm of raw.demos) for (let c = dm.start, i = 0; i < dm.moves.length; i++) s.push(c), a.push(dm.moves[i]), (c = step(W, c, dm.moves[i]));
	return { s, a };
}

// ------------------------------------------------------------------ the policy network, from its weights
/** The weights of MLP('net', [2, H, 4], 'tanh') by state_dict name. */
export type Params = Record<string, { shape: number[]; data: ArrayLike<number> }>;
/** The forward pass on one cell: the hidden layer h = tanh(W₀x + b₀), the scores z = W₂h + b₂, softmax p, and the move
 *  with the highest score (ties to the smallest index). Plain loops in a fixed order (the Python reference repeats it). */
export function policyAt(P: Params, x: [number, number]) {
	const W0 = P['net.0.weight'], b0 = P['net.0.bias'].data, W2 = P['net.2.weight'], b2 = P['net.2.bias'].data, H = W0.shape[0];
	const h: number[] = [], z: number[] = [];
	for (let j = 0; j < H; j++) h.push(Math.tanh(W0.data[j * 2] * x[0] + W0.data[j * 2 + 1] * x[1] + b0[j]));
	for (let k = 0; k < 4; k++) {
		let v = b2[k];
		for (let j = 0; j < H; j++) v += W2.data[k * H + j] * h[j];
		z.push(v);
	}
	const mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), sum = e[0] + e[1] + e[2] + e[3];
	return { h, z, p: e.map((v) => v / sum), a: z.indexOf(mx) };
}
/** The policy's move on every cell (−1 on walls, pits and the goal). */
export const policyMap = (W: World, P: Params) => W.cell.map((c, s) => (c === OPEN ? policyAt(P, featuresOf(W, s)).a : -1));

// ------------------------------------------------------------------ windy rollouts
/** The wind of one evaluation: for every episode and move, whether a gust blows the agent one cell down instead of its
 *  move. The same wind blows on every policy (the expert, cloning, DAgger), so their results compare fairly. */
export type Wind = boolean[][];
export function windOf(rng: Rng, episodes: number, T: number, p: number): Wind {
	return Array.from({ length: episodes }, () => Array.from({ length: T }, () => rng.next() < p));
}
/** the gust's direction: down */
export const GUST = 2;
export type Outcome = 1 | -1 | 0; // goal, pit, out of moves
/** Play one episode from the start: `pick(s)` chooses the move, the wind may replace it. */
export function playEpisode(W: World, pick: (s: number) => number, wind: Wind, e: number, T: number) {
	const cells = [W.start], moves: number[] = [], gusts: boolean[] = [];
	let s = W.start;
	for (let t = 0; t < T && !isEnd(W, s); t++) {
		const a = pick(s), sl = wind[e][t], a2 = sl ? GUST : a;
		s = step(W, s, a2);
		moves.push(a), gusts.push(sl), cells.push(s);
	}
	const outcome: Outcome = W.cell[s] === GOAL ? 1 : W.cell[s] === PIT ? -1 : 0;
	return { cells, moves, gusts, outcome };
}
/** How often a policy reaches the goal under a wind: the share of episodes that end at the goal. */
export function successRate(W: World, pick: (s: number) => number, wind: Wind, T: number): number {
	let n = 0;
	for (let e = 0; e < wind.length; e++) n += playEpisode(W, pick, wind, e, T).outcome === 1 ? 1 : 0;
	return n / wind.length;
}
/** The evaluation every number on the page uses: 20 episodes of at most 30 moves, wind from seed 7. */
export const EVAL = { episodes: 20, T: 30, seed: 7 };
