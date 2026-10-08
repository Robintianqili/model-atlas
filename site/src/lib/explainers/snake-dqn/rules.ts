// snake-dqn: the snake game (pure TS), the same game as pipeline/snake-dqn/game.py (its docstring has the rules).
// Cells are numbered row by row from the top left (cell = y · 9 + x). The snake starts on cells 40, 49, 58 heading up.
// Actions are relative: 0 straight on, 1 turn left, 2 turn right. The k-th food goes to empty cell number
// rng(seed).fork('food' + k).int(empty cells). A crash (wall or body) gives −1 and ends the game, food gives +1.
import { createRng } from '../../atlas/core/rng.ts';

export const L = (en: string, zh: string) => ({ en, zh });
export const SIZE = 9, CELLS = 81, HUNGER = 100;
const DY = [-1, 0, 1, 0], DX = [0, 1, 0, -1], TURN = [0, 3, 1];
export type End = '' | 'wall' | 'self' | 'full';
export interface Game { seed: number; snake: number[]; dir: number; food: number; score: number; moves: number; hunger: number; over: End }
export const ACTION = [L('straight on', '直走'), L('turn left', '左转'), L('turn right', '右转')];
export const ENDS: Record<string, { en: string; zh: string }> = { wall: L('it hit the wall', '撞墙了'), self: L('it ran into its own body', '撞上了自己的身体'), full: L('the board is full', '棋盘满了'), hungry: L(`${HUNGER} moves without food, so the game was stopped`, `${HUNGER} 步没吃到食物，这盘被叫停了`) };

function place(g: Game): void {
	const taken = new Set(g.snake), empty: number[] = [];
	for (let c = 0; c < CELLS; c++) if (!taken.has(c)) empty.push(c);
	if (!empty.length) [g.food, g.over] = [-1, 'full'];
	else g.food = empty[createRng(g.seed).fork(`food${g.score}`).int(empty.length)];
}
export function newGame(seed: number): Game {
	const g: Game = { seed, snake: [40, 49, 58], dir: 0, food: -1, score: 0, moves: 0, hunger: 0, over: '' };
	place(g);
	return g;
}
/** The cell the head moves to with action a (−1: off the board), and the new heading. */
export function nextCell(g: Game, a: number): { c: number; dir: number } {
	const dir = (g.dir + TURN[a]) % 4, y = Math.floor(g.snake[0] / SIZE) + DY[dir], x = (g.snake[0] % SIZE) + DX[dir];
	return { c: y >= 0 && y < SIZE && x >= 0 && x < SIZE ? y * SIZE + x : -1, dir };
}
export const safe = (g: Game, a: number) => { const { c } = nextCell(g, a); return c >= 0 && (c === g.food || !g.snake.slice(0, -1).includes(c)); };
/** One move (pure: a new game), its reward and whether the snake ate. */
export function step(g: Game, a: number): { g: Game; r: number; ate: boolean } {
	const { c, dir } = nextCell(g, a), n: Game = { ...g, snake: g.snake.slice(), dir, moves: g.moves + 1, hunger: g.hunger + 1 };
	if (c < 0 || (c !== g.food && g.snake.slice(0, -1).includes(c))) return { g: { ...n, over: c < 0 ? 'wall' : 'self' }, r: -1, ate: false };
	n.snake.unshift(c);
	if (c !== g.food) {
		n.snake.pop();
		return { g: n, r: 0, ate: false };
	}
	[n.score, n.hunger] = [g.score + 1, 0];
	place(n);
	return { g: n, r: 1, ate: true };
}
/** The game after a list of moves (moves after it ended are dropped). */
export function replay(seed: number, moves: number[]): Game {
	let g = newGame(seed);
	for (const a of moves) if (!g.over) g = step(g, a).g;
	return g;
}
/** The network's input: 3 planes of 9 × 9 (head; body, (L − i) / L on the i-th cell behind the head; food). */
export function encode(g: Game): Float32Array {
	const x = new Float32Array(3 * CELLS), n = g.snake.length;
	x[g.snake[0]] = 1;
	for (let i = 1; i < n; i++) x[CELLS + g.snake[i]] = (n - i) / n;
	if (g.food >= 0) x[2 * CELLS + g.food] = 1;
	return x;
}
export const argmax = (q: ArrayLike<number>) => { let b = 0; for (let i = 1; i < q.length; i++) if (q[i] > q[b]) b = i; return b; };
/** Where a cell is, in words (narration never says "cell 31"). */
export const where = (c: number) => L(`row ${Math.floor(c / SIZE) + 1}, column ${(c % SIZE) + 1}`, `第 ${Math.floor(c / SIZE) + 1} 行第 ${(c % SIZE) + 1} 列`);
