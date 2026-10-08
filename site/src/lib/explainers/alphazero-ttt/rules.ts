// alphazero-ttt: tic-tac-toe rules (pure TS), the board as the network reads it, and an exact solver.
// A board is 9 numbers, cells 0..8 row by row: 1 = X, −1 = O, 0 = empty. X moves first. Mirrors pipeline/alphazero-ttt/game.py.
import type { I18n } from '../../atlas/types/index.ts';

export type Board = number[];
export const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
export const EMPTY: Board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
export const L = (en: string, zh: string) => ({ en, zh });
export const MARK = ['O', '', 'X'];
export const markOf = (s: number) => MARK[s + 1];

export const filled = (b: Board) => b.filter((v) => v !== 0).length;
export const toMove = (b: Board) => (filled(b) % 2 === 0 ? 1 : -1);
export function winLine(b: Board): number[] | null {
	for (const l of LINES) if (b[l[0]] !== 0 && b[l[0]] === b[l[1]] && b[l[1]] === b[l[2]]) return l;
	return null;
}
export const winner = (b: Board) => { const l = winLine(b); return l ? b[l[0]] : 0; };
export const isOver = (b: Board) => winner(b) !== 0 || filled(b) === 9;
export const legal = (b: Board) => b.flatMap((v, i) => (v === 0 ? [i] : []));
export function play(b: Board, cell: number): Board {
	const n = b.slice();
	n[cell] = toMove(b);
	return n;
}
/** At a finished board: the result for the player to move there (−1 lost, 0 draw). */
export const resultForMover = (b: Board) => { const w = winner(b); return w === 0 ? 0 : w === toMove(b) ? 1 : -1; };
/** The network's input: the board from the side of the player to move (+1 mine, −1 theirs, 0 empty). */
export const view = (b: Board) => b.map((v) => v * toMove(b) || 0);

/** The legal prefix of a move list: moves on taken cells and moves after the game ended are dropped. */
export function cleanMoves(moves: unknown): number[] {
	const out: number[] = [];
	let b = EMPTY;
	for (const c of Array.isArray(moves) ? moves : []) if (Number.isInteger(c) && c >= 0 && c < 9 && b[c] === 0 && !isOver(b)) { out.push(c); b = play(b, c); }
	return out;
}
export const boardOf = (moves: number[]) => cleanMoves(moves).reduce(play, EMPTY);

const memo = new Map<string, number>();
/** The game's value for the player to move with perfect play from both sides: 1 win, 0 draw, −1 loss. */
export function solve(b: Board): number {
	const k = b.join();
	let v = memo.get(k);
	if (v === undefined) {
		v = (isOver(b) ? resultForMover(b) : Math.max(...legal(b).map((c) => -solve(play(b, c))))) || 0;
		memo.set(k, v);
	}
	return v;
}
/** The value of each legal move for the player making it (perfect play afterwards). */
export const moveValue = (b: Board, cell: number) => -solve(play(b, cell)) || 0;
/** Why an empty cell is urgent on an unfinished board: 1 it wins at once, −1 it blocks the other player's three in a row, 0 neither. */
export function urgent(b: Board, cell: number): number {
	if (b[cell] !== 0 || isOver(b)) return 0;
	if (winner(play(b, cell)) !== 0) return 1;
	const o = b.slice();
	o[cell] = -toMove(b);
	return winner(o) !== 0 ? -1 : 0;
}

const lineMemo = new Map<string, number>();
/** Lines of play (complete games) from b. */
export function lines(b: Board): number {
	const k = b.join();
	let v = lineMemo.get(k);
	if (v === undefined) {
		v = isOver(b) ? 1 : legal(b).reduce((s, c) => s + lines(play(b, c)), 0);
		lineMemo.set(k, v);
	}
	return v;
}
/** Positions in the full game tree from b, every line counted separately (b included). */
export function treeSize(b: Board): number {
	const k = 't' + b.join();
	let v = lineMemo.get(k);
	if (v === undefined) {
		v = isOver(b) ? 1 : 1 + legal(b).reduce((s, c) => s + treeSize(play(b, c)), 0);
		lineMemo.set(k, v);
	}
	return v;
}
export const RESULT: I18n[] = [L('a loss', '输'), L('a draw', '平局'), L('a win', '赢')];
/** Cell names (narration never says "cell 4"). */
export const CELL: I18n[] = [L('the top-left corner', '左上角'), L('the top middle', '上边中间'), L('the top-right corner', '右上角'), L('the middle left', '左边中间'), L('the centre', '正中间'), L('the middle right', '右边中间'), L('the bottom-left corner', '左下角'), L('the bottom middle', '下边中间'), L('the bottom-right corner', '右下角')];
/** One arrow per cell, pointing from the centre (the tree view's labels). */
export const ARROW = ['↖', '↑', '↗', '←', '•', '→', '↙', '↓', '↘'];
