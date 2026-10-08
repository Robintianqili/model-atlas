// alphazero-ttt: local input adapter 'ttt-board' (InputKind 'game'). The learner's raw input (TttRaw) is a position, given
// as the moves that led to it (X first), plus their guess at step 2 and their game against the search at step 9.
// The position becomes the network's two inputs: `board` (9 numbers from the side of the player to move: +1 mine,
// −1 theirs, 0 empty) and `mask` (0 for an empty cell, −∞ for a taken one; all 0 on a full board, so the numbers stay finite).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { L, boardOf, cleanMoves, isOver, legal, markOf, view, winner } from '../../rules.ts';

export type TttRaw = { moves: number[]; guess?: number | null; game?: number[]; aiFirst?: boolean };
export function rawOf(raw: unknown): TttRaw {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<TttRaw>;
	return { moves: cleanMoves(r.moves), guess: Number.isInteger(r.guess) ? r.guess : null, game: Array.isArray(r.game) ? r.game.filter(Number.isInteger) : [], aiFirst: !!r.aiFirst };
}
export function overNote(moves: number[]) {
	const w = winner(boardOf(moves));
	const res = w ? L(`${markOf(w)} has three in a row`, `${markOf(w)} 已经连成三个`) : L('the board is full: a draw', '棋盘已满：平局');
	return L(`The game on this board is over (${res.en}). The network and the search look at positions where a move is still to be made: undo a move, or pick a position.`, `这盘棋已经结束（${res.zh}）。网络和搜索要看的是还有棋可走的局面：撤回一步，或者换一个局面。`);
}

const adapter: InputAdapter<TttRaw> = {
	id: 'ttt-board',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const r = rawOf(raw), b = boardOf(r.moves), free = legal(b);
		const board = { shape: [9], data: Float32Array.from(view(b)) };
		const mask = { shape: [9], data: Float32Array.from(b, (v) => (v === 0 || !free.length ? 0 : -Infinity)) };
		return {
			tensors: { board, mask },
			steps: [
				{ id: 'board', label: L('The board as the network reads it: +1 for the player to move, −1 for the other player, 0 for empty', '网络读到的棋盘：轮到走棋的一方记 +1，对手记 −1，空格记 0'), tensor: board },
				{ id: 'mask', label: L('The mask: 0 for an empty cell, −∞ for a taken one', '掩码：空格记 0，已占的格子记 −∞'), tensor: mask }
			],
			empty: isOver(b),
			notes: isOver(b) ? [overNote(r.moves)] : []
		};
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { positions?: { id: string; en: string; zh: string; moves: number[] }[] } | undefined;
		return (f?.positions ?? []).map((p) => ({ id: p.id, label: L(p.en, p.zh), raw: { moves: p.moves } }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: L('the empty board: X to move, nine choices', '空棋盘：X 先走，有九种走法'), raw: { moves: [] } },
		{ id: 'one-left', label: L('one empty cell left', '只剩一个空格'), raw: { moves: [0, 4, 8, 2, 6, 3, 5, 7] } },
		{ id: 'two-left', label: L('two empty cells left: O must block', '只剩两个空格：O 必须堵'), raw: { moves: [0, 4, 8, 2, 6, 3, 5] } }
	],
	layer0: { view: 'alphazero-ttt/board', narration: { what: L('The board becomes {n|int} numbers, one per cell.', '棋盘变成 {n|int} 个数，每个格子一个。') } }
};

export default adapter;
