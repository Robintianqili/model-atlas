// snake-dqn: local input adapter 'snake-board' (InputKind 'game'). The learner's raw input (SnakeRaw) is a position,
// given as a food seed and the moves played from the start (0 straight on, 1 left, 2 right), plus `draw` (which random
// draw the ε-greedy step uses) and `game` (the learner's own game at step 10, from the start of the same seed).
// The position becomes the network's input `board`: 3 planes of 9 × 9 (head, body, food; rules.ts encode).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { CELLS, ENDS, L, encode, replay } from '../../rules.ts';

export type SnakeRaw = { seed: number; moves: number[]; draw?: number; game?: number[] };
const acts = (x: unknown) => (Array.isArray(x) ? x.filter((a) => a === 0 || a === 1 || a === 2) : []);
export function rawOf(raw: unknown): Required<SnakeRaw> {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<SnakeRaw>;
	const n = (v: unknown, d: number) => (Number.isInteger(v) && (v as number) >= 0 ? (v as number) : d);
	return { seed: n(r.seed, 1), moves: acts(r.moves), draw: n(r.draw, 0), game: acts(r.game) };
}
export const posOf = (raw: unknown) => { const r = rawOf(raw); return replay(r.seed, r.moves); };
export const crashNote = (why: string) => L(`On this board the snake has crashed (${ENDS[why]?.en ?? why}), so there is no move left to choose. Undo a move, or pick a position.`, `在这块棋盘上，蛇已经撞了（${ENDS[why]?.zh ?? why}），没有下一步可选了。撤回一步，或者换一个局面。`);
const PLANES = [L('Plane 1, the head: 1 where the head is', '第 1 层，蛇头：蛇头所在的格子记 1'), L('Plane 2, the body: brightest right behind the head, dimmest at the tail', '第 2 层，身体：紧跟蛇头的一节最亮，尾巴最暗'), L('Plane 3, the food: 1 where the food is', '第 3 层，食物：食物所在的格子记 1')];

const adapter: InputAdapter<SnakeRaw> = {
	id: 'snake-board',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const g = posOf(raw), x = encode(g);
		return {
			tensors: { board: { shape: [3, 9, 9], data: x } },
			steps: PLANES.map((label, k) => ({ id: `plane${k}`, label, tensor: { shape: [9, 9], data: x.slice(k * CELLS, (k + 1) * CELLS) } })),
			empty: !!g.over,
			notes: g.over ? [crashNote(g.over)] : []
		};
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { positions?: { id: string; en: string; zh: string; seed: number; moves: number[] }[] } | undefined;
		return (f?.positions ?? []).map((p) => ({ id: p.id, label: L(p.en, p.zh), raw: { seed: p.seed, moves: p.moves } }));
	},
	edgeFixtures: () => [
		{ id: 'start', label: L('the start: length 3, heading up', '开局：长度 3，朝上'), raw: { seed: 3, moves: [] } },
		{ id: 'wall-ahead', label: L('the head next to the top wall', '蛇头贴着上面的墙'), raw: { seed: 3, moves: [0, 0, 0, 0] } },
		{ id: 'crashed', label: L('a crashed snake', '已经撞墙的蛇'), raw: { seed: 3, moves: [0, 0, 0, 0, 0] } }
	],
	layer0: { view: 'snake-dqn/step', narration: { what: L('The board becomes {n|int} numbers: three planes of 9 × 9.', '棋盘变成 {n|int} 个数：三层，每层 9 × 9。') } }
};

export default adapter;
