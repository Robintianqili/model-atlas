// gridworld: local input adapter 'gridworld' (InputKind 'game'). The learner's raw input is a GameRaw (world.ts): the
// maze rows, their own slot-machine pulls and the Q-learning settings. The board becomes an [H, W] tensor of cell codes.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { DEFAULTS, goalDistance, toRaw, worldOf, type GameRaw } from '../../world.ts';

export const UNREACHABLE = { en: 'The goal cannot be reached from the start: no path leads there, so no episode can ever end at the goal.', zh: '从起点到不了终点：没有路通向那里，所以没有任何回合能在终点结束。' };

const adapter: InputAdapter<GameRaw> = {
	id: 'gridworld',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const W = worldOf(raw);
		const board = { shape: [W.h, W.w], data: Float32Array.from(W.cell) };
		return { tensors: { board }, steps: [{ id: 'board', label: { en: 'Your maze (0 open, 1 wall, 2 goal, 3 pit)', zh: '你的迷宫（0 空地，1 墙，2 终点，3 陷阱）' } }], empty: false, notes: goalDistance(W) < 0 ? [UNREACHABLE] : [] };
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { boards?: { id: string; en: string; zh: string; rows: string[] }[] } | undefined;
		return (f?.boards ?? []).map((b) => ({ id: b.id, label: { en: b.en, zh: b.zh }, raw: toRaw({ ...DEFAULTS, rows: b.rows }) }));
	},
	edgeFixtures: () => [
		{ id: 'pulled', label: { en: 'the classic maze after twelve pulls of your own', zh: '经典迷宫，加上你自己拉的十二次' }, raw: toRaw({ ...DEFAULTS, pulls: [0, 1, 2, 3, 3, 1, 3, 2, 3, 1, 3, 3] }) },
		{ id: 'no-goal', label: { en: 'a maze with a pit and no goal', zh: '只有陷阱、没有终点的迷宫' }, raw: toRaw({ ...DEFAULTS, rows: ['....', '.P..', 'S...'] }) },
		{ id: 'tiny', label: { en: 'the smallest maze: two by two', zh: '最小的迷宫：二乘二' }, raw: toRaw({ ...DEFAULTS, rows: ['S.', '.G'] }) }
	],
	layer0: { view: 'gridworld/grid-board', narration: { what: { en: 'Your maze is a grid of {n|int} cells.', zh: '你的迷宫是一个有 {n|int} 个格子的网格。' } } }
};

export default adapter;
