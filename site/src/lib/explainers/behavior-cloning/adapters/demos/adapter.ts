// behavior-cloning: local input adapter 'demos' (InputKind 'game'). The learner's raw input is a BcRaw (world.ts): the
// map, their demonstrations, the gust chance and the probed cell. The network reads one cell as a [2] tensor named
// 'input': its column and row scaled to [−1, 1] (the probed cell; the views show the forward pass there).
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { DEFAULTS, expertDemo, featuresOf, pairsOf, toRaw, worldOf, type BcRaw, type Demo } from '../../world.ts';

const at = (map: string, x: number, y: number) => y * worldOf(map).w + x;
const ex = (map: string, x: number, y: number): Demo => expertDemo(worldOf(map), at(map, x, y));
const P = (id: string, en: string, zh: string, raw: Partial<BcRaw>) => ({ id, label: { en, zh } as I18n, raw: toRaw({ ...DEFAULTS, ...raw }) });
/** Generated presets (no presets file): the default input first. */
export const PRESETS = [
	P('one', 'One demonstration', '一条示范', {}),
	P('recover', 'Plus two recoveries', '再加两条纠正示范', { demos: [...DEFAULTS.demos, ex('field', 3, 4), ex('field', 7, 2)] }),
	P('corner', 'The corner map', '拐角地图', { map: 'corner', demos: [ex('corner', 0, 0)] })
];
export const EMPTY_NOTE: I18n = { en: 'No demonstrations yet: pick a start cell and make some moves.', zh: '还没有示范：选一个起始格子，然后走几步。' };

const adapter: InputAdapter<BcRaw> = {
	id: 'demos',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const g = toRaw(raw), W = worldOf(g.map), n = pairsOf(g).s.length;
		const t = { shape: [2], data: Float32Array.from(featuresOf(W, g.probe)) };
		return { tensors: { input: t }, steps: [{ id: 'cell', label: { en: 'The probed cell: column and row, scaled to −1 to 1', zh: '探查的格子：列号和行号，缩放到 −1 到 1 之间' }, tensor: t }], empty: n === 0, notes: n === 0 ? [EMPTY_NOTE] : [] };
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		P('calm', 'no wind at all', '完全无风', { wind: 0 }),
		P('stormy', 'a gust on half of all moves', '一半的步都有阵风', { wind: 0.5 }),
		P('short', 'a single move next to the goal', '只有终点旁边的一步', { demos: [ex('field', 7, 0)] }),
		P('none', 'no demonstrations', '没有示范', { demos: [] })
	],
	layer0: { view: 'layer.tensor', narration: { what: { en: 'The network sees a cell as {n|int} numbers: its column and row.', zh: '网络把一个格子看成 {n|int} 个数：它的列号和行号。' } } }
};

export default adapter;
