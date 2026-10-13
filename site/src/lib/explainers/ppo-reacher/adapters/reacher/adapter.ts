// ppo-reacher: local input adapter 'reacher' (InputKind 'game'). The learner's raw input is a ReachRaw (sim.ts): the target
// of the episode the lesson follows (episode 0 of every batch) and PPO's clip ε (the task step changes it). The tensor
// the page shows is the target [x, y] in metres.
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { DEFAULT_EPS, epsOf, targetOf, type Pt, type ReachRaw } from '../../sim.ts';

const P = (id: string, en: string, zh: string, target: Pt) => ({ id, label: { en, zh } as I18n, raw: { target, eps: DEFAULT_EPS } as ReachRaw });
/** Generated presets (no presets file): the default target first. */
export const PRESETS = [
	P('left', 'Up and to the left', '左上方', [-0.3, 0.6]),
	P('high', 'Straight up, far', '正上方，较远', [0, 0.9]),
	P('right', 'Far right, low', '右边远处，偏低', [0.9, 0.1]),
	P('near', 'Close to the shoulder', '靠近肩膀', [0.25, 0.15])
];

const adapter: InputAdapter<ReachRaw> = {
	id: 'reacher',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const g = targetOf(raw), t = { shape: [2], data: Float32Array.from(g) };
		return { tensors: { input: t }, steps: [{ id: 'target', label: { en: 'Your target [x, y] in m', zh: '你的目标 [x, y]（m）' }, tensor: t }], notes: [epsOf(raw) >= 100 ? { en: 'PPO clip ε: off (no clipping)', zh: 'PPO 截断 ε：关闭（不截断）' } : { en: `PPO clip ε = ${epsOf(raw)}`, zh: `PPO 截断 ε = ${epsOf(raw)}` }], empty: false };
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		P('at-tip', 'the target where the fingertip starts', '目标就在指尖的起点', [0.5, 0.5]),
		P('far-left', 'the far left end of the reach, on the shoulder line', '够得着的最左端，在肩膀那条线上', [-1, 0])
	],
	layer0: { view: 'layer.tensor', narration: { what: { en: 'Your target is {n|int} numbers: x and y in metres.', zh: '你的目标是 {n|int} 个数：以米为单位的 x 和 y。' } } }
};

export default adapter;
