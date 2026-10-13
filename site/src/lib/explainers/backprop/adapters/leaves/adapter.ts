// backprop: local input adapter 'leaves'. The learner's input is the four leaves of the toy graph, { x, w, b, y }: the
// input x, the weight w, the bias b and the target y, set with four sliders. The page reads them as a [4] tensor named
// 'input'. Pure TS (the Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { LEAVES, START, leavesOf, type Leaves } from '../../data.ts';

/** Starting leaves the widget offers as buttons (and lint runs every step on). */
export const STARTS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; raw: Leaves }[] = [
	{ id: 'start', label: { en: 'Start (x = 1.5, w = 0.8, b = −0.5, y = 0.9)', zh: '起点（x = 1.5，w = 0.8，b = −0.5，y = 0.9）' }, short: { en: 'Start', zh: '起点' }, raw: { ...START } },
	{ id: 'wrong-sign', label: { en: 'Wrong sign (w = −1.2)', zh: '符号反了（w = −1.2）' }, short: { en: 'Wrong sign', zh: '符号反了' }, raw: { x: 1, w: -1.2, b: 0.5, y: 0.8 } },
	{ id: 'nearly-right', label: { en: 'Nearly right (a ≈ y)', zh: '几乎答对（a ≈ y）' }, short: { en: 'Nearly right', zh: '几乎答对' }, raw: { x: 1, w: 1, b: 0, y: 0.75 } }
];

const adapter: InputAdapter<Leaves> = {
	id: 'leaves',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const l = leavesOf(raw);
		const t = { shape: [4], data: Float64Array.from(LEAVES, (d) => l[d.id]) };
		return {
			tensors: { input: t },
			steps: [{ id: 'leaves', label: { en: 'The four leaves (x, w, b, y)', zh: '四个叶子节点 (x, w, b, y)' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'corner-low', label: { en: 'every leaf at its lowest', zh: '每个叶子节点都调到最低' }, raw: Object.fromEntries(LEAVES.map((d) => [d.id, d.range[0]])) as Leaves },
		{ id: 'corner-high', label: { en: 'every leaf at its highest', zh: '每个叶子节点都调到最高' }, raw: Object.fromEntries(LEAVES.map((d) => [d.id, d.range[1]])) as Leaves },
		{ id: 'zero-input', label: { en: 'the input x at 0', zh: '输入 x 为 0' }, raw: { ...START, x: 0 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The graph starts from {n|int} numbers, its leaves: the input x, the weight w, the bias b and the target y.',
				zh: '这张图从 {n|int} 个数出发，也就是它的叶子节点：输入 x、权重 w、偏置 b 和目标 y。'
			}
		}
	}
};

export default adapter;
