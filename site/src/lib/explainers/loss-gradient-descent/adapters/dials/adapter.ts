// loss-gradient-descent: local input adapter 'dials'. The learner's input is the neuron's two numbers, the weight w
// and the bias b, set with two dials ({ w, b }). The page reads them as a [2] tensor named 'input'. Pure TS (the
// Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { B_RANGE, W_RANGE, dialsOf, type Dials } from '../../data.ts';

/** Starting dials the widget offers as buttons (and lint runs every step on). */
export const STARTS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; raw: Dials }[] = [
	{ id: 'flat', label: { en: 'Flat line (w = 0, b = 0)', zh: '水平线（w = 0，b = 0）' }, short: { en: 'Flat', zh: '水平' }, raw: { w: 0, b: 0 } },
	{ id: 'steep', label: { en: 'Too steep (w = 3, b = −1)', zh: '太陡（w = 3，b = −1）' }, short: { en: 'Too steep', zh: '太陡' }, raw: { w: 3, b: -1 } },
	{ id: 'downhill', label: { en: 'Wrong way (w = −1, b = 1)', zh: '方向反了（w = −1，b = 1）' }, short: { en: 'Wrong way', zh: '方向反了' }, raw: { w: -1, b: 1 } }
];

const adapter: InputAdapter<Dials> = {
	id: 'dials',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const d = dialsOf(raw);
		const t = { shape: [2], data: Float64Array.of(d.w, d.b) };
		return {
			tensors: { input: t },
			steps: [{ id: 'dials', label: { en: 'Your dials (w, b)', zh: '你的旋钮 (w, b)' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'corner-low', label: { en: 'both dials at their lowest', zh: '两个旋钮都调到最低' }, raw: { w: W_RANGE[0], b: B_RANGE[0] } },
		{ id: 'corner-high', label: { en: 'both dials at their highest', zh: '两个旋钮都调到最高' }, raw: { w: W_RANGE[1], b: B_RANGE[1] } },
		{ id: 'close', label: { en: 'dials close to the best line', zh: '旋钮接近最佳直线' }, raw: { w: 1.2, b: 0.4 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The neuron is just {n|int} numbers, the weight w and the bias b. Its answer for an input x is w·x + b.',
				zh: '这个神经元只有 {n|int} 个数：权重 w 和偏置 b。它对输入 x 的回答是 w·x + b。'
			}
		}
	}
};

export default adapter;
