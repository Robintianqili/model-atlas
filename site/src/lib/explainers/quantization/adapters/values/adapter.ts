// quantization: local input adapter 'values'. The learner's input is one number x (for the bit-level format widgets) and
// the eight numbers of the rounding ledger ({ x, vals }). The page reads them as a [9] tensor named 'input': x, then the
// eight values. Pure TS (the Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { rawOf, START, type Raw } from '../../quant.ts';

/** Sets of eight values the widget offers as buttons (and lint runs every step on). */
export const PRESETS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; vals: number[] }[] = [
	{ id: 'weights', label: { en: 'Weight-like (small, both signs)', zh: '像权重（数值小，有正有负）' }, short: { en: 'Weights', zh: '权重' }, vals: START.vals },
	{ id: 'outlier', label: { en: 'One outlier', zh: '一个离群值' }, short: { en: 'Outlier', zh: '离群值' }, vals: [0.31, -0.12, 0.05, -0.47, 0.22, 0.09, -0.26, 6] },
	{ id: 'positive', label: { en: 'All positive (after a ReLU)', zh: '全是正数（ReLU 之后）' }, short: { en: 'Positive', zh: '正数' }, vals: [0.31, 0.02, 0.05, 0.47, 0.22, 0.09, 0.6, 0.4] }
];
/** Numbers the format widget offers as buttons. */
export const X_PRESETS = [0.1, 1 / 3, 70000, 0.00001];

const adapter: InputAdapter<Raw> = {
	id: 'values',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const r = rawOf(raw);
		const t = { shape: [9], data: Float64Array.of(r.x, ...r.vals) };
		return {
			tensors: { input: t },
			steps: [{ id: 'values', label: { en: 'Your number x and your eight values', zh: '你的数 x 和你的八个值' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'zeros', label: { en: 'eight zeros and x = 0', zh: '八个 0，x = 0' }, raw: { x: 0, vals: [0, 0, 0, 0, 0, 0, 0, 0] } },
		{ id: 'equal', label: { en: 'eight equal values, a large x', zh: '八个相等的值，x 很大' }, raw: { x: 70000, vals: [2, 2, 2, 2, 2, 2, 2, 2] } },
		{ id: 'tiny', label: { en: 'tiny values, a tiny x', zh: '非常小的值，x 也非常小' }, raw: { x: 1e-6, vals: [1e-4, -2e-4, 3e-4, -5e-5, 1e-5, 2e-4, -3e-4, 4e-4] } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your input is {n|int} numbers: x for the format widgets, then the eight values of the ledger.',
				zh: '你的输入是 {n|int} 个数：先是格式演示用的 x，然后是账本里的八个值。'
			}
		}
	}
};

export default adapter;
