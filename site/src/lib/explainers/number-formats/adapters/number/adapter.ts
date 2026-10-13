// number-formats: local input adapter 'number'. The learner's input is one number x ({ x }); the page reads it as a [1]
// tensor named 'input'. Pure TS (the Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { rawOf, START, type Raw } from '../../fmt.ts';

/** Numbers the widget offers as buttons: the start, a third, a typical weight, a small gradient, two large numbers and a
 *  negative one. */
export const X_PRESETS: { x: number; label: { en: string; zh: string } }[] = [
	{ x: START.x, label: { en: '0.1', zh: '0.1' } },
	{ x: 1 / 3, label: { en: '1/3', zh: '1/3' } },
	{ x: 0.02, label: { en: '0.02 (a weight)', zh: '0.02（一个权重）' } },
	{ x: 3e-6, label: { en: '3e−6 (a gradient)', zh: '3e−6（一个梯度）' } },
	{ x: 300, label: { en: '300', zh: '300' } },
	{ x: 70000, label: { en: '70,000', zh: '70,000' } },
	{ x: -2.5, label: { en: '−2.5', zh: '−2.5' } }
];

const adapter: InputAdapter<Raw> = {
	id: 'number',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const t = { shape: [1], data: Float64Array.of(rawOf(raw).x) };
		return { tensors: { input: t }, steps: [{ id: 'number', label: { en: 'Your number x', zh: '你的数 x' }, tensor: t }], empty: false };
	},
	edgeFixtures: () => [
		{ id: 'zero', label: { en: 'x = 0', zh: 'x = 0' }, raw: { x: 0 } },
		{ id: 'tiny', label: { en: 'a tiny x, below fp16’s smallest number', zh: '很小的 x，比 fp16 最小的数还小' }, raw: { x: 1e-9 } },
		{ id: 'huge', label: { en: 'x near fp32’s largest number', zh: '接近 fp32 最大值的 x' }, raw: { x: 3e38 } },
		{ id: 'negative', label: { en: 'a negative x', zh: '负的 x' }, raw: { x: -2.5 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: { what: { en: 'Your input is one number, x.', zh: '你的输入是一个数 x。' } }
	}
};

export default adapter;
