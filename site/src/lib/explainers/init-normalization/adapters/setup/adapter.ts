// init-normalization: local input adapter 'setup'. The learner's input is the toy network's setup: the spread σ of its
// starting weights (five named initializations as buttons, or a slider), its activation function (tanh or ReLU) and
// an optional normalization after every layer (BatchNorm or LayerNorm). The page reads σ as a [1] tensor named
// 'input'. Pure TS (Capture.svelte next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { INIT_STD, START, STD_RANGE, setupOf, type Setup } from '../../data.ts';

/** Setups the lesson's lint runs every step on (besides the start). */
export const STARTS: { id: string; label: { en: string; zh: string }; raw: Setup }[] = [
	{ id: 'start', label: { en: 'tanh, Xavier, no normalization', zh: 'tanh，Xavier，不归一化' }, raw: START },
	{ id: 'relu-he', label: { en: 'ReLU, He, no normalization', zh: 'ReLU，He，不归一化' }, raw: { act: 'relu', std: INIT_STD.he, norm: 'none' } },
	{ id: 'relu-large-bn', label: { en: 'ReLU, too large, BatchNorm', zh: 'ReLU，过大，BatchNorm' }, raw: { act: 'relu', std: INIT_STD.large, norm: 'batch' } }
];

const adapter: InputAdapter<Setup> = {
	id: 'setup',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const s = setupOf(raw);
		const t = { shape: [1], data: Float64Array.of(s.std) };
		return {
			tensors: { input: t },
			steps: [{ id: 'std', label: { en: 'The spread σ of the starting weights', zh: '初始权重的标准差 σ' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'zero-ln', label: { en: 'every weight 0, LayerNorm', zh: '所有权重为 0，LayerNorm' }, raw: { act: 'tanh', std: 0, norm: 'layer' } },
		{ id: 'max-relu', label: { en: 'the largest σ, ReLU', zh: '最大的 σ，ReLU' }, raw: { act: 'relu', std: STD_RANGE[1], norm: 'none' } },
		{ id: 'min-tanh', label: { en: 'the smallest σ above 0, tanh', zh: '大于 0 的最小 σ，tanh' }, raw: { act: 'tanh', std: STD_RANGE[0], norm: 'none' } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The setting is {n|int} number, the spread σ of every starting weight.',
				zh: '这个设置是 {n|int} 个数：每个初始权重的标准差 σ。'
			}
		}
	}
};

export default adapter;
