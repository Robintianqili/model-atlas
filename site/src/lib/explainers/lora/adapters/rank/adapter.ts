// lora: local input adapter 'rank'. The learner's input is LoRA's rank r, a whole number from 1 to 8 ({ r }), set with a
// slider or a button. The page reads it as a [1] tensor named 'input'. Pure TS (the Capture next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { R_MAX, R_MIN, rankOf } from '../../data.ts';

export type Rank = { r: number };

/** Ranks the widget offers as buttons (and lint runs every step on). */
export const STARTS: { id: string; label: { en: string; zh: string }; raw: Rank }[] = [
	{ id: 'r1', label: { en: 'Rank 1', zh: '秩 1' }, raw: { r: 1 } },
	{ id: 'r2', label: { en: 'Rank 2', zh: '秩 2' }, raw: { r: 2 } },
	{ id: 'r4', label: { en: 'Rank 4', zh: '秩 4' }, raw: { r: 4 } },
	{ id: 'r8', label: { en: 'Rank 8', zh: '秩 8' }, raw: { r: 8 } }
];

const adapter: InputAdapter<Rank> = {
	id: 'rank',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const t = { shape: [1], data: Float64Array.of(rankOf(raw)) };
		return { tensors: { input: t }, steps: [{ id: 'rank', label: { en: 'Your rank r', zh: '你选的秩 r' }, tensor: t }], empty: false };
	},
	edgeFixtures: () => [
		{ id: 'r3', label: { en: 'rank 3', zh: '秩 3' }, raw: { r: 3 } },
		{ id: 'too-big', label: { en: 'a rank above the slider (clamped to 8)', zh: '超出滑块的秩（截到 8）' }, raw: { r: R_MAX + 4 } },
		{ id: 'too-small', label: { en: 'rank 0 (clamped to 1)', zh: '秩 0（截到 1）' }, raw: { r: R_MIN - 1 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your input is one number, the rank r: how many columns B has and how many rows A has.',
				zh: '你的输入只有一个数，也就是秩 r：B 有几列，A 有几行。'
			}
		}
	}
};

export default adapter;
