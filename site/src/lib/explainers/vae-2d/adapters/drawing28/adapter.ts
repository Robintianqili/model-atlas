// vae-2d: the kit's drawing28 adapter (its Capture.svelte is used as it is) plus the model's second input, the noise ε:
// two numbers from a standard bell curve, drawn from the kit rng (vae.ts epsOf: the raw input's `draw` picks the draw).
// The drawing's 784 numbers are the kit's own (TinyCNN's preprocessStrokes), so every drawing model sees the same input.
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { type DrawingRaw } from '../../../../atlas/adapters/drawing28/adapter.ts';
import { epsOf } from '../../vae.ts';

const adapter: InputAdapter<DrawingRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const p = kit.preprocess(raw, spec, opts);
		const eps = { shape: [2], data: Float32Array.from(epsOf(raw)) };
		const step = {
			id: 'eps',
			label: { en: 'The noise ε: 2 numbers from a bell curve', zh: '噪声 ε：从钟形曲线里抽出的 2 个数' },
			note: { en: 'The noise ε for step 4: 2 random numbers from a bell curve (“Draw another ε” there changes them).', zh: '第 4 步用的噪声 ε：从钟形曲线里抽出的 2 个随机数（在那一步按“再抽一次 ε”可以换）。' },
			tensor: eps
		};
		return { ...p, tensors: { ...p.tensors, eps }, steps: [...p.steps, step] };
	}
};

export default adapter;
