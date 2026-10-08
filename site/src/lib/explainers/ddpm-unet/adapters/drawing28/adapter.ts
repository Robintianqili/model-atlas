// ddpm-unet: the kit's drawing28 adapter (its Capture.svelte is used as it is) plus the model's two other inputs: the
// noise ε (784 numbers from a bell curve, ddpm.ts epsNoise: the same for every drawing) and the timestep t (the raw
// input's `t`, default T_TRACE). The drawing's 784 numbers are the kit's own (TinyCNN's preprocessStrokes).
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { type DrawingRaw } from '../../../../atlas/adapters/drawing28/adapter.ts';
import { SIDE, T, epsNoise, tOf } from '../../ddpm.ts';

let EPS: Float32Array | null = null;

const adapter: InputAdapter<DrawingRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const p = kit.preprocess(raw, spec, opts);
		const eps = { shape: [1, SIDE, SIDE], data: (EPS ??= epsNoise()) };
		const t = { shape: [1], data: Float32Array.of(tOf(raw)) };
		const steps = [
			{ id: 'eps', label: { en: 'The noise ε: 784 numbers from a bell curve', zh: '噪声 ε：从钟形曲线里抽出的 784 个数' }, tensor: eps },
			{ id: 't', label: { en: 'The step t: how much noise to mix in', zh: '步数 t：要混进多少噪声' }, note: { en: `t = ${t.data[0]} of ${T} steps.`, zh: `t = ${t.data[0]}（共 ${T} 步）。` } }
		];
		return { ...p, tensors: { ...p.tensors, eps, t }, steps: [...p.steps, ...steps] };
	}
};

export default adapter;
