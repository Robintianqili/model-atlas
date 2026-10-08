// hello-xor-2 (copied from hello-xor, DEVKIT §4.1): local input adapter 'xor-point' (design §4.9). The learner's input is one point (x1, x2); the model reads
// it as a [2] tensor named 'input'. Pure TS (the Capture component next to it is the UI part).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';

export type XorPoint = [number, number];

/** Any raw value -> a finite point: [x1, x2] arrays, { x: [x1, x2] } presets; anything else is (0, 0). */
export function toPoint(raw: unknown): XorPoint {
	const a = Array.isArray(raw) ? raw : raw && typeof raw === 'object' && Array.isArray((raw as { x?: unknown }).x) ? (raw as { x: unknown[] }).x : [];
	const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
	return [n(a[0]), n(a[1])];
}

const adapter: InputAdapter<XorPoint> = {
	id: 'xor-point',
	kind: 'points2d',
	preprocess(raw): Preprocessed {
		const p = toPoint(raw);
		const t = { shape: [2], data: Float32Array.from(p) };
		return {
			tensors: { input: t },
			steps: [{ id: 'point', label: { en: 'Your point (x1, x2)', zh: '你的点 (x1, x2)' }, tensor: t }],
			empty: false
		};
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { points?: { id: string; en: string; zh: string; x: XorPoint }[] } | undefined;
		return (f?.points ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: toPoint(p.x) }));
	},
	edgeFixtures: () => [
		{ id: 'origin', label: { en: 'the point (0, 0)', zh: '点 (0, 0)' }, raw: [0, 0] },
		{ id: 'far', label: { en: 'a point far from the training points, (3, -2)', zh: '离训练点很远的点 (3, -2)' }, raw: [3, -2] }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your point is just {n|int} numbers, x1 and x2. That is all the network ever sees.',
				zh: '你的点就是 {n|int} 个数：x1 和 x2。网络看到的只有这些。'
			}
		}
	}
};

export default adapter;
