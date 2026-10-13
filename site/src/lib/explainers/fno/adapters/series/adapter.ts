// fno: local input adapter 'series' (it replaces the kit's for this package; the kit's Capture.svelte is kept: chips,
// drawing a stretch of the curve, arrow keys). The learner's raw input is a CurveRaw: the 64 starting heights `y` (a
// preset's or drawn ones), the preset it came from (`ic`, only to name it) and the number of kept modes `keep` set by
// the slider on the stage. It becomes the model inputs `input` [64, 1] (float32; another length is stretched to 64 points
// by linear interpolation) and `keep` [1].
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { I, N, PRESETS, presetY } from '../../heat.ts';

export const K = 16; // the modes the network keeps (model.py MODES; claims.test.ts checks it against the spec)
export interface CurveRaw { y?: number[]; ic?: string; keep?: number }
export const curveRaw = (id: string): CurveRaw => ({ ic: id, y: presetY(id) });
export const rawOf = (raw: unknown): CurveRaw => (raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as CurveRaw) : {});
export const keepOf = (raw: unknown) => {
	const v = rawOf(raw).keep;
	return typeof v === 'number' && Number.isFinite(v) ? Math.min(K, Math.max(1, Math.round(v))) : K;
};

/** The 64 starting heights the model reads (empty when nothing is drawn). */
export function valuesOf(raw: unknown): Float32Array {
	const y = (rawOf(raw).y ?? []).map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0));
	if (y.length === N) return Float32Array.from(y);
	if (y.length <= 1) return y.length ? new Float32Array(N).fill(y[0]) : new Float32Array(0);
	const out = new Float32Array(N);
	for (let i = 0; i < N; i++) {
		const x = (i * (y.length - 1)) / (N - 1), a = Math.floor(x), b = Math.min(y.length - 1, a + 1);
		out[i] = y[a] + (x - a) * (y[b] - y[a]);
	}
	return out;
}

/** The preset a raw input still equals, or null when the learner drew on it. */
export function presetOf(raw: unknown): (typeof PRESETS)[number] | null {
	const r = rawOf(raw), p = PRESETS.find((q) => q.id === r.ic);
	if (!p) return null;
	const y = valuesOf(raw), ref = presetY(p.id);
	return y.length === N && y.every((v, i) => Math.abs(v - ref[i]) < 1e-4) ? p : null;
}

const adapter: InputAdapter<CurveRaw> = {
	id: 'series',
	kind: 'series',
	preprocess(raw): Preprocessed {
		const x = valuesOf(raw), keep = Float32Array.of(keepOf(raw));
		const notes = x.length ? [] : [I('There is no curve yet: pick one, or draw one.', '还没有曲线：选一条，或者画一条。')];
		return {
			tensors: { input: { shape: [x.length, 1], data: x }, keep: { shape: [1], data: keep } },
			steps: [
				{ id: 'curve', label: I('The starting temperature: 64 heights around the ring', '起始温度：环上的 64 个高度'), tensor: { shape: [x.length], data: x } },
				{ id: 'keep', label: I('Modes kept (the slider on the stage)', '保留的模式数（舞台上的滑块）'), tensor: { shape: [1], data: keep } }
			],
			empty: !x.length,
			notes
		};
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: I(p.en, p.zh), raw: curveRaw(p.id) })),
	edgeFixtures: () => [
		{ id: 'empty', label: I('No curve', '没有曲线'), raw: { y: [] } },
		{ id: 'flat', label: I('A flat line at 0.5', '一条高 0.5 的平线'), raw: { y: Array(N).fill(0.5) } },
		{ id: 'short', label: I('A drawing of 5 values, stretched to 64', '画了 5 个值，拉伸成 64 个'), raw: { y: [0, 1, -0.5, 0.8, 0] } },
		{ id: 'spike-keep1', label: I('One spike, 1 mode kept', '一个尖峰，只保留 1 个模式'), raw: { y: Array.from({ length: N }, (_, i) => (i === 32 ? 1.5 : 0)), keep: 1 } }
	],
	layer0: { view: 'fno/field', narration: { what: I('A curve is {n|int} numbers.', '一条曲线是 {n|int} 个数。') } }
};

export default adapter;
