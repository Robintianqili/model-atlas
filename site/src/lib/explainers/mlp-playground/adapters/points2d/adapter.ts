// mlp-playground: the kit's points2d adapter, with one change: its input step carries no tensor, because the page's
// input section draws that [n, 2] tensor as a heatmap column, 1,600 px tall for 200 points (kit issue in the notes).
// The kit's Capture.svelte is used as it is.
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { type PointsRaw } from '../../../../atlas/adapters/points2d/adapter.ts';

const adapter: InputAdapter<PointsRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const p = kit.preprocess(raw, spec, opts);
		return { ...p, steps: p.steps.map((s) => ({ id: s.id, label: s.label })) };
	}
};

export default adapter;
