// flow-2d: the kit's points2d adapter (its Capture.svelte is used as it is), with two changes:
//   - presets: smiley, two moons and a ring, 600 points each (the kit's generators; the moons' classes are ignored: a
//     flow model learns one shape). static/models/flow-2d/presets.json lists their ids;
//   - the `input` tensor is [x, y, t] of the pair the lesson follows (flow.ts showcase(): a pair of the first batch,
//     drawn exactly as trainer 'fit' draws it), so the live-model source traces the network on that pair.
import type { I18n, InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { presetRaw, type PointsRaw } from '../../../../atlas/adapters/points2d/adapter.ts';
import type { Points2DName } from '../../../../atlas/data/points2d.ts';
import { along, showcase } from '../../flow.ts';

export const LABELS: Record<string, I18n> = { smiley: { en: 'Smiley', zh: '笑脸' }, moons: { en: 'Two moons', zh: '两个月牙' }, ring: { en: 'Ring', zh: '圆环' } };
export const presetOf = (id: string): PointsRaw => presetRaw(id as Points2DName, { n: 600 });

const adapter: InputAdapter<PointsRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const pre = kit.preprocess(raw, spec, opts);
		const tensors = { ...pre.tensors };
		delete tensors.input;
		const q = showcase(raw)?.pair;
		if (q) tensors.input = { shape: [3], data: Float64Array.of(along(q.x0, q.x1, q.t), along(q.y0, q.y1, q.t), q.t) };
		return { ...pre, tensors };
	},
	presets(assets) {
		const ids = ((assets?.presets as { ids?: string[] } | undefined)?.ids ?? Object.keys(LABELS)).filter((id) => id in LABELS);
		return ids.map((id) => ({ id, label: LABELS[id], raw: presetOf(id) }));
	}
};

export default adapter;
