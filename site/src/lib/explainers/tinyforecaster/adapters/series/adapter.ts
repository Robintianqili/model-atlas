// tinyforecaster: the kit's series adapter (its Capture.svelte is used as it is) with two changes:
//   - preprocess adds `window` [24, 1]: the last training window (data.ts), the input of the 'live-model' graph, and
//     shows that window (not the whole series) as "what the model sees";
//   - presets come from static/models/tinyforecaster/presets.json (pipeline/tinyforecaster/export.py): the five
//     forecasting series, without the kit's damped oscillator (observation points) and flat line.
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { SERIES_PRESETS, seriesPresetRaw, type SeriesPresetName, type SeriesRaw } from '../../../../atlas/adapters/series/adapter.ts';
import { dataOf, L } from '../../data.ts';

type PresetsFile = { presets?: { id: SeriesPresetName; seed?: number }[] };

const adapter: InputAdapter<SeriesRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const p = kit.preprocess(raw, spec, opts);
		if (p.empty) return p;
		const d = dataOf(raw);
		const k = d.windows - 1;
		const win = d.y.slice(k, k + L);
		// the input section draws the steps: the window the network reads (lesson steps 3 to 5), not the whole series
		// (the kit draws a step's tensor without its label, so the note, shown under it, says what it is)
		const say = { en: 'The last training window: the 24 values the network reads in lesson steps 3 to 5, one colored cell per value.', zh: '最后一个训练窗口：网络在课程第 3 到第 5 步读的 24 个值，每个值一个色块。' };
		const steps = [{ id: 'window', label: say, note: say, tensor: { shape: [L], data: win } }];
		return { ...p, steps, tensors: { ...p.tensors, window: { shape: [L, 1], data: win } } };
	},
	presets(assets) {
		const list = (assets.presets as PresetsFile | undefined)?.presets ?? [];
		return list.flatMap((p) => {
			const s = SERIES_PRESETS.find((x) => x.id === p.id);
			return s ? [{ id: p.id, label: s.label, raw: seriesPresetRaw(p.id, { seed: p.seed ?? 1 }) }] : [];
		});
	}
};

export default adapter;
