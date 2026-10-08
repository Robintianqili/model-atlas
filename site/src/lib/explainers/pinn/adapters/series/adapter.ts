// pinn: the kit's 'series' adapter (and its Capture.svelte, used as it is) with three changes: the presets are only
// damped oscillators (presets.json: seed and noise of the kit's 'oscillator' preset), the edge fixtures are inputs a PINN
// must survive (no points, one point, points spread over the whole time), and the input step names what the networks
// see, the points, without a tensor (the page would draw the [k, 2] list as a small unlabeled heatmap; the Capture
// already draws the points).
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { OSCILLATOR, seriesPresetRaw, type SeriesRaw } from '../../../../atlas/adapters/series/adapter.ts';
import { dampedOscillator } from '../../../../atlas/data/series.ts';

const OSC = seriesPresetRaw('oscillator');
type PresetFile = { presets?: { id: string; en: string; zh: string; seed: number; noise: number }[] };

const adapter: InputAdapter<SeriesRaw> = {
	...kit,
	preprocess(raw, spec, opts) {
		const p = kit.preprocess(raw, spec, opts);
		const obs = p.tensors.obs as { shape: number[]; data: Float64Array } | undefined;
		const note = { en: 'There are no points: the networks have no data to learn from. Press the plot to add some.', zh: '没有数据点：网络没有数据可学。在图上按一下就能添加。' };
		return { ...p, steps: [{ id: 'points', label: { en: 'What the networks see: your points, one (t, u) pair each', zh: '网络看到的：你的数据点，每个是一对 (t, u)' } }], notes: obs ? [] : [note] };
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as PresetFile | undefined;
		return (f?.presets ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: seriesPresetRaw('oscillator', { seed: p.seed, noise: p.noise }) }));
	},
	edgeFixtures: () => [
		{ id: 'no-points', label: { en: 'No points', zh: '没有数据点' }, raw: { ...OSC, points: [] } },
		{ id: 'one-point', label: { en: 'One point', zh: '一个数据点' }, raw: { ...OSC, points: [{ t: 0.2, y: -0.45 }] } },
		{ id: 'spread', label: { en: 'Points over the whole time', zh: '数据点铺满整段时间' }, raw: { ...OSC, points: [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95].map((t) => ({ t, y: dampedOscillator([t], OSCILLATOR)[0] })) } }
	]
};

export default adapter;
