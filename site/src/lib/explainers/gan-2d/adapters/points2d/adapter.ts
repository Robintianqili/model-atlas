// gan-2d: the kit's points2d adapter (the kit's Capture.svelte is used as it is, with options.classes = 1: one kind of
// point, the real data), with the page's own presets: presets.json (pipeline/gan-2d/export.py) lists their ids and
// names; the points come from data.ts (TS is the source of truth for synthetic data).
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import kit, { type PointsRaw } from '../../../../atlas/adapters/points2d/adapter.ts';
import { PRESETS } from '../../data.ts';

type File = { presets?: { id: string; en: string; zh: string }[] };

const adapter: InputAdapter<PointsRaw> = {
	...kit,
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as File | undefined;
		return (f?.presets ?? []).flatMap((p) => PRESETS.filter((q) => q.id === p.id).map((q) => ({ id: q.id, label: { en: p.en, zh: p.zh }, raw: q.raw })));
	}
};

export default adapter;
