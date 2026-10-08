// gan-2d: the real data. The learner's points (the points2d adapter's raw value; classes are ignored: every point is
// "real") or one of the page's presets. The presets are seeded generators written here (TS is the source of truth for
// synthetic data: the twin reads the batches TS drew). Coordinates lie in about [−1, 1]².
import type { I18n } from '../../atlas/types/index.ts';
import type { PointsRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { pointsOf } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { createRng } from '../../atlas/core/rng.ts';

export type PresetId = 'blob' | 'ring' | 'two' | 'eight' | 'smiley';
const r4 = (v: number) => Math.round(v * 1e4) / 1e4;

/** n points around `centres` (taken in turn), Gaussian with spread s; rounded to 4 decimals (they travel as JSON). */
function blobs(n: number, centres: [number, number][], s: number, seed: number): PointsRaw {
	const rng = createRng(seed);
	const points = Array.from({ length: n }, (_, i) => {
		const [cx, cy] = centres[i % centres.length];
		return { x: r4(cx + s * rng.normal()), y: r4(cy + s * rng.normal()) };
	});
	return { points };
}

/** the eight blobs of the mode-collapse step: centres on a circle of radius 0.8 */
export const EIGHT_CENTRES: [number, number][] = Array.from({ length: 8 }, (_, k) => [r4(0.8 * Math.cos((k * Math.PI) / 4)), r4(0.8 * Math.sin((k * Math.PI) / 4))]);
export const EIGHT_SPREAD = 0.05;

export const PRESETS: { id: PresetId; label: I18n; raw: PointsRaw }[] = [
	{ id: 'blob', label: { en: 'One blob', zh: '一团点' }, raw: blobs(300, [[0.35, 0.3]], 0.15, 11) },
	{ id: 'ring', label: { en: 'Ring', zh: '圆环' }, raw: { preset: { name: 'ring', n: 400, seed: 1 } } },
	{ id: 'two', label: { en: 'Two blobs', zh: '两团点' }, raw: blobs(300, [[-0.5, -0.4], [0.5, 0.45]], 0.12, 12) },
	{ id: 'eight', label: { en: 'Eight blobs', zh: '八团点' }, raw: blobs(400, EIGHT_CENTRES, EIGHT_SPREAD, 13) },
	{ id: 'smiley', label: { en: 'Smiley', zh: '笑脸' }, raw: { preset: { name: 'smiley', n: 600, seed: 1 } } }
];
export const presetOf = (id: PresetId): PointsRaw => PRESETS.find((p) => p.id === id)!.raw;

// one Points2D per raw object (the worker and the views keep the same raw object)
const cache = new WeakMap<object, Points2D>();
/** The real points of a run: the source's opt.data (a fixed set, e.g. the eight blobs) else the learner's points. */
export function realOf(raw: unknown, opt?: { data?: unknown } | null): Points2D {
	const r = opt?.data ?? raw;
	if (!r || typeof r !== 'object') return pointsOf(r);
	let p = cache.get(r);
	if (!p) cache.set(r, (p = pointsOf(r)));
	return p;
}
