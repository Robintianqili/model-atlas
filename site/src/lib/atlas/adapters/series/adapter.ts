// Input adapter 'series' (design §4.9; catalog InputKind 'series'): a 1-D series in time order (the forecaster) and,
// optionally, a few observation points (the PINN). Captured with a curve the learner draws or with draggable points.
// Pure TS: a trainer in a worker reads the same raw value with seriesOf().
//
// Raw value (plain data):
//   { preset?: { name, n?, seed?, noise? },   a generator from atlas/data/series.ts (or the damped oscillator)
//     y?: number[], t?: number[],             values the learner drew (y replaces the preset's values; t defaults to 0..n−1)
//     points?: [{ t, y }] }                   observation points (sorted by t)
// Preprocessed tensors: t [n] and y [n]; obs [k, 2] (t, y per row) when there are observation points; `input` [1] = one
// time value (options.at, default the last t) so a 'live-model' source can trace a single input.
import type { I18n, Tensor } from '../../types/core.ts';
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';
import { dampedOscillator, linspace, observations, randomWalk, seasonal, sine, square, trend } from '../../data/series.ts';
import { createRng } from '../../core/rng.ts';

export type SeriesPresetName = 'sine' | 'seasonal' | 'trend' | 'square' | 'randomWalk' | 'oscillator' | 'flat';

export interface SeriesRaw {
	preset?: { name: SeriesPresetName; n?: number; seed?: number; noise?: number };
	y?: number[];
	t?: number[];
	points?: { t: number; y: number }[];
}

export interface SeriesValue {
	name: string;
	t: Float64Array;
	y: Float64Array;
	points: { t: number; y: number }[];
}

export interface SeriesPreset {
	id: SeriesPresetName;
	label: I18n;
	n: number;
}

export const SERIES_PRESETS: SeriesPreset[] = [
	{ id: 'sine', label: { en: 'Sine wave', zh: '正弦波' }, n: 96 },
	{ id: 'seasonal', label: { en: 'Trend + seasons + noise', zh: '趋势 + 季节 + 噪声' }, n: 120 },
	{ id: 'square', label: { en: 'Square wave', zh: '方波' }, n: 96 },
	{ id: 'trend', label: { en: 'Straight trend', zh: '直线趋势' }, n: 96 },
	{ id: 'randomWalk', label: { en: 'Random walk', zh: '随机游走' }, n: 120 },
	{ id: 'oscillator', label: { en: 'Damped oscillator + points', zh: '阻尼振子 + 观测点' }, n: 200 },
	{ id: 'flat', label: { en: 'Flat line', zh: '一条平线' }, n: 48 }
];

export const MAX_VALUES = 2000;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Are two raw values the same series (preset name, length, seed and noise, drawn values and points)? A Capture marks
 *  the preset chip that equals the learner's input with it (K1: it compared the preset name only, so three presets of
 *  one generator all looked picked). */
export function sameSeries(a: unknown, b: unknown): boolean {
	const canon = (r: unknown) => {
		const v = (r && typeof r === 'object' ? r : {}) as SeriesRaw;
		const p = v.preset;
		return JSON.stringify([p ? [p.name, p.n ?? null, p.seed ?? 1, p.noise ?? null] : null, v.y ?? null, v.t ?? null, v.points ?? null]);
	};
	return canon(a) === canon(b);
}

/** The underdamped oscillator the PINN preset uses (ζ = 0.15, ω = 12 on t ∈ [0, 1]). */
export const OSCILLATOR = { zeta: 0.15, omega: 12 };

function presetValue(p: NonNullable<SeriesRaw['preset']>): SeriesValue {
	const n = Math.max(0, Math.min(MAX_VALUES, Math.floor(p.n ?? SERIES_PRESETS.find((s) => s.id === p.name)?.n ?? 96)));
	const rng = createRng((p.seed ?? 1) >>> 0);
	const noise = finite(p.noise) ? p.noise : undefined;
	switch (p.name) {
		case 'sine': {
			const s = sine(n, noise ? { noise } : {}, noise ? rng : undefined);
			return { name: s.name, t: s.t, y: s.y, points: [] };
		}
		case 'seasonal': {
			const s = seasonal(n, rng, noise !== undefined ? { noise } : {});
			return { name: s.name, t: s.t, y: s.y, points: [] };
		}
		case 'trend': {
			const s = trend(n, noise ? { noise } : {}, noise ? rng : undefined);
			return { name: s.name, t: s.t, y: s.y, points: [] };
		}
		case 'square': {
			const s = square(n);
			return { name: s.name, t: s.t, y: s.y, points: [] };
		}
		case 'randomWalk': {
			const s = randomWalk(n, rng);
			return { name: s.name, t: s.t, y: s.y, points: [] };
		}
		case 'oscillator': {
			const t = linspace(0, 1, n);
			const y = dampedOscillator(t, OSCILLATOR);
			const obs = observations((tt) => dampedOscillator(tt, OSCILLATOR), rng, { n: 8, tMax: 1, frac: 0.4, noise: noise ?? 0.02 });
			return { name: 'oscillator', t, y, points: Array.from(obs.t, (ti, i) => ({ t: ti, y: obs.u[i] })) };
		}
		default: {
			const t = new Float64Array(n);
			for (let i = 0; i < n; i++) t[i] = i;
			return { name: 'flat', t, y: new Float64Array(n), points: [] };
		}
	}
}

/** The series a raw value describes (deterministic). */
export function seriesOf(raw: unknown): SeriesValue {
	const r = (raw && typeof raw === 'object' ? raw : {}) as SeriesRaw;
	let v: SeriesValue = r.preset ? presetValue(r.preset) : { name: 'drawn', t: new Float64Array(0), y: new Float64Array(0), points: [] };
	if (Array.isArray(r.y)) {
		const y = Float64Array.from(r.y.slice(0, MAX_VALUES).map((x) => (finite(x) ? x : 0)));
		let t: Float64Array;
		if (Array.isArray(r.t) && r.t.length >= y.length) t = Float64Array.from(r.t.slice(0, y.length).map((x, i) => (finite(x) ? x : i)));
		else if (v.t.length === y.length) t = v.t;
		else t = Float64Array.from({ length: y.length }, (_, i) => i);
		v = { name: r.preset ? `${v.name} (edited)` : 'drawn', t, y, points: v.points };
	}
	if (Array.isArray(r.points)) {
		v = { ...v, points: r.points.filter((p) => finite(p?.t) && finite(p?.y)).map((p) => ({ t: p.t, y: p.y })) };
	}
	v.points.sort((a, b) => a.t - b.t);
	return v;
}

export function seriesPresetRaw(name: SeriesPresetName, o: { n?: number; seed?: number; noise?: number } = {}): SeriesRaw {
	return { preset: { name, ...(o.n !== undefined ? { n: o.n } : {}), seed: o.seed ?? 1, ...(o.noise !== undefined ? { noise: o.noise } : {}) } };
}

const LABEL_SERIES: I18n = { en: 'The series: one number per time step', zh: '序列：每个时间步一个数' };

export function preprocessSeries(raw: unknown, _spec: unknown, opts: Record<string, unknown> = {}): Preprocessed {
	const v = seriesOf(raw);
	const n = v.y.length;
	const tensors: Record<string, Tensor> = { t: { shape: [n], data: v.t }, y: { shape: [n], data: v.y } };
	if (v.points.length) {
		const obs = new Float64Array(v.points.length * 2);
		v.points.forEach((p, i) => {
			obs[2 * i] = p.t;
			obs[2 * i + 1] = p.y;
		});
		tensors.obs = { shape: [v.points.length, 2], data: obs };
	}
	if (n > 0) {
		const at = finite(opts.at) ? (opts.at as number) : v.t[n - 1];
		tensors.input = { shape: [1], data: Float64Array.of(at) };
	}
	const notes: I18n[] = [];
	if (n === 0 && !v.points.length) notes.push({ en: 'There is no series yet. Draw one, or pick a preset.', zh: '还没有序列。画一条，或者选一个预设。' });
	else if (n > 1) {
		let lo = Infinity;
		let hi = -Infinity;
		for (const y of v.y) {
			lo = Math.min(lo, y);
			hi = Math.max(hi, y);
		}
		if (hi - lo < 1e-12) notes.push({ en: 'The series is flat: there is no pattern to learn, only one value.', zh: '这条序列是平的：没有规律可学，只有一个值。' });
	}
	return { tensors, steps: [{ id: 'series', label: LABEL_SERIES, tensor: tensors.y }], empty: n === 0 && v.points.length === 0, notes };
}

const adapter: InputAdapter<SeriesRaw> = {
	id: 'series',
	kind: 'series',
	preprocess: preprocessSeries,
	presets(assets) {
		// K1: a presets file ({ presets: [{ id, name?, en?, zh?, seed?, noise?, n? }] }, name defaults to id) gives the list;
		// without one, every generator
		const f = (assets?.['presets.json'] ?? assets?.presets) as { presets?: { id: string; name?: string; en?: string; zh?: string; seed?: number; noise?: number; n?: number }[] } | undefined;
		if (Array.isArray(f?.presets) && f.presets.length)
			return f.presets.flatMap((p) => {
				const name = (p.name ?? p.id) as SeriesPresetName;
				const gen = SERIES_PRESETS.find((x) => x.id === name);
				if (!gen) return [];
				const o: { seed?: number; noise?: number; n?: number } = {};
				if (finite(p.seed)) o.seed = p.seed;
				if (finite(p.noise)) o.noise = p.noise;
				if (finite(p.n)) o.n = p.n;
				return [{ id: p.id, label: p.en ? { en: p.en, zh: p.zh ?? p.en } : gen.label, raw: seriesPresetRaw(name, o) }];
			});
		return SERIES_PRESETS.map((p) => ({ id: p.id, label: p.label, raw: seriesPresetRaw(p.id) }));
	},
	edgeFixtures: () => [
		{ id: 'flat', label: { en: 'A flat series', zh: '一条平的序列' }, raw: seriesPresetRaw('flat') },
		{ id: 'empty', label: { en: 'No values', zh: '没有值' }, raw: { y: [] } },
		{ id: 'one-value', label: { en: 'One value', zh: '一个值' }, raw: { y: [0.4] } }
	],
	layer0: {
		view: 'curve1d',
		narration: {
			what: { en: 'A series is a list of numbers in time order: {n|int} values here, one per time step.', zh: '序列就是按时间排好的一串数：这里有 {n|int} 个值，每个时间步一个。' },
			how: { en: 'The curve joins them so you can see the shape, but the network only ever sees the numbers.', zh: '把它们连成曲线是为了让你看出形状，但网络看到的只是这些数。' }
		}
	}
};

export default adapter;
