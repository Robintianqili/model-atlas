// Input adapter 'points2d' (design §4.9; catalog InputKind 'points2d'): a set of 2-D points, labelled by class (MLP
// playground) or unlabelled (flow-2d). Pure TS: the trainer in a worker reads the same raw value with pointsOf().
//
// Raw value (plain data, so it can be posted to a worker and stored in a URL-free way):
//   { preset?: { name, n?, seed?, noise? },   a seeded generator from atlas/data/points2d.ts (TS is the source of truth)
//     points?: [{ x, y, c? }],                 points the learner painted (added after the preset's)
//     removed?: number[] }                     preset points the learner erased (indices into the preset)
// Preprocessed tensors: xy [n, 2] and label [n] (class indices as numbers), plus `input` [2] = one point (options.point,
// default 0) so a 'live-model' source can trace a single point. Coordinates are float64 (training toys use f64).
import type { I18n, Tensor } from '../../types/core.ts';
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';
import { POINTS2D, type Points2D, type Points2DName } from '../../data/points2d.ts';
import { createRng } from '../../core/rng.ts';

export interface PaintedPoint {
	x: number;
	y: number;
	/** class index (default 0) */
	c?: number;
}

export interface PointsRaw {
	preset?: { name: Points2DName; n?: number; seed?: number; noise?: number };
	points?: PaintedPoint[];
	removed?: number[];
}

export interface PointPreset {
	id: Points2DName;
	label: I18n;
	/** two classes (classifiers) or one (flow-2d) */
	labelled: boolean;
	n: number;
}

export const POINT_PRESETS: PointPreset[] = [
	{ id: 'moons', label: { en: 'Two moons', zh: '两个月牙' }, labelled: true, n: 200 },
	{ id: 'circles', label: { en: 'Circle in a ring', zh: '环中圆' }, labelled: true, n: 200 },
	{ id: 'xor', label: { en: 'XOR quadrants', zh: '异或象限' }, labelled: true, n: 200 },
	{ id: 'spiral', label: { en: 'Spiral', zh: '螺旋' }, labelled: true, n: 300 },
	{ id: 'gaussians', label: { en: 'Two blobs', zh: '两团点' }, labelled: true, n: 200 },
	{ id: 'smiley', label: { en: 'Smiley', zh: '笑脸' }, labelled: false, n: 600 },
	{ id: 'ring', label: { en: 'Ring', zh: '圆环' }, labelled: false, n: 600 }
];

/** The largest number of points a raw value may hold (painted + preset). */
export const MAX_POINTS = 4000;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** The points a raw value describes (deterministic: the same raw value gives the same points everywhere). */
export function pointsOf(raw: unknown): Points2D {
	const r = (raw && typeof raw === 'object' ? raw : {}) as PointsRaw;
	const xs: number[] = [];
	const cs: number[] = [];
	let name = 'painted';
	let classes = 1;
	if (r.preset && r.preset.name in POINTS2D) {
		const gen = POINTS2D[r.preset.name];
		const n = Math.max(0, Math.min(MAX_POINTS, Math.floor(r.preset.n ?? 200)));
		const opts: Record<string, number> = finite(r.preset.noise) ? { noise: r.preset.noise } : {};
		const p = (gen as (n: number, rng: ReturnType<typeof createRng>, o?: Record<string, number>) => Points2D)(n, createRng((r.preset.seed ?? 1) >>> 0), opts);
		const gone = new Set((r.removed ?? []).filter((i) => Number.isInteger(i)));
		for (let i = 0; i < p.n; i++) {
			if (gone.has(i)) continue;
			xs.push(p.xy[2 * i], p.xy[2 * i + 1]);
			cs.push(p.label[i]);
		}
		name = p.name;
		classes = p.classes;
	}
	for (const q of r.points ?? []) {
		if (xs.length / 2 >= MAX_POINTS) break;
		if (!finite(q?.x) || !finite(q?.y)) continue;
		const c = Number.isInteger(q.c) && (q.c as number) >= 0 ? (q.c as number) : 0;
		xs.push(q.x, q.y);
		cs.push(c);
		classes = Math.max(classes, c + 1);
	}
	const n = cs.length;
	return { name, xy: Float64Array.from(xs), label: Int32Array.from(cs), n, classes };
}

/** Class counts (index = class). */
export function classCounts(p: Points2D): number[] {
	const out = new Array<number>(Math.max(1, p.classes)).fill(0);
	for (let i = 0; i < p.n; i++) out[p.label[i]] = (out[p.label[i]] ?? 0) + 1;
	return out;
}

/** A preset's raw value. */
export function presetRaw(name: Points2DName, o: { n?: number; seed?: number; noise?: number } = {}): PointsRaw {
	const def = POINT_PRESETS.find((p) => p.id === name);
	return { preset: { name, n: o.n ?? def?.n ?? 200, seed: o.seed ?? 1, ...(o.noise !== undefined ? { noise: o.noise } : {}) } };
}

const LABEL_POINTS: I18n = { en: 'Points: x and y (and a class)', zh: '点：x 和 y（以及类别）' };

export function preprocessPoints(raw: unknown, _spec: unknown, opts: Record<string, unknown> = {}): Preprocessed {
	const p = pointsOf(raw);
	const xy: Tensor = { shape: [p.n, 2], data: p.xy };
	const label: Tensor = { shape: [p.n], data: Float64Array.from(p.label) };
	const tensors: Record<string, Tensor> = { xy, label };
	const notes: I18n[] = [];
	if (p.n > 0) {
		const k = Math.max(0, Math.min(p.n - 1, Math.floor(Number(opts.point ?? 0)) || 0));
		tensors.input = { shape: [2], data: Float64Array.of(p.xy[2 * k], p.xy[2 * k + 1]) };
	} else notes.push({ en: 'There are no points yet. Add some, or pick a preset.', zh: '还没有点。加几个点，或者选一个预设。' });
	const counts = classCounts(p);
	const wantClasses = Number(opts.classes ?? (p.classes > 1 ? p.classes : 1));
	if (p.n > 0 && wantClasses > 1 && counts.filter((c) => c > 0).length < 2) {
		notes.push({ en: 'Every point has the same class, so there is no boundary to learn.', zh: '所有点都是同一类，所以没有边界可学。' });
	}
	return { tensors, steps: [{ id: 'points', label: LABEL_POINTS, tensor: xy }], empty: p.n === 0, notes };
}

const adapter: InputAdapter<PointsRaw> = {
	id: 'points2d',
	kind: 'points2d',
	preprocess: preprocessPoints,
	presets(assets) {
		// options.presets (an id list) may be given through assets.presets2d; default: every preset
		const only = (assets?.presets2d as string[] | undefined) ?? null;
		return POINT_PRESETS.filter((p) => !only || only.includes(p.id)).map((p) => ({ id: p.id, label: p.label, raw: presetRaw(p.id) }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'No points', zh: '没有点' }, raw: { points: [] } },
		{ id: 'one-point', label: { en: 'One point', zh: '一个点' }, raw: { points: [{ x: 0.2, y: -0.3, c: 1 }] } },
		{
			id: 'one-class',
			label: { en: 'Only one class', zh: '只有一类' },
			raw: { points: [-0.6, -0.2, 0.1, 0.5, 0.8].map((x, i) => ({ x, y: 0.3 * Math.sin(3 * x) + 0.1 * i, c: 0 })) }
		}
	],
	layer0: {
		view: 'plane2d',
		narration: {
			what: { en: 'Each point is two numbers: its x and its y. Together they form a table of {shape} numbers, {n|int} in all.', zh: '每个点是两个数：它的 x 和 y。合起来是一张 {shape} 的数表，一共 {n|int} 个数。' },
			how: { en: 'When the points have classes, the color of a point is its class: the answer the network has to learn.', zh: '如果点有类别，点的颜色就是它的类别：网络要学会给出的答案。' }
		}
	}
};

export default adapter;
