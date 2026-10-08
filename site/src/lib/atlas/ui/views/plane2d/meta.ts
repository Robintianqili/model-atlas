// View 'plane2d' (kinds B and C): a 2-D plane with layers, all optional:
//   points   [n, 2] data points (default 'pre.xy', the points2d adapter), colored by `labels` [n] (default 'pre.label')
//   grid     [R, R] values over the plane, row 0 at the top (default: the state field 'grid', e.g. a classifier's derived
//            P(class 1)); gridKind 'prob' (two-class tint + the 0.5 boundary), 'sequential' or 'diverging'
//   vectors  [K, 4] arrows (x, y, dx, dy) (default: the state field 'field'), or vectorsAt(t, state) with a time slider
//   paths    [N, K, 2] trajectories (Euler sampling), samples [M, 2] generated points
// Props: points, labels, grid, gridKind, vectors, vectorsAt, paths, samples, time { min, max, value?, step?, label? },
//   bounds [x0, x1, y0, y1] (default ±1.25), classes (I18n names), highlight (point index),
//   pointValues [n] (the model's value at each data point: exact probe readouts, e.g. a derived field) or
//   valueAt(x, y, state) -> number (the model at any place). Without them the probe reads the grid between its cell
//   centres and says "≈" (K0: it read the nearest of 40 × 40 cells, 0.536 for a point whose share is 0.5367).
// Each reference is a tensor, 'pre.<name>', a state field, or 'field:key' (ui/train/refs.ts).
// Focus { field: 'xy' | 'points', index: [i] } rings point i.
import type { AnyTensor, I18n, Tensor } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import { fmt3, fmtInt, fmtPct } from '../../../core/format.ts';
import { resolveRef, type Ref } from '../../train/refs.ts';

export interface PlaneBounds {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}

export interface PlaneData {
	points: AnyTensor | undefined;
	labels: AnyTensor | undefined;
	grid: AnyTensor | undefined;
	gridKind: 'prob' | 'sequential' | 'diverging';
	vectors: AnyTensor | undefined;
	paths: AnyTensor | undefined;
	samples: AnyTensor | undefined;
	bounds: PlaneBounds;
	classes: I18n[] | null;
	highlight: number | null;
	/** exact values at the data points (props.pointValues), or the model at any place (props.valueAt) */
	pointValues: AnyTensor | undefined;
	valueAt: ((x: number, y: number) => number) | null;
	/** labelled: more than one class among the points */
	labelled: boolean;
	classCounts: number[];
}

function rows2(t: AnyTensor | undefined): number {
	return t && t.shape.length === 2 && t.shape[1] >= 2 ? t.shape[0] : 0;
}

/** Everything the view draws, resolved from the stage props (pure). `t` is the time slider value for vectorsAt. */
export function planeData(p: Pick<StageProps, 'rt' | 'at' | 'props' | 'focus'>, t?: number): PlaneData {
	const props = p.props ?? {};
	const has = (k: string) => Object.prototype.hasOwnProperty.call(props, k);
	const points = resolveRef((has('points') ? props.points : 'pre.xy') as Ref, p.rt, p.at);
	const labels = resolveRef((has('labels') ? props.labels : 'pre.label') as Ref, p.rt, p.at);
	const grid = resolveRef((has('grid') ? props.grid : 'grid') as Ref, p.rt, p.at);
	let vectors = resolveRef((has('vectors') ? props.vectors : 'field') as Ref, p.rt, p.at);
	if (typeof props.vectorsAt === 'function' && p.at?.state) {
		const time = props.time as { value?: number; min?: number } | undefined;
		try {
			vectors = (props.vectorsAt as (t: number, s: unknown) => Tensor)(t ?? time?.value ?? time?.min ?? 0, p.at.state);
		} catch {
			vectors = undefined;
		}
	}
	const paths = resolveRef(props.paths as Ref, p.rt, p.at);
	const pointValues = resolveRef(props.pointValues as Ref, p.rt, p.at);
	const va = props.valueAt;
	const valueAt = typeof va === 'function' ? (x: number, y: number) => Number((va as (x: number, y: number, s: unknown) => number)(x, y, p.at?.state)) : null;
	const samples = resolveRef(props.samples as Ref, p.rt, p.at);
	const b = props.bounds as number[] | undefined;
	const bounds = Array.isArray(b) && b.length === 4 ? { x0: b[0], x1: b[1], y0: b[2], y1: b[3] } : { x0: -1.25, x1: 1.25, y0: -1.25, y1: 1.25 };
	const n = rows2(points);
	const counts: number[] = [];
	if (labels && labels.data.length >= n)
		for (let i = 0; i < n; i++) {
			const c = Math.max(0, Math.round(labels.data[i]));
			counts[c] = (counts[c] ?? 0) + 1;
		}
	else if (n) counts[0] = n;
	for (let c = 0; c < counts.length; c++) counts[c] = counts[c] ?? 0;
	let highlight: number | null = typeof props.highlight === 'number' ? props.highlight : null;
	for (const f of p.focus ?? []) {
		if ((f.field === 'xy' || f.field === 'points') && Array.isArray(f.index) && typeof f.index[0] === 'number') highlight = f.index[0] as number;
	}
	if (highlight !== null && (highlight < 0 || highlight >= n)) highlight = null;
	return {
		points: n ? points : undefined,
		labels,
		grid: grid && grid.shape.length === 2 && grid.data.length > 0 ? grid : undefined,
		gridKind: props.gridKind === 'sequential' || props.gridKind === 'diverging' ? props.gridKind : 'prob',
		vectors: vectors && vectors.shape.length === 2 && vectors.shape[1] >= 4 ? vectors : undefined,
		paths: paths && paths.shape.length === 3 && paths.shape[2] >= 2 ? paths : undefined,
		samples: rows2(samples) ? samples : undefined,
		bounds,
		classes: Array.isArray(props.classes) ? (props.classes as I18n[]) : null,
		highlight,
		pointValues: pointValues && pointValues.data.length >= n ? pointValues : undefined,
		valueAt,
		labelled: counts.filter((c) => c > 0).length > 1,
		classCounts: counts
	};
}

/** The grid value under a point of the plane (nearest cell), NaN outside. */
export function gridAt(d: PlaneData, x: number, y: number): number {
	const g = d.grid;
	if (!g) return NaN;
	const [R, C] = g.shape;
	const c = Math.floor(((x - d.bounds.x0) / (d.bounds.x1 - d.bounds.x0)) * C);
	const r = Math.floor(((d.bounds.y1 - y) / (d.bounds.y1 - d.bounds.y0)) * R);
	if (r < 0 || r >= R || c < 0 || c >= C) return NaN;
	return g.data[r * C + c];
}

/** The grid between its cell centres (bilinear; the grid's values are taken at cell centres), NaN outside the plane. */
export function gridLerp(d: PlaneData, x: number, y: number): number {
	const g = d.grid;
	if (!g) return NaN;
	const [R, C] = g.shape;
	const u = ((x - d.bounds.x0) / (d.bounds.x1 - d.bounds.x0)) * C;
	const w = ((d.bounds.y1 - y) / (d.bounds.y1 - d.bounds.y0)) * R;
	if (!(u >= 0 && u <= C && w >= 0 && w <= R)) return NaN;
	const fx = Math.max(0, Math.min(C - 1, u - 0.5));
	const fy = Math.max(0, Math.min(R - 1, w - 0.5));
	const c0 = Math.floor(fx), r0 = Math.floor(fy);
	const c1 = Math.min(C - 1, c0 + 1), r1 = Math.min(R - 1, r0 + 1);
	const a = fx - c0, b = fy - r0;
	const at = (r: number, c: number) => g.data[r * C + c];
	return (1 - b) * ((1 - a) * at(r0, c0) + a * at(r0, c1)) + b * ((1 - a) * at(r1, c0) + a * at(r1, c1));
}

/** The value at data point i: exact from pointValues or valueAt, else read from the grid (exact: false). */
export function pointValue(d: PlaneData, i: number): { v: number; exact: boolean } {
	if (d.pointValues && i < d.pointValues.data.length) return { v: d.pointValues.data[i], exact: true };
	if (!d.points) return { v: NaN, exact: false };
	const x = d.points.data[2 * i],
		y = d.points.data[2 * i + 1];
	if (d.valueAt) {
		const v = d.valueAt(x, y);
		if (Number.isFinite(v)) return { v, exact: true };
	}
	return { v: gridLerp(d, x, y), exact: false };
}

/** Share of points whose value is on their own class's side of 0.5 (two-class 'prob' grids only). */
export function agreement(d: PlaneData): number {
	if (!d.points || !(d.grid || d.pointValues || d.valueAt) || d.gridKind !== 'prob' || !d.labels) return NaN;
	const n = d.points.shape[0];
	let ok = 0;
	let seen = 0;
	for (let i = 0; i < n; i++) {
		const v = pointValue(d, i).v;
		if (!Number.isFinite(v)) continue;
		seen++;
		if ((v >= 0.5 ? 1 : 0) === Math.round(d.labels.data[i])) ok++;
	}
	return seen ? ok / seen : NaN;
}

const meta = {
	key: 'plane2d',
	title: { en: 'Points on a plane', zh: '平面上的点' },
	kinds: ['train', 'process'],
	accepts: (o) => (o.axes ? o.axes.join(',') === 'p,xy' : o.field === 'xy' || o.field === 'grid'),
	describe(p: StageProps): I18n {
		const d = planeData(p);
		const n = d.points ? d.points.shape[0] : 0;
		const parts: I18n[] = [];
		if (n === 0) parts.push({ en: 'The plane has no points yet.', zh: '平面上还没有点。' });
		else if (d.labelled) {
			const cs = d.classCounts.map((c, k) => `${fmtInt(c)}`);
			parts.push({ en: `${fmtInt(n)} points in ${d.classCounts.length} classes (${cs.join(' and ')} points).`, zh: `${fmtInt(n)} 个点，分成 ${d.classCounts.length} 类（分别 ${cs.join(' 和 ')} 个）。` });
		} else parts.push({ en: `${fmtInt(n)} points.`, zh: `${fmtInt(n)} 个点。` });
		if (d.grid) {
			let lo = Infinity,
				hi = -Infinity;
			for (const v of d.grid.data) {
				lo = Math.min(lo, v);
				hi = Math.max(hi, v);
			}
			if (d.gridKind === 'prob') {
				const agree = agreement(d);
				parts.push({
					en: `The background is the network's answer everywhere: its share for class 1 goes from ${fmt3(lo)} to ${fmt3(hi)}.${Number.isFinite(agree) ? ` ${fmtPct(agree)} of the points lie on the side of their own class.` : ''}`,
					zh: `背景是网络在每个位置的回答：属于第 1 类的比例从 ${fmt3(lo)} 到 ${fmt3(hi)}。${Number.isFinite(agree) ? `${fmtPct(agree)} 的点落在自己那一类的一侧。` : ''}`
				});
			} else parts.push({ en: `The background shows values from ${fmt3(lo)} to ${fmt3(hi)}.`, zh: `背景显示的值从 ${fmt3(lo)} 到 ${fmt3(hi)}。` });
		}
		if (d.vectors) parts.push({ en: `${fmtInt(d.vectors.shape[0])} arrows show which way points are pushed.`, zh: `${fmtInt(d.vectors.shape[0])} 个箭头显示点被推向哪里。` });
		if (d.paths) parts.push({ en: `${fmtInt(d.paths.shape[0])} paths of ${fmtInt(d.paths.shape[1] - 1)} steps each.`, zh: `${fmtInt(d.paths.shape[0])} 条路径，每条 ${fmtInt(d.paths.shape[1] - 1)} 步。` });
		if (d.samples) parts.push({ en: `${fmtInt(d.samples.shape[0])} generated points.`, zh: `${fmtInt(d.samples.shape[0])} 个生成的点。` });
		return { en: parts.map((x) => x.en).join(' '), zh: parts.map((x) => x.zh).join('') };
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { classifierRun, demoProps } = await import('../../train/demo.ts');
		return demoProps(classifierRun('coarse', 300), 'last', {}, [{ field: 'xy', index: [3], role: 'accent' }]);
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
