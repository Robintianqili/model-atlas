// View 'curve1d' (kinds B and C): one or more curves over time, observation points and window overlays.
// Props (StageProps.props):
//   x?: Ref                      the time axis of the data (default 'pre.t', the series adapter)
//   series?: { y: Ref; x?: Ref; label?: I18n; kind?: 'data' | 'model' | 'truth' | 'error'; domain?: 'unit' }[]
//                                default: the data ('pre.y') and, when the state has one, the model's curve 'pred'.
//                                A [N, 2] tensor is read as (x, y) rows; domain 'unit' maps x ∈ [−1, 1] onto the data's
//                                time range (the fit trainer's derived 'pred').
//   points?: Ref                 [k, 2] observation points (default 'pre.obs')
//   windows?: { from: number; to: number; label?: I18n }[]   shaded time windows (forecaster inputs and targets)
//   yRange?: [number, number]; xLabel?: I18n; yLabel?: I18n (K1: drawn along the y axis); highlight?: number (an index into
//   the first curve); legend?: true (K1: a legend for a single curve too; two or more curves always get one)
// Focus { field: 'y', index: [i] } marks value i of the first curve.
import type { AnyTensor, I18n } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import { fmt3, fmtInt } from '../../../core/format.ts';
import { resolveRef, type Ref } from '../../train/refs.ts';

export type CurveKind = 'data' | 'model' | 'truth' | 'error';

export interface Line {
	label: I18n;
	kind: CurveKind;
	x: Float64Array;
	y: Float64Array;
}

export interface CurveData {
	lines: Line[];
	points: { x: number; y: number }[];
	windows: { from: number; to: number; label?: I18n }[];
	x0: number;
	x1: number;
	y0: number;
	y1: number;
	highlight: number | null;
}

const KIND_LABEL: Record<CurveKind, I18n> = {
	data: { en: 'data', zh: '数据' },
	model: { en: 'the network', zh: '网络' },
	truth: { en: 'true curve', zh: '真实曲线' },
	error: { en: 'error', zh: '误差' }
};

type SeriesSpec = { y: Ref; x?: Ref; label?: I18n; kind?: CurveKind; domain?: 'unit' };

function lineOf(s: SeriesSpec, defaultX: AnyTensor | undefined, p: Pick<StageProps, 'rt' | 'at'>, span: [number, number] | null): Line | null {
	const yt = resolveRef(s.y, p.rt, p.at);
	if (!yt || yt.data.length === 0) return null;
	const kind = s.kind ?? 'data';
	let x: Float64Array;
	let y: Float64Array;
	if (yt.shape.length === 2 && yt.shape[1] >= 2) {
		const n = yt.shape[0];
		x = new Float64Array(n);
		y = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			x[i] = yt.data[i * yt.shape[1]];
			y[i] = yt.data[i * yt.shape[1] + 1];
		}
	} else {
		y = Float64Array.from(yt.data);
		const xt = s.x ? resolveRef(s.x, p.rt, p.at) : defaultX;
		x = xt && xt.data.length === y.length ? Float64Array.from(xt.data) : Float64Array.from({ length: y.length }, (_, i) => i);
	}
	if (s.domain === 'unit' && span) {
		const [a, b] = span;
		for (let i = 0; i < x.length; i++) x[i] = a + ((x[i] + 1) / 2) * (b - a);
	}
	return { label: s.label ?? KIND_LABEL[kind], kind, x, y };
}

/** Everything the view draws (pure). */
export function curveData(p: Pick<StageProps, 'rt' | 'at' | 'props' | 'focus'>): CurveData {
	const props = p.props ?? {};
	const defaultX = resolveRef((props.x ?? 'pre.t') as Ref, p.rt, p.at);
	let specs = props.series as SeriesSpec[] | undefined;
	if (!Array.isArray(specs)) {
		specs = [{ y: 'pre.y', kind: 'data' }];
		if (resolveRef('pred', p.rt, p.at)) specs.push({ y: 'pred', kind: 'model', domain: 'unit' });
	}
	const span: [number, number] | null = defaultX && defaultX.data.length ? [defaultX.data[0], defaultX.data[defaultX.data.length - 1]] : null;
	const lines: Line[] = [];
	for (const s of specs) {
		const l = lineOf(s, defaultX, p, span);
		if (l) lines.push(l);
	}
	const pt = resolveRef((Object.prototype.hasOwnProperty.call(props, 'points') ? props.points : 'pre.obs') as Ref, p.rt, p.at);
	const points: { x: number; y: number }[] = [];
	if (pt && pt.shape.length === 2 && pt.shape[1] >= 2) for (let i = 0; i < pt.shape[0]; i++) points.push({ x: pt.data[i * pt.shape[1]], y: pt.data[i * pt.shape[1] + 1] });
	let x0 = Infinity,
		x1 = -Infinity,
		y0 = Infinity,
		y1 = -Infinity;
	const take = (x: number, y: number) => {
		if (!Number.isFinite(x) || !Number.isFinite(y)) return;
		x0 = Math.min(x0, x);
		x1 = Math.max(x1, x);
		y0 = Math.min(y0, y);
		y1 = Math.max(y1, y);
	};
	for (const l of lines) for (let i = 0; i < l.x.length; i++) take(l.x[i], l.y[i]);
	for (const q of points) take(q.x, q.y);
	const yr = props.yRange as number[] | undefined;
	if (Array.isArray(yr) && yr.length === 2) {
		y0 = yr[0];
		y1 = yr[1];
	}
	if (!Number.isFinite(x0)) {
		x0 = 0;
		x1 = 1;
		y0 = 0;
		y1 = 1;
	}
	if (x1 === x0) x1 = x0 + 1;
	if (y1 === y0) {
		y0 -= 0.5;
		y1 += 0.5;
	}
	let highlight: number | null = typeof props.highlight === 'number' ? props.highlight : null;
	for (const f of p.focus ?? []) if (f.field === 'y' && Array.isArray(f.index) && typeof f.index[0] === 'number') highlight = f.index[0] as number;
	if (highlight !== null && (!lines[0] || highlight < 0 || highlight >= lines[0].y.length)) highlight = null;
	return { lines, points, windows: Array.isArray(props.windows) ? (props.windows as CurveData['windows']) : [], x0, x1, y0, y1, highlight };
}

/** Linear interpolation of a line at x (NaN outside its range). */
export function interp(l: Line, x: number): number {
	const n = l.x.length;
	if (!n) return NaN;
	if (x < Math.min(l.x[0], l.x[n - 1]) || x > Math.max(l.x[0], l.x[n - 1])) return NaN;
	let lo = 0,
		hi = n - 1;
	const up = l.x[n - 1] >= l.x[0];
	while (hi - lo > 1) {
		const mid = (lo + hi) >> 1;
		if (up ? l.x[mid] <= x : l.x[mid] >= x) lo = mid;
		else hi = mid;
	}
	const a = l.x[lo],
		b = l.x[hi];
	const u = b === a ? 0 : (x - a) / (b - a);
	return l.y[lo] + u * (l.y[hi] - l.y[lo]);
}

/** Mean |model − data| over the data's x values (the gap the training tries to close), NaN without both lines. */
export function meanGap(d: CurveData): number {
	const data = d.lines.find((l) => l.kind === 'data');
	const model = d.lines.find((l) => l.kind === 'model');
	if (!data || !model) return NaN;
	let s = 0,
		n = 0;
	for (let i = 0; i < data.x.length; i++) {
		const m = interp(model, data.x[i]);
		if (!Number.isFinite(m) || !Number.isFinite(data.y[i])) continue;
		s += Math.abs(m - data.y[i]);
		n++;
	}
	return n ? s / n : NaN;
}

const meta = {
	key: 'curve1d',
	title: { en: 'Curve over time', zh: '随时间变化的曲线' },
	kinds: ['train', 'process'],
	accepts: (o) => (o.axes ? o.axes.join(',') === 't' : o.field === 'y' || o.field === 'pred'),
	describe(p: StageProps): I18n {
		const d = curveData(p);
		const parts: I18n[] = [];
		const first = d.lines[0];
		if (!first) parts.push({ en: 'There is no curve yet.', zh: '还没有曲线。' });
		else {
			let lo = Infinity,
				hi = -Infinity;
			for (const v of first.y) {
				if (!Number.isFinite(v)) continue;
				lo = Math.min(lo, v);
				hi = Math.max(hi, v);
			}
			parts.push({
				en: `A curve of ${fmtInt(first.y.length)} values from t = ${fmt3(first.x[0])} to ${fmt3(first.x[first.x.length - 1])}, between ${fmt3(lo)} and ${fmt3(hi)}.`,
				zh: `一条有 ${fmtInt(first.y.length)} 个值的曲线，t 从 ${fmt3(first.x[0])} 到 ${fmt3(first.x[first.x.length - 1])}，取值在 ${fmt3(lo)} 和 ${fmt3(hi)} 之间。`
			});
		}
		const gap = meanGap(d);
		if (Number.isFinite(gap)) parts.push({ en: `The network's curve is ${fmt3(gap)} away from the data on average.`, zh: `网络的曲线与数据平均相差 ${fmt3(gap)}。` });
		else if (d.lines.length > 1) parts.push({ en: `${d.lines.length - 1} more curve(s) are drawn.`, zh: `另外还画了 ${d.lines.length - 1} 条曲线。` });
		if (d.points.length) parts.push({ en: `${fmtInt(d.points.length)} observation points are marked.`, zh: `标出了 ${fmtInt(d.points.length)} 个观测点。` });
		if (d.windows.length) parts.push({ en: `${fmtInt(d.windows.length)} time windows are shaded.`, zh: `有 ${fmtInt(d.windows.length)} 个时间窗口被涂上了底色。` });
		return { en: parts.map((x) => x.en).join(' '), zh: parts.map((x) => x.zh).join('') };
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { demoProps, fitRun } = await import('../../train/demo.ts');
		return demoProps(fitRun(400), 'last', { windows: [{ from: 60, to: 83, label: { en: 'input window', zh: '输入窗口' } }] });
	},
	minHeight: 240
} satisfies ViewMeta;

export default meta;
