// tinyforecaster: the processes (kind C).
//   windows   how the series becomes training examples: the whole series (with the held-out part), then one frame per
//             training window: L values in, the next H values (the target) out. Input: the series as plain numbers, so
//             the Python reference (pipeline/tinyforecaster/reference.py) reads the same list.
//   forecast  trainer 'lstm' run to the end (the same frames as the 'train' source with the same seed and options), then
//             the kit's rollout (process/library/rollout.ts) over the held-out part: each forecast is appended to the
//             history and read back as input, so errors can compound; then the seasonal-naive baseline on the same part.
//   walks     the limits step: trainer 'lstm' on three random walks (input: their values; the manifest passes the series
//             adapter's preset with seeds 1 to 3), one frame per walk: its one-step forecasts of the held-out part and
//             the running averages. Its trainings skip the per-snapshot metrics (not shown; the weights are the same).
//   npm run atlas -- run tinyforecaster windows --frames 0..5;  npm run atlas -- run tinyforecaster forecast --frames 30..60
import type { Control, ProcessDef, ProcessRun, Tensor, TrainState } from '../../atlas/types/index.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { rolloutProcess, type RolloutState } from '../../atlas/process/library/rollout.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { dataOf, L, seasonalNaive, type Data } from './data.ts';
import type { Forecaster } from './model.ts';
import { forecastAll, netOf } from './model.ts';
import { lstm, runData, type Opt } from './trainer.ts';

const T = (en: string, zh: string) => ({ en, zh });

export type WinOpt = { L: number; H: number };
export type WinState = { series: Tensor; window: Tensor; target: Tensor; start: number };

export const windows: ProcessDef<unknown, WinOpt, WinState> = {
	id: 'windows',
	kind: 'algorithm',
	title: T('Cut the series into training examples', '把序列切成训练样例'),
	phases: {
		series: { label: T('The whole series', '整条序列'), kind: 'input' },
		window: {
			label: T('One window and its target', '一个窗口和它的目标'),
			kind: 'observe',
			formula: {
				tex: '(y_{ {{start}} }, \\dots, y_{ {{end}} }) \\;\\to\\; y_{ {{next}} } = {{target}}',
				values: (f) => ({ start: f.metrics!.start, end: f.metrics!.end, next: f.metrics!.end + 1, target: f.metrics!.target })
			}
		}
	},
	schema: {
		series: { kind: 'tensor', label: T('The series', '序列'), role: 'data', tol: 1e-12 },
		window: { kind: 'tensor', label: T('The window (model input)', '窗口（模型输入）'), role: 'data', tol: 1e-12 },
		target: { kind: 'tensor', label: T('The next values (target)', '接下来的值（目标）'), role: 'target', tol: 1e-12 },
		start: { kind: 'int', label: T('Where the window starts', '窗口的起点'), tol: 'exact' }
	},
	defaults: { L, H: 1 },
	*run(input, ctx): ProcessRun<WinState, WinOpt> {
		const d = dataOf(input);
		const { L: W, H } = ctx.opt;
		const count = Math.max(0, d.split - W - H + 1);
		yield {
			phase: 'series',
			state: { series: { shape: [d.n], data: d.y.slice() } },
			metrics: { n: d.n, hold: d.hold, split: d.split, count, L: W, H }
		};
		for (let k = 0; k < count && !ctx.signal.cancelled; k++) {
			yield {
				phase: 'window',
				step: k,
				state: { window: { shape: [W], data: d.y.slice(k, k + W) }, target: { shape: [H], data: d.y.slice(k + W, k + W + H) }, start: k },
				metrics: { k, start: k, end: k + W - 1, target: d.y[k + W], count, L: W, H }
			};
		}
	}
};

const inner = trainingProcess(lstm);
export type FcState = TrainState & RolloutState & { naive?: Tensor; one?: Tensor };
type FcOpt = Opt & { horizon?: number };

export const forecast: ProcessDef<unknown, FcOpt, FcState> = {
	...inner,
	id: 'forecast',
	title: T('Train, then forecast the held-out part', '先训练，再预测留出的部分'),
	phases: {
		...inner.phases,
		start: { label: T('Forecast from the split', '从分界处开始预测'), kind: 'input' },
		window: { label: T('Read the last values', '读出最近的数值'), kind: 'observe' },
		predict: { label: T('Predict, append, repeat', '预测、追加、重复'), kind: 'output' },
		naive: { label: T('The seasonal-naive baseline', '季节性朴素基准'), kind: 'output' }
	},
	// tolerances for the Python reference: 200 Adam steps in PyTorch float64 drift from TS by up to about 1e-10 (notes),
	// so everything computed from trained weights is compared at 1e-9; the data and the baseline stay at 1e-12
	schema: {
		...inner.schema,
		loss: { ...inner.schema.loss, tol: 1e-9 },
		params: { ...inner.schema.params, tol: 1e-9 },
		...rolloutProcess({ id: 'r', predict: () => 0 }).schema,
		naive: { kind: 'tensor', label: T('Seasonal-naive forecast', '季节性朴素预测'), role: 'output', tol: 1e-12 },
		one: { kind: 'tensor', label: T('One-step forecasts from the true values', '用真实值做的单步预测'), role: 'output', tol: 1e-9 }
	},
	defaults: { ...lstm.defaults },
	*run(input, ctx): ProcessRun<FcState, FcOpt> {
		let params: Record<string, Tensor> | undefined;
		const g = inner.run(input, ctx as never);
		let c: Control<FcOpt> | undefined;
		const d = runData(input, ctx.opt);
		for (let r = g.next(); !r.done; r = g.next(c as never)) {
			if (r.value.state.params) params = r.value.state.params;
			// the rollout's path (tracked by changes) exists from the first frame: NaN until predicted
			if (r.value.phase === 'init') (r.value.state as Partial<FcState>).path = { shape: [d.hold], data: new Float64Array(d.hold).fill(NaN) };
			c = yield r.value as never;
		}
		if (!params || ctx.signal.cancelled) return;
		const m = netOf(params, createRng(1));
		const roll = rolloutProcess({ id: 'rollout', predict: (w) => forecastAll(m, w, 1)[0] });
		const opt = { window: L, horizon: d.hold, start: d.split };
		let absSum = 0;
		let early = 0;
		let late = 0;
		const q = Math.max(1, Math.floor(d.hold / 3));
		for (const f of roll.run({ series: d.y }, { ...ctx, opt, detail: 'coarse' } as never)) {
			if (f.phase === 'predict') {
				const h = f.step!;
				const a = Math.abs(f.metrics!.err);
				absSum += a;
				if (h < q) early += a / q;
				if (h >= d.hold - q) late += a / q;
			}
			// the scrubber names a forecast by its frame (the library counts from 0): "forecast 24 of 24" for the last one
			const k = (f.step ?? 0) + 1;
			yield (f.phase === 'predict' ? { ...f, caption: T(`Predict, append, repeat · forecast ${k} of ${d.hold}`, `预测、追加、重复 · 第 ${k} 个预测（共 ${d.hold} 个）`) } : f) as never;
		}
		const nv = seasonalNaive(d);
		const o = oneStep(m, d);
		yield {
			phase: 'naive',
			state: { naive: { shape: [d.hold], data: nv.path }, one: { shape: [d.hold], data: o.path } },
			metrics: { m: nv.m, naiveMae: nv.mae, lstmMae: absSum / d.hold, hold: d.hold, early, late, q, ...o.mae }
		};
	}
};

/** One-step forecasts (each from the true values before it) and the persistence rule "repeat the last value":
 *  on the held-out part (one, persist) and over the training windows (fit, persistFit). Mean absolute misses. */
export function oneStep(m: Forecaster, d: Data): { path: Float64Array; mae: { one: number; persist: number; fit: number; persistFit: number } } {
	const rows = (from: number, to: number) => {
		const x = new Float64Array((to - from) * L);
		for (let t = from; t < to; t++) x.set(d.y.subarray(t - L, t), (t - from) * L);
		return forecastAll(m, x, to - from);
	};
	const mae = (p: Float64Array, from: number) => p.reduce((s, v, k) => s + Math.abs(v - d.y[from + k]), 0) / p.length;
	const persist = (from: number, to: number) => {
		let s = 0;
		for (let t = from; t < to; t++) s += Math.abs(d.y[t - 1] - d.y[t]);
		return s / (to - from);
	};
	const path = rows(d.split, d.n);
	return { path, mae: { one: mae(path, d.split), persist: persist(d.split, d.n), fit: mae(rows(L, d.split), L), persistFit: persist(L, d.split) } };
}

export type WalkState = { series: Tensor; one: Tensor };

/** The trained parameters of one full run of trainer 'lstm' on `raw` (the frames are not kept). */
function trained(raw: unknown, ctx: Parameters<typeof inner.run>[1]): Record<string, Tensor> | undefined {
	let params: Record<string, Tensor> | undefined;
	for (const f of inner.run(raw, ctx)) if (f.state.params) params = f.state.params;
	return params;
}

export const walks: ProcessDef<unknown, Opt, WalkState> = {
	id: 'walks',
	kind: 'training', // so an e2e run caps its steps as it caps every training source (engine/runner.ts)
	title: T('The same network on three random walks', '同一个网络，三条随机游走'),
	phases: { walk: { label: T('One random walk', '一条随机游走'), kind: 'output' } },
	schema: {
		series: { kind: 'tensor', label: T('The walk', '随机游走'), role: 'data', tol: 1e-12 },
		one: { kind: 'tensor', label: T('One-step forecasts of the held-out part', '对留出部分的单步预测'), role: 'output', tol: 1e-9 }
	},
	defaults: { ...lstm.defaults },
	*run(input, ctx): ProcessRun<WalkState, Opt> {
		const list = (Array.isArray(input) ? input : []) as number[][];
		const sum = { fit: 0, persistFit: 0, one: 0, persist: 0 };
		let wins = 0;
		for (let k = 0; k < list.length && !ctx.signal.cancelled; k++) {
			const raw = list[k];
			const params = trained(raw, { ...ctx, detail: 'coarse', opt: { ...ctx.opt, snapshotEvery: ctx.opt.steps } } as never);
			if (!params) return;
			const d = dataOf(raw);
			const o = oneStep(netOf(params, createRng(1)), d);
			for (const key of Object.keys(sum) as (keyof typeof sum)[]) sum[key] += o.mae[key];
			if (o.mae.one < o.mae.persist) wins++;
			const n = k + 1;
			yield {
				phase: 'walk',
				step: k,
				sub: k + 1, // the scrubber reads 'walk 1' to 'walk 4' (play.by 'sub')
				state: { series: { shape: [d.n], data: d.y.slice() }, one: { shape: [d.hold], data: o.path } },
				metrics: { ...o.mae, walk: n, walks: list.length, split: d.split, avgFit: sum.fit / n, avgPersistFit: sum.persistFit / n, avgOne: sum.one / n, avgPersist: sum.persist / n, wins }
			};
		}
	}
};
