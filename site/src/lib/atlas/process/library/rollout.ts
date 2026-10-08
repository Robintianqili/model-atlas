// Autoregressive rollout of a one-step forecaster (tinyforecaster, design §7): each prediction is appended to the
// history and becomes input for the next one, so errors can compound.
//   pred_h = f(last `window` values of history ++ pred_0..pred_{h-1}),  err_h = pred_h − truth_h
// Coarse: 'start', then one 'predict' frame per step. Fine: 'window' (the numbers the model reads) before each 'predict'.
// The predictions so far live in `path` [horizon], tracked by changes (NaN until predicted).
import type { I18n, Tensor } from '../../types/core.ts';
import type { Control, Frame, ProcessDef, RunCtx } from '../../types/process.ts';

export type RolloutOpt = { window: number; horizon: number; start: number };
export type RolloutInput = { series: ArrayLike<number> };
export type RolloutState = { window?: Tensor; pred?: number; truth?: number; err?: number; path: Tensor; h: number };

const L = (en: string, zh: string): I18n => ({ en, zh });

export function rolloutProcess(def: {
	id: string;
	title?: I18n;
	/** one-step-ahead forecast from the last `window` values (oldest first) */
	predict: (window: Float64Array) => number;
	defaults?: Partial<RolloutOpt>;
}): ProcessDef<RolloutInput, RolloutOpt, RolloutState> {
	return {
		id: def.id,
		kind: 'algorithm',
		title: def.title ?? L('Forecast, append, repeat', '预测、追加、重复'),
		phases: {
			start: { label: L('The history the model knows', '模型知道的历史'), kind: 'input' },
			window: { label: L('Read the last values', '读出最近的数值'), kind: 'observe' },
			predict: { label: L('Predict the next value', '预测下一个值'), kind: 'output' }
		},
		schema: {
			window: { kind: 'tensor', label: L('Values the model reads', '模型读到的数值'), shape: ['L'], role: 'data', tol: 1e-12 },
			pred: { kind: 'scalar', label: L('Prediction', '预测'), role: 'output', tol: 1e-9 },
			truth: { kind: 'scalar', label: L('True value', '真实值'), role: 'target', tol: 'exact' },
			err: { kind: 'scalar', label: L('Error', '误差'), tol: 1e-9 },
			path: { kind: 'tensor', label: L('Predictions so far', '目前的预测'), shape: ['H'], track: 'changes', role: 'output', tol: 1e-9 },
			h: { kind: 'int', label: L('Steps ahead', '向前几步'), tol: 'exact' }
		},
		defaults: { window: 24, horizon: 8, start: 24, ...(def.defaults ?? {}) },
		*run(input: RolloutInput, ctx: RunCtx<RolloutOpt>): Generator<Frame<RolloutState>, void, Control<RolloutOpt> | undefined> {
			const { window: W, horizon: H } = ctx.opt;
			const start = Math.max(W, Math.floor(ctx.opt.start));
			const series = Float64Array.from(input.series);
			if (series.length < W) throw new Error(`${def.id}: the series has ${series.length} values, the window needs ${W}`);
			const hist = Array.from(series.subarray(0, Math.min(start, series.length)));
			const path: Tensor = { shape: [H], data: new Float64Array(H).fill(NaN) };
			yield { phase: 'start', state: { path, h: 0 }, metrics: { start, horizon: H } };
			let absSum = 0;
			for (let h = 0; h < H && !ctx.signal.cancelled; h++) {
				const win = Float64Array.from(hist.slice(hist.length - W));
				if (ctx.detail === 'fine') yield { phase: 'window', step: h, state: { window: { shape: [W], data: win.slice() }, h } };
				const pred = def.predict(win);
				const ti = start + h;
				const truth = ti < series.length ? series[ti] : NaN;
				const err = pred - truth;
				path.data[h] = pred;
				hist.push(pred);
				if (Number.isFinite(err)) absSum += Math.abs(err);
				yield {
					phase: 'predict',
					step: h,
					state: { pred, truth, err, h: h + 1 },
					changes: [{ field: 'path', index: [h], from: NaN, to: pred }],
					metrics: { pred, truth, err, absErr: Math.abs(err), mae: absSum / (h + 1) }
				};
			}
		}
	};
}
