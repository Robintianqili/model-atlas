// tinyforecaster: trainer 'lstm' (kind B; process 'train:lstm'). The forecaster of model.ts learns, live, to predict the
// value after each training window of the learner's series (data.ts: windows of L = 24 values that end before the
// held-out part). Adam lr 0.01, 16 windows drawn at random per step, mean squared error, 200 steps (the e2e cap, so the
// screenshots show the same numbers as a browser), a snapshot every 10.
// Metrics at the start and at every snapshot, over every training window: `mse` (the loss on all of them) and `mae`
// (the average size of a forecast's miss). Twin: pipeline/tinyforecaster/twin.py.
//   npm run atlas -- run tinyforecaster train --frames 0..20
import type { Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import { input } from '../../atlas/autograd/var.ts';
import { dataOf, L, type Data } from './data.ts';
import { Forecaster, forecastAll } from './model.ts';

export type Opt = TrainOpt & { data?: unknown };

const sets = new WeakMap<Data, { x: Float64Array; y: Float64Array; count: number }>();
/** Every training window as rows [windows, L] and their targets [windows] (one copy per series; read-only). */
export function trainingSet(d: Data): { x: Float64Array; y: Float64Array; count: number } {
	const hit = sets.get(d);
	if (hit) return hit;
	const count = d.windows;
	const x = new Float64Array(count * L);
	const y = new Float64Array(count);
	for (let k = 0; k < count; k++) {
		x.set(d.y.subarray(k, k + L), k * L);
		y[k] = d.y[k + L];
	}
	const set = { x, y, count };
	sets.set(d, set);
	return set;
}

/** The series a run uses: opt.data (a fixed series, e.g. the random walk of the last step) else the learner's. */
export const runData = (raw: unknown, opt?: { data?: unknown }) => dataOf(opt?.data ?? raw);

/** mse and mae of the one-step forecasts over every training window. */
export function fitOf(m: Forecaster, d: Data): { mse: number; mae: number } {
	const s = trainingSet(d);
	const p = forecastAll(m, s.x, s.count);
	let se = 0;
	let ae = 0;
	for (let k = 0; k < s.count; k++) {
		const e = p[k] - s.y[k];
		se += e * e;
		ae += Math.abs(e);
	}
	return { mse: se / s.count, mae: ae / s.count };
}

export const lstm: TrainerDef<unknown, Forecaster, { x: Tensor; y: Tensor }, Opt> = {
	id: 'lstm',
	title: { en: 'Train the forecaster on the windows', zh: '在窗口上训练预测模型' },
	defaults: { seed: 1, lr: 0.01, batch: 16, steps: 200, snapshotEvery: 10, optimizer: 'adam' },
	model: (rng) => new Forecaster(rng),
	batch(raw, _step, rng, opt) {
		const s = trainingSet(runData(raw, opt));
		const B = opt.batch;
		const x = new Float64Array(B * L);
		const y = new Float64Array(B);
		for (let i = 0; i < B; i++) {
			const k = rng.int(s.count);
			x.set(s.x.subarray(k * L, (k + 1) * L), i * L);
			y[i] = s.y[k];
		}
		return { x: { shape: [B, L], data: x }, y: { shape: [B, 1], data: y } };
	},
	loss: (m, b, rec) => ({ loss: mse(m.forward(input(b.x), rec), input(b.y)) }),
	metrics: (m, raw, opt) => fitOf(m, runData(raw, opt))
};
