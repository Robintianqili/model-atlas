// tiny-tcn: trainer 'tcn' (kind B; process 'train:tcn'). The TCN of model.ts learns, live, to predict the next change
// at every training position of the learner's series at once: one pass over the changes of the training part gives
// all `count` outputs (t = RF..split−2), and the loss is their mean squared error against c_{t+1}. Full batch: every
// step reads the same data, so no step draws anything. Adam lr 0.01, 200 steps (the e2e cap, so the screenshots show
// the same numbers as a browser), a snapshot every 10. Metrics at the start and at every snapshot: `mse` (the same
// loss, all positions) and `mae` (the average size of a one-step miss). Twin: pipeline/tiny-tcn/twin.py.
//   npm run atlas -- run tiny-tcn train --frames 0..20
import type { Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { input } from '../../atlas/autograd/var.ts';
import { dataOf, RF, type Data } from './data.ts';
import { forwardPlain, TCN, type Params } from './model.ts';

export type Opt = TrainOpt & { data?: unknown };

/** The series a run uses: opt.data (the limits step's fixed series) else the learner's. */
export const runData = (raw: unknown, opt?: { data?: unknown }) => dataOf(opt?.data ?? raw);

/** The network's input (the changes of the training part, [1, 1, split]) and the targets c_{t+1} of t = RF..split−2. */
export function batchOf(d: Data): { x: Tensor; y: Tensor } {
	return { x: { shape: [1, 1, d.split], data: d.c.slice(0, d.split) }, y: { shape: [1, 1, d.count], data: d.c.slice(RF + 1, d.split) } };
}

/** mse and mae of the one-step forecasts over every training position (plain numbers). */
export function fitOf(P: Params, d: Data): { mse: number; mae: number } {
	const out = forwardPlain(P, d.c.subarray(0, d.split)).out;
	let se = 0;
	let ae = 0;
	for (let t = RF; t <= d.split - 2; t++) {
		const e = out[t] - d.c[t + 1];
		se += e * e;
		ae += Math.abs(e);
	}
	return { mse: se / d.count, mae: ae / d.count };
}

export const paramsOf = (m: TCN): Params => Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, { shape: v.shape, data: v.data }]));

export const tcn: TrainerDef<unknown, TCN, { x: Tensor; y: Tensor }, Opt> = {
	id: 'tcn',
	title: { en: 'Train the TCN on every position at once', zh: '在所有位置上同时训练 TCN' },
	defaults: { seed: 1, lr: 0.01, batch: 1, steps: 200, snapshotEvery: 10, optimizer: 'adam' },
	model: (rng) => new TCN(rng),
	batch: (raw, _step, _rng, opt) => batchOf(runData(raw, opt)),
	loss(m, b, rec) {
		const T = b.x.shape[2];
		const n = b.y.shape[2];
		// the outputs at t = T − 1 − n .. T − 2 (= RF .. split − 2) predict the targets c_{t+1}
		return { loss: mse(op.slice(m.forward(input(b.x), rec), 2, T - 1 - n, T - 1), input(b.y)) };
	},
	metrics: (m, raw, opt) => fitOf(paramsOf(m), runData(raw, opt))
};
