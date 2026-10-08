// tinyforecaster claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, on every fixture where it is about the learner's series. Training runs 200 steps, the
// e2e cap, so the browser, the e2e screenshots and these tests see the same runs. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/tinyforecaster/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor, TrainState } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { input, noGrad } from '../../atlas/autograd/var.ts';
import { seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { dataOf, L, seasonalNaive, WALK_SEEDS } from './data.ts';
import { forecastAll, HIDDEN, netOf } from './model.ts';
import { lstm, trainingSet } from './trainer.ts';
import { forecast, walks, windows } from './process.ts';
import lesson, { startChange } from './lesson.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('tinyforecaster', SITE)).pkg;
const lastFrame = (t: Tape) => t.frame(t.length - 1);
const paramsAt = (t: Tape, i: number) => (t.state(i) as Partial<TrainState>).params!;
const series = (name: Parameters<typeof seriesPresetRaw>[0]) => Array.from(dataOf(seriesPresetRaw(name)).y);
/** One 'forecast' run per series and step count, shared by the tests (training is the slow part). */
const runs = new Map<string, Tape>();
function fc(y: number[], steps = lstm.defaults.steps): Tape {
	const key = `${steps}|${JSON.stringify(y)}`;
	let t = runs.get(key);
	if (!t) runs.set(key, (t = record(forecast, y, { steps }, { seed: 1 })));
	return t;
}
const lastSnap = (t: Tape) => {
	let j = -1;
	for (let i = 0; i < t.length; i++) if (t.frame(i).phase === 'snapshot') j = i;
	return j;
};
const walkInput = () => WALK_SEEDS.map((seed) => Array.from(dataOf(seriesPresetRaw('randomWalk', { seed })).y));
/** One run of the walks process with the page's options (shared: three trainings). */
let walkTape: Tape | undefined;
const walkRun = () => (walkTape ??= record(walks, walkInput(), {}, { seed: 1 }));

test('page: 10 steps; windows of 24 values; 16 memory numbers; 1,233 parameters; Adam lr 0.01, 16 windows a step, 200 steps', () => {
	assert.equal(lesson.steps.length, 10);
	assert.equal(L, 24);
	assert.equal(HIDDEN, 16);
	assert.equal(lstm.model(createRng(1).fork('init'), lstm.defaults).params().reduce((s, p) => s + p.size, 0), 1233);
	assert.deepEqual([lstm.defaults.optimizer, lstm.defaults.lr, lstm.defaults.batch, lstm.defaults.steps, lstm.defaults.snapshotEvery], ['adam', 0.01, 16, 200, 10]);
	assert.equal((manifest.sources!.init as unknown as { opt: { steps: number } }).opt.steps, 0, 'steps 3 to 6 use the untrained (random) starting weights');
});

test('step 1: the held-out values are never trained on (training windows and their targets end before the split)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const d = dataOf(rt.input);
			const s = trainingSet(d);
			assert.equal(s.count, d.split - L);
			assert.equal(s.count - 1 + L, d.split - 1, 'the last target is the last value of the training part');
			assert.ok(d.hold >= 1 && d.split + d.hold === d.n);
		},
		opts
	);
});

test('step 2: neighbouring windows share 23 of their 24 values (the window slides one step)', () => {
	const t = record(windows, series('seasonal'), {}, { seed: 1 });
	for (let i = 2; i < t.length; i++) {
		const a = (t.state(i - 1) as { window: Tensor }).window.data;
		const b = (t.state(i) as { window: Tensor }).window.data;
		assert.equal(t.frame(i).step! - t.frame(i - 1).step!, 1);
		for (let k = 0; k < L - 1; k++) assert.equal(b[k], a[k + 1]);
	}
	assert.equal(t.length - 1, 120 - 24 - 24, '72 windows for the 120 values of the start series');
});

test('step 3 look: training changes every gate (unit 0 at the last step; the start series and a drawn one)', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			const d = dataOf(rt.input);
			const k = d.windows - 1;
			const gates = (p: Record<string, Tensor>) => {
				const rec: Record<string, number> = {};
				noGrad(() => netOf(p, createRng(1)).forward(input({ shape: [1, L], data: d.y.slice(k, k + L) }), (id, v) => (rec[id] = v.value.data[0])));
				return rec;
			};
			for (const steps of [lstm.defaults.steps]) {
				const t = fc(Array.from(d.y), steps);
				const a = gates(paramsAt(t, 0));
				const b = gates(paramsAt(t, lastSnap(t)));
				for (const g of ['i', 'f', 'g', 'o']) assert.ok(Math.abs(a[`lstm#${g}`] - b[`lstm#${g}`]) > 1e-6, `${fx.id} @${steps}: gate ${g}`);
			}
		},
		{ ...opts, only: ['preset:seasonal', 'drawn'] }
	);
});

test('step 3 look and step 4: the last h of memory number 0 is o·tanh(c) and row 0 of what the head reads', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			await rt.ensure('net0', 'last');
			const t = rt.sources.net0.trace!.t;
			const last = (id: string) => t[id].data[(L - 1) * HIDDEN];
			assert.ok(Math.abs(last('lstm#h') - last('lstm#o') * Math.tanh(last('lstm#c'))) <= 1e-12, fx.id);
			assert.equal(t.last.data[0], last('lstm'), `${fx.id}: row 0 of the head's input is that h`);
		},
		opts
	);
});

test('steps 4-5: forecast = last value + predicted change, on the random starting weights (net0 = the seeded init)', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return assert.equal(fx.id, 'edge:empty');
			await rt.ensure('net0', 'last');
			const tr = rt.sources.net0.trace!;
			assert.ok(Math.abs(tr.t.out.data[0] - (tr.t.lastv.data[0] + tr.t.head.data[0])) <= 1e-15);
			await rt.ensure('init', 'last');
			const d = dataOf(rt.input);
			const k = d.windows - 1;
			const f = forecastAll(netOf(paramsAt(rt.sources.init.tape!, 0), createRng(1)), d.y.slice(k, k + L), 1)[0];
			assert.ok(Math.abs(f - tr.t.out.data[0]) <= 1e-12, `${fx.id}: trace ${tr.t.out.data[0]} vs init weights ${f}`);
		},
		opts
	);
});

test('step 6: the mse metric is the mean squared miss over all training windows; the look holds (changes stay small)', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			await rt.ensure('train', 'first');
			const t = rt.sources.train.tape!;
			const d = dataOf(rt.input);
			const s = trainingSet(d);
			const f = forecastAll(netOf(paramsAt(t, 0), createRng(1)), s.x, s.count);
			const mse = f.reduce((a, v, k) => a + (v - s.y[k]) ** 2, 0) / s.count;
			assert.ok(Math.abs(mse - t.frame(0).metrics!.mse) <= 1e-12);
			// "follow the series one step late": every starting forecast is within 0.1 of its window's last value
			assert.ok(startChange(d, paramsAt(t, 0)) < 0.1, `${fx.id}: ${startChange(d, paramsAt(t, 0))}`);
		},
		opts
	);
});

test('step 7: the mean squared error over all windows falls', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			for (const steps of [lstm.defaults.steps]) {
				const t = fc(Array.from(dataOf(rt.input).y), steps);
				const last = t.frame(lastSnap(t)).metrics!.mse;
				assert.ok(last < t.frame(0).metrics!.mse, `${fx.id} @${steps}: ${t.frame(0).metrics!.mse} -> ${last}`);
			}
		},
		opts
	);
});

test('step 7: the train source and the forecast process train the same network (same weights, same mse)', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			await rt.ensure('train', 'last');
			const a = rt.sources.train.tape!;
			const b = fc(Array.from(dataOf(rt.input).y));
			const pa = paramsAt(a, a.length - 1);
			const pb = paramsAt(b, lastSnap(b));
			for (const k of Object.keys(pa)) assert.deepEqual(Array.from(pa[k].data), Array.from(pb[k].data), `${fx.id}: ${k}`);
		},
		{ ...opts, only: ['preset:seasonal', 'edge:one-value'] }
	);
});

test('step 8: each forecast reads the forecasts before it (a rollout), on every fixture', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			const d = dataOf(rt.input);
			const t = fc(Array.from(d.y));
			const m = netOf(paramsAt(t, lastSnap(t)), createRng(1));
			const hist = Array.from(d.y.subarray(0, d.split));
			const path = (t.state(t.length - 1) as { path: Tensor }).path.data;
			for (let h = 0; h < d.hold; h++) {
				const p = forecastAll(m, hist.slice(-L), 1)[0];
				assert.equal(path[h], p, `${fx.id} forecast ${h}`);
				hist.push(p);
			}
		},
		opts
	);
});

test('step 9: m is the lag (1 to 24) with the smallest one-step miss on the training part; m = 1 is “the same as now”; the rule repeats the last m known values and never reads the held-out part', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			const d = dataOf(rt.input);
			const nv = seasonalNaive(d);
			for (let lag = 1; lag <= L; lag++) {
				let s = 0;
				for (let t = L; t < d.split; t++) s += Math.abs(d.y[t] - d.y[t - lag]);
				assert.ok(s / (d.split - L) >= nv.fit - 1e-12, `${fx.id}: lag ${lag} beats m = ${nv.m}`);
			}
			if (nv.m === 1) for (const v of nv.path) assert.equal(v, d.y[d.split - 1]);
			for (let k = 0; k < d.hold; k++) {
				const from = d.split + k - nv.m * (Math.floor(k / nv.m) + 1);
				assert.ok(from >= d.split - nv.m && from < d.split && nv.path[k] === d.y[from], `${fx.id}: forecast ${k} copies position ${from}`);
			}
		},
		opts
	);
});

test('step 9 task and quiz: on the sine wave the rule wins; on the start series the network wins', () => {
	for (const steps of [lstm.defaults.steps]) {
		const sine = lastFrame(fc(series('sine'), steps)).metrics!;
		assert.ok(sine.naiveMae < 1e-9 && sine.lstmMae > sine.naiveMae, `sine @${steps}: ${sine.lstmMae} vs ${sine.naiveMae}`);
		const start = lastFrame(fc(series('seasonal'), steps)).metrics!;
		assert.ok(start.lstmMae < start.naiveMae, `start @${steps}: ${start.lstmMae} vs ${start.naiveMae}`);
	}
	const d = dataOf(seriesPresetRaw('sine'));
	assert.deepEqual([d.n, d.hold, d.windows], [96, 24, 48], 'quiz: 96 values, 24 held out, 48 windows');
});

test('step 10, its look, the prequestion and the misconception: on the 3 walks the network fits the training parts better than repeating the last value and forecasts new values worse; it wins only on walk 1 (the Random walk preset, by luck) and, where it loses, expects rises while the walk falls', () => {
	for (const steps of [lstm.defaults.steps]) {
		const t = steps === lstm.defaults.steps ? walkRun() : record(walks, walkInput(), { steps }, { seed: 1 });
		assert.equal(t.length, 3);
		const m = lastFrame(t).metrics!;
		assert.ok(m.avgFit < m.avgPersistFit, `@${steps}: fit ${m.avgFit} vs ${m.avgPersistFit}`);
		assert.ok(m.avgOne > m.avgPersist, `@${steps}: one ${m.avgOne} vs ${m.avgPersist}`);
		const won: number[] = [];
		for (let i = 0; i < t.length; i++) {
			const f = t.frame(i);
			const s = t.state(i) as { series: Tensor; one: Tensor };
			const y = s.series.data;
			const split = f.metrics!.split;
			if (f.metrics!.one < f.metrics!.persist) {
				won.push(f.sub!);
				continue;
			}
			// a walk it loses: its forecasts lean up (mean predicted change > 0) while the held-out part ends lower than it starts
			const change = Array.from(s.one.data).reduce((a, v, k) => a + v - y[split + k - 1], 0) / s.one.data.length;
			assert.ok(change > 0 && y[y.length - 1] < y[split - 1], `walk ${f.sub}: change ${change}, ${y[split - 1]} -> ${y[y.length - 1]}`);
		}
		// the prequestion's "except by luck": one walk is won, so either side can win one walk
		assert.deepEqual(won, [1], `@${steps}: won ${won}`);
	}
	// "the same for everyone; walk 1 is the Random walk preset": the walks do not depend on the learner's input
	assert.deepEqual(walkInput(), (manifest.sources!.walks as unknown as { input: () => unknown }).input());
	assert.deepEqual(walkInput()[0], series('randomWalk'));
	// "each value is the last one plus a fresh random step": the preset is a Gaussian random walk (step 0.1)
	const y = walkInput()[0];
	const steps = y.slice(1).map((v, i) => v - y[i]);
	const sd = Math.sqrt(steps.reduce((s, v) => s + v * v, 0) / steps.length);
	assert.ok(sd > 0.08 && sd < 0.12, `step size ${sd}`);
});

test('step 10: the recorded walks (source walksSaved, static/models/tinyforecaster/tapes/walks) equal a fresh run of the walks process', async () => {
	const fresh = walkRun();
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('walksSaved', 'last');
			const t = rt.sources.walksSaved.tape!;
			assert.equal(t.length, fresh.length);
			for (let i = 0; i < t.length; i++) {
				assert.deepEqual([t.frame(i).phase, t.frame(i).step, t.frame(i).sub, t.frame(i).metrics], [fresh.frame(i).phase, fresh.frame(i).step, fresh.frame(i).sub, fresh.frame(i).metrics]);
				const a = t.state(i) as { series: Tensor; one: Tensor };
				const b = fresh.state(i) as { series: Tensor; one: Tensor };
				assert.deepEqual([Array.from(a.series.data), Array.from(a.one.data)], [Array.from(b.series.data), Array.from(b.one.data)], `walk ${i + 1}`);
			}
		},
		{ ...opts, only: ['preset:seasonal'] }
	);
});

test('step 5 and try this: the square wave jumps every 8 steps and its late forecasts miss far more; the trend goes on rising', () => {
	const sq = series('square');
	for (let i = 0; i + 8 < sq.length; i += 8) assert.ok(sq.slice(i, i + 8).every((v) => v === sq[i]) && sq[i + 8] === -sq[i]);
	for (const steps of [lstm.defaults.steps]) {
		const m = lastFrame(fc(sq, steps)).metrics!;
		assert.ok(m.late > 10 * m.early, `square @${steps}: ${m.early} -> ${m.late}`);
		const d = dataOf(seriesPresetRaw('trend'));
		const t = fc(Array.from(d.y), steps);
		const path = (t.state(t.length - 1) as { path: Tensor }).path.data;
		assert.ok(path.every((v, k) => k === 0 || v > path[k - 1]), 'the forecast keeps rising');
		assert.ok(path[path.length - 1] > Math.max(...d.y.subarray(0, d.split)) + 0.4, 'far above every training value');
		assert.ok(lastFrame(t).metrics!.lstmMae < 1e-3);
	}
});
