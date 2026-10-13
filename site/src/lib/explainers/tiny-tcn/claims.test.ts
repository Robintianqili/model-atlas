// tiny-tcn claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, on every fixture where it is about the learner's series. Training runs 200 steps, the e2e cap,
// so the browser, the e2e screenshots and these tests see the same runs. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/tiny-tcn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, TrainState } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { input, noGrad } from '../../atlas/autograd/var.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { seriesPresetRaw, type SeriesPresetName } from '../../atlas/adapters/series/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { DILATIONS, KERNEL, LONG_PERIOD, RF, dataOf, longSquare } from './data.ts';
import { CH, forwardPlain, TCN, type Params } from './model.ts';
import { batchOf, paramsOf, tcn } from './trainer.ts';
import { fitArima, forecastArima, P_MAX, Q_MAX, stable } from './arima.ts';
import { arimaOf, tcnPath } from './process.ts';
import { headAt, longStretch, nudged, tStar, treeOffsets } from './look.ts';
import lesson from './lesson.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('tiny-tcn', SITE)).pkg;
const train = trainingProcess(tcn);
const runs = new Map<string, Tape>();
/** One training run per series (seed 1, the page's options), shared by the tests. */
function trained(raw: unknown): Tape {
	const key = JSON.stringify(raw);
	let t = runs.get(key);
	if (!t) runs.set(key, (t = record(train, raw, {}, { seed: 1 })));
	return t;
}
const lastParams = (t: Tape) => {
	let p: Params | undefined;
	for (let i = 0; i < t.length; i++) p = (t.state(i) as Partial<TrainState>).params ?? p;
	return p!;
};
const firstParams = (t: Tape) => (t.state(0) as TrainState).params;
const preset = (name: SeriesPresetName) => seriesPresetRaw(name);
const compare = (raw: unknown) => {
	const d = dataOf(raw);
	const mae = (p: number[]) => p.reduce((s, v, h) => s + Math.abs(v - d.y[d.split + h]), 0) / d.hold;
	return { d, tcn: mae(tcnPath(lastParams(trained(raw)), d)), arima: arimaOf(d).mae };
};

test('page: 11 steps; kernel 2, dilations 1/2/4/8, reach 16; 8 features; 441 weights; Adam lr 0.01, 200 steps; ARIMA p ≤ 5, q ≤ 2', () => {
	assert.equal(lesson.steps.length, 11);
	const at = (id: string) => (lesson.steps as { id?: string }[]).findIndex((x) => x.id === id) + 1;
	assert.deepEqual([at('head'), at('train'), at('arima')], [7, 8, 9], 'the narration names step 7 (the followed forecast), step 8 (training) and step 9 (ARIMA)');
	assert.deepEqual([KERNEL, ...DILATIONS, RF, CH], [2, 1, 2, 4, 8, 16, 8]);
	assert.equal(1 + DILATIONS.reduce((s, d) => s + d, 0), 16, 'pre-question and step 4: 1 + 1 + 2 + 4 + 8 = 16');
	assert.equal(1 + [...DILATIONS, 16].reduce((s, d) => s + d, 0), 32, 'quiz and step 11 look: a fifth layer with dilation 16 reaches 32');
	assert.ok(32 > LONG_PERIOD / 2, 'step 11 look: 32 changes reach the last jump, 20 steps back');
	assert.equal(new TCN(createRng(1).fork('init')).params().reduce((s, p) => s + p.size, 0), 441);
	assert.deepEqual([tcn.defaults.optimizer, tcn.defaults.lr, tcn.defaults.steps], ['adam', 0.01, 200]);
	assert.deepEqual([P_MAX, Q_MAX], [5, 2]);
	assert.equal((manifest.sources!.init as unknown as { opt: { steps: number } }).opt.steps, 0, 'steps 3 to 7 use the untrained (random) starting weights');
	assert.deepEqual([3, 2, 1, 0, -1].map((l) => treeOffsets(l).length), [1, 2, 4, 8, 16], 'step 5: 1, 2, 4, 8 nodes and 16 changes');
	assert.deepEqual(treeOffsets(-1), Array.from({ length: 16 }, (_, i) => i));
});

test('the plain-number network (views and narration) equals the autograd forward, ≤ 1e-12, on every fixture', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			const d = dataOf(rt.input);
			for (const P of [firstParams(trained(rt.input)), lastParams(trained(rt.input))]) {
				const m = new TCN(createRng(1));
				for (const [k, v] of Object.entries(m.named())) v.data.set(P[k].data);
				const b = batchOf(d);
				const y = noGrad(() => m.forward(input(b.x))).data;
				const p = forwardPlain(P, d.c.subarray(0, d.split)).out;
				for (let i = 0; i < d.split; i++) assert.ok(Math.abs(y[i] - p[i]) <= 1e-12, `${fx.id} @${i}`);
			}
		},
		opts
	);
});

test('steps 3-5: causal and a reach of exactly 16 changes: later and older changes leave the output at t* unchanged; the newest moves it', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (rt.pre?.empty) return;
			const d = dataOf(rt.input);
			const t = tStar(d);
			for (const P of [firstParams(trained(rt.input)), lastParams(trained(rt.input))]) {
				const base = forwardPlain(P, d.c.subarray(0, d.split)).out;
				const later = d.c.slice(0, d.split);
				for (let k = t + 1; k < d.split; k++) later[k] += 5;
				const older = d.c.slice(0, d.split);
				for (let k = 0; k <= t - RF; k++) older[k] -= 3;
				assert.equal(forwardPlain(P, later).out[t], base[t], `${fx.id}: a later change reaches t*`);
				assert.equal(forwardPlain(P, older).out[t], base[t], `${fx.id}: a change older than t* − 15 reaches t*`);
				const f = headAt(P, d).forecast;
				assert.equal(nudged(P, d, RF), f, `${fx.id}: step 5 "stays exactly"`);
				assert.notEqual(nudged(P, d, 0), f, `${fx.id}: step 5 "moves"`);
			}
		},
		opts
	);
});

test('step 2 look: a straight trend has equal changes (a flat line of changes)', () => {
	const d = dataOf(preset('trend'));
	for (let t = 2; t < d.n; t++) assert.ok(Math.abs(d.c[t] - d.c[1]) < 1e-12);
});

test('step 6 look and residual blocks: h = skip + ReLU(conv); where the ReLU gives 0 the output equals the input exactly', () => {
	const d = dataOf(preset('seasonal'));
	const pl = forwardPlain(firstParams(trained(preset('seasonal'))), d.c.subarray(0, d.split));
	let zeros = 0;
	for (let l = 1; l < 4; l++)
		for (let k = 0; k < pl.h[l].length; k++) {
			assert.equal(pl.h[l][k], pl.h[l - 1][k] + pl.a[l][k]);
			if (pl.a[l][k] === 0) (zeros++, assert.equal(pl.h[l][k], pl.h[l - 1][k]));
		}
	assert.ok(zeros > 0);
});

test('step 8: the loss is the mean squared error of all training positions (= the mse metric), and it falls', () => {
	for (const name of ['seasonal', 'sine', 'square'] as SeriesPresetName[]) {
		const t = trained(preset(name));
		const f0 = t.frame(0).metrics!;
		const fl = t.frame(t.length - 1).metrics!;
		const m = new TCN(createRng(1));
		for (const [k, v] of Object.entries(m.named())) v.data.set(firstParams(t)[k].data);
		const loss = noGrad(() => tcn.loss(m, batchOf(dataOf(preset(name))), () => {}, tcn.defaults).loss).value.data[0];
		assert.ok(Math.abs(loss - f0.mse) <= 1e-12 * f0.mse, name);
		assert.ok(fl.mse < f0.mse / 10, name);
	}
});

test('step 9: ARIMA finds the exact rule of a sine (ARIMA(2, 1, 0), 3 coefficients) and of a trend; its fit only reads the training part', () => {
	const s = arimaOf(dataOf(preset('sine')));
	assert.deepEqual([s.fit.p, s.fit.q], [2, 0]);
	assert.ok(Math.abs(s.fit.phi[0] - 2 * Math.cos((2 * Math.PI) / 24)) < 1e-4 && Math.abs(s.fit.phi[1] + 1) < 1e-4);
	assert.ok(s.mae < 1e-4, `sine: ${s.mae}`);
	assert.ok(arimaOf(dataOf(preset('trend'))).mae < 1e-6);
	const d = dataOf(preset('seasonal'));
	const y2 = d.y.slice();
	for (let k = d.split; k < d.n; k++) y2[k] += 7;
	assert.deepEqual(arimaOf(dataOf(Array.from(y2))).path, arimaOf(d).path, 'the held-out part changes nothing');
	assert.ok(stable(s.fit.phi), 'kept orders are stable');
});

test('step 9: ARIMA’s AR part is one causal convolution of width p over the changes', () => {
	const d = dataOf(preset('seasonal'));
	const z = Array.from(d.c.subarray(1, d.split));
	const f = fitArima(z);
	assert.equal(f.q, 0, 'the start series picks a pure AR order');
	const next = forecastArima(f, z, d.y[d.split - 1], 1).changes[0];
	const conv = f.mu + f.phi.reduce((s, w, i) => s + w * z[z.length - 1 - i], 0);
	assert.ok(Math.abs(next - conv) < 1e-15);
});

test('step 10, tryThis and misconception: the TCN wins on the start series and the square wave; ARIMA wins on the sine and the trend', () => {
	const seas = compare(preset('seasonal'));
	assert.ok(seas.tcn < seas.arima, 'the start series: the task is not done yet');
	const sq = compare(preset('square'));
	assert.ok(sq.tcn < sq.arima, 'square: the TCN sees the jump 8 steps back, ARIMA (p ≤ 5) does not');
	for (const name of ['sine', 'trend'] as SeriesPresetName[]) {
		const c = compare(preset(name));
		assert.ok(c.arima < c.tcn && c.arima < 1e-4, `${name}: ARIMA lies on the truth and wins (the task can be done)`);
	}
});

test('step 11: on the long square every stretch of 16 zero changes gets one forecast, for any weights, and misses each jump', () => {
	const t = trained(longSquare());
	assert.deepEqual(t.frame(t.length - 1).step, tcn.defaults.steps, 'trained like the learner’s series');
	for (const P of [firstParams(t), lastParams(t)]) {
		const s = longStretch(P);
		assert.ok(s.same, 'the same forecast at every time of the stretch');
		assert.ok(s.b - s.a + 1 >= 2 && s.jump !== s.stay);
	}
	const P = lastParams(t);
	const d = dataOf(longSquare());
	const out = forwardPlain(P, d.c.subarray(0, d.split)).out;
	// every jump among the training targets c_{RF+1}..c_{split−1} (20, 40, 60, 80): each follows 16 zero changes and is missed
	let n = 0;
	for (let j = RF + 1; j <= d.split - 1; j++) {
		if (d.c[j] === 0) continue;
		n++;
		assert.equal(j % (LONG_PERIOD / 2), 0, `a jump only every ${LONG_PERIOD / 2} steps (${j})`);
		assert.ok(d.c.subarray(j - RF, j).every((v) => v === 0), `the jump at ${j} follows 16 zero changes`);
		const miss = Math.abs(d.y[j - 1] + out[j - 1] - d.y[j]);
		assert.ok(miss > 1.5, `the jump at ${j}: miss ${miss}`);
	}
	assert.equal(n, 4);
	for (const Q of [firstParams(t), P]) assert.equal(longStretch(Q).jumps, n, 'step 11: "each of the 4 jumps it trained on"');
});

test('tryThis square: the square preset jumps every 8 steps', () => {
	const d = dataOf(preset('square'));
	for (let k = 1; k < d.n; k++) assert.equal(d.c[k] !== 0, k % 8 === 0, `change at ${k}`);
});
