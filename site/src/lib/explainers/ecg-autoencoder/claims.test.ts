// ecg-autoencoder claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, over every fixture where it depends on the input (withFixtures: the drawn fixture,
// the adapter's empty / flat / short / top, the 6 presets). `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/ecg-autoencoder/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { dims } from './lesson.ts';
import { KINDS, N, beat, paramsOf } from './beats.ts';
import { GOAL, ODD, THR0, THR_MAX, THR_MIN, THR_STEP, countAt, errorsOf, rebuildOf, recOf, rocOf, testsetOf, thrOf } from './ecg.ts';
import { sweep, data } from './process.ts';
import { PRESETS } from './adapters/series/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('ecg-autoencoder', SITE).then((r) => r.pkg);
const rtP = pkgP.then((pkg) => createRuntime(pkg, opts));
const ST = (a: ArrayLike<number>) => Array.from(a).slice(60, 76).reduce((s, v) => s + v, 0) / 16; // the ST segment's mean height

test('structure: 140 heights -> 32 -> a 4-number code -> 32 -> 140; the parameter count; 4,000 normal training beats', async () => {
	const rt = await rtP;
	assert.deepEqual(dims(rt), { n: 140, h: 32, c: 4 });
	assert.equal(rt.num('fact:params'), rt.spec!.params);
	assert.equal(manifest.catalog!.params!.en, rt.spec!.params.toLocaleString('en-US'));
	assert.equal(rt.num('fact:code'), 4, 'the titles and looks say 4 numbers');
	assert.equal(rt.num('fact:trainN'), 4000);
	const kb = rt.spec!.tensors.reduce((s, t) => s + t.length, 0) * 4 / 1000;
	assert.ok(Math.round(kb) === 38, `catalog size: 38 KB of weights (${kb} KB)`);
});

test('the test set: the TS model on the TS beats gives the scores PyTorch wrote (testset.json), within 1e-5 relative', async () => {
	const rt = await rtP, s = testsetOf(rt)!;
	assert.equal(s.scores.length, 400);
	let worst = 0;
	s.kinds.forEach((k, i) => {
		const x = beat(KINDS[k], createRng(2026).fork(`test${i}`));
		const xr = rt.model!.forward({ input: { shape: [N], data: x } }, { record: 'layers' }).t.out.data;
		worst = Math.max(worst, Math.abs(errorsOf(x, xr).score - s.scores[i]) / s.scores[i]);
	});
	assert.ok(worst < 1e-5, `largest relative difference ${worst}`);
	const a = countAt(s, 0);
	assert.deepEqual([a.neg, a.pos, ...a.of.slice(1)], [200, 200, 50, 50, 50, 50], 'step "threshold": 200 normal, 200 odd, 50 of each kind');
});

test('steps "error", "score", "draw": the errors, the score and the verdict on every fixture', async () => {
	await withFixtures(await pkgP, (rt) => {
		const r = recOf(rt);
		if (!r) return assert.ok(rt.pre?.empty, 'no rebuild only on an empty input');
		assert.equal(r.e.length, N);
		const ss = r.e.reduce((t, v) => t + v * v, 0);
		assert.ok(Math.abs(r.score - Math.sqrt(ss / N)) < 1e-12, 'score = the root of the mean of the squared errors');
		assert.ok(r.e.every((v) => Math.abs(v) <= Math.abs(r.e[r.worst])), 'the ringed point has the largest error');
		const thr = thrOf(rt.input);
		assert.ok(thr >= THR_MIN && thr <= THR_MAX);
	}, opts);
});

test('step "odd" and the predict answer: the rebuild of a beat with a raised ST segment misses most of the raise', async () => {
	const rt = await rtP, odd = rebuildOf(rt.model!, ODD.kind, ODD.seed), twin = rebuildOf(rt.model!, 'normal', ODD.seed);
	const raise = ST(odd.x) - ST(twin.x), rebuilt = ST(odd.xr) - ST(twin.xr);
	assert.ok(raise > 0.1, `the preset's ST segment is raised by ${raise}`);
	assert.ok(rebuilt < 0.5 * raise, `the rebuild lifts it by ${rebuilt}, less than half of ${raise}`);
	assert.ok(odd.worst >= 60 && odd.worst < 76, `the largest error (point ${odd.worst}) is over the raised stretch`);
	assert.ok(odd.score > 1.5 * twin.score && odd.score > THR0, 'the gap is large: the odd beat scores well above its normal twin and the threshold');
	for (let s = 1; s <= 200; s++) assert.equal(paramsOf('normal', createRng(s)).st, 0, 'no normal (training) beat has a raised ST segment');
	// "they vary": the 12 training beats of the first panel differ in spike height, timing and noise
	const g = record(data, null, data.defaults, { seed: 2026 }), b = (g.state(0) as { beats: { data: Float32Array } }).beats.data;
	const tops = Array.from({ length: 12 }, (_, i) => Math.max(...b.subarray(i * N, (i + 1) * N)));
	assert.ok(Math.max(...tops) - Math.min(...tops) > 0.15, 'spikes of different heights');
	const ps = Array.from({ length: 12 }, (_, i) => paramsOf('normal', createRng(2026).fork(`train${i}`)));
	assert.ok(Math.max(...ps.map((p) => p.T.c)) - Math.min(...ps.map((p) => p.T.c)) > 0.02 && Math.max(...ps.map((p) => p.noise)) > 1.5 * Math.min(...ps.map((p) => p.noise)), 'earlier or later, more or less noise');
	assert.ok(Array.from({ length: 12 }, (_, i) => b.subarray(i * N, (i + 1) * N)).every((y, i) => y.every((v, j) => v === beat('normal', createRng(2026).fork(`train${i}`))[j])), 'the gallery beats are the first training beats');
});

test('step "threshold" and the task: the piles overlap; the goal is reachable on the slider and not met at the start', async () => {
	const rt = await rtP, s = testsetOf(rt)!;
	const normal = s.scores.filter((_, i) => s.kinds[i] === 0), odd = s.scores.filter((_, i) => s.kinds[i] > 0);
	assert.ok(Math.min(...odd) < Math.max(...normal), 'no threshold separates the two piles completely');
	assert.ok(rt.num('fact:thr95') <= THR0, '95 in 100 normal validation beats score below the starting threshold');
	const met = (thr: number) => { const a = countAt(s, thr); return a.tp >= GOAL.tp && a.fp <= GOAL.fp; };
	assert.ok(!met(THR0), 'the start does not meet the goal');
	const task = lesson.steps.find((st) => 'id' in st && st.id === 'threshold') as { task: { prompt: { en: string; zh: string } } };
	for (const txt of [task.task.prompt.en, task.task.prompt.zh]) for (const v of [GOAL.tp, GOAL.fp, countAt(s, 0).pos]) assert.ok(txt.includes(String(v)), `the task prompt states ${v}`);
	const grid = Array.from({ length: Math.round((THR_MAX - THR_MIN) / THR_STEP) + 1 }, (_, i) => +(THR_MIN + i * THR_STEP).toFixed(4));
	assert.ok(grid.includes(THR0) && grid.filter(met).length >= 2, `slider positions that meet the goal: ${grid.filter(met).join(', ')}`);
	// quiz 1: a lower threshold catches more odd beats and raises more false alarms
	const lo = countAt(s, 0.025), hi = countAt(s, THR0);
	assert.ok(lo.tp > hi.tp && lo.fp > hi.fp);
});

test('step "roc": the sweep matches the ROC of the test set, and its area is the chance an odd beat outscores a normal one', async () => {
	const rt = await rtP, s = testsetOf(rt)!;
	const tape = record(sweep, { scores: s.scores, kinds: s.kinds }, {}, { seed: 1 });
	const roc = rocOf(s), last = tape.frame(tape.length - 1);
	assert.equal(last.phase, 'area');
	for (let k = 1; k < tape.length - 1; k++) {
		const p = (tape.state(k) as { point: { data: Float64Array } }).point.data;
		assert.ok(p[0] === roc[k][0] && p[1] === roc[k][1], `frame ${k}`);
	}
	let wins = 0;
	for (let i = 0; i < 400; i++) for (let j = 0; j < 400; j++) if (s.kinds[i] > 0 && s.kinds[j] === 0) wins += s.scores[i] > s.scores[j] ? 1 : s.scores[i] === s.scores[j] ? 0.5 : 0;
	assert.ok(Math.abs(last.metrics!.auc - wins / (200 * 200)) < 1e-12, 'trapezoid area = share of (odd, normal) pairs where the odd beat scores higher');
	assert.ok(Math.abs(last.metrics!.auc - rt.num('fact:testAuc')) < 1e-9, 'the same AUC train.py measured');
	// step "score": odd beats tend to score higher, but not always
	assert.ok(rt.num('fact:medianOdd') > rt.num('fact:medianNormal') && Math.min(...s.scores.filter((_, i) => s.kinds[i] > 0)) < rt.num('fact:medianNormal'), 'some odd beats score below the median normal beat');
});

test('step "limits", the try-this and the misconception: noisy beats score high, missing P waves often pass, the code size', async () => {
	const rt = await rtP, s = testsetOf(rt)!, a = countAt(s, THR0);
	// the try-this setups are presets of the input widget (the adapter's PRESETS, presetsFromAdapter)
	for (const tt of manifest.page!.tryThis!) assert.ok(PRESETS.some((p) => p.id === (tt.setup as { preset?: string }).preset), `try-this ${tt.id} opens a preset`);
	const pre = (id: string) => { const p = PRESETS.find((q) => q.id === id)!; return rebuildOf(rt.model!, p.kind, p.seed); };
	const noisy = pre('noisy');
	assert.ok(noisy.score > 1.5 * THR0, `the noisy preset scores far above the threshold (${noisy.score})`);
	const med = (k: number) => { const v = s.scores.filter((_, i) => s.kinds[i] === k).sort((x, y) => x - y); return (v[24] + v[25]) / 2; };
	assert.ok(med(4) > THR0, 'a noisy recording scores high: the median noisy test beat is flagged');
	assert.ok(a.of[1] - a.per[1] >= a.of[1] / 3 && a.of[3] - a.per[3] >= a.of[3] / 3, `small changes often pass at the starting threshold: P ${a.of[1] - a.per[1]} of ${a.of[1]}, ST ${a.of[3] - a.per[3]} of ${a.of[3]}`);
	for (let s2 = 1; s2 <= 200; s2++) { const p = paramsOf('noP', createRng(s2)), q = paramsOf('normal', createRng(s2)); assert.ok(p.P.a <= 0.6 * q.P.a + 1e-12, 'the P wave is shrunk (to at most 60%) or missing'); }
	const noP = pre('noP').score;
	assert.ok(noP >= THR0 && noP < 1.15 * THR0, `try-this noP: the preset only just crosses the starting threshold (${noP})`);
	const auc = (k: string) => rt.num(`fact:${k}`), err = (k: string) => rt.num(`fact:${k}`);
	assert.ok(auc('testAuc') > auc('c2Auc') && auc('testAuc') > auc('c8Auc') && auc('testAuc') > auc('c16Auc'), 'the AUC peaks at 4');
	assert.ok(auc('c16Auc') < auc('c8Auc') && err('c16TrainScore') < err('trainScore'), 'with 16 numbers normal beats are rebuilt more closely, but odd beats stand out less: the AUC falls');
	assert.ok(err('c2TrainScore') > err('trainScore') && err('trainScore') > err('c8TrainScore') && err('c8TrainScore') > err('c16TrainScore'), 'the rebuild error on normal beats keeps falling');
	assert.ok(noP < pre('wideQRS').score, 'try-this: a small part leaves a smaller error than a wide QRS');
});
