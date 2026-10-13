// fno claim tests: every sentence of the lesson and the page that is not computed live is asserted here, over every
// fixture where it depends on the input (withFixtures: the lesson fixtures, the adapter's empty / flat / short / spike,
// the 5 presets). `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/fno/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import { dims } from './lesson.ts';
import { AMP, FAMILIES, N, STEPS, fdGain, heightOf, initial, lowPass, presetY, solve, spectrum } from './heat.ts';
import { CH, GOAL, T, col, errByKeep, fewest, heights, learnedGain, liftOf, mixMode, predict, stateOf, taller } from './fno.ts';
import { K } from './adapters/series/adapter.ts';
import { TALL } from './views/field/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('fno', SITE).then((r) => r.pkg);
const rtP = pkgP.then((pkg) => createRuntime(pkg, opts));
const steep = (u: ArrayLike<number>) => Math.max(...Array.from(u, (v, j) => Math.abs(u[(j + 1) % u.length] - v)));
const wave = (n: number, a = 1) => Array.from({ length: N }, (_, j) => a * Math.cos((2 * Math.PI * n * j) / N));

test('structure: 64 points, 4 channels, 33 modes, 16 kept; the parameter count and the catalog size', async () => {
	const rt = await rtP;
	assert.deepEqual(dims(rt), { n: 64, w: 4, m: 33, K: 16, steps: 50 });
	assert.equal(K, 16, 'the slider and the adapter use the spec’s 16 modes');
	assert.equal(rt.num('fact:params'), rt.spec!.params);
	assert.equal(manifest.catalog!.params!.en, rt.spec!.params.toLocaleString('en-US'));
	const kb = readFileSync(join(SITE, 'static/models/fno/weights.bin')).length / 1000;
	assert.equal(Math.round(kb), 4, `catalog size: 4 KB of weights (${kb} KB)`);
});

test('the solver: a wave of mode n shrinks by fdGain(n); short waves fade first (prequestion, steps "solver", "keep", quiz 1)', () => {
	for (let n = 0; n <= N / 2; n++) {
		const s = spectrum(solve(wave(n)));
		assert.ok(Math.abs(heightOf(s.re[n], s.im[n], n) - fdGain(n)) < 1e-9, `mode ${n}`);
		if (n > 0) assert.ok(fdGain(n) < fdGain(n - 1), 'every higher mode keeps a smaller share');
	}
	assert.ok(fdGain(K) < 1e-4, 'the dropped modes were about to vanish: mode 16 keeps less than 0.01%');
	for (const id of ['square', 'waves']) assert.ok(steep(solve(presetY(id))) < 0.5 * steep(presetY(id)), `${id}: sharp corners and wiggles flatten`);
	const bump = (w: number) => Array.from({ length: N }, (_, j) => Math.exp(-0.5 * ((j - 32) / (N * w)) ** 2));
	assert.ok(Math.max(...solve(bump(0.05))) < 0.8 && Math.max(...solve(bump(0.2))) > 0.95, 'a narrow bump sinks, a wide hill only slowly');
	const w = spectrum(solve(presetY('waves')));
	assert.ok(heightOf(w.re[12], w.im[12], 12) < 1e-3 && heightOf(w.re[2], w.im[2], 2) > 0.8 * 0.6, 'tryThis "waves": the wiggle is gone, the slow wave shrank a little');
});

test('steps "lift", "fft", "keep", "mix": channels are copies of the curve; the FFT and the cut as heat.ts computes them', async () => {
	await withFixtures(await pkgP, (rt) => {
		const s = stateOf(rt);
		if (!s) return assert.ok(rt.pre?.empty, 'no trace only on an empty input');
		const ch = CH(rt), L = liftOf(rt), lifted = col(T(rt, 'lift'), ch);
		assert.equal(L.length, 4);
		assert.ok(L.every((l) => Math.abs(l.a) <= Math.abs(L[ch].a)), 'the followed channel is the one scaled most');
		for (let c = 0; c < 4; c++) col(T(rt, 'lift'), c).forEach((v, j) => assert.ok(Math.abs(v - (L[c].a * s.u0[j] + L[c].b)) < 1e-5, 'a scaled and shifted copy'));
		const sp = spectrum(lifted), h = heights(T(rt, 'f0.fft'), ch);
		h.forEach((v, k) => assert.ok(Math.abs(v - heightOf(sp.re[k], sp.im[k], k)) < 1e-5, `FFT mode ${k}`));
		const kept = heights(T(rt, 'f0.keep'), ch);
		kept.forEach((v, k) => assert.ok(k < s.keep ? Math.abs(v - h[k]) < 1e-6 : v === 0, `kept mode ${k}`));
		const k = mixMode(rt), e = rt.model!.explain(rt.sources.fwd!.trace!, 'f0.mix', [k, ch, 0]);
		assert.equal(e.terms!.length, 8, 'the real part of a mode adds up 8 products');
		if (k > 1) assert.ok(kept.slice(1).every((v) => v <= kept[k]), 'step "mix": mode k is the tallest kept wave');
		if (k === 0) assert.ok(kept.slice(1).every((v) => v <= 1e-6), 'step "mix": mode 0 only when no kept wave has any height');
	});
	const rt = await rtP, lifted = col(T(rt, 'lift'), CH(rt));
	assert.ok(steep(lowPass(lifted, K)) < steep(lifted), 'the square pulse rebuilt from the kept modes has rounder corners');
});

test('steps "predict", "filter": the network matches the solver; the learned filter sits on the solver’s', async () => {
	const rt = await rtP, g = learnedGain(rt.model!);
	g.forEach((v, n) => assert.ok(Math.abs(v - fdGain(n)) < 0.01, `mode ${n}: learned ${v}, solver ${fdGain(n)}`));
	assert.ok(stateOf(rt)!.err < 0.01, 'tryThis "square": the network still matches the solver');
	assert.ok(rt.num('fact:testRel') < 0.01 && rt.num('fact:testN') === 400 && rt.num('fact:trainN') === 4000);
	const smooth = presetY('random');
	assert.ok(errByKeep(rt.model!, smooth)[7] < 0.01, 'quiz 1: 8 modes change a smooth curve’s prediction little');
});

test('step "modes": the task can be done on every preset and is not done at the start', async () => {
	await withFixtures(await pkgP, (rt) => {
		const s = stateOf(rt);
		if (!s || !rt.pre?.tensors) return;
		const errs = errByKeep(rt.model!, s.u0), f = fewest(errs);
		assert.ok(errs.every(Number.isFinite));
		if ((rt.input as { ic?: string } | null)?.ic) assert.ok(f >= 1 && errs[f - 1] < GOAL, 'a preset can reach the target');
	});
	const rt = await rtP, s = stateOf(rt)!;
	assert.ok(s.keep === K && fewest(errByKeep(rt.model!, s.u0)) < K, 'the start input does not already sit on the answer');
	assert.ok(Math.abs(predict(rt.model!, s.u0)[0] - s.pred[0]) < 1e-6, 'predict() equals the trace');
	const wild = Array.from({ length: N }, (_, j) => (j % 8 < 4 ? 1.5 : -1.5));
	assert.equal(fewest(errByKeep(rt.model!, wild)), 0, 'a drawing unlike the training curves can miss the target with every choice: the task then takes all the modes');
});

test('step "limits": heat scales with the curve; training curves stay within AMP; a taller curve is predicted worse', async () => {
	const u = presetY('waves'), a = solve(u.map((v) => TALL * v)), b = solve(u);
	a.forEach((v, j) => assert.ok(Math.abs(v - TALL * b[j]) < 1e-12, '4 times taller in, 4 times taller out'));
	let top = 0;
	for (let i = 0; i < 400; i++) for (const v of initial(FAMILIES[i % 4], createRng(2026).fork(`test${i}`))) top = Math.max(top, Math.abs(v));
	assert.ok(top <= AMP, `no training curve is taller than ${AMP} (${top})`);
	await withFixtures(await pkgP, (rt) => {
		const s = stateOf(rt), t = taller(rt, TALL);
		if (!s || !t) return;
		assert.ok(t.err > 2 * s.err, `the taller curve's error ${t.err} against ${s.err}`);
		if (s.keep === K) assert.ok(t.err > TALL * s.err, `with all ${K} modes the error grows more than the curve: ${t.err} against ${TALL} × ${s.err}`);
	});
	assert.equal(STEPS, 50);
});
