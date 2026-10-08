// loss-gradient-descent claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept
// page (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the starting dials
// and the adapter's edge fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/loss-gradient-descent/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tensor, TrainState } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { input, noGrad } from '../../atlas/autograd/var.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { bestLine, dialsOf, gradientOf, maxStableRate, scoreOf, XS, YS } from './data.ts';
import { gd, realNet } from './process.ts';
import { MOONS, PARAMS, SIZES } from './net.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'loss-gradient-descent';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

test('steps "points", "score": the data are not on one line, so even the best line scores above 0 (pitfall 2); a line through every point would score 0', () => {
	const best = bestLine();
	assert.ok(best.score > 0.01, `best score ${best.score}`);
	// the definition: a score of 0 needs every error to be 0
	assert.equal(scoreOf(0, 0), YS.reduce((a, y) => a + y * y, 0) / XS.length);
});

test('step "gradient": autograd (backpropagation) gives the same slopes as the formulas mean(2·e·x) and mean(2·e), on every fixture', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const i = await rt.ensure('gd', { phase: 'grad', step: 0 });
			const f = rt.sources.gd.tape!.frame(i);
			const d = dialsOf(rt.input);
			const g = gradientOf(d.w, d.b);
			assert.ok(close(f.metrics!.gw, g.gw) && close(f.metrics!.gb, g.gb), `autograd ${f.metrics!.gw}, ${f.metrics!.gb} vs ${g.gw}, ${g.gb}`);
		},
		opts
	);
});

test('step "predict" (and the toy): one step lowers the score, but not to the lowest possible; the toy\'s preview equals the run\'s update', async () => {
	const best = bestLine().score;
	await withFixtures(
		await pkg(),
		async (rt) => {
			const i = await rt.ensure('gd', { phase: 'update', step: 0 });
			const m = rt.sources.gd.tape!.frame(i).metrics!;
			assert.ok(m.loss1 < m.loss, `the score went ${m.loss} -> ${m.loss1}`);
			assert.ok(m.loss1 > best + 1e-9, `one step reached the bottom: ${m.loss1} vs ${best}`);
			// concept.ts toy: w1, b1 and the new score from the formulas
			const g = gradientOf(m.w, m.b);
			assert.ok(close(m.w - m.lr * g.gw, m.w1) && close(m.b - m.lr * g.gb, m.b1));
			assert.ok(close(scoreOf(m.w1, m.b1), m.loss1));
		},
		opts
	);
});

test('step "descend": each step is shorter than the one before (the slope shrinks near the bottom)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('run', 'last');
			const s = rt.sources.run.tape!.state(rt.sources.run.tape!.length - 1) as { path: Tensor };
			const p = s.path.data;
			let prev = Infinity;
			for (let k = 1; k < s.path.shape[0]; k++) {
				const len = Math.hypot(p[2 * k] - p[2 * k - 2], p[2 * k + 1] - p[2 * k - 1]);
				assert.ok(len < prev || len < 1e-12, `step ${k} is ${len}, the one before ${prev}`);
				prev = len;
			}
		},
		opts
	);
});

test('step "too-big", compare and pitfall 1: above 2/3 the run fails, below it settles; η = 0.7 ends above where it started and η = 0.01 is slower than η = 0.1', async () => {
	// the largest stable rate: 2 / the largest curvature of the bowl (power iteration on the Hessian 2·[[mean x², mean x], [mean x, 1]])
	const n = XS.length;
	const H = [
		[(2 * XS.reduce((a, x) => a + x * x, 0)) / n, (2 * XS.reduce((a, x) => a + x, 0)) / n],
		[(2 * XS.reduce((a, x) => a + x, 0)) / n, 2]
	];
	let v = [1, 0.3];
	let lam = 0;
	for (let k = 0; k < 200; k++) {
		const u = [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]];
		lam = Math.hypot(u[0], u[1]) / Math.hypot(v[0], v[1]);
		v = u;
	}
	assert.ok(close(maxStableRate(), 2 / lam, 1e-9), `maxStableRate ${maxStableRate()} vs ${2 / lam}`);
	assert.ok(close(maxStableRate(), 2 / 3, 1e-12), 'two thirds for these points');
	// any rate above it: the steep direction grows; just below it the run settles (from the flat line, 300 updates)
	for (const lr of [0.68, 0.7, 0.9]) assert.ok(last(record(gd, { w: 0, b: 0 }, { lr, steps: 300 })).metrics!.loss > 1e3, `lr ${lr} did not blow up`);
	assert.ok(last(record(gd, { w: 0, b: 0 }, { lr: 0.65, steps: 300 })).metrics!.loss < bestLine().score + 1e-6);
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('rates', 'last');
			const t = rt.sources.rates.tape!;
			const m = last(t).metrics!;
			assert.ok(m.lossBig > t.frame(0).metrics!.lossBig, `η ${m.lrBig}: ${t.frame(0).metrics!.lossBig} -> ${m.lossBig}`);
			assert.ok(m.lossSlow > m.loss, `η ${m.lrSlow} ${m.lossSlow} vs η ${m.lr} ${m.loss}`);
		},
		opts
	);
});

test('step "limits": the network has 114 weights; two random starts reach similar low scores with different weights, and the landscape between them is not a bowl', async () => {
	assert.equal(PARAMS, 114);
	const runs = [1, 2].map((seed) => record(realNet, null, { steps: 300 }, { seed }));
	const ends = runs.map((t) => t.state(t.length - 1) as TrainState);
	const losses = runs.map((t) => last(t).metrics!.loss);
	for (const l of losses) assert.ok(l < 0.05, `score ${l}`);
	const wa = ends[0].params['net.0.weight'].data[0];
	const wb = ends[1].params['net.0.weight'].data[0];
	assert.ok(Math.abs(wa - wb) > 0.2, `first weights ${wa} and ${wb}`);
	// not one bowl: the score halfway between the two ends is higher than at both ends (a bowl is never higher between two
	// of its points than at both of them)
	const net = new MLP('net', SIZES, 'tanh', createRng(1));
	const named = net.named();
	for (const k of Object.keys(named)) {
		const a = ends[0].params[k].data;
		const b = ends[1].params[k].data;
		named[k].value.data.set(a.map((x, i) => (x + b[i]) / 2));
	}
	const mid = noGrad(() => crossEntropy(net.forward(input({ shape: [MOONS.n, 2], data: MOONS.xy })), MOONS.label).item());
	assert.ok(mid > Math.max(...losses), `halfway ${mid} vs ends ${losses}`);
	// the lesson's own source: the pair's second run is the seed-2 run
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('real', 'last');
			const m = last(rt.sources.real.tape!).metrics!;
			assert.ok(close(m.loss, losses[0], 1e-9) && close(m.loss2, losses[1], 1e-9), `pair ${m.loss}, ${m.loss2} vs ${losses}`);
			assert.ok(close(m.w0, wa, 1e-9) && close(m.w0b, wb, 1e-9));
		},
		{ ...opts, only: ['flat'] }
	);
});

test('concept "real": the network\'s first update is plain gradient descent (old - lr * g = new)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const i = await rt.ensure('realStep', { phase: 'update', step: 0 });
			const s = rt.sources.realStep.tape!.state(i) as TrainState;
			assert.equal(s.optimizer, 'sgd');
			const g = s.grads!['net.0.weight'].data;
			const p = (s.prev as Record<string, Tensor>)['net.0.weight'].data;
			const w = s.params['net.0.weight'].data;
			const lr = Number(s.lr);
			assert.ok(lr > 0);
			for (let k = 0; k < w.length; k++) assert.ok(close(w[k], p[k] - lr * g[k]));
		},
		{ ...opts, only: ['flat'] }
	);
});
