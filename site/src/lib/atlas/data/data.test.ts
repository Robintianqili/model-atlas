// W1-B1: seeded synthetic data (points2d, series). TS is the source of truth, so the generators must be
// deterministic per seed, balanced and in the documented ranges; the TS inputs of the torch twins must reproduce.
// Run from site/:  node --test src/lib/atlas/data/data.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { POINTS2D, pointsTensors, sampleBatch } from './points2d.ts';
import { dampedOscillator, linspace, observations, randomWalk, seasonal, sine, square, trend, windows } from './series.ts';
import { makeRng } from '../../../../tests/atlas/autograd/rng-stub.ts';
import { loadFixture } from '../../../../tests/atlas/autograd/support.ts';
import { makePinnInputs, makeTrainInputs } from '../../../../tests/atlas/autograd/inputs.ts';

test('every 2-D generator is deterministic per seed, balanced and roughly in [−1.5, 1.5]²', () => {
	for (const [name, gen] of Object.entries(POINTS2D)) {
		const a = gen(101, makeRng(3));
		const b = gen(101, makeRng(3));
		const c = gen(101, makeRng(4));
		assert.deepEqual(a.xy, b.xy, `${name}: same seed, same points`);
		assert.notDeepEqual(a.xy, c.xy, `${name}: other seed, other points`);
		assert.equal(a.n, 101);
		assert.ok(a.xy.every((v) => Number.isFinite(v) && Math.abs(v) < 1.5), `${name}: range`);
		if (a.classes === 2) {
			const ones = a.label.reduce((s, v) => s + v, 0);
			assert.equal(ones, 50, `${name}: alternating labels`);
		} else assert.ok(a.label.every((v) => v === 0));
	}
	const x = POINTS2D.xor(200, makeRng(1));
	for (let i = 0; i < x.n; i++) assert.equal(x.xy[2 * i] * x.xy[2 * i + 1] > 0 ? 0 : 1, x.label[i], 'xor labels follow the quadrant');
	const t = pointsTensors(x);
	assert.deepEqual(t.x.shape, [200, 2]);
	const bt = sampleBatch(x, 16, makeRng(2));
	assert.deepEqual(bt.x.shape, [16, 2]);
	for (let i = 0; i < 16; i++) assert.equal(bt.label[i], x.label[bt.index[i]]);
});

test('series: shapes, windows, and the analytic oscillator solves its ODE', () => {
	assert.deepEqual(Array.from(linspace(0, 1, 5)), [0, 0.25, 0.5, 0.75, 1]);
	assert.equal(sine(48, { period: 24 }).y[6], 1);
	assert.equal(trend(10, { slope: 1, intercept: 0 }).y[9], 9);
	assert.deepEqual(Array.from(square(8, { period: 4, amplitude: 1 }).y), [1, 1, -1, -1, 1, 1, -1, -1]);
	assert.deepEqual(seasonal(30, makeRng(1)).y, seasonal(30, makeRng(1)).y);
	assert.equal(randomWalk(20, makeRng(1)).y[0], 0);
	assert.throws(() => sine(5, { noise: 0.1 }), /needs an rng/);
	const w = windows([0, 1, 2, 3, 4, 5, 6], 3, 2);
	assert.equal(w.count, 3);
	assert.deepEqual(Array.from(w.x), [0, 1, 2, 1, 2, 3, 2, 3, 4]);
	assert.deepEqual(Array.from(w.y), [3, 4, 4, 5, 5, 6]);
	assert.equal(windows([1, 2], 3, 1).count, 0);
	// u'' + 2ζω u' + ω² u ≈ 0 by central differences
	const osc = { zeta: 0.15, omega: 12 };
	const h = 1e-4;
	for (const t0 of [0.1, 0.37, 0.8]) {
		const [um, u0, up] = dampedOscillator([t0 - h, t0, t0 + h], osc);
		const d1 = (up - um) / (2 * h);
		const d2 = (up - 2 * u0 + um) / (h * h);
		assert.ok(Math.abs(d2 + 2 * osc.zeta * osc.omega * d1 + osc.omega ** 2 * u0) < 1e-3 * osc.omega ** 2);
	}
	assert.equal(dampedOscillator([0], osc)[0], 1);
	const obs = observations((tt) => dampedOscillator(tt, osc), makeRng(4), { n: 8, frac: 0.4 });
	assert.ok(obs.t.every((v, i) => v >= 0 && v <= 0.4 && (i === 0 || v >= obs.t[i - 1])));
});

test('the local Rng stand-in matches the documented mulberry32 test vectors (types/core.ts)', () => {
	const r42 = makeRng(42);
	assert.deepEqual([r42.next(), r42.next(), r42.next(), r42.next()].map((v) => v.toFixed(10)), ['0.6011037519', '0.4482905590', '0.8524657935', '0.6697340414']);
	const r1 = makeRng(1);
	assert.deepEqual([r1.next(), r1.next(), r1.next()].map((v) => v.toFixed(10)), ['0.6270739406', '0.0027357212', '0.5274470400']);
	const f = makeRng(7);
	f.next();
	assert.equal(f.fork('init').next(), makeRng(7).fork('init').next(), 'fork does not depend on how much was consumed');
});

test('the TS-written twin inputs reproduce from the generators (fixtures are current)', () => {
	assert.deepEqual(JSON.parse(JSON.stringify(makeTrainInputs())), loadFixture('train_inputs.json'));
	assert.deepEqual(JSON.parse(JSON.stringify(makePinnInputs())), loadFixture('pinn_inputs.json'));
});
