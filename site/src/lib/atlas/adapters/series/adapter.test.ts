// series adapter (W1-B2): presets, drawn values, observation points, edge fixtures.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import adapter, { OSCILLATOR, preprocessSeries, SERIES_PRESETS, seriesOf, seriesPresetRaw } from './adapter.ts';
import { dampedOscillator, seasonal } from '../../data/series.ts';
import { createRng } from '../../core/rng.ts';

test('presets are the atlas/data generators, seeded', () => {
	const s = seriesOf(seriesPresetRaw('seasonal', { n: 60, seed: 4 }));
	const ref = seasonal(60, createRng(4));
	assert.deepEqual(Array.from(s.y), Array.from(ref.y));
	for (const p of SERIES_PRESETS) assert.equal(seriesOf(seriesPresetRaw(p.id)).y.length, p.n, p.id);
	const osc = seriesOf(seriesPresetRaw('oscillator'));
	assert.equal(osc.points.length, 8);
	assert.ok(osc.points.every((p, i) => i === 0 || p.t >= osc.points[i - 1].t), 'sorted by time');
	assert.ok(Math.abs(osc.y[0] - dampedOscillator([0], OSCILLATOR)[0]) < 1e-15);
});

test('drawn values replace the preset values and keep its time axis; points override', () => {
	const base = seriesOf(seriesPresetRaw('sine', { n: 5 }));
	const s = seriesOf({ ...seriesPresetRaw('sine', { n: 5 }), y: [1, 2, 3, 4, 5], points: [{ t: 3, y: 0 }, { t: 1, y: 1 }] });
	assert.deepEqual(Array.from(s.y), [1, 2, 3, 4, 5]);
	assert.deepEqual(Array.from(s.t), Array.from(base.t));
	assert.deepEqual(s.points, [{ t: 1, y: 1 }, { t: 3, y: 0 }]);
	const drawn = seriesOf({ y: [0.5, Infinity, -1] });
	assert.deepEqual(Array.from(drawn.y), [0.5, 0, -1]);
	assert.deepEqual(Array.from(drawn.t), [0, 1, 2]);
});

test('preprocess: t, y, obs and input; edge fixtures (flat, empty, one value)', () => {
	const pre = preprocessSeries(seriesPresetRaw('oscillator', { n: 50 }), null, {});
	assert.deepEqual(pre.tensors.t.shape, [50]);
	assert.deepEqual(pre.tensors.obs.shape, [8, 2]);
	assert.deepEqual(pre.tensors.input.shape, [1]);
	const fx = Object.fromEntries(adapter.edgeFixtures().map((f) => [f.id, adapter.preprocess(f.raw, null, {})]));
	assert.ok(fx.flat.notes?.some((n) => /flat/.test(n.en)));
	assert.equal(fx.empty.empty, true);
	assert.equal(fx['one-value'].tensors.y.shape[0], 1);
	assert.equal(adapter.layer0.view, 'curve1d');
});
