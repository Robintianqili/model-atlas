// points2d adapter (W1-B2): deterministic raw → points, painting and erasing, edge fixtures, the live-model `input`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import adapter, { classCounts, pointsOf, POINT_PRESETS, presetRaw, preprocessPoints } from './adapter.ts';
import { moons } from '../../data/points2d.ts';
import { createRng } from '../../core/rng.ts';

test('a preset raw value gives the seeded generator points, the same everywhere', () => {
	const raw = presetRaw('moons', { n: 120, seed: 3 });
	const a = pointsOf(raw);
	const b = pointsOf(structuredClone(raw)); // a worker receives a structured clone
	const ref = moons(120, createRng(3));
	assert.deepEqual(Array.from(a.xy), Array.from(ref.xy));
	assert.deepEqual(Array.from(a.label), Array.from(ref.label));
	assert.deepEqual(Array.from(b.xy), Array.from(a.xy));
	assert.deepEqual(classCounts(a), [60, 60]);
});

test('painted points are added after the preset, erased preset points are removed, junk is ignored', () => {
	const raw = { ...presetRaw('xor', { n: 10 }), removed: [0, 3], points: [{ x: 0.5, y: -0.5, c: 1 }, { x: NaN, y: 0 }, { x: 0.1, y: 0.2 }] };
	const p = pointsOf(raw);
	const base = pointsOf(presetRaw('xor', { n: 10 }));
	assert.equal(p.n, 8 + 2);
	assert.equal(p.xy[0], base.xy[2]); // preset point 1 is first now
	assert.deepEqual([p.xy[16], p.xy[17], p.label[8]], [0.5, -0.5, 1]);
	assert.equal(p.label[9], 0);
	for (const pr of POINT_PRESETS) assert.equal(pointsOf(presetRaw(pr.id)).n, pr.n, pr.id);
});

test('preprocess: xy [n, 2], label [n], input [2] for one point; empty and one-class notes', () => {
	const pre = preprocessPoints(presetRaw('circles', { n: 50 }), null, { point: 4 });
	assert.deepEqual(pre.tensors.xy.shape, [50, 2]);
	assert.deepEqual(pre.tensors.label.shape, [50]);
	assert.deepEqual(Array.from(pre.tensors.input.data), [pre.tensors.xy.data[8], pre.tensors.xy.data[9]]);
	assert.equal(pre.empty, false);
	// a classifier package declares input.options { classes: 2 }; the one-class note is for classifiers
	const fx = Object.fromEntries(adapter.edgeFixtures().map((f) => [f.id, adapter.preprocess(f.raw, null, { classes: 2 })]));
	assert.equal(adapter.preprocess(adapter.edgeFixtures()[2].raw, null, {}).notes?.length, 0, 'no class note for an unlabelled set');
	assert.equal(fx.empty.empty, true);
	assert.ok(!('input' in fx.empty.tensors));
	assert.ok(fx.empty.notes?.length);
	assert.equal(fx['one-point'].tensors.xy.shape[0], 1);
	assert.ok(fx['one-class'].notes?.some((n) => /same class/.test(n.en)), 'one-class note');
	assert.equal(adapter.layer0.view, 'plane2d');
	assert.ok(adapter.presets!({}).length >= 5);
});
