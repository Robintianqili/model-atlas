// pointnet claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (presets and edge fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/pointnet/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { T, argmax, crit, cutRaw, maxDiff, variants } from './cloud.ts';
import { cloudOf, normalize, presetPoints, STRAY, withStray, type ShapeName } from './shapes.ts';
import adapter, { type CloudRaw } from './adapters/points2d/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('pointnet', SITE)).pkg;
const ready = async (rt: ExplainerRuntime) => (await rt.ensure('fwd', 'last'), rt);
const fwd = (rt: ExplainerRuntime, xy: Float32Array) => rt.model!.forward({ input: { shape: [xy.length / 2, 2], data: xy } }, { record: 'layers' }).t;
const rowsOf = (x: Tensor, idx: number[]) => Float32Array.from(idx.flatMap((i) => [x.data[2 * i], x.data[2 * i + 1]]));
const SHAPES: ShapeName[] = ['star', 'square', 'triangle', 'circle', 'ring'];
const at = async (raw: unknown) => ready(await createRuntime(await pkg(), { ...opts, input: raw }));

test('order (predict) and quiz: the shuffled table gives exactly the same pooled row and probabilities, on every fixture', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const v = variants(rt);
		if (!T(rt, 'input')) return;
		assert.equal(maxDiff(T(rt, 'pool')!.data, v!.shuf!.pool), 0);
		assert.equal(maxDiff(T(rt, 'softmax')!.data, v!.shuf!.p), 0);
	}, opts);
});

test('critical: the critical rows alone give exactly the same probabilities; each live feature has one winner', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const c = crit(rt);
		if (!c) return;
		assert.equal(maxDiff(T(rt, 'softmax')!.data, variants(rt)!.only!.p), 0);
		assert.equal(c.win.filter((i) => i >= 0).length + c.dead, c.F);
		c.win.forEach((i, f) => assert.ok(i < 0 ? c.max[f] === 0 : c.max[f] > 0));
	}, opts);
});

test('features: a row gives the same features in any table that contains it (each point alone through the shared MLP)', async () => {
	const rt = await at({ points: presetPoints('star') });
	const x = T(rt, 'input')!, h = T(rt, 'relu2')!;
	const F = h.shape[1];
	for (const i of [0, 7, 40]) assert.deepEqual(Array.from(fwd(rt, rowsOf(x, [i])).relu2.data), Array.from(h.data.subarray(i * F, (i + 1) * F)));
});

test('cut: erasing a non-critical row leaves the pooled row as it was; after erasing the critical rows, new ones take over', async () => {
	for (const k of SHAPES) {
		const rt = await at({ points: presetPoints(k) });
		const x = T(rt, 'input')!, c = crit(rt)!;
		const all = Array.from({ length: c.n }, (_, i) => i);
		const non = all.find((i) => !c.pts.includes(i))!;
		assert.equal(maxDiff(fwd(rt, rowsOf(x, all.filter((i) => i !== non))).pool.data, T(rt, 'pool')!.data), 0, k);
		const h2 = fwd(rt, rowsOf(x, all.filter((i) => !c.pts.includes(i)))).relu2;
		assert.ok(crit({ sources: { fwd: { trace: { t: { relu2: h2 } } } } } as unknown as ExplainerRuntime)!.pts.length > 0, k);
	}
});

test('cut (task): one press keeps the answer and a few more change it, on every preset; cut keeps the first answer and the points', async () => {
	for (const k of SHAPES) {
		let raw: CloudRaw = { points: presetPoints(k) };
		const first = argmax(T(await at(raw), 'softmax')!.data);
		let presses = 0, now = first;
		while (now === first && presses < 6) {
			const rt = await at(raw);
			raw = cutRaw(raw, crit(rt)!.pts, argmax(T(rt, 'softmax')!.data));
			presses++;
			now = argmax(T(await at(raw), 'softmax')!.data);
			if (presses === 1) assert.equal(now, first, `${k}: the first press changed the answer`);
		}
		assert.notEqual(now, first, `${k}: still the same answer after ${presses} presses`);
		assert.equal(raw.cut!.n, presses);
		assert.equal(raw.cut!.cls, first);
		assert.deepEqual(raw.cut!.back, presetPoints(k));
		console.log(`  ${k}: the answer changes after ${presses} presses`);
	}
});

test('pre-question: critical points sit farther out than the average point (tips and corners) on the star and the square', async () => {
	for (const k of ['star', 'square'] as const) {
		const rt = await at({ points: presetPoints(k) });
		const x = T(rt, 'input')!, c = crit(rt)!;
		const r = (i: number) => Math.hypot(x.data[2 * i], x.data[2 * i + 1]);
		const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
		assert.ok(mean(c.pts.map(r)) > mean(Array.from({ length: c.n }, (_, i) => r(i))), k);
	}
});

test('stray: the stray point is the right edge of the box, the centre moves toward it, the shape shrinks and the maxima shift', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const x = T(rt, 'input');
		if (!x) return;
		const s = withStray(x.data), n = s.n;
		for (let i = 0; i < n - 1; i++) assert.ok(s.xy[2 * i] < s.xy[2 * n - 2]);
		assert.ok(s.scale < 1 && s.cx > 0);
		assert.ok(maxDiff(T(rt, 'pool')!.data, variants(rt)!.stray!.pool) > 0);
		assert.equal(STRAY, 3);
	}, opts);
});

test('output: the network answers with one of its 5 shapes, shares adding up to 1, on every fixture', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const p = T(rt, 'softmax');
		if (!p) return;
		assert.equal(p.data.length, 5);
		assert.ok(Math.abs(Array.from(p.data).reduce((a, b) => a + b, 0) - 1) < 1e-5);
	}, opts);
});

test('points: the input is centred on the middle of the box, the farthest point at distance 1; presets are recognized', async () => {
	for (const k of SHAPES) {
		const raw = { points: presetPoints(k) };
		const x = adapter.preprocess(raw, null, {}).tensors.input;
		const m = normalize(cloudOf(raw).flatMap((q) => [q.x, q.y]));
		let far = 0, x0 = Infinity, x1 = -Infinity;
		for (let i = 0; i < m.n; i++) (far = Math.max(far, Math.hypot(x.data[2 * i], x.data[2 * i + 1]))), (x0 = Math.min(x0, x.data[2 * i])), (x1 = Math.max(x1, x.data[2 * i]));
		assert.ok(Math.abs(far - 1) < 1e-6 && Math.abs(x0 + x1) < 1e-6, k);
		const rt = await at(raw);
		assert.equal(['circle', 'square', 'triangle', 'star', 'ring'][argmax(T(rt, 'softmax')!.data)], k);
	}
});

test('try this (ring): with the inner circle erased, the ring is read as a circle', async () => {
	const pts = presetPoints('ring').filter((q) => Math.hypot(q.x, q.y) > 0.65);
	const rt = await at({ points: pts });
	assert.equal(argmax(T(rt, 'softmax')!.data), 0);
});
