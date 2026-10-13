// tinymoe claim tests: every sentence of the lesson and the manifest that is not computed live is asserted here, on
// every fixture where it depends on the input (the adapter's edge fixtures and the presets).
// `atlas check` runs this file (AK408 on failure).   node --test src/lib/explainers/tinymoe/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { dispatch } from './process.ts';
import { cfg, idsOf, row, routeOf, spaceExpert, T } from './routing.ts';
import { drops } from './lesson.ts';
import { topK } from './ops/moe.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const ROOT = process.env.ATLAS_ROOT ?? join(SITE, SITE.includes('.work') ? '../../../..' : '..');
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinymoe', SITE).then((l) => l.pkg);
const each = async (fn: (rt: AtlasRuntime) => void | Promise<void>) => withFixtures(await pkgP, async (rt) => (idsOf(rt).length ? fn(rt) : undefined), opts);

test('steps 1, 2, 5, 10: two blocks of 4 experts (32 → 64 → 32, ReLU), top 2, a router without bias, a last LayerNorm and a linear head over every character', async () => {
	await withFixtures(await pkgP, (rt) => {
		const spec = rt.spec!;
		const c = cfg(rt);
		const moe = spec.layers.filter((l) => l.op === 'moe');
		assert.equal(moe.length, c.L);
		assert.equal(c.L, 2);
		for (const l of moe) assert.equal((l.params as { k: number }).k, c.K);
		assert.deepEqual([c.E, c.K, c.d, c.H], [4, 2, 32, 64]);
		const shape = (n: string) => spec.tensors.find((t) => t.name === n)?.shape;
		assert.deepEqual(shape('B1.moe.fc'), [c.E, c.H, c.d]);
		assert.deepEqual(shape('B1.moe.out'), [c.E, c.d, c.H]);
		assert.deepEqual(shape('B1.router.weight'), [c.E, c.d]);
		assert.equal(shape('B1.router.bias'), undefined, 'the router has no bias');
		const ids = spec.layers.map((l) => l.id);
		assert.ok(ids.indexOf('ln') < ids.indexOf('head') && ids.at(-1) === 'probs');
		assert.equal(shape('head.weight')![0], rt.num('fact:vocab'));
		// one character uses everything except the experts it skips: (E − K) experts per block
		const expert = c.H * c.d + c.H + c.d * c.H + c.d;
		assert.equal(rt.num('fact:active'), rt.num('fact:params') - c.L * (c.E - c.K) * expert);
		assert.equal(spec.tensors.reduce((a, t) => a + t.shape.reduce((x, y) => x * y, 1), 0), rt.num('fact:params'));
	}, { ...opts, only: ['preset:lily'] });
});

test('steps 4 and 5, misconception: only the top 2 experts run; the others give 0; the gates are the two chances rescaled to add up to 1; top 2 by chance = top 2 by score', async () => {
	await each((rt) => {
		const { E, K, d, n } = cfg(rt);
		for (const b of [1, 2])
			for (let t = 0; t < n; t++) {
				const r = routeOf(rt, b, t);
				const on = r.g.flatMap((v, e) => (v > 0 ? [e] : []));
				assert.deepEqual(on, [r.a, r.b].sort((x, y) => x - y));
				assert.ok(Math.abs(r.ga + r.gb - 1) < 1e-6);
				assert.ok(Math.abs(r.ga - r.pa / (r.pa + r.pb)) < 1e-6);
				assert.deepEqual(topK(r.z, K).i, topK(r.p, K).i, 'softmax keeps the order of the scores');
				const ex = T(rt, `B${b}.moe#e`)!;
				for (let e = 0; e < E; e++) if (!on.includes(e)) for (let i = 0; i < d; i++) assert.equal(ex.data[(t * E + e) * d + i], 0);
			}
		for (let t = 0; t < n; t++) assert.ok(Math.abs(row(rt, 'B1.route', t).reduce((a, v) => a + v, 0) - 1) < 1e-5);
	});
});

test('step 6 and its predict: block 2 has an expert that takes every space, period and comma and almost every new line of the held-out stories', async () => {
	await withFixtures(await pkgP, (rt) => {
		const s = spaceExpert(rt, 2);
		const share = (c: string) => s.top.find((x) => x.char === c)?.share ?? 0;
		assert.equal(share(' '), 1);
		assert.ok(share('.') >= 0.999 && share(',') >= 0.999, `. ${share('.')} , ${share(',')}`);
		assert.ok(share('\n') >= 0.99, `new line ${share('\n')}`);
		// "far from random": in each block, most experts take some character every time it occurs
		const routing = (rt.spec as unknown as { routing: { moe: { top: { share: number }[][] }[] } }).routing.moe;
		for (const b of routing) assert.ok(b.top.filter((list) => list[0].share >= 0.99).length >= 3);
	}, { ...opts, only: ['preset:lily'] });
});

test('step 7: the balancing loss is 1 for an even split and grows when one expert gets a large share and a large chance', () => {
	const run = (p: number[][]) => {
		const tape = record(dispatch, { ids: p.map(() => 1), p, block: 1, k: 2 }, { cf: 1 });
		return tape.frame(tape.length - 1).metrics!.aux;
	};
	const even = [[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]];
	const flat = [[0.25, 0.25, 0.25, 0.25]];
	const hog = [[0.6, 0.3, 0.05, 0.05], [0.6, 0.3, 0.05, 0.05], [0.6, 0.05, 0.3, 0.05], [0.6, 0.05, 0.05, 0.3]];
	assert.ok(Math.abs(run(even) - 1) < 1e-12);
	assert.ok(Math.abs(run(flat) - 1) < 1e-12);
	assert.ok(run(hog) > 1.4, String(run(hog)));
});

test('step 8 and its task: the start text does not finish the task, one letter over and over does (half the slots dropped)', async () => {
	await withFixtures(await pkgP, (rt, fx) => {
		if (!idsOf(rt).length) return;
		const d = drops(rt)!;
		const same = new Set(idsOf(rt)).size === 1;
		if (fx.raw === 'Once upon a time, there was a little girl named Lily.') assert.ok(d.dropped / d.slots < 0.4);
		if (same && idsOf(rt).length >= 8) assert.ok(d.dropped / d.slots >= 0.4, `dropped ${d.dropped} of ${d.slots}`);
		assert.equal(d.cap, Math.ceil((2 * idsOf(rt).length) / 4));
	}, opts);
});

test('step 9: the dense twin has about as many parameters as one character uses in the MoE, and was trained the same way', () => {
	const tm = JSON.parse(readFileSync(join(ROOT, 'pipeline/tinymoe/train_metrics.json'), 'utf8'));
	assert.ok(Math.abs(tm.dense.params - tm.moe.active) / tm.moe.active < 0.02);
	for (const k of ['steps', 'batch', 'window', 'lr', 'seed', 'optimizer', 'train_chars']) assert.equal(tm.dense[k], tm.moe[k], k);
	assert.equal(tm.noaux.alpha, 0);
});
