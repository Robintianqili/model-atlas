// tiny-unet-seg claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, over every fixture where it is about the learner's picture (the four presets and the
// adapter's edge cases) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/tiny-unet-seg/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { readReference } from '../../atlas/graph/refs.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { dims } from './lesson.ts';
import { EDGES, rawOf, truthOf } from './adapters/shapes32/adapter.ts';
import { CLASSES, PRESETS, S, TRAIN_R, presetRaw, render } from './scenes.ts';
import { borders, classMap, scoreOf, unsure } from './seg.ts';
import { LIMITS } from './views/mask/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const ROOT = [1, 2, 3, 4].map((k) => join(SITE, ...Array(k).fill('..'))).find((p) => existsSync(join(p, 'pipeline', 'tiny-unet-seg')))!;
const pkgP = loadPackage('tiny-unet-seg', SITE).then((r) => r.pkg);
const step = (id: string) => lesson.steps.find((s) => 'id' in s && s.id === id) as { task?: { done: (rt: never, at: never) => boolean } };

test('page and catalog: parameter count, weight size, classes, set sizes and the lesson length match the spec and facts', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const spec = rt.spec!;
	assert.equal(spec.params, 29644);
	assert.equal(rt.num('fact:params'), spec.params);
	assert.ok(manifest.catalog!.params!.en.includes('29,644'));
	const bytes = readFileSync(join(SITE, 'static/models/tiny-unet-seg/weights.bin')).length;
	assert.equal(bytes, spec.params * 4);
	assert.equal(Math.round(bytes / 1024), 116, '116 KB of weights');
	assert.deepEqual((spec as unknown as { classes: { id: string }[] }).classes.map((c) => c.id), CLASSES.map((c) => c.id), 'background, circle, square or triangle');
	assert.equal(S * S, 1024, 'goal and title: 1,024 answers for a 32×32 picture');
	assert.equal(rt.num('fact:trainN'), 16000, 'Training intro: 16,000 scenes');
	assert.equal(rt.num('fact:testN'), 2000, 'Training intro: 2,000 test scenes');
	assert.equal(lesson.steps.length, 11, 'lessonNote: eleven steps');
	assert.match(readFileSync(join(ROOT, 'pipeline/tiny-unet-seg/make_data.ts'), 'utf8'), /'big', 500, \{ rr: \[9, 13\] \}/, 'limits view: bigger shapes are 9 to 13 px');
});

test('the structure the narration names: 3×3 convolutions with padding, 2×2 pooling, upsampling without weights, two concatenations, a 1×1 head', async () => {
	const spec = (await createRuntime(await pkgP, opts)).spec!;
	const L = (id: string) => spec.layers.find((l) => l.id === id)!;
	for (const id of ['enc1.conv1', 'enc1.conv2', 'enc2.conv1', 'bott.conv2', 'dec1.conv2']) assert.deepEqual([L(id).params?.kernel, L(id).params?.padding], [3, 1], id);
	assert.equal(L('pool1').params?.kernel ?? 2, 2);
	for (const id of ['up2', 'up1']) assert.ok(!L(id).w && !spec.tensors.some((t) => t.name.startsWith(`${id}.`)), `${id} learns nothing`);
	assert.deepEqual(L('cat1').in, ['enc1', 'up1'], 'skip 1: the maps of down 1 next to the upsampled ones');
	assert.deepEqual(L('cat2').in, ['enc2', 'up2']);
	assert.equal(L('dec1.conv1').in[0], 'cat1', 'the next convolution reads both');
	assert.equal(L('logits').params?.kernel, 1, 'a last 1×1 convolution');
	assert.equal(L('probs').params?.axis, 0, 'a softmax for every pixel: over the 4 classes');
});

test('the page renders exactly the bytes PyTorch read (reference.json inputs, written from runs/data/cases.bin)', async () => {
	const refs = await readReference(fsFetch, join(SITE, 'static/models/tiny-unet-seg'));
	const all = [...PRESETS.map((p) => ({ id: p.id, raw: p.raw })), ...EDGES];
	for (const c of refs) {
		const mine = render(all.find((a) => a.id === c.id)!.raw).image;
		assert.deepEqual(Array.from(c.inputs.input.data), Array.from(mine), c.id);
	}
	assert.ok(refs.length >= 4);
});

test('windows: a bottleneck cell sees nearly the whole picture; through the bottleneck an output pixel sees all of it', async () => {
	const d = dims(await createRuntime(await pkgP, opts));
	assert.ok(d.rf3 >= d.side - 2 && d.rf3 < d.side, `rf3 ${d.rf3}`);
	assert.equal(d.w3, d.side, 'w3: the whole picture (also the misconception box)');
	assert.ok(d.rf1 < d.rf2 && d.rf2 < d.rf3, 'each level sees more');
	assert.ok(d.w1 < d.w2 && d.w2 < d.w3);
});

test('blurry (predict) and skip: without skips the shapes are found, but more pixels are unsure and at least as many are wrong (every preset)', async () => {
	let seen = 0;
	await withFixtures(await pkgP, (rt, fx) => {
		if (!PRESETS.some((p) => fx.id === p.id)) return;
		seen++;
		const s = scoreOf(rt)!, n = scoreOf(rt, 'noskip')!;
		for (const c of n.per) if (c.k > 0 && c.truth > 0) assert.ok(c.iou > 0.5, `${fx.id}: shape ${c.k} found`);
		assert.ok(unsure(rt, 'noskip') > unsure(rt), `${fx.id}: more unsure pixels without skips`);
		assert.ok(n.wrong >= s.wrong, `${fx.id}: border pixels flip`);
	}, opts);
	assert.equal(seen, PRESETS.length);
	const rt = await createRuntime(await pkgP, opts);
	assert.ok(rt.num('fact:noskipEdgeAcc') < rt.num('fact:testEdgeAcc'));
});

test('softmax: the least sure pixel lies on a border on every preset (the narration picks the sentence live elsewhere)', async () => {
	let seen = 0;
	await withFixtures(await pkgP, (rt, fx) => {
		if (!PRESETS.some((p) => fx.id === p.id)) return;
		seen++;
		const m = classMap(rt)!;
		let b = 0;
		for (let i = 1; i < S * S; i++) if (m.p[i] < m.p[b]) b = i;
		const e = borders(truthOf(rt.input));
		const near = [b, b - 1, b + 1, b - S, b + S].some((i) => i >= 0 && i < S * S && e[i]);
		assert.ok(near, `${fx.id}: least sure pixel ${b} near a border`);
	}, opts);
	assert.equal(seen, PRESETS.length);
});

test('task: does well on scenes like training, and a giant circle or a painted ring breaks it; presets never count as done', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.ok(rt.num('fact:testMiou') > 0.9, 'does well on scenes like its training set');
	const done = step('task').task!.done;
	for (const id of ['giant', 'ring']) {
		rt.setInput(EDGES.find((e) => e.id === id)!.raw);
		assert.ok(scoreOf(rt)!.worst!.iou < 0.5, `${id}: a shape class under one half`);
		assert.ok(done(rt as never, {} as never), `${id}: the task is done`);
	}
	rt.setInput(EDGES.find((e) => e.id === 'giant')!.raw);
	const giant = scoreOf(rt)!;
	assert.ok(giant.per.some((c) => c.k > 1 && c.pred > 0), 'try-this giant: part of the circle gets another class');
	assert.ok(14 / TRAIN_R[1] > 1.5 && 14 / TRAIN_R[1] < 2, 'almost twice as big as any in training');
	for (const p of PRESETS) {
		rt.setInput(presetRaw(p.id));
		assert.ok(!done(rt as never, {} as never), `${p.id}: a preset is not the learner's own picture`);
		assert.ok(rawOf(rt.input).preset === p.id);
	}
});

test('limits: the three sets unlike training all score far below the test set; training sizes 3 to 8, filled, brighter', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const test = rt.num('fact:testMiou');
	for (const l of LIMITS.slice(1)) assert.ok(rt.num(`fact:${l.fact}`) < test - 0.25, l.fact);
	assert.deepEqual(TRAIN_R, [3, 8]);
	assert.equal(rt.num('fact:oodN'), 500);
});
