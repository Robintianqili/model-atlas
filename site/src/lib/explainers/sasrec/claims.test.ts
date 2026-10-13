// sasrec claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, on every fixture where it depends on the input (the five start histories and the adapter's four
// edge fixtures, through withFixtures). `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/sasrec/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import { TASK_CATS, TASK_CLICKS, taskDone } from './lesson.ts';
import { C, ITEMS, N_ITEMS, NEW_ITEM, PRESETS, SPLIT, session } from './world.ts';
import { K, T, attn, clicksOf, dim, popTop, ranked, reversed, row, runOn } from './seq.ts';
// step "order" names two other pages' models (read only, never changed here): their claims are checked on their code
import { createRng } from '../../atlas/core/rng.ts';
import { MF } from '../matrix-factorization/trainer.ts';
import { N_USERS as MF_USERS } from '../matrix-factorization/world.ts';
import { N_ITEMS as TT_ITEMS, userFeatures } from '../two-tower/world.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('sasrec', SITE).then((r) => r.pkg);
const facts = JSON.parse(readFileSync(join(SITE, 'static/models/sasrec/facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const train = Array.from({ length: SPLIT.train }, (_, s) => session(s));
const ran = (rt: ExplainerRuntime) => clicksOf(rt).length > 0;
const layer = (rt: ExplainerRuntime, id: string) => rt.spec!.layers.find((l) => l.id === id)!;

test('page and catalog: 2,144 learned numbers (31 × 16 item table, 12 position rows, one block), about 25 KB; the training intro; the quiz', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(rt.spec!.params, 2144);
	assert.equal(facts.params.value, 2144);
	assert.deepEqual(rt.model!.weights['item.weight'].shape, [N_ITEMS, 16]);
	assert.deepEqual(rt.model!.weights['pos.weight'].shape, [12, 16]);
	assert.equal(rt.spec!.layers.filter((l) => l.op === 'attention_scores').length, 1, 'one attention block');
	assert.ok(manifest.catalog!.params!.en.startsWith('2,144 ') && manifest.catalog!.params!.en.includes('31 × 16') && manifest.catalog!.params!.en.includes('12 position rows'));
	const kb = ['spec.json', 'weights.bin', 'facts.json'].reduce((s, f) => s + statSync(join(SITE, 'static/models/sasrec', f)).size, 0) / 1024;
	const said = Number(/About (\d+) KB/.exec(manifest.catalog!.size!.en)![1]);
	assert.ok(Math.abs(kb - said) < 1.5, `${kb.toFixed(1)} KB, the catalog says ${said}`);
	assert.deepEqual([facts.trainSessions.value, facts.epochs.value, facts.valSessions.value, facts.testSessions.value], [8000, 60, 1000, 1000]);
	const intro = manifest.page.sections!.intros!.training!.en;
	assert.ok(intro.includes('8,000 synthetic sessions') && intro.includes('60 epochs') && intro.includes('1,000 validation sessions') && intro.includes('1,000 other, test sessions'));
	const n = 4, allowed = (n * (n + 1)) / 2;
	assert.deepEqual([n * n, allowed, n * n - allowed], [16, 10, 6], 'quiz');
});

test('misconception and step "clicks": no user table, only ids and order; 31 products in 6 categories with ids 0 … 30; the task', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.ok(Object.keys(rt.model!.weights).every((k) => !/user/.test(k)));
	assert.deepEqual((rt.spec!.inputs as { name: string }[]).map((x) => x.name), ['ids']);
	assert.equal(N_ITEMS, 31);
	assert.equal(C, 6);
	assert.deepEqual([TASK_CLICKS, TASK_CATS], [5, 2]);
	assert.ok(!taskDone(PRESETS[0].raw), 'the start history does not finish the task by itself');
	assert.ok(taskDone({ clicks: [...PRESETS[0].raw.clicks, 2] }));
	assert.deepEqual((rt.spec!.classes as { en: string }[]).map((x) => x.en), ITEMS.map((x) => x.name.en), 'export.py read the names from world.ts');
});

test('step "embed": every click is its product row plus its position row', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const c = clicksOf(rt), E = rt.model!.weights['item.weight'].data, Pw = rt.model!.weights['pos.weight'].data, x = T(rt, 'x')!.data;
		c.forEach((i, t) => { for (let d = 0; d < 16; d++) assert.ok(Math.abs(x[t * 16 + d] - (E[i * 16 + d] + Pw[t * 16 + d])) < 1e-6); });
	}, opts);
});

test('predict "attend": the most recent click gets the largest weight on every history; earlier clicks of its category get more than clicks of other categories; one click reads only itself', async () => {
	await withFixtures(await pkgP, (rt, fx) => {
		if (!ran(rt)) return;
		const a = attn(rt)!, c = clicksOf(rt), n = c.length, cat = ITEMS[c[n - 1]].cat;
		assert.ok(Math.abs(a.reduce((s, v) => s + v, 0) - 1) < 1e-5);
		assert.ok(a.every((v, j) => j === n - 1 || v < a[n - 1]), `${fx.id}: ${a.map((v) => v.toFixed(3))}`);
		if (n === 1) assert.ok(Math.abs(a[0] - 1) < 1e-6);
		const same = a.slice(0, -1).filter((_, j) => ITEMS[c[j]].cat === cat), other = a.slice(0, -1).filter((_, j) => ITEMS[c[j]].cat !== cat);
		const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
		if (same.length && other.length) assert.ok(mean(same) > mean(other), fx.id);
	}, opts);
	assert.ok(facts.attLastIsMax.value > 0.9);
});

test('steps "score", "mask" and "mix": layer normalization before q, k, v; scores masked above the diagonal (weights exactly 0); a residual add, a two-layer FFN, a last layer normalization; the training count', async () => {
	const rt = await createRuntime(await pkgP, opts);
	for (const id of ['q', 'k', 'v']) assert.deepEqual(layer(rt, id).in, ['ln1']);
	assert.deepEqual(layer(rt, 'add1').in, ['x', 'att.mix']);
	assert.deepEqual(['ffn.fc1', 'ffn.relu', 'ffn.fc2', 'add2', 'ln', 'logits'].map((id) => layer(rt, id).in[0]), ['ln2', 'ffn.fc1', 'ffn.relu', 'add1', 'add2', 'ln']);
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const n = clicksOf(rt).length, w = T(rt, 'att.w')!.data;
		for (let q = 0; q < n; q++) for (let j = q + 1; j < n; j++) assert.equal(w[q * n + j], 0);
	}, opts);
	assert.equal(facts.trainTargets.value, train.reduce((s, x) => s + x.length - 1, 0));
});

test('step "scores": a score is the dot product of the last vector with the product’s row of the same item table (no bias)', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const n = clicksOf(rt).length, h = row(T(rt, 'ln'), n - 1)!, s = row(T(rt, 'logits'), n - 1)!, E = rt.model!.weights['item.weight'].data, d = dim(rt);
		for (let i = 0; i < N_ITEMS; i++) assert.ok(Math.abs(s[i] - h.reduce((a, v, k) => a + v * E[i * d + k], 0)) < 1e-5);
	}, opts);
	const rt = await createRuntime(await pkgP, opts);
	assert.equal((layer(rt, 'logits') as { w?: { weight?: string } }).w?.weight, 'item.weight', 'tied: the same table');
	assert.equal(rt.model!.weights['logits.bias'], undefined);
});

test('step "topk": the top list leaves out the history; the hit rates of the narration are measured, and SASRec beats both baselines', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const r = ranked(rt), c = clicksOf(rt);
		assert.equal(r.length, N_ITEMS - c.length);
		assert.ok(r.slice(0, K).every((x) => !c.includes(x.i)));
	}, opts);
	assert.ok(facts.hit5.value > facts.hit5Markov.value && facts.hit5Markov.value > facts.hit5Pop.value);
});

test('try-this "late": after three baking clicks and running shoes, the top 5 is mostly fitness, led by the sports watch; reversed, baking products lead', async () => {
	const p = PRESETS[4].raw;
	assert.deepEqual(p.clicks.map((i) => ITEMS[i].cat), [5, 5, 5, 3]);
	const rt = await createRuntime(await pkgP, { ...opts, input: p });
	const top = ranked(rt).slice(0, K);
	assert.equal(ITEMS[top[0].i].name.en, 'Sports watch');
	assert.ok(top.filter((x) => ITEMS[x.i].cat === 3).length > K / 2);
	assert.deepEqual(reversed(rt).top.slice(0, 2).map((x) => ITEMS[x.i].cat), [5, 5]);
});

test('step "order": reversing uses the same model (a forward pass equals the trace on the same ids); shuffling the test histories lowers the hit rate', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const again = runOn(rt, clicksOf(rt))!;
		assert.deepEqual(Array.from(again.probs.data), Array.from(T(rt, 'probs')!.data));
	}, opts);
	assert.ok(facts.hit5Shuffled.value < facts.hit5.value);
});

test('step "order": matrix factorization learns one row per user; the two-tower page reads your picks as a bag (its user features do not change with their order)', () => {
	assert.deepEqual(new MF(createRng(1), 0.5).named()['user.weight'].shape, [MF_USERS, 2]);
	for (const picks of [[0, 5, 9], [3, 1, 12, 7], [2, 8, 14, 11, 6, 4]].map((p) => p.map((i) => i % TT_ITEMS))) {
		const a = userFeatures([0, 2], picks), b = userFeatures([0, 2], [...picks].reverse()), c = userFeatures([0, 2], [...picks.slice(1), picks[0]]);
		for (const x of [b, c]) a.forEach((v, k) => assert.ok(Math.abs(v - x[k]) < 1e-6, `feature ${k}`));
	}
});

test('step "limits": no training session clicked the drone (it was only ever a wrong answer); the 5 most clicked products and their larger share of the top-5 slots', async () => {
	const clicks = Array.from({ length: N_ITEMS }, () => 0);
	for (const s of train) for (const i of s) clicks[i]++;
	assert.equal(clicks[NEW_ITEM], 0);
	assert.ok(Array.from({ length: SPLIT.all }, (_, s) => session(s)).every((s) => !s.includes(NEW_ITEM)), 'not in any session at all');
	const rt = await createRuntime(await pkgP, opts);
	assert.deepEqual(rt.spec!.clicks, clicks);
	const order = [...clicks.keys()].sort((a, b) => clicks[b] - clicks[a] || a - b);
	assert.deepEqual(popTop(rt), order.slice(0, 5));
	assert.ok(facts.popShareRec.value > facts.popShareTarget.value);
	assert.ok(facts.newItemInTop5.value >= 0 && facts.newItemPlaceMean.value > K);
});
