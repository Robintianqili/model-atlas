// crnn-ocr claim tests: every sentence of the lesson and the manifest that is not computed live is asserted here, on
// every fixture where it depends on the input (lesson fixtures, the adapter's edge fixtures and the presets).
// `atlas check` runs this file (AK408 on failure).   node --test src/lib/explainers/crnn-ocr/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { bestPath, collapse, forward, idsOf, pathLog, probsOf, textOf } from './ctc.ts';
import { lossOf } from './process.ts';
import { H, SYMBOLS, W } from './glyphs.ts';
import { stripOf } from './views/ctc/data.ts';
import { PRESETS } from './adapters/word-strokes/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('crnn-ocr', SITE).then((l) => l.pkg);
const tr = (rt: AtlasRuntime, id: string): Tensor => rt.sources.fwd.trace!.t[id];
const each = async (fn: (rt: AtlasRuntime) => void | Promise<void>) => withFixtures(await pkgP, async (rt) => (rt.pre?.empty ? undefined : fn(rt)), opts);
const strip = (rt: AtlasRuntime) => stripOf({ rt, at: { index: 0 }, props: {} })!;

test('manifest, steps 1, 3, 5, 6: the sizes in the text (a strip much wider than tall, 32 columns, 11 symbols, two LSTMs)', async () => {
	await withFixtures(await pkgP, (rt) => {
		assert.ok(W > 2 * H, 'step 1: the strip is much wider than tall');
		assert.deepEqual(tr(rt, 'probs').shape, [32, 11], 'prequestion and goals: 11 symbols in each of 32 columns');
		assert.deepEqual(SYMBOLS.slice(1), '0123456789'.split(''), 'step 11: the digits and the blank, nothing else (no letter)');
		assert.equal(rt.spec!.layers.filter((l) => l.op === 'lstm').length, 2, 'step 5: a second LSTM');
		assert.equal(tr(rt, 'cols').shape[1], tr(rt, 'relu2').shape[0] * tr(rt, 'relu2').shape[1], 'step 3: k2 maps × ph rows');
	}, { ...opts, only: ['preset:2048'] });
});

test('step 3: one column of the maps saw 8 pixels of the strip across (the receptive field the narration computes)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const box = rt.model!.receptiveField('relu2', { y0: 0, x0: 10, y1: 7, x1: 10 });
		assert.ok(box);
		assert.equal(box.x1 - box.x0 + 1, 8);
	}, { ...opts, only: ['preset:2048'] });
});

test('steps 4 and 5: the left-to-right state at a column ignores every column to its right; "both" joins the two LSTMs', async () => {
	await each((rt) => {
		const x = tr(rt, 'input');
		const ink = Float32Array.from(x.data);
		const cut = 40; // change every pixel right of x = 40: columns 0..15 of the left-to-right LSTM see none of it
		for (let r = 0; r < H; r++) for (let c = cut; c < W; c++) ink[r * W + c] = 1 - ink[r * W + c];
		const b = rt.model!.forward({ input: { shape: x.shape, data: ink } }).t;
		const [f0, f1] = [tr(rt, 'fwd'), b.fwd];
		const D = f0.shape[1];
		for (let t = 0; t < 16; t++) for (let j = 0; j < D; j++) assert.equal(f1.data[t * D + j], f0.data[t * D + j], `fwd column ${t}`);
		const both = tr(rt, 'both');
		const back = tr(rt, 'bwd_back');
		for (let t = 0; t < both.shape[0]; t++)
			for (let j = 0; j < D; j++) {
				assert.equal(both.data[t * 2 * D + j], f0.data[t * D + j]);
				assert.equal(both.data[t * 2 * D + D + j], back.data[t * D + j]);
			}
	});
});

test('step 6 (predict) and step 7: collapse merges repeats, then drops blanks; 1 1 reads 1 and 1 – 1 reads 11', () => {
	assert.equal(textOf(collapse([2, 2])), '1');
	assert.equal(textOf(collapse([2, 0, 2])), '11');
	assert.equal(textOf(collapse([0, 3, 3, 0, 0, 1, 0, 5, 9, 9, 0])), '2048');
});

test('step 6 (predict): on every preset the first and last columns are blanks, and the doubled 1s and 0s of 1100 are kept apart by blanks', async () => {
	await withFixtures(await pkgP, (rt) => {
		const p = bestPath(probsOf(tr(rt, 'probs'))!);
		assert.equal(p[0], 0);
		assert.equal(p[p.length - 1], 0);
	}, { ...opts, only: PRESETS.map((q) => `preset:${q.id}`) });
	await withFixtures(await pkgP, (rt) => {
		const p = bestPath(probsOf(tr(rt, 'probs'))!);
		assert.equal(textOf(collapse(p)), '1100');
		const s = p.map((k) => SYMBOLS[k]).join('').replace(/(.)\1+/g, '$1');
		assert.match(s, /1–1/);
		assert.match(s, /0–0/);
	}, { ...opts, only: ['preset:1100'] });
});

test('steps 7-9: the decode process reads the greedy path; many paths spell the text; the loss is −log of their total chance', async () => {
	await each(async (rt) => {
		await rt.ensure('decode', 'last');
		const tape = rt.sources.decode.tape!;
		const f = tape.frame(tape.length - 1);
		const q = probsOf(tr(rt, 'probs'))!;
		const best = bestPath(q);
		assert.equal((tape.state(tape.length - 1) as { text: string }).text, textOf(collapse(best)));
		const mt = f.metrics as Record<string, number>;
		assert.ok(Math.abs(mt.logPath - pathLog(q, best)) < 1e-12);
		assert.ok(mt.logText >= mt.logPath, 'the text is at least as likely as one of its paths');
		if (textOf(collapse(best))) assert.ok(mt.paths > 1, 'many paths spell the same text');
		const s = strip(rt);
		const want = s.wrote || s.text;
		const fw = forward(q, idsOf(want));
		if (fw.paths > 0) assert.ok(Math.abs(lossOf(mt).loss + fw.logP) < 1e-9, 'step 9: the formula is −log of the total chance of the meant text');
	});
});

test('step 10: the 1100 preset with an extra stroke is a misread (the task can be done); the presets are read right', async () => {
	await withFixtures(await pkgP, (rt) => {
		const s = strip(rt);
		assert.notEqual(s.text, s.wrote);
	}, { ...opts, only: ['messy-1100'] });
	await withFixtures(await pkgP, (rt) => {
		const s = strip(rt);
		const raw = rt.input as { text: string };
		if (/^[0-9]+$/.test(raw.text)) assert.equal(s.text, raw.text);
	}, { ...opts, only: PRESETS.map((q) => `preset:${q.id}`) });
});

test('step 11: written HEX comes out as digits only (letters were never trained on; every column picks a digit or the blank)', async () => {
	await withFixtures(await pkgP, (rt) => assert.match(strip(rt).text, /^[0-9]+$/), { ...opts, only: ['preset:hex'] });
});

test('steps 9 and 11: messiness costs it (messier words, glyphs.ts wild 2.2 = the top of the training range 1 to 2.2, are read less often than the neatest, wild 1)', async () => {
	const { readFile } = await import('node:fs/promises');
	const f = JSON.parse(await readFile(join(SITE, 'static/models/crnn-ocr/facts.json'), 'utf8')).facts;
	assert.ok(f.wildAcc.value < f.testAcc.value - 0.05, 'clearly fewer messier words read right');
	assert.match(f.wildAcc.unit, /wild 2\.2/);
	assert.equal(f.testWords.value, 7000);
});
