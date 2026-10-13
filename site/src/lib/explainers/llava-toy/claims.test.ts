// llava-toy claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the lesson's two house questions, the adapter's blank / dot / no-words /
// long, the 12 presets) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/llava-toy/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import type { ExplainerRuntime } from '../../atlas/types/index.ts';
import { answerOf, dims } from './lesson.ts';
import { CLASSES, HOUSE, N_IMG, VOCAB, idsOf, textTensor, tokenize, type QRaw } from './adapters/drawing-question/adapter.ts';
import { INK, gridOf } from './views/patch-grid/grid.ts';
import drawing28 from '../../atlas/adapters/drawing28/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('llava-toy', SITE).then((r) => r.pkg);
const rows = (t: Tensor, from: number, to: number) => Array.from(t.data.subarray(from * t.shape[1], to * t.shape[1]));
/** another input through the same graph (rt.model): the answer of its last row */
const ask = (rt: ExplainerRuntime, raw: QRaw) => {
	const image = { shape: [1, 28, 28], data: drawing28.preprocess(typeof raw.b64 === 'string' ? { b64: raw.b64 } : { strokes: (raw.strokes ?? []) as never[] }, rt.spec!, {}).tensors.input.data as Float32Array };
	return answerOf(rt.model!.forward({ image, text: textTensor(idsOf(raw)) }, { record: 'layers' }).t.probs);
};

test('page texts and step "vit": the structural numbers match the spec; TinyViT is read one block before its end', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const D = dims(rt);
	assert.deepEqual([D.n, D.k, D.dv, D.dh, D.dl, D.V, D.V1, D.vb, D.vbAll, D.heads, D.C, D.others], [16, 7, 48, 64, 64, 20, 19, 3, 4, 4, 10, 9]);
	assert.equal(D.vb, D.vbAll - 1, 'the second-to-last block, as LLaVA reads CLIP');
	const text = JSON.stringify(manifest.page);
	for (const s of ['28×28', '16 image tokens', '64 numbers per row', '48 numbers per patch', '10 things']) assert.ok(text.includes(s), s);
	assert.equal(rt.spec!.layers.find((l) => l.id === 'feat')!.outShape[0], N_IMG, 'drop_cls keeps the 16 patch rows');
	assert.equal(rt.num('fact:trainable') + rt.num('fact:vitRead'), rt.spec!.params, 'trained + frozen = every weight of the graph');
	assert.equal(INK, 0.5, 'the view and stats.py (INK = 0.5) use the same "has ink" rule');
	rt.close();
});

test('step "vit": the frozen TinyViT weights are exactly the tinyvit package’s (never changed in training)', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const tv = await createRuntime((await loadPackage('tinyvit', SITE)).pkg, opts);
	const names = Object.keys(rt.model!.weights).filter((n) => n.startsWith('vit.'));
	assert.ok(names.length > 40);
	let n = 0;
	for (const name of names) {
		const a = rt.model!.weights[name].data;
		const b = tv.model!.weights[name.slice(4)].data;
		assert.deepEqual(Array.from(a), Array.from(b), name);
		n += a.length;
	}
	assert.equal(n, rt.num('fact:vitRead'), 'the frozen weights the narration counts');
	rt.close();
	tv.close();
});

test('steps "mask" and "ask": no image token reads the question, so the image tokens are the same whatever is asked', async () => {
	const pkg = await pkgP;
	const traces: Record<string, { seq: number[]; out: number[] }> = {};
	await withFixtures(
		pkg,
		(rt, fx) => {
			const tr = rt.sources.fwd.trace!;
			for (const b of [1, 2]) {
				const a = tr.t[`lm.L${b}.attn`];
				const [H, Tq, Tk] = a.shape;
				for (let h = 0; h < H; h++) for (let q = 0; q < N_IMG; q++) for (let k = N_IMG; k < Tk; k++) assert.equal(a.data[(h * Tq + q) * Tk + k], 0, `${fx.id}: block ${b}`);
			}
			traces[fx.id] = { seq: rows(tr.t['lm.seq'], 0, N_IMG), out: rows(tr.t['lm.L2.add2'], 0, N_IMG) };
		},
		opts
	);
	assert.deepEqual(traces['house-yes'].out, traces['house-car'].out, 'same drawing, two questions: identical image rows after block 2');
	assert.deepEqual(traces['house-yes'].seq, traces['house-car'].seq);
});

test('step "input": TinyViT gets exactly the numbers its own page (drawing28) makes from the same drawing', async () => {
	await withFixtures(
		await pkgP,
		(rt, fx) => {
			const r = rt.input as { b64?: string; strokes?: unknown[] };
			const own = drawing28.preprocess(typeof r.b64 === 'string' ? { b64: r.b64 } : { strokes: (r.strokes ?? []) as never[] }, rt.spec!, {}).tensors.input;
			assert.deepEqual(Array.from(rt.sources.fwd.trace!.t.image.data), Array.from(own.data), fx.id);
		},
		opts
	);
});

test('step "look" (predict) and "where": the answer row reads mostly the image tokens, more where there is ink', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.ok(rt.num('fact:imgShare1') > 0.5 && rt.num('fact:imgShare2') > 0.5);
	assert.ok(rt.num('fact:inkLean1') > 1 && rt.num('fact:topInked1') > 0.9, 'patches with ink get the most');
	assert.ok(rt.num('fact:inkLean1') < 4, 'the empty ones are not ignored');
	rt.close();
	await withFixtures(
		await pkgP,
		(r, fx) => {
			const g = gridOf({ rt: r, source: 'fwd', at: { index: 0 }, props: { block: 1 } })!;
			assert.ok(g.imgShare > 0.5, `${fx.id}: ${g.imgShare}`);
			assert.ok(Math.abs(g.imgShare + g.words.reduce((s, x) => s + x.w, 0) - 1) < 1e-5, 'the weights add up to 1');
		},
		opts
	);
});

test('step "answer": close to TinyViT’s own head, so the projector passes on almost all it needs', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const gap = rt.num('fact:vitAcc') - rt.num('fact:whatAcc');
	assert.ok(gap > 0 && gap < 0.02, String(gap));
	assert.ok(rt.num('fact:agreeVit') > 0.95);
	rt.close();
});

test('task "ask": the house says yes to “is this a house ?” and no to “is it a car ?”; the start drawing is a house', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(answerOf(rt.sources.fwd.trace!.t.probs).label, 'house', 'the start input');
	assert.equal(ask(rt, { strokes: HOUSE, question: 'is this a house ?' }).label, 'yes');
	assert.equal(ask(rt, { strokes: HOUSE, question: 'is it a car ?' }).label, 'no');
	for (const q of ['what is this ?', 'is this a house ?', 'is it an apple ?']) assert.ok(tokenize(q).words.every((w) => w.id > 0), q);
	rt.close();
});

test('step "limits" and try-this "ladder": no word for a ladder or “I do not know”; the ladder gets one of the 10 things', async () => {
	assert.ok(!VOCAB.includes('ladder') && !VOCAB.includes('know'));
	assert.equal(manifest.page.tryThis![0].setup.preset, 'heldout-ladder');
	const rt = await createRuntime(await pkgP, opts);
	const lad = (rt as unknown as { presets: { id: string; raw: unknown }[] }).presets.find((p) => p.id === 'heldout-ladder')!.raw;
	const a = ask(rt, lad as QRaw);
	assert.ok(CLASSES.includes(a.label));
	assert.equal(a.id, rt.num('fact:ladderAnswer'), 'facts.json: the same answer in TS as in PyTorch');
	assert.ok(Math.abs(a.p - rt.num('fact:ladderP')) < 1e-4);
	assert.equal(rt.num('fact:heldOutClassWord'), 1);
	rt.close();
});
