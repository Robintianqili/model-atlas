// self-supervised-objectives claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that
// is not computed live is asserted here, over every fixture (the lesson's, the adapter's empty / one / long, the four
// presets) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/self-supervised-objectives/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IntTensor, StepDef, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { contentOf, layout, PRESETS, splitAt } from './adapters/masked-text/adapter.ts';
import { DRAWING, FLAGS, VOCAB } from './source_data.ts';
import { BART, GRID, JEPA, MAE, T5, bartSpans, cropViews, jepaBlock, maePatches, t5Spans } from './patterns.ts';
import { gap, nextProbs, row, T } from './text.ts';
import { hide } from './process.ts';
import { objectives } from './objectives.ts';
import { pitfalls } from './concept.ts';
import lesson from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('self-supervised-objectives', SITE).then((r) => r.pkg);
const json = (p: string) => JSON.parse(readFileSync(join(SITE, 'static', p), 'utf8'));
const bertSpec = json('models/tinybert/spec.json');
const bertFacts = json('models/tinybert/facts.json').facts as Record<string, { value: number }>;
const gptSpec = json('models/tinygpt/spec.json');
const facts = json('models/self-supervised-objectives/facts.json').facts as Record<string, { value: number }>;
const i32 = (a: number[]): IntTensor => ({ shape: [a.length], data: Int32Array.from(a), dtype: 'i32' });
const runs = (m: boolean[]) => m.filter((x, i) => x && (i === 0 || !m[i - 1])).length;

test('source_data.ts is an unchanged copy of TinyBERT’s vocabulary and token flags, and of a TinyViT preset drawing', () => {
	assert.deepEqual(VOCAB, bertSpec.vocab);
	assert.equal(FLAGS, bertSpec.tokenFlags);
	const house = json('models/tinyvit/presets.json').classes.find((c: { id: string }) => c.id === 'house');
	assert.equal(DRAWING.b64, house.items[0].b64);
	assert.equal(atob(DRAWING.b64).length, GRID.side * GRID.side);
});

test('facts.json quotes TinyBERT’s and TinyGPT’s measured numbers unchanged', () => {
	for (const k of ['accBoth', 'accLeft', 'accGpt', 'cmpN', 'maskRate', 'maskShare', 'randomShare', 'keepShare', 'valAccMasked']) assert.equal(facts[k].value, bertFacts[k].value, k);
	assert.equal(facts.gptValLoss.value, gptSpec.training.metrics.val_loss);
	assert.equal(facts.gptUniformLoss.value, gptSpec.training.metrics.uniform_baseline_loss);
	assert.ok(Math.abs(facts.gptUniformLoss.value - Math.log(gptSpec.vocabSize)) < 1e-3, '"blind guessing": every one of the 1,024 tokens equally likely');
});

test('the copied adapter lays out TinyBERT’s input exactly as tinybert’s presets.json records it (sentence split, [CLS], [SEP], [MASK])', () => {
	for (const p of json('models/tinybert/presets.json').items as { ids: number[]; mask: number; bert: number[]; seg: number[] }[]) {
		const L = layout(p.ids, p.mask);
		assert.deepEqual(L.ids, p.bert);
		assert.deepEqual(L.seg, p.seg);
	}
	for (const p of PRESETS) assert.ok(json('models/tinybert/presets.json').items.some((q: { text: string; ids: number[] }) => q.text === p.raw.text && q.ids.join() === p.raw.ids.join()), p.id);
});

test('steps "next-token" and "every": a causal mask, so TinyGPT’s guess at a position never changes with a later token; one answer per token after the first', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const g = gap(rt);
		if (g.n < 2) return;
		const model = rt.sources.gpt.model!;
		const changed = [...g.c];
		changed[g.n - 1] = (changed[g.n - 1] + 1) % 1023;
		const a = T(rt, 'probs', 'gpt')!;
		const b = model.forward({ input: i32(changed) }).t.probs as Tensor;
		for (let r = 0; r < g.n - 1; r++) assert.deepEqual(Array.from(row(b, r)!), Array.from(row(a, r)!), `row ${r}`);
		const N = nextProbs(rt);
		assert.equal(N.targets, g.n - 1);
		assert.ok(Math.abs(N.loss - N.ps.reduce((s, q) => s - Math.log(q.p), 0) / N.targets) < 1e-12);
	}, opts);
});

test('steps "gap", "click" and "limits": TinyBERT reads both sides of the gap; the measured shares rank TinyBERT (both sides) > TinyGPT > TinyBERT (left only)', async () => {
	assert.ok(facts.accBoth.value > facts.accGpt.value && facts.accGpt.value > facts.accLeft.value);
	await withFixtures(await pkgP, async (rt) => {
		const g = gap(rt);
		// the "limits" look: with the last text token hidden, only [SEP] follows the gap in TinyBERT's input (the left-only setting)
		if (g.n >= 1) {
			const L = layout(g.c, g.n - 1);
			assert.deepEqual(L.ids.slice(L.where[g.n - 1] + 1), [bertSpec.special.sep], 'nothing but [SEP] after the last token');
		}
		if (g.n < 2 || g.k >= g.n - 1) return;
		const model = rt.sources.bert.model!;
		const changed = contentOf(rt.input as never);
		changed[g.n - 1] = (changed[g.n - 1] + 1) % 1023;
		const L = layout(changed, g.k);
		const b = model.forward({ ids: i32(L.ids), segment: i32(L.seg) }).t.probs as Tensor;
		const a = T(rt, 'probs')!;
		const diff = Array.from(row(a, g.m)!).reduce((s, v, i) => Math.max(s, Math.abs(v - row(b, g.m)![i])), 0);
		assert.ok(diff > 1e-6, 'a token after the gap changes TinyBERT’s guess');
	}, opts);
});

test('step "eighty" and the first pitfall: the replayed draw uses TinyBERT’s 15% / 80-10-10; it grades fewer answers than the causal objective on texts of three or more tokens', async () => {
	assert.deepEqual([hide.defaults.rate, hide.defaults.pMask, hide.defaults.pRandom, 1 - hide.defaults.pMask - hide.defaults.pRandom].map((v) => Math.round(v * 100)), ['maskRate', 'maskShare', 'randomShare', 'keepShare'].map((k) => Math.round(facts[k].value * 100)));
	assert.match(bertSpec.training.task, /15% of text tokens chosen.*80% \[MASK\], 10% random, 10% unchanged/);
	assert.match(pitfalls[0].truth.en, /Only 80% do; 10% become a random token and 10% stay/);
	await withFixtures(await pkgP, async (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		await rt.ensure('hide', 'last');
		const tape = rt.sources.hide.tape!;
		const m = tape.frame(tape.length - 1).metrics!;
		assert.ok(m.hidden >= 1);
		if (g.n >= 3) assert.ok(m.hidden < g.n - 1, `${m.hidden} of ${g.n}`);
	}, opts);
});

test('step "nsp": TinyBERT was trained without next-sentence prediction (no pair label, only token guesses)', () => {
	assert.deepEqual(bertSpec.outputs, ['probs']);
	assert.deepEqual(bertSpec.layers.at(-1).outShape, ['T', bertSpec.vocabSize]);
	assert.doesNotMatch(bertSpec.training.task, /next.sentence|NSP/i);
	assert.match(bertSpec.training.task, /masked-language modelling/);
});

test('step "spans": T5’s 15% in spans of mean 3 and BART’s 30% with Poisson(3) lengths, drawn on every fixture as the narration says', async () => {
	assert.deepEqual([T5.rate, T5.meanSpan, BART.rate, BART.lambda], [0.15, 3, 0.3, 3]);
	await withFixtures(await pkgP, async (rt) => {
		const n = gap(rt).n;
		const t = t5Spans(n);
		const b = bartSpans(n);
		if (n < 2) return assert.equal(t.noise, 0);
		assert.equal(t.noise, Math.min(Math.max(Math.round(n * T5.rate), 1), n - 1));
		assert.equal(t.span.filter((s) => s >= 0).length, t.noise);
		assert.equal(t.spans, runs(t.span.map((s) => s >= 0)));
		assert.equal(t.inLen, n - t.noise + t.spans);
		assert.equal(t.outLen, t.noise + t.spans + 1);
		assert.equal(b.hidden, Math.max(Math.round(n * BART.rate), 1));
		assert.equal(b.runs, runs(b.masked));
		assert.equal(b.outLen, n);
		const o = objectives(rt.input as never);
		assert.equal(o.find((x) => x.id === 'span')!.lines[1].chips.length, t.outLen, 'the T5 target row has the narrated length');
		assert.equal(o.find((x) => x.id === 'denoise')!.lines[0].chips.filter((c) => c.text === '[MASK]').length, b.runs, 'one [MASK] per BART run');
		assert.equal(o.find((x) => x.id === 'nsp')!.lines.length, splitAt(gap(rt).c) >= 0 ? 2 : 1, 'a real pair only when the text has two parts');
	}, opts);
});

test('step "patches": TinyViT’s 4 x 4 grid of 7 x 7-pixel patches, and MAE’s 75% (12 of 16 hidden)', () => {
	const patch = json('models/tinyvit/spec.json').layers.find((l: { id: string }) => l.id === 'patch');
	assert.equal(patch.params.kernel, GRID.patch);
	assert.equal(patch.params.stride, GRID.patch);
	assert.deepEqual(patch.outShape.slice(1), [GRID.n, GRID.n]);
	assert.equal(GRID.side, GRID.n * GRID.patch);
	assert.equal(MAE.rate, 0.75);
	const m = maePatches();
	assert.equal(m.hidden.filter(Boolean).length, 12);
	assert.equal(m.k, 12);
	assert.match(pitfalls[1].truth.en, /BERT hides 15% of the tokens, MAE 75% of the image patches/);
	// JEPA: one block of neighbouring patches (I-JEPA's targets are blocks), as the narration and the quiz say
	const j = jepaBlock();
	const cells = j.hidden.flatMap((h, i) => (h ? [[Math.floor(i / GRID.n), i % GRID.n]] : []));
	assert.equal(cells.length, JEPA.block * JEPA.block);
	for (const [r, c] of cells) assert.ok(r >= j.row && r < j.row + JEPA.block && c >= j.col && c < j.col + JEPA.block);
	const patches = (lesson.steps as StepDef[]).find((s) => s.id === 'patches')!;
	assert.match((patches.narration as { en: string }).en, /JEPA hides a block of neighbouring patches/);
	assert.equal(Math.round(facts.maskRate.value * 100), 15);
});

test('step "views": two crops of the same drawing, inside it, the second mirrored', () => {
	const [a, b] = cropViews();
	for (const c of [a, b]) assert.ok(c.x >= 0 && c.y >= 0 && c.x + c.size <= GRID.side && c.y + c.size <= GRID.side);
	assert.equal(a.size, b.size);
	assert.deepEqual([a.flip, b.flip], [false, true]);
});
