// masked-lm claim tests (DEVKIT §8.4 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the lesson's, the adapter's empty / one / long, the five presets) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/masked-lm/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IntTensor, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { layout, MASK, MAXC, PRESETS, TEXT_VOCAB } from './adapters/masked-text/adapter.ts';
import { FLAGS, STORIES, VOCAB } from './source_data.ts';
import { mlmLoss, row, T, text } from './text.ts';
import { batch } from './process.ts';
import { pitfalls } from './concept.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('masked-lm', SITE).then((r) => r.pkg);
const json = (p: string) => JSON.parse(readFileSync(join(SITE, 'static', p), 'utf8'));
const bertSpec = json('models/tinybert/spec.json');
const bertFacts = json('models/tinybert/facts.json').facts as Record<string, { value: number }>;
const gptSpec = json('models/tinygpt/spec.json');
const facts = json('models/masked-lm/facts.json').facts as Record<string, { value: number }>;
const i32 = (a: number[]): IntTensor => ({ shape: [a.length], data: Int32Array.from(a), dtype: 'i32' });
const argmax = (p: ArrayLike<number>) => Array.from(p).reduce((j, v, i, a) => (v > a[j] ? i : j), 0);

test('source_data.ts is an unchanged copy of TinyBERT’s vocabulary, token flags and example sentences', () => {
	assert.deepEqual(VOCAB, bertSpec.vocab);
	assert.equal(FLAGS, bertSpec.tokenFlags);
	const items = json('models/tinybert/presets.json').items as { text: string; ids: number[]; mask: number }[];
	assert.deepEqual(STORIES.map((s) => [s.text, s.ids.join(), s.mask]), items.map((p) => [p.text, p.ids.join(), p.mask]));
});

test('facts.json quotes TinyBERT’s and TinyGPT’s measured numbers unchanged; the context is pipeline/tinybert N_MAX', () => {
	for (const k of ['maskRate', 'maskShare', 'randomShare', 'keepShare', 'valLoss', 'valN', 'valAcc', 'valAccMasked', 'accBoth', 'accLeft', 'accGpt', 'cmpN', 'steps', 'batch']) assert.equal(facts[k].value, bertFacts[k].value, k);
	assert.equal(facts.gptValLoss.value, gptSpec.training.metrics.val_loss);
	assert.equal(facts.context.value, MAXC);
	assert.equal(bertSpec.context ?? 64, MAXC + 3, '64 positions: [CLS], two [SEP] and 61 text tokens');
});

test('the adapter lays out TinyBERT’s input exactly as tinybert’s presets.json records it (sentence split, [CLS], [SEP], [MASK])', () => {
	for (const p of json('models/tinybert/presets.json').items as { ids: number[]; mask: number; bert: number[]; seg: number[] }[]) {
		const L = layout(p.ids, p.ids.map((t, i) => (i === p.mask ? MASK : t)));
		assert.deepEqual(L.ids, p.bert);
		assert.deepEqual(L.seg, p.seg);
	}
	assert.equal(PRESETS.length, STORIES.length);
});

test('steps "hook" and "draw": [MASK] is a special token of its own that real text never contains (TinyGPT’s tokenizer has ids 0 to 1023 only)', () => {
	assert.equal(VOCAB[MASK], '[MASK]');
	assert.equal(bertSpec.special.mask, MASK);
	assert.equal(json('models/tinygpt/tokenizer.json').vocabSize, MASK);
	assert.equal(TEXT_VOCAB, bertSpec.special.eot, 'random tokens are drawn from the text tokens below <|endoftext|>');
});

test('steps "eighty" and "draw", the first pitfall and the quiz: the draw uses TinyBERT’s 15% / 80-10-10, every chosen token is graded, and a full batch settles near the rule', async () => {
	const d = batch.defaults;
	assert.deepEqual([d.rate, d.pMask, d.pRandom, 1 - d.pMask - d.pRandom].map((v) => Math.round(v * 100)), ['maskRate', 'maskShare', 'randomShare', 'keepShare'].map((k) => Math.round(facts[k].value * 100)));
	assert.equal(d.big, facts.batch.value, 'a full training batch: 512 examples');
	assert.equal(d.bigLen, facts.context.value, 'of the longest window, 61 text tokens');
	assert.match(bertSpec.training.task, /15% of text tokens chosen.*80% \[MASK\], 10% random, 10% unchanged/);
	assert.match(pitfalls[0].truth.en, /Only 80% do; 10% become a random token and 10% stay/);
	// TinyBERT's evaluation graded every chosen token: the [MASK] positions are 80% of the graded ones
	assert.ok(Math.abs(facts.valAccMasked.value - bertSpec.training.metrics['[MASK] tokens right']) < 1e-4);
	assert.ok(Math.abs(bertFacts.valMaskedN.value / facts.valN.value - facts.maskShare.value) < 0.01);
	assert.equal(Math.round(1 / facts.maskRate.value), 7, '"about one answer per seven tokens"');
	await withFixtures(await pkgP, async (rt) => {
		if (!text(rt).n) return;
		await rt.ensure('batch', 'last');
		const tape = rt.sources.batch.tape!;
		const m = tape.frame(tape.length - 1).metrics!;
		assert.equal(m.chosen, m.masked + m.random + m.kept);
		assert.ok(Math.abs(m.bigChosen / m.bigN - d.rate) < 0.01, 'chosen share near 15%');
		assert.ok(Math.abs(m.bigMasked / m.bigChosen - d.pMask) < 0.02 && Math.abs(m.bigRandom / m.bigChosen - d.pRandom) < 0.02, 'near 80/10/10');
	}, opts);
});

test('steps "copy" and "loss" and the misconception: at the visible positions TinyBERT already ranks the token it sees first almost everywhere', async () => {
	let seen = 0;
	let first = 0;
	await withFixtures(await pkgP, async (rt) => {
		if (!(rt.input as { text?: string } | null)?.text) return; // real sentences only (edge:long is 70 letters in a row)
		const L = mlmLoss(rt);
		seen += L.seen.length;
		first += L.seenRight;
	}, opts);
	assert.ok(seen > 50 && first / seen > 0.9, `${first} of ${seen}`);
});

test('step "context": a causal mask, so TinyGPT’s guess for a token never changes with a later token; TinyBERT’s does', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const x = text(rt);
		if (x.n < 3 || x.focus < 0 || x.focus >= x.n - 1) return;
		const later = [...x.c];
		later[x.n - 1] = (later[x.n - 1] + 1) % TEXT_VOCAB;
		const g = rt.sources.gpt.model!.forward({ input: i32(later) }).t.probs as Tensor;
		for (let r = 0; r < x.n - 1; r++) assert.deepEqual(Array.from(row(g, r)!), Array.from(row(T(rt, 'probs', 'gpt'), r)!), `row ${r}`);
		const shown = x.shown.map((s, k) => (k === x.n - 1 ? later[k] : s));
		const L = layout(x.c, shown);
		const b = rt.sources.bert.model!.forward({ ids: i32(L.ids), segment: i32(L.seg) }).t.probs as Tensor;
		const a = row(T(rt, 'probs'), x.where[x.focus])!;
		const diff = Array.from(row(b, x.where[x.focus])!).reduce((s, v, i) => Math.max(s, Math.abs(v - a[i])), 0);
		assert.ok(diff > 1e-6, 'a token after the gap changes TinyBERT’s guess');
	}, opts);
});

test('step "limits" and the second pitfall: TinyBERT (both sides) > TinyGPT > TinyBERT (left only); TinyBERT’s loss is the lower one; hiding the last token leaves only [SEP] after it', async () => {
	assert.ok(facts.accBoth.value > facts.accGpt.value && facts.accGpt.value > facts.accLeft.value);
	assert.ok(facts.valLoss.value < facts.gptValLoss.value);
	await withFixtures(await pkgP, async (rt) => {
		const x = text(rt);
		if (!x.n) return;
		const L = layout(x.c, x.c.map((t, k) => (k === x.n - 1 ? MASK : t)));
		assert.deepEqual(L.ids.slice(L.where[x.n - 1] + 1), [bertSpec.special.sep]);
	}, opts);
});

test('step "crowd" and the third pitfall: each gap is filled in one pass without the model’s other guesses (filling one gap with its top guess changes the guess at another), and with two tokens of every three hidden the guesses are right less often than one gap at a time', async () => {
	let together = 0;
	let alone = 0;
	await withFixtures(await pkgP, async (rt) => {
		const x = text(rt);
		if (x.masked.length < 2) return;
		const bert = rt.sources.bert.model!;
		const [a, b] = x.masked;
		const P = T(rt, 'probs')!;
		const fill = x.shown.map((s, k) => (k === b ? argmax(row(P, x.where[b])!) : s));
		const L = layout(x.c, fill);
		const q = bert.forward({ ids: i32(L.ids), segment: i32(L.seg) }).t.probs as Tensor;
		const diff = Array.from(row(q, x.where[a])!).reduce((s, v, i) => Math.max(s, Math.abs(v - row(P, x.where[a])![i])), 0);
		assert.ok(diff > 1e-6, 'its guess at one gap would change had it seen its guess at the other');
		if (x.masked.length * 2 < x.n - 1) return;
		together += mlmLoss(rt).right;
		for (const k of x.masked) {
			const one = layout(x.c, x.c.map((t, i) => (i === k ? MASK : t)));
			const p = bert.forward({ ids: i32(one.ids), segment: i32(one.seg) }).t.probs as Tensor;
			if (argmax(row(p, one.where[k])!) === x.c[k]) alone++;
		}
	}, opts);
	assert.ok(together < alone, `${together} right with two of every three hidden, ${alone} one gap at a time`);
});

test('the hook, step "context" and the compare panel: TinyBERT was trained on TinyGPT’s own training text, so on the same stories with the same tokenizer', () => {
	assert.match(gptSpec.training.dataset, /TinyStoriesV2-GPT4-train/);
	assert.match(bertSpec.training.dataset, /TinyGPT.s train.bin.*TinyStoriesV2-GPT4-train.*TinyGPT.s tokenizer/);
});

test('step "limits": accBoth, accLeft and accGpt were measured with one gap at a time, TinyGPT on the same gaps', () => {
	const m = bertSpec.training.metrics as Record<string, number>;
	assert.ok(Math.abs(m['one gap, both sides'] - facts.accBoth.value) < 1e-4);
	assert.ok(Math.abs(m['one gap, left side only'] - facts.accLeft.value) < 1e-4);
	assert.ok(Math.abs(m['TinyGPT on the same gaps'] - facts.accGpt.value) < 1e-4);
});

test('the manifest is a concept on TinyBERT and TinyGPT, read from their own folders', () => {
	assert.equal(manifest.type, 'concept');
	assert.deepEqual([manifest.sources!.bert, manifest.sources!.gpt].map((s) => (s as { assets?: string }).assets), ['models/tinybert', 'models/tinygpt']);
});
