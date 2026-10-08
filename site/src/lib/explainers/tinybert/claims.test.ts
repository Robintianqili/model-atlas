// tinybert claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the lesson's, the adapter's empty / one / long, the 5 presets) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/tinybert/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { BPETokenizer } from '../../models/bpe.ts';
import { CLS, MASK, SEP, layout, splitAt, type Spec } from './adapters/masked-text/adapter.ts';
import { attnRow, bertGuess, bigFeature, gap, gptGuess, rightHead, row, T, twinOf } from './bert.ts';
import { after, dims, mixOf, softmaxOf } from './lesson.ts';
import { hide } from './process.ts';
import { chipsOf } from './views/chips/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinybert', SITE).then((r) => r.pkg);
const json = (p: string) => JSON.parse(readFileSync(join(SITE, 'static', p), 'utf8'));

test('step "input": the same tokenizer as TinyGPT; [CLS] first, [SEP] last (and one more after sentence A), [MASK] at the gap', async () => {
	assert.equal(readFileSync(join(SITE, 'static/models/tinybert/tokenizer.json'), 'utf8'), readFileSync(join(SITE, 'static/models/tinygpt/tokenizer.json'), 'utf8'));
	const spec = json('models/tinybert/spec.json') as { vocab: string[] };
	const tok = new BPETokenizer(json('models/tinygpt/tokenizer.json'));
	for (let id = 0; id < 1024; id++) assert.equal(spec.vocab[id], tok.tokenToDisplay(id), `display of token ${id}`);
	assert.deepEqual(spec.vocab.slice(1024), ['[MASK]', '[CLS]', '[SEP]']);
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		assert.equal(g.ids[0], CLS);
		assert.equal(g.ids[g.len - 1], SEP);
		assert.equal(g.ids.filter((x) => x === SEP).length, g.split >= 0 ? 2 : 1);
		assert.equal(g.ids[g.m], MASK, 'the model reads [MASK], never the hidden token');
		assert.equal(g.ids.filter((x) => x === MASK).length, 1);
		assert.equal(g.len, g.n + (g.split >= 0 ? 3 : 2));
	}, opts);
});

test('the page splits sentences exactly as training did (data.py layout, written into presets.json by export.py)', async () => {
	const spec = json('models/tinybert/spec.json') as Spec;
	for (const p of json('models/tinybert/presets.json').items as { ids: number[]; mask: number; bert: number[]; seg: number[] }[]) {
		const L = layout(p.ids, p.mask, spec);
		assert.deepEqual(L.ids, p.bert);
		assert.deepEqual(L.seg, p.seg);
	}
	assert.ok(splitAt((json('models/tinybert/presets.json').items[0] as { ids: number[] }).ids, spec) >= 0, 'the start sentence has two segments');
});

test('steps "rows" and "both": the same blocks as TinyGPT (size, heads, MLP, GELU), no segment rows and a causal mask only in TinyGPT', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const D = dims(rt);
	const gpt = rt.sources.gpt.model!.spec;
	assert.deepEqual([D.d, D.heads, D.blocks], [gpt.dModel, gpt.nHead, gpt.nLayer]);
	assert.equal(rt.spec!.layers.find((l) => l.id === 'B1.fc')!.outShape[1], gpt.dMLP);
	assert.equal(rt.spec!.layers.find((l) => l.id === 'B1.gelu')!.params?.approx, 'tanh');
	assert.ok(gpt.layers.some((l) => l.op === 'gelu' || l.op === 'gelu_tanh'));
	assert.deepEqual(gpt.layers.filter((l) => l.op === 'embedding' || l.op === 'position_embedding').map((l) => l.id), ['tok_emb', 'pos_emb'], 'TinyGPT: token and position rows only');
	assert.ok(rt.spec!.layers.some((l) => l.id === 'seg' && l.op === 'embedding'));
	assert.equal(gpt.layers.filter((l) => l.op === 'causal_mask').length, D.blocks);
	assert.ok(!rt.spec!.layers.some((l) => l.op === 'causal_mask' || l.params?.causal), 'TinyBERT has no causal mask');
	assert.equal(rt.spec!.layers.find((l) => l.id === 'logits')!.w?.weight, 'tok.weight', 'the output compares with the rows of the token table');
	assert.equal(Math.abs(rt.spec!.params - (gpt.params as number)) / (gpt.params as number) < 0.01, true, 'about the same size');
	rt.close();
});

test('steps "both" and "softmax": TinyGPT gives exactly 0 weight to every later token; TinyBERT gives weight after the gap', async () => {
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		for (let b = 0; b < 4; b++) {
			const w = T(rt, `L${b}.attn.weights`, 'gpt')!;
			const [H, n] = w.shape;
			for (let h = 0; h < H; h++) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) assert.equal(w.data[(h * n + i) * n + j], 0);
		}
		const a = T(rt, 'B1.attn')!;
		let right = 0;
		for (let j = g.m + 1; j < g.len; j++) right += a.data[g.m * g.len + j];
		assert.ok(right > 0, 'head 0 of block 1 reads after the gap');
	}, opts);
});

test('step "limits": TinyGPT’s guess for the gap uses only the tokens before it', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const g = gap(rt);
		if (g.k < 1) return;
		const p0 = Float64Array.from(gptGuess(rt).p!);
		const c = g.c.slice();
		for (let i = g.k; i < c.length; i++) c[i] = 262; // change the gap and everything after it
		rt.setInput({ ids: c, mask: g.k });
		const p1 = gptGuess(rt).p!;
		assert.ok(Array.from(p0).every((v, i) => Math.abs(v - p1[i]) < 1e-6));
	}, opts);
});

test('steps "train" and "limits": the masking rule and the measured comparison', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(hide.defaults.rate, rt.num('fact:maskRate'));
	assert.equal(hide.defaults.pMask, rt.num('fact:maskShare'));
	assert.equal(hide.defaults.pRandom, rt.num('fact:randomShare'));
	assert.ok(Math.abs(1 - hide.defaults.pMask - hide.defaults.pRandom - rt.num('fact:keepShare')) < 1e-12);
	assert.equal(hide.defaults.vocab, 1023, 'random tokens are text tokens (0 to 1022)');
	assert.ok(rt.num('fact:accBoth') > rt.num('fact:accLeft'), 'both sides beat the left side alone');
	assert.ok(rt.num('fact:accLeft') < rt.num('fact:accGpt'), 'TinyBERT with the left side alone is below TinyGPT');
	assert.ok(rt.num('fact:valAccMasked') < 1 && rt.num('fact:valAccMasked') > 0.5);
	rt.close();
});

// ---------------------------------------------------------------- added by the teaching review
const P = (rt: Parameters<typeof gap>[0], props: Record<string, unknown> = {}) => ({ rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'live', autoplay: false, onend: () => {}, props }) as unknown as Parameters<typeof chipsOf>[0];

test('step "input": [SEP] at the end, and one more after the first sentence only (part A); the rest, any number of sentences, is part B', async () => {
	const seen = { three: false };
	await withFixtures(await pkgP, (rt, fx) => {
		const g = gap(rt);
		if (!g.n || g.split < 0) return;
		const firstSep = g.ids.indexOf(SEP);
		assert.equal(firstSep, g.where[g.split] + 1, 'the first [SEP] follows the first sentence');
		for (let i = 0; i < g.n; i++) assert.equal(g.seg[g.where[i]], i > g.split ? 1 : 0);
		if (fx.id === 'three') seen.three = g.ids.filter((x) => x === SEP).length === 2;
	}, opts);
	assert.ok(seen.three, 'a text of three sentences still has two [SEP]');
});

test('steps "input" and "hide": the dashed chips are exactly the special tokens; [CLS] and [SEP] cannot be clicked to hide', async () => {
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		for (const mode of ['show', 'pick']) {
			const c = chipsOf(P(rt, { mode }));
			c.items.forEach((x, q) => assert.equal(!!x.special, g.ids[q] >= 1024));
			c.toText.forEach((k, q) => assert.equal(k < 0, g.ids[q] === CLS || g.ids[q] === SEP));
		}
	}, opts);
});

test('step "rows": token row + position row + segment row = the input of block 1, at the outlined feature', async () => {
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		const f = bigFeature(rt);
		const v = (id: string) => row(T(rt, id), g.m)![f];
		assert.ok(Math.abs(v('tok') + v('pos') + v('seg') - v('embed')) < 1e-5);
		const e = row(T(rt, 'embed'), g.m)!;
		assert.ok(Array.from(e).every((x) => Math.abs(x) <= Math.abs(e[f])), 'the outlined feature is the largest');
	}, opts);
});

test('step "score": a score is the query times the key, number by number, added up, divided by the square root of the head size', async () => {
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		const D = dims(rt);
		const h = rightHead(rt);
		const q = row(T(rt, 'B1.q'), g.m)!;
		const k = row(T(rt, 'B1.k'), after(rt))!;
		let dot = 0;
		for (let c = h * D.dh; c < (h + 1) * D.dh; c++) dot += q[c] * k[c];
		const s = attnRow(T(rt, 'B1.scores'), h, g.m)![after(rt)];
		assert.ok(Math.abs(dot / Math.sqrt(D.dh) - s) < 1e-4 * Math.max(1, Math.abs(s)));
		assert.ok(after(rt) > g.m, 'the key worked out is after the gap');
	}, opts);
});

test('step "softmax": e to the power (score - largest), divided by the sum over the row, is the traced weight; the three sides add up to 1', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!gap(rt).n) return;
		const s = softmaxOf(rt);
		assert.ok(Math.abs(s.ex / s.sum - s.w) < 1e-5);
		assert.ok(Math.abs(s.left + s.self + s.right - 1) < 1e-5);
	}, opts);
});

test('step "mix": the [MASK] row’s value mix is the weighted sum of the values; the narrated part is the largest one and lies in the head’s features', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const g = gap(rt);
		if (!g.n) return;
		await rt.ensure('fwd', 'last');
		const D = dims(rt);
		const h = rightHead(rt);
		let f = h * D.dh;
		const mix = T(rt, 'B1.mix')!;
		for (let c = h * D.dh; c < (h + 1) * D.dh; c++) if (Math.abs(mix.data[g.m * D.d + c]) > Math.abs(mix.data[g.m * D.d + f])) f = c;
		rt.sel = { ...rt.sel, layer: 'B1.mix', unit: [g.m, f] };
		const x = mixOf(rt);
		assert.ok(x.f >= x.lo && x.f <= x.hi);
		const a = attnRow(T(rt, 'B1.attn'), h, g.m)!;
		const v = T(rt, 'B1.v')!;
		let y = 0;
		let big = 0;
		for (let j = 0; j < a.length; j++) {
			y += a[j] * v.data[j * D.d + f];
			big = Math.max(big, Math.abs(a[j] * v.data[j * D.d + f]));
		}
		assert.ok(Math.abs(y - x.y) < 1e-5);
		assert.ok(Math.abs(Math.abs(x.av) - big) < 1e-9);
	}, opts);
});

test('step "guess" and try-this "first": a word with a space in front is another token; with the first word hidden, TinyBERT’s top guesses all carry a space', async () => {
	const spec = json('models/tinybert/spec.json') as { vocab: string[] };
	const tom = spec.vocab.indexOf('Tom');
	assert.ok(tom >= 0 && spec.vocab.indexOf(' Tom') >= 0 && spec.vocab.indexOf(' Tom') !== tom);
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		const b = bertGuess(rt);
		assert.ok(b.order.slice(0, 5).every((i) => g.word(i).startsWith(' ')));
		const tw = twinOf(rt, g.hidden);
		assert.equal(g.word(tw), ' Tom');
		assert.ok(b.order.indexOf(tw) < b.order.indexOf(g.hidden), '“␣Tom” ranks above “Tom”');
	}, { ...opts, only: ['first'] });
});

test('step "train" (chips caption): exactly the bytes 0x80-0xFF show as <0x..>, pieces of characters that take several bytes', () => {
	const spec = json('models/tinybert/spec.json') as { vocab: string[] };
	for (let i = 0; i < 256; i++) assert.equal(/^<0x[0-9A-F]{2}>$/.test(spec.vocab[i]), i >= 0x80, `byte ${i}`);
});

test('try-this "colour" and the predict of step "guess": hiding “red” in the box sentence, another word comes first and “red” is still among the top 5', async () => {
	await withFixtures(await pkgP, (rt) => {
		const g = gap(rt);
		assert.equal(g.word(g.hidden), ' red');
		const b = bertGuess(rt);
		assert.notEqual(b.top, g.hidden);
		assert.ok(b.rank >= 2 && b.rank <= 5);
	}, { ...opts, only: ['colour'] });
});
