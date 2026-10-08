// R0 teaching review (design §8.2, integration 1b step 3): the op templates as a learner meets them inside auto
// lessons. planLesson() composes each op step from what + how (or the short `repeat` beat) on the shipped TinyCNN,
// TinyGPT and hello-xor specs; every step is rendered here in en and zh with the op's vars(), exactly as the lesson
// layer renders it, and checked for:
//   - no unresolved placeholder ("…"), no "[object Object]", real Chinese in zh;
//   - 2-4 sentences and at most 90 English words (AK407's narration shape);
//   - the structure-dependent phrases: one input channel vs several (conv2d), a bias or none (linear, conv2d),
//     a residual add vs token + position (add), what softmax shares mean (attention, vocabulary, classes),
//     GELU's tanh shortcut, the attention scale as 1/√d, consistent shape brackets.
// Run from site/: node --test src/lib/atlas/ops/narration.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORE_OPS, coreRegistry } from './index.ts';
import { biasWords, dependsOn, words } from './common.ts';
import { normalizeSpec } from '../graph/legacy.ts';
import { planLesson } from '../lesson/plan.ts';
import { render, sentenceCount, wordCount, MISSING, type Lang } from '../lesson/narrate.ts';
import type { AtlasLayer, AtlasSpec, ExplainerManifest, ExplainerRuntime, I18n, StepDef } from '../types/index.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const loadSpec = (slug: string): AtlasSpec => normalizeSpec(JSON.parse(readFileSync(join(SITE, 'static', 'models', slug, 'spec.json'), 'utf8')));
const MANIFEST = (slug: string): ExplainerManifest => ({
	slug,
	type: 'model',
	kinds: ['trace'],
	quality: 'draft',
	devOnly: true,
	page: { goals: [], prequestion: { id: 'q', question: { en: 'q', zh: 'q' }, options: [{ en: 'a', zh: 'a' }], answer: 0, explain: { en: 'e', zh: 'e' } } },
	credits: []
});
/** The op steps only need their vars (structure); nothing else of the runtime is read. */
const RT = { slug: 'r0', spec: null, model: null, input: null, pre: null, sources: {}, sel: {}, fact: () => undefined, num: () => NaN, ensure: async () => 0, setInput: () => {} } as unknown as ExplainerRuntime;

function say(step: StepDef, lang: Lang): string {
	const r = render(step.narration as never, RT, { index: 0 }, lang);
	assert.deepEqual(r.missing, [], `${step.id} ${lang}: unresolved {${r.missing.join('}, {')}}`);
	assert.ok(!r.error, `${step.id} ${lang}: vars() threw ${r.error}`);
	assert.ok(!r.text.includes(MISSING) && !r.text.includes('[object Object]') && !r.text.includes('undefined'), `${step.id} ${lang}: ${r.text}`);
	return r.text;
}

const SPECS = ['tinycnn', 'tinygpt', 'hello-xor'] as const;
const plans = Object.fromEntries(SPECS.map((s) => [s, planLesson(loadSpec(s), CORE_OPS, MANIFEST(s)).filter((st) => st.id.startsWith('op:'))])) as Record<
	(typeof SPECS)[number],
	StepDef[]
>;
const text = (slug: (typeof SPECS)[number], id: string, lang: Lang = 'en') => {
	const st = plans[slug].find((s) => s.id === id);
	assert.ok(st, `${slug}: no auto step ${id} (have ${plans[slug].map((s) => s.id).join(', ')})`);
	return say(st, lang);
};

for (const slug of SPECS) {
	test(`R0: every auto op step of ${slug} renders in en and zh with 2-4 sentences and at most 90 words`, () => {
		assert.ok(plans[slug].length >= 2, `${slug}: ${plans[slug].length} op steps`);
		for (const st of plans[slug]) {
			const en = say(st, 'en');
			const zh = say(st, 'zh');
			const n = sentenceCount(en, 'en');
			assert.ok(n >= 2 && n <= 4, `${slug} ${st.id}: ${n} sentences: ${en}`);
			assert.ok(wordCount(en) <= 90, `${slug} ${st.id}: ${wordCount(en)} words`);
			const nz = sentenceCount(zh, 'zh');
			assert.ok(nz >= 2 && nz <= 4, `${slug} ${st.id} zh: ${nz} sentences: ${zh}`);
			assert.ok(/[一-鿿]/.test(zh), `${slug} ${st.id}: zh is Chinese`);
			if (st.look) for (const lang of ['en', 'zh'] as const) assert.ok(!/^(Look|看)/.test(render(st.look as never, RT, { index: 0 }, lang).text), `${st.id}: the look hint repeats the UI's "Look at:" label`);
		}
	});
}

test('R0 conv2d: one input channel is a 3×3 grid of 9 numbers ("all 1 input channels" is gone); the repeat beat counts 8 channels', () => {
	const first = text('tinycnn', 'op:conv1_1');
	assert.match(first, /^A kernel is a 3×3 grid of 9 learned numbers\. It slides over the input one cell at a time;/);
	assert.match(first, /the 9 products are summed, plus one learned bias\./);
	assert.match(first, /This layer has 8 kernels, one per output channel, so the output has shape \[8×26×26\]\.$/);
	assert.doesNotMatch(first, /all 1 input channels/);
	assert.match(text('tinycnn', 'op:conv1_1', 'zh'), /^卷积核是一个 3×3 的网格，共 9 个学来的数。它在输入上滑动，每次移动一格；/);
	const rep = text('tinycnn', 'op:conv1_2');
	assert.match(rep, /reads 8 channels, not one, so each kernel is 8 stacked 3×3 grids\. Each output cell sums 72 products/);
	// a first conv that already reads several channels (an RGB input), with stride 2, padding 1 and no bias
	const spec = loadSpec('tinycnn');
	const layer: AtlasLayer = { ...spec.layers[1], id: 'rgb', params: { kernel: 3, stride: 2, padding: 1, inChannels: 3, outChannels: 4 } };
	const s2: AtlasSpec = { ...spec, tensors: [{ name: 'rgb.weight', shape: [4, 3, 3, 3], offset: 0, length: 108 }], layers: [layer] };
	const v = coreRegistry.get('conv2d').def.vars!(layer, s2) as Record<string, unknown>;
	assert.deepEqual(v.kernelIs, { en: '27 learned numbers: a 3×3 grid for each of the 3 input channels', zh: '是 27 个学来的数：3 个输入通道各有一个 3×3 的网格' });
	assert.equal((v.slideHow as unknown as I18n).en, '2 cells at a time, with one ring of zeros added around the input first');
	assert.equal((v.biasClause as unknown as I18n).en, '; this layer has no bias');
});

test('R0 linear: a bias is stated when the layer has one and denied when it has none (TinyGPT tied logits); q/k/v name their heads', () => {
	assert.match(text('tinycnn', 'op:dense'), /^Each of the 10 outputs is a weighted sum of all 256 inputs: .*added up, plus one learned bias\./);
	const q = text('tinygpt', 'op:L0.attn.q');
	assert.match(q, /plus one learned bias\./);
	assert.match(q, /Attention reads these 76 outputs as 4 heads of 19 numbers each\.$/);
	assert.match(text('hello-xor', 'op:fc1'), /^Each of the 2 outputs is a weighted sum of all 2 inputs/);
	const spec = loadSpec('tinygpt');
	const logits = spec.layers.find((l) => l.id === 'logits')!;
	const v = coreRegistry.get(logits.op).def.vars!(logits, spec) as Record<string, unknown>;
	assert.deepEqual(v.biasClause, biasWords(false).biasClause);
	assert.equal((v.biasIs as unknown as I18n).en, 'there is no bias, so b =');
	assert.equal((v.headsNote as unknown as I18n).en, '');
});

test('R0 add: token + position is not called a residual connection; the block adds are', () => {
	const embed = text('tinygpt', 'op:embed');
	assert.match(embed, /with a taken from “Token embedding” and b from “Position embedding”\. Neither input is computed from the other/);
	assert.doesNotMatch(embed, /residual/i);
	assert.match(text('tinygpt', 'op:embed', 'zh'), /a 来自“词元嵌入”，b 来自“位置嵌入”。两个输入互不依赖/);
	const spec = loadSpec('tinygpt');
	assert.equal(dependsOn(spec, 'L0.attn.proj', 'embed'), true);
	assert.equal(dependsOn(spec, 'pos_emb', 'tok_emb'), false);
	for (const id of ['L0.resid1', 'L0.resid2', 'L3.resid2']) {
		const l = spec.layers.find((x) => x.id === id)!;
		const v = coreRegistry.get('add').def.vars!(l, spec) as Record<string, unknown>;
		assert.match((v.addRole as unknown as I18n).en, /^This is a residual connection: b was computed from a/, id);
	}
});

test('R0 softmax: the shares are named for their place (attention weights, next-token or class probabilities)', () => {
	assert.match(text('tinygpt', 'op:L0.attn.weights'), /its shares are the attention weights, how much that query reads from each key\.$/);
	const role = (slug: 'tinycnn' | 'tinygpt', id: string) => {
		const spec = loadSpec(slug);
		const l = spec.layers.find((x) => x.id === id)!;
		return ((coreRegistry.get('softmax').def.vars!(l, spec) as Record<string, unknown>).softmaxRole as unknown as I18n).en;
	};
	assert.equal(role('tinygpt', 'probs'), ' Here the shares are the model’s probabilities for each of its 1,024 tokens to come next.');
	assert.equal(role('tinycnn', 'softmax'), ' Here the shares are the model’s probabilities for its 10 classes.');
});

test('R0 shapes, attention scale, layer norm ε, GELU shortcut', () => {
	assert.match(text('tinycnn', 'op:pool1'), /shrinks the maps from \[8×24×24\] to \[8×12×12\]\./);
	assert.match(text('tinycnn', 'op:flatten'), /^Flatten lays the \[16×4×4\] grid out as one long list of 256 numbers/);
	const sc = text('tinygpt', 'op:L0.attn.scores');
	assert.match(sc, /dot product of their 19 numbers, multiplied by 0\.229 \(that is, divided by √19\)\./);
	assert.doesNotMatch(sc, /T×T/);
	assert.match(text('tinygpt', 'op:L0.ln1'), /ε = 0\.00001 so that a flat row never divides by 0\./);
	assert.match(text('tinygpt', 'op:L0.mlp.gelu'), /This layer computes Φ with a common tanh shortcut/);
	// every phrase var is bilingual
	for (const def of CORE_OPS) assert.ok(def.narration.what.zh, def.op);
	assert.deepEqual(words('a', 'b'), { en: 'a', zh: 'b' });
});

test('R0: every CORE_OP template gives an auto step of 2-4 sentences and at most 90 words (what + how; repeat alone), as lint counts them', () => {
	for (const def of CORE_OPS) {
		const n = def.narration;
		const parts: [string, string][] = [['what + how', [n.what.en, n.how?.en].filter(Boolean).join(' ')]];
		if (n.repeat) parts.push(['repeat', n.repeat.en]);
		for (const [field, en] of parts) {
			const s = sentenceCount(en, 'en');
			assert.ok(s >= 2 && s <= 4, `${def.op} ${field}: ${s} sentences`);
			assert.ok(wordCount(en) <= 90, `${def.op} ${field}: ${wordCount(en)} words`);
		}
	}
});
