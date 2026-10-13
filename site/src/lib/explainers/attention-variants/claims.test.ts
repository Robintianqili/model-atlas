// attention-variants claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the adapter's presets and
// edge fixtures and the start input) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/attention-variants/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { configOf, divisors, kindOf, MLA, perBlock, perToken, PRESETS, presetConfig, scoreMacs, total, withKv, type Config } from './data.ts';
import { H, pooled } from './process.ts';
import { PROBE_IDS } from './evalset.ts';
import lesson from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'attention-variants';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', SLUG, 'facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const fact = (k: string) => facts[k].value;
const P = (id: string) => presetConfig(PRESETS.find((p) => p.id === id)!);
const gpt = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'spec.json'), 'utf8'));

/** TinyGPT's weights as the page loads them. */
function tinygpt(): Record<string, Tensor> {
	const buf = readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'weights.bin'));
	const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
	const w: Record<string, Tensor> = {};
	for (const t of gpt.tensors) w[t.name] = { shape: t.shape, data: all.slice(t.offset, t.offset + t.length) };
	return w;
}
async function shareOf(rt: ExplainerRuntime) {
	await rt.ensure('share', 'last');
	const t = rt.sources.share.tape!;
	return (phase: string, step = 0): Frame => t.frame(t.find({ phase, step }));
}
async function cacheOf(rt: ExplainerRuntime) {
	await rt.ensure('cache', 'first');
	return configOf({ ...rt.sources.cache.tape!.frame(0).metrics, preset: 'custom' });
}

test('step "cache": the formula 2 × layers × KV heads × head width counts exactly the keys and values TinyGPT computes per token', () => {
	const m = fromParts(gpt, tinygpt(), CORE_OPS);
	const T = PROBE_IDS.length;
	const tr = m.forward({ shape: [T], data: Int32Array.from(PROBE_IDS), dtype: 'i32' }, { record: 'layers' }).t;
	let n = 0;
	for (let l = 0; l < gpt.nLayer; l++) n += tr[`L${l}.attn.k`].data.length + tr[`L${l}.attn.v`].data.length;
	const c = P('tinygpt');
	assert.equal(n / T, c.layers * perBlock(c));
	assert.deepEqual([c.layers, c.heads, c.kvHeads, c.dHead], [gpt.nLayer, gpt.nHead, gpt.nHead, gpt.dHead]);
	// each new query is compared with the key of every earlier token: the causal mask leaves exactly i + 1 keys for query i
	const w = tr['L0.attn.weights'];
	for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) assert.equal(w.data[i * T + j] > 0, j <= i, `query ${i}, key ${j}`);
});

test('step "mha": TinyGPT gives every query head its own key and value projection (as many KV heads as query heads)', () => {
	const w = tinygpt();
	assert.equal(w['L0.attn.k.weight'].shape[0], gpt.nHead * gpt.dHead);
	assert.equal(w['L0.attn.v.weight'].shape[0], gpt.nHead * gpt.dHead);
	assert.equal(kindOf(P('tinygpt')), 0);
});

test('predict and quiz: one shared KV head makes the cache H times smaller; 8 of 32 KV heads make it a quarter of MHA', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const c = await cacheOf(rt);
		if (c.latent > 0) return;
		assert.equal(perToken(withKv(c, 1)) * c.heads, perToken(withKv(c, c.heads)));
		for (const k of divisors(c.heads)) assert.equal(perToken(withKv(c, k)) * c.heads, perToken(withKv(c, c.heads)) * k);
	}, opts);
	const l3 = P('llama3-8b');
	assert.equal(perToken(l3) * 4, perToken(withKv(l3, 32)));
});

test('step "gqa": MHA and MQA are the two ends of GQA (one query head per group, or one group)', () => {
	for (const p of PRESETS) {
		const c = presetConfig(p);
		assert.equal(kindOf(withKv(c, c.heads)), 0);
		assert.equal(kindOf(withKv(c, 1)), c.heads === 1 ? 0 : 2);
	}
});

test('step "shared": sharing = the mean of the group’s trained key and value heads; each head still scores q·k̄/√d with its own query, and heads 0 and 1 peak at different positions', async () => {
	const w = tinygpt();
	const d = gpt.dHead;
	const p = pooled(w, 2, [2], d);
	const K0 = w['L2.attn.k.weight'].data;
	const K = p['L2.attn.k.weight'].data;
	const D = gpt.dModel;
	for (let i = 0; i < d * D; i++) {
		assert.ok(Math.abs(K[i] - (K0[i] + K0[d * D + i]) / 2) < 1e-6);
		assert.equal(K[i], K[d * D + i]);
	}
	// the scores the stage shows are q·k̄/√d from the pooled model's own q and k
	const T = PROBE_IDS.length;
	const tr = fromParts(gpt, p, CORE_OPS).forward({ shape: [T], data: Int32Array.from(PROBE_IDS), dtype: 'i32' }, { record: 'layers' }).t;
	const q = tr['L2.attn.q'].data;
	const k = tr['L2.attn.k'].data;
	await withFixtures(await pkg(), async (rt) => {
		const f = (await shareOf(rt))('shared', 0);
		const s = (f.state as { s: Tensor }).s.data;
		for (const h of [0, 1])
			for (let j = 0; j < T; j++) {
				let dot = 0;
				for (let c = 0; c < d; c++) dot += q[(T - 1) * D + h * d + c] * k[j * D + h * d + c];
				assert.ok(Math.abs(dot / Math.sqrt(d) - s[h * T + j]) < 1e-4, `head ${h}, position ${j}`);
			}
		assert.notEqual(f.metrics!.arg0, f.metrics!.arg1, 'heads 0 and 1 peak at different positions');
	}, opts);
});

test('step "task": the KV-heads field offers exactly the divisors of the query heads, and a non-divisor snaps to the nearest one (ties to the smaller)', () => {
	assert.deepEqual(divisors(64), [1, 2, 4, 8, 16, 32, 64]);
	assert.equal(configOf({ heads: 12, kvHeads: 5 }).kvHeads, 4); // 4 and 6 are equally near: the smaller wins
	assert.equal(configOf({ heads: 12, kvHeads: 7 }).kvHeads, 6);
	assert.equal(configOf({ heads: 71, kvHeads: 8 }).kvHeads, 1);
	const done = lesson.steps.find((s) => 'id' in s && s.id === 'task')!;
	const fn = (done as unknown as { task: { done: (rt: unknown) => boolean } }).task.done;
	assert.equal(fn({ input: { ...P('llama2-70b'), kvHeads: 64 } }), true);
	assert.equal(fn({ input: P('llama2-70b') }), false);
});

test('step "presets": the published configurations (layers, query heads, KV heads, head width), 2-byte numbers except TinyGPT', () => {
	const cfg = (c: Config) => [c.layers, c.heads, c.kvHeads, c.dHead];
	assert.deepEqual(cfg(P('gpt2-small')), [12, 12, 12, 64]); // GPT-2 small: 12 layers, width 768 = 12 heads × 64
	assert.deepEqual(cfg(P('llama2-7b')), [32, 32, 32, 128]); // Llama 2 7B: MHA
	assert.deepEqual(cfg(P('llama2-70b')), [80, 64, 8, 128]); // Llama 2 70B: GQA, 8 KV heads
	assert.deepEqual(cfg(P('llama3-8b')), [32, 32, 8, 128]); // Llama 3 8B: GQA, 8 KV heads
	assert.deepEqual(cfg(P('falcon-7b')), [32, 71, 1, 64]); // Falcon-7B: width 4,544 = 71 × 64, multi-query
	assert.deepEqual([P('deepseek-v2').layers, P('deepseek-v2').heads, P('deepseek-v2').dHead, P('deepseek-v2').latent], [60, 128, 128, MLA.dc + MLA.dr]);
	assert.deepEqual([MLA.dc, MLA.dr], [512, 64]);
	for (const p of PRESETS) assert.equal(p.cfg.bytes, p.id === 'tinygpt' ? 4 : 2);
	// "Llama 2 70B ... caches less than Llama 2 7B"
	assert.ok(perToken(P('llama2-70b')) < perToken(P('llama2-7b')));
	assert.equal(perToken(P('llama2-7b')), 512 * 1024);
});

test('step "mla": MLA caches d_c + d_R numbers per token and layer, the cache of GQA with (d_c + d_R) / 2d KV heads', () => {
	const c = P('deepseek-v2');
	assert.equal(perBlock(c), MLA.dc + MLA.dr);
	assert.equal(perBlock(withKv(c, 1)) * ((MLA.dc + MLA.dr) / (2 * c.dHead)), perBlock(c));
	assert.equal(kindOf(c), 3);
});

test('step "convert": converting raises perplexity and KL as the KV heads go from 4 to 1; each head had learned its own keys (the heads of a group differ)', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const F = await shareOf(rt);
		const [a, b, c] = [F('eval', 0).metrics!, F('eval', 1).metrics!, F('eval', 2).metrics!];
		assert.ok(a.ppl < b.ppl && b.ppl < c.ppl);
		assert.ok(a.kl === 0 && a.kl < b.kl && b.kl < c.kl);
		assert.ok(b.ppl > 2 * a.ppl, 'much worse');
	}, opts);
	const w = tinygpt();
	const d = gpt.dHead;
	const D = gpt.dModel;
	for (let l = 0; l < gpt.nLayer; l++) {
		const K = w[`L${l}.attn.k.weight`].data;
		for (let h = 0; h < H; h += 2) {
			let diff = 0;
			let norm = 0;
			for (let i = 0; i < d * D; i++) {
				diff += (K[h * d * D + i] - K[(h + 1) * d * D + i]) ** 2;
				norm += K[h * d * D + i] ** 2;
			}
			assert.ok(Math.sqrt(diff / norm) > 0.5, `block ${l}, heads ${h} and ${h + 1}`);
		}
	}
});

test('step "uptrain": 1,500 steps are 5% of TinyGPT’s training (Ainslie et al.’s share); uptraining repairs most of the loss but not all, and GQA ends closer to MHA than MQA', () => {
	assert.equal(fact('pretrainSteps'), gpt.training.steps);
	assert.equal(fact('upSteps') / fact('pretrainSteps'), 0.05);
	for (const v of ['Gqa', 'Mqa']) {
		const [p0, p1, m] = [fact(`ppl${v}0`), fact(`ppl${v}1500`), fact('pplMha1500')];
		assert.ok((p0 - p1) / (p0 - m) > 0.5, `${v}: most of it`);
		assert.ok(p1 > m, `${v}: not all of it`);
	}
	assert.ok(Math.abs(fact('pplGqa1500') - fact('pplMha1500')) < Math.abs(fact('pplMqa1500') - fact('pplMha1500')));
	assert.equal(fact('pplMha0'), 6.8786); // the trained model on the same 64,000 tokens (the quantization page's bigPpl0)
});

test('step "limits": the score arithmetic is the same for every number of KV heads; the cache grows with every token', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const c = await cacheOf(rt);
		for (const k of divisors(c.heads)) assert.equal(scoreMacs(withKv(c, k)), scoreMacs(c));
		assert.equal(total({ ...c, tokens: c.tokens + 1 }) - total(c), perToken(c));
	}, opts);
});

test('manifest and hook: with long contexts the KV cache can outgrow the weights (Llama 2 7B, MHA, 2-byte numbers, 32,768 tokens)', () => {
	const c = { ...P('llama2-7b'), tokens: 32768 };
	assert.ok(total(c) > 7e9 * 2);
});

test('lessonNote: eleven steps', () => {
	assert.equal(lesson.steps.length, 11);
});
