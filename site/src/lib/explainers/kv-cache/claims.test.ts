// kv-cache claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the lesson's fixtures,
// the adapter's edge fixtures, the presets and the start input) through withFixtures. `atlas check` runs this file.
// Run alone: node --test src/lib/explainers/kv-cache/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { tokenizerOf } from '../../atlas/adapters/text-bpe/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { cacheBytes, dimsOf, EOT, KVCache, run } from './gpt.ts';
import { gptOf } from './explainer.ts';
import { NEW, ONE_UPON, PRESETS, PROMPT_MAX, promptA, promptOf, TASK_SAME, taskDone } from './data.ts';
import { VOCAB } from './vocab.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'kv-cache';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', SLUG, 'facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const fact = (k: string) => facts[k].value;
const tok = tokenizerOf(JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'tokenizer.json'), 'utf8')))!;

async function framesOf(rt: ExplainerRuntime, source: string): Promise<Frame[]> {
	await rt.ensure(source, 'last');
	const t = rt.sources[source].tape!;
	return Array.from({ length: t.length }, (_, i) => t.frame(i));
}

test('vocab.ts is TinyGPT’s tokenizer, and every preset’s ids are its encoding of the preset’s texts', () => {
	assert.equal(VOCAB.length, 1024);
	VOCAB.forEach((v, id) => assert.equal(v, tok.tokenToDisplay(id), `token ${id}`));
	for (const p of [...PRESETS.map((x) => x.raw), ONE_UPON]) {
		assert.deepEqual(p.ids, tok.encode(p.text), p.text);
		assert.deepEqual(p.ids2, tok.encode(p.text2), p.text2);
	}
	assert.equal(tok.tokenToDisplay(EOT), '<|endoftext|>');
});

test('preset labels state the shared prefix; step "break" task: “Once” → “One” keeps every later token in place', () => {
	const shared = (r: { ids: number[]; ids2: number[] }) => r.ids.findIndex((v, i) => v !== r.ids2[i]);
	for (const p of PRESETS) assert.ok(p.label.en.includes(`(${shared(p.raw)} shared tokens)`) && (p.label.zh ?? '').includes(`共享 ${shared(p.raw)} 个词元`), p.id);
	assert.equal(ONE_UPON.ids2.length, PRESETS[0].raw.ids2.length);
	assert.ok(taskDone(ONE_UPON));
	for (const p of PRESETS) assert.ok(!taskDone(p.raw), `preset ${p.id} must not already do the task`);
});

test('the page keeps the first 40 prompt tokens and generates 24 new ones, filling TinyGPT’s 64-token window (no position beyond it)', async () => {
	assert.equal(PROMPT_MAX + NEW, 64);
	assert.deepEqual(promptOf([]), [EOT]);
	assert.equal(promptOf(Array.from({ length: 70 }, () => 97)).length, PROMPT_MAX);
	const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'spec.json'), 'utf8'));
	assert.equal(spec.context, 64);
	assert.deepEqual(spec.tensors.find((t: { name: string }) => t.name === 'pos_emb.weight').shape, [64, 76]);
	await withFixtures(
		await pkg(),
		async (rt) => {
			const f = await framesOf(rt, 'gen');
			assert.equal(f.length, NEW);
			assert.ok(f[0].metrics!.P <= PROMPT_MAX);
			assert.equal(f.at(-1)!.metrics!.n, f[0].metrics!.P + NEW - 1);
		},
		opts
	);
});

test('steps "predict", "cache", "same" and the misconception: earlier keys and values never change, both runs pick the same tokens with identical logits, and the cached attention rows are the uncached ones', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const f = await framesOf(rt, 'gen');
			for (const x of f) {
				assert.equal(x.metrics!.dKV, 0, `dKV at pass ${x.step}`);
				assert.equal(x.metrics!.dLogit, 0, `dLogit at pass ${x.step}`);
				assert.equal(x.metrics!.token, x.metrics!.tokFull);
			}
			const z = f.at(-1)!.metrics!;
			assert.equal(z.sameAll, z.maxNew);
			assert.equal(z.dLogitMax, 0);
			assert.ok(z.dKit < 1e-4, `the kit's forward pass differs by ${z.dKit}`);
			// the uncached run's attention rows (one pass over every token read) equal the cached run's, row by row
			const g = gptOf(rt)!;
			const d = dimsOf(g.spec);
			const st = rt.sources.gen.tape!.state(f.length - 1) as unknown as { ids: { data: Float64Array }; att: { data: Float32Array } };
			const ids = Array.from(st.ids.data).filter((v) => v >= 0).slice(0, z.n);
			const att = new Float32Array(d.L * d.C * d.C);
			run(g, ids, new KVCache(d), { macs: 0 }, att);
			for (let i = 0; i < att.length; i++) assert.equal(att[i], st.att.data[i], `attention entry ${i}`);
		},
		opts
	);
});

test('step "generate": each pass picks the most likely next token (greedy), the kit’s own forward pass agreeing', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const f = await framesOf(rt, 'gen');
			const g = gptOf(rt)!;
			const st = rt.sources.gen.tape!.state(0) as unknown as { ids: { data: Float64Array } };
			const ids = Array.from(st.ids.data).filter((v) => v >= 0);
			assert.deepEqual(ids.slice(0, f[0].metrics!.P), promptA(rt.input as never));
			for (const x of f) {
				const lc = run(g, ids.slice(0, x.metrics!.n), new KVCache(dimsOf(g.spec)), { macs: 0 });
				let best = 0;
				for (let i = 1; i < lc.length; i++) if (lc[i] > lc[best]) best = i;
				assert.equal(x.metrics!.token, best);
				assert.equal(ids[x.metrics!.n], best);
			}
		},
		opts
	);
});

test('step "cost" (and its look): uncached work per pass grows in step with the length, cached work stays nearly flat (a straight total), the gap widens; pass 0 costs the same either way', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const m = (await framesOf(rt, 'gen')).map((x) => x.metrics!);
			assert.equal(m[0].macsFull, m[0].macsCache, 'the prefill costs the same');
			// nearly linear in the length: every extra token read adds about the same work (attention adds a little more each time)
			const step = m.slice(1).map((x, k) => x.macsFull - m[k].macsFull);
			assert.ok(Math.min(...step) > 0 && Math.max(...step) / Math.min(...step) < 1.2, `uncached work added per token: ${step}`);
			const c = m.slice(1).map((x) => x.macsCache);
			assert.ok(Math.max(...c) / Math.min(...c) < 1.05, `cached work per pass: ${c}`);
			for (let k = 2; k < m.length; k++) {
				assert.ok(m[k].macsFull > m[k - 1].macsFull);
				assert.ok(m[k].cumFull - m[k].cumCache > m[k - 1].cumFull - m[k - 1].cumCache);
			}
		},
		opts
	);
});

test('step "memory" and quiz 1: cache size = layers × heads × tokens × head size × 2 × bytes; the large models from their configurations', () => {
	assert.equal(cacheBytes({ L: 2, H: 2, dh: 8 } as never, 10), 2560);
	const d = dimsOf(JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'spec.json'), 'utf8')));
	assert.deepEqual([d.L, d.H, d.dh, d.C, d.H * d.dh], [4, 4, 19, 64, 76]);
	assert.equal(cacheBytes(d, 64), 155648);
	assert.equal(fact('gpt2Heads') * fact('gpt2DHead'), 768);
	assert.equal(fact('llamaHeads') * fact('llamaDHead'), 4096);
	assert.equal(fact('gpt2Layers') * fact('gpt2Heads') * fact('gpt2Ctx') * fact('gpt2DHead') * 2 * fact('halfBytes'), 37748736);
	assert.equal(fact('llamaLayers') * fact('llamaHeads') * fact('llamaCtx') * fact('llamaDHead') * 2 * fact('halfBytes'), 2147483648);
});

test('step "prefix": reusing the shared prefix gives B exactly the logits of a run from scratch, and saves work whenever something is shared', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const [f] = await framesOf(rt, 'prefix');
			const m = f.metrics!;
			assert.equal(m.dReuse, 0);
			assert.equal(m.computed + m.reused, m.PB);
			if (m.reused > 0) assert.ok(m.macsReuse < m.macsB);
			else assert.equal(m.macsReuse, m.macsB);
		},
		opts
	);
});

test('step "break", quiz 2 and the last pitfall: after the first difference, layer 0 keys and values of the same token in the same place match, later layers differ, so nothing past the difference is reusable', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const [f] = await framesOf(rt, 'prefix');
			const m = f.metrics!;
			if (m.nSame > 0) {
				assert.equal(m.dK0, 0);
				assert.ok(m.dKdeep > 0 && m.dVdeep > 0, 'later layers differ');
			}
			if (rt.input === ONE_UPON || JSON.stringify(rt.input) === JSON.stringify(ONE_UPON)) {
				assert.equal(m.reused, 0, 'quiz 2: none of B reuses A’s cache');
				assert.ok(m.nSame >= TASK_SAME);
				assert.ok(m.dKdeep > 0.1 && m.dWrong > 0.01, `dKdeep ${m.dKdeep}, dWrong ${m.dWrong}`);
			}
		},
		opts
	);
});
