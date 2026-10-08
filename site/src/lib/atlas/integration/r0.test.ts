// The R0 review's kit issues (LOG.md "DevKit R0 review: op narration"), checked on the shipped TinyGPT and TinyCNN assets
// with the manifests the Lesson Lab synthesizes for them:
//   1. the auto softmax step on attention weights [H, T, T] picks the LAST query's row (then the argmax), not the trivial
//      1.0 of the first query;
//   2. a language model's output step speaks of the next token and its vocabulary, with the token's text ("#501" never);
//   3. the group step names the blocks as the page labels them (Block 2 to Block 4, not "L1 to L3");
//   4. the input step of a text model shows the tokens (kit.tokens), not the token embedding;
//   5. glossary terms follow the structure: token + position is not a "residual connection"; a hidden linear layer's
//      outputs are not "logits"; a stride-1, unpadded convolution does not link "stride" or "padding".
// Run from site/: node --test src/lib/atlas/integration/r0.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeSpec } from '../graph/legacy.ts';
import { synthManifest } from '../ui/synth.ts';
import { CORE_OPS } from '../ops/index.ts';
import { compileLesson } from '../lesson/compile.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { stepAt } from '../runtime/sources.ts';
import { fsFetch } from '../core/io.ts';
import { textBpe, textInput } from '../adapters/text-bpe/adapter.ts';
import drawing28 from '../adapters/drawing28/adapter.ts';
import type { PackageModules } from '../types/index.ts';

const STATIC = join(fileURLToPath(new URL('../../../../', import.meta.url)), 'static');

async function open(slug: 'tinygpt' | 'tinycnn') {
	const dir = `models/${slug}`;
	const spec = normalizeSpec(JSON.parse(readFileSync(join(STATIC, dir, 'spec.json'), 'utf8')));
	let options: Record<string, unknown> | undefined;
	let start: unknown;
	if (slug === 'tinygpt') {
		const tok = JSON.parse(readFileSync(join(STATIC, dir, 'tokenizer.json'), 'utf8'));
		options = { tokenizer: tok };
		start = textInput(tok, 'Once upon a time, there was a little girl named Lily. She');
	}
	const manifest = synthManifest(spec, { slug, dir, presets: slug === 'tinycnn' ? 'presets.json' : null, auto: ['spec.json', 'weights.bin'], options, start });
	const pkg: PackageModules = { slug, manifest, lesson: null, ops: CORE_OPS, processes: {}, adapters: { 'text-bpe': textBpe, drawing28 } };
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC });
	const c = compileLesson(null, { spec: rt.spec, ops: pkg.ops, manifest, adapters: pkg.adapters, processes: {} });
	const step = (id: string) => {
		const s = c.steps.find((x) => x.id === id || x.id === id.replace(/[^A-Za-z0-9_-]/g, '-'));
		assert.ok(s, `step ${id} in ${c.steps.map((x) => x.id).join(', ')}`);
		return s;
	};
	return { rt, c, step };
}

test('R0 on TinyGPT: last-query softmax pick, next-token output, block names, token input step, structural terms', async () => {
	const { rt, step } = await open('tinygpt');
	const T = (rt.pre!.tensors.input as { shape: number[] }).shape[0];
	// 1. attention weights [4, T, T]: the last query's row
	const sm = step('op:L0.attn.weights');
	await stepAt(rt, sm);
	assert.equal(rt.sel.unit![1], T - 1, `query ${rt.sel.unit} is the last of ${T}`);
	const w = rt.sources.fwd.trace!.t['L0.attn.weights'];
	const [h, q, k] = rt.sel.unit!;
	let best = -Infinity;
	for (let hh = 0; hh < 4; hh++) for (let kk = 0; kk < T; kk++) best = Math.max(best, w.data[(hh * T + T - 1) * T + kk]);
	assert.equal(w.data[(h * T + q) * T + k], best, 'the argmax of the last rows');
	assert.ok(best < 1, 'not the trivial 1.0 of the first query');
	assert.deepEqual(sm.terms, ['softmax', 'attention']);
	// 2. the output step
	const out = step('output');
	const po = await stepAt(rt, out);
	const n = out.narration(rt, po.at);
	assert.match(n.en, /next token is “[^”]+”, with [\d.]+%/);
	assert.match(n.en, /1,024 tokens of its vocabulary/);
	assert.doesNotMatch(n.en, /#\d|classes/);
	assert.match(n.zh!, /下一个词元/);
	// 3. the group step
	const g = step('group:L1');
	const pg = await stepAt(rt, g);
	assert.equal(g.narration(rt, pg.at).en.split(' repeat')[0], 'Block 2 to Block 4');
	assert.match(g.narration(rt, pg.at).en, /same steps as Block 1,/);
	assert.match(g.narration(rt, pg.at).zh!, /^第 2 块到第 4 块重复第 1 块/);
	// 4. the input step
	assert.equal(step('input').view, 'kit.tokens');
	// 5. terms: token + position is not a residual connection; q is a hidden linear layer (no logits)
	assert.deepEqual(step('op:embed').terms, []);
	assert.deepEqual(step('op:L0.attn.q').terms, ['dense-layer', 'weight', 'bias']);
	rt.close();
});

test('R0 on TinyCNN: conv terms follow stride and padding; the dense layer gives logits; class output wording unchanged', async () => {
	const { rt, step } = await open('tinycnn');
	assert.deepEqual(step('op:conv1_1').terms, ['convolution', 'kernel', 'feature-map', 'receptive-field']);
	assert.deepEqual(step('op:dense').terms, ['dense-layer', 'weight', 'bias', 'logit']);
	const out = step('output');
	const po = await stepAt(rt, out);
	assert.match(out.narration(rt, po.at).en, /^The model answers \S+ with [\d.<]+%.*among the 10 classes/);
	rt.close();
});
