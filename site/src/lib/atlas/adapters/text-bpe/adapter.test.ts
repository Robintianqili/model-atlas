// text-bpe (W1-D1): TinyGPT's tokenizer, the context window (last tokens kept, as tinygpt.ts does), edge fixtures that
// hold for byte-level BPE, and the TinyGPT auto lesson on the real adapter (lint 0 errors, en and zh).
// Run from site/: node --test src/lib/atlas/adapters/text-bpe/adapter.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORE_OPS } from '../../ops/index.ts';
import { fsFetch } from '../../core/io.ts';
import { normalizeSpec } from '../../graph/legacy.ts';
import { compileLesson } from '../../lesson/compile.ts';
import { fixtureCases, lintLesson } from '../../lesson/lint.ts';
import { createRuntime } from '../../runtime/runtime.ts';
import type { PackageModules } from '../../types/index.ts';
import { synthManifest } from '../../ui/synth.ts';
import { BPETokenizer } from '../../../models/bpe.ts';
import { fromWeights } from '../../../models/tinygpt.ts';
import { rawIds, textBpe, textInput, tokenizerOf } from './adapter.ts';

const SITE = fileURLToPath(new URL('../../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const DIR = join(STATIC, 'models', 'tinygpt');
const tokJson = JSON.parse(readFileSync(join(DIR, 'tokenizer.json'), 'utf8'));
const rawSpec = JSON.parse(readFileSync(join(DIR, 'spec.json'), 'utf8'));
const spec = normalizeSpec(rawSpec);
const TEXT = 'Once upon a time, there was a little girl named Lily. She';

test('text-bpe: ids come from TinyGPT’s own tokenizer; ids win over text; bare text needs the tokenizer', () => {
	const tok = new BPETokenizer(tokJson);
	const r = textInput(tokJson, TEXT);
	assert.deepEqual(r.ids, tok.encode(TEXT));
	assert.equal(tokenizerOf(tokJson), tokenizerOf(tokJson), 'memoized per tokenizer.json object');
	const p = textBpe.preprocess(r, spec, {});
	assert.deepEqual(Array.from(p.tensors.input.data), r.ids);
	assert.equal((p.tensors.input as { dtype?: string }).dtype, 'i32');
	assert.deepEqual(rawIds({ text: 'ignored', ids: [5, 6] }).ids, [5, 6]);
	assert.throws(() => textBpe.preprocess('hello', spec, {}), /tokenizer/);
	assert.deepEqual(Array.from(textBpe.preprocess('hello', spec, { tokenizer: tokJson }).tensors.input.data), tok.encode('hello'));
	assert.throws(() => textBpe.preprocess({ ids: [5000] }, spec, {}), /vocabulary/);
});

test('text-bpe: the edge fixtures are honest for this tokenizer, and a long input keeps its last 64 tokens', () => {
	const tok = new BPETokenizer(tokJson);
	const fx = Object.fromEntries(textBpe.edgeFixtures().map((f) => [f.id, f.raw as { text?: string; ids: number[] }]));
	assert.deepEqual(fx.one.ids, tok.encode(fx.one.text!), "'a' is the single token 97");
	const empty = textBpe.preprocess(fx.empty, spec, {});
	assert.equal(empty.empty, true);
	assert.ok(empty.notes?.[0]?.zh);
	const long = textBpe.preprocess(fx.long, spec, {});
	assert.deepEqual(long.tensors.input.shape, [64]);
	assert.deepEqual(Array.from(long.tensors.input.data), fx.long.ids.slice(6));
	assert.match(long.notes![0].en, /last 64 tokens/);
	// the same truncation as tinygpt.ts
	const wb = readFileSync(join(DIR, 'weights.bin'));
	const flat = new Float32Array(wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength));
	const gpt = fromWeights(rawSpec, rawSpec.tensors, flat);
	assert.deepEqual(gpt.forward(fx.long.ids).ids, Array.from(long.tensors.input.data));
});

test('TinyGPT auto lesson on the real text-bpe adapter: lint 0 errors on a story prompt and the edge fixtures (en, zh)', async () => {
	const manifest = synthManifest(spec, { dir: 'models/tinygpt', auto: ['spec.json', 'weights.bin', 'tokenizer.json'], options: { tokenizer: tokJson }, start: textInput(tokJson, TEXT) });
	const pkg: PackageModules = { slug: 'tinygpt', manifest, lesson: null, ops: CORE_OPS, processes: {}, adapters: { 'text-bpe': textBpe } };
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC });
	assert.deepEqual(rt.warnings, []);
	assert.equal(rt.sources.fwd.status, 'ready');
	const c = compileLesson(null, { spec: rt.spec, ops: CORE_OPS, manifest, adapters: pkg.adapters });
	const fixtures = [{ id: 'story', label: { en: 'story', zh: '故事' }, raw: textInput(tokJson, TEXT) }, ...fixtureCases(pkg, rt.presets)];
	const issues = await lintLesson(c, (fx) => createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: fx.raw }), fixtures);
	const errors = issues.filter((i) => i.level === 'error');
	const warns = [...new Set(issues.filter((i) => i.level === 'warn').map((i) => `${i.code}${i.step ? ' ' + i.step : ''}`))];
	console.log(`ℹ TinyGPT auto lesson (text-bpe): ${c.steps.length} steps over ${fixtures.length} fixtures; ${errors.length} errors; warnings: ${warns.join('; ')}`);
	assert.deepEqual(errors.map((e) => `${e.code} ${e.step ?? ''} ${e.fixture ?? ''} ${e.lang ?? ''} ${e.message}`), []);
	rt.close();
});
