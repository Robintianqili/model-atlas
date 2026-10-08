// W1-A acceptance 5 and the hand-written validators: both shipped specs validate after normalizeSpec (ops, attrs,
// weights, inferShape at the dim extremes); legacy.ts does what design §3.2 says; reference and facts validators.
// (Lives in graph/ because W1-A owns schema/{spec,ref,facts}.ts but not other files in schema/.)
// Run from site/: node --test src/lib/atlas/graph/schema.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeSpec } from './legacy.ts';
import { fromParts } from './executor.ts';
import { parseReference } from './refs.ts';
import { CORE_OPS } from '../ops/index.ts';
import { assertSpec, errorsOf, formatIssues, validateSpec } from '../schema/spec.ts';
import { validateRef } from '../schema/ref.ts';
import { readFacts, validateFacts } from '../schema/facts.ts';
import { f32ToB64, i32ToB64 } from '../core/io.ts';
import { intTensor } from '../core/tensor.ts';
import type { AtlasSpec } from '../types/index.ts';

const MODELS = fileURLToPath(new URL('../../../../static/models/', import.meta.url));
const raw = (slug: string, file = 'spec.json') => readFileSync(join(MODELS, slug, file), 'utf8');

test('acceptance 5: both shipped specs validate after normalizeSpec (CORE_OPS, attrs, weights, inferShape at dim extremes)', (t) => {
	for (const slug of ['tinycnn', 'tinygpt']) {
		const text = raw(slug);
		const bytes = readFileSync(join(MODELS, slug, 'weights.bin')).byteLength;
		const issues = validateSpec(JSON.parse(text), { ops: CORE_OPS, weightsBytes: bytes, folder: slug });
		const errs = errorsOf(issues);
		assert.equal(errs.length, 0, `${slug}:\n${formatIssues(errs)}`);
		t.diagnostic(`${slug}: 0 errors, ${issues.length} warnings${issues.length ? ': ' + issues.map((i) => `${i.path} ${i.message}`).join('; ') : ''}`);
		assert.equal(raw(slug), text, 'the shipped file is never rewritten');
	}
});

test('normalizeSpec: TinyGPT per design §3.2', () => {
	const src = JSON.parse(raw('tinygpt'));
	const before = JSON.stringify(src);
	const s = normalizeSpec(src);
	assert.equal(JSON.stringify(src), before, 'argument not mutated');
	assert.deepEqual(s.inputs, [{ name: 'input', adapter: 'text-bpe', kind: 'text', shape: ['T'], dtype: 'i32', axes: ['t'] }]);
	assert.deepEqual(s.dims, { T: { min: 1, max: 64, label: { en: 'tokens', zh: '词元数' } } });
	const logits = s.layers.find((l) => l.id === 'logits')!;
	assert.equal(logits.op, 'linear');
	assert.deepEqual(logits.w, { weight: 'tok_emb.weight' });
	assert.deepEqual(logits.params, { accumulate: 'bias-first' });
	const q = s.layers.find((l) => l.id === 'L0.attn.q')!;
	assert.deepEqual(q.w, { weight: 'L0.attn.q.weight', bias: 'L0.attn.q.bias' });
	assert.deepEqual(q.params, { nHead: 4, dHead: 19, accumulate: 'bias-first' });
	assert.equal(s.layers.find((l) => l.id === 'L0.attn.masked')!.op, 'causal_mask');
	assert.deepEqual(s.layers.find((l) => l.id === 'tok_emb')!.in, ['input']);
	assert.deepEqual(s.layers.find((l) => l.id === 'L0.attn.scores')!.axes, ['h', 'tq', 'tk']);
	assert.deepEqual(logits.axes, ['t', 'vocab']);
	assert.equal(s.parity.bar, 1e-4, 'tolerance read as bar');
	assert.equal(s.parity.pass, true, 'passed read as pass');
	assert.equal(s.atlas?.spec, '1.1');
	assert.deepEqual(s.outputs, ['probs']);
	assert.deepEqual(normalizeSpec(s), s, 'idempotent');
	assert.ok(!s.layers.some((l) => l.params && 'transposed' in l.params), 'transposed dropped');
});

test('normalizeSpec: TinyCNN per design §3.2', () => {
	const s = normalizeSpec(JSON.parse(raw('tinycnn')));
	assert.deepEqual(s.inputs, [{ name: 'input', adapter: 'drawing28', kind: 'drawing', shape: [1, 28, 28], dtype: 'f32', axes: ['c', 'y', 'x'] }]);
	const input = s.layers[0];
	assert.equal(input.op, 'input');
	assert.deepEqual(input.in, ['input']);
	assert.equal(s.layers.find((l) => l.id === 'dense')!.op, 'dense', 'dense resolves through its alias');
	assert.equal(s.layers.find((l) => l.id === 'conv1_1')!.params!.accumulate, undefined, 'TinyCNN keeps the default bias-last');
	assert.deepEqual(s.layers.find((l) => l.id === 'dense')!.axes, ['class']);
	assert.deepEqual(s.layers.find((l) => l.id === 'flatten')!.axes, ['f']);
	assert.equal(s.dims, undefined);
	assert.equal(s.parity.bar, 1e-4);
});

test('validateSpec catches broken specs (AK301 / AK302)', () => {
	const base = (): AtlasSpec => normalizeSpec(JSON.parse(raw('tinycnn')));
	const codes = (s: unknown) => errorsOf(validateSpec(s, { ops: CORE_OPS })).map((i) => `${i.code} ${i.path}`);
	let s = base();
	s.layers[3].outShape = [8, 25, 25];
	assert.ok(codes(s).some((c) => c.startsWith('AK302')), 'outShape mismatch is AK302');
	s = base();
	s.layers[1].op = 'conv3d';
	assert.ok(codes(s).some((c) => c.includes('.op')), 'unregistered op');
	s = base();
	s.layers[1].params = { ...s.layers[1].params, kernal: 3 };
	assert.ok(codes(s).some((c) => c.includes('params.kernal')), 'unknown attribute');
	s = base();
	s.layers[1].params = { ...s.layers[1].params, stride: 1.5 };
	assert.ok(codes(s).some((c) => c.includes('params.stride')), 'wrong attribute type');
	s = base();
	s.layers[2].in = ['nowhere'];
	assert.ok(codes(s).some((c) => c.includes('.in')), 'dangling input');
	s = base();
	s.layers[2].id = 'relu#1';
	assert.ok(codes(s).some((c) => c.includes('.id')), "'#' in an id");
	s = base();
	s.tensors[1].length = 9;
	assert.ok(codes(s).length > 0, 'tensor length');
	s = base();
	s.tensors = s.tensors.filter((x) => x.name !== 'conv1_1.weight');
	assert.ok(codes(s).some((c) => c.includes("'conv1_1'")), 'missing weight');
	const g = normalizeSpec(JSON.parse(raw('tinygpt')));
	g.dims!.T.max = 65; // position table has 64 rows
	assert.ok(codes(g).some((c) => c.startsWith('AK302')), 'T = 65 overflows the position table at the dim extreme');
	assert.throws(() => assertSpec({ slug: 'x' }), /invalid/);
	assert.deepEqual(errorsOf(validateSpec(42)).length, 1);
});

test('validateRef: both shipped references, and an atlas-ref-1 file written from the executor', () => {
	for (const slug of ['tinycnn', 'tinygpt']) {
		const spec = normalizeSpec(JSON.parse(raw(slug)));
		const issues = errorsOf(validateRef(JSON.parse(raw(slug, 'reference.json')), { spec }));
		assert.equal(issues.length, 0, `${slug}: ${JSON.stringify(issues.slice(0, 3))}`);
	}
	// build a small atlas-ref-1 from a live TinyGPT trace and read it back
	const spec = normalizeSpec(JSON.parse(raw('tinygpt')));
	const w = readFileSync(join(MODELS, 'tinygpt', 'weights.bin'));
	const m = fromParts(spec, w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength) as ArrayBuffer, CORE_OPS);
	const ids = [428, 437, 258];
	const tr = m.forward(intTensor([3], ids));
	const keep = ['embed', 'L0.attn.masked', 'probs'];
	const ref = {
		format: 'atlas-ref-1',
		model: 'tinygpt',
		version: spec.version,
		against: 'pytorch-fp32-cpu',
		note: 'test',
		cases: [{ id: 'c1', label: 'three tokens', edge: false, inputs: { input: { shape: [3], dtype: 'int32', b64: i32ToB64(ids) } }, order: keep, tensors: Object.fromEntries(keep.map((k) => [k, { shape: tr.t[k].shape, dtype: 'float32', b64: f32ToB64(tr.t[k].data) }])) }]
	};
	assert.deepEqual(errorsOf(validateRef(ref, { spec })), []);
	const back = parseReference(ref);
	assert.deepEqual(Array.from(back[0].inputs.input.data), ids);
	assert.equal(back[0].tensors['L0.attn.masked'].data[1], -Infinity, '−∞ survives');
	const bad = structuredClone(ref);
	bad.cases[0].tensors.embed.b64 = bad.cases[0].tensors.embed.b64.slice(4);
	bad.cases[0].tensors['nope#x'] = bad.cases[0].tensors.probs;
	const errs = errorsOf(validateRef(bad, { spec })).map((i) => i.path);
	assert.ok(errs.some((p) => p.endsWith('embed.b64')), 'payload length');
	assert.ok(errs.some((p) => p.includes('nope#x')), 'unknown id');
});

test('validateFacts / readFacts', () => {
	const ok = { format: 'atlas-facts/1', facts: { test_acc: { value: 0.9681, unit: 'fraction', source: 'pipeline/tinycnn/train.py', measured: '2026-10-05, test split' }, best: { value: { en: 'house', zh: '房子' }, source: 'x.py', measured: 'now' } } };
	assert.deepEqual(errorsOf(validateFacts(ok)), []);
	assert.deepEqual(readFacts(ok), { test_acc: 0.9681, best: { en: 'house', zh: '房子' } });
	const bad = { format: 'atlas-facts/2', facts: { '1x': { value: NaN, source: '', measured: 3 } } };
	const paths = errorsOf(validateFacts(bad)).map((i) => i.path);
	assert.deepEqual(paths.sort(), ['facts.1x', 'facts.1x.measured', 'facts.1x.source', 'facts.1x.value', 'format'].sort());
	assert.throws(() => readFacts(bad), /invalid/);
});
