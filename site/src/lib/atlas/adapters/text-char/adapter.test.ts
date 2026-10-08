// text-char (W1-D1): the alphabet from the spec or options, unknown characters (unk id or dropped, with a note), the
// context window (last characters kept), edge fixtures, and a char model's executor run on the adapter's output.
// Run from site/: node --test src/lib/atlas/adapters/text-char/adapter.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CORE_OPS } from '../../ops/index.ts';
import { fromParts } from '../../graph/executor.ts';
import type { AtlasSpec } from '../../types/index.ts';
import { charVocab, textChar } from './adapter.ts';

const VOCAB = ' abcdefghijklmnopqrstuvwxyz.';

test('text-char: ids are alphabet positions; unknown characters are dropped or mapped to unk, with a note', () => {
	const p = textChar.preprocess('hi there', { vocab: VOCAB }, {});
	assert.deepEqual(Array.from(p.tensors.input.data), [...'hi there'].map((c) => VOCAB.indexOf(c)));
	assert.equal((p.tensors.input as { dtype?: string }).dtype, 'i32');
	const q = textChar.preprocess('Hi!', { vocab: VOCAB }, {});
	assert.deepEqual(Array.from(q.tensors.input.data), [VOCAB.indexOf('i')]);
	assert.match(q.notes![0].en, /left out: H !/);
	const u = textChar.preprocess('Hi!', { chars: [...VOCAB, '?'], unk: '?' }, {});
	assert.deepEqual(Array.from(u.tensors.input.data), [28, VOCAB.indexOf('i'), 28]);
	assert.match(u.notes![0].en, /unknown/);
	assert.equal(charVocab(null, { vocab: 'ab' }).chars.length, 2, 'options.vocab wins');
	assert.throws(() => textChar.preprocess('x', {}, {}), /vocabulary/);
});

test('text-char: edge fixtures and the context window', () => {
	const spec = { vocab: VOCAB, dims: { T: { min: 1, max: 128 } } };
	const fx = Object.fromEntries(textChar.edgeFixtures().map((f) => [f.id, textChar.preprocess(f.raw, spec, {})]));
	assert.equal(fx.empty.empty, true);
	assert.ok(fx.empty.notes?.[0]?.zh);
	assert.deepEqual(fx.one.tensors.input.shape, [1]);
	assert.deepEqual(fx.long.tensors.input.shape, [128]);
	const text = 'once upon a time '.repeat(16);
	assert.deepEqual(Array.from(fx.long.tensors.input.data), [...text.slice(-128)].map((c) => VOCAB.indexOf(c)));
	assert.match(fx.long.notes!.at(-1)!.en, /last 128 characters/);
});

test('text-char feeds a char model (embedding -> linear) through the executor', () => {
	const V = VOCAB.length;
	const D = 3;
	const spec = {
		slug: 'char-demo',
		name: { en: 'char demo', zh: '字符演示' },
		version: 'test',
		params: V * D + D * V + V,
		precision: 'fp32',
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'test' },
		inputs: [{ name: 'input', adapter: 'text-char', kind: 'text', shape: ['T'], dtype: 'i32', axes: ['t'] }],
		dims: { T: { min: 1, max: 16, label: { en: 'characters', zh: '字符数' } } },
		vocab: VOCAB,
		tensors: [
			{ name: 'emb.weight', shape: [V, D], offset: 0, length: V * D },
			{ name: 'head.weight', shape: [V, D], offset: V * D, length: V * D },
			{ name: 'head.bias', shape: [V], offset: 2 * V * D, length: V }
		],
		layers: [
			{ id: 'emb', op: 'embedding', label: { en: 'Embedding', zh: '嵌入' }, in: ['input'], outShape: ['T', D] },
			{ id: 'head', op: 'linear', label: { en: 'Head', zh: '输出头' }, in: ['emb'], outShape: ['T', V] }
		],
		parity: { against: 'none', max_abs_diff: null, checked: null }
	} as unknown as AtlasSpec;
	const w = new Float32Array(2 * V * D + V).map((_, i) => ((i * 37) % 11) / 10 - 0.5);
	const model = fromParts(spec, w.buffer, CORE_OPS);
	const pre = textChar.preprocess('a cat.', spec, {});
	const trace = model.forward(pre.tensors);
	assert.deepEqual(trace.t.head.shape, [6, V]);
	assert.ok(Array.from(trace.t.head.data).every(Number.isFinite));
});
