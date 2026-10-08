// The CORE_OPS contract (design §4.3, §8.1 W1-A deliverables): 20 ops + 3 aliases, each with forward, attrs, explain,
// receptive (spatial and elementwise ops), formula, bilingual narration (same placeholders in en and zh, AK403; no
// literal multi-digit numbers, AK402), vars, a parity bar and a pytorch line; glossary terms exist; schema/ops.json is
// generated from this list and up to date.
// Run from site/: node --test src/lib/atlas/ops/ops.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { CORE_OPS, coreRegistry, createRegistry } from './index.ts';
import { manifestText, buildManifest } from './manifest.ts';
import { GLOSSARY_BY_ID } from '../../catalog/glossary.ts';
import type { AtlasLayer } from '../types/index.ts';

const EXPECTED = [
	'input', 'conv2d', 'linear', 'relu', 'tanh', 'sigmoid', 'silu', 'gelu', 'maxpool2d', 'flatten',
	'concat', 'add', 'embedding', 'position_embedding', 'layernorm', 'attention_scores', 'causal_mask', 'softmax', 'attention_mix', 'lstm'
];
const PLACEHOLDER = /\{([A-Za-z_][\w.:#-]*)(?:\|([a-z0-9]+))?\}/g;
const FMTS = new Set(['n3', 'int', 'pct', 'signed', 'shape', 'label', 'tok']);
const SPATIAL = new Set(['conv2d', 'maxpool2d']);
const ELEMENTWISE = new Set(['relu', 'tanh', 'sigmoid', 'silu', 'gelu', 'add']);

test('CORE_OPS: exactly the 20 ops of design §4.3, plus aliases dense, linear_tied, gelu_tanh', () => {
	assert.deepEqual(CORE_OPS.map((d) => d.op), EXPECTED);
	const names = coreRegistry.list();
	for (const a of ['dense', 'linear_tied', 'gelu_tanh']) assert.ok(names.includes(a), `alias ${a}`);
	assert.equal(names.length, 23);
	assert.equal(coreRegistry.get('dense').def.op, 'linear');
	assert.equal(coreRegistry.get('linear_tied').def.op, 'linear');
	assert.deepEqual(coreRegistry.get('gelu_tanh').params, { approx: 'tanh' });
	assert.throws(() => coreRegistry.get('nope'), /not registered/);
	// a later op of the same name shadows a core op (local ops)
	const local = { ...CORE_OPS[3], label: { en: 'local relu', zh: '本地 relu' } };
	assert.equal(createRegistry(...CORE_OPS, local).get('relu').def.label.en, 'local relu');
});

for (const def of CORE_OPS) {
	test(`op contract: ${def.op}`, () => {
		assert.equal(typeof def.forward, 'function', 'forward');
		assert.equal(typeof def.inferShape, 'function', 'inferShape');
		assert.equal(typeof def.explain, 'function', 'explain');
		assert.equal(typeof def.vars, 'function', 'vars');
		assert.equal(typeof def.pytorch, 'function', 'pytorch');
		assert.ok(def.version >= 1);
		assert.ok(def.label.en && def.label.zh, 'label en + zh');
		assert.ok(def.formula && def.formula.includes('{{'), 'formula with live {{values}}');
		assert.ok(def.parity && def.parity.bar > 0 && def.parity.bar <= 1e-4, 'parity bar');
		assert.ok(def.zoom && def.category, 'zoom and category');
		assert.ok(def.unit && typeof def.unit.axes === 'function', 'unit axes');
		for (const [k, a] of Object.entries(def.attrs)) {
			assert.ok(['int', 'float', 'bool', 'string', 'int[]', 'float[]'].includes(a.type), `${k}: attr type`);
		}
		if (SPATIAL.has(def.op)) assert.equal(typeof def.receptive, 'function', 'spatial ops map receptive fields');
		if (ELEMENTWISE.has(def.op)) assert.deepEqual(def.receptive!({ y0: 1, x0: 2, y1: 3, x1: 4 }, {}), { y0: 1, x0: 2, y1: 3, x1: 4 });
		// narration: every field bilingual, same placeholder set, known formats, no literal multi-digit numbers (AK402)
		assert.ok(def.narration.what?.en && def.narration.what?.zh, 'narration.what en + zh');
		for (const [field, txt] of Object.entries(def.narration)) {
			if (!txt) continue;
			assert.ok(txt.en?.trim() && txt.zh?.trim(), `narration.${field}: en and zh`);
			const en = [...txt.en.matchAll(PLACEHOLDER)];
			const zh = [...txt.zh!.matchAll(PLACEHOLDER)];
			assert.deepEqual([...new Set(en.map((m) => m[1]))].sort(), [...new Set(zh.map((m) => m[1]))].sort(), `narration.${field}: en/zh placeholders differ (AK403)`);
			for (const m of [...en, ...zh]) if (m[2]) assert.ok(FMTS.has(m[2]), `narration.${field}: unknown format |${m[2]}`);
			for (const lang of ['en', 'zh'] as const) {
				const bare = txt[lang]!.replace(PLACEHOLDER, '');
				assert.ok(!/\d*\.\d+|\d{2,}/.test(bare), `narration.${field}.${lang}: literal number in "${bare}" (AK402)`);
			}
			const words = txt.en.split(/\s+/).filter(Boolean).length;
			assert.ok(words <= 90, `narration.${field}.en has ${words} words (≤ 90)`);
		}
		for (const term of def.terms ?? []) assert.ok(GLOSSARY_BY_ID[term], `glossary term '${term}' does not exist`);
		const layer: AtlasLayer = { id: 'demo', op: def.op, label: { en: 'demo' }, in: ['x'], outShape: [2, 3, 3], params: { kernel: 3, hidden: 4, nHead: 2, dHead: 3 } };
		const line = def.pytorch!(layer);
		assert.ok(typeof line === 'string' && line.length > 3, 'pytorch line');
	});
}

test('schema/ops.json is generated from CORE_OPS and up to date', () => {
	const file = fileURLToPath(new URL('../schema/ops.json', import.meta.url));
	const cur = readFileSync(file, 'utf8');
	assert.equal(cur, manifestText(), 'run: node src/lib/atlas/ops/manifest.ts --write');
	const m = buildManifest();
	assert.equal(m.length, 20);
	const conv = m.find((e) => e.op === 'conv2d')!;
	assert.deepEqual(Object.keys(conv).slice(0, 6), ['op', 'version', 'attrs', 'aliases', 'weights', 'subs']);
	assert.deepEqual(conv.attrs.kernel, { type: 'int', required: true, doc: 'kernel height and width (square)' });
	assert.deepEqual(m.find((e) => e.op === 'linear')!.aliases, { dense: {}, linear_tied: {} });
	assert.deepEqual(m.find((e) => e.op === 'lstm')!.subs.map((s) => s.name), ['i', 'f', 'g', 'o', 'c', 'h']);
	assert.ok(!cur.startsWith('﻿') && !cur.includes('\r\n'), 'LF, no BOM');
});
