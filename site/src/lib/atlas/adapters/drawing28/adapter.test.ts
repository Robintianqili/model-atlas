// drawing28 (W1-D1): the same 784 numbers as the TinyCNN page, edge fixtures, presets, and the TinyCNN auto lesson run on
// the real adapter in Node (every step × every fixture × en/zh, lint 0 errors), as the Lesson Lab runs it in the browser.
// Run from site/: node --test src/lib/atlas/adapters/drawing28/adapter.test.ts
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
import { render } from '../../lesson/narrate.ts';
import { createRuntime } from '../../runtime/runtime.ts';
import { stepAt } from '../../runtime/sources.ts';
import type { PackageModules } from '../../types/index.ts';
import { synthManifest } from '../../ui/synth.ts';
import { decodePreset, preprocessStrokes } from '../../../models/tinycnn.ts';
import { drawing28, drawingInput, simplifyStroke } from './adapter.ts';

const SITE = fileURLToPath(new URL('../../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const presetsFile = JSON.parse(readFileSync(join(STATIC, 'models', 'tinycnn', 'presets.json'), 'utf8'));

test('drawing28: strokes, presets and ink grids give TinyCNN’s own 784 numbers', () => {
	const strokes = [{ points: [{ x: 0.2, y: 0.2 }, { x: 0.21, y: 0.2005 }, { x: 0.5, y: 0.6 }, { x: 0.8, y: 0.3 }] }, { points: [{ x: 0.3, y: 0.8 }, { x: 0.7, y: 0.8 }] }];
	const a = drawing28.preprocess({ strokes }, null, {});
	const b = preprocessStrokes(strokes.map((s) => simplifyStroke(s.points)));
	assert.deepEqual(Array.from(a.tensors.input.data), Array.from(b.input), 'same as the page path (simplify, then preprocessStrokes)');
	assert.deepEqual(a.tensors.input.shape, [1, 28, 28]);
	const p0 = presetsFile.classes[0].items[0].b64;
	assert.deepEqual(Array.from(drawing28.preprocess({ b64: p0 }, null, {}).tensors.input.data), Array.from(decodePreset(p0)));
	const ink = Array.from({ length: 784 }, (_, i) => (i % 29 === 0 ? 1.5 : i % 31 === 0 ? -1 : 0));
	const c = drawingInput(ink);
	assert.equal(c.source, 'ink');
	assert.ok(c.input.every((v) => v >= 0 && v <= 1), 'ink is clamped to [0, 1]');
	assert.throws(() => drawing28.preprocess({ ink: [1, 2, 3] }, null, {}), /784/);
});

test('drawing28: edge fixtures (blank is empty with a note; dot and small are not; no NaN)', () => {
	const fx = Object.fromEntries(drawing28.edgeFixtures().map((f) => [f.id, drawing28.preprocess(f.raw, null, {})]));
	assert.deepEqual(Object.keys(fx), ['blank', 'dot', 'small']);
	assert.equal(fx.blank.empty, true);
	assert.ok(fx.blank.notes?.[0]?.zh, 'the empty note is bilingual');
	for (const id of ['dot', 'small']) {
		assert.equal(fx[id].empty, false, id);
		assert.ok(Array.from(fx[id].tensors.input.data).every(Number.isFinite), `${id}: finite`);
	}
	assert.equal(drawing28.preprocess(null, null, {}).empty, true);
});

test('drawing28: presets are one drawing per class plus the held-out classes, labelled in en and zh', () => {
	const ps = drawing28.presets!({ 'presets.json': presetsFile });
	assert.equal(ps.length, presetsFile.classes.length + presetsFile.heldOut.length);
	assert.ok(ps.every((p) => p.label.en && p.label.zh));
	assert.ok(ps.some((p) => p.id.startsWith('heldout-')));
});

const tinycnnPkg = (): PackageModules => {
	const spec = normalizeSpec(JSON.parse(readFileSync(join(STATIC, 'models', 'tinycnn', 'spec.json'), 'utf8')));
	return { slug: 'tinycnn', manifest: synthManifest(spec, { dir: 'models/tinycnn', presets: 'presets.json' }), lesson: null, ops: CORE_OPS, processes: {}, adapters: { drawing28 } };
};

test('TinyCNN auto lesson on the real drawing28 adapter: every class preset is classified as its class; lint has 0 errors', async () => {
	const pkg = tinycnnPkg();
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC });
	assert.deepEqual(rt.warnings, []);
	assert.equal(rt.presets.length, 12);
	// presets.json says the base-class presets are drawings the model gets right with p > 0.9
	for (const p of rt.presets.filter((x) => !x.id.startsWith('heldout-'))) {
		rt.setInput(p.raw);
		const top = rt.fact('top.label') as { en: string };
		assert.equal(top.en, p.label.en, `preset ${p.id}`);
		assert.ok(rt.num('top.p') > 0.9, `preset ${p.id}: p ${rt.num('top.p')}`);
	}
	const c = compileLesson(null, { spec: rt.spec, ops: CORE_OPS, manifest: pkg.manifest, adapters: pkg.adapters });
	assert.ok(c.steps.length >= 5);
	const fixtures = fixtureCases(pkg, rt.presets);
	assert.equal(fixtures.length, 15);
	const issues = await lintLesson(c, (fx) => createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: fx.raw }), fixtures);
	const errors = issues.filter((i) => i.level === 'error');
	const warns = [...new Set(issues.filter((i) => i.level === 'warn').map((i) => `${i.code}${i.step ? ' ' + i.step : ''}`))];
	console.log(`ℹ TinyCNN auto lesson (drawing28): ${c.steps.map((s) => s.id).join(', ')}; ${fixtures.length} fixtures; ${errors.length} errors; warnings: ${warns.join('; ')}`);
	assert.deepEqual(errors.map((e) => `${e.code} ${e.step ?? ''} ${e.fixture ?? ''} ${e.lang ?? ''} ${e.message}`), []);
	// the input step's narration on a preset quotes the real tensor size
	rt.setInput(rt.presets[0].raw);
	const pos = await stepAt(rt, c.steps[0]);
	const en = render(c.steps[0].src.narration as never, rt, pos.at, 'en').text;
	const zh = render(c.steps[0].src.narration as never, rt, pos.at, 'zh').text;
	assert.match(en, /784 numbers in a \[1×28×28\] grid/);
	assert.match(zh, /784 个数/);
	rt.close();
});
