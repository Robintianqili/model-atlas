// The runtime layer: createRuntime source handling, immutability (KIT rule 5), localRunner, precomputed tapes,
// live-model, fakeRuntime, withFixtures.
// Run from site/:  node --test src/lib/atlas/runtime/runtime.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { ProcessDef } from '../types/process.ts';
import type { PackageModules } from '../types/runtime.ts';
import type { Tensor } from '../types/core.ts';
import { createRuntime } from './runtime.ts';
import { localRunner } from './local.ts';
import { fakeRuntime } from './fake.ts';
import { effectiveProvenance, stepAt } from './sources.ts';
import { clampIndex, pickUnit, resolveCursor } from './selection.ts';
import { compileLesson } from '../lesson/compile.ts';
import { withFixtures } from '../lesson/fixtures.ts';
import { lesson6 } from '../lesson/fixtures/lesson6.ts';
import { memoryFetch, SEQ_FILES, SEQ_SPEC, seqFetch, seqManifest, seqPackage, unroll, type CountState } from '../lesson/fixtures/seq.ts';
import { record } from '../process/record.ts';
import { saveTape } from '../process/io.ts';

test('createRuntime: loads spec, facts, presets; trace on the first preset; facts and built-ins resolve', async () => {
	const rt = await createRuntime(seqPackage(lesson6), { fetch: seqFetch, base: '' });
	assert.deepEqual(rt.warnings, []);
	assert.equal(rt.spec?.slug, 'fixture-seq');
	assert.equal(rt.input, 'hello world');
	assert.equal(rt.sources.fwd.status, 'ready');
	assert.equal(rt.sources.fwd.tape!.length, 11, 'one frame per layer');
	assert.equal(rt.sources.unroll.status, 'idle', 'process sources start on demand');
	assert.equal(rt.num('fact:test_accuracy'), 0.9681);
	assert.equal(rt.fact('layer.B0.lin.shape'), '[11×4]');
	assert.ok(Number.isFinite(rt.num('top.p')));
	assert.ok(Number.isNaN(rt.num('fact:nope')));
	await assert.rejects(rt.ensure('nope', 'first'), /unknown source 'nope'/);
});

test('ensure runs a process only as far as the cursor needs; setInput replaces objects and restarts the process', async () => {
	const rt = await createRuntime(seqPackage(lesson6), { fetch: seqFetch, base: '', input: 'abcabcabca' });
	let changes = 0;
	rt.onChange(() => changes++);
	const i = await rt.ensure('unroll', { phase: 'read', step: 3 });
	assert.equal(i, 3);
	const tape = rt.sources.unroll.tape!;
	assert.ok(tape.length >= 4 && tape.length < 10, `recorded ${tape.length} of 10 frames`);
	assert.equal(await rt.ensure('unroll', 'last'), 9);
	assert.equal((tape.state(9) as CountState).count.data[0], 4);
	const before = { sources: rt.sources, fwd: rt.sources.fwd, sel: rt.sel, pre: rt.pre };
	await new Promise((r) => setTimeout(r, 5));
	changes = 0;
	rt.setInput('zz');
	await Promise.resolve();
	assert.equal(changes, 1, 'one notification per burst');
	assert.notEqual(rt.sources, before.sources, 'sources record replaced');
	assert.notEqual(rt.sources.fwd, before.fwd, 'trace source replaced');
	assert.notEqual(rt.pre, before.pre);
	assert.equal(rt.sources.unroll.status, 'idle');
	assert.equal(await rt.ensure('unroll', 'last'), 1);
	assert.equal((rt.sources.unroll.tape!.state(1) as CountState).count.data[25], 2);
	// clamping: a cursor past the end comes back clamped, and find() says it was not recorded
	assert.equal(await rt.ensure('unroll', 40), 1);
	assert.deepEqual(resolveCursor(rt.sources.unroll.tape, 40), { index: 1, found: false });
	rt.close();
});

test('implicit scan sources follow the lesson selection; stepAt builds At with the unit explanation', async () => {
	const pkg = seqPackage(lesson6);
	const rt = await createRuntime(pkg, { fetch: seqFetch, base: '', input: 'quiet' });
	const steps = compileLesson(lesson6, { spec: SEQ_SPEC, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes }).steps;
	const scanStep = steps.find((s) => s.source === 'scan:B0.lin')!;
	const pos = await stepAt(rt, scanStep);
	assert.equal(pos.found, true);
	const tape = rt.sources['scan:B0.lin'].tape!;
	assert.equal(tape.length, 4, 'one row of 4 outputs: rows mode fixes the selected row');
	const row = rt.sel.unit![0];
	assert.deepEqual((tape.state(0) as { unit: number[] }).unit, [row, 0]);
	assert.ok(pos.at.unit && Math.abs(pos.at.unit.result - pos.at.unit.traced) < 1e-6);
	// a different selection rebuilds the scan
	rt.sel = { layer: 'B0.lin', unit: [(row + 1) % 5, 0] };
	await rt.ensure('scan:B0.lin', 'first');
	assert.deepEqual((rt.sources['scan:B0.lin'].tape!.state(0) as { unit: number[] }).unit, [(row + 1) % 5, 0]);
	// trace steps: at.unit only when the selection is on the frame's layer
	const squash = steps.find((s) => s.id === 'squash')!;
	const p2 = await stepAt(rt, squash);
	assert.equal(p2.at.layer, 'B0.act');
	assert.equal(p2.at.unit?.kind, 'elementwise');
	assert.equal(effectiveProvenance(rt, squash).mode, 'live');
	rt.close();
});

test('precomputed tapes: a tape source and preferPrecomputed load atlas-tape/1 files with provenance precomputed', async () => {
	const t = record(unroll, 'precomputed text', {}, { seed: 1 });
	const { header, blob } = saveTape(t);
	const files = { ...SEQ_FILES, 'models/fixture-seq/tapes/count.tape.json': JSON.parse(JSON.stringify(header)), 'models/fixture-seq/tapes/count.tape.bin': blob };
	const manifest = { ...seqManifest, sources: { ...seqManifest.sources, saved: { kind: 'tape' as const, file: 'tapes/count' }, unroll: { kind: 'process' as const, process: 'unroll', precomputed: 'tapes/count' } } };
	const pkg: PackageModules = { ...seqPackage(lesson6), manifest };
	const rt = await createRuntime(pkg, { fetch: memoryFetch(files), base: '', preferPrecomputed: true });
	assert.equal(await rt.ensure('saved', 'last'), t.length - 1);
	assert.equal(rt.sources.saved.provenance, 'precomputed');
	await rt.ensure('unroll', 'last');
	assert.equal(rt.sources.unroll.provenance, 'precomputed');
	assert.equal(effectiveProvenance(rt, { provenance: { mode: 'live' }, source: 'unroll' }).mode, 'precomputed', 'a live step on a precomputed source shows precomputed');
	assert.equal((rt.sources.unroll.tape!.state(t.length - 1) as CountState).count.data[4], (t.state(t.length - 1) as CountState).count.data[4]);
});

test('live-model: a trace on the train source weights through ProcessDef.liveSpec; a clear error without it', async () => {
	type S = { step: number; w: number };
	const trainer: ProcessDef<unknown, { steps: number }, S> & { liveSpec?: (s: S) => { spec: typeof SEQ_SPEC; weights: Record<string, Tensor> } } = {
		id: 'train:toy',
		kind: 'training',
		title: { en: 'toy', zh: '玩具' },
		phases: { snapshot: { label: { en: 'snapshot', zh: '快照' }, kind: 'update' } },
		schema: { step: { kind: 'int', label: { en: 'step', zh: '步' } }, w: { kind: 'scalar', label: { en: 'w', zh: 'w' } } },
		defaults: { steps: 3 },
		*run(_i, ctx) {
			for (let s = 0; s < ctx.opt.steps; s++) yield { phase: 'snapshot', step: s, state: { step: s, w: s * 0.1 }, metrics: { loss: 1 / (s + 1) } };
		}
	};
	const base = seqPackage(lesson6);
	const weights = new Float32Array(SEQ_FILES['models/fixture-seq/weights.bin'] as ArrayBuffer);
	trainer.liveSpec = (s) => {
		const W: Record<string, Tensor> = {};
		for (const e of SEQ_SPEC.tensors) W[e.name] = { shape: e.shape, data: weights.slice(e.offset, e.offset + e.length).map((v) => v * (1 + s.w)) };
		return { spec: SEQ_SPEC, weights: W };
	};
	const manifest = { ...seqManifest, kinds: ['trace' as const, 'train' as const], sources: { fwd: { kind: 'trace' as const }, train: { kind: 'train' as const, trainer: 'toy' }, live: { kind: 'live-model' as const, of: 'train' } } };
	const pkg: PackageModules = { ...base, manifest, processes: { ...base.processes, 'train:toy': trainer } };
	const rt = await createRuntime(pkg, { fetch: seqFetch, base: '', runner: localRunner });
	await rt.ensure('train', 'last');
	assert.equal(rt.num('train.step'), 2);
	assert.equal(rt.num('train.loss'), 1 / 3);
	const i = await rt.ensure('live', { layer: 'prob' });
	assert.equal(rt.sources.live.status, 'ready');
	assert.ok(i > 0);
	const p = rt.sources.live.trace!.t.prob.data;
	assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 1e-6);
	// a deep link straight to the live-model step (afternoon test): the train source has not run yet. The runtime runs it
	// to its end first, so the step shows the same numbers as stepping in through the train step.
	const deep = await createRuntime(pkg, { fetch: seqFetch, base: '', runner: localRunner });
	assert.equal(deep.sources.train.status, 'idle', 'nothing has run yet');
	const pos = await stepAt(deep, { source: 'live', at: () => ({ layer: 'prob' }), select: () => ({}) });
	assert.equal(pos.error, undefined, `deep link: ${pos.error}`);
	assert.ok(pos.found && pos.index > 0);
	assert.equal(deep.sources.live.status, 'ready');
	assert.equal(deep.num('train.step'), 2, 'the train source ran to its end');
	assert.deepEqual(Array.from(deep.sources.live.trace!.t.prob.data), Array.from(p), 'same numbers as stepping in');
	deep.close();
	delete trainer.liveSpec;
	rt.setInput('again');
	await rt.ensure('live', 'first');
	assert.match(rt.sources.live.error ?? '', /liveSpec/);
});

test('fakeRuntime and selection helpers', async () => {
	const tape = record(unroll, 'abc', {}, { seed: 1 });
	const rt = fakeRuntime({ tapes: { c: tape }, facts: { 'opt.gamma': 0.9 } });
	assert.equal(rt.num('opt.gamma'), 0.9);
	assert.equal(await rt.ensure('c', 'last'), 2);
	assert.equal(await rt.ensure('c', 99), 2);
	assert.equal(rt.sources.c.provenance, 'illustrative');
	await assert.rejects(rt.ensure('x', 0));
	assert.equal(clampIndex(tape, -5), 0);
	assert.equal(clampIndex(null, 3), -1);
	const t: Tensor = { shape: [2, 2, 3], data: Float32Array.from([1, 2, 3, 4, 5, 6, -9, 0, 0, 0, 7, 0]) };
	assert.deepEqual(pickUnit(t, 'argmax'), [1, 1, 1]);
	assert.deepEqual(pickUnit(t, 'argmaxAbs'), [1, 0, 0]);
	assert.deepEqual(pickUnit(t, 'argmax', { ch: 0 }), [0, 1, 2]);
	assert.deepEqual(pickUnit(t, 'center'), [1, 1, 1]);
	assert.deepEqual(pickUnit(t, { unit: [5, 0, 1] }), [1, 0, 1]);
});

test('withFixtures runs a claim on every fixture and preset, and names the fixtures that fail (AK408)', async () => {
	const pkg = seqPackage(lesson6);
	const { ran } = await withFixtures(pkg, (rt) => {
		if (rt.pre?.empty) return;
		const p = rt.sources.fwd.trace!.t.prob.data;
		assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 1e-6);
	}, { fetch: seqFetch, base: '' });
	assert.deepEqual(ran, ['mixed', 'edge:empty', 'edge:one', 'edge:long', 'preset:hello', 'preset:zz']);
	await assert.rejects(
		withFixtures(pkg, (rt) => assert.ok(String(rt.input).length > 3, 'claim: inputs are long'), { fetch: seqFetch, base: '' }),
		/AK408 claim failed on 2 of 6 fixtures:[\s\S]*edge:empty[\s\S]*edge:one/
	);
});
