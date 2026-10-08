// W1-D1 pure helpers of the atlas UI, in Node: kitspec, synth, scrub (coarse train tapes alternate 'step' and 'snapshot';
// scans), fallback, and the live-model input decision (liveinput.ts: the traced example comes from the lesson selection).
// Run from site/: node --test src/lib/atlas/ui/ui.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORE_OPS } from '../ops/index.ts';
import { fromParts, loadGraph } from '../graph/executor.ts';
import { normalizeSpec } from '../graph/legacy.ts';
import { fsFetch } from '../core/io.ts';
import { createRng } from '../core/rng.ts';
import { MLP, loadParams } from '../autograd/nn.ts';
import { input, noGrad, type Var } from '../autograd/var.ts';
import { crossEntropy } from '../autograd/loss.ts';
import { trainingProcess } from '../train/trainer.ts';
import { moons, sampleBatch, pointsTensors, type Points2D } from '../data/points2d.ts';
import { record } from '../process/record.ts';
import { scanTape, traceToTape } from '../process/adapters.ts';
import type { TrainOpt, TrainerDef } from '../types/index.ts';
import { layerNodes, numericShape, parityBadge, toKitSpec } from './kitspec.ts';
import { layerCountQuestion, synthManifest } from './synth.ts';
import { frameLabel, playFrames, scrubMarks } from './scrub.ts';
import { describeFallback, fallbackModel, shownTerms } from './fallback.ts';
import { liveModelInput, selectedRow } from './liveinput.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const spec = (slug: string) => normalizeSpec(JSON.parse(readFileSync(join(STATIC, 'models', slug, 'spec.json'), 'utf8')));

test('toKitSpec: numeric shapes, unmeasured parity left out, PredictQuestion compatible', () => {
	const gpt = spec('tinygpt');
	const k = toKitSpec(gpt, { T: 9 });
	assert.ok(k.layers!.every((l) => l.outShape!.every((d) => typeof d === 'number')), 'every outShape is numeric');
	assert.deepEqual(k.layers!.find((l) => l.id === 'tok_emb')!.outShape, [9, gpt.layers.find((l) => l.id === 'tok_emb')!.outShape[1] as number]);
	assert.equal(typeof k.parity?.max_abs_diff, 'number');
	const unmeasured = { ...gpt, parity: { ...gpt.parity, max_abs_diff: null, checked: null } };
	assert.equal(toKitSpec(unmeasured).parity, undefined, 'null max_abs_diff -> no parity -> the badge prints no tolerance');
	assert.equal(parityBadge(unmeasured).tolerance, undefined);
	assert.deepEqual(numericShape(['T', 76], gpt, {}), [64, 76], 'T falls back to dims.T.max');
	assert.deepEqual(layerNodes(gpt).find((l) => l.id === 'tok_emb')!.outShape, ['T', 76], 'overview labels keep symbolic dims');
	const cnn = toKitSpec(spec('tinycnn'));
	assert.equal(cnn.parity?.max_abs_diff, 2.86e-6);
	assert.deepEqual(cnn.input?.shape, [1, 28, 28]);
});

test('synthManifest: adapter, presets and output view from the spec; the predict question is answered by the spec', () => {
	const s = spec('tinycnn');
	const m = synthManifest(s, { dir: 'models/tinycnn', presets: 'presets.json' });
	assert.equal(m.input?.adapter, 'drawing28');
	assert.equal(m.input?.presets, 'presets.json');
	assert.equal(m.output?.view, 'kit.probbars');
	const q = layerCountQuestion(s);
	assert.equal(q.options[q.answer].en, String(s.layers.filter((l) => l.op !== 'input').length));
	assert.ok(q.question.zh && q.explain.zh && q.options.every((o) => o.zh));
});

// a coarse training tape: 'init', then 'snapshot' every snapshotEvery steps with 'step' frames in between
type Opt = TrainOpt & { hidden: number };
const PLAY: TrainerDef<Points2D, MLP, { x: Var; y: Int32Array }, Opt> = {
	id: 'play',
	title: { en: 'Play', zh: '演练' },
	defaults: { seed: 1, lr: 0.05, batch: 16, steps: 60, snapshotEvery: 10, optimizer: 'adam', hidden: 5 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, 2], 'tanh', rng),
	batch(data, _k, rng, opt) {
		const b = sampleBatch(data, opt.batch, rng);
		return { x: input(b.x), y: b.label };
	},
	loss: (m, b, rec) => ({ loss: crossEntropy(m.forward(b.x, rec), b.y) })
};

test('scrub: play ranges over coarse train tapes (step / snapshot alternate) and scans', async () => {
	const p = trainingProcess(PLAY);
	const data = moons(40, createRng(3), { noise: 0.1 });
	const tape = record(p, data, {}, { seed: 2, detail: 'coarse' });
	const phases = new Set(Array.from({ length: tape.length }, (_, i) => tape.frame(i).phase));
	assert.ok(phases.has('step') && phases.has('snapshot'), `phases ${[...phases].join(', ')}`);
	const all = playFrames(tape, 'first', 'last');
	assert.equal(all.length, tape.length);
	const snaps = playFrames(tape, 'first', 'last', ['snapshot']);
	assert.ok(snaps.length >= 6 && snaps.every((i) => tape.frame(i).phase === 'snapshot'), `${snaps.length} snapshot frames`);
	assert.ok(snaps.every((i) => (tape.state(i) as { params?: unknown }).params), 'every snapshot frame has weights');
	// alternating short segments are not listed one by one: evenly spaced marks labelled by training step
	const segs = tape.segments().length;
	const marks = scrubMarks(tape, all);
	assert.ok(segs > 12 && marks.length <= 8 && marks.length >= 2, `${segs} segments -> ${marks.length} marks`);
	assert.equal(tape.frame(all[0]).phase, 'init', 'a run starts with its init frame (no step number: labelled by phase)');
	assert.ok(marks.slice(1).every((m) => /^step \d+$/.test(m.label.en)), marks.map((m) => m.label.en).join(', '));
	assert.match(frameLabel(tape, snaps[1]).en, /step \d+/);
	// a scan: one long run -> marks named by output row
	const model = await loadGraph(fsFetch, join(STATIC, 'models', 'tinycnn'), CORE_OPS);
	const x = new Float32Array(784);
	for (let i = 250; i < 520; i += 3) x[i] = 1;
	const trace = model.forward({ input: { shape: [1, 28, 28], data: x } });
	const scan = scanTape(model, trace, 'conv1_1', [2]);
	const sf = playFrames(scan, 'first', 'last');
	assert.equal(sf.length, 26 * 26);
	const sm = scrubMarks(scan, sf, { unit: 'position' });
	assert.equal(sm[0].label.en, 'row 0', 'rows count from 0, as the TensorGrid readout');
	assert.equal(sm[sm.length - 1].label.en, 'row 25');
	// the generic stage reads the scan frame's arithmetic
	const rt = { sources: { s: { kind: 'scan', status: 'ready', provenance: 'live', tape: scan, trace } }, spec: model.spec } as never;
	const at = { index: 100, frame: scan.frame(100), state: scan.state(100), layer: 'conv1_1', unit: scan.state(100).explain };
	const fm = fallbackModel({ rt, source: 's', at });
	assert.equal(fm.kind, 'unit');
	assert.equal(fm.read?.id, 'input');
	assert.equal(shownTerms(fm.unit!).terms.length, 9);
	const d = describeFallback({ rt, source: 's', at });
	assert.match(d.en, /9 products/);
	assert.ok(d.zh && /乘积/.test(d.zh));
	// a trace frame
	const tt = traceToTape(model.spec, trace);
	const i = tt.find({ layer: 'softmax' });
	const fm2 = fallbackModel({ rt: { sources: { fwd: { kind: 'trace', status: 'ready', provenance: 'live', tape: tt, trace } }, spec: model.spec } as never, source: 'fwd', at: { index: i, frame: tt.frame(i), state: tt.state(i), layer: 'softmax' } });
	assert.equal(fm2.kind, 'tensor');
});

test('live-model input comes from the lesson selection: the selected point, traced on the live weights, equals the Module', () => {
	const p = trainingProcess(PLAY);
	const data = moons(30, createRng(5), { noise: 0.1 });
	const tape = record(p, data, { steps: 30 }, { seed: 4, detail: 'coarse' });
	const state = tape.state(tape.length - 1) as { params: Record<string, { shape: number[]; data: Float64Array }> };
	const { spec: s2, weights } = (p as unknown as { liveSpec: (s: unknown, o?: unknown) => { spec: never; weights: never } }).liveSpec(state, { steps: 30 });
	const model = fromParts(s2, weights, CORE_OPS);
	const pts = pointsTensors(data);
	const pre = { tensors: { points: pts.x }, steps: [], empty: false };
	for (const r of [0, 7, 29, 99]) {
		const sel = { p: r };
		const li = liveModelInput(model.spec, pre as never, sel);
		assert.deepEqual(li.missing, []);
		const row = Math.min(29, r);
		assert.equal(li.row, row, 'clamped row');
		const trace = model.forward(li.feed);
		const out = trace.t[trace.order[trace.order.length - 1]].data;
		const m = PLAY.model(createRng(0), { ...PLAY.defaults, steps: 30 });
		loadParams(m, state.params);
		const xy = Float64Array.from(Float32Array.from(pts.x.data.subarray(row * 2, row * 2 + 2)));
		const ref = noGrad(() => m.forward(input({ shape: [1, 2], data: xy }))).value.data;
		for (let k = 0; k < ref.length; k++) assert.ok(Math.abs(out[k] - ref[k]) <= 1e-6 * Math.max(1, Math.abs(ref[k])), `row ${row} logit ${k}: ${out[k]} vs ${ref[k]}`);
	}
	// an adapter that already gives one example of the declared shape is used as is
	const one = { tensors: { input: { shape: [2], data: Float32Array.from([0.3, -0.2]) } }, steps: [], empty: false };
	assert.equal(liveModelInput(model.spec, one as never, {}).row, null);
	assert.equal(selectedRow({ unit: [4, 1] }), 4);
	assert.equal(selectedRow({}), null);
});
