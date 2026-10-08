// W1-C acceptance 2 (bpeTape reproduces trainBPE on pipeline/parity/bpe_cases.json) plus the other tape adapters and the
// process library. The models/*.ts imports here are test-only parity references (the pure files never import them).
// Run from site/:  node --test src/lib/atlas/process/adapters.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { trainBPE, BPETokenizer, type TokenizerJSON } from '../../models/bpe.ts';
import { load as loadTinyGPT, samplingDistribution as gptDistribution } from '../../models/tinygpt.ts';
import { bpeTape, generateTape, scanUnits, traceToTape } from './adapters.ts';
import { record } from './record.ts';
import { checkDeterminism } from './determinism.ts';
import { samplingDistribution, samplingProcess, type SamplingState } from './library/sampling.ts';
import { eulerProcess, type EulerState } from './library/euler.ts';
import { rolloutProcess, type RolloutState } from './library/rollout.ts';
import { ATLAS_ROOT, ATLAS_SITE, fsFetch, readJson } from './fixtures/paths.ts';
import { createRng } from '../runtime/seams.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { Tensor, Trace } from '../types/core.ts';

type BpeCases = { train: { text: string; numMerges: number; merges: [number, number][]; counts: number[] }[] };

test('acceptance 2: bpeTape reproduces trainBPE on pipeline/parity/bpe_cases.json (merges and counts, Python reference)', (tc) => {
	const cases = readJson<BpeCases>(join(ATLAS_ROOT, 'pipeline', 'parity', 'bpe_cases.json'));
	const tok = readJson<TokenizerJSON>(join(ATLAS_SITE, 'static', 'models', 'tinygpt', 'tokenizer.json'));
	assert.equal(cases.train.length, 3);
	let merges = 0;
	for (const c of cases.train) {
		const r = trainBPE(c.text, c.numMerges, tok.pretokenize.pattern);
		const t = bpeTape(r);
		assert.equal(t.length, r.steps.length + 1);
		assert.equal(t.frame(0).phase, 'start');
		assert.equal(t.frame(0).metrics!.tokenCount, r.initialTokenCount);
		// frame by frame: the merge, its count and the token count after it
		const fromTape = Array.from({ length: t.length - 1 }, (_, k) => t.frame(k + 1));
		assert.deepEqual(fromTape.map((f) => (f.state as { pair: number[] }).pair), c.merges, 'merges equal the Python trainer');
		assert.deepEqual(fromTape.map((f) => (f.state as { count: number }).count), c.counts, 'counts equal the Python trainer');
		assert.deepEqual(fromTape.map((f) => f.metrics!.tokenCount), r.steps.map((s) => s.tokenCount));
		assert.deepEqual(fromTape.map((f) => f.step), r.steps.map((s) => s.rank));
		// state(i).mergeTable (tracked by changes) holds exactly the merges learned so far
		for (const i of [0, 1, Math.floor(t.length / 2), t.length - 1]) {
			const table = (t.state(i) as { mergeTable: Tensor }).mergeTable;
			const rows = Array.from({ length: i }, (_, k) => [table.data[2 * k], table.data[2 * k + 1]]);
			assert.deepEqual(rows, c.merges.slice(0, i), `mergeTable at ${i}`);
			if (i < r.steps.length) assert.equal(table.data[2 * i], -1);
		}
		// the merges from the tape tokenize like trainBPE's merges
		const fromTable = Array.from({ length: r.steps.length }, (_, k) => [(t.state(t.length - 1) as { mergeTable: Tensor }).mergeTable.data[2 * k], (t.state(t.length - 1) as { mergeTable: Tensor }).mergeTable.data[2 * k + 1]] as [number, number]);
		const a = BPETokenizer.fromMerges(fromTable, tok.pretokenize.pattern);
		const b = BPETokenizer.fromMerges(r.merges, tok.pretokenize.pattern);
		assert.deepEqual(a.encode(c.text.slice(0, 400)), b.encode(c.text.slice(0, 400)));
		merges += r.steps.length;
	}
	assert.equal(merges, cases.train.reduce((a, c) => a + c.merges.length, 0));
	tc.diagnostic(`${cases.train.length} cases, ${merges} merges: tape == trainBPE == Python`);
});

test('library/sampling equals TinyGPT generate() token for token (sampled, top-k, top-p and greedy)', async () => {
	const dir = join(ATLAS_SITE, 'static', 'models', 'tinygpt');
	const model = await loadTinyGPT(fsFetch(dir) as unknown as Parameters<typeof loadTinyGPT>[0], '');
	const tok = new BPETokenizer(readJson<TokenizerJSON>(join(dir, 'tokenizer.json')));
	const prompt = tok.encode('Once upon a time, there was a little');
	const proc = samplingProcess({ id: 'sample', logits: (ids) => model.nextTokenProbs(ids).logits, context: model.context });
	for (const o of [
		{ seed: 42, temperature: 0.8, topK: 20, topP: 1 },
		{ seed: 7, temperature: 1.2, topK: 0, topP: 0.9 },
		{ seed: 1, temperature: 0, topK: 0, topP: 1 }
	]) {
		const g = model.generate(prompt, { seed: o.seed, temperature: o.temperature, topK: o.topK, topP: o.topP, maxNew: 10 });
		const t = record(proc, { ids: prompt }, { maxNew: 10, temperature: o.temperature, topK: o.topK, topP: o.topP, stopId: model.vocabSize - 1 }, { seed: o.seed });
		const samples = Array.from({ length: t.length }, (_, i) => t.frame(i)).filter((f) => f.phase === 'sample');
		assert.deepEqual(samples.map((f) => (f.state as SamplingState).token), g.newIds, `tokens ${JSON.stringify(o)}`);
		assert.deepEqual(samples.map((f) => (f.state as SamplingState).u), g.steps.map((s) => s.u ?? NaN), 'same random numbers');
		assert.deepEqual(samples.map((f) => (f.state as SamplingState).p), g.steps.map((s) => s.p), 'same sampling probabilities');
		samples.forEach((f, k) => assert.ok(Math.abs((f.state as SamplingState).pModel! - g.steps[k].pModel) < 1e-6));
		// generateTape wraps the generate() result without changing it
		const gt = generateTape(g);
		assert.equal(gt.length, g.steps.length + 1);
		assert.deepEqual((gt.state(gt.length - 1) as { ids: number[] }).ids, g.ids);
		assert.deepEqual(Array.from({ length: g.steps.length }, (_, k) => (gt.frame(k + 1).state as { token: number }).token), g.newIds);
	}
	// samplingDistribution is the same function
	const rng = createRng(5);
	for (let k = 0; k < 20; k++) {
		const logits = Float32Array.from({ length: 50 }, () => rng.normal() * 3);
		const args = [[1, 0, 1], [0.7, 5, 1], [1.3, 0, 0.8], [0.5, 10, 0.5]][k % 4] as [number, number, number];
		assert.deepEqual(Array.from(samplingDistribution(logits, ...args)), Array.from(gptDistribution(logits, ...args)));
	}
	// fine detail adds logits and distribution frames and stays deterministic
	const det = checkDeterminism(proc, { ids: prompt.slice(0, 5) }, { maxNew: 4, temperature: 0.9 }, { seed: 3, detail: 'fine' });
	assert.equal(det.ok, true, det.problems.join('\n'));
	assert.equal(det.frames, 1 + 4 * 3);
});

test('traceToTape: one frame per trace id, sub-steps after their layer, cursor { layer } finds the layer', () => {
	const spec = {
		slug: 'demo',
		version: 'v',
		layers: [
			{ id: 'input', op: 'input', in: [], outShape: [2], label: { en: 'Input', zh: '输入' } },
			{ id: 'L0.attn', op: 'x', in: ['input'], outShape: [2], label: { en: 'Attention', zh: '注意力' } },
			{ id: 'out', op: 'softmax', in: ['L0.attn'], outShape: [2], label: { en: 'Out', zh: '输出' } }
		]
	} as unknown as AtlasSpec;
	const ten = (v: number[]): Tensor => ({ shape: [v.length], data: Float32Array.from(v) });
	const trace: Trace = { order: ['input', 'L0.attn', 'L0.attn#h0', 'L0.attn#h1', 'out'], t: { input: ten([1, 2]), 'L0.attn': ten([3, -4]), 'L0.attn#h0': ten([5, 6]), 'L0.attn#h1': ten([7, 8]), out: ten([0.25, 0.75]) } };
	const t = traceToTape(spec, trace);
	assert.deepEqual(Array.from({ length: t.length }, (_, i) => t.frame(i).phase), ['input', 'layer', 'sub', 'sub', 'output']);
	assert.equal(t.find({ layer: 'L0.attn' }), 1);
	assert.equal(t.find({ layer: 'L0.attn', last: true }), 3);
	assert.equal(t.frame(3).sub, 1);
	assert.equal((t.state(2) as { id: string }).id, 'L0.attn#h0');
	assert.equal(t.frame(1).metrics!.min, -4);
	assert.equal(t.state(4).out, trace.t.out, 'tensors are shared, not copied');
	assert.deepEqual(scanUnits([2, 3, 4], [1]).length, 12);
	assert.deepEqual(scanUnits([2, 3, 4], [1, -1, 2]), [[1, 0, 2], [1, 1, 2], [1, 2, 2]]);
	assert.equal(scanUnits([4, 26, 26], [0], { stride: 2 }).length, 169);
});

test('library/euler: x_K = (1 − h)^K x0 for v = −x; fine detail adds velocity frames', () => {
	const proc = eulerProcess({ id: 'euler', field: (x) => x.map((v) => -v) });
	const points: Tensor = { shape: [3, 2], data: Float64Array.from([1, 2, -1, 0.5, 3, -2]) };
	const t = record(proc, { points }, { steps: 10 }, { seed: 1 });
	assert.equal(t.length, 11);
	const end = t.state(10) as EulerState;
	let x = Array.from(points.data);
	for (let k = 0; k < 10; k++) x = x.map((v) => v + 0.1 * -v);
	assert.deepEqual(Array.from(end.x.data), x);
	assert.ok(Math.abs(end.t - 1) < 1e-12);
	const fine = record(proc, { points }, { steps: 10 }, { seed: 1, detail: 'fine' });
	assert.equal(fine.length, 21);
	assert.equal(fine.frame(1).phase, 'velocity');
	assert.equal(checkDeterminism(proc, { points }, { steps: 30 }, { seed: 2, detail: 'fine' }).ok, true);
});

test('library/rollout: predictions feed back; path is tracked by changes; deterministic', () => {
	const series = Array.from({ length: 40 }, (_, i) => Math.sin(i / 3));
	const proc = rolloutProcess({ id: 'rollout', predict: (w) => 2 * w[w.length - 1] - w[w.length - 2] });
	const t = record(proc, { series }, { window: 6, horizon: 8, start: 20 }, { seed: 1, keyEvery: 4 });
	assert.equal(t.length, 9);
	const s = t.state(8) as RolloutState;
	// recompute by hand
	const hist = series.slice(0, 20);
	const preds: number[] = [];
	for (let h = 0; h < 8; h++) {
		const p = 2 * hist[hist.length - 1] - hist[hist.length - 2];
		preds.push(p);
		hist.push(p);
	}
	assert.deepEqual(Array.from(s.path.data), preds);
	assert.equal(s.truth, series[27]);
	assert.ok(Number.isNaN((t.state(3) as RolloutState).path.data[5]));
	assert.ok(t.metric('absErr')[8] > t.metric('absErr')[1], 'errors compound');
	const det = checkDeterminism(proc, { series }, { window: 6, horizon: 30, start: 8 }, { seed: 1 });
	assert.equal(det.ok, true, det.problems.join('\n'));
});
