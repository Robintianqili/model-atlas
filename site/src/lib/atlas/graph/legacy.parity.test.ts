// W1-A acceptance 1–3 (design §8.1): the kit executor on the two shipped specs, read through legacy.ts.
//   1. TinyCNN: bit for bit equal to tinycnn.ts on the 3 reference inputs, the 60 class presets and the 12 held-out
//      presets; within 1e-4 of the PyTorch reference.
//   2. TinyGPT: within 1e-6 relative of tinygpt.ts at T = 1, 9, 22, 37, 64 (T = 1 and 64 are tokenized fixed strings);
//      within 1e-4 of the shipped PyTorch reference (T = 9, 22, 37). T = 64 against PyTorch is reported (not gated) when
//      the git-ignored pipeline/parity/verify_tinygpt/verify_ref.json exists.
//   3. explain() equals the trace (≤ 1e-5) for every unit of every TinyCNN layer and 2,000 sampled TinyGPT units.
// This test imports the legacy runtimes in src/lib/models/ only to compare against them (tests are exempt from the
// atlas dependency rule; no runtime file under atlas/ imports models/).
// Run from site/: node --test src/lib/atlas/graph/legacy.parity.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromParts } from './executor.ts';
import { checkExplain, pickUnit } from './explain.ts';
import { readReference } from './refs.ts';
import { CORE_OPS } from '../ops/index.ts';
import { findRepoRoot } from '../process/fixtures/paths.ts';
import { bitEqual, maxDiff } from '../core/compare.ts';
import { f32FromB64, fsFetch } from '../core/io.ts';
import { intTensor } from '../core/tensor.ts';
import * as tinycnn from '../../models/tinycnn.ts';
import * as tinygpt from '../../models/tinygpt.ts';
import { BPETokenizer, type TokenizerJSON } from '../../models/bpe.ts';
import type { GraphModel, Trace } from '../types/index.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const MODELS = join(SITE, 'static', 'models');
const ab = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
const json = (p: string) => JSON.parse(readFileSync(p, 'utf8'));

/** ATLAS_ROOT: $ATLAS_ROOT, else a lane's lane.json root (K4: a lane outside the repo), else the nearest ancestor of the
 *  site folder with CONTRACTS.md and pipeline/. */
function atlasRoot(): string | null {
	return findRepoRoot(SITE);
}

// ------------------------------------------------------------------ TinyCNN
const cnnSpec = json(join(MODELS, 'tinycnn', 'spec.json'));
const cnnW = ab(readFileSync(join(MODELS, 'tinycnn', 'weights.bin')));
const cnn: GraphModel = fromParts(cnnSpec, cnnW, CORE_OPS);
const cnnLegacy = tinycnn.fromParts(cnnSpec, cnnW);

function cnnInputs(): { label: string; x: Float32Array }[] {
	const out: { label: string; x: Float32Array }[] = [];
	for (const r of json(join(MODELS, 'tinycnn', 'reference.json')).inputs) out.push({ label: `ref ${r.label}`, x: Float32Array.from(r.input) });
	const presets = json(join(MODELS, 'tinycnn', 'presets.json'));
	for (const c of presets.classes) c.items.forEach((it: { b64: string }, i: number) => out.push({ label: `preset ${c.id}#${i}`, x: tinycnn.decodePreset(it.b64) }));
	for (const c of presets.heldOut) c.items.forEach((it: { b64: string }, i: number) => out.push({ label: `held-out ${c.id}#${i}`, x: tinycnn.decodePreset(it.b64) }));
	return out;
}

test('acceptance 1: TinyCNN executor == tinycnn.ts bit for bit (3 references + 60 presets + 12 held-out); ≤ 1e-4 vs PyTorch', async (t) => {
	const inputs = cnnInputs();
	const counts = { ref: 0, preset: 0, 'held-out': 0 } as Record<string, number>;
	let tensors = 0;
	for (const { label, x } of inputs) {
		counts[label.split(' ')[0]]++;
		const a = cnn.forward({ shape: [1, 28, 28], data: x });
		const b = cnnLegacy.forward(x);
		assert.deepEqual(a.order, b.order, `${label}: trace order`);
		for (const id of b.order) {
			assert.deepEqual(a.t[id].shape, b.t[id].shape, `${label} ${id}: shape`);
			const eq = bitEqual(a.t[id].data, b.t[id].data);
			assert.ok(eq.equal, `${label} ${id}: differs at ${eq.at}: ${a.t[id].data[eq.at]} vs ${b.t[id].data[eq.at]}`);
			tensors++;
		}
	}
	assert.deepEqual(counts, { ref: 3, preset: 60, 'held-out': 12 });
	// against PyTorch
	const refs = await readReference(fsFetch, join(MODELS, 'tinycnn'));
	let worst = 0;
	for (const r of refs) {
		const tr = cnn.forward(r.inputs.input);
		for (const id of r.order) worst = Math.max(worst, maxDiff(tr.t[id].data, r.tensors[id].data).max);
	}
	assert.ok(worst <= 1e-4, `vs PyTorch ${worst}`);
	t.diagnostic(`TinyCNN: ${inputs.length} inputs, ${tensors} traced tensors bit-identical to tinycnn.ts; vs PyTorch reference max |diff| ${worst.toExponential(3)} (bar 1e-4; shipped ${cnnSpec.parity.max_abs_diff})`);
});

test('TinyCNN receptive fields and unit details agree with tinycnn.ts', () => {
	assert.deepEqual(cnn.receptiveField('pool2', { y0: 3, x0: 3, y1: 3, x1: 3 }), { y0: 12, x0: 12, y1: 27, x1: 27 });
	for (const L of cnnSpec.layers as { id: string; outShape: number[] }[]) {
		if (L.outShape.length !== 3) {
			assert.deepEqual(cnn.receptiveField(L.id, { y0: 0, x0: 0, y1: 0, x1: 0 }), { y0: 0, x0: 0, y1: 27, x1: 27 }, `${L.id}: whole image`);
			continue;
		}
		const [, H, W] = L.outShape;
		for (let y = 0; y < H; y += 3)
			for (let x = 0; x < W; x += 2) {
				assert.deepEqual(cnn.receptiveField(L.id, { y0: y, x0: x, y1: y, x1: x }), cnnLegacy.receptiveField(L.id, y, x), `${L.id} (${y},${x})`);
			}
	}
	const x = Float32Array.from(json(join(MODELS, 'tinycnn', 'reference.json')).inputs[1].input);
	const tr = cnn.forward({ shape: [1, 28, 28], data: x });
	const lt = cnnLegacy.forward(x);
	const conv = cnn.explain(tr, 'conv1_2', [3, 10, 12]);
	const cd = cnnLegacy.convDetail(lt, 'conv1_2', 3, 10, 12);
	assert.equal(conv.result, cd.traced);
	assert.equal(conv.terms!.length, cd.inChannels * cd.kernel * cd.kernel);
	assert.deepEqual(conv.reads[0].box, cd.window);
	assert.equal(conv.groups!.length, 8);
	assert.ok(Math.abs(conv.groups![2].sum - cd.channelSums[2]) < 1e-6);
	const pool = cnn.explain(tr, 'pool1', [2, 5, 7]);
	const pd = cnnLegacy.poolDetail(lt, 'pool1', 2, 5, 7);
	assert.equal(pool.result, pd.value);
	assert.equal(pool.values.argmax, pd.argmax);
	const dense = cnn.explain(tr, 'dense', [5]);
	const dd = cnnLegacy.denseDetail(lt, 5);
	assert.equal(dense.result, dd.traced);
	assert.equal(dense.groups!.length, 16);
	for (let c = 0; c < 16; c++) assert.ok(Math.abs(dense.groups![c].sum - dd.channelSums[c]) < 1e-5);
	assert.equal(dense.values.zeros, dd.zeroInputs);
	const sm = cnn.explain(tr, 'softmax', [5]);
	const sd = cnnLegacy.softmaxDetail(lt);
	assert.equal(sm.values.max, sd.max);
	assert.ok(Math.abs(sm.values.sum - sd.sumExp) < 1e-12);
	assert.equal(sm.values.top, sd.top);
	assert.deepEqual(pickUnit(cnn, tr, 'softmax', 'argmax'), [sd.top]);
});

test('acceptance 3a: explain() equals the trace for every unit of every TinyCNN layer (3 reference inputs)', (t) => {
	let units = 0;
	let max = 0;
	for (const r of json(join(MODELS, 'tinycnn', 'reference.json')).inputs) {
		const tr = cnn.forward({ shape: [1, 28, 28], data: Float32Array.from(r.input) });
		const c = checkExplain(cnn, tr);
		assert.equal(c.failures.length, 0, JSON.stringify(c.failures.slice(0, 3)));
		assert.deepEqual(Object.keys(c.perLayer), tr.order, 'every layer explained');
		units += c.units;
		max = Math.max(max, c.max);
	}
	assert.ok(max <= 1e-5);
	t.diagnostic(`TinyCNN explain = trace: ${units} units (every unit of all 14 layers × 3 inputs), max |result − traced| ${max.toExponential(2)}`);
});

// ------------------------------------------------------------------ TinyGPT
const gptSpec = json(join(MODELS, 'tinygpt', 'spec.json'));
const gptW = ab(readFileSync(join(MODELS, 'tinygpt', 'weights.bin')));
const gpt: GraphModel = fromParts(gptSpec, gptW, CORE_OPS);
const gptLegacy = tinygpt.fromWeights(gptSpec, gptSpec.tensors, new Float32Array(gptW));
const tok = new BPETokenizer(json(join(MODELS, 'tinygpt', 'tokenizer.json')) as TokenizerJSON);
const LONG =
	'Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big red ball in the park. She ran to the ball and kicked it high into the sky. Her friend Tom came to play with her, and they laughed and played until the sun went down.';

function gptCases(): { T: number; ids: number[]; why: string }[] {
	const ref = json(join(MODELS, 'tinygpt', 'reference.json'));
	const one = tok.encode('Once');
	const long = tok.encode(LONG);
	assert.equal(one.length, 1, '"Once" is one token');
	assert.ok(long.length >= 64, 'the long string has at least 64 tokens');
	return [
		{ T: 1, ids: one, why: 'tokenized "Once"' },
		...ref.prompts.map((p: { ids: number[]; text: string }) => ({ T: p.ids.length, ids: p.ids, why: `reference prompt ${JSON.stringify(p.text.slice(0, 24))}…` })),
		{ T: 64, ids: long.slice(0, 64), why: 'first 64 tokens of a fixed story' }
	];
}

test('acceptance 2: TinyGPT executor within 1e-6 relative of tinygpt.ts at T = 1, 9, 22, 37, 64; within 1e-4 of PyTorch', async (t) => {
	const cases = gptCases();
	assert.deepEqual(cases.map((c) => c.T), [1, 9, 22, 37, 64]);
	const lines: string[] = [];
	for (const c of cases) {
		const a = gpt.forward(intTensor([c.T], c.ids));
		const b = gptLegacy.forward(c.ids);
		assert.deepEqual(a.order, b.order, `T=${c.T}: trace order`);
		let rel = 0;
		let bitSame = true;
		for (const id of b.order) {
			const d = maxDiff(a.t[id].data, b.t[id].data, { relative: true });
			assert.ok(!d.nan && d.max <= 1e-6, `T=${c.T} ${id}: relative diff ${d.max}`);
			rel = Math.max(rel, d.max);
			if (!bitEqual(a.t[id].data, b.t[id].data).equal) bitSame = false;
		}
		lines.push(`T=${c.T} (${c.why}): max relative diff vs tinygpt.ts ${rel.toExponential(2)}${bitSame ? ' (bit-identical)' : ''}, ${a.meta!.ms!.toFixed(1)} ms`);
	}
	const refs = await readReference(fsFetch, join(MODELS, 'tinygpt'));
	let worst = 0;
	for (const r of refs) {
		const tr = gpt.forward(r.inputs.input);
		let m = 0;
		for (const id of r.order) {
			const d = maxDiff(tr.t[id].data, r.tensors[id].data);
			assert.ok(!d.nan, `${r.id} ${id}: NaN or unmatched infinity`);
			m = Math.max(m, d.max);
		}
		assert.ok(m <= 1e-4, `${r.id}: ${m}`);
		lines.push(`vs PyTorch reference ${r.id}: max |diff| ${m.toExponential(3)}`);
		worst = Math.max(worst, m);
	}
	lines.push(`vs PyTorch (T = 9, 22, 37): max |diff| ${worst.toExponential(3)} (bar 1e-4; shipped ${gptSpec.parity.max_abs_diff})`);
	// T = 64 against PyTorch: local, git-ignored fixture; reported, never gating
	const root = atlasRoot();
	const vref = root ? join(root, 'pipeline', 'parity', 'verify_tinygpt', 'verify_ref.json') : null;
	if (vref && existsSync(vref)) {
		const v = json(vref);
		for (const p of v.prompts as { name: string; ids: number[]; kept: number; order: string[]; t32: Record<string, { b64: string }> }[]) {
			const ids = p.ids.slice(-64);
			if (ids.length !== 64 && ids.length !== 1) continue;
			const tr = gpt.forward(intTensor([ids.length], ids));
			let m = 0;
			for (const id of p.order) m = Math.max(m, maxDiff(tr.t[id].data, f32FromB64(p.t32[id].b64)).max);
			lines.push(`report only, verify_ref.json ${p.name} (T=${ids.length}) vs PyTorch: max |diff| ${m.toExponential(3)}`);
		}
	} else lines.push('verify_ref.json not found: T = 64 against PyTorch not reported');
	for (const l of lines) t.diagnostic(l);
});

test('acceptance 3b: explain() equals the trace on 2,000 sampled TinyGPT units', (t) => {
	const cases = gptCases();
	const tr: Trace = gpt.forward(intTensor([22], cases[2].ids));
	const c = checkExplain(gpt, tr, { sample: 2000, seed: 20261006 });
	assert.equal(c.units, 2000);
	assert.equal(c.failures.length, 0, JSON.stringify(c.failures.slice(0, 3)));
	assert.ok(c.max <= 1e-5);
	const layers = Object.keys(c.perLayer).length;
	// the T = 64 trace too (the longest context), 500 more units
	const tr64 = gpt.forward(intTensor([64], cases[4].ids));
	const c64 = checkExplain(gpt, tr64, { sample: 500, seed: 7 });
	assert.equal(c64.failures.length, 0, JSON.stringify(c64.failures.slice(0, 3)));
	t.diagnostic(`TinyGPT explain = trace: 2000 units over ${layers} of 66 layers at T=22, max |diff| ${c.max.toExponential(2)}; +500 units at T=64, max ${c64.max.toExponential(2)}`);
	// a unit outside the tensor is a RangeError, not a wrong answer
	assert.throws(() => gpt.explain(tr, 'logits', [22, 0]), RangeError);
	// a masked score explains as −∞
	const m = gpt.explain(tr, 'L0.attn.masked', [0, 1, 5]);
	assert.equal(m.result, -Infinity);
	assert.equal(m.traced, -Infinity);
});

test('executor guards: wrong input shapes and dims out of range fail loudly', () => {
	assert.throws(() => cnn.forward({ shape: [1, 27, 28], data: new Float32Array(27 * 28) }), /axis 1 is 27/);
	assert.throws(() => gpt.forward(intTensor([65], new Array(65).fill(1))), /outside 1\.\.64/);
	assert.throws(() => gpt.forward(intTensor([2], [1, 5000])), /outside 0\.\.1023/);
});
