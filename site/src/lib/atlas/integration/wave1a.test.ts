// Integration 1a (design §8.1): checks that need two wave-1a packages at once.
//   1. toSpec (W1-B1) -> executor (W1-A) forward equals the autograd Module forward (<= 1e-6 relative), the spec
//      validates against CORE_OPS, and explain() equals the trace on every unit.
//   2. trainingProcess (W1-B1) passes the process determinism harness (W1-C), coarse and fine.
//   3. planLesson (W1-C) on the real CORE_OPS (W1-A) and the shipped TinyCNN / TinyGPT specs: the reported metrics of
//      design §5.2 (TinyCNN coverage of the hand-made lesson ids, TinyGPT step count), and lint-clean auto lessons.
//   4. a 'live-model' source (W1-C runtime) on a real trainingProcess (W1-B1 liveSpec + toSpec) through the executor.
// Run from site/: node --test src/lib/atlas/integration/wave1a.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORE_OPS } from '../ops/index.ts';
import { fromParts } from '../graph/executor.ts';
import { checkExplain } from '../graph/explain.ts';
import { normalizeSpec } from '../graph/legacy.ts';
import { validateSpec } from '../schema/spec.ts';
import { createRng } from '../core/rng.ts';
import { fsFetch } from '../core/io.ts';
import { MLP, Linear, loadParams, type Activation } from '../autograd/nn.ts';
import { input, noGrad, type Var } from '../autograd/var.ts';
import { crossEntropy } from '../autograd/loss.ts';
import type { ToSpecOpts } from '../autograd/tospec.ts';
import { trainingProcess } from '../train/trainer.ts';
import { moons, sampleBatch, type Points2D } from '../data/points2d.ts';
import { checkDeterminism } from '../process/determinism.ts';
import { planLesson } from '../lesson/plan.ts';
import { compileLesson } from '../lesson/compile.ts';
import { lintLesson } from '../lesson/lint.ts';
import { createRuntime } from '../runtime/runtime.ts';
import type { AtlasSpec, ExplainerManifest, InputAdapter, PackageModules, Rng, TrainOpt, TrainerDef } from '../types/index.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b));

// ------------------------------------------------------------------ 1. toSpec -> executor = Module forward
type Spec1 = { spec: AtlasSpec; weights: Float32Array };
type Case = { name: string; inF: number; make: (rng: Rng) => { forward(x: Var): Var; toSpec(o: { slug: string } & ToSpecOpts): Spec1 }; opts?: ToSpecOpts };
const ACTS: Activation[] = ['tanh', 'relu', 'sigmoid', 'silu', 'gelu', 'gelu_tanh'];
const CASES: Case[] = [
	...ACTS.map((act): Case => ({ name: `MLP 2-8-8-2 ${act}`, inF: 2, make: (r) => new MLP('net', [2, 8, 8, 2], act, r), opts: { inputAxes: ['xy'], outputAxes: ['class'] } })),
	{ name: 'MLP 1-32-32-1 tanh (PINN)', inF: 1, make: (r) => new MLP('net', [1, 32, 32, 1], 'tanh', r) },
	{ name: 'MLP 4-6-3 relu, no bias', inF: 4, make: (r) => new MLP('mlp', [4, 6, 3], 'relu', r, { bias: false }) },
	{ name: 'Linear 5-3', inF: 5, make: (r) => new Linear('head', 5, 3, r) },
	{ name: 'flow-2d MLP 3-64-64-64-2 silu, inputs xy + t (concat)', inF: 3, make: (r) => new MLP('net', [3, 64, 64, 64, 2], 'silu', r), opts: { inputs: [{ name: 'xy', size: 2, axes: ['xy'] }, { name: 't', size: 1 }] } }
];

test('toSpec -> executor forward equals the Module forward (<= 1e-6 relative); the spec validates; explain = trace', () => {
	for (const c of CASES) {
		const rng = createRng(7);
		const m = c.make(rng.fork('init'));
		const { spec, weights } = m.toSpec({ slug: 'live', ...(c.opts ?? {}) });
		const issues = validateSpec(JSON.parse(JSON.stringify(spec)), { ops: CORE_OPS, weightsBytes: weights.byteLength });
		assert.deepEqual(issues.filter((i) => i.level === 'error'), [], `${c.name}: spec errors`);
		const model = fromParts(spec, weights.slice().buffer, CORE_OPS);
		const xr = rng.fork('x');
		let worst = 0;
		let explainMax = 0;
		for (let n = 0; n < 40; n++) {
			const x32 = Float32Array.from({ length: c.inF }, () => xr.normal() * 1.5);
			const ref = noGrad(() => m.forward(input({ shape: [1, c.inF], data: Float64Array.from(x32) }))).value.data;
			let at = 0;
			const ins = c.opts?.inputs
				? Object.fromEntries(c.opts.inputs.map((d) => [d.name, { shape: [d.size], data: x32.slice(at, (at += d.size)) }]))
				: { input: { shape: [c.inF], data: x32 } };
			const trace = model.forward(ins);
			const out = trace.t[spec.outputs![0]];
			assert.equal(out.data.length, ref.length, `${c.name}: output size`);
			for (let i = 0; i < ref.length; i++) worst = Math.max(worst, rel(out.data[i], ref[i]));
			if (n < 4) explainMax = Math.max(explainMax, checkExplain(model, trace).max);
		}
		console.log(`ℹ ${c.name}: ops ${spec.layers.map((l) => l.op).join(',')}; executor vs Module max rel ${worst.toExponential(2)}; explain vs trace ${explainMax.toExponential(2)}`);
		assert.ok(worst <= 1e-6, `${c.name}: executor vs Module max rel ${worst}`);
		assert.ok(explainMax <= 1e-5, `${c.name}: explain vs trace ${explainMax}`);
	}
});

// ------------------------------------------------------------------ 2. trainingProcess through W1-C's harness
type Opt = TrainOpt & { hidden: number };
export const PLAYGROUND: TrainerDef<Points2D, MLP, { x: Var; y: Int32Array }, Opt> = {
	id: 'playground',
	title: { en: 'Playground', zh: '演练场' },
	defaults: { seed: 1, lr: 0.03, batch: 16, steps: 150, snapshotEvery: 10, optimizer: 'adam', hidden: 6 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, opt.hidden, 2], 'tanh', rng),
	batch(data, _k, rng, opt) {
		const b = sampleBatch(data, opt.batch, rng);
		return { x: input(b.x), y: b.label };
	},
	loss: (m, b, rec) => ({ loss: crossEntropy(m.forward(b.x, rec), b.y) })
};

test('trainingProcess passes the W1-C determinism harness (coarse and fine)', () => {
	const p = trainingProcess(PLAYGROUND);
	const data = moons(120, createRng(99), { noise: 0.1 });
	for (const detail of ['coarse', 'fine'] as const) {
		const r = checkDeterminism(p, data, { steps: 150 }, { seed: 4, detail, samples: 200 });
		console.log(`ℹ ${detail}: frames ${r.frames}, sameFrames ${r.sameFrames}, sameStates ${r.sameStates}, replay ${r.replayChecked} (${r.replayMismatches.length} mismatches)`);
		assert.ok(r.ok, `${detail}: ${r.problems.join(' | ')}`);
	}
});

// ------------------------------------------------------------------ 3. planLesson on the real ops
const MANIFEST = (slug: string, adapter?: string): ExplainerManifest => ({
	slug,
	type: 'model',
	kinds: ['trace'],
	quality: 'draft',
	devOnly: true,
	assets: { dir: `models/${slug}`, auto: ['spec.json', 'weights.bin'] },
	...(adapter ? { input: { adapter } } : {}),
	page: { goals: [], prequestion: { id: 'q', question: { en: 'q', zh: 'q' }, options: [{ en: 'a', zh: 'a' }], answer: 0, explain: { en: 'e', zh: 'e' } } },
	credits: []
});
const loadSpec = (slug: string): AtlasSpec => normalizeSpec(JSON.parse(readFileSync(join(STATIC, 'models', slug, 'spec.json'), 'utf8')));
// TinyCNN's hand-made lesson ids (src/lib/pages/tinycnn) and the auto step that teaches the same idea.
const HAND_TO_AUTO: Record<string, string> = { pixels: 'input', slide: 'op:conv1_1', relu: 'op:relu1_1', combine: 'op:conv1_2', pool: 'op:pool1', flatten: 'op:flatten', dense: 'op:dense', softmax: 'output' };
const HAND_IDS = ['pixels', 'slide', 'kernels', 'relu', 'combine', 'pool', 'grow', 'flatten', 'dense', 'softmax'];

test('planLesson on CORE_OPS: TinyCNN covers 8 of the 10 hand-made lesson ids; TinyGPT step count', () => {
	const cnn = planLesson(loadSpec('tinycnn'), CORE_OPS, MANIFEST('tinycnn'));
	const ids = cnn.map((s) => s.id);
	console.log(`ℹ TinyCNN auto plan (${ids.length}): ${ids.join(', ')}`);
	const covered = HAND_IDS.filter((h) => HAND_TO_AUTO[h] && ids.includes(HAND_TO_AUTO[h]));
	console.log(`ℹ TinyCNN hand ids covered ${covered.length}/10: ${covered.map((h) => `${h}=${HAND_TO_AUTO[h]}`).join(', ')}; author steps: ${HAND_IDS.filter((h) => !covered.includes(h)).join(', ')}`);
	assert.equal(covered.length, 8);
	const gpt = planLesson(loadSpec('tinygpt'), CORE_OPS, MANIFEST('tinygpt'));
	console.log(`ℹ TinyGPT auto plan (${gpt.length}): ${gpt.map((s) => s.id).join(', ')}`);
	assert.ok(gpt.length >= 5, `TinyGPT plan has ${gpt.length} steps`);
	for (const slug of ['tinycnn', 'tinygpt']) {
		const c = compileLesson(null, { spec: loadSpec(slug), ops: CORE_OPS, manifest: MANIFEST(slug) });
		assert.deepEqual(c.issues.filter((i) => i.level === 'error'), [], `${slug}: static compile errors`);
	}
});

test('the TinyCNN auto lesson lints clean on the real model (reference input, blank, dot; en and zh)', async () => {
	const ref = JSON.parse(readFileSync(join(STATIC, 'models', 'tinycnn', 'reference.json'), 'utf8'));
	const first = (ref.inputs ?? ref.cases)?.[0];
	const flat: number[] = (first?.input ?? first?.data ?? first)?.flat?.(3) ?? [];
	assert.equal(flat.length, 784, 'reference input 0 is 28x28');
	// A raw 28x28 adapter: the real drawing28 adapter arrives in wave 1b (W1-D1).
	const raw28: InputAdapter = {
		id: 'raw28',
		kind: 'drawing',
		preprocess: (raw) => {
			const d = Float32Array.from(raw as number[]);
			return { tensors: { input: { shape: [1, 28, 28], data: d } }, steps: [], empty: d.every((v) => v === 0) };
		},
		edgeFixtures: () => [
			{ id: 'blank', label: { en: 'blank', zh: '空白' }, raw: new Array(784).fill(0) },
			{ id: 'dot', label: { en: 'dot', zh: '一个点' }, raw: Array.from({ length: 784 }, (_, i) => (i === 406 ? 1 : 0)) }
		],
		layer0: { view: 'layer.tensor', narration: { what: { en: 'Your drawing becomes {n|int} numbers, arranged as {shape}.', zh: '你的画变成 {n|int} 个数，排成 {shape}。' } } }
	};
	const pkg: PackageModules = { slug: 'tinycnn', manifest: { ...MANIFEST('tinycnn', 'raw28'), output: { view: 'kit.probbars' } }, lesson: null, ops: CORE_OPS, processes: {}, adapters: { raw28 } };
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: flat });
	const c = compileLesson(null, { spec: rt.spec, ops: CORE_OPS, manifest: pkg.manifest, adapters: pkg.adapters });
	const fixtures = [{ id: 'ref0', label: { en: 'reference 0', zh: '参考 0' }, raw: flat }, ...raw28.edgeFixtures().map((e) => ({ ...e, id: `edge:${e.id}` }))];
	const issues = await lintLesson(c, (fx) => createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: fx.raw }), fixtures);
	const errors = issues.filter((i) => i.level === 'error');
	const warns = [...new Set(issues.filter((i) => i.level === 'warn').map((i) => `${i.code}${i.step ? ' ' + i.step : ''}`))];
	console.log(`ℹ TinyCNN auto lesson: ${c.steps.length} steps, ${errors.length} lint errors; warnings: ${warns.join('; ') || 'none'}`);
	assert.deepEqual(errors.map((e) => `${e.code} ${e.step ?? ''} ${e.fixture ?? ''} ${e.lang ?? ''} ${e.message}`), []);
});

test('the TinyGPT auto lesson lints clean on the real model (7 tokens, 1 token, more than the 64-token context)', async () => {
	// A token-id adapter: the real text-bpe adapter arrives in wave 1b (W1-D1). It keeps the last 64 ids, as tinygpt.ts does.
	const ids: InputAdapter = {
		id: 'ids',
		kind: 'text',
		preprocess: (raw) => {
			const a = (raw as number[]).slice(-64);
			return { tensors: { input: { shape: [a.length], data: Int32Array.from(a), dtype: 'i32' } }, steps: [], empty: a.length === 0 };
		},
		edgeFixtures: () => [
			{ id: 'one', label: { en: 'one token', zh: '一个词元' }, raw: [101] },
			{ id: 'long', label: { en: '70 tokens', zh: '70 个词元' }, raw: Array.from({ length: 70 }, (_, i) => (i * 37) % 1000) }
		],
		layer0: { view: 'layer.tensor', narration: { what: { en: 'Your text becomes {n|int} token ids. That is the shape {shape}.', zh: '你的文字变成 {n|int} 个词元 id。形状是 {shape}。' } } }
	};
	const pkg: PackageModules = { slug: 'tinygpt', manifest: MANIFEST('tinygpt', 'ids'), lesson: null, ops: CORE_OPS, processes: {}, adapters: { ids } };
	const mid = [464, 219, 257, 640, 11, 612, 373];
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: mid });
	const c = compileLesson(null, { spec: rt.spec, ops: CORE_OPS, manifest: pkg.manifest, adapters: pkg.adapters });
	const fixtures = [{ id: 'mid', label: { en: '7 tokens', zh: '7 个词元' }, raw: mid }, ...ids.edgeFixtures().map((e) => ({ ...e, id: `edge:${e.id}` }))];
	const issues = await lintLesson(c, (fx) => createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: fx.raw }), fixtures);
	const errors = issues.filter((i) => i.level === 'error');
	const warns = [...new Set(issues.filter((i) => i.level === 'warn').map((i) => `${i.code}${i.step ? ' ' + i.step : ''}`))];
	console.log(`ℹ TinyGPT auto lesson: ${c.steps.length} steps, ${errors.length} lint errors; warnings: ${warns.join('; ') || 'none'}`);
	assert.deepEqual(errors.map((e) => `${e.code} ${e.step ?? ''} ${e.fixture ?? ''} ${e.lang ?? ''} ${e.message}`), []);
});

test('live-model source: a kind-A trace on the trained weights of a real trainingProcess equals the Module forward', async () => {
	const data = moons(60, createRng(5), { noise: 0.1 });
	// raw = the data set; the traced input is point 0 (an mlp-playground-style "trace one point").
	const pts: InputAdapter<Points2D> = {
		id: 'pts',
		kind: 'points2d',
		preprocess: (raw) => ({ tensors: { input: { shape: [2], data: Float32Array.from(raw.xy.subarray(0, 2)) } }, steps: [], empty: raw.n === 0 }),
		edgeFixtures: () => [],
		layer0: { view: 'layer.tensor', narration: { what: { en: 'The points.', zh: '这些点。' } } }
	};
	const p = trainingProcess(PLAYGROUND);
	const manifest: ExplainerManifest = {
		...MANIFEST('playground', 'pts'),
		kinds: ['train', 'trace'],
		sources: { train: { kind: 'train', trainer: 'playground', opt: { steps: 40, hidden: 5 } }, pt: { kind: 'live-model', of: 'train' } }
	};
	const pkg: PackageModules = { slug: 'playground', manifest, lesson: null, ops: CORE_OPS, processes: { 'train:playground': p }, adapters: { pts: pts as InputAdapter } };
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC, input: data });
	await rt.ensure('train', 'last');
	await rt.ensure('pt', 'last');
	assert.equal(rt.sources.pt.status, 'ready', rt.sources.pt.error);
	const tape = rt.sources.train.tape!;
	const params = (tape.state(tape.length - 1) as { params: Record<string, { shape: number[]; data: Float64Array }> }).params;
	assert.equal(params['net.0.weight'].shape[0], 5, 'the source opt (hidden 5) reached the rebuilt model');
	const m = PLAYGROUND.model(createRng(0), { ...PLAYGROUND.defaults, hidden: 5 });
	loadParams(m, params);
	const ref = noGrad(() => m.forward(input({ shape: [1, 2], data: Float64Array.from(Float32Array.from(data.xy.subarray(0, 2))) }))).value.data;
	const trace = rt.sources.pt.trace!;
	const out = trace.t[trace.order[trace.order.length - 1]].data;
	let worst = 0;
	for (let i = 0; i < ref.length; i++) worst = Math.max(worst, rel(out[i], ref[i]));
	console.log(`ℹ live-model after ${rt.num('train.step')} steps: ${trace.order.join(' -> ')}; vs Module max rel ${worst.toExponential(2)}`);
	assert.ok(worst <= 1e-6, `live-model vs Module ${worst}`);
});
