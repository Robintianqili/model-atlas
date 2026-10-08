// tinyclip claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, most of them over every fixture (the six presets and the adapter's four edge cases) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/tinyclip/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { record } from '../../atlas/process/record.ts';
import { readReference } from '../../atlas/graph/refs.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { ORDER, PRESET_CAPS } from './lesson.ts';
import { nce, synth } from './process.ts';
import { FIXED, MAX_WORDS, PARTNERS, VOCAB, crcBytes, render, strokesFor, strokesOf, tokenize } from './scenes.ts';
import { drawingInput } from '../../atlas/adapters/drawing28/adapter.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { captionsData } from './views/match/data.ts';
import { rawOf } from './adapters/shapes-captions/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinyclip', SITE).then((r) => r.pkg);
const len = (a: ArrayLike<number>) => Math.sqrt(Array.from(a).reduce((s, x) => s + x * x, 0));
const rows = (t: Tensor) => Array.from({ length: t.data.length / t.shape[t.shape.length - 1] }, (_, k) => t.data.subarray(k * t.shape[t.shape.length - 1], (k + 1) * t.shape[t.shape.length - 1]));
const stateAt = (tp: Tape, phase: string) => tp.state(tp.find({ phase })) as Record<string, Tensor>;

test('page and catalog: parameter count, weight size, word list and limits match the spec and facts', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const spec = rt.spec!;
	assert.equal(spec.params, 15905);
	assert.equal(rt.num('fact:params'), spec.params);
	assert.ok(manifest.catalog!.params!.en.includes('15,905'));
	const bytes = readFileSync(join(SITE, 'static/models/tinyclip/weights.bin')).length;
	assert.equal(bytes, spec.params * 4);
	assert.equal(Math.round(bytes / 1024), 62, '62 KB of weights');
	assert.deepEqual((spec as unknown as { vocab: string[] }).vocab, VOCAB, 'Python (model.py) and TS (scenes.ts) use the same word list');
	assert.equal(rt.num('fact:vocab'), VOCAB.length - 1, '23 words');
	assert.equal(spec.dims!.L.max, MAX_WORDS);
	const out = (id: string) => spec.layers.find((l) => l.id === id)!.outShape;
	assert.deepEqual([out('img.proj'), out('txt.proj'), out('img.flat')], [[32], ['K', 32], [400]], '32 numbers per tower; 400 features');
	assert.ok(manifest.page.goals[0].en.includes('32 numbers') && lesson.title.en.includes('TinyCLIP'));
	assert.equal(spec.dims!.K.max, 3, 'three caption boxes');
	assert.equal(rt.num('fact:trainN'), 20000);
	const intro = manifest.page.sections!.intros!.training!;
	for (const s of [`${rt.num('fact:epochs')} epochs`, '20,000 training pairs', `batches of ${rt.num('fact:batch')}`, `${rt.num('fact:testN').toLocaleString('en-US')} test pairs`]) assert.ok(intro.en.includes(s), s);
	rt.close();
});

test('training data: Python trained on the bytes TS renders (CRC-32 of the first chunk and of the fixed drawings)', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const tape = record(synth, null, { n: 1000, chunk: 1000, twoShare: 0.25 }, { seed: 1 });
	assert.equal(tape.frame(0).phase, 'fixed');
	assert.equal(tape.frame(0).metrics!.hash, rt.num('fact:fixedHash'), 'fixed drawings, as Python read them');
	assert.equal(tape.frame(1).metrics!.hash, rt.num('fact:chunkHash0'), 'training chunk 0, as Python read it');
	assert.equal(crcBytes((tape.state(1) as { images: Tensor }).images.data as Float32Array), rt.num('fact:chunkHash0'));
	rt.close();
});

test('parity cases: the reference inputs are the drawings and word ids the page computes for the same fixed drawings', async () => {
	const cases = await readReference(fsFetch, join(SITE, 'static/models/tinyclip'));
	assert.ok(cases.length >= 10);
	for (const c of cases) {
		const f = FIXED.find((x) => x.id === c.id);
		if (!f) continue; // 'no-words': a caption without a known word (an all-padding row)
		assert.deepEqual(Array.from(c.inputs.image.data), Array.from(render(strokesFor(f))), `${c.id}: image`);
		const ids = f.captions.map((x) => tokenize(x).ids).filter((r) => r.length);
		const L = Math.max(...ids.map((r) => r.length));
		assert.deepEqual(Array.from(c.inputs.text.data), ids.flatMap((r) => [...r, ...Array(L - r.length).fill(0)]), `${c.id}: text`);
	}
});

test('steps "image" and "caption": the two towers are built as narrated', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const ops = (pre: string) => rt.spec!.layers.filter((l) => l.id.startsWith(pre)).map((l) => l.op);
	assert.deepEqual(ops('img.'), ['conv2d', 'relu', 'maxpool2d', 'conv2d', 'relu', 'maxpool2d', 'flatten', 'linear', 'l2norm']);
	assert.deepEqual(ops('txt.'), ['embedding', 'mean_pool', 'linear', 'l2norm']);
	assert.equal(lesson.steps.length, 10);
	rt.close();
});

test('steps "normalize" and "cosine": length 1, a cosine between −1 and 1, score = s · cosine (every fixture)', async () => {
	await withFixtures(
		await pkgP,
		(rt) => {
			const t = rt.sources.fwd.trace!.t;
			assert.ok(Math.abs(len(t['img.norm'].data) - 1) < 1e-6);
			const s = Math.min(Math.exp(rt.model!.weights['sim.logit_scale'].data[0]), 100);
			rows(t['txt.norm']).forEach((b, k) => {
				assert.ok(Math.abs(len(b) - 1) < 1e-6 || len(b) === 0);
				const cos = Array.from(b).reduce((x, v, d) => x + v * t['img.norm'].data[d], 0);
				assert.ok(Math.abs(cos) <= 1 + 1e-6);
				assert.ok(Math.abs(t.sim.data[k] - s * cos) <= 1e-5 * s);
			});
		},
		opts
	);
});

test('steps "matrix" and "gradient": five fixed partners; with the starting weights the true pairs do not stand out; every true pair is pulled up and every wrong pair pushed down (every fixture)', async () => {
	assert.equal(PARTNERS.length, 5);
	await withFixtures(
		await pkgP,
		async (rt) => {
			for (const src of ['nce0', 'nce']) {
				await rt.ensure(src, 'last');
				const tp = rt.sources[src].tape!;
				const m = tp.frame(tp.find({ phase: 'scores' })).metrics!;
				assert.equal(m.N, 6);
				if (src === 'nce0') assert.ok(Math.abs(m.diagMean - m.offMean) < 1.5, `before training: diagonal ${m.diagMean} vs ${m.offMean}`);
				const G = stateAt(tp, 'grad').G.data;
				for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) assert.ok(i === j ? G[i * 6 + j] < 0 : G[i * 6 + j] > 0, `${src} G[${i},${j}] = ${G[i * 6 + j]}`);
			}
		},
		opts
	);
});

test('step "order" (breakIt): swapping the two shape words gives the same caption vector and the same score with any drawing', async () => {
	await withFixtures(
		await pkgP,
		(rt) => {
			const r = captionsData(rt, ORDER);
			assert.equal(r[0].score, r[1].score);
			assert.equal(r[0].share, r[1].share);
			assert.notEqual(r[0].score, r[2].score, '“below” instead of “above” changes the words, so the score');
		},
		opts
	);
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(rt.num('fact:orderTies'), rt.num('fact:twoN'), 'every two-shape test drawing tied (stats.py)');
	assert.ok(rt.num('fact:orderMaxGap') < 1e-4, 'the ties in float32 PyTorch differ only by rounding');
	rt.close();
});

test('task "captions": a preset does not count, nor a new drawing or a typo with the preset captions; three different captions, one of them the learner own, do', async () => {
	const step = lesson.steps.find((s) => 'id' in s && s.id === 'captions') as { task: { done: (rt: unknown, at: unknown) => boolean } };
	const pkg = await pkgP;
	const preset = rawOf(FIXED[0]);
	const own = { ...preset, preset: undefined, captions: ['a big circle in the middle', 'a small circle', 'a big square'] };
	assert.ok(PRESET_CAPS.has(tokenize('a small circle').ids.join(' ')) && !PRESET_CAPS.has(tokenize('a big square').ids.join(' ')));
	const drawn = { ...preset, preset: undefined, strokes: [[{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.8 }]] };
	const typo = { ...preset, preset: undefined, captions: ['A big circle in the middle!', ...preset.captions!.slice(1)] };
	for (const [raw, want] of [[preset, false], [drawn, false], [typo, false], [own, true], [{ ...own, captions: ['a circle', 'A circle!', 'a star'] }, false]] as const) {
		const rt = await createRuntime(pkg, { ...opts, input: raw });
		assert.equal(step.task.done(rt, { index: 0 }), want);
		rt.close();
	}
});

test('step "pairs": the captions use every one of the 23 words; the frame keeps size and place, which the crop alone would erase', () => {
	const tape = record(synth, null, { n: 1000, chunk: 1000, twoShare: 0.25 }, { seed: 1 });
	const used = new Set(((tape.state(1) as { captions: string[] }).captions).flatMap((c) => c.split(' ')));
	assert.deepEqual([...used].sort(), VOCAB.slice(1).sort());
	/** the ink's box inside the frame (the frame takes the outer 3 pixels) */
	const box = (img: Float32Array, m: number) => {
		const xs: number[] = [];
		for (let y = m; y < 28 - m; y++) for (let x = m; x < 28 - m; x++) if (img[y * 28 + x] > 0.5) xs.push(x);
		return { mid: (Math.min(...xs) + Math.max(...xs)) / 2, w: Math.max(...xs) - Math.min(...xs) };
	};
	for (let shape = 0; shape < 5; shape++) {
		const small = strokesOf({ kind: 'one', shape, size: 0, place: 0 }, createRng(shape + 1));
		const big = strokesOf({ kind: 'one', shape, size: 1, place: 4 }, createRng(shape + 1));
		const a = box(render(small), 3), b = box(render(big), 3);
		assert.ok(a.mid < 11 && Math.abs(b.mid - 13.5) < 2 && b.w > a.w + 2, `shape ${shape}: with the frame, left/small vs middle/big: ${JSON.stringify([a, b])}`);
		const na = box(drawingInput({ strokes: small }).input, 0), nb = box(drawingInput({ strokes: big }).input, 0);
		assert.ok(Math.abs(na.mid - nb.mid) <= 1.5 && Math.abs(na.w - nb.w) <= 2, `shape ${shape}: without the frame both fill the grid: ${JSON.stringify([na, nb])}`);
	}
});

test('step "caption" with no known word: no rows to average, so the average is all zeros and the caption vector is the bias', async () => {
	const pkg = await pkgP;
	const raw = { ...rawOf(FIXED[0]), preset: undefined, captions: ['blue', 'hello there', ''] };
	const rt = await createRuntime(pkg, { ...opts, input: raw });
	await rt.ensure('fwd', 'last');
	const t = rt.sources.fwd.trace!.t;
	assert.ok(Array.from(t['txt.pool'].data).every((v) => v === 0));
	const bias = rt.model!.weights['txt.proj.bias'].data;
	assert.ok(Array.from(t['txt.proj'].data).every((v, d) => Math.abs(v - bias[d]) < 1e-7));
	rt.close();
});

test('step "matrix" (predict): with one vector for every drawing and caption, every score is s, the shares are even and the loss is ln N', () => {
	const N = 6, D = 32, s = 21.1;
	const v = Array.from({ length: D }, (_, d) => (d === 0 ? 1 : 0));
	const tape = record(nce, { N, D, scale: s, I: Array.from({ length: N }, () => v).flat(), T: Array.from({ length: N }, () => v).flat() }, {}, { seed: 1 });
	assert.ok(Array.from(stateAt(tape, 'scores').L.data).every((x) => Math.abs(x - s) < 1e-12), 'every score is the largest possible, s');
	assert.ok(Array.from(stateAt(tape, 'softmax').P.data).every((x) => Math.abs(x - 1 / N) < 1e-12), 'even shares');
	const m = tape.frame(tape.find({ phase: 'softmax' })).metrics!;
	assert.ok(Math.abs(m.loss - Math.log(N)) < 1e-12, 'the loss of an even guess');
});
