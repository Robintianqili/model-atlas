// contrastive-learning claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, most of them on every fixture (the three starting batches and the adapter's edge
// batches) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/contrastive-learning/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { readReference } from '../../atlas/graph/refs.ts';
import type { AnyTensor } from '../../atlas/types/index.ts';
import { CKPTS, ckSource, scaleIn } from './batch.ts';
import { lossAt, offMax, tauCurve } from './data.ts';
import { VOCAB } from './scenes.ts';
import { partner } from './process.ts';
import { START, pairsOf } from './adapters/batch/adapter.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('contrastive-learning', SITE).then((r) => r.pkg);
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
async function frames(rt: ExplainerRuntime, src: string) {
	await rt.ensure(src, 'last');
	const tp = rt.sources[src].tape!;
	const at = (phase: string) => tp.find({ phase });
	return {
		state: (phase: string) => tp.state(at(phase)) as Record<string, Tensor>,
		m: (phase: string) => (tp.frame(at(phase)) as Frame).metrics!,
		tape: tp
	};
}

test('steps "pairs" and "cell": every vector has length 1, and each cell is the sum of the products of two vectors', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'nce');
		const { I, T, C } = f.state('scores');
		const N = C.shape[0], D = I.shape[1];
		assert.equal(N, pairsOf(rt.input).length, `${fx.id}: one row per pair`);
		assert.equal(D, 32, '32 numbers per vector');
		for (let i = 0; i < N; i++) {
			const len = (t: Tensor) => Math.sqrt(Array.from(t.data.subarray(i * D, (i + 1) * D)).reduce((s, x) => s + x * x, 0));
			assert.ok(close(len(I), 1, 1e-5) && close(len(T), 1, 1e-5), `${fx.id}: vector ${i} has length 1`);
			for (let j = 0; j < N; j++) {
				let s = 0;
				for (let d = 0; d < D; d++) s += I.data[i * D + d] * T.data[j * D + d];
				assert.ok(close(C.data[i * N + j], s, 1e-12), `${fx.id}: cell (${i}, ${j})`);
				assert.ok(C.data[i * N + j] <= 1 + 1e-6 && C.data[i * N + j] >= -1 - 1e-6, 'a cosine lies between −1 and 1');
			}
		}
	}, opts);
});

test('step "softmax": the row loss is −log of the share, and InfoNCE is the mean of the row and the column losses', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'nce');
		const s = f.m('scores'), m = f.m('softmax');
		assert.ok(close(m.l0, -Math.log(m.p00), 1e-12), fx.id);
		assert.ok(close(m.loss, (m.rowLoss + m.colLoss) / 2, 1e-12), `${fx.id}: rows and columns averaged`);
		assert.ok(close(lossAt(f.state('scores').C.data, s.N, s.tau), m.loss, 1e-9), `${fx.id}: data.ts lossAt (the curve) gives the same loss`);
		const P = f.state('softmax').P;
		for (let i = 0; i < s.N; i++) assert.ok(close(Array.from(P.data.subarray(i * s.N, (i + 1) * s.N)).reduce((a, b) => a + b, 0), 1, 1e-12), 'the shares of a row add up to one');
	}, opts);
});

test('step "hot" and the quiz: a very large τ gives the loss of a blind guess, log N (log 8 ≈ 2.08)', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'nce');
		const s = f.m('scores');
		assert.ok(close(f.m('softmax').chance, Math.log(s.N), 1e-12));
		assert.ok(Math.abs(lossAt(f.state('scores').C.data, s.N, 1e6) - Math.log(s.N)) < 1e-5, `${fx.id}: τ → ∞ gives log N`);
	}, opts);
	assert.equal(Math.log(8).toFixed(2), '2.08');
	for (const N of [2, 5, 8]) assert.ok(close(lossAt(new Float64Array(N * N).fill(0.3), N, 0.1), Math.log(N), 1e-12), 'equal scores: log N, growing with the batch (pitfall)');
});

test('step "drag": all partners first -> the loss falls all the way as τ shrinks; else it cannot reach 0, and below an inner lowest point it climbs again', async () => {
	const seen = new Set<string>();
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'nce');
		const s = f.m('scores'), m = f.m('softmax');
		const c = tauCurve(f.state('scores').C.data, s.N);
		const wrong = 2 * s.N - m.rowRight - m.colRight;
		if (wrong === 0) {
			for (let k = 1; k < c.loss.length; k++) assert.ok(c.loss[k] >= c.loss[k - 1] - 1e-12, `${fx.id}: the loss grows with τ`);
			assert.ok(c.best.atEdge);
			seen.add('right');
		} else {
			// a row or column whose own partner is not strictly first gives it a share of at most 1/2: −log ≥ log 2
			const floor = (wrong * Math.log(2)) / (2 * s.N);
			assert.ok(c.loss.every((l) => l >= floor - 1e-12), `${fx.id}: the loss cannot fall to 0 (at least ${floor})`);
			assert.ok(c.best.loss > 0);
			if (!c.best.atEdge) assert.ok(c.loss[0] > c.best.loss, `${fx.id}: below the lowest point the loss climbs again`);
			seen.add(c.best.atEdge ? 'floor' : 'climb');
		}
	}, opts);
	assert.deepEqual([...seen].sort(), ['climb', 'floor', 'right'], 'the fixtures reach all three branches of the narration');
});

test('step "gradient": the true pairs get a negative gradient, the wrong ones a positive one: their row and column shares over 2N', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'nce');
		const N = f.m('scores').N;
		const { P, Q } = f.state('softmax');
		const G = f.state('grad').G;
		for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
			const k = i * N + j;
			const want = (P.data[k] + Q.data[k] - (i === j ? 2 : 0)) / (2 * N);
			assert.ok(close(G.data[k], want, 1e-9), `${fx.id}: G(${i}, ${j})`);
			assert.ok(i === j ? G.data[k] <= 0 : G.data[k] >= 0, `${fx.id}: sign of G(${i}, ${j})`);
		}
	}, opts);
});

test('steps "train" and "learned": the checkpoints, their temperatures and the measured test numbers', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const last = CKPTS[CKPTS.length - 1];
	assert.equal(last, rt.num('fact:updates'), 'the last checkpoint is the last update of the run');
	assert.equal(rt.num('fact:updates'), rt.num('fact:epochs') * Math.floor(rt.num('fact:trainN') / rt.num('fact:batch')));
	assert.equal(CKPTS.length, rt.num('fact:checkpoints'));
	for (const u of CKPTS) {
		const spec = JSON.parse(readFileSync(join(SITE, `static/models/contrastive-learning/ck/${u}/spec.json`), 'utf8'));
		assert.ok(spec.name.en.includes(`after ${u} updates`), `ck/${u} holds the weights after ${u} updates`);
		assert.ok(rt.sources[ckSource(u)].model, `ck${u} loads`);
	}
	const tau = (src: string) => 1 / scaleIn(rt.sources[src].model);
	assert.ok(close(tau(ckSource(0)), rt.num('fact:tauFirst'), 1e-6) && close(tau(ckSource(last)), rt.num('fact:tauLast'), 1e-6), 'τ is a weight of the checkpoints');
	assert.ok(close(tau('clip'), rt.num('fact:tauClip'), 1e-6), 'the shipped TinyCLIP learned its own τ');
	assert.equal(Number(rt.num('fact:tauFirst').toPrecision(3)), 0.07, 'CLIP’s starting temperature');
	assert.ok(rt.num('fact:tauLast') < rt.num('fact:tauFirst'));
	assert.equal(rt.num('fact:batches6'), Math.floor(rt.num('fact:testN') / 6), 'the test pairs in batches of 6');
	assert.ok(rt.num('fact:acc6First') < 0.2 && rt.num('fact:acc6Last') > 0.9);
	rt.setInput(START);
	const f = await frames(rt, 'progress');
	assert.equal(f.tape.length, CKPTS.length, 'one frame per checkpoint');
	const m0 = f.tape.frame(0).metrics!, mL = f.tape.frame(f.tape.length - 1).metrics!;
	assert.ok(m0.diag - m0.off < 0.1 && mL.diag - mL.off > 0.5, 'from noise to a diagonal on the starting batch');
	rt.close();
});

test('checkpoints (kind A parity): every checkpoint’s TS forward pass matches PyTorch within 1e-4, layer by layer', async () => {
	const rt = await createRuntime(await pkgP, opts);
	for (const u of CKPTS) {
		const model = rt.sources[ckSource(u)].model!;
		const cases = await readReference(fsFetch, join(SITE, `static/models/contrastive-learning/ck/${u}`));
		assert.ok(cases.length >= 3, `ck/${u}: reference cases`);
		let max = 0;
		for (const c of cases) {
			const tr = model.forward(c.inputs as Record<string, AnyTensor>, { record: 'all' });
			for (const id of c.order) {
				const ref = c.tensors[id], mine = tr.t[id];
				if (!ref) continue;
				assert.ok(mine && mine.data.length === ref.data.length, `ck/${u} ${c.id}: layer ${id}`);
				for (let k = 0; k < ref.data.length; k++) max = Math.max(max, Math.abs(mine.data[k] - ref.data[k]));
			}
		}
		assert.ok(max <= 1e-4, `ck/${u}: max |diff| ${max}`);
	}
	rt.close();
});

test('step "simclr": two views of each drawing side by side, and the loss over all views', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const f = await frames(rt, 'simclr');
		const v = f.m('views'), l = f.m('loss');
		const M = v.M;
		assert.equal(M, 2 * pairsOf(rt.input).length, `${fx.id}: two views per drawing`);
		assert.equal(l.chance, Math.log(M - 1));
		const P = f.state('loss').P;
		let loss = 0;
		for (let k = 0; k < M; k++) {
			assert.equal(P.data[k * M + k], 0, 'a view is never its own partner');
			assert.ok(close(Array.from(P.data.subarray(k * M, (k + 1) * M)).reduce((a, b) => a + b, 0), 1, 1e-9));
			loss -= Math.log(P.data[k * M + partner(k)]) / M;
		}
		assert.ok(close(loss, l.loss, 1e-9), `${fx.id}: the mean of −log the partner’s share`);
		assert.ok(v.negMax <= 1 && v.pos0 <= 1 + 1e-6);
	}, opts);
});

test('step "limits": on the starting batch the largest wrong share is a real false negative (“a circle” for the big circle)', async () => {
	const rt = await createRuntime(await pkgP, opts);
	rt.setInput(START);
	const f = await frames(rt, 'nce');
	const P = f.state('softmax').P, G = f.state('grad').G;
	const n = P.shape[0];
	const best = offMax(P.data, n);
	const pairs = pairsOf(START);
	assert.deepEqual([best.i, best.j], [0, 1], 'drawing 0 with caption 1');
	assert.equal(pairs[0].from, 'circle', 'drawing 0 is the big circle');
	assert.equal(pairs[1].caption, 'a circle', 'caption 1 fits it too');
	assert.ok(G.data[best.i * n + best.j] > 0, 'and the loss still pushes that score down');
	rt.close();
});

test('step "invariance": the captions name the place, and the image tower keeps it on the starting batch', async () => {
	for (const w of ['left', 'right', 'top', 'bottom', 'middle']) assert.ok(VOCAB.includes(w), w);
	const rt = await createRuntime(await pkgP, opts);
	rt.setInput(START);
	const f = await frames(rt, 'simclr');
	const m = f.m('shift');
	assert.ok(m.shift0 < m.jitter0 - 0.05, `moved across ${m.shift0} vs distorted ${m.jitter0}`);
	rt.close();
});

test('page: goals, credits and the concept catalog block', async () => {
	assert.equal(manifest.page.goals.length, 4);
	assert.ok(manifest.credits.some((c) => c.url.includes('2103.00020')) && manifest.credits.some((c) => c.url.includes('2002.05709')) && manifest.credits.some((c) => c.url.includes('1807.03748')));
	assert.equal(manifest.sources!.clip.kind === 'trace' && manifest.sources!.clip.assets, 'models/tinyclip', 'the shipped TinyCLIP, read only');
});
