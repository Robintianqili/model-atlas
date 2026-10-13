// lora claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page (concept.ts)
// and the manifest that is not computed live is asserted here, over every fixture (the four ranks and the adapter's edge
// fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/lora/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { Adam } from '../../atlas/autograd/optim.ts';
import { param } from '../../atlas/autograd/var.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { BASE_LOSS, D, GOAL, GPT2_PARAMS, MIN_RANK, N, R_MAX, SPECTRUM, TINYGPT, WORLD, floorOf, gpt2Lora, loraCount, memoryOf, rankOf } from './data.ts';
import { lora, ranks } from './process.ts';
import { countRows } from './views/counter/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'lora';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const sq = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v * v, 0);
type T = { shape: number[]; data: Float64Array };

/** The singular values of a D × D matrix: square roots of the eigenvalues of MᵀM (cyclic Jacobi), largest first. */
function singularValues(M: Float64Array): number[] {
	const A = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => {
		let s = 0;
		for (let k = 0; k < D; k++) s += M[k * D + i] * M[k * D + j];
		return s;
	}));
	for (let sweep = 0; sweep < 60; sweep++)
		for (let p = 0; p < D; p++)
			for (let q = p + 1; q < D; q++) {
				if (Math.abs(A[p][q]) < 1e-300) continue;
				const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
				const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
				const c = 1 / Math.sqrt(t * t + 1);
				const s = t * c;
				for (let k = 0; k < D; k++) {
					const akp = A[k][p];
					const akq = A[k][q];
					A[k][p] = c * akp - s * akq;
					A[k][q] = s * akp + c * akq;
				}
				for (let k = 0; k < D; k++) {
					const apk = A[p][k];
					const aqk = A[q][k];
					A[p][k] = c * apk - s * aqk;
					A[q][k] = s * apk + c * aqk;
				}
			}
	return A.map((row, i) => Math.sqrt(Math.max(0, row[i]))).sort((a, b) => b - a);
}

test('the toy world: XᵀX = 64·I, so the loss of a change ΔW is ‖ΔW − ΔW*‖²/16; the needed change has the strengths of SPECTRUM (two strong ones, 2 and 1.2, then a weak tail)', () => {
	const { X } = WORLD;
	for (let i = 0; i < D; i++)
		for (let j = 0; j < D; j++) {
			let s = 0;
			for (let n = 0; n < N; n++) s += X[n * D + i] * X[n * D + j];
			assert.ok(Math.abs(s - (i === j ? N : 0)) < 1e-11, `XᵀX[${i}, ${j}] = ${s}`);
		}
	const sv = singularValues(WORLD.dWstar);
	sv.forEach((v, k) => assert.ok(Math.abs(v - SPECTRUM[k]) < 1e-9, `s${k} ${v} vs ${SPECTRUM[k]}`));
	assert.equal(SPECTRUM[0], 2);
	assert.equal(SPECTRUM[1], 1.2);
	for (let k = 2; k < D; k++) assert.ok(SPECTRUM[k] <= 0.3, 'a weak tail');
	// step "start": the new task's answers are the outputs of the changed layer W0 + ΔW* on the 64 inputs;
	// "the third panel is what is still missing" / the start loss: the base layer's error on the new task, computed directly
	let e = 0;
	for (let n = 0; n < N; n++)
		for (let o = 0; o < D; o++) {
			let y = 0;
			let yChanged = 0;
			for (let i = 0; i < D; i++) {
				y += X[n * D + i] * WORLD.W0[o * D + i];
				yChanged += X[n * D + i] * (WORLD.W0[o * D + i] + WORLD.dWstar[o * D + i]);
			}
			assert.ok(Math.abs(yChanged - WORLD.Y[n * D + o]) < 1e-12, `answer ${n}, ${o}`);
			e += (y - WORLD.Y[n * D + o]) ** 2;
		}
	assert.ok(close(e / (N * D), BASE_LOSS, 1e-12), `base ${e / (N * D)} vs ${BASE_LOSS}`);
	assert.ok(close(BASE_LOSS, sq(WORLD.dWstar) / D, 1e-12));
});

test('every frame of the lora source: loss = ‖ΔW − ΔW*‖²/16, never below the rank\'s floor (Eckart–Young: "no amount of training gets the loss below"), and ΔW = (α/r)·B·A', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('fit', 'last');
			const t = rt.sources.fit.tape!;
			const r = rankOf(rt.input);
			for (let i = 0; i < t.length; i++) {
				const m = t.frame(i).metrics!;
				const s = t.state(i) as { A: T; B: T; dW: T };
				assert.equal(m.r, r);
				const dist = sq(s.dW.data.map((v, k) => v - WORLD.dWstar[k])) / D;
				assert.ok(close(m.loss, dist, 1e-9), `frame ${i}: loss ${m.loss} vs ${dist}`);
				assert.ok(m.loss >= floorOf(r) - 1e-12, `frame ${i}: loss ${m.loss} below the floor ${floorOf(r)}`);
				for (let o = 0; o < D; o++)
					for (let j = 0; j < D; j++) {
						let v = 0;
						for (let k = 0; k < r; k++) v += s.B.data[o * r + k] * s.A.data[k * D + j];
						assert.ok(close(s.dW.data[o * D + j], m.scale * v, 1e-12));
					}
			}
			// "after n updates" counts updates: the last frame is the run's length
			assert.equal(last(t).metrics!.step, t.length - 1);
		},
		opts
	);
});

test('steps "predict", "start": B starts as all zeros, so ΔW = 0, W = W0 and the loss is the base model\'s; the merged layer gives the same outputs as the two paths (pitfall 3)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const i = await rt.ensure('fit', { phase: 'start' });
			const t = rt.sources.fit.tape!;
			const s = t.state(i) as { A: T; B: T; dW: T };
			assert.ok(s.B.data.every((v) => v === 0) && s.dW.data.every((v) => v === 0), 'B and ΔW are all zeros');
			assert.ok(s.A.data.some((v) => v !== 0) && s.A.data.every((v) => Math.abs(v) <= 0.25), 'A: small random numbers');
			assert.ok(close(t.frame(i).metrics!.loss, BASE_LOSS, 1e-12));
			// pitfall 3 (merge): (W0 + ΔW)·x = W0·x + (α/r)·B·(A·x) for every input, after training
			await rt.ensure('fit', 'last');
			const e = t.state(t.length - 1) as { A: T; B: T; dW: T };
			const m = last(t).metrics!;
			const r = m.r;
			for (let n = 0; n < N; n += 7) {
				const x = WORLD.X.subarray(n * D, n * D + D);
				const ax = Array.from({ length: r }, (_, k) => x.reduce((a, v, j) => a + e.A.data[k * D + j] * v, 0));
				for (let o = 0; o < D; o++) {
					const merged = x.reduce((a, v, j) => a + (WORLD.W0[o * D + j] + e.dW.data[o * D + j]) * v, 0);
					const paths = x.reduce((a, v, j) => a + WORLD.W0[o * D + j] * v, 0) + m.scale * ax.reduce((a, v, k) => a + e.B.data[o * r + k] * v, 0);
					assert.ok(Math.abs(merged - paths) < 1e-12, `${merged} vs ${paths}`);
				}
			}
		},
		opts
	);
});

test('step "first" and pitfall 2: A\'s gradient is exactly 0 while B = 0, so update 1 leaves A unchanged; B\'s is not, and each of its numbers moves by about the learning rate; from update 2 on both learn', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('fit', { phase: 'step', step: 2 });
			const t = rt.sources.fit.tape!;
			const m0 = t.frame(0).metrics!;
			const m1 = t.frame(1).metrics!;
			const m2 = t.frame(2).metrics!;
			assert.equal(m0.gA, 0);
			assert.ok(m0.gB > 0.1, `B's gradient ${m0.gB}`);
			assert.equal(m1.dA, 0);
			assert.ok(Math.abs(m1.dB - m1.lr) < 0.01 * m1.lr, `B moved ${m1.dB}, lr ${m1.lr}`);
			assert.ok(m1.loss < m0.loss);
			assert.ok(m2.dA > 0 && m2.dB > 0, 'both move in update 2');
		},
		opts
	);
});

test('step "train": the loss is what is missing, so 1 − loss/base is the share of ΔW* captured; after training it is at least 70% but below 100%', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('fit', 'last');
			const m = last(rt.sources.fit.tape!).metrics!;
			const covered = 1 - m.loss / m.base;
			assert.ok(covered >= 0.7 && covered < 1, `covered ${covered}`);
		},
		opts
	);
});

test('steps "spectrum", "task": the floors; rank 2 is the smallest rank under the goal, live and analytically; every trained run ends within 2% of its floor up to rank 3', async () => {
	assert.equal(MIN_RANK, 2);
	assert.ok(floorOf(1) > GOAL && floorOf(2) < GOAL);
	const r1 = last(record(lora, { r: 1 }, lora.defaults, { seed: 1 })).metrics!.loss;
	const r2 = last(record(lora, { r: 2 }, lora.defaults, { seed: 1 })).metrics!.loss;
	assert.ok(r1 > GOAL && r2 < GOAL, `trained: rank 1 ${r1}, rank 2 ${r2}`);
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('fit', 'last');
			const m = last(rt.sources.fit.tape!).metrics!;
			if (m.r <= 3) assert.ok(m.loss < floorOf(m.r) * 1.02, `rank ${m.r}: ${m.loss} vs floor ${floorOf(m.r)}`);
			assert.equal(m.trainable, loraCount(m.r, D, D));
		},
		opts
	);
});

test('compare, step "limits" and pitfall 1: the biggest drop is from rank 1 to 2; rank 1 stays at its floor while full fine-tuning gets near 0; rank 8 of a 16 × 16 layer trains as many numbers as full fine-tuning; with an even spectrum every small rank is far behind', async () => {
	const t = record(ranks, null, ranks.defaults, { seed: 1 });
	const m = last(t).metrics!;
	assert.ok(m.l1 - m.l2 > 3 * (m.l2 - m.l8), `drops ${m.l1 - m.l2} vs ${m.l2 - m.l8}`);
	assert.ok(m.l1 > m.l2 && m.l2 > m.l4 && m.l4 > m.l8 && m.l8 > m.lfull);
	assert.ok(close(m.l1, floorOf(1), 1e-4), `rank 1 ${m.l1} vs ${floorOf(1)}`);
	assert.ok(m.lfull < 1e-4, `full ${m.lfull}`);
	for (const k of ['l1', 'l2', 'l4', 'l8', 'lfull']) assert.ok(close(t.frame(0).metrics![k], BASE_LOSS, 1e-12), 'every run starts at the base loss');
	assert.equal(loraCount(8, D, D), D * D);
	// the lesson's rank-2 run is the comparison's rank-2 run
	assert.ok(close(last(record(lora, { r: 2 }, lora.defaults, { seed: 1 })).metrics!.loss, m.l2, 1e-12));
	// pitfall 1: past rank 2 the further drop is small next to the step from 1 to 2
	assert.ok(floorOf(2) - floorOf(R_MAX) < 0.2 * (floorOf(1) - floorOf(2)));
	// "a change with many strong directions": with 16 equal strengths, rank r keeps only r/16 of it
	for (let r = 1; r <= 4; r++) assert.ok((D - r) / D >= 0.75);
});

test('steps "count", "scale" and the quiz: r·(d_in + d_out); GPT-2 small has 124,439,808 parameters; TinyGPT\'s MLP layer is 304 × 76; the share shrinks as the layers grow; 72 matrices', () => {
	assert.equal(loraCount(4, 1000, 1000), 8000);
	assert.equal(GPT2_PARAMS, 124_439_808);
	const spec = JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/spec.json'), 'utf8')) as { dModel: number; dMLP: number };
	assert.equal(spec.dModel, TINYGPT.d);
	assert.equal(spec.dMLP, TINYGPT.dMLP);
	for (let r = 1; r <= R_MAX; r++) {
		const rows = countRows(r);
		for (let k = 1; k < rows.length; k++) assert.ok(rows[k].share < rows[k - 1].share, `rank ${r}: ${rows[k].id}`);
		assert.equal(gpt2Lora(r), r * 12 * (4 * 1536 + 2 * 3840), '12 blocks x (4 attention + 2 MLP) = 72 matrices');
	}
});

test('step "memory", the misconception and concept "real": Adam keeps two numbers per trainable number; at rank 8 LoRA trains about 1% of GPT-2 small but needs about a quarter of the memory; the frozen weights dominate', () => {
	const p = param('p', { shape: [3], data: [1, 2, 3] });
	p.grad = param('g', { shape: [3], data: [0.1, -0.2, 0.3] });
	const adam = new Adam({ p }, { lr: 0.01 });
	adam.step();
	const st = adam.state().p;
	assert.equal(st.m!.data.length, 3);
	assert.equal(st.v!.data.length, 3);
	const share = gpt2Lora(8) / GPT2_PARAMS;
	assert.ok(share > 0.009 && share < 0.012, `share ${share}`);
	for (let r = 1; r <= R_MAX; r++) {
		const m = memoryOf(r);
		const f = m.lora.total / m.full.total;
		assert.ok(f > 0.24 && f < 0.28, `rank ${r}: ${f}`);
		assert.ok(m.lora.weights > 0.9 * m.lora.total, 'the weights are most of the LoRA bar');
		assert.ok(m.lora.weights >= m.full.weights && m.lora.weights < 1.02 * m.full.weights, 'about as long as the full bar\'s weights');
		assert.ok(m.qlora.total < m.lora.total);
	}
});
