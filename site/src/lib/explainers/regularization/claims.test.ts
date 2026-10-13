// regularization claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest (goals, quiz, misconception, glossary) that is not computed live is asserted here. The
// runs on the starting knobs are recorded once and shared; the learner-knob claims run over the lesson's fixtures.
// `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/regularization/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { backward, param } from '../../atlas/autograd/var.ts';
import { Adam, SGD } from '../../atlas/autograd/optim.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { DATA_SEED, EVERY, PATIENCE, POOL, START, STEPS, TARGET, bestMiss, paramCountOf, splitOf } from './data.ts';
import { RegNet, p1Of, type Params } from './net.ts';
import { VARIANTS, cnn, fit, methods } from './process.ts';
import { STARTS } from './adapters/knobs/adapter.ts';
import { CNN, shiftPixels } from './cnn.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

let M: Tape | null = null;
/** the 'methods' source of the page (input ignored, seed 1), recorded once */
const methodsTape = (): Tape => (M ??= record(methods, null, {}) as Tape);
const end = () => last(methodsTape()).metrics!;

test('the data: training and test points split the same pool, both sets balanced; the start has 40 training and 160 test points', () => {
	for (const ratio of [0.1, 0.2, 0.5, 0.9]) {
		const { train, test } = splitOf(0.3, ratio);
		assert.equal(train.n + test.n, POOL);
		for (const s of [train, test]) {
			const ones = Array.from(s.label).filter((l) => l === 1).length;
			assert.equal(ones * 2, s.n, `${s.name} at ratio ${ratio} is not balanced`);
		}
	}
	const s = splitOf(START.noise, START.ratio);
	assert.equal(s.train.n, 40);
	assert.equal(s.test.n, 160);
	assert.equal(DATA_SEED, 7);
});

test('step "split": the network never trains on the test points (moving a test point changes no training step)', () => {
	// the run on the start knobs, and the same run with every test point moved far away, give identical training losses
	const a = record(fit, { ...START }, { steps: 30 });
	const s = splitOf(START.noise, START.ratio);
	const saved = Float64Array.from(s.test.xy);
	try {
		s.test.xy.fill(9);
		const b = record(fit, { ...START }, { steps: 30 });
		for (let i = 0; i < a.length; i++) assert.equal(a.frame(i).metrics!.trainLoss, b.frame(i).metrics!.trainLoss);
	} finally {
		s.test.xy.set(saved);
	}
});

test('steps "curves", "memorize", misconception, prequestion: on the start knobs the training loss falls almost to 0 and every training point is right, while the test loss climbs back up', () => {
	const m = end();
	assert.ok(m.trainLoss < 0.01, `training loss ${m.trainLoss}`);
	assert.equal(m.trainAcc, 1);
	assert.ok(m.testAcc < 1);
	assert.ok(m.testLoss > m.best + 0.1, `test ${m.testLoss}, lowest ${m.best}`);
	assert.equal(paramCountOf(32), 1218);
	assert.equal(paramCountOf(2), 18);
});

test('step "predict-size": the small network ends with the lower test loss and the higher training loss', () => {
	const m = end();
	assert.equal(m.widthSmall, 2);
	assert.equal(m.width, 32);
	assert.ok(m.testLossSmall < m.testLoss, `small ${m.testLossSmall} vs big ${m.testLoss}`);
	assert.ok(m.trainLossSmall > m.trainLoss);
	assert.ok(m.trainLoss < 0.01);
});

test('step "early-stop" and pitfall 1: it stops PATIENCE checks after the lowest test loss, which comes long before the training loss stops falling; "full" is the full run\'s end', async () => {
	const t = record(fit, null, { stopEarly: 1 });
	const m = last(t).metrics!;
	assert.equal(m.stopped, 1);
	assert.equal(m.step, m.bestStep + PATIENCE * EVERY);
	assert.ok(close(m.full, end().testLoss), `full ${m.full} vs ${end().testLoss}`);
	const kept = t.frame(m.bestStep / EVERY).metrics!;
	assert.equal(kept.step, m.bestStep);
	assert.equal(kept.testLoss, m.best);
	// the training loss keeps falling well after the lowest test loss
	assert.ok(end().trainLoss < kept.trainLoss / 10, `training loss ${kept.trainLoss} at the lowest test loss, ${end().trainLoss} at the end`);
	assert.ok(m.bestStep < STEPS / 3);
});

test('step "l2", glossary "weight-decay": the penalty term is λ·Σw² over the weights, its slope is 2λw, and it shrinks the weights and the test loss', () => {
	const m = end();
	assert.ok(close(m.totalL2, m.trainLossL2 + m.l2L2 * m.sumSqL2, 1e-12));
	assert.ok(close(m.penaltyL2, m.l2L2 * m.sumSqL2, 1e-12));
	assert.ok(m.sumSqL2 < m.sumSq / 10 && m.testLossL2 < m.testLoss);
	// autograd: d(λ·Σw²)/dw = 2λw
	const w = param('w', { shape: [3], data: [0.5, -1.2, 2] });
	const lam = 0.003;
	backward(op.scale(op.sum(op.square(w)), lam));
	w.grad!.data.forEach((g, i) => assert.ok(close(g, 2 * lam * w.value.data[i], 1e-12)));
	// the biases are not in Σw²: the net's weights() are the three weight matrices
	const net = new RegNet(4, createRng(1));
	assert.deepEqual(
		net.weights().map((v) => v.name),
		['net.0.weight', 'net.2.weight', 'net.4.weight']
	);
});

test('step "dropout", quiz 1, pitfall 2: each unit is off with probability p, a new choice every update, kept units scaled by 1/(1 − p); testing drops nothing', () => {
	const t = record(fit, null, { every: 1, steps: 24, dropout: 0.5 });
	let off = 0;
	let all = 0;
	let sum = 0;
	let prev: Float64Array | null = null;
	for (let i = 1; i < t.length; i++) {
		const mask = (t.state(i) as { mask: Tensor }).mask.data;
		for (const v of mask) {
			assert.ok(v === 0 || v === 2, `mask value ${v}`);
			if (v === 0) off++;
			sum += v;
			all++;
		}
		if (prev) assert.ok(mask.some((v, j) => v !== prev![j]), `update ${i}: the same units again`);
		prev = Float64Array.from(mask);
	}
	assert.ok(Math.abs(off / all - 0.5) < 0.05, `share off ${off / all}`);
	assert.ok(Math.abs(sum / all - 1) < 0.1, `mean mask ${sum / all}`);
	// testing: the reported test loss is the network without masks (net.ts scores, no dropout)
	const dr = record(fit, null, { dropout: 0.5, steps: 40 });
	const s = dr.state(dr.length - 1) as { params: Params };
	const { test } = splitOf(START.noise, START.ratio);
	const p1 = p1Of(s.params, test.xy, test.n);
	let loss = 0;
	for (let i = 0; i < test.n; i++) loss -= Math.log(test.label[i] === 1 ? p1[i] : 1 - p1[i]);
	assert.ok(close(loss / test.n, last(dr).metrics!.testLoss, 1e-9));
	assert.ok(close(1 / (1 - 0.25), 4 / 3, 1e-12));
});

test('step "augment": the nudged points differ at every update, augmentation lowers the test loss here, and TinyCNN was trained with shifts, turns and stroke changes', () => {
	const t = record(fit, null, { every: 1, steps: 5, aug: 0.1 });
	let prev: Float64Array | null = null;
	for (let i = 1; i < t.length; i++) {
		const a = (t.state(i) as { aug: Tensor }).aug.data;
		if (prev) assert.ok(a.every((v, j) => v !== prev![j]));
		prev = Float64Array.from(a);
	}
	assert.ok(end().testLossAug < end().testLoss);
	const py = tinycnnFile('train.py');
	const common = tinycnnFile('common.py');
	assert.match(py, /if not a\.noaug: x = augment\(x, gen\)/);
	assert.match(common, /def augment\(x, gen, shift=0\.12, scale=\(0\.8, 1\.1\), rot_deg=12\.0, thick_p=0\.6\)/);
});

test('step "task": the start ends above the goal; each fix the hint names ends below it with a smaller gap than the start', () => {
	const gap = (raw: object) => {
		const m = last(record(fit, raw, {})).metrics!;
		return { test: m.testLoss, gap: m.testLoss - m.trainLoss };
	};
	const s = gap({ ...START });
	assert.ok(s.test >= TARGET);
	for (const id of ['weight-decay', 'dropout', 'more-data', 'small']) {
		const g = gap(STARTS.find((x) => x.id === id)!.raw);
		assert.ok(g.test < TARGET, `${id}: ${g.test}`);
		assert.ok(g.gap < s.gap, `${id}: gap ${g.gap} vs ${s.gap}`);
	}
});

test('step "too-much", pitfall 4: the largest λ underfits (small weights, both losses high, worse than λ = 0.003); no run sorts every test point, and most misses sit closer to the other moon\'s training points', () => {
	const m = end();
	assert.equal(m.l2Heavy, 0.1);
	assert.ok(m.sumSqHeavy < m.sumSqL2 / 3);
	assert.ok(m.trainLossHeavy > 0.4 && m.testLossHeavy > m.testLossL2 && m.trainLossHeavy > m.trainLossL2);
	assert.ok(m.trainAccHeavy < m.trainAccL2);
	// every run gets some test point wrong; the best one's misses
	const suffixes = ['', ...VARIANTS.map((v) => v.suffix)];
	for (const s of suffixes) assert.ok(m['testAcc' + s] < 1, `${s}: ${m['testAcc' + s]}`);
	const miss = bestMiss(m);
	const bestS = suffixes.find((s) => Math.round((1 - m['testAcc' + s]) * m.nTest) === miss)!;
	const v = VARIANTS.find((x) => x.suffix === bestS);
	const t = record(fit, null, { ...(v?.knobs ?? {}), aug: v?.aug ?? 0 });
	const params = (t.state(t.length - 1) as { params: Params }).params;
	const { train, test } = splitOf(START.noise, START.ratio);
	const p1 = p1Of(params, test.xy, test.n);
	let wrong = 0;
	let other = 0;
	for (let i = 0; i < test.n; i++) {
		if ((p1[i] > 0.5 ? 1 : 0) === test.label[i]) continue;
		wrong++;
		let bj = 0;
		let bd = Infinity;
		for (let j = 0; j < train.n; j++) {
			const d = (train.xy[2 * j] - test.xy[2 * i]) ** 2 + (train.xy[2 * j + 1] - test.xy[2 * i + 1]) ** 2;
			if (d < bd) [bd, bj] = [d, j];
		}
		if (train.label[bj] !== test.label[i]) other++;
	}
	assert.equal(wrong, miss);
	assert.ok(other * 2 > wrong, `${other} of ${wrong}`);
});

test('concept "compare": every fix ends with a lower test loss than the plain run', () => {
	const m = end();
	for (const s of ['L2', 'Drop', 'Aug', 'Small']) assert.ok(m['testLoss' + s] < m.testLoss, `${s}: ${m['testLoss' + s]} vs ${m.testLoss}`);
});

/** A file of pipeline/tinycnn/: under ATLAS_ROOT (the CLI sets it, also for a lane outside the repo, CONTRACTS "Lanes
 *  outside the repo"), else found above the site folder (the main tree's site/ or a lane in site/.work/<lane>/site/). */
function tinycnnFile(rel: string): string {
	const root = process.env.ATLAS_ROOT;
	if (root && existsSync(join(root, 'pipeline', 'tinycnn', rel))) return readFileSync(join(root, 'pipeline', 'tinycnn', rel), 'utf8');
	let dir = SITE;
	for (let k = 0; k < 6; k++) {
		const p = join(dir, 'pipeline', 'tinycnn', rel);
		if (existsSync(p)) return readFileSync(p, 'utf8');
		dir = dirname(dir);
	}
	throw new Error(`pipeline/tinycnn/${rel} not found above ${SITE}`);
}

test('concept "real": the TinyCNN numbers are its shipped run (export.py default) and its training settings', () => {
	assert.match(tinycnnFile('export.py'), /ap\.add_argument\("--ckpt", default="runs\/c8e60\/best\.pt"\)/);
	const m = JSON.parse(tinycnnFile('runs/c8e60/metrics.json'));
	assert.equal(CNN.params, m.params);
	assert.equal(CNN.epochs, m.epochs);
	assert.equal(CNN.bestEpoch, m.best_epoch);
	assert.ok(close(CNN.valAcc[CNN.bestEpoch - 1], m.val_acc, 1e-4));
	assert.ok(close(CNN.testAcc, m.test_acc, 1e-5));
	assert.equal(CNN.weightDecay, m.args.wd);
	assert.equal(m.args.noaug, false);
	m.history.forEach((h: { train_loss: number; val_acc: number }, i: number) => {
		assert.ok(close(CNN.trainLoss[i], h.train_loss, 1e-4) && close(CNN.valAcc[i], h.val_acc, 1e-4), `epoch ${i + 1}`);
	});
	// the best epoch is kept by validation accuracy (train.py), and the process marks the same epoch
	assert.match(tinycnnFile('train.py'), /if va > best\[0\]: best = /);
	const t = record(cnn, null, {});
	assert.equal(last(t).metrics!.bestEpoch, CNN.bestEpoch);
	assert.ok(close(shiftPixels, 1.68, 1e-12));
	// weight decay only on the weights (AdamW groups), as the L2 penalty here
	assert.match(tinycnnFile('train.py'), /AdamW\(\[\{"params": decay, "weight_decay": a\.wd\}, \{"params": nodecay, "weight_decay": 0\}\]/);
});

test('the lesson\'s input fixtures train and end at a finite test loss', () => {
	for (const s of STARTS) assert.ok(Number.isFinite(last(record(fit, s.raw, { steps: 20 })).metrics!.testLoss));
});

test('pitfall 5, glossary "weight-decay": with plain gradient descent the L2 penalty shrinks every weight by the same share; with Adam it does not; AdamW shrinks the weights directly', () => {
	const lam = 0.003;
	const w0 = [1, 2, -0.5];
	/** one optimizer step on the penalty λ·Σw² alone (penalty false: a loss whose gradient is 0), the new weights */
	const stepWith = (make: (named: Record<string, ReturnType<typeof param>>) => { step(): void }, penalty: boolean) => {
		const w = param('w', { shape: [3], data: w0.slice() });
		const opt = make({ w });
		backward(op.scale(op.sum(op.square(w)), penalty ? lam : 0));
		opt.step();
		return Array.from(w.value.data);
	};
	// SGD: w ← w − lr·2λw = (1 − 2·lr·λ)·w, the same share for every weight
	const sgd = stepWith((n) => new SGD(n, { lr: 0.1 }), true);
	sgd.forEach((v, i) => assert.ok(close(v / w0[i], 1 - 2 * 0.1 * lam, 1e-12)));
	// Adam with the penalty in the loss: its gradient is rescaled like any other, so the shares differ (here each
	// weight moves by about lr, whatever its size)
	const adam = stepWith((n) => new Adam(n, { lr: 0.01 }), true);
	const shares = adam.map((v, i) => 1 - v / w0[i]);
	assert.ok(Math.abs(shares[0] - shares[1]) > 1e-3, `Adam shares ${shares}`);
	// AdamW (decoupled): every weight shrinks by the same factor 1 − lr·wd, outside the gradient step
	const adamw = stepWith((n) => new Adam(n, { lr: 0.01, weightDecay: 0.05, decoupled: true }), false);
	adamw.forEach((v, i) => assert.ok(close(v / w0[i], 1 - 0.01 * 0.05, 1e-9), `AdamW ${i}: ${v}`));
	assert.match(tinycnnFile('train.py'), /torch\.optim\.AdamW\(/);
});
