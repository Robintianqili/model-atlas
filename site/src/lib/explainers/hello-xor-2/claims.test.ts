// hello-xor-2 claim tests: every factual sentence of the lesson that is not computed live is asserted here, over every
// fixture (adapter edge fixtures and presets) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/hello-xor-2/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { xorSigmoid } from './trainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);

test('step "stuck": from seed 2 the 2-2-1 sigmoid network stops with a loss above 0.1 and at most 3 points right; more steps do not help; a third hidden unit fixes this start', async () => {
	const pkg = (await loadPackage('hello-xor-2', SITE)).pkg;
	const p = trainingProcess(xorSigmoid);
	const two = last(record(p, null, { hidden: 2 }, { seed: 2 }));
	const longer = last(record(p, null, { hidden: 2, steps: 3000 }, { seed: 2 }));
	const three = last(record(p, null, { hidden: 3 }, { seed: 2 }));
	assert.ok(two.metrics!.loss > 0.1 && two.metrics!.correct <= 3, `seed 2, 2 hidden: loss ${two.metrics!.loss}, correct ${two.metrics!.correct}`);
	assert.ok(longer.metrics!.loss > 0.1, `"more steps do not help": 3,000 steps still give ${longer.metrics!.loss}`);
	assert.equal(three.metrics!.correct, 4, 'seed 2 with 3 hidden units gets all 4 points right');
	// the lesson's own source says the same on every fixture (the learner's point does not change training)
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('stuck', 'last');
			const f = last(rt.sources.stuck.tape!);
			assert.equal(rt.sources.stuck.tape!.seed, 2);
			assert.ok(f.metrics!.loss > 0.1 && f.metrics!.correct < 4);
		},
		opts
	);
});

test('step "train": the default run (seed 1) gets all four points right', async () => {
	const pkg = (await loadPackage('hello-xor-2', SITE)).pkg;
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('train', 'last');
			const f = last(rt.sources.train.tape!);
			assert.equal(f.metrics!.correct, 4);
			assert.ok(f.metrics!.loss < 0.05, `loss ${f.metrics!.loss}`);
		},
		opts
	);
});

test('step "sgd-update": the slowed-down step starts wrong, lowers the loss, is plain SGD (old - lr * g = new), and the output gradient is (p - y) / 4', async () => {
	const pkg = (await loadPackage('hello-xor-2', SITE)).pkg;
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('step1', 'last');
			const t = rt.sources.step1.tape!;
			const frames = Array.from({ length: t.length }, (_, i) => t.frame(i));
			const up = frames.find((f) => f.phase === 'update')!.metrics!;
			assert.ok(Math.abs(up.old - up.lr * up.g - up.new) < 1e-15, 'the narrated update rule is the one that ran');
			assert.ok(up.lossAfter < up.loss, `loss ${up.loss} -> ${up.lossAfter}`);
			const fwd = frames.filter((f) => f.phase === 'forward');
			assert.equal(fwd.length, 4);
			const wrong = fwd.filter((f) => Math.abs(f.metrics!.yhat - f.metrics!.target) > 0.1);
			assert.ok(wrong.length > 0, '"its random starting weights" get some point wrong');
			// "the gradient at each output score is the probability minus the target, divided by 4": the output bias
			// gradient is the sum of those, and a finite difference of the loss in the output bias agrees
			const gb2 = (frames.find((f) => f.phase === 'backward')!.state.gb2 as { data: ArrayLike<number> }).data[0];
			const sum = fwd.reduce((s, f) => s + (f.metrics!.yhat - f.metrics!.target) / 4, 0);
			assert.ok(Math.abs(gb2 - sum) < 1e-15, `gb2 ${gb2} vs sum of (p - y) / 4 = ${sum}`);
			const bce = (z: number, y: number) => {
				const p = 1 / (1 + Math.exp(-z));
				return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
			};
			const L = (db: number) => fwd.reduce((s, f) => s + bce(f.metrics!.z + db, f.metrics!.target), 0) / 4;
			const h = 1e-6;
			assert.ok(Math.abs((L(h) - L(-h)) / (2 * h) - gb2) < 1e-8, 'finite difference agrees with (p - y) / 4');
		},
		opts
	);
});

test('step "train": "the network you met first was trained the same way in PyTorch" (Adam, lr 0.05, full batch, binary cross-entropy)', async () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'hello-xor-2', 'spec.json'), 'utf8')) as { training: { optimizer: string; steps: number } };
	const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'hello-xor-2', 'facts.json'), 'utf8')) as { facts: Record<string, { value: number }> };
	assert.equal(spec.training.optimizer, `Adam lr ${xorSigmoid.defaults.lr}, full batch, binary cross-entropy`);
	assert.equal(xorSigmoid.defaults.optimizer, 'adam');
	assert.equal(xorSigmoid.defaults.batch, 4, 'full batch: all four points every step');
	assert.equal(facts.facts.trainSteps.value, spec.training.steps);
});

test('step "answer": the exported network puts every XOR point on the right side of one half, and its output is the sigmoid of the score', async () => {
	const pkg = (await loadPackage('hello-xor-2', SITE)).pkg;
	const seen: string[] = [];
	await withFixtures(
		pkg,
		async (rt, fx) => {
			const tr = rt.sources.fwd.trace!;
			const z = tr.t.fc2.data[0];
			const p = tr.t.out.data[0];
			assert.ok(Math.abs(p - 1 / (1 + Math.exp(-z))) < 1e-6, `${fx.id}: out ${p} is sigmoid(${z})`);
			if (fx.kind !== 'preset') return;
			const x = rt.pre!.tensors.input.data;
			const target = (x[0] > 0.5) !== (x[1] > 0.5) ? 1 : 0;
			assert.equal(p > 0.5 ? 1 : 0, target, `${fx.id}: output ${p}`);
			seen.push(fx.id);
		},
		opts
	);
	assert.deepEqual(seen.sort(), ['preset:p00', 'preset:p01', 'preset:p10', 'preset:p11']);
});
