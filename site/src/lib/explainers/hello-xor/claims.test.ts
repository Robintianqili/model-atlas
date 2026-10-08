// hello-xor claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson that is not computed
// live is asserted here, over every fixture (adapter edge fixtures and presets) through withFixtures. `atlas check`
// runs this file (AK408 on failure). Run alone: node --test src/lib/explainers/hello-xor/claims.test.ts
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
import { xor } from './trainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);

test('step "stuck": from seed 1 the 2-2-1 network stops with a loss above 0.1 and at most 3 points right; a third hidden unit fixes this start', async () => {
	const pkg = (await loadPackage('hello-xor', SITE)).pkg;
	const p = trainingProcess(xor);
	const two = last(record(p, null, { hidden: 2 }, { seed: 1 }));
	const longer = last(record(p, null, { hidden: 2, steps: 3000 }, { seed: 1 }));
	const three = last(record(p, null, { hidden: 3 }, { seed: 1 }));
	assert.ok(two.metrics!.loss > 0.1 && two.metrics!.correct <= 3, `seed 1, 2 hidden: loss ${two.metrics!.loss}, correct ${two.metrics!.correct}`);
	assert.ok(longer.metrics!.loss > 0.1, `"more steps do not help": 3,000 steps still give ${longer.metrics!.loss}`);
	assert.equal(three.metrics!.correct, 4, 'seed 1 with 3 hidden units gets all 4 points right');
	// the lesson's own source says the same on every fixture (the learner's point does not change training)
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('stuck', 'last');
			const f = last(rt.sources.stuck.tape!);
			assert.ok(f.metrics!.loss > 0.1 && f.metrics!.correct < 4);
		},
		opts
	);
});

test('step "train": the default run (seed 5) gets all four points right', async () => {
	const pkg = (await loadPackage('hello-xor', SITE)).pkg;
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('train', 'last');
			const f = last(rt.sources.train.tape!);
			assert.equal(f.metrics!.correct, 4);
			assert.ok(f.metrics!.loss < 1e-6, `loss ${f.metrics!.loss}`);
		},
		opts
	);
});

test('steps "sgd-forward" / "sgd-update": the slowed-down step starts wrong, lowers the loss, and is plain SGD (old - lr * g = new)', async () => {
	const pkg = (await loadPackage('hello-xor', SITE)).pkg;
	await withFixtures(
		pkg,
		async (rt) => {
			await rt.ensure('step1', 'last');
			const t = rt.sources.step1.tape!;
			const frames = Array.from({ length: t.length }, (_, i) => t.frame(i));
			const up = frames.find((f) => f.phase === 'update')!.metrics!;
			assert.ok(Math.abs(up.old - up.lr * up.g - up.new) < 1e-15, 'the narrated update rule is the one that ran');
			assert.ok(up.lossAfter < up.loss, `loss ${up.loss} -> ${up.lossAfter}`);
			const wrong = frames.filter((f) => f.phase === 'forward' && Math.abs(f.metrics!.yhat - f.metrics!.target) > 0.1);
			assert.ok(wrong.length > 0, '"its random starting weights" get some point wrong');
		},
		opts
	);
});

test('step "train": "the network you met first was trained the same way in PyTorch" (Adam, lr 0.05, full batch, mean squared error)', async () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'hello-xor', 'spec.json'), 'utf8')) as { training: { optimizer: string; steps: number } };
	const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'hello-xor', 'facts.json'), 'utf8')) as { facts: Record<string, { value: number }> };
	assert.equal(spec.training.optimizer, `Adam lr ${xor.defaults.lr}, full batch, mean squared error`);
	assert.equal(xor.defaults.optimizer, 'adam');
	assert.equal(xor.defaults.batch, 4, 'full batch: all four points every step');
	assert.equal(facts.facts.trainSteps.value, spec.training.steps);
});

test('step "answer": the exported network puts every XOR point on the right side of 0.5', async () => {
	const pkg = (await loadPackage('hello-xor', SITE)).pkg;
	const seen: string[] = [];
	await withFixtures(
		pkg,
		async (rt, fx) => {
			if (fx.kind !== 'preset') return;
			const y = rt.sources.fwd.trace!.t.fc2.data[0];
			const x = rt.pre!.tensors.input.data;
			const target = (x[0] > 0.5) !== (x[1] > 0.5) ? 1 : 0;
			assert.equal(y > 0.5 ? 1 : 0, target, `${fx.id}: output ${y}`);
			seen.push(fx.id);
		},
		opts
	);
	assert.deepEqual(seen.sort(), ['preset:p00', 'preset:p01', 'preset:p10', 'preset:p11']);
});
