// ddpm-unet claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/ddpm-unet/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadTape } from '../../atlas/process/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import { FEW, FILM, SCHED, SEEDS, T, mse, modelInput, schedule, withExtra } from './ddpm.ts';
import { gallery, sample, type ModelIn } from './process.ts';
import { traceOf } from './views/pics/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('ddpm-unet', SITE).then((r) => r.pkg);
const CLASSES = ['star', 'umbrella', 'house'];
const x = (tp: Tape, i: number) => (tp.state(i) as { x: Tensor }).x.data;
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
async function modelIn(seed: number): Promise<ModelIn> {
	const rt = await createRuntime(await pkgP, opts);
	const m = modelInput(rt)!;
	rt.close();
	return { ...m, seed } as ModelIn;
}
function galleryTape(): Tape {
	const stem = join(SITE, 'static', 'models', 'ddpm-unet', 'tapes', 'gallery.tape');
	const h = JSON.parse(readFileSync(`${stem}.json`, 'utf8'));
	const b = readFileSync(`${stem}.bin`);
	return loadTape(h, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), gallery);
}

test('steps "input" and "limits", the page: three kinds, 36,000 drawings, the parameter count, no stored drawings, no text input', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.deepEqual((rt.spec!.kinds as { id: string }[]).map((k) => k.id), CLASSES, 'stars, umbrellas and houses; not a cat');
	assert.equal(rt.num('fact:classesN'), CLASSES.length);
	assert.equal(rt.num('fact:trainN'), 12000 * CLASSES.length);
	assert.equal(rt.num('fact:params'), rt.spec!.params, 'learned numbers (the ᾱ table is an attribute of q_sample, not a weight)');
	assert.equal(manifest.catalog!.params!.en, rt.num('fact:params').toLocaleString('en-US'));
	assert.equal(rt.spec!.tensors.reduce((s, t) => s + t.length, 0), rt.spec!.params, 'the weights are the U-Net’s numbers: no drawings');
	assert.deepEqual(rt.spec!.inputs!.map((i) => i.name), ['input', 'eps', 't'], 'quiz: the network reads x_t (from the drawing and the noise) and t, no text');
	const after = rt.spec!.layers.slice(rt.spec!.layers.findIndex((l) => l.id === 'xt') + 1);
	assert.ok(after.every((l) => !l.in.includes('input') && !l.in.includes('eps')), 'step "job": the U-Net is shown only x_t and t');
	let d = SITE;
	for (let i = 0; i < 6 && !existsSync(join(d, 'pipeline', 'ddpm-unet', 'train.py')); i++) d = join(d, '..');
	const train = readFileSync(join(d, 'pipeline', 'ddpm-unet', 'train.py'), 'utf8');
	assert.ok(train.includes('F.mse_loss(m(') && train.includes('torch.randint(1, T + 1'), 'predict: random steps, mean squared error on the noise');
	rt.close();
});

test('steps "mix" and "forward": the schedule is the exported one, ᾱ only falls, √ᾱ² + √(1 − ᾱ)² = 1, stepwise noising = the one-jump formula', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const tab = rt.spec!.layers.find((l) => l.id === 'xt')!.params!.abar as number[];
	assert.equal(tab.length, T);
	for (let i = 0; i < T; i++) assert.ok(Math.abs(tab[i] - SCHED.abar[i]) <= 1e-7 * SCHED.abar[i] + 1e-12, `ᾱ_${i + 1}`);
	assert.equal(rt.num('fact:beta1'), SCHED.beta[0]);
	assert.equal(rt.num('fact:betaT'), SCHED.beta[T - 1]);
	for (let i = 1; i < T; i++) assert.ok(SCHED.abar[i] < SCHED.abar[i - 1], 'the drawing’s weight keeps falling');
	assert.ok(Math.sqrt(SCHED.abar[T - 1]) < 0.01, 'at t = T less than 1% of the drawing is left: nothing to see');
	// x_t = √(1 − β_t)·x_{t−1} + √β_t·z, step by step: the drawing's factor and the noise variance equal √ᾱ_t and 1 − ᾱ_t
	let f = 1, v = 0;
	for (let i = 0; i < T; i++) {
		f *= Math.sqrt(1 - SCHED.beta[i]);
		v = (1 - SCHED.beta[i]) * v + SCHED.beta[i];
		assert.ok(Math.abs(f - Math.sqrt(SCHED.abar[i])) < 1e-12 && Math.abs(v - (1 - SCHED.abar[i])) < 1e-12, `step ${i + 1}`);
	}
	assert.deepEqual(Array.from(schedule().abar), Array.from(SCHED.abar));
	rt.close();
});

test('steps "mix" and "job": x_t mixes the stretched drawing and ε with weights whose squares add to 1, and 2x₀ − 1 = (x_t − √(1 − ᾱ)·ε) / √ᾱ', async () => {
	await withFixtures(await pkgP, (rt) => {
		const c = traceOf({ rt });
		if (!c) return assert.ok(rt.pre?.empty, 'only an empty pad has no trace');
		assert.ok(Math.abs(c.a * c.a + c.b * c.b - 1) < 1e-6);
		for (let i = 0; i < c.x0.length; i += 37) assert.ok(Math.abs((c.xt[i] - c.b * c.eps[i]) / c.a - (2 * c.x0[i] - 1)) < 1e-4, `pixel ${i}`);
		assert.ok(c.x0[c.i] === Math.max(...c.x0), 'the marked pixel has the most ink');
	}, opts);
});

test('steps "seeds" and "pick": the precomputed gallery equals the live run of the same seed; 12 seeds give 12 different drawings', async () => {
	const g = galleryTape();
	assert.equal(g.length, SEEDS * (T / FILM + 1));
	for (const seed of [0, 5, 11]) {
		const run = record(sample, await modelIn(seed), sample.defaults, { seed: 1 });
		for (let k = 0; k <= T / FILM; k++) {
			const i = g.find({ step: seed, sub: k });
			assert.equal(g.frame(i).metrics!.t, T - k * FILM);
			assert.ok(maxDiff(x(g, i), x(run, k * FILM)) < 1e-4, `seed ${seed}, t = ${T - k * FILM}`);
		}
	}
	const finals = Array.from({ length: SEEDS }, (_, s) => x(g, g.find({ step: s, sub: T / FILM })));
	for (let a = 0; a < SEEDS; a++) for (let b = a + 1; b < SEEDS; b++) assert.ok(mse(finals[a], finals[b]) > 0.01, `seeds ${a} and ${b} differ`);
	const same = record(sample, await modelIn(3), { ...sample.defaults, stride: FEW }, { seed: 1 });
	const again = record(sample, await modelIn(3), { ...sample.defaults, stride: FEW }, { seed: 1 });
	assert.deepEqual(Array.from(x(same, same.length - 1)), Array.from(x(again, again.length - 1)), 'the same seed gives the same noise and the same drawing');
});

test('step "few": ten big steps remove the same noise in all as the hundred small ones (each α of a big step = ten small α)', async () => {
	const few = record(sample, await modelIn(0), { ...sample.defaults, stride: FEW }, { seed: 1 });
	assert.equal(few.length - 1, T / FEW);
	let prod = 1;
	for (let i = 1; i < few.length; i++) {
		const m = few.frame(i).metrics!;
		let ten = 1;
		for (let t = m.t; t > m.tNext; t--) ten *= 1 - SCHED.beta[t - 1];
		assert.ok(Math.abs(m.alpha - ten) < 1e-12, `t = ${m.t}`);
		prod *= m.alpha;
	}
	assert.ok(Math.abs(prod - SCHED.abar[T - 1]) < 1e-15);
});

test('step "unet": pooling halves the maps twice (28 → 14 → 7), two skip connections concatenate, t enters every time-biased block', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const L = rt.spec!.layers;
	assert.deepEqual(L.filter((l) => l.op === 'maxpool2d').map((l) => l.outShape.slice(1)), [[14, 14], [7, 7]]);
	assert.deepEqual(L.filter((l) => l.op === 'concat').map((l) => l.in), [['up.up', 'down.silu'], ['top.up', 'in.silu']]);
	assert.deepEqual(L.filter((l) => l.op === 'add').map((l) => l.in[1].endsWith('.time')), [true, true, true], 'the time bias is added in three blocks, no residual add');
	rt.close();
});

test('step "pick": the task is done only for a seed other than 0', async () => {
	const step = (await pkgP).lesson!.steps.find((s) => 'id' in s && s.id === 'pick') as { task: { done: (rt: unknown, at: unknown) => boolean } };
	assert.equal(step.task.done({ input: { strokes: [] } }, {}), false);
	assert.equal(step.task.done({ input: withExtra(null, { seed: 4 }) }, {}), true);
});
