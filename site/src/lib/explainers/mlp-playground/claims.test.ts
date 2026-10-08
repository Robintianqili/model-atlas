// mlp-playground claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here; per-fixture claims run through withFixtures (lesson fixtures + the adapter's edge
// fixtures). `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/mlp-playground/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { fit } from './trainer.ts';
import { lrPair, play } from './process.ts';
import manifest, { SPIRAL } from './explainer.ts';
import { dataOf, evalPoints, neuronGrids, type Params } from './net.ts';
import { EPS } from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('mlp-playground', SITE)).pkg;
const last = (t: Tape) => t.frame(t.length - 1);
const losses = (t: Tape) => Array.from({ length: t.length }, (_, i) => t.frame(i).metrics?.loss).filter((v): v is number => typeof v === 'number');
const finiteAll = (t: Tape) => {
	for (let i = 0; i < t.length; i++) {
		const f = t.frame(i) as Frame<Record<string, unknown>>;
		if (f.metrics && Object.values(f.metrics).some((v) => !Number.isFinite(v))) return false;
		for (const p of Object.values((f.state.params ?? {}) as Params)) if (p.data.some((v) => !Number.isFinite(v))) return false;
	}
	return true;
};
const src = (name: string) => manifest.sources![name] as { opt: Record<string, unknown> };

test('page code section: the PyTorch snippet has the settings of the trainer (2 -> 4 -> 4 -> 2, tanh, Adam lr 0.03, 300 steps, all points)', async () => {
	const { readFileSync } = await import('node:fs');
	const code = readFileSync(join(SITE, 'src/lib/explainers/mlp-playground/sections/Code.svelte'), 'utf8');
	assert.ok(code.includes('nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)') && code.includes('lr=0.03') && code.includes('range(300)'));
	assert.deepEqual([fit.defaults.hidden, fit.defaults.act, fit.defaults.optimizer, fit.defaults.lr, fit.defaults.steps, fit.defaults.batch], [[4, 4], 'tanh', 'adam', 0.03, 300, 0]);
});

test('page and quiz: the default network 2 -> 4 -> 4 -> 2 has 42 weights and biases', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.deepEqual([2, ...fit.defaults.hidden, 2], [2, 4, 4, 2]);
	assert.equal(m.params().reduce((s, p) => s + p.size, 0), 42);
});

test('M2: on the start data (two moons) the loss drops by at least 50% within 300 steps; on every fixture it falls', async () => {
	const t = record(play, presetRaw('moons'), { steps: 300 }, { seed: 1 });
	const l = losses(t);
	assert.ok(l[300] <= 0.5 * l[0], `moons: ${l[0]} -> ${l[300]}`);
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('run', 'last');
			const tape = rt.sources.run.tape!;
			if (tape.length === 0) return assert.equal(fx.id, 'edge:empty');
			const ls = losses(tape);
			assert.ok(ls[ls.length - 1] < ls[0], `${fx.id}: ${ls[0]} -> ${ls[ls.length - 1]}`);
		},
		opts
	);
});

test('M2: no NaN or infinity in 2,000 steps on any preset, at learning rates 0.003 to 3 (Adam) and 3 (SGD)', () => {
	for (const name of ['moons', 'circles', 'xor', 'spiral', 'gaussians'] as const)
		for (const o of [{ lr: 0.003 }, { lr: 0.03 }, { lr: 0.3 }, { lr: 3 }, { optimizer: 'sgd' as const, lr: 3 }]) {
			const t = record(play, presetRaw(name), { ...o, steps: 2000 }, { seed: 1 });
			assert.equal(losses(t).length, 2001, `${name} ${JSON.stringify(o)}: the run stopped early`);
			assert.ok(finiteAll(t), `${name} ${JSON.stringify(o)}: a non-finite number`);
		}
	assert.deepEqual(['lr0003', 'lr03', 'lr3', 'run'].map((n) => (manifest.sources![n] as { opt: { lr?: number } }).opt.lr ?? fit.defaults.lr), [0.003, 0.3, 3, 0.03], 'the Training section presets');
	for (const hidden of [[2, 2], [8, 8]]) assert.ok(finiteAll(record(play, SPIRAL, { hidden, lr: 3, steps: 2000 }, { seed: 1 })), `spiral ${hidden} lr 3`);
});

test('prequestion and try-this: without a hidden layer one straight line cannot separate the ring; four lines per layer can', () => {
	const flat = last(record(play, presetRaw('circles'), { hidden: [], steps: 2000 }, { seed: 1 }));
	const deep = last(record(play, presetRaw('circles'), {}, { seed: 1 }));
	assert.ok(flat.metrics!.accuracy <= 0.75, `no hidden layer: ${flat.metrics!.accuracy}`);
	assert.equal(deep.metrics!.accuracy, 1);
});

test('step "weighted-sum": z = w0 x + w1 y + b and a = tanh(z) for the traced point, from the same weights', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('pt', { layer: 'net.0' });
			const tr = rt.sources.pt.trace;
			const x = rt.pre?.tensors.input?.data;
			if (!tr || !x) return assert.ok(rt.pre?.empty);
			const t = rt.sources.train.tape!;
			const p = (t.state(t.length - 1) as { params: Params }).params;
			const z = p['net.0.weight'].data[0] * x[0] + p['net.0.weight'].data[1] * x[1] + p['net.0.bias'].data[0];
			assert.ok(Math.abs(tr.t['net.0'].data[0] - z) < 1e-5, `${tr.t['net.0'].data[0]} vs ${z}`);
			assert.ok(Math.abs(tr.t['net.1'].data[0] - Math.tanh(tr.t['net.0'].data[0])) < 1e-6);
		},
		opts
	);
});

test('steps "loss", "gradient", "update": the loss is the mean of the points\' losses, a positive gradient means growing the weight raises the loss and a nudge of EPS changes it by about g × EPS, the random network is unsure everywhere, and the update is old − lr × g and lowers the loss', async () => {
	assert.ok(Math.abs(-Math.log(0.5) - 0.69) < 0.005, 'predict "loss": −ln of one half is about 0.69');
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('sgd1', 'last');
			const t = rt.sources.sgd1.tape!;
			if (t.length === 0) return assert.equal(fx.id, 'edge:empty');
			const frames = Array.from({ length: t.length }, (_, i) => ({ f: t.frame(i), s: t.state(i) as { params: Params; grads: Params } }));
			const pts = dataOf(rt.input);
			const init = frames.find((x) => x.f.phase === 'init')!.s.params;
			const L0 = frames.find((x) => x.f.phase === 'loss')!.f.metrics!.loss;
			assert.ok(Math.abs(evalPoints(init, 'tanh', pts).mean - L0) < 1e-12, `${fx.id}: mean of the points' losses`);
			const up = frames.find((x) => x.f.phase === 'update')!;
			const g = up.s.grads;
			let best = { key: '', k: 0, v: 0 };
			for (const [key, w] of Object.entries(g)) if (key.endsWith('.weight')) w.data.forEach((v, k) => Math.abs(v) > Math.abs(best.v) && (best = { key, k, v }));
			const bump = (eps: number) => {
				const q: Params = { ...init, [best.key]: { shape: init[best.key].shape, data: Float64Array.from(init[best.key].data) } as Tensor };
				q[best.key].data[best.k] += eps;
				return evalPoints(q, 'tanh', pts).mean;
			};
			if (Math.abs(best.v) > 1e-6) assert.equal(Math.sign(bump(1e-4) - bump(-1e-4)), Math.sign(best.v), `${fx.id}: the sign of the gradient`);
			// "nudging its weight up by EPS changes the loss by dl, close to g × EPS" (within 15%)
			const dl = bump(EPS) - bump(0);
			if (Math.abs(best.v) > 1e-3) assert.ok(Math.abs(dl - best.v * EPS) <= 0.15 * Math.abs(best.v * EPS), `${fx.id}: nudge ${dl} vs g × eps ${best.v * EPS}`);
			// "pale because it is unsure everywhere": before training the share for class 1 stays near one half on the whole plane
			const g0 = neuronGrids(init, 'tanh', true).data;
			assert.ok(Math.min(...g0) > 0.3 && Math.max(...g0) < 0.7, `${fx.id}: the random network's shares ${Math.min(...g0)} .. ${Math.max(...g0)}`);
			const I = init[best.key].shape[1];
			const u = updateOf({ index: t.length - 1, frame: up.f, state: up.s, tape: t }, best.key, [Math.floor(best.k / I), best.k % I]);
			assert.ok(Math.abs(u.old - u.lr * u.g - u.new) < 1e-15);
			assert.ok(evalPoints(up.s.params, 'tanh', pts).mean < L0, `${fx.id}: the loss falls`);
		},
		opts
	);
});

test('step "spiral": 2 neurons per layer stay below 70% of the spiral even after 1,000 steps; 8 per layer pass 95% in 300', () => {
	for (const [name, hidden] of [['spiral2', [2, 2]], ['spiral4', [4, 4]], ['spiral8', [8, 8]]] as const) {
		assert.deepEqual(src(name).opt.hidden, hidden);
		assert.equal(src(name).opt.data, SPIRAL);
	}
	const at = (hidden: number[], steps: number) => last(record(play, SPIRAL, { hidden, steps }, { seed: 1 })).metrics!.accuracy;
	assert.ok(at([2, 2], 300) < 0.7 && at([2, 2], 1000) < 0.7, '"training longer does not fix this"');
	assert.ok(at([8, 8], 300) >= 0.95, 'the task: 8 neurons per layer learn the spiral');
	assert.equal(dataOf(SPIRAL).n, 300);
});

test('step "high-lr": same spiral, same 8-neuron network, same start; at learning rate 3 the loss jumps up after one step, keeps rising on many steps and never comes near learning rate 0.03, also when the run is perturbed in its last digits', () => {
	assert.deepEqual(src('lrs').opt.hidden, [8, 8]);
	assert.equal(src('lrs').opt.data, SPIRAL);
	assert.deepEqual([src('lrs').opt.lr, src('lrs').opt.lrHi], [0.03, 3]);
	const lo = record(play, SPIRAL, { hidden: [8, 8], lr: 0.03 }, { seed: 1 });
	const hi = record(play, SPIRAL, { hidden: [8, 8], lr: 3 }, { seed: 1 });
	assert.deepEqual(lo.state(0).params, hi.state(0).params, 'same start');
	const pair = record(lrPair, null, src('lrs').opt, { seed: 1 });
	for (let i = 0; i < pair.length; i++) {
		const m = pair.frame(i).metrics!;
		if (typeof m.hiLoss === 'number') assert.equal(m.hiLoss, losses(hi)[m.step], 'lrPair carries the lr 3 run');
	}
	// The lr-3 run is chaotic: Edge and Node already differ in the 16th digit of its loss after one step, and by step 300
	// the curves differ visibly. So the narration states only what holds for every such run: check it on the run and on
	// four copies whose spiral differs by 1e-15 to 1e-13 (relative) in a few coordinates, at 300 steps and at e2e's 200.
	const P = dataOf(SPIRAL);
	const nudged = (eps: number) => ({ points: Array.from({ length: P.n }, (_, i) => ({ x: P.xy[2 * i] * (i % 7 === 0 ? 1 + eps : 1), y: P.xy[2 * i + 1], c: P.label[i] })) });
	const ups = (l: number[]) => l.slice(1).filter((v, i) => v > l[i]).length / (l.length - 1);
	for (const eps of [0, 1e-15, -1e-15, 1e-14, 1e-13]) {
		const t = record(lrPair, nudged(eps), { ...src('lrs').opt, data: null }, { seed: 1 });
		const lh = Array.from({ length: t.length }, (_, i) => t.frame(i).metrics?.hiLoss).filter((v): v is number => typeof v === 'number');
		const ll = losses(t);
		assert.ok(lh[1] > 5 * lh[0], `eps ${eps}: the jump ${lh[0]} -> ${lh[1]}`);
		for (const n of [200, 300]) {
			const after = lh.slice(1, n + 1);
			assert.ok(ups(after) > 0.25 && ups(ll.slice(1, n + 1)) < 0.05, `eps ${eps}, ${n} steps: "keeps overshooting": rises on ${ups(after)} vs ${ups(ll.slice(1, n + 1))} of the steps`);
			assert.ok(Math.min(...after) > 10 * ll[n], `eps ${eps}, ${n} steps: "never gets below" ${Math.min(...after)} vs lr 0.03 at ${ll[n]}`);
			assert.ok(Math.max(...lh.slice(0, n + 1)) / Math.min(...ll.slice(0, n + 1)) > 30, 'the loss-curve view picks its log scale (look: "on a log scale")');
		}
	}
});
