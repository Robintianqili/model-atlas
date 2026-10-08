// gan-2d claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, per fixture through withFixtures (lesson fixtures, the adapter's edge fixtures, the presets).
// Claims about a whole GAN run (a chaotic system: the browser's last digits differ from Node's, so its run differs too)
// are checked on the page's seed and on four other seeds, so they hold for runs like the one the learner sees.
//   node --test src/lib/explainers/gan-2d/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { backward, zeroGrad, type Var } from '../../atlas/autograd/var.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { gan } from './trainer.ts';
import manifest from './explainer.ts';
import { EIGHT_CENTRES, presetOf, realOf } from './data.ts';
import { dGrid, gen, nearCount, MODE_MIN, Z_SHOW, type Params } from './net.ts';
import { alongArrows, dLossParts, gLossOf, sceneOf } from './scene.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('gan-2d', SITE)).pkg;
const P = trainingProcess(gan);
const optOf = (name: string) => (manifest.sources![name] as { opt: Record<string, unknown> }).opt;
const metric = (t: Tape, k: string) => Array.from({ length: t.length }, (_, i) => t.frame(i).metrics?.[k]).filter((v): v is number => typeof v === 'number');
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const SEEDS = [1, 2, 3, 4, 5];

test('steps "generator" and "discriminator", page params: G is 2 -> 16 -> 16 -> 2, D is 2 -> 16 -> 16 -> 1, tanh between layers; 691 parameters', () => {
	const m = gan.model(createRng(1).fork('init'), gan.defaults);
	assert.deepEqual([m.gen.sizes, m.disc.sizes, m.gen.act, m.disc.act], [[2, 16, 16, 2], [2, 16, 16, 1], 'tanh', 'tanh']);
	assert.equal(m.params().reduce((s, p) => s + p.size, 0), 691);
	assert.ok(manifest.catalog?.params?.en.includes('691 parameters'));
});

test('step "noise": every noise number lies between −1 and 1, drawn uniformly (the shown 240 and every training batch)', () => {
	const t = record(P, presetOf('ring'), { steps: 20 }, { seed: 1, detail: 'fine' });
	const zs = [Array.from(Z_SHOW)];
	for (let i = 0; i < t.length; i++) if (t.frame(i).phase === 'forward') zs.push(Array.from(((t.state(i) as { batch: Record<string, Tensor> }).batch.z.data)));
	const all = zs.flat();
	assert.ok(all.every((v) => v >= -1 && v <= 1));
	// uniform on [−1, 1]: mean 0, variance 1/3
	assert.ok(Math.abs(mean(all)) < 0.05 && Math.abs(mean(all.map((v) => v * v)) - 1 / 3) < 0.03);
});

test('step "discriminator": untrained, D is unsure everywhere (its answer is within 0.1 of ½ all over the plane)', () => {
	const t = record(P, presetOf('ring'), { steps: 0 }, { seed: 1 });
	const g = dGrid((t.state(0) as { params: Params }).params).data;
	assert.ok(g.every((v) => Math.abs(v - 0.5) < 0.1), `${Math.min(...g)} .. ${Math.max(...g)}`);
});

test('prequestion and step "g-step": G never sees the real points (its gradient does not change when they do)', () => {
	const m = gan.model(createRng(1).fork('init'), gan.defaults);
	const grads = (x: number[]) => {
		zeroGrad(m.params());
		const b = { x: { shape: [x.length / 2, 2], data: Float64Array.from(x) }, z: { shape: [2, 2], data: Float64Array.of(0.1, -0.4, 0.7, 0.2) }, turn: [1] };
		backward(gan.loss(m, b, () => {}, gan.defaults).loss as Var);
		return m.gen.params().map((p) => Array.from(p.grad!.data));
	};
	assert.deepEqual(grads([0.5, 0.5, -0.3, 0.2]), grads([-0.9, 0.1]));
});

test('steps "d-step" and "g-step": each net moves only on its turn; on every input with points D’s step lowers its loss on the batch; on G’s step D stays fixed, G’s loss falls, and most points move along their arrow', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('slow', 'last');
			const t = rt.sources.slow.tape!;
			const at = (step: number) => {
				const i = t.find({ phase: 'update', step });
				return { index: i, frame: t.frame(i), state: t.state(i), tape: t };
			};
			const d = sceneOf(rt, at(0), 'slow')!;
			const g = sceneOf(rt, at(1), 'slow')!;
			// each net moves only on its own turn
			for (const [s, net] of [[d, 'gen.'], [g, 'disc.']] as const) for (const k of Object.keys(s.params)) if (k.startsWith(net)) assert.deepEqual(Array.from(s.params[k].data), Array.from(s.before![k].data), `${fx.id}: ${k} moved`);
			if (!d.real.n) return;
			const [a, b] = dLossParts(d.before!, d.realShown, d.fakeBefore!);
			const [a1, b1] = dLossParts(d.params, d.realShown, d.fake);
			assert.ok(a1 + b1 < a + b, `${fx.id}: D loss ${a + b} -> ${a1 + b1}`);
			assert.ok(gLossOf(g.params, g.fake) < gLossOf(g.before!, g.fakeBefore!), `${fx.id}: G loss`);
			assert.ok(alongArrows(g) >= 0.9 * (g.fake.length / 2), `${fx.id}: ${alongArrows(g)} along`);
			assert.ok(g.dFake > g.dFakeBefore, `${fx.id}: D(generated) ${g.dFakeBefore} -> ${g.dFake}`);
		},
		opts
	);
});

test('step "watch": G learns more slowly than D (learning rates 0.002 and 0.01), D and G take turns 1 : 1', () => {
	assert.ok((gan.defaults.lrG as number) < gan.defaults.lr && gan.defaults.dSteps === 1);
	assert.equal(optOf('train').lrG, undefined, 'the lesson run uses the defaults');
	assert.equal(optOf('slower').lrG, undefined, '"G 5 times slower (this page’s setting)"');
	assert.equal(gan.defaults.lr / (gan.defaults.lrG as number), 5);
	assert.equal(optOf('same').lrG, gan.defaults.lr, '"G as fast as D"');
});

test('step "game" and the misconception: the losses do not fall to 0 but wobble around 2 ln 2 and ln 2, and D’s loss already starts within 0.05 of 2 ln 2 (every input; the ring on seeds 1-5)', async () => {
	// D says ½ everywhere (all logits 0): D's loss 2 ln 2, G's loss ln 2
	const zero: Params = { 'disc.0.weight': { shape: [1, 2], data: Float64Array.of(0, 0) }, 'disc.0.bias': { shape: [1], data: Float64Array.of(0) } };
	const pts = Float64Array.of(0.3, -0.2, 0.9, 0.1);
	assert.ok(Math.abs(dLossParts(zero, pts, pts).reduce((s, v) => s + v) - 2 * Math.LN2) < 1e-15 && Math.abs(gLossOf(zero, pts) - Math.LN2) < 1e-15);
	const check = (t: Tape, id: string) => {
		const dl = metric(t, 'dLoss').slice(-200);
		const gl = metric(t, 'gLoss').slice(-200);
		assert.ok(Math.min(...dl) > 0.5 && Math.min(...gl) > 0.3, `${id}: a loss near 0`);
		assert.ok(Math.abs(mean(dl) - 2 * Math.LN2) < 0.3 && Math.abs(mean(gl) - Math.LN2) < 0.4, `${id}: ${mean(dl)} ${mean(gl)}`);
		assert.ok(Math.abs(metric(t, 'dLoss')[0] - 2 * Math.LN2) < 0.05, `${id}: D's loss at step 0 is ${metric(t, 'dLoss')[0]}`);
	};
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('train', 'last');
			if (realOf(rt.input).n) check(rt.sources.train.tape!, fx.id);
		},
		opts
	);
	for (const seed of SEEDS) {
		check(record(P, presetOf('ring'), {}, { seed }), `ring seed ${seed}`);
	}
});

test('steps "generator" and "game": untrained, G squeezes the whole noise square into a patch less than 0.3 wide (the page’s seed; the init does not depend on the points)', () => {
	const t = record(P, presetOf('ring'), { steps: 0 }, { seed: 1 });
	const fake = gen((t.state(0) as { params: Params }).params, Z_SHOW);
	const span = (o: number) => Math.max(...fake.filter((_, i) => i % 2 === o)) - Math.min(...fake.filter((_, i) => i % 2 === o));
	assert.ok(Math.max(span(0), span(1)) < 0.3, `patch ${span(0)} x ${span(1)}`);
});

test('no points of yours (runOnEmpty): every source still runs; on your (empty) points nothing moves, the eight-blob runs train on their fixed data', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('train', 'last');
			const t = rt.sources.train.tape!;
			const [p0, p1] = [0, t.length - 1].map((i) => (t.state(i) as { params: Params }).params);
			assert.ok(t.length > 1 && Object.keys(p0).every((k) => p0[k].data.every((v, i) => v === p1[k].data[i])), `train: ${t.length} frames, weights moved`);
			for (const s of ['same', 'slower']) {
				await rt.ensure(s, 'last');
				const u = rt.sources[s].tape!;
				assert.ok(u.length > 1 && Number.isFinite(u.frame(u.length - 1).metrics?.modes), `${s}: ${u.length} frames`);
			}
		},
		{ ...opts, only: ['edge:empty'] }
	);
});

/** the blobs covered at each snapshot of the second half of a run, and every blob covered at some time in it */
function lateModes(t: Tape): { late: number[]; seen: Set<number> } {
	const late: number[] = [];
	const seen = new Set<number>();
	for (let i = 0; i < t.length; i++) {
		const f = t.frame(i);
		if (f.phase !== 'snapshot' || (f.step ?? 0) < 1500) continue;
		late.push(f.metrics!.modes);
		const fake = gen((t.state(i) as { params: Params }).params, Z_SHOW);
		EIGHT_CENTRES.forEach(([x, y], k) => nearCount(x, y, fake) >= MODE_MIN && seen.add(k));
	}
	return { late, seen };
}

test('step "collapse": with G as fast as D on eight blobs, G covers only a few blobs at times and hops between them; with G 5 times slower it keeps at least 4 (seeds 1-5)', () => {
	for (const seed of SEEDS) {
		const { late, seen } = lateModes(record(P, presetOf('eight'), optOf('same'), { seed }));
		assert.ok(Math.min(...late) <= 2, `seed ${seed}: at least ${Math.min(...late)} blobs all the time`);
		assert.ok(seen.size >= 5, `seed ${seed}: only ${seen.size} different blobs covered in the second half`);
		const slower = lateModes(record(P, presetOf('eight'), optOf('slower'), { seed })).late;
		assert.ok(Math.min(...slower) >= 4 && Math.min(...slower) > Math.min(...late), `seed ${seed}: G 5 times slower drops to ${Math.min(...slower)} blobs`);
	}
});
