// pinn claim tests (design §4.6 "Claims are tests", §8.3 M9): every sentence of the lesson and the page that is not
// computed live is asserted here, on the three presets (the inputs a learner starts from) and, where a sentence speaks
// about them, on the edge inputs. Runs are recorded exactly as the page runs them (same trainers, options and seed).
//   node --test src/lib/explainers/pinn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { record } from '../../atlas/process/record.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { OSCILLATOR, seriesOf, seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';
import { dampedOscillator } from '../../atlas/data/series.ts';
import manifest from './explainer.ts';
import adapter from './adapters/series/adapter.ts';
import * as trainers from './trainer.ts';
import { netAt, pinn } from './trainer.ts';
import { probe } from './process.ts';
import { N, T, TRUTH, meanSq, pointsOf, valuesAt } from './osc.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const presetFile = JSON.parse(readFileSync(join(SITE, 'static/models/pinn/presets.json'), 'utf8'));
const PRESETS = adapter.presets!({ 'presets.json': presetFile });
const NO_POINTS = adapter.edgeFixtures().find((f) => f.id === 'no-points')!.raw;
const SOURCES = manifest.sources as Record<string, { kind: string; trainer?: string; opt?: Record<string, unknown> }>;
const TRAIN = Object.entries(SOURCES).filter(([, d]) => d.kind === 'train');

const tapes = new Map<string, Tape>();
/** a train source run on an input, as the page runs it (memoized for the whole file) */
function run(source: string, raw: unknown, key: string, extra: Record<string, unknown> = {}): Tape {
	const k = `${source}|${key}|${JSON.stringify(extra)}`;
	let t = tapes.get(k);
	if (!t) {
		const d = SOURCES[source];
		const { seed, ...opt } = d.opt ?? {};
		t = record(trainingProcess((trainers as unknown as Record<string, typeof pinn>)[d.trainer!]), raw, { ...opt, ...extra }, { seed: Number(seed ?? 1) });
		tapes.set(k, t);
	}
	return t;
}
type End = { m: Record<string, number>; cv: Tensor; params: Record<string, Tensor> };
function end(t: Tape): End {
	let i = t.length - 1;
	while (i > 0 && t.frame(i).phase !== 'snapshot') i--;
	const s = t.state(i) as { curves: Tensor; params: Record<string, Tensor> };
	return { m: t.frame(i).metrics ?? {}, cv: s.curves, params: s.params };
}
const first = (t: Tape) => t.frame(t.find({ phase: 'snapshot', step: 0 })).metrics ?? {};
const u = (e: End) => e.cv.data.subarray(0, N);
const maxAbsOn = (v: ArrayLike<number>, a: number, b: number) => Math.max(...Array.from(T).flatMap((t, i) => (t >= a && t <= b ? [Math.abs(v[i])] : [])));
const each = (fn: (raw: unknown, id: string) => void) => PRESETS.forEach((p) => fn(p.raw, p.id));

test('the presets are the damped oscillator with eight points in the first 40% of the time', () => {
	assert.equal(PRESETS.length, 3);
	for (const p of PRESETS) {
		assert.equal((p.raw as { preset: { name: string } }).preset.name, 'oscillator');
		const obs = adapter.preprocess(p.raw, null, {}).tensors.obs!;
		assert.equal(obs.shape[0], 8);
		for (let i = 0; i < 8; i++) assert.ok(obs.data[2 * i] >= 0 && obs.data[2 * i] <= 0.4);
		// step "data": the line the input box draws (the series' y) is the true curve, without noise
		const s = seriesOf(p.raw);
		assert.ok(s.y.length > 0);
		s.y.forEach((y, i) => assert.equal(y, dampedOscillator([s.t[i]], OSCILLATOR)[0]));
	}
});

test('step "loss" look: the data part falls first while the start part grows for a while; all three end far below their start', () => {
	each((raw, id) => {
		const t = run('pinn', raw, id);
		const snaps = Array.from({ length: t.length }, (_, i) => t.frame(i)).filter((f) => f.phase === 'snapshot').map((f) => f.metrics ?? {});
		const [a, z] = [snaps[0], snaps[snaps.length - 1]];
		assert.ok(Math.max(...snaps.map((m) => m.start)) > 10 * a.start, `${id}: the start part grows for a while`);
		const peak = snaps.findIndex((m) => m.start === Math.max(...snaps.map((q) => q.start)));
		assert.ok(snaps[peak].data < a.data, `${id}: by then the data part has fallen (${snaps[peak].data} < ${a.data})`);
		for (const k of ['data', 'physics', 'start']) assert.ok(z[k] < 0.05 * a[k], `${id}: ${k} ends far below its start (${a[k]} -> ${z[k]})`);
	});
});

test('try-this "One bad measurement": the plain network bends through the bad point; the PINN does not', () => {
	const setup = (manifest.page!.tryThis![0].setup as { input: unknown }).input;
	const pts = pointsOf(setup);
	const k = Array.from(pts.t).indexOf(0.25);
	assert.ok(Math.abs(pts.u[k] - dampedOscillator([0.25], OSCILLATOR)[0]) > 0.5, 'the point at t = 0.25 is far off the curve');
	const at = (s: string) => valuesAt(netAt(end(run(s, setup, 'outlier')).params, {}), [0.25])[0];
	assert.ok(Math.abs(at('plain') - pts.u[k]) < 0.01, `the plain network reaches it (${at('plain')})`);
	assert.ok(Math.abs(at('pinn') - pts.u[k]) > 0.3, `the PINN does not (${at('pinn')})`);
});

test('step "law": the true curve swings back and forth and dies down', () => {
	const peaks: number[] = [];
	for (let i = 1; i + 1 < N; i++) if (Math.abs(TRUTH[i]) >= Math.abs(TRUTH[i - 1]) && Math.abs(TRUTH[i]) >= Math.abs(TRUTH[i + 1])) peaks.push(Math.abs(TRUTH[i]));
	assert.ok(peaks.length >= 3, `${peaks.length} swings`);
	for (let i = 1; i < peaks.length; i++) assert.ok(peaks[i] < peaks[i - 1], 'every swing is smaller than the one before');
	assert.equal(TRUTH[0], 1);
});

test('M9 + step "loss": plain, the PINN and every λ and wrong-c source start from the same weights (rng.fork("init"))', () => {
	const raw = PRESETS[0].raw;
	const ref = run('plain', raw, 'init', { steps: 0 }).state(0) as { params: Record<string, Tensor> };
	for (const [name] of TRAIN) {
		const s = run(name, raw, 'init', { steps: 0 }).state(0) as { params: Record<string, Tensor> };
		for (const [k, v] of Object.entries(ref.params)) assert.deepEqual(Array.from(s.params[k].data), Array.from(v.data), `${name} ${k}`);
	}
	assert.equal(Object.values(ref.params).reduce((s, t) => s + t.data.length, 0), 1153);
});

test('steps "plain", "side", "after" and the predict answer: the plain network fits the points and wanders after them; the PINN stays close', () => {
	each((raw, id) => {
		const [a, b] = [end(run('plain', raw, id)), end(run('pinn', raw, id))];
		assert.ok(a.m.data < 0.01 && b.m.data < 0.01, `${id}: both do well at the points (${a.m.data}, ${b.m.data})`);
		assert.ok(a.m.err > 0.3, `${id}: the plain network goes its own way (${a.m.err})`);
		assert.ok(b.m.err < 0.05, `${id}: the PINN carries the swing on (${b.m.err})`);
		assert.ok(a.m.err > 10 * b.m.err, `${id}: ${a.m.err} vs ${b.m.err}`);
		assert.ok(meanSq(a.cv, 3) > 100 * meanSq(b.cv, 3), `${id}: the strips tell them apart`);
	});
});

test('e2e cap (200 steps): the PINN is already closer than the plain network after t = 0.4', () => {
	// the snapshot at step 200 of a full run holds the weights after 200 updates, as a run capped at 200 steps ends
	const at200 = (t: Tape) => t.frame(t.find({ phase: 'snapshot', step: 200 })).metrics ?? {};
	each((raw, id) => {
		const [a, b] = [at200(run('plain', raw, id)), at200(run('pinn', raw, id))];
		assert.ok(a.err > 2 * b.err, `${id}: ${a.err} vs ${b.err}`);
	});
});

test('step "deriv": the probe trains the plain network\'s weights; finite differences agree at h = 0.001 and fail at h = 1e−7', async () => {
	each((raw, id) => {
		const t = record(probe, raw, {}, { seed: 1 });
		const m = (i: number) => t.frame(i).metrics ?? {};
		const at3 = t.find({ phase: 'fd', step: 3 });
		assert.equal(m(at3).h, 0.001);
		const [d1, d2] = [m(at3).d1, m(at3).d2];
		assert.ok(m(at3).e1 < 1e-4 * Math.max(1, Math.abs(d1)) && m(at3).e2 < 1e-3 * Math.max(1, Math.abs(d2)), `${id}: agree at h = 0.001`);
		assert.ok(m(at3).eTiny > 100 * m(at3).e2, `${id}: rounding errors take over at h = ${m(at3).hTiny}`);
		// "the bend sits in the last few of the 16 digits the computer keeps for u": h²·u″, the part of
		// u(t+h) − 2u(t) + u(t−h) that carries the bend, is below 1e−12 of u itself (float64 keeps about 16 digits)
		assert.ok(m(at3).hTiny ** 2 * Math.abs(d2) < 1e-12 * Math.abs(m(0).u), `${id}: h²·u″ = ${m(at3).hTiny ** 2 * Math.abs(d2)} vs u = ${m(0).u}`);
		const e = end(run('plain', raw, id));
		assert.ok(Math.abs(valuesAt(netAt(e.params, {}), [0.7])[0] - m(0).u) < 1e-12, `${id}: the same plain network`);
	});
	const facts = JSON.parse(readFileSync(join(SITE, 'static/models/pinn/facts.json'), 'utf8')).facts;
	assert.ok(facts.derivParity.value <= 1e-10, 'u′, u″, residual and physics-loss gradients match torch.autograd.grad(create_graph=True)');
});

test('M9 λ task and its hint: λ = 0 wanders, λ = 100 lets go of the points (a nearly flat curve), λ = 1 is the closest', () => {
	assert.deepEqual([trainers.pinnLam0.defaults.lam, trainers.pinnLam001.defaults.lam, SOURCES.lam100.opt!.lam, pinn.defaults.lam], [0, 0.01, 100, 1]);
	assert.deepEqual(['lam0', 'lam001', 'lam100', 'pinn'].map((s) => SOURCES[s].trainer), ['pinnLam0', 'pinnLam001', 'pinn', 'pinn']);
	each((raw, id) => {
		const [l0, l1, l100] = ['lam0', 'pinn', 'lam100'].map((s) => end(run(s, raw, id)));
		assert.ok(l0.m.err > 5 * l1.m.err, `${id}: λ = 0 wanders (${l0.m.err} vs ${l1.m.err})`);
		assert.ok(l100.m.data > 10 * l1.m.data, `${id}: λ = 100 no longer passes through the points (${l100.m.data})`);
		assert.ok(maxAbsOn(u(l100), 0.4, 1) < 0.1, `${id}: λ = 100 is nearly flat after the points`);
		assert.ok(l1.m.err < Math.min(l0.m.err, l100.m.err), `${id}: λ = 1 is closer than 0 and 100`);
	});
	const [p] = PRESETS;
	const l001 = end(run('lam001', p.raw, p.id));
	assert.ok(end(run('pinn', p.raw, p.id)).m.err < l001.m.err && l001.m.err < end(run('lam0', p.raw, p.id)).m.err, 'start input: λ = 0.01 lies between');
});

test('M9 wrong-c breakIt and the misconception: c = 0 gives a confident wrong curve', () => {
	assert.equal(SOURCES.wrongC.trainer, 'pinnWrongC');
	assert.equal(trainers.pinnWrongC.defaults.c, 0);
	each((raw, id) => {
		const wt = run('wrongC', raw, id);
		const [w, r] = [end(wt), end(run('pinn', raw, id))];
		assert.ok(w.m.loss < 0.05 * first(wt).loss, `${id}: its loss still falls far below the start`);
		assert.ok(w.m.loss > r.m.loss, `${id}: its loss is higher than the right law's`);
		assert.ok(w.m.err > 5 * r.m.err, `${id}: far from the truth after the data (${w.m.err} vs ${r.m.err})`);
		assert.ok(maxAbsOn(u(w), 0.7, 1) > 2 * maxAbsOn(TRUTH, 0.7, 1), `${id}: its swing does not die down`);
		// the look: its strip (checked against its own law, c = 0) stays pale, far paler than the plain network's
		const p = end(run('plain', raw, id));
		const res = (e: End) => e.cv.data.subarray(3 * N, 4 * N);
		assert.ok(maxAbsOn(res(w), 0, 1) < 0.25, `${id}: pale strip (${maxAbsOn(res(w), 0, 1)})`);
		assert.ok(meanSq(w.cv, 3) < meanSq(p.cv, 3) / 10, `${id}: far paler than the plain network's (${meanSq(w.cv, 3)} vs ${meanSq(p.cv, 3)})`);
	});
	const [w, r] = [end(run('wrongC', PRESETS[0].raw, PRESETS[0].id)), end(run('pinn', PRESETS[0].raw, PRESETS[0].id))];
	assert.ok(w.m.err > 10 * r.m.err, 'on the start input more than ten times');
	// with no points, both laws end on the same curve (the lesson says so on that input)
	const [wn, rn] = [end(run('wrongC', NO_POINTS, 'none')), end(run('pinn', NO_POINTS, 'none'))];
	assert.ok(Math.max(...Array.from(u(wn), (v, i) => Math.abs(v - u(rn)[i]))) < 0.05, 'no points: the same curve');
	// step "after" on that input: nothing sets the size of the swing, and the PINN settles on u = 0
	assert.ok(maxAbsOn(u(rn), 0, 1) < 0.05, `no points: the PINN is nearly flat at u = 0 (${maxAbsOn(u(rn), 0, 1)})`);
});

test('manifest: the wrong-c compare label and the λ buttons name the options their sources run', async () => {
	const { default: lesson } = await import('./lesson.ts');
	const step = lesson.steps.find((s) => 'id' in s && s.id === 'wrong-c') as { compare: { a: string; label: { en: string }[] } };
	assert.equal(step.compare.a, 'wrongC');
	assert.ok(step.compare.label[0].en.endsWith(`c = ${trainers.pinnWrongC.defaults.c}`));
	assert.equal(seriesPresetRaw('oscillator').preset!.seed, 1);
});
