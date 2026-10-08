// matrix-factorization claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is
// not computed live is asserted here, on every fixture where it depends on the input (presets + the adapter's edge
// fixtures). `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/matrix-factorization/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, TrainState } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { record } from '../../atlas/process/record.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { fit } from './trainer.ts';
import { nudge, nudgeRating } from './process.ts';
import { MOVIES, N_ITEMS, N_OTHERS, N_USERS, OTHERS, PRESETS, YOU, factorsOf, initFactors, known, predict, ranked, vecU, vecV, type Factors } from './world.ts';
import { closest } from './picture.ts';
import { RATE_GOAL, fitsYou, sameKindNeighbours } from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('matrix-factorization', SITE)).pkg;
const factorsAt = (t: Tape, i: number): Factors => factorsOf((t.state(i) as Partial<TrainState>).params)!;
const last = (t: Tape) => factorsAt(t, t.length - 1);

test('page: (24 + 1 people + 20 movies) × 2 = 90 numbers; the sections say 4,000 steps, one rating per step, learning rate 0.05', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.deepEqual([N_OTHERS, N_USERS, N_ITEMS], [24, 25, 20]);
	assert.equal(m.params().reduce((s, p) => s + p.value.data.length, 0), 90);
	assert.deepEqual([fit.defaults.steps, fit.defaults.batch, fit.defaults.lr, fit.defaults.optimizer], [4000, 1, 0.05, 'sgd']);
	const training = readFileSync(join(SITE, 'src/lib/explainers/matrix-factorization/sections/Training.svelte'), 'utf8');
	assert.ok(training.includes('4,000 steps') && training.includes('learning rate 0.05') && training.includes('one random known rating per step'));
	assert.equal(1 * 3 + 2 * -1, 1, 'quiz: (1, 2)·(3, −1)');
});

test('step "matrix": each other person rated only a few movies (8 to 12 of 20); the ratings are whole numbers from −2 to +2', () => {
	for (let u = 0; u < N_OTHERS; u++) {
		const n = OTHERS.filter((x) => x.u === u).length;
		assert.ok(n >= 8 && n <= 12, `person ${u}: ${n}`);
	}
	assert.ok(OTHERS.every((x) => Number.isInteger(x.r) && x.r >= -2 && x.r <= 2));
	assert.equal(RATE_GOAL, 10, 'task: "at least ten movies"');
});

test('prequestion, misconception, step "factors": training sees only (person, movie, rating), yet on every input at least 15 of the 20 movies have a nearest neighbour of their own kind (more than at the start)', async () => {
	const b = fit.batch(PRESETS[0].raw, 0, createRng(3), fit.defaults);
	assert.deepEqual(Object.keys(b).sort(), ['i', 'r', 'u']);
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('train', 'last');
			const t = rt.sources.train.tape!;
			const end = sameKindNeighbours(last(t)), start = sameKindNeighbours(factorsAt(t, 0));
			assert.ok(end >= 15 && end > start, `${fx.id}: ${start} -> ${end}`);
		},
		opts
	);
});

test('predict "dot": the marked movie points the way your arrow points (the largest cosine of its pool) and its predicted rating is positive, on every input', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('train', 'last');
			const f = last(rt.sources.train.tape!);
			const c = closest(f, rt.input);
			const cos = (i: number) => predict(f, YOU, i) / Math.hypot(...vecV(f, i)) / Math.hypot(...vecU(f, YOU));
			assert.ok(cos(c.i) > 0 && predict(f, YOU, c.i) > 0, `${fx.id}: ${predict(f, YOU, c.i)}`);
			const mine = (rt.input as { ratings: (number | null)[] }).ratings;
			for (let i = 0; i < N_ITEMS; i++) if (!c.unrated || mine[i] === null) assert.ok(cos(i) <= cos(c.i) + 1e-12, `${fx.id}: movie ${i}`);
		},
		opts
	);
});

test('steps "error" and "update": the nudge starts from the trainer’s own starting numbers, is exactly one SGD step of the trainer on that rating (old − lr × g), and brings the prediction closer to the rating', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('nudge', 'last');
			const t = rt.sources.nudge.tape!;
			const m = t.frame(t.length - 1).metrics!;
			const want = nudgeRating(rt.input);
			assert.deepEqual([m.user, m.item, m.r], [want.user, want.item, want.r]);
			const { U, V } = initFactors(createRng(1).fork('init'), fit.defaults.init);
			assert.deepEqual([m.u1, m.u2, m.v1, m.v2], [U[2 * m.user], U[2 * m.user + 1], V[2 * m.item], V[2 * m.item + 1]]);
			assert.ok(Math.abs(m.eAfter) < Math.abs(m.e), `${fx.id}: |${m.eAfter}| < |${m.e}|`);
			// the same step on the trainer itself, its batch forced to this rating
			const one = { ...fit, batch: () => ({ u: Int32Array.of(m.user), i: Int32Array.of(m.item), r: Float64Array.of(m.r) }) };
			const ft = record(trainingProcess(one), rt.input, { steps: 1 }, { seed: 1, detail: 'fine' });
			const up = ft.length - 1;
			const at = { index: up, frame: ft.frame(up), state: ft.state(up), tape: ft };
			const u = updateOf(at, 'user.weight', [m.user, 0]);
			assert.deepEqual([u.old, u.g, u.lr, u.new], [m.u1, m.gu1, m.lr, m.nu1]);
			assert.equal(updateOf(at, 'item.weight', [m.item, 1]).new, m.nv2);
			assert.equal(ft.frame(2).phase, 'loss');
			assert.equal(ft.frame(2).metrics!.loss, m.loss);
			// step "update": raising u₁ by a tiny h raises the prediction by v₁h and the loss (e + v₁h)² by about 2ev₁h
			assert.equal(m.gu1, 2 * m.e * m.v1);
			const h = 1e-6, pred = (u1: number) => u1 * m.v1 + m.u2 * m.v2;
			assert.ok(Math.abs(pred(m.u1 + h) - pred(m.u1) - m.v1 * h) < 1e-12, fx.id);
			const slope = ((pred(m.u1 + h) - m.r) ** 2 - (pred(m.u1) - m.r) ** 2) / h;
			assert.ok(Math.abs(slope - m.gu1) <= 1e-4 * Math.max(1, Math.abs(m.gu1)), `${fx.id}: ${slope} vs ${m.gu1}`);
		},
		opts
	);
	assert.equal(nudge.defaults.lr, fit.defaults.lr);
});

test('step "watch": SGD takes one known rating per step; the average squared error falls on every input', async () => {
	const k = known(PRESETS[0].raw);
	const rng = createRng(5);
	for (let s = 0; s < 50; s++) {
		const b = fit.batch(PRESETS[0].raw, s, rng, fit.defaults);
		assert.equal(b.r.length, 1);
		assert.ok(Array.from({ length: k.n }).some((_, j) => k.u[j] === b.u[0] && k.i[j] === b.i[0] && k.r[j] === b.r[0]));
	}
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('train', 'last');
			const t = rt.sources.train.tape!;
			const a = t.frame(0).metrics!, z = t.frame(t.length - 1).metrics!;
			assert.ok(z.mse < 0.5 * a.mse, `${fx.id}: ${a.mse} -> ${z.mse}`);
			assert.equal(z.step, fit.defaults.steps);
			// "your arrow settles where it fits your ratings" / "stays where it started, because you rated nothing"
			// (an input whose error does not halve gets the live "as well as two numbers can" sentence instead)
			if (Number.isFinite(z.you)) assert.ok(fitsYou(a.you, z.you), `${fx.id}: yours ${a.you} -> ${z.you}`);
			else assert.deepEqual(vecU(last(t), YOU), vecU(factorsAt(t, 0), YOU), fx.id);
		},
		opts
	);
});

test('step "recs": the predictions are dot products and can go past +2 (the sample profile’s top pick does)', async () => {
	const t = record(trainingProcess(fit), PRESETS[0].raw, {}, { seed: 1 });
	const f = last(t);
	const best = Math.max(...MOVIES.map((_, i) => predict(f, YOU, i)));
	assert.ok(best > 2, `${best}`);
});

test('try-this "dark": likes three crime films and hates a comedy; top three are sci-fi; three of the five crime films are rated; the arrow points between the crime and sci-fi dots, nearer sci-fi; a sci-fi film got +1', () => {
	const raw = PRESETS[1].raw, r = raw.ratings;
	const of = (kind: number) => MOVIES.map((_, i) => i).filter((i) => MOVIES[i].kind === kind);
	const [comedy, scifi, crime] = [of(0), of(2), of(3)];
	assert.equal(crime.length, 5);
	assert.deepEqual(crime.filter((i) => r[i] !== null).map((i) => r[i]! > 0), [true, true, true]);
	assert.ok(comedy.some((i) => r[i] === -2) && scifi.some((i) => r[i] === 1));
	const f = last(record(trainingProcess(fit), raw, {}, { seed: 1 }));
	assert.deepEqual(ranked(f, raw).slice(0, 3).map((x) => MOVIES[x.i].kind), [2, 2, 2]);
	const dir = (v: number[]) => Math.atan2(v[1], v[0]);
	const centre = (ids: number[]) => [0, 1].map((d) => ids.reduce((s, i) => s + vecV(f, i)[d], 0) / ids.length);
	const gap = (a: number, b: number) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
	const you = dir(vecU(f, YOU)), s = dir(centre(scifi)), c = dir(centre(crime));
	assert.ok(Math.abs(gap(you, s) + gap(you, c) - gap(s, c)) < 1e-9, 'between');
	assert.ok(gap(you, s) < gap(you, c), 'nearer sci-fi');
});

test('step "cold": a row that no rating involves never changes (the new user ends at the start, on every input); a movie nobody rated would not move either', async () => {
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			await rt.ensure('cold', 'last');
			const t = rt.sources.cold.tape!;
			assert.deepEqual(vecU(last(t), YOU), vecU(factorsAt(t, 0), YOU), fx.id);
		},
		opts
	);
	// every gradient row outside the batch's person and movie is exactly 0
	const ft = record(trainingProcess(fit), PRESETS[0].raw, { steps: 20 }, { seed: 1, detail: 'fine' });
	for (let i = 0; i < ft.length; i++) {
		if (ft.frame(i).phase !== 'backward') continue;
		const s = ft.state(i) as TrainState, b = (ft.state(i - 2) as TrainState).batch!;
		const [u, it] = [b.u.data[0], b.i.data[0]];
		s.grads!['item.weight'].data.forEach((g, k) => Math.floor(k / 2) !== it && assert.equal(g, 0));
		s.grads!['user.weight'].data.forEach((g, k) => Math.floor(k / 2) !== u && assert.equal(g, 0));
	}
});
