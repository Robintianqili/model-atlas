// kalman-filter claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on
// every fixture (lesson fixtures, the adapter's edge fixtures) where it depends on the input. `atlas check` runs this file.
//   node --test src/lib/explainers/kalman-filter/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { kf } from './process.ts';
import { dataOf, KDEMO, kfPredict, kfRun, kfStart, kfUpdate, movingAverage, NOISE, presetTruth, Q0, QBIG, QS, R0, rmse, RS, START, TURN_AT, W, type TrackRaw } from './track.ts';
import { history, runCase, turnStats } from './views/track/meta.ts';
import manifest, { TURN } from './explainer.ts';
import lesson from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('kalman-filter', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const of = (t: Tape, phase: string) => all(t).filter((f) => f.phase === phase);
const at = (t: Tape, phase: string, k: number) => of(t, phase).find((f) => f.step === k)!.metrics!;
const run = (raw: TrackRaw, o: { fixed?: boolean; q?: number; r?: number } = {}) => record(kf, { ...dataOf(raw, o), ...(o.q ? { q: o.q } : {}), ...(o.r ? { r: o.r } : {}) }, {}, { seed: 1 });
const done = (t: Tape) => of(t, 'done')[0].metrics!;
const steps = lesson.steps as { id: string; task?: { done: (rt: unknown, at: unknown) => boolean } }[];
async function tapeOf(rt: AtlasRuntime, source: string): Promise<Tape | null> {
	await rt.ensure(source, 'last');
	return rt.sources[source].tape ?? null;
}
/** run fn on the 'cycle' and 'run' tapes of every fixture with a track */
async function each(fn: (t: Tape, raw: unknown, source: string) => void): Promise<void> {
	await withFixtures(await pkgP, async (rt) => {
		for (const s of ['cycle', 'run']) {
			const t = await tapeOf(rt, s);
			if (!dataOf(rt.input).z.length) return assert.equal(t?.length ?? 0, 0, 'no track, no frames');
			fn(t!, rt.input, s);
		}
	}, opts);
}
const close = (a: number, b: number, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
/** sum of squared second differences: how wiggly a line is */
const rough = (y: number[]) => y.slice(2).reduce((s, v, i) => s + (v - 2 * y[i + 1] + y[i]) ** 2, 0);

test('lesson note: ten steps; steps 2-7 work through second KDEMO of the cycle source', () => {
	assert.equal(lesson.steps.length, 10);
	assert.deepEqual(steps.map((s) => s.id), ['noisy', 'predict', 'grow', 'measure', 'gain', 'update', 'shrink', 'run', 'tune', 'limits']);
	assert.equal(manifest.input?.start, START);
});

test('noisy: every reading is the true position plus a random error; rmseZ is their root mean square', () =>
	each((t) => {
		const s = t.state(0) as { truth: Tensor; z: Tensor }, truth = Array.from(s.truth.data), z = Array.from(s.z.data);
		assert.ok(z.every((v, i) => v !== truth[i]), 'every reading is off');
		close(t.frame(0).metrics!.rmseZ, rmse(z, truth));
	}));

test('noisy: the noise is seeded normal noise of standard deviation NOISE (the readings repeat for one seed)', () => {
	const a = dataOf(START), b = dataOf(START), rng = createRng(1);
	assert.deepEqual(a.z, b.z);
	a.z.forEach((v, i) => close(v, a.truth[i] + NOISE * rng.normal()));
});

test('predict: the motion model keeps the velocity and moves the position by it (Δt = 1 s)', () =>
	each((t) => {
		for (const f of of(t, 'predict')) {
			const m = f.metrics!;
			assert.equal(m.vp, m.v);
			assert.equal(m.pp, m.p + m.v);
		}
	}));

test('grow: predicting always adds doubt (the variance and the band grow)', () =>
	each((t) => {
		for (const f of of(t, 'predict')) assert.ok(f.metrics!.Pp00 > f.metrics!.P00 && f.metrics!.sdp > f.metrics!.sd);
	}));

test('measure: the gap y is the reading minus the prediction', () =>
	each((t) => {
		for (const f of of(t, 'measure')) assert.equal(f.metrics!.y, f.metrics!.z - f.metrics!.pp);
	}));

test('gain (predict answer): at the worked second K < 1/2 because P⁻ < R, and the new estimate is closer to the prediction', () =>
	each((t, _raw, source) => {
		if (source !== 'cycle') return;
		const g = at(t, 'gain', KDEMO), u = at(t, 'update', KDEMO);
		assert.ok(g.K < 0.5 && g.Pp00 < g.r, `K ${g.K}`);
		assert.ok(Math.abs(u.p - u.pp) < Math.abs(u.p - u.z));
	}));

test('gain: K = 0 keeps the prediction, K = 1 takes the reading, K > 1/2 when the prediction is the less certain one', () => {
	const b = { p: 3, v: 1, P00: 0, P01: 0, P11: 1 };
	assert.equal(kfUpdate(b, 7, 1).b.p, 3);
	assert.equal(kfUpdate({ ...b, P00: 2 }, 7, 0).b.p, 7);
	assert.ok(kfUpdate({ ...b, P00: 2 }, 7, 1).K > 0.5);
	// the quiz: P = 3, R = 1 gives K = 0.75
	assert.equal(kfUpdate({ ...b, P00: 3 }, 7, 1).K, 0.75);
});

test('update: the estimate moves the share K of the gap; the velocity is corrected with its own gain', () =>
	each((t) => {
		for (const f of of(t, 'update')) {
			const m = f.metrics!;
			close(m.p, m.pp + m.K * m.y);
			close(m.v, m.vp + m.Kv * m.y);
			if (m.y !== 0) close((m.p - m.pp) / (m.z - m.pp), m.K, 1e-9);
		}
	}));

test('shrink: the new variance (1 − K) P⁻ is smaller than both P⁻ and R, and equals K × R', () =>
	each((t) => {
		const r = t.frame(0).metrics!.r;
		for (const f of of(t, 'update')) {
			const m = f.metrics!;
			assert.ok(m.P00 < m.Pp00 && m.P00 < r);
			close(m.P00, m.K * r);
		}
	}));

test('run: the filter, its history and the grid search use the same arithmetic; the scores are root mean squares', () =>
	each((t, raw, source) => {
		const d = dataOf(raw, { fixed: source === 'cycle' }), h = history(t, t.length - 1), k = kfRun(d.z, d.q, d.r), m = done(t);
		assert.deepEqual(h.est, k.est);
		assert.deepEqual(h.sd, k.sd);
		close(m.rmseKf, rmse(k.est, d.truth));
		close(m.rmseMa, rmse(movingAverage(d.z, W), d.truth));
		let best = Infinity;
		for (const q of QS) for (const r of RS) best = Math.min(best, rmse(kfRun(d.z, q, r).est, d.truth));
		assert.equal(m.bestRmse, best);
		assert.ok(m.bestRmse <= m.rmseKf);
	}));

test('run, prequestion: a moving average trails behind a moving cart (it averages where the cart was)', () => {
	const truth = presetTruth('cruise'), ma = movingAverage(truth, W), v = truth[1] - truth[0];
	for (let k = W - 1; k < truth.length; k++) close(ma[k] - truth[k], (-(W - 1) / 2) * v, 1e-9);
	assert.ok(v > 0);
});

test('prequestion, start page: the filter beats both the latest readings and the moving average on the start track', () => {
	const m = done(run(START, { fixed: true }));
	assert.ok(m.rmseKf < m.rmseZ && m.rmseKf < m.rmseMa);
});

test('tune: a small R or a large Q makes the filter trust the readings (a larger final K) and copy their noise; a large R or a small Q makes it react late', () => {
	const K = (q: number, r: number) => {
		let b = kfStart(0, r), K = 0;
		for (let k = 1; k < 60; k++) {
			const u = kfUpdate(kfPredict(b, q), 0, r);
			(b = u.b), (K = u.K);
		}
		return K;
	};
	for (const r of RS) for (let i = 1; i < QS.length; i++) assert.ok(K(QS[i], r) > K(QS[i - 1], r));
	for (const q of QS) for (let i = 1; i < RS.length; i++) assert.ok(K(q, RS[i]) < K(q, RS[i - 1]));
	const cruise = dataOf({ preset: 'cruise', seed: 1 }), turn = dataOf(TURN);
	assert.ok(rough(kfRun(cruise.z, 1, 0.1).est) > 10 * rough(kfRun(cruise.z, 0.001, 8).est), 'trusting the readings copies their noise');
	const late = turnStats(run(TURN, { q: 0.001, r: 8 }), 1e9, TURN_AT), quick = turnStats(run(TURN, { q: 1, r: 0.1 }), 1e9, TURN_AT);
	assert.ok(late.worst > quick.worst && late.post > quick.post, 'trusting the model reacts late');
	assert.ok(turn.z.length);
});

test('tune (task): the starting settings do not finish the task on the start track; the best pair always can', async () => {
	const task = steps.find((s) => s.id === 'tune')!.task!;
	const start = run(START), best = done(start);
	assert.equal(task.done(null, { frame: of(start, 'done')[0] }), false);
	const tuned = run({ ...START, q: best.bestQ, r: best.bestR });
	assert.equal(task.done(null, { frame: of(tuned, 'done')[0] }), true);
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'run');
		if (!t?.length) return;
		const m = done(t), d = dataOf(rt.input);
		assert.equal(rmse(kfRun(d.z, m.bestQ, m.bestR).est, d.truth), m.bestRmse);
	}, opts);
});

test('limits: after the sudden turn the estimate overshoots, the truth leaves the band; a larger Q recovers sooner but is worse before the turn', () => {
	const a = run(TURN, { fixed: true }), b = run(TURN, { fixed: true, q: QBIG }), h = history(a, a.length - 1), truth = presetTruth('turn');
	assert.equal(Q0, 0.01);
	assert.ok(h.est[TURN_AT + 1] > truth[TURN_AT + 1] && h.est[TURN_AT + 2] > truth[TURN_AT + 2], 'it keeps going');
	const sa = turnStats(a, a.length - 1, TURN_AT), sb = turnStats(b, b.length - 1, TURN_AT);
	assert.ok(sa.out > sb.out && sa.out >= 3, `outside ${sa.out} vs ${sb.out}`);
	assert.ok(sb.pre > sa.pre && sb.post < sa.post && sb.worst < sa.worst);
	// the sources the step compares are these runs
	assert.equal(done(b).q, QBIG);
	assert.equal(done(a).r, R0);
});

test('limits (look): the band never depends on the readings, and the surprise y jumps after the turn', () => {
	const turn = run(TURN, { fixed: true }), cruise = run({ preset: 'cruise', seed: 4 }, { fixed: true });
	assert.deepEqual(history(turn, turn.length - 1).sd, history(cruise, cruise.length - 1).sd);
	const y = of(turn, 'measure').map((f) => f.metrics!.y), before = Math.sqrt(y.slice(5, TURN_AT - 1).reduce((s, v) => s + v * v, 0) / (TURN_AT - 6));
	assert.ok(Math.max(...y.slice(TURN_AT, TURN_AT + 3).map(Math.abs)) > 2 * before);
});

test('glossary (variance): ±2 standard deviations hold a normal error most of the time (about 95 in 100)', () => {
	const rng = createRng(11);
	let inside = 0;
	for (let i = 0; i < 20000; i++) if (Math.abs(rng.normal()) <= 2) inside++;
	assert.ok(inside / 20000 > 0.94 && inside / 20000 < 0.96);
});

test('try this: on the wave Q = 0.3 beats the starting Q; on the steady cruise the best pair has the smallest Q and the largest R', () => {
	const wave = dataOf({ preset: 'wave', seed: 1 }, { fixed: true });
	assert.ok(rmse(kfRun(wave.z, 0.3, R0).est, wave.truth) < rmse(kfRun(wave.z, Q0, R0).est, wave.truth));
	const m = done(run({ preset: 'cruise', seed: 1 }));
	assert.equal(m.bestQ, QS[0]);
	assert.equal(m.bestR, RS[RS.length - 1]);
});

// review (teaching reviewer, 2026-10-07): claims behind the reworded steps

test('predict: the filter started from the first reading (velocity 0) and makes its two moves every second', () =>
	each((t) => {
		const f = t.frame(0), z = (t.state(0) as { z: Tensor }).z.data;
		assert.equal(f.phase, 'start');
		assert.equal(f.metrics!.p, z[0]);
		assert.equal(f.metrics!.v, 0);
		const n = z.length;
		assert.equal(of(t, 'predict').length, n - 1);
		assert.equal(of(t, 'update').length, n - 1);
	}));

test('grow (look and formula): P⁻ adds the position’s doubt, the velocity’s, twice their covariance and Q/4; Q/4 is the push a/2 of a push a of variance Q', async () => {
	await each((t) => {
		for (const f of of(t, 'predict')) {
			const m = f.metrics!;
			close(m.Pp00, m.P00 + 2 * m.P01 + m.P11 + m.q / 4);
		}
	});
	// with no doubt before, predicting leaves exactly the push: variance q for the velocity, q/4 for the position, q/2 shared
	const b = kfPredict({ p: 0, v: 0, P00: 0, P01: 0, P11: 0 }, 0.08);
	close(b.P11, 0.08);
	close(b.P00, 0.08 / 4);
	close(b.P01, 0.08 / 2);
});

test('update: the surprise also corrects the velocity, in the same direction (its gain Kv is positive at the worked second)', () =>
	each((t, _raw, source) => {
		if (source !== 'cycle') return;
		const u = at(t, 'update', KDEMO);
		assert.ok(u.Kv > 0, `Kv ${u.Kv}`);
		assert.ok(Math.sign(u.v - u.vp) === Math.sign(u.y) || u.y === 0);
	}));

test('shrink (look): the ellipse narrows more across (position) than up (velocity) at every update', () =>
	each((t) => {
		for (let i = 1; i < t.length; i++) {
			if (t.frame(i).phase !== 'update') continue;
			const A = (t.state(i - 1) as { P: Tensor }).P.data, B = (t.state(i) as { P: Tensor }).P.data;
			assert.ok(B[0] / A[0] < B[3] / A[3], `frame ${i}: ${B[0] / A[0]} vs ${B[3] / A[3]}`);
		}
	}));

test('run (verdict): wins = the filter beats both; tune = the best pair beats both; model = not even the best pair, only on a track that jumps', async () => {
	const seen = new Set<string>();
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'run');
		if (!t?.length) return;
		const m = done(t), c = runCase(m), truth = dataOf(rt.input).truth;
		seen.add(c);
		if (c === 'wins') assert.ok(m.rmseKf < m.rmseZ && m.rmseKf < m.rmseMa);
		else assert.ok(m.rmseKf >= Math.min(m.rmseZ, m.rmseMa));
		if (c === 'tune') assert.ok(m.bestRmse < m.rmseZ && m.bestRmse < m.rmseMa);
		if (c === 'model') {
			assert.ok(m.bestRmse >= Math.min(m.rmseZ, m.rmseMa));
			// the cart does not move the way a steady-speed model expects: its speed jumps by 5 m/s or more in one second
			assert.ok(Math.max(...truth.slice(2).map((v, i) => Math.abs(v - 2 * truth[i + 1] + truth[i]))) >= 5);
		}
	}, opts);
	assert.ok(seen.has('wins') && seen.has('tune') && seen.has('model'), [...seen].join(','));
	// the start track: the filter wins with the starting settings
	assert.equal(runCase(done(run(START))), 'wins');
});
