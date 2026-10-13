// optimizers claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here; lesson sources run on every fixture through
// withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/optimizers/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { input, param } from '../../atlas/autograd/var.ts';
import { Adam } from '../../atlas/autograd/optim.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { CLIFF, LAMBDA, RACERS, SCHED, SURFACES, clipByNorm, leagueKey, lrAt, surfaceOf, type ScheduleId } from './data.ts';
import { cliff, league, race, range, sched } from './process.ts';
import { MOONS, PARAMS } from './net.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('optimizers', SITE)).pkg;
const last = (t: { length: number; frame(i: number): Frame }) => t.frame(t.length - 1);
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const VALLEY = { surface: 'valley', x: -4, y: 1 };
const raceOn = (id: 'valley' | 'saddle' | 'bumpy', y?: number) => record(race, { surface: id, x: surfaceOf(id).start[0], y: y ?? surfaceOf(id).start[1] }, { steps: 80 });

test('the surfaces: each gradient matches finite differences; every listed bottom has loss 0 and gradient 0; the network has 114 weights', () => {
	for (const s of SURFACES) {
		for (const [fx, fy] of [[0.2, 0.3], [0.7, 0.1], [0.5, 0.9], [0.9, 0.6]]) {
			const x = s.xr[0] + fx * (s.xr[1] - s.xr[0]);
			const y = s.yr[0] + fy * (s.yr[1] - s.yr[0]);
			const h = 1e-6;
			const g = s.grad(x, y);
			const nx = (s.f(x + h, y) - s.f(x - h, y)) / (2 * h);
			const ny = (s.f(x, y + h) - s.f(x, y - h)) / (2 * h);
			assert.ok(close(g[0], nx, 1e-5) && close(g[1], ny, 1e-5), `${s.id} at (${x}, ${y}): ${g} vs ${nx}, ${ny}`);
		}
		for (const q of s.min) {
			assert.ok(Math.abs(s.f(q[0], q[1])) < 1e-15, `${s.id} bottom`);
			assert.ok(Math.hypot(...s.grad(q[0], q[1])) < 1e-12);
		}
	}
	assert.equal(PARAMS, 114);
	assert.equal(MOONS.n, 120);
});

test('the race runs the kit optimizers by their formulas: SGD old − η·g, momentum v ← μ·v + g, AdamW decays first', () => {
	const t = record(race, VALLEY, { steps: 12 }, { detail: 'fine' });
	for (let i = 0; i < t.length; i++) {
		const f = t.frame(i);
		const m = f.metrics!;
		if (f.phase === 'sgd') assert.ok(close(m.x1, m.x - m.lr * m.gx, 1e-12) && close(m.y1, m.y - m.lr * m.gy, 1e-12));
		if (f.phase === 'momentum') {
			assert.ok(close(m.bx1, m.mu * m.bx + m.gx, 1e-12) && close(m.by1, m.mu * m.by + m.gy, 1e-12));
			assert.ok(close(m.x1, m.x - m.lr * m.bx1, 1e-12) && close(m.y1, m.y - m.lr * m.by1, 1e-12));
		}
		if (f.phase === 'adamw') {
			assert.equal(m.wd, LAMBDA);
			const s = (v: number, mh: number, sv: number) => (1 - m.lr * m.wd) * v - (m.lr * mh) / (sv + m.eps);
			assert.ok(close(m.x1, s(m.x, m.mhx, m.shx), 1e-12) && close(m.y1, s(m.y, m.mhy, m.shy), 1e-12), `adamw step ${f.step}`);
		}
	}
});

test('step "sgd" and the pre-question: SGD flips y across the floor while x creeps; after 10 updates x is still far from the bottom', () => {
	const t = raceOn('valley');
	let flips = 0;
	for (let k = 1; k <= 10; k++) if (Math.sign(t.frame(k).metrics!.sgdY) !== Math.sign(t.frame(k - 1).metrics!.sgdY)) flips++;
	assert.ok(flips >= 5, `flips ${flips}`);
	assert.ok(t.frame(1).metrics!.sgdY < 0, 'the first update passes the floor');
	assert.ok(t.frame(10).metrics!.sgdX < -3, `x after 10: ${t.frame(10).metrics!.sgdX}`);
});

test('step "momentum" (predict) and pitfall 1: after 20 updates momentum is further along than SGD but still swinging; along the way the slope along x keeps one sign', () => {
	const t = raceOn('valley');
	const m = t.frame(20).metrics!;
	assert.ok(m.momentumX > m.sgdX + 0.5, `${m.momentumX} vs ${m.sgdX}`);
	assert.ok(Math.abs(m.momentumY) > 0.1 && Math.abs(m.sgdY) < 1e-3, `y: ${m.momentumY} vs ${m.sgdY}`);
	for (let k = 0; k <= 20; k++) assert.ok(t.frame(k).metrics!.momentumX < 1, 'x stays below the bottom, so the slope along x keeps its sign');
	let crossings = 0;
	for (let k = 1; k < t.length; k++) if (Math.sign(t.frame(k).metrics!.momentumY) !== Math.sign(t.frame(k - 1).metrics!.momentumY)) crossings++;
	assert.ok(crossings >= 3, `momentum crosses the floor ${crossings} times`);
});

test('step "momentum-update": at update 9 the slope along x agrees with the velocity and the new velocity is larger in size; across y they disagree and partly cancel', () => {
	const t = record(race, VALLEY, { steps: 12 }, { detail: 'fine' });
	const m = t.frame(t.find({ phase: 'momentum', step: 9 })).metrics!;
	assert.ok(m.gx * m.bx > 0 && Math.abs(m.bx1) > Math.abs(m.bx), `along: g ${m.gx}, v ${m.bx} -> ${m.bx1}`);
	assert.ok(m.gy * m.by < 0 && Math.abs(m.by1) < Math.min(Math.abs(m.by), Math.abs(m.gy)), `across: g ${m.gy}, v ${m.by} -> ${m.by1}`);
});

test('step "adam" and quiz 1: Adam’s first update moves every weight by η against its slope, whatever the slope’s size', () => {
	const t = record(race, VALLEY, { steps: 1 }, { detail: 'fine' });
	const m = t.frame(t.find({ phase: 'adam', step: 0 })).metrics!;
	assert.ok(Math.abs(m.gy / m.gx) > 10);
	assert.ok(close(m.dx, -Math.sign(m.gx) * m.lr, 1e-6) && close(m.dy, -Math.sign(m.gy) * m.lr, 1e-6), `${m.dx}, ${m.dy}`);
	const p = param('p', { shape: [2], data: [0, 0] });
	const o = new Adam([p], { lr: 0.1 });
	p.grad = input({ shape: [2], data: [0.2, -40] });
	o.step();
	assert.ok(close(p.data[0], -0.1, 1e-6) && close(p.data[1], 0.1, 1e-6), `quiz: ${p.data[0]}, ${p.data[1]}`);
});

test('step "yours": from a start exactly on the saddle’s ridge no optimizer leaves it', () => {
	const t = raceOn('saddle', 0);
	const m = last(t).metrics!;
	for (const r of RACERS) assert.equal(m[`${r.key}Y`], 0);
});

test('step "league", the misconception and the compare panel: three landscapes, three different winners; on the bumpy bowl SGD and Adam stop in dips that momentum rolls past', () => {
	const t = record(league, null, { steps: 80 });
	const m = last(t).metrics!;
	const win = (s: 'valley' | 'saddle' | 'bumpy') => RACERS.reduce((b, r) => (m[leagueKey(s, r.key)] < m[leagueKey(s, b.key)] ? r : b)).key;
	assert.deepEqual([win('valley'), win('saddle'), win('bumpy')], ['adam', 'sgd', 'momentum']);
	const b = last(raceOn('bumpy')).metrics!;
	const s = surfaceOf('bumpy');
	// each sits in a dip: slow gradient descent from where it stopped settles in a local bottom above momentum's loss
	for (const k of ['sgd', 'adam'] as const) {
		let [x, y] = [b[`${k}X`], b[`${k}Y`]];
		for (let i = 0; i < 5000; i++) {
			const g = s.grad(x, y);
			x -= 0.05 * g[0];
			y -= 0.05 * g[1];
		}
		assert.ok(Math.hypot(...s.grad(x, y)) < 1e-9 && s.f(x, y) > 1.5 * b.momentumLoss && b[`${k}Loss`] > 1.5 * b.momentumLoss, `${k}: dip at ${s.f(x, y)}, momentum ${b.momentumLoss}`);
	}
});

test('steps "range" and "schedule": the range test falls fastest, then bottoms out, then blows up; the demo peak η sits just below the blow-up and the test came close to it', () => {
	const t = record(range, null, { steps: 100 });
	const e = last(t).metrics!;
	assert.ok(e.steepLr < e.lowLr && e.lowLr < e.blowLr, `${e.steepLr} ${e.lowLr} ${e.blowLr}`);
	assert.ok(SCHED.peak < e.blowLr && e.blowLr < 1.25 * SCHED.peak, `peak ${SCHED.peak}, blow-up ${e.blowLr}`);
	let near = false;
	for (let i = 0; i < t.length - 1; i++) {
		const f = t.frame(i).metrics!;
		if (f.lr >= 0.95 * SCHED.peak && f.lr <= SCHED.peak && f.loss < e.startLoss) near = true;
	}
	assert.ok(near, 'the range test came close to the peak with a loss below its start');
});

test('steps "schedule", "warmup", pitfall 2 and quiz 3: warm-up climbs to the peak in 20 steps; at the peak every schedule without warm-up ends no better than a coin flip, every one with warm-up trains', () => {
	assert.equal(lrAt('constant', true, 0), 0.2);
	assert.equal(lrAt('cosine', true, SCHED.warm - 1), SCHED.peak);
	for (const schedule of ['constant', 'step', 'cosine'] as ScheduleId[]) {
		const cold = last(record(sched, { schedule, warmup: false }, { steps: SCHED.steps })).metrics!;
		const warm = last(record(sched, { schedule, warmup: true }, { steps: SCHED.steps })).metrics!;
		assert.ok(cold.loss > Math.LN2 - 0.005, `${schedule} without warm-up ends at ${cold.loss}`);
		// Edge ends step decay with warm-up at 0.0534, Node at 0.0214 (a chaotic run at η = 4: last-bit differences of Math.tanh
		// grow); the claim is the size of the gap, not the digits
		assert.ok(warm.loss < 0.1, `${schedule} with warm-up ends at ${warm.loss}`);
		assert.ok(cold.lossBase > Math.LN2, 'the constant cold start fails');
	}
});

test('step "clip", pitfall 3 and quiz 2: without clipping the run is thrown off the cliff; with clipping every step is at most η·c long, points the way the gradient does, and the run ends bouncing across the floor', () => {
	const c = clipByNorm([30, 40], 1);
	assert.ok(close(c.g[0], 0.6, 1e-6) && close(c.g[1], 0.8, 1e-6));
	const t = record(cliff, { clip: false }, { steps: CLIFF.steps });
	const e = last(t).metrics!;
	assert.ok(e.plainGone >= 2 && e.clipGone === 0, `plain gone at ${e.plainGone}`);
	assert.ok(t.frame(e.plainGone).metrics!.plainLoss > CLIFF.blowUp);
	assert.ok(t.frame(e.plainGone - 1).metrics!.plainW > 0.95, 'the update before lands next to the cliff');
	const s = surfaceOf('cliff');
	const losses: number[] = [];
	for (let i = 1; i < t.length; i++) {
		const m = t.frame(i).metrics!;
		const p = t.frame(i - 1).metrics!;
		assert.ok(m.clipLen <= CLIFF.lr * CLIFF.maxNorm + 1e-12, `step ${i}: ${m.clipLen}`);
		const g = s.grad(p.clipW, p.clipB);
		const d = [m.clipW - p.clipW, m.clipB - p.clipB];
		assert.ok(Math.abs(g[0] * d[1] - g[1] * d[0]) <= 1e-9 * Math.hypot(...g) * Math.hypot(...d) + 1e-15 && g[0] * d[0] + g[1] * d[1] <= 0, 'the step points against the gradient');
		losses.push(m.clipLoss);
	}
	const tail = losses.slice(-6);
	for (let i = 1; i < tail.length; i++) assert.ok(Math.sign(tail[i] - tail[i - 1]) !== Math.sign(tail[i - 1] - (i > 1 ? tail[i - 2] : losses[losses.length - 7])), 'the end alternates up and down');
	assert.ok(losses[losses.length - 1] > 100 * Math.min(...losses), 'it bounces above its lowest point');
});

test('every lesson source runs on every fixture (deep links: each step can be the first one entered)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			for (const s of ['valley', 'race', 'league', 'cliff', 'sched']) {
				await rt.ensure(s, 'last');
				assert.ok((rt.sources[s].tape as Tape).length > 1, s);
			}
		},
		opts
	);
});
