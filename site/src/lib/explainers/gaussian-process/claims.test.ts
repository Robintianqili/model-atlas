// gaussian-process claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on
// every fixture (lesson fixtures, the adapter's edge fixtures) where it depends on the input. `atlas check` runs this file.
//   node --test src/lib/explainers/gaussian-process/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { gp } from './process.ts';
import { dataOf, evidence, G, GRID, kern, LONG, LS, PRESETS, posterior, SHORT, SN2, START, STEP, type GpRaw } from './gp.ts';
import manifest from './explainer.ts';
import lesson, { CLOSE } from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('gaussian-process', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const of = (t: Tape, phase: string) => all(t).filter((f) => f.phase === phase);
const one = (t: Tape, phase: string) => of(t, phase)[0].metrics!;
const arr = (t: Tape, i: number, k: string) => Array.from(((t.state(i) as Record<string, Tensor>)[k]).data);
const run = (raw: GpRaw, o: { ls?: number } = {}) => record(gp, dataOf(raw, o), {}, { seed: 1 });
/** run fn on the 'gp' tape of every fixture with points */
async function each(fn: (t: Tape, raw: unknown) => void): Promise<void> {
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('gp', 'last');
		const t = rt.sources.gp.tape ?? null;
		if (!dataOf(rt.input).x.length) return assert.equal(t?.length ?? 0, 0, 'no points, no frames');
		fn(t!, rt.input);
	}, opts);
}
const close = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
/** sum of squared second differences: how wiggly a curve on the grid is */
const rough = (y: number[]) => y.slice(2).reduce((s, v, i) => s + (v - 2 * y[i + 1] + y[i]) ** 2, 0);

test('lesson note: eleven steps in this order', () => {
	assert.deepEqual(lesson.steps.map((s) => (s as { id: string }).id), ['prior', 'kernel', 'first', 'gap', 'points', 'matrix', 'samples', 'evidence', 'tune', 'cost', 'choice']);
	assert.equal(manifest.input?.start, START);
});

test('prior: mean 0 and band ±2√σf² at every x; the curves are drawn with the seeded draws', () =>
	each((t) => {
		const p = one(t, 'prior');
		assert.ok(arr(t, 0, 'mean').every((v) => v === 0));
		arr(t, 0, 'sd').forEach((v) => close(2 * v, p.band0));
		close(p.band0, 2 * Math.sqrt(p.sf2));
	}));

test('prior: a shorter length-scale makes the prior functions wiggle faster (same draws, five seeds)', () => {
	for (let seed = 1; seed <= 5; seed++) {
		const r = [LS[0], 1, LS[LS.length - 1]].map((ls) => { const v = arr(run({ preset: 'gap', seed }, { ls }), 0, 'samples'); return [0, 1, 2].reduce((a, k) => a + rough(v.slice(k * G, (k + 1) * G)), 0); });
		assert.ok(r[0] > r[1] && r[1] > r[2], `seed ${seed}: ${r.join(', ')}`);
	}
});

test('kernel: K holds k(xᵢ, xⱼ) = σf² exp(−d²/(2ℓ²)) with σf² + σn² on the diagonal; the pairs are the closest and the farthest', () =>
	each((t, raw) => {
		const d = dataOf(raw), k = one(t, 'kernel'), K = arr(t, t.length - 1, 'K'), n = d.x.length;
		for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) close(K[i * n + j], d.sf2 * Math.exp(-((d.x[i] - d.x[j]) ** 2) / (2 * d.ls * d.ls)) + (i === j ? SN2 : 0), 1e-12);
		close(k.kd, d.sf2 + SN2);
		if (n < 2) return assert.equal(k.iN, undefined);
		const ds = d.x.flatMap((a, i) => d.x.slice(i + 1).map((b) => Math.abs(a - b)));
		close(k.dN, Math.min(...ds));
		close(k.dF, Math.max(...ds));
		close(k.kN, kern(0, k.dN, d.ls, d.sf2));
	}));

test('first: at x₀ the mean is σf²/(σf² + σn²) × y₀, almost y₀; at one length-scale the kernel is σf² e^−½ and the band is wider but narrower than the prior', () =>
	each((t, raw) => {
		const d = dataOf(raw), a = of(t, 'add')[0].metrics!;
		close(a.mAt, (d.sf2 / (d.sf2 + SN2)) * d.y[0]);
		assert.ok(a.w > 0.95, 'σn² is small next to every σf² on the slider');
		close(Math.abs(a.xp - d.x[0]), d.ls);
		close(a.kp, d.sf2 * Math.exp(-0.5));
		assert.ok(Math.abs(a.mp) <= Math.abs(a.mAt) + 1e-12, 'the mean moves part of the way');
		assert.ok(a.sdp > a.sdAt && a.sdp < a.sd0, 'the band at xₚ is wider than at x₀, narrower than the prior');
		// look: farther away (three length-scales) the band is back at (nearly) its full prior width
		const sd = arr(t, all(t).findIndex((f) => f.phase === 'add'), 'sd');
		GRID.forEach((x, g) => Math.abs(x - d.x[0]) >= 3 * d.ls && assert.ok(sd[g] > 0.99 * a.sd0, `x = ${x}`));
	}));

test('gap (predict): the band is widest inside the gap, narrow at the points at its ends, most of the prior band once the gap is three length-scales wide', () =>
	each((t) => {
		const d = one(t, 'done');
		if (d.gw === undefined) return;
		if (d.gwl >= 1) assert.ok(d.sdMid > d.sdA && d.sdMid > d.sdB, 'wider in the middle than at the ends');
		else assert.ok(d.sdMid - Math.min(d.sdA, d.sdB) < 0.1 * d.sd0, 'a narrow gap: it barely widens');
		assert.ok(d.sdA < 0.3 * d.sd0 && d.sdB < 0.3 * d.sd0, 'narrow at the points');
		if (d.gwl >= 3) assert.ok(d.sdMid > 0.5 * d.sd0, 'most of the prior band');
	}));

test('gap (predict): in a gap much narrower than a length-scale the band barely widens', () => {
	for (const ls of LS) {
		const p = posterior([-0.25 * ls, 0.25 * ls], [0.3, -0.2], ls, 1, SN2, [-0.25 * ls, 0, 0.25 * ls]);
		assert.ok(p.sd[1] < 0.1, `ℓ = ${ls}: ${p.sd[1]}`);
	}
});

test('points: each new point pulls the mean at its x towards its y and pinches the band there', () =>
	each((t, raw) => {
		const d = dataOf(raw);
		for (let i = 1; i < d.x.length; i++) {
			const before = posterior(d.x.slice(0, i), d.y.slice(0, i), d.ls, d.sf2, SN2, [d.x[i]]), after = posterior(d.x.slice(0, i + 1), d.y.slice(0, i + 1), d.ls, d.sf2, SN2, [d.x[i]]);
			assert.ok(Math.abs(after.mean[0] - d.y[i]) <= Math.abs(before.mean[0] - d.y[i]) + 1e-12, `point ${i}: pulled towards`);
			assert.ok(after.sd[0] <= before.sd[0] + 1e-12, `point ${i}: pinched`);
		}
	}));

test('matrix: (K + σn² I) α = y, and the mean anywhere is Σ αᵢ k(x, xᵢ)', () =>
	each((t, raw) => {
		const d = dataOf(raw), n = d.x.length, i = all(t).findIndex((f) => f.phase === 'kernel'), K = arr(t, i, 'K'), a = arr(t, i, 'alpha');
		for (let r = 0; r < n; r++) close(K.slice(r * n, (r + 1) * n).reduce((s, v, j) => s + v * a[j], 0), d.y[r], 1e-8);
		const last = all(t).filter((f) => f.phase === 'add').length, mean = arr(t, i + last, 'mean');
		GRID.forEach((x, g) => close(mean[g], a.reduce((s, v, j) => s + v * kern(x, d.x[j], d.ls, d.sf2), 0), 1e-9));
		close(one(t, 'kernel').mq, a.reduce((s, v, j) => s + v * kern(0, d.x[j], d.ls, d.sf2), 0));
	}));

test('samples: xFar is the grid point farthest from every point, and there the samples spread more than next to point 0', () =>
	each((t, raw) => {
		const d = dataOf(raw), s = one(t, 'sample'), gap = (x: number) => Math.min(...d.x.map((v) => Math.abs(x - v)));
		assert.ok(GRID.every((x) => gap(x) <= s.dFar + 1e-12));
		assert.ok(s.spreadFar > s.spreadNear, `${s.spreadFar} vs ${s.spreadNear}`);
	}));

test('evidence: log p = fit + complexity + constant; the complexity term is lower for more flexible (shorter) length-scales; the best is the highest on the slider', () =>
	each((t, raw) => {
		const d = dataOf(raw), e = one(t, 'evidence'), i = all(t).findIndex((f) => f.phase === 'evidence'), l = arr(t, i, 'lmls');
		close(e.lml, e.fit + e.cpx + e.cst);
		close(e.cst, -0.5 * d.x.length * Math.log(2 * Math.PI));
		const c = LS.map((ls) => evidence(d.x, d.y, ls, d.sf2, SN2).cpx);
		c.slice(1).forEach((v, j) => assert.ok(v >= c[j] - 1e-12, `ℓ = ${LS[j + 1]}`));
		close(e.bestLml, Math.max(...l));
		assert.ok(e.drop >= 0);
	}));

test('tune: too short a length-scale wiggles and balloons between the points, too long a one cannot follow them', () => {
	for (const p of PRESETS) {
		const s = run({ preset: p.id }, { ls: LS[0] }), l = run({ preset: p.id }, { ls: LS[LS.length - 1] });
		const meanOf = (t: Tape) => arr(t, all(t).findIndex((f) => f.phase === 'sample'), 'mean');
		assert.ok(rough(meanOf(s)) > rough(meanOf(l)), `${p.id}: wiggles`);
		assert.ok(one(s, 'done').avgBand > one(l, 'done').avgBand, `${p.id}: balloons`);
		assert.ok(one(l, 'done').maxRes > one(s, 'done').maxRes, `${p.id}: cannot follow`);
	}
});

test('tune: the task is done at the best length-scale of the start points (ℓ = 2), not at the start (ℓ = 1)', () => {
	const at = (raw: GpRaw) => ({ frame: of(run(raw), 'evidence')[0] });
	const step = lesson.steps.find((s) => (s as { id: string }).id === 'tune') as { task: { done: (rt: unknown, at: unknown) => boolean } };
	assert.equal(step.task.done(null, at(START)), false);
	assert.equal(step.task.done(null, at({ preset: 'gap', ls: 2 })), true);
	assert.ok(one(run(START), 'evidence').drop > CLOSE);
});

test('cost: the Cholesky loop does (n³ − n)/3 multiplications and subtractions, about n³/3; ten times the points cost 100 times the memory and 1000 times the work', () =>
	each((t) => {
		const d = one(t, 'done'), n = d.n;
		let ops = 0;
		for (let j = 0; j < n; j++) ops += 2 * (j + (n - j - 1) * j);
		assert.equal(ops, (n ** 3 - n) / 3);
		close(d.ops, n ** 3 / 3);
		close(d.mem, n * n);
		close(d.memBig / d.mem, (d.big / n) ** 2);
		assert.equal((10 * n) ** 2 / n ** 2, 100);
		assert.equal((10 * n) ** 3 / n ** 3, 1000);
	}));

test('choice: on the step data the short length-scale overshoots and balloons, the long one misses the points beside the step and overshoots too', () => {
	const s = one(run(STEP, { ls: SHORT }), 'done'), l = one(run(STEP, { ls: LONG }), 'done');
	assert.ok(s.peak > s.ymax && l.peak > l.ymax, 'both overshoot');
	assert.ok(s.avgBand > 2 * l.avgBand, 'the short one balloons');
	assert.ok(l.maxRes > 10 * s.maxRes && l.maxRes > 0.3, 'the long one misses');
	const x = dataOf(STEP).x, near = x.reduce((b, v) => (Math.abs(v) < Math.abs(b) ? v : b), Infinity);
	const p = posterior(x, dataOf(STEP).y, LONG, 1, SN2, [near]);
	close(Math.abs(p.mean[0] - Math.sign(near)), l.maxRes, 1e-9);
});

test('page: the quiz numbers (exp −½, −2, −4.5) and the 95-in-100 band of a normal distribution', () => {
	assert.equal(Math.exp(-0.5).toFixed(2), '0.61');
	assert.equal(Math.exp(-2).toFixed(2), '0.14');
	assert.equal(Math.exp(-4.5).toFixed(3), '0.011');
	let s = 0;
	const N = 200000;
	for (let i = 0; i < N; i++) {
		const z = -2 + (4 * (i + 0.5)) / N;
		s += (Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI)) * (4 / N);
	}
	assert.equal(Math.round(s * 100), 95);
	assert.equal(G, GRID.length);
});
