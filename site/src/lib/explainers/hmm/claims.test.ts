// hmm claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (lesson fixtures, the adapter's edge fixtures, the presets) where it depends on the input. Most claims are
// checked against brute force: every one of the 2ⁿ weather paths scored on its own. `atlas check` runs this file.
//   node --test src/lib/explainers/hmm/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import type { Frame, Tape } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { sample } from './process.ts';
import { A0, B0, dataOf, FREE, jointOf, K, M, MINP, naiveOf, PI0, RAIN, setProb, START, START_OBS, viterbiOf, YEAR, type HmmData } from './hmm.ts';
import { latOf } from './views/lattice/meta.ts';
import manifest from './explainer.ts';
import lesson from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('hmm', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const of = (t: Tape, phase: string) => all(t).filter((f) => f.phase === phase);
const close = (a: number, b: number, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1e-300, Math.abs(b)), `${a} vs ${b}`);
async function tapeOf(rt: AtlasRuntime, source: string): Promise<Tape | null> {
	await rt.ensure(source, 'last');
	return rt.sources[source].tape ?? null;
}
/** every weather path of n days with its joint probability */
function paths(d: HmmData): { p: number[]; pr: number }[] {
	const n = d.obs.length, out: { p: number[]; pr: number }[] = [];
	for (let c = 0; c < K ** n; c++) {
		const p = Array.from({ length: n }, (_, t) => Math.floor(c / K ** t) % K);
		out.push({ p, pr: jointOf(d, p) });
	}
	return out;
}
/** fn on the forward and Viterbi tapes of every fixture with at least one day, with brute force */
async function each(fn: (d: HmmData, fwd: Tape, vit: Tape, all: { p: number[]; pr: number }[]) => void): Promise<void> {
	await withFixtures(await pkgP, async (rt) => {
		const d = dataOf(rt.input), fwd = await tapeOf(rt, 'fwd'), vit = await tapeOf(rt, 'vit');
		if (!d.obs.length) return assert.equal((fwd?.length ?? 0) + (vit?.length ?? 0), 0, 'no days, no frames');
		fn(d, fwd!, vit!, paths(d));
	}, opts);
}

test('lesson note: ten steps in this order, with predict, task and limits steps', () => {
	assert.deepEqual(lesson.steps.map((s) => (s as { id: string }).id), ['story', 'sample', 'path', 'forward', 'fill', 'viterbi', 'back', 'task', 'limits', 'rnn']);
	assert.equal(manifest.input?.start, START);
});

test('tables: every row of every fixture sums to 1 with entries of at least MINP; broken rows fall back to the start', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const d = dataOf(rt.input);
		for (const row of [d.pi, ...d.A, ...d.B]) {
			close(row.reduce((a, b) => a + b, 0), 1);
			assert.ok(row.every((v) => v >= MINP / 100 - 1e-12));
		}
	}, opts);
	assert.deepEqual(dataOf({ obs: [0], pi: [70, 70], A: [A0[0], [50, 60]], B: [[1, 2, 3], B0[1]] }).pi, PI0.map((v) => v / 100));
});

test('Tables sliders: setProb keeps a row summing to 100, every entry within [MINP, 100 − MINP·(n−1)]', () => {
	for (const row of [[75, 25], [50, 35, 15], [5, 5, 90], [34, 33, 33]])
		for (let j = 0; j < row.length; j++)
			for (let v = 0; v <= 100; v += 5) {
				const r = setProb(row, j, v);
				assert.equal(r.reduce((a, b) => a + b, 0), 100, `${row} ${j} ${v}`);
				assert.ok(r.every((x) => Number.isInteger(x) && x >= MINP && x <= 100 - MINP * (row.length - 1)), `${r}`);
			}
});

test('sample: each draw lands in its interval of the row it was drawn from; later days use the transition row of the day before', () => {
	for (const raw of [START, { obs: [0], A: [[95, 5], [10, 90]] }]) {
		const { pi, A, B } = dataOf(raw), t = record(sample, { pi, A, B }, { days: 12 }, { seed: 1 });
		for (const f of of(t, 'day')) {
			const m = f.metrics!, row = m.t ? A[m.prev] : pi, cum = (r: number[], i: number) => r.slice(0, i).reduce((a, b) => a + b, 0);
			assert.ok(m.u >= m.lo && m.u < m.hi && m.v >= m.alo && m.v < m.ahi);
			close(m.lo, cum(row, m.s), 1e-12);
			close(m.alo, cum(B[m.s], m.act), 1e-12);
		}
	}
});

test('path: the guess gives each day the weather under which its activity is likelier, and its product is the one narrated', () =>
	each((d, fwd) => {
		const m = fwd.frame(0).metrics!, g = latOf(fwd, 0).guess;
		g.forEach((s, t) => assert.ok(d.B[s][d.obs[t]] >= d.B[1 - s][d.obs[t]]));
		close(m.joint, d.pi[g[0]] * d.B[g[0]][d.obs[0]] * g.slice(1).reduce((p, s, t) => p * d.A[g[t]][s] * d.B[s][d.obs[t + 1]], 1), 1e-12);
		if (d.obs.length > 1) close(m.two, m.pi * m.b0 * (m.a1 * m.b1));
	}));

test('predict "hmm-paths" and quiz: n days of two weathers give 2ⁿ paths; ten days give 1,024 paths and 20 cells', () =>
	each((d, fwd, _v, ps) => {
		assert.equal(ps.length, 2 ** d.obs.length);
		assert.equal(fwd.frame(0).metrics!.paths, ps.length);
		assert.equal(K ** 10, 1024);
		assert.equal(new Set(ps.map((x) => x.p.join(''))).size, ps.length);
	}));

test('forward / fill: the last column sums to the total over every path; the shares are the brute-force posteriors; column sums shrink every day', () =>
	each((d, fwd, _v, ps) => {
		const m = of(fwd, 'end')[0].metrics!, n = d.obs.length, L = latOf(fwd, fwd.length - 1);
		const total = ps.reduce((a, x) => a + x.pr, 0), sunny = ps.filter((x) => x.p[n - 1] === 0).reduce((a, x) => a + x.pr, 0);
		close(m.evidence, total, 1e-9);
		close(m.s0, sunny / total, 1e-9);
		for (let t = 0; t < n; t++) for (let j = 0; j < K; j++) close(L.val[t][j], alphaBrute(d, ps, t, j), 1e-9);
		const sums = L.val.map((r) => r[0] + r[1]);
		for (let t = 1; t < n; t++) assert.ok(sums[t] < sums[t - 1], `day ${t}`);
	}));
/** α[t][j] by brute force: the probability of the first t + 1 activities over every path prefix ending in j */
function alphaBrute(d: HmmData, ps: { p: number[] }[], t: number, j: number): number {
	const seen = new Set<string>();
	let s = 0;
	for (const x of ps) {
		const pre = x.p.slice(0, t + 1), k = pre.join('');
		if (pre[t] !== j || seen.has(k)) continue;
		seen.add(k);
		s += jointOf({ ...d, obs: d.obs.slice(0, t + 1) }, pre);
	}
	return s;
}

test('viterbi / back: the traced path is the brute-force best path; δ of its last cell is its probability; share = best / total', () =>
	each((d, _f, vit, ps) => {
		const m = of(vit, 'done')[0].metrics!, L = latOf(vit, vit.length - 1), best = ps.reduce((a, x) => (x.pr > a.pr ? x : a));
		assert.deepEqual(L.path, best.p);
		close(m.pBest, best.pr, 1e-9);
		close(m.share, best.pr / ps.reduce((a, x) => a + x.pr, 0), 1e-9);
		// "one per weather state": every cell keeps exactly one back-pointer, to the best way in
		for (const f of of(vit, 'cell')) assert.equal(f.metrics!.best, Math.max(f.metrics!.c0, f.metrics!.c1));
	}));

test('back: where Viterbi and the activity-alone guess differ, the transitions of the best path outweigh that one activity', () =>
	each((d, _f, vit) => {
		const L = latOf(vit, vit.length - 1), best = jointOf(d, L.path);
		L.path.forEach((s, t) => {
			if (s === L.naive[t]) return;
			const flip = L.path.slice();
			flip[t] = L.naive[t];
			assert.ok(d.B[L.naive[t]][d.obs[t]] > d.B[s][d.obs[t]], 'the activity alone prefers the other weather');
			assert.ok(jointOf(d, flip) < best, 'yet the path with that one day changed is less likely');
		});
		assert.equal(of(vit, 'done')[0].metrics!.diff, L.path.filter((s, t) => s !== L.naive[t]).length);
	}));

test('back: "far from certain" is said only when the best path holds less than half of the probability', async () => {
	const step = lesson.steps.find((s) => (s as { id: string }).id === 'back') as { narration: { vars: (rt: unknown, at: unknown) => Record<string, unknown> } };
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'vit');
		if (!t?.length) return;
		const v = step.narration.vars(rt, { tape: t, index: t.length - 1, frame: t.frame(t.length - 1) }), share = t.frame(t.length - 1).metrics!.share;
		assert.equal(JSON.stringify(v.sure).includes('far from certain'), share < 0.5);
		assert.ok(share > 0 && share <= 1);
	}, opts);
});

test('task: the starting tables leave the path unchanged; the hint (stickier weather, or cleaning likelier on sunny days) changes it on the start days', () => {
	const run = (raw: object) => viterbiOf(dataOf(raw)).path.join('');
	const base = run({ obs: START_OBS });
	assert.notEqual(run({ obs: START_OBS, A: [[95, 5], [10, 90]] }), base);
	assert.notEqual(run({ obs: START_OBS, B: [[40, 25, 35], [10, 30, 60]] }), base);
	const fx = (lesson.fixtures ?? []).find((f) => f.id === 'clean-sunny')!;
	assert.notEqual(run(fx.raw as object), base);
});

test('limits: in the made-up stretch every rainy spell lasts RAIN days; the counted chain gets the mean right but its likeliest spell is one day, for any stay probability below 1', () => {
	const spells: number[] = [];
	for (let t = 0; t < YEAR.length; t++) if (YEAR[t] === 1 && (t === 0 || YEAR[t - 1] === 0)) spells.push(YEAR.slice(t).findIndex((s) => s === 0));
	assert.ok(spells.length > 1 && spells.every((l) => l === RAIN) && YEAR[YEAR.length - 1] === 0);
	const nRR = YEAR.slice(1).filter((s, t) => s === 1 && YEAR[t] === 1).length, nR = YEAR.slice(0, -1).filter((s) => s === 1).length;
	close(1 / (1 - nRR / nR), RAIN, 1e-12);
	for (let a = 0.05; a < 1; a += 0.05) assert.ok((1 - a) > a * (1 - a), `stay ${a}: one day beats two`);
});

test('rnn: an HMM can count rainy days with more states: Rainy-1 → Rainy-2 → Rainy-3 → Sunny gives spells of exactly three days', () => {
	const A = [[0.8, 0.2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], B = [[0.5, 0.35, 0.15], ...Array(3).fill([0.1, 0.3, 0.6])];
	const t = record(sample, { pi: [1, 0, 0, 0], A, B }, { days: 200 }, { seed: 3 });
	const w = Array.from((t.state(t.length - 1) as { weather: { data: ArrayLike<number> } }).weather.data), rainy = w.map((s) => (s > 0 ? 1 : 0));
	const lens: number[] = [];
	for (let i = 0; i < rainy.length; i++) if (rainy[i] && (!i || !rainy[i - 1])) lens.push(rainy.slice(i).findIndex((s) => !s));
	const whole = lens.filter((l) => l >= 0);
	assert.ok(whole.length > 5 && whole.every((l) => l === 3), `${lens}`);
});

test('rnn: the free numbers of the tables, and the char-RNN facts copied from its own files', () => {
	assert.equal(FREE, (K - 1) + K * (K - 1) + K * (M - 1));
	assert.equal(FREE, 7);
	const mine = JSON.parse(readFileSync(join(SITE, 'static/models/hmm/facts.json'), 'utf8')).facts;
	const theirs = JSON.parse(readFileSync(join(SITE, 'static/models/char-rnn/facts.json'), 'utf8')).facts;
	const spec = JSON.parse(readFileSync(join(SITE, 'static/models/char-rnn/spec.json'), 'utf8'));
	for (const k of ['counterUnit', 'counterR']) assert.equal(mine[k].value, theirs[k].value);
	assert.equal(mine.charRnnParams.value, theirs.params.value);
	assert.equal(mine.charRnnHidden.value, spec.layers.find((l: { op: string }) => l.op === 'lstm').outShape.at(-1));
});

test('story: the defaults make walks likelier on sunny days and cleaning likelier on rainy ones, and weather tends to last', () => {
	assert.ok(B0[0][0] > B0[1][0] && B0[1][2] > B0[0][2]);
	assert.ok(A0[0][0] > 50 && A0[1][1] > 50);
	assert.deepEqual(naiveOf(dataOf(START)), [0, 0, 0, 1, 0, 1, 0, 1]);
});
