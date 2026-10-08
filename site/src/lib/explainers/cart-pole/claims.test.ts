// cart-pole claim tests: every sentence of the lesson (and the page) that is not computed live is asserted here, on every
// fixture (the presets and the adapter's edge fixtures) where it depends on the input. `atlas check` runs this file
// (AK408 on failure).   node --test src/lib/explainers/cart-pole/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape, Tensor, TrainState } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { advantages, physics, policy, RANGES, returns, startOf, START, TAU, TH_LIMIT, X_LIMIT, type Params, type State } from './sim.ts';
import { paramsOf, reinforce } from './trainer.ts';
import { policyEpisode } from './views/scene/meta.ts';
import { cellsOf } from './views/round/meta.ts';
import lesson from './lesson.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('cart-pole', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const D = reinforce.defaults;
const FRESH = paramsOf(reinforce.model(createRng(1).fork('init'), D));
/** every start of a 3 × 3 × 3 × 3 grid over the slider ranges */
const GRID: State[] = [];
for (const a of [0, 0.5, 1]) for (const b of [0, 0.5, 1]) for (const c of [0, 0.5, 1]) for (const d of [0, 0.5, 1]) GRID.push([a, b, c, d].map((f, i) => RANGES[i][0] + f * (RANGES[i][1] - RANGES[i][0])) as State);
async function run(rt: AtlasRuntime, source: string): Promise<Tape> {
	await rt.ensure(source, 'last');
	const t = rt.sources[source].tape;
	assert.ok(t?.length, `${source} has frames`);
	return t!;
}
const lastState = (t: Tape) => t.state(t.length - 1) as Partial<TrainState> & { prev?: Params; batch?: Record<string, Tensor> };

test('manifest and lesson: every source has seed 1; 4 → 16 → 1 has 97 parameters; the train predict names the real number of rounds', async () => {
	for (const [k, s] of Object.entries(manifest.sources!)) assert.equal((s as { seed?: number }).seed ?? (s as { opt?: { seed?: number } }).opt?.seed ?? 1, 1, k);
	assert.equal(Object.values(FRESH).reduce((n, t) => n + t.data.length, 0), 97);
	assert.ok(manifest.catalog?.params?.en.includes('97'));
	const train = lesson.steps.find((s) => 'id' in s && s.id === 'train') as { predict: { question: { en: string } } };
	assert.ok(train.predict.question.en.includes(`${D.steps} times`));
});

test('policy: "close to a coin flip": the fresh policy gives p(left) between 0.35 and 0.65 at every start the sliders allow (grid)', () => {
	for (const s of GRID) {
		const p = policy(FRESH, s).p;
		assert.ok(p > 0.35 && p < 0.65, `p(left) ${p} at [${s}]`);
	}
});

test('physics: pushing right tips the pole left (θ̈ < 0) and pushing left tips it right, at every start the sliders allow (grid)', () => {
	for (const s of GRID)
		for (const left of [0, 1]) {
			const f = physics(s, left);
			assert.ok(f.thAcc * Math.sign(f.force) < 0, `θ̈ ${f.thAcc} for force ${f.force} at [${s}]`);
			assert.equal(f.s2[2], s[2] + TAU * s[3], 'θ′ = θ + τ θ̇ (the formula)');
			assert.equal(f.s2[3], s[3] + TAU * f.thAcc, 'θ̇′ = θ̇ + τ θ̈');
		}
});

test('sample, episode-end, returns (every fixture): left exactly when u < p(left); the end reason is the limit crossed; one reward per push; G = 1 + γ·G(next), largest at the first step', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const fs = all(await run(rt, 'episode'));
		const acts = fs.filter((f) => f.phase === 'act'), moves = fs.filter((f) => f.phase === 'move'), end = fs.find((f) => f.phase === 'end')!.metrics!;
		for (const a of acts) assert.equal(a.metrics!.left, a.metrics!.u < a.metrics!.p ? 1 : 0);
		assert.equal(end.len, acts.length);
		const last = moves[moves.length - 1].metrics!;
		const reason = Math.abs(last.th2) > TH_LIMIT ? 1 : Math.abs(last.x2) > X_LIMIT ? 2 : 3;
		assert.equal(end.end, reason, 'the end reason matches the final state');
		for (const m of moves.slice(0, -1)) assert.equal(m.metrics!.done, 0, 'no limit crossed before the end');
		const G = fs.filter((f) => f.phase === 'return').map((f) => f.metrics!).sort((a, b) => a.t - b.t);
		assert.deepEqual(G.map((g) => g.G), returns(end.len, D.gamma));
		for (let t = 0; t + 1 < G.length; t++) assert.ok(G[t].G > G[t + 1].G, 'returns fall toward the end');
	}, opts);
});

test('advantage, pulls (every fixture): episode 0 of the round is the episode followed; the others start near upright; the last step of every episode has A < 0; the longest episode’s first step has the largest return; the pulls add up to the output-bias gradient', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const ep = all(await run(rt, 'episode')).filter((f) => f.phase === 'act').map((f) => [f.metrics!.x, f.metrics!.xd, f.metrics!.th, f.metrics!.thd]);
		const t1 = await run(rt, 'step1');
		const b = (t1.state(all(t1).findIndex((f) => f.phase === 'backward')) as { batch: Record<string, Tensor>; grads: Record<string, Tensor> });
		const rows = cellsOf(b.batch), N = b.batch.adv.data.length;
		assert.equal(rows[0].length, ep.length, 'episode 0 has the length of the episode followed');
		ep.forEach((s, t) => assert.deepEqual(Array.from(b.batch.x.data.slice(4 * t, 4 * t + 4)), s));
		for (let e = 1; e < rows.length; e++) {
			const i0 = Array.from(b.batch.ep.data).indexOf(e);
			for (let k = 0; k < 4; k++) assert.ok(Math.abs(b.batch.x.data[4 * i0 + k]) <= START, 'a random start near upright');
		}
		for (const r of rows) assert.ok(r[r.length - 1].A < 0, 'the last step before the end has A < 0');
		const G = Array.from(b.batch.ret.data), longest = rows.reduce((m, r, e) => (r.length > rows[m].length ? e : m), 0);
		assert.equal(rows[longest][0].G, Math.max(...G));
		assert.deepEqual(Array.from(b.batch.adv.data), advantages(G), 'A depends on the returns only, not on the pushes');
		const sum = rows.flat().reduce((s, c) => s + c.pull, 0);
		assert.ok(Math.abs(sum - b.grads['net.2.bias'].data[0]) < 1e-12, `pulls ${sum} vs gradient ${b.grads['net.2.bias'].data[0]}`);
		for (const c of rows.flat()) {
			const chosen = c.left ? c.p : 1 - c.p;
			// gradient descent moves z against the pull: z rises when pull < 0, which raises p(left); "toward its own push" iff A > 0
			const towardOwn = (c.pull < 0) === (c.left === 1);
			if (c.A !== 0) assert.equal(towardOwn, c.A > 0);
			assert.ok(Math.abs(Math.abs(c.pull) - (Math.abs(c.A) * (1 - chosen)) / N) < 1e-15, 'harder when the push was unlikely');
		}
	}, opts);
});

test('pulls, update: "raising the output bias on its own makes left more likely" (and lowering it right), at every start the sliders allow (grid)', () => {
	const up = { ...FRESH, 'net.2.bias': { shape: [1], data: Float64Array.from([FRESH['net.2.bias'].data[0] + 0.01]) } };
	const down = { ...FRESH, 'net.2.bias': { shape: [1], data: Float64Array.from([FRESH['net.2.bias'].data[0] - 0.01]) } };
	for (const s of GRID) {
		const p = policy(FRESH, s).p;
		assert.ok(policy(up, s).p > p && policy(down, s).p < p, `at [${s}]`);
	}
});

test('update (every fixture): plain SGD; one update moves p(left) at the start by less than 0.05', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const s = lastState(await run(rt, 'step1'));
		assert.equal(s.optimizer, 'sgd');
		const start = startOf(rt.input);
		assert.ok(Math.abs(policy(s.params!, start).p - policy(s.prev!, start).p) < 0.05);
	}, opts);
});

test('train (every fixture): the average episode length climbs to at least 85% of the step limit and to 5 times round 0, and dips on some rounds', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const ml = all(await run(rt, 'train')).map((f) => f.metrics?.meanLen).filter((v): v is number => typeof v === 'number');
		assert.equal(ml.length, D.steps + 1);
		assert.ok(Math.max(...ml) >= 0.85 * D.maxLen && Math.max(...ml) >= 5 * ml[0], `max ${Math.max(...ml)}, round 0 ${ml[0]}`);
		assert.ok(ml.some((v, i) => i > 0 && v < ml[i - 1]), 'not smoothly');
	}, opts);
});

test('trained (every fixture): training starts from the fresh policy; the scene replays the episode followed with the same draws', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const t = await run(rt, 'train');
		const P0 = t.state(0) as Partial<TrainState>;
		for (const [k, v] of Object.entries(FRESH)) assert.deepEqual(Array.from(P0.params![k].data), Array.from(v.data), k);
		const ep = all(await run(rt, 'episode')).filter((f) => f.phase === 'act');
		const re = policyEpisode(P0.params!, rt.input);
		assert.equal(re.len, ep.length);
		ep.forEach((f, i) => assert.equal(re.left[i], f.metrics!.left));
	}, opts);
});

test('seeds (every fixture): run 1 is the first rounds of the train source exactly; the other seeds start from other weights', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const s = all(await run(rt, 'seeds')).filter((f) => f.phase !== 'init');
		const tr = all(await run(rt, 'train')).filter((f) => f.phase !== 'init');
		s.forEach((f, i) => assert.equal(f.metrics!.len1, tr[i].metrics!.meanLen, `round ${i}`));
		assert.ok(s.some((f) => f.metrics!.len2 !== f.metrics!.len1));
	}, opts);
});

test('quiz: an episode of 3 steps with γ = 0.5 gives the first step a return of 1.75', () => {
	assert.equal(returns(3, 0.5)[0], 1.75);
});
