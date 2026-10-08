// gridworld claim tests: every sentence of the lesson that is not computed live is asserted here, on every fixture
// (adapter edge fixtures and presets) where it depends on the input. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/gridworld/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { compileLesson } from '../../atlas/lesson/compile.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { stepAt } from '../../atlas/runtime/sources.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { bandit, race } from './process.ts';
import { BEST, BOARDS, GAMMAS, GOAL, OPEN, PIT, PROBS, goalDistance, step, toRaw, valueIteration, worldOf } from './world.ts';
import { UNREACHABLE } from './adapters/gridworld/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('gridworld', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const SEED = 22; // the bandit sources' seed (explainer.ts)

async function run(rt: AtlasRuntime, source: string): Promise<Tape> {
	await rt.ensure(source, 'last');
	return rt.sources[source].tape!;
}
async function narration(rt: AtlasRuntime, id: string) {
	const pkg = await pkgP;
	const s = compileLesson(pkg.lesson, { spec: null, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes }).steps.find((x) => x.id === id)!;
	const pos = await stepAt(rt, s);
	return { text: s.narration(rt, pos.at), at: pos.at };
}

test('explainer: the bandit sources use the seed these tests check', async () => {
	const src = (await pkgP).manifest.sources!;
	for (const k of ['you', 'greedy', 'eps', 'ucb', 'race']) assert.equal((src[k] as { seed?: number }).seed, SEED, k);
});

test('b-greedy, b-eps: the first pull paid 1 coin, and greedy pulled that machine every time; it never tried the best machine', () => {
	const t = record(bandit, null, { policy: 'greedy' }, { seed: SEED });
	const first = t.frame(2).metrics!;
	const N = Array.from((t.state(t.length - 1) as { N: Tensor }).N.data);
	assert.equal(first.r, 1);
	assert.equal(N[first.arm], 200, `pulls per machine: ${N}`);
	assert.notEqual(first.arm, BEST);
	assert.equal(N[BEST], 0, 'greedy never tried the best machine');
});

test('b-regret: each pull costs pBest − p; the race repeats every single run; regret is deterministic per seed; greedy climbs in a straight line', () => {
	const r = record(race, null, {}, { seed: SEED, recordDraws: true });
	(['greedy', 'eps', 'ucb'] as const).forEach((policy, k) => {
		const ups = all(record(bandit, null, { policy }, { seed: SEED })).filter((f) => f.phase === 'update');
		ups.forEach((f, t) => {
			assert.equal(r.frame(t).metrics![`r${k}`], f.metrics!.regret, `${policy} pull ${t}`);
			const before = t ? ups[t - 1].metrics!.regret : 0;
			assert.ok(Math.abs(f.metrics!.regret - before - (PROBS[BEST] - PROBS[f.metrics!.arm])) < 1e-12);
		});
	});
	const again = record(race, null, {}, { seed: SEED, recordDraws: true });
	for (let k = 0; k < 4; k++) assert.deepEqual(Array.from(again.metric(`r${k}`)), Array.from(r.metric(`r${k}`)));
	const other = record(race, null, {}, { seed: SEED + 1, recordDraws: true });
	assert.notDeepEqual(Array.from(other.metric('r3')), Array.from(r.metric('r3')), 'another seed gives another run');
	const g = Array.from(r.metric('r0'));
	g.forEach((v, t) => assert.ok(Math.abs(v - (t + 1) * (g[0])) < 1e-9, 'greedy: the same cost every pull'));
});

test('b-ucb: the shown pick has fewer pulls, a bigger bonus and a higher score than the machine with the best average', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const m = (await narration(rt, 'b-ucb')).at.frame!.metrics!;
		const q = [0, 1, 2, 3].map((k) => m[`q${k}`]), rival = q.indexOf(Math.max(...q));
		assert.notEqual(rival, m.arm);
		assert.ok(m[`n${m.arm}`] < m[`n${rival}`] && m[`b${m.arm}`] > m[`b${rival}`] && m.score > q[rival] + m[`b${rival}`]);
	}, { ...opts, only: ['preset:classic'] });
});

test('b-average: Q <- Q + (r − Q) / n is the plain average of the coins', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await run(rt, 'you');
		assert.equal(t.length, 1 + 2 * toRaw(rt.input).pulls.length);
		for (let i = 0; i < t.length; i++) {
			const s = t.state(i) as { Q: Tensor; N: Tensor; R: Tensor };
			for (let k = 0; k < 4; k++) assert.ok(Math.abs(s.Q.data[k] - (s.N.data[k] ? s.R.data[k] / s.N.data[k] : 0)) < 1e-12);
		}
	}, opts);
});

test('g-world: only the goal (+1) and the pit (−1) pay, and only they end the episode; walls and the edge block', async () => {
	await withFixtures(await pkgP, (rt) => {
		const W = worldOf(rt.input);
		W.cell.forEach((c, s) => {
			if (c !== OPEN) return;
			for (let a = 0; a < 4; a++) {
				const { s2, r, done } = step(W, s, a);
				assert.equal(r, W.cell[s2] === GOAL ? 1 : W.cell[s2] === PIT ? -1 : 0);
				assert.equal(done, W.cell[s2] === GOAL || W.cell[s2] === PIT);
				assert.notEqual(W.cell[s2], 1, 'never inside a wall');
			}
		});
	}, opts);
});

test('g-qtable, g-choose, g-episode: Q starts at 0, so the first move is random and the first episode wanders, changing at most one Q', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await run(rt, 'q');
		assert.ok(Array.from((t.state(0) as { Q: Tensor }).Q.data).every((v) => v === 0));
		const ep0 = all(t).filter((f) => f.sub === 0);
		for (const f of ep0.filter((x) => x.phase === 'act')) assert.ok(f.metrics!.explore === 1 || f.metrics!.ties === 4, 'all four tied');
		const ups = ep0.filter((f) => f.phase === 'update'), moved = ups.filter((f) => f.metrics!.new !== f.metrics!.old);
		assert.ok(moved.length <= 1 && (!moved.length || (moved[0] === ups[ups.length - 1] && moved[0].metrics!.r !== 0)), 'only the last move, into the goal or the pit, changes a Q');
		for (const f of ups.slice(0, -1)) assert.ok(f.metrics!.r === 0 && f.metrics!.maxNext === 0 && f.metrics!.target === 0, 'every earlier target is 0');
	}, opts);
});

test('g-target, g-spread, misconception: value starts at the goal; the first success sets one positive Q; the reused value was learned earlier', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await run(rt, 'q'), f = all(t);
		const W = worldOf(rt.input);
		const firstPos = f.find((x) => x.phase === 'update' && x.metrics!.new > 0);
		if (goalDistance(W) < 0) {
			assert.equal(firstPos, undefined, 'no positive value without a reachable goal');
			assert.equal(f[f.length - 1].metrics!.goals, 0);
			return;
		}
		assert.equal(W.cell[firstPos!.metrics!.s2], GOAL, 'the first positive Q is a move into the goal');
		const end = f.findIndex((x) => x.phase === 'end' && x.metrics!.how === 1);
		assert.equal(Array.from((t.state(end) as { Q: Tensor }).Q.data).filter((v) => v > 0).length, 1);
		const { at } = await narration(rt, 'g-target');
		const m = at.frame!.metrics!;
		assert.ok(m.maxNext > 0 && f.slice(0, at.index).some((x) => x.phase === 'update' && x.metrics!.s === m.s2 && x.metrics!.new > 0));
	}, opts);
});

test('g-policy, g-gamma: value iteration gives the exact value γ^(d−1) at the start; far cells shrink much faster with γ = 0.5; the γ step quotes the vi sources', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const W = worldOf(rt.input), d = goalDistance(W);
		const [v05, v95] = GAMMAS.map((g) => valueIteration(W, g));
		GAMMAS.forEach((g, i) => assert.ok(Math.abs([v05, v95][i][W.start] - (d < 0 ? 0 : g ** (d - 1))) < 1e-12, `γ = ${g}`));
		W.cell.forEach((_, s) => v95[s] > 0 && v95[s] <= 0.95 ** 4 + 1e-12 && assert.ok(v05[s] / v95[s] < 0.1, `cell ${s}`));
		for (const [src, v] of [['vi05', v05], ['vi95', v95]] as const) {
			const t = await run(rt, src);
			assert.ok(Math.abs(t.frame(t.length - 1).metrics!.vStart - v[W.start]) < 1e-15);
		}
	}, opts);
});

test('unreachable goal: the adapter, the world, the values and the limits step all say so', async () => {
	await withFixtures(await pkgP, async (rt) => {
		assert.deepEqual(rt.pre!.notes, [UNREACHABLE]);
		assert.match((await narration(rt, 'g-world')).text.en, /no path leads from S to the goal/);
		assert.match((await narration(rt, 'g-spread')).text.zh!, /没有哪个回合能走到终点/);
		assert.match((await narration(rt, 'g-limits')).text.en, /the goal cannot be reached/);
		assert.match((await narration(rt, 'g-policy')).text.en, /No path leads from S to the goal/);
		assert.match((await narration(rt, 'g-gamma')).text.en, /the start is worth 0 with any γ/);
		const t = await run(rt, 'q');
		assert.ok(Array.from((t.state(t.length - 1) as { Q: Tensor }).Q.data).every((v) => v <= 0), 'no positive value');
	}, { ...opts, only: ['preset:walled'] });
});

test('presets.json (written by pipeline/gridworld/export.py) has the boards of world.ts', () => {
	const f = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'gridworld', 'presets.json'), 'utf8')) as { boards: { id: string; rows: string[] }[] };
	assert.deepEqual(Object.fromEntries(f.boards.map((b) => [b.id, b.rows])), BOARDS);
});
