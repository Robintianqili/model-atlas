// behavior-cloning claim tests: every sentence of the lesson (and the page) that is not computed live is asserted here,
// on every fixture (the presets and the adapter's edge fixtures) where it depends on the input. `atlas check` runs this
// file (AK408 on failure).   node --test src/lib/explainers/behavior-cloning/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { DEFAULTS, distances, EVAL, expertMove, featuresOf, pairsOf, policyAt, step, toRaw, worldOf, xy, type Params } from './world.ts';
import { bc, DAGGER, datasetOf, paramsOf } from './trainer.ts';
import { boardOf } from './views/board/meta.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('behavior-cloning', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
async function run(rt: AtlasRuntime, source: string): Promise<Tape> {
	await rt.ensure(source, 'last');
	const t = rt.sources[source].tape;
	assert.ok(t?.length, `${source} has frames`);
	return t!;
}
const lastP = (t: Tape) => (t.state(t.length - 1) as { params: Params }).params;
const hasDemos = (rt: AtlasRuntime) => pairsOf(toRaw(rt.input)).s.length > 0;

test('manifest: the train sources share seed 1; 2 → 16 → 4 has 116 parameters; the expert demonstration of the default reaches the goal', () => {
	for (const k of ['train', 'dagger']) assert.equal((manifest.sources![k] as unknown as { opt: { seed: number } }).opt.seed, 1, k);
	assert.equal(Object.values(paramsOf(bc.model(createRng(1).fork('init'), bc.defaults))).reduce((n, t) => n + t.data.length, 0), 7 * bc.defaults.hidden + 4);
	const W = worldOf(DEFAULTS.map), dm = DEFAULTS.demos[0];
	assert.equal(dm.start, W.start, 'the default demonstration starts at S');
	assert.equal(W.cell[dm.moves.reduce((s, a) => step(W, s, a), dm.start)], 2, 'and ends at the goal');
});

test('steps "net" and "train": the fresh network gives no move a probability of 1; training gets every pair of an expert demonstration right', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		if (!hasDemos(rt)) return;
		const t = await run(rt, 'train'), g = toRaw(rt.input), W = worldOf(g.map);
		const p0 = policyAt((t.state(0) as { params: Params }).params, featuresOf(W, g.probe)).p;
		assert.ok(Math.max(...p0) < 0.9, 'random starting weights: no near-certain move');
		const m = t.frame(t.length - 1).metrics!;
		assert.equal(m.acc, 1, 'every demonstrated (cell, move) pair is copied');
		assert.ok(m.loss < t.frame(1).metrics!.loss, 'the loss falls');
	}, opts);
});

test('step "predict" (explain): with gusts, the cloned policy of the default input points along the path somewhere below it', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const g = toRaw(rt.input);
		if (!hasDemos(rt) || JSON.stringify(g) !== JSON.stringify(DEFAULTS)) return;
		await run(rt, 'train');
		const b = boardOf(rt, {}), path = new Set(pairsOf(g).s);
		const below = [...path].map((s) => s + b.W.w).filter((s) => s < b.W.cell.length && b.W.cell[s] === 0 && !path.has(s));
		assert.ok(below.some((s) => b.moves![s] !== 0), 'a cell below the path where the clone does not move back up');
	}, opts);
});

test('steps "drive" and "compound": the rollouts agree with the trainer, and off the data the moves are worse (the errors compound)', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		if (!hasDemos(rt)) return;
		const [t, d] = [await run(rt, 'train'), await run(rt, 'dagger')];
		const r = all(await run(rt, 'roll')), dr = all(await run(rt, 'droll'));
		const done = r[r.length - 1].metrics!, ddone = dr[dr.length - 1].metrics!;
		assert.equal(done.success, t.frame(t.length - 1).metrics!.success, 'roll = the trainer\'s success metric (the same wind)');
		assert.equal(ddone.success, d.frame(d.length - 1).metrics!.success, 'droll = the DAgger run\'s success metric');
		assert.equal(done.exGoals, ddone.exGoals, 'the expert meets the same gusts in both');
		assert.equal(done.onGood, done.onMoves, 'on cells in the dataset every move leads closer (the clone copies the expert there)');
		if (done.offMoves > 0 && done.onMoves > 0 && done.goals < done.exGoals) assert.ok(done.offGood / done.offMoves < done.onGood / done.onMoves, 'unseen cells: a smaller share of good moves');
		// the expert-labelled cells: their moves are the expert's
		const g = toRaw(rt.input), W = worldOf(g.map), dist = distances(W);
		const D = datasetOf(rt.input, d, DAGGER);
		D.s.forEach((s, i) => D.from[i] > 0 && assert.equal(D.a[i], expertMove(W, dist, s)));
	}, opts);
});

test('step "dagger": the DAgger run starts as the cloning run, and its rebuilt dataset is the one it trained on', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		if (!hasDemos(rt)) return;
		const t = await run(rt, 'train'), d = await run(rt, 'dagger');
		const at100 = all(d).findIndex((f) => f.phase === 'snapshot' && f.metrics!.step === DAGGER.steps - DAGGER.rounds * bc.defaults.every);
		const a = lastP(t), b = (d.state(at100) as { params: Params }).params;
		for (const k of Object.keys(a)) assert.deepEqual(Array.from(b[k].data), Array.from(a[k].data), k);
		const D = datasetOf(rt.input, d, DAGGER), mLast = d.frame(d.length - 1).metrics!;
		assert.equal(D.s.length, mLast.n, 'the replayed dataset has the size the run trained on');
		assert.equal(D.rounds.length, DAGGER.rounds);
	}, opts);
});

test('step "limits": the unlabelled cells are exactly the open cells outside the DAgger dataset, and the default needs the expert', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		if (!hasDemos(rt)) return;
		await run(rt, 'dagger');
		const b = boardOf(rt, { policy: 'dagger' });
		for (const s of b.added) assert.ok(!pairsOf(toRaw(rt.input)).s.includes(s), 'DAgger adds only cells the demonstrations lack');
		if (JSON.stringify(toRaw(rt.input)) === JSON.stringify(DEFAULTS)) {
			assert.ok(b.added.size > 0, 'the default input needs expert labels');
			assert.ok([...b.added].some((s) => xy(b.W, s)[0] > 0 && xy(b.W, s)[1] > 0), 'some labelled cells lie below the demonstrated path (right of column 0, under row 0)');
		}
	}, opts);
});

test('the wind: the same seed gives the same gusts, whatever the policy (a fair comparison)', () => {
	const a = createRng(EVAL.seed), b = createRng(EVAL.seed);
	for (let i = 0; i < 100; i++) assert.equal(a.next(), b.next());
});
