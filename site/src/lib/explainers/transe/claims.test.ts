// transe claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (the lesson's, the adapter's edge fixtures and the presets) where it depends on the input. `atlas check` runs it.
//   node --test src/lib/explainers/transe/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { backward, type Var as AVar } from '../../atlas/autograd/var.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest, { SEED } from './explainer.ts';
import lesson from './lesson.ts';
import { rank, rankInput } from './process.ts';
import { TransE, transe } from './trainer.ts';
import { CAPITAL_OF, CONTINENT, ENT, factsOf, hasFact, id, LOCATED_IN, NE, NR, ofType, PT_FACT, SPEAKS, START, START_FACTS, type Fact } from './kg.ts';
import { arrowOnMap, dist, lastVecs, onMap, pca2, ranked, spread, worked, type Vecs } from './space.ts';
import { GAMMA } from './views/step/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('transe', SITE).then((l) => l.pkg);
const at = (t: Tape, i: number) => ({ index: i, frame: t.frame(i), state: t.state(i), tape: t });
const train = (raw: unknown, seed = SEED) => lastVecs(record(trainingProcess(transe), raw, {}, { seed, maxFrames: 100000 }));
const nearest = (v: Vecs, h: number, r: number, kind: number) => ranked(v, h, r).find((c) => ENT[c.x].type === kind)!.x;
async function tapeOf(rt: AtlasRuntime, s: string): Promise<Tape | null> {
	await rt.ensure(s, 'last');
	return rt.sources[s]?.tape ?? null;
}
const WITH_PT = { ...START, facts: [...START_FACTS, PT_FACT] };

test('kg: 34 entities of four types, 3 relations; the start leaves out where Portugal is, and only that; Canada speaks two languages', () => {
	assert.equal(NE, 34);
	assert.deepEqual([0, 1, 2, 3].map((k) => ofType(k).length), [11, 11, 4, 8]);
	assert.equal(NR, 3);
	assert.ok(!START_FACTS.some((f) => f[0] === PT_FACT[0] && f[1] === LOCATED_IN));
	for (const c of ofType(0)) if (c !== PT_FACT[0]) assert.ok(hasFact(START_FACTS, [c, LOCATED_IN, CONTINENT[c]]));
	assert.deepEqual(START_FACTS.filter((f) => f[0] === id('Canada') && f[1] === SPEAKS).map((f) => f[2]), [id('English'), id('French')]);
	assert.equal(CONTINENT[PT_FACT[0]], id('Europe'));
});

test('margin (predict): once the copy is γ farther than the fact, the loss and every gradient are 0', () => {
	const m = new TransE(createRng(1), 8), I = (x: number) => Int32Array.of(x);
	const E = m.ent.value.data, R = m.rel.value.data;
	// place the fact's tail 0.2 from h + r and the copy's tail 1.5 from it: d(fact) = 0.2, d(copy) = 1.5 > 0.2 + γ
	for (let k = 0; k < 8; k++) (E[2 * 8 + k] = E[0 * 8 + k] + R[k] + (k === 1 ? 0.2 : 0)), (E[3 * 8 + k] = E[0 * 8 + k] + R[k] + (k === 0 ? 1.5 : 0));
	const loss = transe.loss(m, { h: I(0), r: I(0), t: I(2), h2: I(0), t2: I(3) }, () => {}, transe.defaults).loss as unknown as AVar;
	assert.equal(loss.item(), 0);
	backward(loss);
	for (const p of m.params()) assert.ok(!p.grad || p.grad.data.every((g) => g === 0));
	// the page's quiz: max(0, 1 + 0.3 − 0.9) = 0.4
	assert.ok(Math.abs(Math.max(0, 1 + 0.3 - 0.9) - 0.4) < 1e-12);
});

test('update: step 0 of `one` is step 0 of `run`; its loss is positive, the fact gets closer, the copy farther, the loss falls; new = old − lr × g', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const one = await tapeOf(rt, 'one');
		if (!one?.length) return assert.equal(factsOf(rt.input).length, 0);
		const i = Array.from({ length: one.length }, (_, k) => k).find((k) => one.frame(k).phase === 'update')!;
		const w = worked(at(one, i), GAMMA)!;
		assert.ok(w.loss > 0, `loss ${w.loss}`);
		assert.ok(w.dPos1 < w.dPos && w.dNeg1 > w.dNeg && w.loss1 < w.loss);
		assert.equal(w.new, w.old - w.lr * w.g);
		assert.ok(w.h2 !== w.t2 && (w.h2 === w.h) !== (w.t2 === w.t), 'the copy swaps exactly one end, never into a self-loop');
		const run = await tapeOf(rt, 'run');
		assert.equal(run!.frame(1).metrics!.loss, one.frame(i).metrics!.loss);
	}, opts);
});

test('arrows: an arrow keeps its direction on the map (map of h + r = map of h + map of r); every capital lands nearer its own country than the others', async () => {
	const v = train(START), p = pca2(v.E), h = v.E[id('Paris')], r = v.R[CAPITAL_OF];
	const a = onMap(p, h.map((x, k) => x + r[k])), b = onMap(p, h).map((x, k) => x + arrowOnMap(p, r)[k]);
	assert.ok(Math.abs(a[0] - b[0]) < 1e-12 && Math.abs(a[1] - b[1]) < 1e-12);
	await withFixtures(await pkgP, async (rt) => {
		const vv = lastVecs(await tapeOf(rt, 'run')), fs = factsOf(rt.input).filter((f) => f[1] === CAPITAL_OF);
		if (fs.length < 2) return;
		for (const [c, , t] of fs) for (const x of ofType(0)) if (x !== t) assert.ok(dist(vv, c, CAPITAL_OF, t) < dist(vv, c, CAPITAL_OF, x), `${c} ${x}`);
	}, opts);
});

test('rank: the process scores all 34 entities, any type, and its ranking is the distance order; Paris + capital of → France first (prequestion)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'rank');
		if (!t?.length) return assert.equal(factsOf(rt.input).length, 0);
		const sc = Array.from({ length: t.length }, (_, i) => t.frame(i)).filter((f) => f.phase === 'score');
		assert.deepEqual(sc.map((f) => f.metrics!.x), Array.from({ length: NE }, (_, x) => x));
		const v = lastVecs(rt.sources.run.tape), q = (rt.input as { query?: [number, number] }).query ?? START.query;
		assert.equal(t.frame(t.length - 1).metrics!.t1, ranked(v, q[0], q[1])[0].x);
	}, opts);
	const v = train(START);
	assert.equal(ranked(v, id('Paris'), CAPITAL_OF)[0].x, id('France'));
	assert.equal(record(rank, rankInput(null, 0, 0, []), {}, { seed: 1 }).length, 0);
});

test('portugal (predict): with the start facts the nearest continent is South America (Brazil’s, which shares Portuguese), and Portugal sits nearer Brazil than two countries on average; with the fact added it is Europe', () => {
	const v = train(START);
	assert.equal(nearest(v, PT_FACT[0], LOCATED_IN, 2), id('South America'));
	assert.equal(nearest(v, id('Brazil'), LOCATED_IN, 2), id('South America'));
	assert.ok(dist(v, PT_FACT[0], SPEAKS, id('Portuguese')) < 1 && dist(v, id('Brazil'), SPEAKS, id('Portuguese')) < 1);
	assert.ok(spread(v, [PT_FACT[0], id('Brazil')]) < spread(v, ofType(0)));
	// the try-this and the task: one added fact moves the answer to Europe
	assert.equal(nearest(train(WITH_PT), PT_FACT[0], LOCATED_IN, 2), id('Europe'));
	// not an accident of one seed: South America first for most seeds, Europe once the fact is added for nearly all
	let sa = 0, eu = 0;
	for (let s = 1; s <= 12; s++) (sa += nearest(train(START, s), PT_FACT[0], LOCATED_IN, 2) === id('South America') ? 1 : 0), (eu += nearest(train(WITH_PT, s), PT_FACT[0], LOCATED_IN, 2) === id('Europe') ? 1 : 0);
	assert.ok(sa >= 8 && eu >= 11, `${sa} ${eu}`);
});

test('limits: one-to-many pulls English and French together; many-to-one pulls the countries of a continent together (6 seeds)', () => {
	for (let s = 1; s <= 6; s++) {
		const v = train(START, s);
		assert.ok(spread(v, [id('English'), id('French')]) < 0.75 * spread(v, ofType(3)), `seed ${s}`);
		let a = 0, n = 0, b = 0, m = 0;
		for (const x of ofType(0)) for (const y of ofType(0)) if (x < y) CONTINENT[x] === CONTINENT[y] ? ((a += spread(v, [x, y])), n++) : ((b += spread(v, [x, y])), m++);
		assert.ok(a / n < b / m, `seed ${s}`);
	}
});

test('page: “Ten steps”; a predict, a task and a limits step; the catalog’s parameter count; every fact of the start trains to rank 1', () => {
	assert.equal(lesson.steps.length, 10);
	assert.ok(lesson.steps.some((s) => 'predict' in s && s.predict) && lesson.steps.some((s) => 'task' in s && s.task) && lesson.steps.some((s) => 'kindOf' in s && s.kindOf === 'limits'));
	assert.equal((NE + NR) * transe.defaults.dim, 296);
	assert.match(manifest.catalog!.params!.en, /\(34 entities \+ 3 relations\) × 8 = 296/);
	const tape = record(trainingProcess(transe), START, {}, { seed: SEED, maxFrames: 100000 });
	assert.equal(tape.frame(tape.length - 1).metrics!.hits1, 1);
	const f: Fact = START_FACTS[0];
	assert.deepEqual(f, [id('Paris'), CAPITAL_OF, id('France')]);
});
