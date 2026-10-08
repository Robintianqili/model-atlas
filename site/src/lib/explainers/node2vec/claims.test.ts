// node2vec claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (the lesson's, the adapter's edge fixture and the presets) where it depends on the input. `atlas check` runs it.
//   node --test src/lib/explainers/node2vec/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { KARATE_EDGES } from '../../atlas/data/karate.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest, { SEED } from './explainer.ts';
import lesson, { twins } from './lesson.ts';
import { walks } from './process.ts';
import { sg } from './trainer.ts';
import { exampleWalk, FACTIONS, inputOf, LEN, moveOf, N, PQ, START, WALK_SEED, walksOf, WALKS, WIN } from './walks.ts';
import { cosine, factionStats, lastVecs, worked } from './vectors.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('node2vec', SITE).then((l) => l.pkg);
const at = (t: Tape, i: number) => ({ index: i, frame: t.frame(i), state: t.state(i), tape: t });
const train = (raw: unknown, seed: number) => lastVecs(record(trainingProcess(sg), raw, {}, { seed, maxFrames: 100000 }));
async function tapeOf(rt: AtlasRuntime, s: string): Promise<Tape | null> {
	await rt.ensure(s, 'last');
	return rt.sources[s]?.tape ?? null;
}

test('walks: the process frames are the walks the trainer learns from (same seed); pairs stay inside a walk, within the window', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const W = walksOf(rt.input), t = record(walks, rt.input, {}, { seed: WALK_SEED, maxFrames: 100000 });
		const rebuilt = Array.from({ length: WALKS * N }, (_, w) => [w % N]);
		for (let i = 0; i < t.length; i++) rebuilt[t.frame(i).metrics!.w].push(t.frame(i).metrics!.x);
		assert.deepEqual(rebuilt, W.walks);
		for (const p of W.pairs) assert.ok(Math.abs(p.i - p.j) <= WIN && W.walks[p.w][p.i] === p.c && W.walks[p.w][p.j] === p.o);
		// "a walk of LEN members": every walk that moves has exactly LEN members
		for (const w of W.walks) assert.ok(w.length === 1 || w.length === LEN);
	}, opts);
});

test('step: back weighs 1/p, a friend of t 1, any other 1/q; the chances add up to 1; the example move has a previous member', async () => {
	// a hand case: v = 1 came from t = 0; friends of 1: 0 (back), 2 (also a friend of 0), 3 and 4 (farther)
	const nb = [[1, 2], [0, 2, 3, 4], [0, 1], [1], [1]];
	const m = moveOf(nb, 0, 1, 1, 0.5);
	assert.deepEqual(m.kind, [0, 1, 2, 2]);
	assert.deepEqual(m.a, [1, 1, 2, 2]);
	// the quiz: v has 4 friends (t, one friend of t, two others), p = 1, q = 0.5: going back has 1/6
	assert.equal(m.prob[0], 1 / 6);
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'walks');
		if (!t?.length) return assert.equal(walksOf(rt.input).pairs.length, 0);
		const ex = Array.from({ length: t.length }, (_, i) => t.frame(i).metrics!).filter((f) => f.ex === 1);
		assert.equal(ex.length, 1);
		assert.ok(ex[0].t >= 0 && ex[0].w === exampleWalk(rt.input));
		assert.ok(Math.abs(ex[0].p0 + ex[0].p1 + ex[0].p2 - 1) < 1e-12);
	}, opts);
});

const W = (p: number, q: number) => walksOf({ ...START, p, q });
test('predict and task: a small q sends walks farther (like DFS), a large q keeps them close (like BFS); a small p keeps them close too', () => {
	for (const p of PQ) {
		// ends farther from the start: q below 1 > q = 1 > q above 1 (adjacent values below 1 may tie within noise)
		const [lo, one, hi] = [[0.25, 0.5], [1], [2, 4]].map((qs) => qs.map((q) => W(p, q).far));
		assert.ok(Math.min(...lo) > one[0] && one[0] > Math.max(...hi), `p ${p}: ${lo} ${one} ${hi}`);
		// meets more different members, step by step: smaller q, larger p
		for (let k = 1; k < PQ.length; k++) assert.ok(W(p, PQ[k - 1]).distinct > W(p, PQ[k]).distinct && W(PQ[k - 1], p).distinct < W(PQ[k], p).distinct);
	}
	for (const q of PQ) assert.ok(W(0.25, q).far < W(4, q).far, `q ${q}: p 0.25 vs 4`);
	// the try-this: p = 0.25, q = 4 ends nearer the start than the start setting
	assert.ok(W(0.25, 4).far < W(START.p, START.q).far);
});

test('nn: a member’s nearest members are mostly its friends and friends of friends (≥ 80 % within 2 hops, the club, both runs’ seeds)', () => {
	for (const raw of [START, { ...START, p: 0.25, q: 4 }, { ...START, p: 4, q: 0.25 }]) {
		const st = factionStats(raw, train(raw, SEED))!;
		let near = 0, all = 0;
		for (let i = 0; i < N; i++) for (const x of st.nn[i].slice(0, 5)) (all += 1), (near += st.H[i][x.j] >= 1 && st.H[i][x.j] <= 2 ? 1 : 0);
		assert.ok(near / all >= 0.8, `${raw.p} ${raw.q}: ${near / all}`);
	}
});

test('update: step 0 of `one` is step 0 of `run` (same seed); the context member’s score rises and the loss on the pair falls', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const one = await tapeOf(rt, 'one');
		if (!one?.length) return assert.equal(walksOf(rt.input).pairs.length, 0);
		const i = Array.from({ length: one.length }, (_, k) => k).find((k) => one.frame(k).phase === 'update')!;
		const w = worked(at(one, i))!;
		assert.ok(w.u1 > w.u[w.o] && w.loss1 < w.loss);
		assert.equal(w.new, w.old - w.lr * w.g);
		const run = await tapeOf(rt, 'run');
		assert.equal(run!.frame(1).metrics!.loss, one.frame(i).metrics!.loss);
	}, opts);
});

test('map: training never reads the sides; on the club they fall apart (cut ≥ 32 of 34, nearest member on the same side ≥ 28)', async () => {
	const src = ['trainer.ts', 'walks.ts', 'process.ts'].map((f) => readFileSync(join(SITE, 'src/lib/explainers/node2vec', f), 'utf8'));
	assert.ok(!/FACTIONS\[/.test(src.join('\n')), 'the trainer, the walks and the process never index FACTIONS');
	await withFixtures(await pkgP, async (rt) => {
		const g = inputOf(rt.input);
		if (JSON.stringify(g.edges) !== JSON.stringify(KARATE_EDGES)) return;
		const st = factionStats(rt.input, lastVecs(await tapeOf(rt, 'run')))!;
		assert.ok(st.same >= 28, `same ${st.same}`);
		const { pca2, project, cutOf } = await import('./vectors.ts');
		const vs = lastVecs(rt.sources.run.tape);
		assert.ok(cutOf(project(vs, pca2(vs))).ok >= 32);
	}, opts);
});

test('misconception: on the club, most pairs join two members of the same side (≥ 75 % for every p and q offered)', () => {
	for (const p of PQ)
		for (const q of PQ) {
			const W = walksOf({ ...START, p, q });
			let same = 0;
			for (let c = 0; c < N; c++) for (let o = 0; o < N; o++) if (FACTIONS[c] === FACTIONS[o]) same += W.cooc[c * N + o];
			assert.ok(same / W.pairs.length >= 0.75, `p ${p} q ${q}: ${same / W.pairs.length}`);
		}
});

test('pq: BFS-like walks (p 0.25, q 4) give nearer nearest members than DFS-like ones (p 4, q 0.25), for six seeds', () => {
	for (let seed = 1; seed <= 6; seed++) {
		const b = factionStats(START, train({ ...START, p: 0.25, q: 4 }, seed))!, d = factionStats(START, train({ ...START, p: 4, q: 0.25 }, seed))!;
		assert.ok(b.hop < d.hop, `seed ${seed}: ${b.hop} vs ${d.hop}`);
	}
});

test('gcn: the GCN page’s number is its facts.json trainAcc × 32; node2vec’s leader rule sorts ≥ 28 of 32 on the club', async () => {
	const ours = JSON.parse(readFileSync(join(SITE, 'static/models/node2vec/facts.json'), 'utf8')).facts.gcnRight.value;
	const gcn = JSON.parse(readFileSync(join(SITE, 'static/models/gcn/facts.json'), 'utf8')).facts.trainAcc.value;
	assert.equal(ours, gcn * 32);
	assert.ok(factionStats(START, train(START, SEED))!.right >= 28);
});

test('limits: members 14, 15, 18, 20, 22 share their friends (32, 33) and end closer than two members picked at random', () => {
	assert.deepEqual(twins(START), [14, 15, 18, 20, 22]);
	for (const i of twins(START)) assert.deepEqual(walksOf(START).nb[i], [32, 33]);
	for (const raw of [START, { ...START, p: 0.25, q: 4 }, { ...START, p: 4, q: 0.25 }]) {
		const vs = train(raw, SEED), g = twins(raw);
		let s = 0, n = 0, all = 0, m = 0;
		for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) (all += cosine(vs[a], vs[b])), m++;
		for (let a = 0; a < g.length; a++) for (let b = a + 1; b < g.length; b++) (s += cosine(vs[g[a]], vs[g[b]])), n++;
		assert.ok(s / n > all / m + 0.3, `${s / n} vs ${all / m}`);
	}
});

test('page: the lesson note’s “Ten steps”; the predict, task and limits steps exist; the start input is the club with p = 1, q = 0.5', () => {
	assert.equal(lesson.steps.length, 10);
	assert.ok(lesson.steps.some((s) => 'predict' in s && s.predict) && lesson.steps.some((s) => 'task' in s && s.task) && lesson.steps.some((s) => 'kindOf' in s && s.kindOf === 'limits'));
	assert.deepEqual(manifest.input?.start, { edges: KARATE_EDGES, focus: 8, p: 1, q: 0.5 });
});
