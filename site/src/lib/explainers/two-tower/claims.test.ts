// two-tower claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, on every fixture where it depends on the input (the five presets and the adapter's four edge
// fixtures, through withFixtures). `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/two-tower/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import { EX, TASK_GENRES, TASK_PICKS, taskDone } from './lesson.ts';
import { APPEAL, G, GENRES, MOVIES, PRESETS, TWINS, itemFeatures, logUser, toPrefs } from './world.ts';
import { K, T, likesOf, rankOne, ranked, reranked, rowOf, shortlist, towerOf } from './recs.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('two-tower', SITE).then((r) => r.pkg);
const facts = JSON.parse(readFileSync(join(SITE, 'static/models/two-tower/facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const chosenGenres = (raw: unknown) => { const p = toPrefs(raw); return new Set([...p.genres, ...p.picks.map((i) => MOVIES[i].genre)]); };

test('page and catalog: 672 learned numbers in the towers plus the 36 × 8 table; the file sizes; the training intro (64, Adam, 40 epochs, 500 test viewers)', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(rt.spec!.params, 960);
	assert.equal(facts.params.value, 672);
	assert.deepEqual(rt.model!.weights['scores.weight'].shape, [MOVIES.length, 8]);
	assert.ok(manifest.catalog!.params!.en.startsWith('672 ') && manifest.catalog!.params!.en.includes('36 × 8'));
	const kb = ['spec.json', 'weights.bin', 'facts.json'].reduce((s, f) => s + statSync(join(SITE, 'static/models/two-tower', f)).size, 0) / 1024;
	const said = Number(/About (\d+) KB/.exec(manifest.catalog!.size!.en)![1]);
	assert.ok(Math.abs(kb - said) < 1, `${kb.toFixed(1)} KB, the catalog says ${said}`);
	assert.deepEqual([facts.batch.value, facts.epochs.value, facts.testViewers.value, facts.valViewers.value], [64, 40, 500, 500]);
	assert.ok(facts.bestEpoch.value >= 1 && facts.bestEpoch.value <= 40);
	const intro = manifest.page.sections!.intros!.training!.en;
	assert.ok(intro.includes('batches of 64 pairs, Adam; the weights of the best of 40 epochs on 500 validation viewers') && intro.includes('500 other, test viewers'));
	assert.equal(1 * 3 + 2 * -1 + 0 * 5, 1, 'quiz');
});

test('step "catalog": every movie is one genre switch on (one-hot), then the year and the length, scaled to about −1 … 1', () => {
	for (const m of MOVIES) {
		const f = itemFeatures(m);
		assert.equal(Array.from(f.slice(0, G)).filter((x) => x === 1).length, 1);
		assert.ok(Array.from(f.slice(0, G)).every((x) => x === 0 || x === 1));
		assert.ok(Math.abs(f[G]) <= 1.05 && Math.abs(f[G + 1]) <= 1.05, m.title.en);
	}
});

test('step "item": the item tower ignores the viewer, and every row of the shipped table equals the tower run live on that movie', async () => {
	const rt = await createRuntime(await pkgP, opts), m = rt.model!;
	for (let i = 0; i < MOVIES.length; i++) {
		const y = towerOf(rt, 'item', i)!.y, row = rowOf(rt, i);
		assert.ok(row.every((v, k) => Math.abs(v - y[k]) < 1e-6), MOVIES[i].title.en);
	}
	const run = (u: number) => m.forward({ user: { shape: [15], data: new Float32Array(15).fill(u) }, item: { shape: [8], data: itemFeatures(MOVIES[EX]) } }, { record: 'layers' }).t['item.fc2'].data;
	assert.deepEqual(Array.from(run(0)), Array.from(run(1)));
	assert.equal(rt.spec!.layers.find((l) => l.id === 'scores')!.op, 'linear');
	assert.equal(rt.model!.weights['scores.bias'], undefined, 'the scoring layer has no bias: a score is the dot product alone');
});

test('step "pick": the task asks for one genre and three movies', () => {
	assert.deepEqual([TASK_GENRES, TASK_PICKS], [1, 3]);
	assert.ok(taskDone(PRESETS[0].raw) && !taskDone(PRESETS[4].raw));
});

test('steps "dot" and "scores": every score is your vector times that movie’s row of the table', async () => {
	await withFixtures(await pkgP, (rt) => {
		const u = T(rt, 'user.fc2')!.data, s = T(rt, 'scores')!.data;
		for (let i = 0; i < MOVIES.length; i++) assert.ok(Math.abs(s[i] - rowOf(rt, i).reduce((a, v, d) => a + v * u[d], 0)) < 1e-5);
	}, opts);
});

test('predict "scores": the best unpicked movie is of a genre you chose (switched on or picked); with nothing chosen, a movie many viewers liked leads', async () => {
	const likes = MOVIES.map((_, i) => i);
	await withFixtures(await pkgP, (rt, fx) => {
		const top = ranked(rt)[0].i, g = chosenGenres(rt.input);
		if (g.size) assert.ok(g.has(MOVIES[top].genre), `${fx.id}: ${MOVIES[top].title.en}`);
		else {
			const order = [...likes].sort((a, b) => likesOf(rt, b) - likesOf(rt, a));
			assert.ok(order.indexOf(top) < MOVIES.length / 4, `${fx.id}: ${MOVIES[top].title.en} is not among the most liked`);
		}
	}, opts);
});

test('try-this "scary": no genre switched on, four horror picks; the two unpicked horror movies lead the shortlist', async () => {
	const p = PRESETS[3].raw;
	assert.equal(p.genres.length, 0);
	assert.deepEqual(p.picks.map((i) => GENRES[MOVIES[i].genre].en), ['horror', 'horror', 'horror', 'horror']);
	const rt = await createRuntime(await pkgP, { ...opts, input: p });
	assert.deepEqual(shortlist(rt).slice(0, 2).map((x) => MOVIES[x.i].genre), [3, 3]);
});

test('steps "shortlist" and "rank": retrieval leaves out your picks and keeps the best K; the TS ranker equals export.py’s float64 ranker on every reference case', async () => {
	const ref = (await createRuntime(await pkgP, opts)).spec!.rankerRef as { case: string; order: number[]; z: number[] }[];
	await withFixtures(await pkgP, (rt, fx) => {
		const s = shortlist(rt), picks = toPrefs(rt.input).picks;
		assert.equal(s.length, K);
		assert.ok(s.every((x) => !picks.includes(x.i)));
		const r = ref.find((x) => x.case === fx.id.replace(/^edge:/, ''));
		if (r) {
			const rr = reranked(rt);
			assert.deepEqual(rr.map((x) => x.i), r.order, fx.id);
			assert.ok(rr.every((x, j) => Math.abs(x.z - r.z[j]) < 1e-4));
		}
	}, opts);
	assert.equal(ref.length, 9);
});

test('step "train": the right answer is the diagonal; loss = mean −log share of the right movie; chance = log B; the gradient is negative on the diagonal and positive elsewhere', async () => {
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('inbatch', 'last');
		const tp = rt.sources.inbatch.tape as Tape, at = (ph: string) => tp.find({ phase: ph });
		const P = (tp.state(at('softmax')) as { P: Tensor }).P, Gr = (tp.state(at('grad')) as { G: Tensor }).G, B = P.shape[0], m = tp.frame(at('softmax')).metrics!;
		const loss = -Array.from({ length: B }, (_, i) => Math.log(P.data[i * B + i])).reduce((a, b) => a + b, 0) / B;
		assert.ok(Math.abs(loss - m.loss) < 1e-12 && Math.abs(m.chance - Math.log(B)) < 1e-12);
		for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) assert.ok(i === j ? Gr.data[i * B + j] < 0 : Gr.data[i * B + j] > 0);
	}, opts);
	assert.equal(facts.trainPairs.value, Array.from({ length: 5000 }, (_, u) => logUser(u).liked.length).reduce((a, b) => a + b, 0));
});

test('step "limits": the twins share every feature, the vector and every score; the logs liked them very differently (the likes extra equals the TS logs); only the ranker separates them', async () => {
	const [a, b] = TWINS;
	assert.deepEqual(Array.from(itemFeatures(MOVIES[a])), Array.from(itemFeatures(MOVIES[b])));
	assert.ok(APPEAL[a] > APPEAL[b], 'something no feature records');
	const likes = MOVIES.map(() => 0);
	for (let u = 0; u < 5000; u++) for (const i of [...logUser(u).history, ...logUser(u).liked]) likes[i]++;
	const rt0 = await createRuntime(await pkgP, opts);
	assert.deepEqual(rt0.spec!.likes, likes);
	assert.deepEqual(rowOf(rt0, a), rowOf(rt0, b));
	await withFixtures(await pkgP, (rt) => {
		const s = T(rt, 'scores')!.data;
		assert.equal(s[a], s[b]);
		assert.ok(rankOne(rt, { i: a, s: s[a] }).p > rankOne(rt, { i: b, s: s[b] }).p);
	}, opts);
	assert.deepEqual([facts.twinLikesA.value, facts.twinLikesB.value], [likes[a], likes[b]]);
});
