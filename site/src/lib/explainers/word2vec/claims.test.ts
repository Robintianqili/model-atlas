// word2vec claim tests: every sentence of the lesson that is not computed live is asserted here, on every fixture (the
// lesson's, the adapter's edge fixtures and the presets) where it depends on the input. `atlas check` runs this file.
//   node --test src/lib/explainers/word2vec/claims.test.ts
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
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { corpusOf, cooccurrence, PETS, SEED, SEED2, type CorpusRaw } from './corpus.ts';
import { cosine, dot, firstPair, neighbours, wordVectors, worked, workedWeight } from './vectors.ts';
import { fit } from './process.ts';
import { sg } from './trainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('word2vec', SITE).then((l) => l.pkg);
const PRESETS = (JSON.parse(readFileSync(join(SITE, 'static/models/word2vec/presets.json'), 'utf8')) as { presets: { id: string; text: string }[] }).presets;
const raw = (id: string): CorpusRaw => ({ text: PRESETS.find((p) => p.id === id)!.text, window: 2 });
const last = (t: Tape) => t.state(t.length - 1);
const vecsOf = (r: CorpusRaw, seed: number) => wordVectors(r, last(record(fit, r, {}, { seed, maxFrames: 100000 })));
async function tapeOf(rt: AtlasRuntime, s: string): Promise<Tape | null> {
	await rt.ensure(s, 'last');
	return rt.sources[s]?.tape ?? null;
}
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

test('presets: four presets, the start input is the first one (pets)', () => {
	assert.deepEqual(PRESETS.map((p) => p.id), ['pets', 'kids', 'story', 'pets-zh']);
	assert.equal(PETS.text, PRESETS[0].text);
});

test('pairs: a window of 2 over a 5-word sentence gives 2 + 3 + 4 + 3 + 2 = 14 pairs; pairs never cross a sentence', () => {
	assert.equal(corpusOf({ text: 'one two three four five.', window: 2 }).pairs.length, 14);
	for (const p of PRESETS) {
		const c = corpusOf(raw(p.id));
		for (const x of c.pairs) assert.ok(Math.abs(x.i - x.j) <= c.window && c.sentences[x.s][x.i] === x.c && c.sentences[x.s][x.j] === x.o);
	}
});

test('fit: the process source makes exactly the frames of trainer sg (the train source)', () => {
	const a = record(fit, PETS, { steps: 200 }, { seed: SEED, maxFrames: 1000 });
	const b = record(trainingProcess(sg), PETS, { steps: 200 }, { seed: SEED, maxFrames: 1000 });
	assert.equal(a.length, b.length);
	for (let i = 0; i < a.length; i++) assert.deepEqual(a.frame(i).metrics, b.frame(i).metrics);
	assert.deepEqual(last(a).params, last(b).params);
});

test('onehot, row, scores: the pair of step 0 is firstPair(); h is row c of W_in; the scores are h · each output vector', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'one');
		const fp = firstPair(rt.input as CorpusRaw, SEED);
		if (!fp) return assert.equal(t?.length ?? 0, 0);
		const w = worked(rt.input as CorpusRaw, { index: 1, frame: t!.frame(1), state: t!.state(1), tape: t! })!;
		assert.deepEqual([w.c, w.o], [fp.c, fp.o]);
		assert.deepEqual(w.h, w.win[w.c]);
		w.wout.forEach((r, j) => assert.ok(close(w.u[j], dot(w.h, r), 1e-12)));
		const run = await tapeOf(rt, 'run');
		assert.equal(run!.frame(1).metrics!.loss, t!.frame(2).metrics!.loss, 'run’s step 0 is the slowed-down step');
	}, opts);
});

test('softmax: with the random start weights every share is within a factor of 2 of 1/V (no word stands out)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'one');
		if (!t?.length) return;
		const w = worked(rt.input as CorpusRaw, { index: 1, frame: t.frame(1), state: t.state(1), tape: t })!;
		for (const p of w.p) assert.ok(p < 2 / w.V && p > 0.5 / w.V, `share ${p}, V ${w.V}`);
	}, opts);
});

test('update: the center row moves by lr·(v_o − Σ p_j v_j), v_o by lr·(1 − p_o)·h, every other v_j by −lr·p_j·h; the score rises and the loss falls', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'one');
		if (!t?.length) return;
		const i = t.find({ phase: 'update', step: 0 });
		const w = worked(rt.input as CorpusRaw, { index: i, frame: t.frame(i), state: t.state(i), tape: t })!;
		const avg = w.h.map((_, k) => w.wout.reduce((s, r, j) => s + w.p[j] * r[k], 0));
		w.h.forEach((_, k) => assert.ok(close(w.win1![w.c][k] - w.win[w.c][k], w.lr * (w.wout[w.o][k] - avg[k]), 1e-10)));
		w.wout.forEach((r, j) => r.forEach((x, k) => assert.ok(close(w.wout1![j][k] - x, w.lr * ((j === w.o ? 1 : 0) - w.p[j]) * w.h[k], 1e-10))));
		w.win.forEach((r, j) => j !== w.c && assert.deepEqual(w.win1![j], r));
		assert.ok(w.u1! > w.u[w.o] && w.loss1! < w.loss);
	}, opts);
});

test('together: the average loss falls on every input', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'run');
		if (!t?.length) return;
		assert.ok(t.frame(t.length - 1).metrics!.avgLoss < t.frame(0).metrics!.avgLoss);
	}, opts);
});

test('together: words with the same context words end up alike (cat, dog, bird; Tim, Lily, Sue), in both runs', () => {
	for (const [id, group] of [['pets', ['cat', 'dog', 'bird']], ['kids', ['tim', 'lily', 'sue']]] as const)
		for (const seed of [SEED, SEED2]) {
			const r = raw(id), w = corpusOf(r).words, vs = vecsOf(r, seed), ids = group.map((g) => w.indexOf(g));
			for (const i of ids) assert.ok(ids.includes(neighbours(vs, i)[0].j), `${id} seed ${seed}: ${w[i]} -> ${w[neighbours(vs, i)[0].j]}`);
			// and pairs with many shared context words are more alike than pairs with few
			const ctx = cooccurrence(corpusOf(r)).map((row) => row.map((n) => n > 0));
			const shared = (a: number, b: number) => ctx[a].filter((x, j) => x && ctx[b][j]).length;
			const [hi, lo] = [[] as number[], [] as number[]];
			for (let a = 0; a < vs.length; a++) for (let b = a + 1; b < vs.length; b++) (shared(a, b) >= 4 ? hi : shared(a, b) <= 1 ? lo : []).push(cosine(vs[a], vs[b]));
			const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;
			assert.ok(mean(hi) > mean(lo) + 0.3, `${id} seed ${seed}: ${mean(hi)} vs ${mean(lo)}`);
		}
});

test('window: the window decides which words end up alike (pets: “the” is nearest to another word with window 1 than with 2)', () => {
	const nn = (w: number) => { const r: CorpusRaw = { text: PRESETS[0].text, window: w }, words = corpusOf(r).words; return words[neighbours(vecsOf(r, SEED), words.indexOf('the'))[0].j]; };
	assert.notEqual(nn(1), nn(2));
});

test('limits: on every preset at least a third of the words change their nearest word with another random start, and similarities move by 0.1 or more on average', () => {
	for (const p of PRESETS) {
		const r = raw(p.id), a = vecsOf(r, SEED), b = vecsOf(r, SEED2), V = a.length;
		const changed = a.filter((_, i) => neighbours(a, i)[0].j !== neighbours(b, i)[0].j).length;
		let d = 0, n = 0;
		for (let i = 0; i < V; i++) for (let j = i + 1; j < V; j++) (d += Math.abs(cosine(a[i], a[j]) - cosine(b[i], b[j]))), n++;
		assert.ok(changed >= V / 3 && d / n >= 0.1, `${p.id}: ${changed} of ${V} changed, mean |Δcos| ${d / n}`);
	}
});

test('adapter: no pairs means an empty input (the training steps show the note), and the cap keeps 40 words', () => {
	assert.equal(corpusOf({ text: 'Cat.' }).pairs.length, 0);
	const many = corpusOf({ text: Array.from({ length: 45 }, (_, i) => `w${i}`).join(' ') });
	assert.equal(many.words.length, 40);
	assert.equal(many.dropped.length, 5);
});

test('the page says 3,000 steps, learning rate 0.1, plain SGD, vectors of 5 numbers: the trainer defaults', async () => {
	assert.deepEqual([sg.defaults.steps, sg.defaults.lr, sg.defaults.optimizer, sg.defaults.dim, sg.defaults.batch], [3000, 0.1, 'sgd', 5, 1]);
	const m = (await pkgP).manifest;
	assert.ok(m.page.lessonNote!.en.includes('3,000') && !!m.page.lessonNote!.zh?.includes('3,000'));
	assert.equal((await pkgP).lesson!.steps.length, 11, 'the lesson note says eleven steps');
	assert.ok(m.page.lessonNote!.en.startsWith('Eleven steps') && !!m.page.lessonNote!.zh?.startsWith('十一步'));
});

// ---- added by the teaching review (notes/review-word2vec.md)

test('pairs: the step and its scrubber open on the first center word with the most context words (the narration’s example)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'pairs');
		if (!t?.length) return;
		const ks: number[] = [], ex: number[] = [];
		for (let i = 0; i < t.length; i++) { const f = t.frame(i); if (f.phase === 'center') { ks.push(f.metrics!.k); if (f.metrics!.ex === 1) ex.push(ks.length - 1); } }
		const kmax = Math.max(...ks);
		assert.deepEqual(ex, [ks.indexOf(kmax)]);
		const c = corpusOf(rt.input as CorpusRaw);
		if (c.sentences.some((s) => s.length > 2 * c.window)) assert.equal(kmax, 2 * c.window, 'a full window when the text has one');
	}, opts);
});

test('update: every output vector moves, the context word’s the most; the worked number is the center row’s largest gradient', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, 'one');
		if (!t?.length) return;
		const i = t.find({ phase: 'update', step: 0 });
		const w = worked(rt.input as CorpusRaw, { index: i, frame: t.frame(i), state: t.state(i), tape: t })!;
		const norm = (j: number) => Math.hypot(...w.wout1![j].map((x, k) => x - w.wout[j][k]));
		for (let j = 0; j < w.V; j++) {
			assert.ok(norm(j) > 0, `output vector ${j} moved`);
			// |Δ v_o| = lr (1 − p_o) |h| ≥ lr p_j |h| = |Δ v_j|, since p_o + p_j ≤ 1 (equal only with two words)
			if (j !== w.o) assert.ok(norm(w.o) >= norm(j) * (1 - 1e-12) && (w.V === 2 || norm(w.o) > norm(j)), `the context word’s output vector moves the most (${j})`);
		}
		const x = workedWeight(w)!;
		assert.ok(close(x.new, x.old - w.lr * x.g, 1e-12));
		assert.ok(w.gin![w.c].every((g) => Math.abs(g) <= Math.abs(x.g)));
	}, opts);
});

test('window: “happy” and “sad” mean opposite things yet end alike on the pets and kids presets, at every window (seeds 11 and 2)', () => {
	for (const id of ['pets', 'kids'])
		for (const window of [1, 2, 3])
			for (const seed of [SEED, SEED2]) {
				const r: CorpusRaw = { text: PRESETS.find((p) => p.id === id)!.text, window }, c = corpusOf(r), vs = vecsOf(r, seed);
				const [a, b] = [c.words.indexOf('happy'), c.words.indexOf('sad')];
				const ctx = cooccurrence(c).map((row) => row.map((n) => n > 0));
				assert.ok(ctx[a].some((x, j) => x && ctx[b][j]), `${id} w${window}: happy and sad share a context word`);
				assert.ok(cosine(vs[a], vs[b]) >= 0.5, `${id} w${window} seed ${seed}: cos ${cosine(vs[a], vs[b])}`);
			}
});

test('try-this “A new animal”: fox never stands with “wanted”, and ends nearest to cat or dog', async () => {
	const m = (await pkgP).manifest, r = m.page.tryThis![0].setup!.input as CorpusRaw, c = corpusOf(r), fox = c.words.indexOf('fox');
	assert.equal(cooccurrence(c)[fox][c.words.indexOf('wanted')], 0);
	for (const seed of [SEED, SEED2]) {
		const vs = vecsOf(r, seed), nn = c.words[neighbours(vs, fox)[0].j];
		assert.ok(['cat', 'dog'].includes(nn), `seed ${seed}: fox -> ${nn}`);
	}
});
