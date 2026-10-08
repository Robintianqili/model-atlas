// gcn claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed live
// is asserted here; per-fixture claims run through withFixtures (lesson fixtures + the adapter's edge fixture).
// `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/gcn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { input, noGrad } from '../../atlas/autograd/var.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { atFor } from '../../atlas/runtime/sources.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { adjacency, forwardGcn, graphOf, hops, KARATE, LABELS, N, neighbours, normalize, POS, sorted, type Params } from './graph.ts';
import { gcn } from './trainer.ts';
import { mp, mpInput, smooth } from './process.ts';
import { ISOLATED } from './adapters/graph/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('gcn', SITE)).pkg;
const json = (f: string) => JSON.parse(readFileSync(join(SITE, 'static/models/gcn', f), 'utf8'));
const T = (rt: ExplainerRuntime) => rt.sources.fwd!.trace!.t;
const weights = (rt: ExplainerRuntime): Params => rt.model!.weights as Params;

test('graph.ts holds Zachary’s club as networkx gives it (karate.json): 34 members, 78 friendships, factions, layout', () => {
	const k = json('karate.json');
	assert.equal(k.n, N);
	assert.equal(N, 34);
	assert.deepEqual(KARATE.edges, k.edges);
	assert.equal(KARATE.edges.length, 78);
	assert.deepEqual(LABELS, k.labels);
	assert.deepEqual(k.clubs, ['Mr. Hi', 'Officer']);
	// the instructor Mr. Hi is member 0 (faction 0), the Officer member 33 (faction 1); they are the two labelled members
	assert.deepEqual([LABELS[0], LABELS[33]], [0, 1]);
	assert.deepEqual(KARATE.labelled, [0, 33]);
	assert.equal(LABELS.length - KARATE.labelled.length, 32);
	POS.forEach((p, i) => assert.ok(Math.abs(p[0] - k.pos[i][0]) < 1e-9 && Math.abs(p[1] - k.pos[i][1]) < 1e-9, `pos ${i}`));
	assert.equal(LABELS.filter((c) => c === 0).length, 17);
});

test('catalog and narration: 156 parameters (34 -> 4 -> 2 graph layers, 2 -> 2 head); the browser trainer has the same', () => {
	const spec = json('spec.json');
	assert.equal(spec.params, 156);
	const shapes = Object.fromEntries(spec.tensors.map((t: { name: string; shape: number[] }) => [t.name, t.shape]));
	assert.deepEqual(shapes, { 'gc1.weight': [4, 34], 'gc1.bias': [4], 'gc2.weight': [2, 4], 'gc2.bias': [2], 'head.weight': [2, 2], 'head.bias': [2] });
	const m = gcn.model(createRng(1).fork('init'), gcn.defaults);
	assert.equal(m.params().reduce((s, p) => s + p.value.data.length, 0), 156);
	assert.deepEqual([gcn.defaults.optimizer, gcn.defaults.lr, gcn.defaults.steps], ['adam', 0.01, 200]);
	// the exported model was trained with the same recipe (pipeline/gcn/train.py, recorded in facts.json and spec.training)
	const f = json('facts.json').facts;
	assert.deepEqual([f.lr.value, f.steps.value], [gcn.defaults.lr, gcn.defaults.steps]);
	assert.match(spec.training.dataset, /labels of members 0 and 33 only/);
});

test('pre-question and misconception: trained on members 0 and 33 only, the network sorts all but one of the 32 others', async () => {
	const facts = json('facts.json').facts;
	assert.equal(facts.trainAcc.value, 31 / 32);
	await withFixtures(await pkg(), async (rt, fx) => {
		if (fx.id !== 'club') return;
		const s = sorted(T(rt).head.data, KARATE.labelled);
		assert.deepEqual([s.rest, s.right, s.wrong], [32, 31, [8]]);
	}, opts);
});

test('the view and the trainer compute the same network as the exported trace (forwardGcn = fwd trace, every fixture)', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const f = forwardGcn(weights(rt), normalize(adjacency(graphOf(rt.input)), N).ahat);
		const t = T(rt);
		f.z.forEach((v, k) => assert.ok(Math.abs(v - t.head.data[k]) < 1e-5, `head ${k}`));
		f.emb.forEach((v, k) => assert.ok(Math.abs(v - t.act2.data[k]) < 1e-5, `act2 ${k}`));
	}, opts);
	const m = gcn.model(createRng(1).fork('init'), gcn.defaults);
	const b = gcn.batch(KARATE, 0, createRng(1), gcn.defaults);
	const z = noGrad(() => m.forward(input(b.ahat), input(b.x))).value.data;
	const f = forwardGcn(Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.value])), b.ahat.data);
	f.z.forEach((v, k) => assert.ok(Math.abs(v - z[k]) < 1e-12));
});

test('steps loops, norm, transform: d = friends + 1, Âᵢⱼ = 1/√(dᵢ dⱼ), the self-loop 1/dᵢ, a busier friend weighs less; row j of X Wᵀ = column j of W', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const g = graphOf(rt.input);
		const t = T(rt);
		const i = g.focus;
		const d = (j: number) => neighbours(g, j).length + 1;
		assert.equal(t['norm#deg'].data[i], d(i));
		for (const j of [i, ...neighbours(g, i)]) assert.ok(Math.abs(t.norm.data[i * N + j] - 1 / Math.sqrt(d(i) * d(j))) < 1e-7);
		for (const a of neighbours(g, i)) for (const b of neighbours(g, i)) if (d(a) > d(b)) assert.ok(t.norm.data[i * N + a] < t.norm.data[i * N + b]);
		const W = weights(rt)['gc1.weight'].data;
		for (let j = 0; j < N; j++) for (let o = 0; o < 4; o++) assert.equal(t['gc1#xw'].data[j * 4 + o], W[o * N + j]);
	}, opts);
	// quiz: 3 and 8 friends -> degrees 4 and 9 -> 1/6
	const star = (k: number, off: number) => Array.from({ length: k }, (_, q) => [off, off + 1 + q] as [number, number]);
	const g2 = graphOf({ edges: [...star(2, 0), [0, 10], ...star(7, 10)], labelled: [], focus: 0 });
	assert.deepEqual([neighbours(g2, 0).length, neighbours(g2, 10).length], [3, 8]);
	assert.ok(Math.abs(normalize(adjacency(g2), N).ahat[0 * N + 10] - 1 / 6) < 1e-15);
});

test('steps messages and tanh: the messages add up to the traced layer 1 (sum + b = gc1, tanh = act1); member 11 alone gets one message of weight 1', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await rt.ensure('mp', 'last');
		const tape = rt.sources.mp!.tape!;
		const g = graphOf(rt.input);
		const msgs = Array.from({ length: tape.length }, (_, k) => tape.frame(k)).filter((f) => f.phase === 'message');
		assert.equal(msgs.length, neighbours(g, g.focus).length + 1);
		const act = tape.frame(tape.length - 1).metrics!;
		const t = T(rt);
		for (let f = 0; f < 4; f++) {
			assert.ok(Math.abs(act[`s${f}`] + act[`b${f}`] - t.gc1.data[g.focus * 4 + f]) < 1e-6);
			assert.equal(act[`h${f}`], t.act1.data[g.focus * 4 + f]);
			assert.ok(Math.abs(Math.tanh(act[`z${f}`]) - act[`h${f}`]) < 1e-6);
		}
	}, opts);
	const rt = await createRuntime(await pkg(), { ...opts, input: ISOLATED });
	const t = record(mp, mpInput(rt), {}, { seed: 1 });
	const m = Array.from({ length: t.length }, (_, k) => t.frame(k)).filter((f) => f.phase === 'message');
	assert.deepEqual([m.length, m[0].metrics!.j, m[0].metrics!.a], [1, 11, 1]);
});

test('step hops (predict): after two layers no message from three or more friendships away reaches the member in focus', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const g = graphOf(rt.input);
		const d = hops(g, g.focus);
		const far = d.map((x, k) => (x < 0 || x >= 3 ? k : -1)).filter((k) => k >= 0);
		const pre = rt.pre!.tensors as Record<string, Tensor>;
		const x = { shape: [N, N], data: Float32Array.from(pre.x.data) };
		for (const k of far) x.data[k * N + k] = 5; // change what every far member sends
		const t2 = rt.model!.forward({ adj: pre.adj, x });
		for (let o = 0; o < 2; o++) assert.equal(t2.t.act2.data[g.focus * 2 + o], T(rt).act2.data[g.focus * 2 + o]);
		if (far.length) assert.ok(far.some((k) => t2.t.act2.data[k * 2] !== T(rt).act2.data[k * 2]), 'the change does reach the far members themselves');
	}, opts);
});

test('step retrain: the twin-checked trainer learns from the labelled members only and sorts most of the club from 0 and 33', async () => {
	// the Training section's intro calls it "lesson step 10"
	assert.equal((await pkg()).lesson!.steps.findIndex((s) => 'id' in s && s.id === 'retrain'), 9);
	const b = gcn.batch({ ...KARATE, labelled: [1, 2] }, 0, createRng(1), gcn.defaults);
	assert.deepEqual(Array.from(b.labelled, (i) => b.y[i]), [0, 0]); // two labels from one faction: the loss sees one faction
	await withFixtures(await pkg(), async (rt, fx) => {
		if (fx.id !== 'club') return;
		await rt.ensure('train', 'last');
		const tape = rt.sources.train!.tape!;
		const last = tape.frame(tape.length - 1);
		assert.equal(last.step, 200);
		assert.ok(last.metrics!.accuracy >= 0.9, `accuracy ${last.metrics!.accuracy}`);
	}, opts);
});

test('step smooth (limits): the leaders’ rows grow alike; colour first spreads, then fades; the 8-layer facts are measured', async () => {
	for (const raw of [KARATE, ISOLATED, { ...KARATE, edges: [...KARATE.edges, [16, 33]] }]) {
		const t = record(smooth, raw, { rounds: 8 }, { seed: 1 });
		const m = (k: number) => t.frame(k).metrics!;
		for (let k = 1; k <= 8; k++) assert.ok(m(k).leaders > m(k - 1).leaders);
		assert.ok(m(2).spread > m(0).spread && m(8).spread < m(2).spread, `spread ${m(0).spread} ${m(2).spread} ${m(8).spread}`);
	}
	const none = record(smooth, { ...KARATE, edges: [] }, { rounds: 8 }, { seed: 1 });
	assert.equal(none.frame(8).metrics!.leaders, 0); // no friendships: nothing is averaged
	const f = json('facts.json').facts;
	assert.ok(f.deepAcc.value >= 0.9 && f.shallowAcc.value >= 0.9 && f.seeds.value === 10 && /8 graph layers/.test(f.deepAcc.source));
});

// ---------------------------------------------------------------- added by the teaching review (notes/review-gcn.md)
test('step messages (review): the scrubber starts on the first message, the member’s own, with weight 1/d; the line under the table is Â × its numbers', async () => {
	const step = (await pkg()).lesson!.steps.find((s) => 'id' in s && s.id === 'messages') as { at: unknown; play: { from: unknown } };
	assert.deepEqual(step.play.from, { phase: 'message' }); // the scrubber starts on the first message, which the narration works out
	assert.deepEqual(step.at, { phase: 'message', last: true });
	await withFixtures(await pkg(), async (rt) => {
		await rt.ensure('mp', 'last');
		const tape = rt.sources.mp!.tape!;
		const g = graphOf(rt.input);
		const first = tape.frame(tape.find({ phase: 'message' })).metrics!;
		assert.equal(first.j, g.focus);
		assert.equal(first.k, 1);
		assert.equal(first.of, neighbours(g, g.focus).length + 1);
		assert.ok(Math.abs(first.a - 1 / (neighbours(g, g.focus).length + 1)) < 1e-7);
		for (let f = 0; f < 4; f++) {
			assert.equal(first[`m${f}`], first.a * first[`x${f}`]);
			assert.equal(first[`s${f}`], first[`m${f}`]);
		}
	}, opts);
});

test('step retrain (review): the task is done only when both labelled members are new (neither 0 nor 33) and the retraining has finished', async () => {
	const step = (await pkg()).lesson!.steps.find((s) => 'id' in s && s.id === 'retrain') as { task: { done: (rt: ExplainerRuntime, at: unknown) => boolean } };
	const end = { tape: { opt: { steps: 200 } }, frame: { step: 200 } };
	const done = (labelled: number[], at: unknown = end) => step.task.done({ input: { ...KARATE, labelled } } as unknown as ExplainerRuntime, at);
	assert.deepEqual([done([0, 33]), done([33, 1]), done([0, 30]), done([1]), done([1, 30]), done([1, 2])], [false, false, false, false, true, true]);
	assert.equal(done([1, 30], { tape: { opt: { steps: 200 } }, frame: { step: 120 } }), false);
	// on the real runtime: the 'same-side' fixture (labels 1 and 2) is done at the end of its run, not halfway
	await withFixtures(await pkg(), async (rt, fx) => {
		if (fx.id !== 'same-side') return;
		await rt.ensure('train', 'last');
		const tape = rt.sources.train!.tape!;
		assert.equal(step.task.done(rt, atFor(rt, 'train', tape.length - 1)), true);
		assert.equal(step.task.done(rt, atFor(rt, 'train', Math.floor(tape.length / 2))), false);
	}, opts);
});

test('step embed (review): on every fixture many members sit near a corner of the square (both numbers past ±0.8)', async () => {
	const { cornerCount } = await import('./views/graph/meta.ts');
	await withFixtures(await pkg(), async (rt, fx) => {
		const c = cornerCount(T(rt).act2.data);
		assert.ok(c >= 10, `${fx.id}: ${c}`);
		if (fx.id === 'club') assert.equal(c, 14);
	}, opts);
});

test('step smooth (review): two leaders not linked by any chain of friendships keep cosine similarity 0 after every round', () => {
	const t = record(smooth, { ...KARATE, edges: KARATE.edges.filter(([a, b]) => a !== 0 && b !== 0) }, { rounds: 8 }, { seed: 1 });
	assert.ok(hops(graphOf({ ...KARATE, edges: KARATE.edges.filter(([a, b]) => a !== 0 && b !== 0) }), 0)[33] < 0);
	for (let k = 0; k <= 8; k++) assert.equal(t.frame(k).metrics!.leaders, 0);
});

test('step embed caption (review): the trained weights carry the recorded friendships in each member’s own column of W (no friendships at all: still 30 of 32)', async () => {
	await withFixtures(await pkg(), async (rt, fx) => {
		if (fx.id !== 'none') return;
		const s = sorted(T(rt).head.data, KARATE.labelled);
		assert.equal(s.right, 30);
		// with no edges, layer 1 of member i is tanh(column i of W + b): nothing but its own column
		const W = weights(rt)['gc1.weight'].data;
		const b = weights(rt)['gc1.bias'].data;
		for (let i = 0; i < N; i++) for (let o = 0; o < 4; o++) assert.ok(Math.abs(T(rt).act1.data[i * 4 + o] - Math.tanh(W[o * N + i] + b[o])) < 1e-6);
	}, opts);
});
