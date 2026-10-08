// gnn-variants claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here; per-fixture claims run through withFixtures (lesson fixtures + the adapter's edge
// fixture). `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/gnn-variants/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { adjacency, forwardGat, graphOf, ISOLATED, KARATE, LABELS, N, neighbours, POS, VARIANTS, type Params } from './graph.ts';
import { gcnRule, ginRule, sageRule, weightOf } from './rules.ts';
import { drop, dropInput, gat, gatInput } from './process.ts';
import { gin_agg, sage_conv } from './ops/gnn.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('gnn-variants', SITE)).pkg;
const json = (f: string) => JSON.parse(readFileSync(join(SITE, 'static/models/gnn-variants', f), 'utf8'));
const T = (rt: ExplainerRuntime) => rt.sources.fwd!.trace!.t;
const near = (a: number[], b: ArrayLike<number>, tol: number, what: string) => a.forEach((v, k) => assert.ok(Math.abs(v - b[k]) <= tol, `${what}[${k}]: ${v} vs ${b[k]}`));
const row = (x: Tensor, r: number, w: number) => Array.from(x.data.subarray(r * w, (r + 1) * w));
const facts = () => json('facts.json').facts as Record<string, { value: number; source: string }>;

test('graph.ts holds Zachary’s club as networkx gives it (karate.json): 34 members, 78 friendships, factions, layout', () => {
	const k = json('karate.json');
	assert.deepEqual([k.n, N, KARATE.edges.length], [34, 34, 78]);
	assert.deepEqual(KARATE.edges, k.edges);
	assert.deepEqual(LABELS, k.labels);
	assert.deepEqual(k.clubs, ['Mr. Hi', 'Officer']);
	assert.deepEqual([LABELS[0], LABELS[33]], [0, 1]); // Mr. Hi is member 0, the Officer member 33
	POS.forEach((p, i) => assert.ok(Math.abs(p[0] - k.pos[i][0]) < 1e-9 && Math.abs(p[1] - k.pos[i][1]) < 1e-9, `pos ${i}`));
});

test('catalog: 808 parameters in four networks (GCN 156, GraphSAGE 300, GIN 184, GAT 168), 3.2 KB of weights; one recipe for all four', () => {
	const spec = json('spec.json');
	assert.equal(spec.params, 808);
	const per = Object.fromEntries(VARIANTS.map((v) => [v, spec.tensors.filter((t: { name: string }) => t.name.startsWith(`${v}.`)).reduce((s: number, t: { length: number }) => s + t.length, 0)]));
	assert.deepEqual(per, { gcn: 156, sage: 300, gin: 184, gat: 168 });
	const f = facts();
	assert.deepEqual(VARIANTS.map((v) => f[`${v}Params`].value), [156, 300, 184, 168]);
	assert.equal((statSync(join(SITE, 'static/models/gnn-variants/weights.bin')).size / 1000).toFixed(1), '3.2');
	assert.match(spec.training.optimizer, /the same for all four networks/);
	assert.match(spec.training.dataset, /labels of members 0 and 33 only/);
	assert.deepEqual([f.steps.value, f.lr.value, f.seeds.value], [200, 0.01, 10]);
});

test('pre-question and step accuracy: the plain GCN is the most accurate on average and the steadiest; GraphSAGE swings the most', () => {
	const f = facts();
	const range = (v: string) => f[`${v}Max`].value - f[`${v}Min`].value;
	for (const v of ['sage', 'gin', 'gat']) assert.ok(f.gcnMean.value > f[`${v}Mean`].value, v);
	for (const v of ['sage', 'gin', 'gat']) assert.ok(range('gcn') < range(v), `steadiest vs ${v}`);
	for (const v of ['gcn', 'gin', 'gat']) assert.ok(range('sage') > range(v), `widest vs ${v}`);
	for (const v of VARIANTS) assert.match(f[`${v}Mean`].source, /seeds 0-9/);
	assert.ok(f.sageMin.value < 0.5, 'step accuracy: GraphSAGE’s worst start is worse than a coin flip');
});

test('step embed: a second graph layer gives 2 numbers per member, and a last linear layer (the head) turns them into one score per faction', () => {
	const spec = json('spec.json');
	for (const v of VARIANTS) {
		const L = (id: string) => spec.layers.find((l: { id: string }) => l.id === id);
		assert.deepEqual(L(`${v}.act2`).outShape, [N, 2], `${v}.act2`);
		assert.equal(L(`${v}.head`).op, 'linear', `${v}.head`);
		assert.deepEqual(L(`${v}.head`).outShape, [N, 2], `${v}.head`);
	}
});

test('step recipe: a one-hot row sends its own column of W; Â = 1/√(dᵢ dⱼ) with d = friends + 1; the weighted messages + b give the traced layer', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const t = T(rt);
		const g = graphOf(rt.input);
		const W = rt.model!.weights['gcn.gc1.weight'].data;
		for (let j = 0; j < N; j++) for (let o = 0; o < 4; o++) assert.equal(t['gcn.gc1#xw'].data[j * 4 + o], W[o * N + j]);
		const d = (j: number) => neighbours(g, j).length + 1;
		for (const j of [g.focus, ...neighbours(g, g.focus)]) assert.ok(Math.abs((weightOf(rt, 'gcn', j) ?? NaN) - 1 / Math.sqrt(d(g.focus) * d(j))) < 1e-7);
		const R = gcnRule(rt);
		near(R.sum.map((s, o) => s + R.b[o]), R.z, 1e-5, 'gcn z');
		near(R.z.map(Math.tanh), R.h, 1e-6, 'gcn tanh');
	}, opts);
});

test('step sage: every friend counts 1/k; the own half of W reads column i, the friends’ half their average; own + average + b = the traced layer', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const t = T(rt);
		const g = graphOf(rt.input);
		const nb = neighbours(g, g.focus);
		const mean = row(t['sage.gc1#mean'], g.focus, N);
		mean.forEach((v, j) => assert.ok(Math.abs(v - (nb.includes(j) ? 1 / nb.length : 0)) < 1e-7, `mean ${j}`));
		const R = sageRule(rt);
		near(R.own.map((v, o) => v + R.mean[o] + R.b[o]), R.z, 1e-5, 'sage z');
		if (!nb.length) assert.deepEqual(R.mean, [0, 0, 0, 0]);
	}, opts);
});

test('step gin: the sum is (1 + ε) × the own row + the friends’ rows; its first linear layer = (1 + ε) × own column + friends’ columns + b₁', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const t = T(rt);
		const g = graphOf(rt.input);
		const nb = neighbours(g, g.focus);
		const R = ginRule(rt);
		const agg = row(t['gin.gc1.agg'], g.focus, N);
		agg.forEach((v, j) => assert.ok(Math.abs(v - (j === g.focus ? 1 + R.eps : nb.includes(j) ? 1 : 0)) < 1e-6, `agg ${j}`));
		const b1 = Array.from(rt.model!.weights['gin.gc1.mlp.0.bias'].data);
		near(R.own.map((v, o) => v + R.fr.reduce((s, c) => s + c[o], 0) + b1[o]), R.pre, 1e-5, 'gin mlp.0');
		near(R.pre.map((v) => Math.max(0, v)), R.relu, 0, 'relu');
	}, opts);
	assert.ok(Math.abs(ginRule(await createRuntime(await pkg(), opts)).eps) > 0.01, 'ε was learned away from 0');
});

test('predict gnn-variants-sum: three equal rows average to that row (GraphSAGE: same result for A and B) but sum to three times it (GIN: different)', () => {
	// members 0 (A) and 4 (B); A has one friend (1), B three (5, 6, 7); every friend sends the row (1, 2)
	const n = 8;
	const adj = new Float32Array(n * n);
	for (const [a, b] of [[0, 1], [4, 5], [4, 6], [4, 7]]) adj[a * n + b] = adj[b * n + a] = 1;
	const h = Float32Array.from({ length: n * 2 }, (_, k) => ([1, 5, 6, 7].includes(Math.floor(k / 2)) ? [1, 2][k % 2] : 0));
	const subs: Record<string, Tensor> = {};
	const ctx = (w: Record<string, Tensor>) => ({ layer: { id: 'L', op: 'x', label: { en: '', zh: '' }, in: [], outShape: [] }, dims: {}, w: (s: string) => w[s], has: () => true, emit: (k: string, x: Tensor) => void (subs[k] = x) });
	const A = { shape: [n, n], data: adj };
	const H = { shape: [n, 2], data: h };
	sage_conv.forward([H, A], {}, ctx({ weight: { shape: [1, 4], data: new Float32Array(4) }, bias: { shape: [1], data: new Float32Array(1) } }));
	assert.deepEqual([...row(subs.mean, 0, 2)], [...row(subs.mean, 4, 2)]);
	const s = gin_agg.forward([H, A], {}, ctx({ eps: { shape: [1], data: Float32Array.of(0) } }));
	assert.deepEqual([row(s, 0, 2), row(s, 4, 2)], [[1, 2], [3, 6]]);
});

test('step score and predict gnn-variants-top: LeakyReLU keeps positives and scales negatives by 0.2; the highest score gets the most weight; member 8 → member 0 (16 friends), member 33 → member 19 (3 friends)', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await rt.ensure('gat', 'last');
		const tape = rt.sources.gat!.tape!;
		const fr = Array.from({ length: tape.length }, (_, k) => tape.frame(k));
		const sc = fr.filter((x) => x.phase === 'score').map((x) => x.metrics!);
		for (const m of sc) assert.equal(m.e, m.raw > 0 ? m.raw : 0.2 * m.raw);
		assert.equal(sc[0].j, sc[0].node, 'the first score row (the one the narration quotes) is the member itself');
		assert.equal(rt.model!.weights['gat.gc1.att_src'].data.length, 4, 'a_src is a row of 4 numbers');
		const sm = fr.find((x) => x.phase === 'softmax')!.metrics!;
		const best = sc.filter((m) => m.j !== m.node).reduce((b, m) => (b === null || m.e > b.e ? m : b), null as null | Record<string, number>);
		if (best) assert.equal(sm.top, best.j);
	}, opts);
	const top = async (focus: number) => {
		const rt = await createRuntime(await pkg(), { ...opts, input: { ...KARATE, focus } });
		await rt.ensure('gat', 'last');
		const tape = rt.sources.gat!.tape!;
		return tape.frame(tape.find({ phase: 'softmax' })).metrics!.top;
	};
	const deg = (j: number) => neighbours(KARATE, j).length;
	assert.deepEqual([await top(8), deg(0), await top(33), deg(19)], [0, 16, 19, 3]);
});

test('steps softmax and mix: the weights lie in (0, 1], add up to 1 and equal the traced α; the messages + b give the traced GAT layer 1', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await rt.ensure('gat', 'last');
		const tape = rt.sources.gat!.tape!;
		const g = graphOf(rt.input);
		const w = (tape.state(tape.find({ phase: 'softmax' })) as { w: Tensor }).w.data;
		const S = [g.focus, ...neighbours(g, g.focus)];
		assert.ok(Math.abs(S.reduce((s, j) => s + w[j], 0) - 1) < 1e-12);
		for (const j of S) assert.ok(w[j] > 0 && w[j] <= 1);
		near(S.map((j) => w[j]), S.map((j) => T(rt)['gat.gc1#alpha'].data[g.focus * N + j]), 1e-6, 'alpha');
		const act = tape.frame(tape.length - 1).metrics!;
		near([0, 1, 2, 3].map((f) => act[`y${f}`]), row(T(rt)['gat.gc1'], g.focus, 4), 1e-6, 'gat.gc1');
		near([0, 1, 2, 3].map((f) => act[`h${f}`]), row(T(rt)['gat.act1'], g.focus, 4), 1e-6, 'gat.act1');
	}, opts);
});

test('step side: GCN, GraphSAGE and GIN weights follow from the friendships alone; only GAT’s change with the members’ numbers', async () => {
	await withFixtures(await pkg(), async (rt, fx) => {
		if (fx.id !== 'club' && fx.id !== 'hub') return;
		const pre = rt.pre!.tensors as Record<string, Tensor>;
		const x = { shape: [N, N], data: Float32Array.from(pre.x.data) };
		for (let j = 0; j < N; j++) x.data[j * N + j] = 1 + j / 10; // other numbers, the same friendships
		const t2 = rt.model!.forward({ adj: pre.adj, x });
		const t = T(rt);
		assert.deepEqual(Array.from(t2.t['gcn.gc1#ahat'].data), Array.from(t['gcn.gc1#ahat'].data));
		const nb = neighbours(graphOf(rt.input), graphOf(rt.input).focus);
		const mean = row(t2.t['sage.gc1#mean'], graphOf(rt.input).focus, N);
		assert.ok(nb.every((j) => mean[j] !== 0) && mean.filter((v) => v !== 0).length === nb.length, 'the average still takes every friend');
		assert.ok(Array.from(t2.t['gat.gc1#alpha'].data).some((v, k) => Math.abs(v - t['gat.gc1#alpha'].data[k]) > 1e-3), 'α moves');
	}, opts);
});

test('step limits and the misconception card: the drop process re-runs the exported GAT; on the club, member 8’s most-attended friend moves its gap least of the two', async () => {
	await withFixtures(await pkg(), async (rt) => {
		const p = dropInput(rt)!;
		const f = forwardGat(p.p as Params, adjacency(graphOf(rt.input)), p.slope);
		const head = T(rt)['gat.head'].data;
		f.gap.forEach((v, r) => assert.ok(Math.abs(v - (head[r * 2 + 1] - head[r * 2])) < 1e-4, `gap ${r}`));
		near(Array.from(f.emb), T(rt)['gat.act2'].data, 1e-5, 'emb');
	}, opts);
	const rt = await createRuntime(await pkg(), opts);
	const v = record(drop, dropInput(rt), {}, { seed: 1 });
	const m = v.frame(v.length - 1).metrics!;
	assert.equal(m.node, 8);
	assert.ok(m.aTopA > m.aTopE && Math.abs(m.dTopA) < Math.abs(m.dTopE) && m.same === 0);
	// misconception card: member 8's largest layer-1 weight goes to member 0, removing that friendship barely moves its
	// score; member 32 gets little layer-1 weight, removing it switches the side; layer 2's weights favour member 32
	assert.deepEqual([m.topA, m.topE, m.topA2, m.same2], [0, 32, 32, 1]);
	assert.ok(Math.abs(m.dTopA) < 0.05 * Math.abs(m.gap) && m.aTopE < 0.05, 'barely moves; little weight');
	assert.ok(m.dTopE * m.gap < 0 && Math.abs(m.dTopE) > Math.abs(m.gap), 'removing 8–32 switches member 8’s side');
	assert.ok(m.agree * 2 < m.members, 'only a minority agree');
	assert.ok(m.agree2 > m.agree, 'layer 2’s largest α agrees with the largest move more often than layer 1’s');
	// layer-2 α in the drop frames equals the traced gat.gc2#alpha
	const a2 = T(rt)['gat.gc2#alpha'].data;
	for (let k = 0; k < v.length; k++) { const x = v.frame(k).metrics!; if (x.a2 !== undefined) assert.ok(Math.abs(x.a2 - a2[8 * N + x.j]) < 1e-6, `a2 ${x.j}`); } // prettier-ignore
	// the task's hint: members with few friends are a good place to start (2 friends: most agree; 5 or more: rarely)
	const agrees = async (f: number) => { const r = await createRuntime(await pkg(), { ...opts, input: { ...KARATE, focus: f } }); const t = record(drop, dropInput(r), {}, { seed: 1 }); return t.frame(t.length - 1).metrics!.same === 1; }; // prettier-ignore
	const share = async (keep: (d: number) => boolean) => { const ids = Array.from({ length: N }, (_, f) => f).filter((f) => keep(neighbours(KARATE, f).length)); let k = 0; for (const f of ids) k += (await agrees(f)) ? 1 : 0; return k / ids.length; }; // prettier-ignore
	const [few, many] = [await share((d) => d === 2), await share((d) => d >= 5)];
	assert.ok(few > 0.5 && many < few, `2 friends: ${few}, 5 or more: ${many}`);
});

test('try-this “Member 11 alone” and the quiz: no friends → GCN and GAT keep the self-loop with weight 1, the average is 0s, GIN’s sum is (1 + ε) × own; 4 friends → 1/4 and 1', async () => {
	const rt = await createRuntime(await pkg(), { ...opts, input: ISOLATED });
	assert.equal(neighbours(graphOf(rt.input), 11).length, 0);
	assert.equal(weightOf(rt, 'gcn', 11), 1);
	assert.ok(Math.abs((weightOf(rt, 'gat', 11) ?? NaN) - 1) < 1e-7);
	assert.deepEqual(sageRule(rt).mean, [0, 0, 0, 0]);
	const agg = row(T(rt)['gin.gc1.agg'], 11, N);
	agg.forEach((v, j) => assert.ok(Math.abs(v - (j === 11 ? 1 + ginRule(rt).eps : 0)) < 1e-6));
	const g = record(gat, gatInput(rt), {}, { seed: 1 });
	assert.equal(Array.from({ length: g.length }, (_, k) => g.frame(k)).filter((f) => f.phase === 'message').length, 1);
	const four = KARATE.edges.flat().find((j) => neighbours(KARATE, j).length === 4)!;
	const r4 = await createRuntime(await pkg(), { ...opts, input: { ...KARATE, focus: four } });
	const fr = neighbours(KARATE, four)[0];
	assert.deepEqual([weightOf(r4, 'sage', fr), weightOf(r4, 'gin', fr)], [0.25, 1]);
	assert.ok((await pkg()).lesson!.steps.some((s) => 'id' in s && s.id === 'side'));
});
