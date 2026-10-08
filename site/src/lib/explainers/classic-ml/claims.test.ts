// classic-ml claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (lesson fixtures, the adapter's edge fixtures) where it depends on the input. `atlas check` runs this file.
//   node --test src/lib/explainers/classic-ml/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { kmeans, knn, knnMap, tree } from './process.ts';
import { cellX, cellY, dataOf, DMAX, gini, KS, majority, patches, R, RINGS, START, type MlData } from './ml.ts';
import lesson from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('classic-ml', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const of = (t: Tape, phase: string) => all(t).filter((f) => f.phase === phase);
const arr = (x: unknown) => Array.from((x as Tensor).data);
async function tapeOf(rt: AtlasRuntime, source: string): Promise<Tape | null> {
	await rt.ensure(source, 'last');
	return rt.sources[source].tape ?? null;
}
async function each(source: string, fn: (t: Tape, d: MlData) => void): Promise<void> {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt, source), d = dataOf(rt.input);
		if (!d.label.length) return assert.equal(t?.length ?? 0, 0, 'no points, no frames');
		fn(t!, d);
	}, opts);
}
/** the training points in a node's box (a cut sends x <= t to the first child; the plane's edge is no bound) */
const inBox = (d: MlData, train: number[], m: Record<string, number>) =>
	train.filter((i) => [0, 1].every((a) => { const v = d.xy[2 * i + a], lo = a ? m.y0 : m.x0, hi = a ? m.y1 : m.x1; return (lo === -1.25 || v > lo) && (hi === 1.25 || v <= hi); }));
/** the largest impurity drop of any cut of these points */
function bestDrop(d: MlData, idx: number[]): number {
	const cnt = (xs: number[]) => Array.from({ length: d.classes }, (_, k) => xs.filter((i) => d.label[i] === k).length), g = gini(cnt(idx), idx.length);
	let best = 0;
	for (const a of [0, 1])
		for (const v of new Set(idx.map((i) => d.xy[2 * i + a]))) {
			const L = idx.filter((i) => d.xy[2 * i + a] <= v), Rr = idx.filter((i) => d.xy[2 * i + a] > v);
			if (L.length && Rr.length) best = Math.max(best, g - ((L.length / idx.length) * gini(cnt(L), L.length) + (Rr.length / idx.length) * gini(cnt(Rr), Rr.length)));
		}
	return best;
}
const dist = (d: MlData, i: number, x: number, y: number) => Math.sqrt((d.xy[2 * i] - x) ** 2 + (d.xy[2 * i + 1] - y) ** 2);

test('lesson note: twelve steps in three chapters', () => {
	assert.equal(lesson.steps.length, 12);
	assert.deepEqual([...new Set(lesson.steps.map((s) => ('id' in s ? s.id.split('-')[0] : '')))], ['knn', 'tree', 'km']);
});

test('knn-dist, knn-vote: straight-line distances; the k nearest inside the circle vote, ties to the smaller class; no training', () =>
	each('knn', (t, d) => {
		const [q, , v] = [t.frame(0), t.frame(1), t.frame(2)], s = t.state(2) as { d: Tensor; nn: Tensor }, ds = arr(s.d), nn = arr(s.nn);
		ds.forEach((x, i) => assert.ok(Math.abs(x - dist(d, i, q.metrics!.qx, q.metrics!.qy)) < 1e-12));
		const k = Math.min(d.k, d.label.length), r = v.metrics!.r;
		assert.equal(nn.length, k);
		assert.equal(r, ds[nn[k - 1]]);
		ds.forEach((x, i) => assert.ok(nn.includes(i) ? x <= r : x >= r, `point ${i} on the right side of the circle`));
		const votes = new Array(d.classes).fill(0);
		nn.forEach((i) => votes[d.label[i]]++);
		const top = Math.max(...votes);
		assert.equal(v.metrics!.pred, votes.indexOf(top), 'the most votes; a tie goes to the smaller class number');
		assert.deepEqual(Object.keys(knn.phases), ['query', 'distances', 'vote'], 'nothing is trained: only distances and a vote');
	}));

test('knn-vote: a tie goes to the smaller class number', () => {
	const t = record(knn, { xy: [0.1, 0, -0.1, 0], label: [1, 0], classes: 2, q: [0, 0], k: 2, depth: 8, kc: 2 }, {}, { seed: 1 });
	assert.equal(t.frame(2).metrics!.tie, 1);
	assert.equal(t.frame(2).metrics!.pred, 0);
});

test('knn-map: with k = 1 every spot takes the class of its nearest point; on the start points k = 1 leaves islands and a larger k smooths them', () => {
	const d = dataOf(START), t = record(knnMap, d, {}, { seed: 1 }), n = d.label.length;
	const g1 = arr(t.state(0).grid);
	for (let r = 0; r < R; r++)
		for (let c = 0; c < R; c++) {
			let b = 0;
			for (let i = 1; i < n; i++) if (dist(d, i, cellX(c), cellY(r)) < dist(d, b, cellX(c), cellY(r))) b = i;
			assert.equal(g1[r * R + c], d.label[b]);
		}
	const edges = (g: number[]) => { let e = 0; for (let r = 0; r < R; r++) for (let c = 0; c < R; c++) e += (c + 1 < R && g[r * R + c] !== g[r * R + c + 1] ? 1 : 0) + (r + 1 < R && g[r * R + c] !== g[(r + 1) * R + c] ? 1 : 0); return e; };
	const last = arr(t.state(t.length - 1).grid);
	assert.equal(t.frame(t.length - 1).metrics!.k, KS[KS.length - 1]);
	assert.ok(patches(g1) > patches(last) + 2, `islands: ${patches(g1)} patches at k = 1, ${patches(last)} at k = 25`);
	assert.ok(edges(g1) > 1.4 * edges(last), `a smoother boundary: ${edges(g1)} cell edges at k = 1, ${edges(last)} at k = 25`);
});

test('tree-gini: G = 1 − Σ p² on the training points; a single class gives 0, C classes in equal shares give the largest, 1 − 1/C; the held-out points never change the tree', () => {
	assert.equal(gini([5, 0], 5), 0);
	assert.equal(gini([30, 10], 40), 0.375, 'the quiz');
	assert.equal(gini([7, 7], 14), 0.5, 'two classes half and half');
	assert.ok(Math.abs(gini([4, 4, 4], 12) - 2 / 3) < 1e-15, 'three classes in equal thirds');
	return each('tree', (t, d) => {
		for (const f of of(t, 'node').concat(of(t, 'gini'))) assert.ok(f.metrics!.gini <= 1 - 1 / d.classes + 1e-12, `G ${f.metrics!.gini} is at most 1 − 1/${d.classes}`);
		const h = t.frame(0).metrics!, g = t.frame(1).metrics!, test = arr(t.state(0).test);
		assert.equal(h.nTest, d.label.length >= 2 ? Math.max(1, Math.floor(d.label.length / 4)) : 0);
		assert.equal(test.filter((x) => x === 1).length, h.nTest);
		const c = Array.from({ length: d.classes }, (_, k) => g[`c${k}`]);
		assert.ok(Math.abs(g.gini - (1 - c.reduce((s, x) => s + (x / g.m) ** 2, 0))) < 1e-12);
		const flipped = { ...d, label: d.label.map((l, i) => (test[i] ? (l + 1) % Math.max(2, d.classes) : l)), classes: Math.max(2, d.classes) };
		const nodes = (x: Tape) => of(x, 'node').map((f) => [f.metrics!.axis, f.metrics!.thr ?? 0].join());
		assert.deepEqual(nodes(record(tree, flipped, {}, { seed: t.seed })), nodes(t), 'relabelling the held-out points leaves every question as it was');
	});
});

test('tree-split: every cut halfway between neighbours on x and y, scored by the weighted Gini; the best has the biggest drop', () =>
	each('tree', (t, d) => {
		const f = of(t, 'node')[0], m = f.metrics!, train = arr(t.state(0).test).map((x, i) => (x ? -1 : i)).filter((i) => i >= 0);
		if (m.axis < 0) {
			const pure = m.gini === 0, oneSpot = train.every((i) => d.xy[2 * i] === d.xy[2 * train[0]] && d.xy[2 * i + 1] === d.xy[2 * train[0] + 1]);
			return assert.ok(pure || oneSpot || train.length === 1, 'a leaf at the root: one class, one spot or one point');
		}
		const cand = arr((t.state(all(t).findIndex((x) => x.phase === 'node')) as { cand: Tensor }).cand);
		const want = [0, 1].reduce((s, a) => s + new Set(train.map((i) => d.xy[2 * i + a])).size - 1, 0);
		assert.equal(cand.length / 3, want, 'one cut between every two neighbouring values, on x and on y');
		const ws = cand.filter((_, j) => j % 3 === 2);
		assert.equal(m.w, Math.min(...ws));
		assert.ok(Math.abs(m.gain - (m.gini - m.w)) < 1e-15);
		const side = train.filter((i) => d.xy[2 * i + m.axis] <= m.thr), cnt = (idx: number[]) => Array.from({ length: d.classes }, (_, k) => idx.filter((i) => d.label[i] === k).length);
		const rest = train.filter((i) => !side.includes(i));
		assert.ok(Math.abs(m.w - ((side.length / train.length) * gini(cnt(side), side.length) + (rest.length / train.length) * gini(cnt(rest), rest.length))) < 1e-12);
	}));

test('tree-grow: a leaf holds one class, has no purer cut or sits 8 deep; it answers with its most common class; leaves = questions + 1', () =>
	each('tree', (t, d) => {
		const nodes = of(t, 'node').map((f) => f.metrics!), splits = nodes.filter((m) => m.axis >= 0).length;
		const train = arr(t.state(0).test).map((x, i) => (x ? -1 : i)).filter((i) => i >= 0);
		for (const m of nodes) {
			const c = Array.from({ length: d.classes }, (_, k) => m[`c${k}`]), idx = inBox(d, train, m);
			assert.equal(m.pred, majority(c));
			assert.equal(idx.length, m.m, `node ${m.node}: its box holds its points`);
			assert.ok(m.depth < DMAX);
			if (m.axis < 0) assert.ok(m.gini === 0 || bestDrop(d, idx) <= 1e-12, `leaf ${m.node}: one class, or no cut makes it purer`);
		}
		assert.equal(of(t, 'depth').at(-1)!.metrics!.leaves, splits + 1);
	}));

test('tree-depth: more depth never fits the training points worse; on the start points the deepest tree overfits single stray points', async () => {
	await each('tree', (t) => {
		const ds = of(t, 'depth').map((f) => f.metrics!), acc = ds.map((x) => x.trainAcc);
		acc.forEach((a, i) => i && assert.ok(a >= acc[i - 1], `depth ${i + 1}: ${a} < ${acc[i - 1]}`));
		// wherever the narration says "the deepest boxes wrap single stray points" (held-out accuracy below its best at
		// depth DMAX), a question below the best depth splits off a box of one or two training points
		const best = ds.filter((x) => x.testAcc >= 0).reduce((b, x) => (x.testAcc > b.testAcc ? x : b), ds[0]);
		if (best && best.testAcc >= 0 && ds[DMAX - 1].testAcc < best.testAcc) {
			const tiny = of(t, 'node').map((f) => f.metrics!).filter((x) => x.axis >= 0 && x.depth >= best.d && (x.nL <= 2 || x.nR <= 2));
			assert.ok(tiny.length >= 1, 'a deep box around one or two stray points');
		}
	});
	const t = record(tree, dataOf(START), {}, { seed: 1 }), ds = of(t, 'depth').map((f) => f.metrics!);
	assert.ok(Math.max(...ds.map((x) => x.testAcc)) > ds[DMAX - 1].testAcc, 'some shallower tree does better on the held-out points (the misconception, too)');
	const leaves = of(t, 'node').map((f) => f.metrics!).flatMap((m) => (m.axis >= 0 ? [m.nL, m.nR] : [])), small = leaves.filter((n) => n <= 2).length;
	assert.ok(small >= 3, `boxes around one or two stray points: ${small}`);
	assert.ok(ds[DMAX - 1].trainAcc > 0.97);
});

test('km-init, km-assign, km-update, km-repeat: random points start; nearest centroid; mean; stop when nothing changes; the total never goes up', () =>
	each('km', (t, d) => {
		const n = d.label.length, init = t.frame(0).metrics!, k = init.k, C0 = arr(t.state(0).C);
		const picks = Array.from({ length: k }, (_, j) => init[`p${j}`]);
		assert.equal(new Set(picks).size, k);
		picks.forEach((i, j) => assert.deepEqual([C0[2 * j], C0[2 * j + 1]], [d.xy[2 * i], d.xy[2 * i + 1]]));
		let tot = Infinity;
		all(t).forEach((f, i) => {
			const s = t.state(i) as { C: Tensor; assign: Tensor }, C = arr(s.C), a = arr(s.assign);
			const d2 = (p: number, j: number) => (d.xy[2 * p] - C[2 * j]) ** 2 + (d.xy[2 * p + 1] - C[2 * j + 1]) ** 2;
			if (f.phase === 'assign') for (let p = 0; p < n; p++) for (let j = 0; j < k; j++) assert.ok(d2(p, a[p]) <= d2(p, j), `point ${p} joins its nearest centroid`);
			if (f.phase === 'update')
				for (let j = 0; j < k; j++) {
					const mine = a.map((x, p) => (x === j ? p : -1)).filter((p) => p >= 0);
					if (mine.length) assert.ok(Math.abs(C[2 * j] - mine.reduce((s2, p) => s2 + d.xy[2 * p], 0) / mine.length) < 1e-12, 'the mean');
				}
			if (f.phase === 'assign' || f.phase === 'update') {
				assert.ok(f.metrics!.inertia <= tot * (1 + 1e-12) + 1e-15, `step ${i}: the total went up`);
				tot = f.metrics!.inertia;
			}
		});
		const assigns = of(t, 'assign');
		assert.equal(assigns.at(-1)!.metrics!.changed, 0);
		assigns.slice(0, -1).forEach((f) => assert.ok(f.metrics!.changed > 0));
		assert.equal(of(t, 'done').length, 1);
	}));

test('quiz: k-means never reads the classes', () => {
	const d = dataOf(START), a = record(kmeans, d, { sweep: 0 }, { seed: 3 }), b = record(kmeans, { ...d, label: d.label.map((l) => 1 - l) }, { sweep: 0 }, { seed: 3 });
	const key = (t: Tape) => all(t).map((f, i) => `${f.phase}:${arr((t.state(i) as { C: Tensor }).C).join()}:${arr((t.state(i) as { assign: Tensor }).assign).join()}`).join('|');
	assert.equal(key(a), key(b));
});

test('km-limits: on the ring the total keeps falling as k grows; with k = 2 the border is straight and each group holds about half of each shape', () => {
	const d = dataOf(RINGS), t = record(kmeans, d, { sweep: 6 }, { seed: 3 }), sw = of(t, 'sweep').map((f) => f.metrics!.inertia);
	assert.equal(sw.length, 6);
	sw.forEach((v, i) => i && assert.ok(v < sw[i - 1], `k = ${i + 1}`));
	const done = of(t, 'done')[0].metrics!, s = t.state(t.length - 1) as { C: Tensor; assign: Tensor }, C = arr(s.C), a = arr(s.assign);
	assert.equal(done.k, 2);
	for (const [g, c] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
		const share = done[`m${g * 2 + c}`] / d.label.filter((l) => l === c).length;
		assert.ok(share > 0.4 && share < 0.6, `group ${g}, class ${c}: ${share}`);
	}
	// the border between two centroids is the straight line where both are equally far: a point's side of it is its group
	const side = (i: number) => (d.xy[2 * i] - (C[0] + C[2]) / 2) * (C[2] - C[0]) + (d.xy[2 * i + 1] - (C[1] + C[3]) / 2) * (C[3] - C[1]) > 0 ? 1 : 0;
	a.forEach((g, i) => assert.equal(g, side(i)));
});

test('try this, rings: the tree carves the disc with cuts on both axes (a staircase)', () => {
	const t = record(tree, dataOf({ ...RINGS, depth: 8 }), {}, { seed: 1 }), axes = new Set(of(t, 'node').map((f) => f.metrics!.axis).filter((a) => a >= 0));
	assert.deepEqual([...axes].sort(), [0, 1]);
	assert.ok(of(t, 'node').filter((f) => f.metrics!.axis >= 0).length >= 4);
});

test('manifest: the sources use the seeds and data these tests check', async () => {
	const src = (await pkgP).manifest.sources! as Record<string, { seed?: number; opt?: Record<string, unknown> }>;
	assert.equal(src.tree.seed, 1);
	assert.equal(src.rings.seed, 3);
	assert.equal(src.rings.opt?.sweep, 6);
});
