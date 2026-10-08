// tinyvit claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the four lines, the adapter's blank / dot / small, the 12 presets) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/tinyvit/claims.test.ts
// The teaching review (notes/review-tinyvit.md) added the tests marked "review".
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, StepDef, Tensor, Trace } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { outputProbs } from '../../atlas/lesson/facts.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { dims, softmaxWorked } from './lesson.ts';
import { GRID, INK, PATCH, SIDE, gridOf, inkedPatches } from './views/patch-grid/grid.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinyvit', SITE).then((r) => r.pkg);
const argmax = (a: ArrayLike<number>) => Array.from(a).reduce((b, v, i, x) => (v > x[b] ? i : b), 0);
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const row = (t: Tensor, r: number) => t.data.subarray(r * t.shape[1], (r + 1) * t.shape[1]);
const top = (tr: Trace, id = 'softmax') => argmax(outputProbs(tr.t[id]));
const step = (id: string) => (lesson.steps as StepDef<ExplainerRuntime>[]).find((s) => s.id === id)!;
const LINES = ['line', 'hline', 'vline', 'corner'];

test('page texts: the structural numbers in the goals, the pre-question, the quiz and the try-this match the spec', async () => {
	const pkg = await pkgP;
	const rt = await createRuntime(pkg, opts);
	const D = dims(rt);
	assert.deepEqual([D.side, D.k, D.grid, D.n, D.px, D.d, D.rows, D.heads, D.dh, D.blocks, D.classes, D.cnnK], [28, 7, 4, 16, 49, 48, 17, 4, 12, 4, 10, 3]);
	const p = rt.spec!.layers.find((l) => l.id === 'patch')!.params as Record<string, number>;
	assert.equal(p.stride, p.kernel, 'kernel = stride: the patches never overlap');
	assert.equal(p.padding, 0);
	assert.equal(D.side, GRID * PATCH);
	assert.equal(SIDE, D.side);
	const text = JSON.stringify(manifest.page);
	for (const s of ['28×28', '17 rows of 48', '16 patches', '√12', '17 scores', '3×3', '10 classes', '"16"', '"17"', '"784"', 'four patches']) assert.ok(text.includes(s), s);
	assert.equal(manifest.page.quiz![0].answer, 1, 'quiz 1: 17 rows');
	assert.equal(rt.num('fact:params'), rt.spec!.params);
	assert.equal(rt.num('fact:inkThreshold'), INK, 'the view and stats.py use the same "has ink" rule');
	assert.equal(rt.num('fact:testN'), 15000);
	rt.close();
});

test('review: try-this "One dot": a single dot is scaled up into the middle, where four patches meet, and inks those four', async () => {
	const pkg = await pkgP;
	const dot = manifest.page.tryThis!.find((t) => t.id === 'dot')!.setup as { input: unknown };
	const rt = await createRuntime(pkg, { ...opts, input: dot.input });
	const ink = inkedPatches(rt.sources.fwd.trace!.t.input.data);
	assert.deepEqual(ink.flatMap((v, k) => (v ? [k] : [])), [5, 6, 9, 10], 'the four patches around the centre');
	const g = gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { mode: 'attention', block: 1, head: rt.num('fact:head') } })!;
	assert.ok(g.inkShare / 4 > (1 - g.self - g.inkShare) / 12, 'the look: the inked patches out-weigh the empty ones');
	rt.close();
});

test('steps "tokens" and "qk" (predict): the [CLS] row and its block-1 query are the same for every drawing; its weights are not', async () => {
	const pkg = await pkgP;
	let cls: Float32Array | null = null;
	let q0: Float32Array | null = null;
	const weights: Record<string, Float32Array> = {};
	await withFixtures(
		pkg,
		(rt, fx) => {
			const tr = rt.sources.fwd.trace!;
			const h = rt.num('fact:head');
			const c = Float32Array.from(row(tr.t.cls, 0));
			const q = Float32Array.from(row(tr.t['B1.q'], 0));
			cls ??= c;
			q0 ??= q;
			assert.equal(maxDiff(c, cls), 0, '[CLS] row');
			assert.ok(maxDiff(q, q0) < 1e-6, 'block-1 [CLS] query');
			const a = tr.t['B1.attn'];
			weights[fx.id] = Float32Array.from(a.data.subarray(h * 17 * 17, h * 17 * 17 + 17));
		},
		opts
	);
	assert.ok(maxDiff(weights['preset:apple'], weights['preset:fish']) > 0.05, 'the weights follow the drawing');
	assert.ok(maxDiff(weights['preset:sun'], weights['preset:house']) > 0.05);
});

test('review: step "qk": dividing by √12 is multiplying by 0.289 (the stage shows × 0.289)', async () => {
	const pkg = await pkgP;
	const rt = await createRuntime(pkg, opts);
	const L = rt.spec!.layers.find((l) => l.id === 'B1.scores')!.params as { scale?: number; dHead: number };
	assert.ok(Math.abs((L.scale ?? 1 / Math.sqrt(L.dHead)) - 1 / Math.sqrt(12)) < 1e-7);
	assert.equal((1 / Math.sqrt(12)).toFixed(3), '0.289');
	rt.close();
});

test('review: step "row": the softmax worked out (e to the score, divided by the sum over all 17 rows) is the traced weight and the largest one, on every fixture', async () => {
	const pkg = await pkgP;
	await withFixtures(
		pkg,
		(rt) => {
			const h = rt.num('fact:head');
			const sm = softmaxWorked(rt, h)!;
			const a = rt.sources.fwd.trace!.t['B1.attn'];
			const traced = a.data[h * 17 * 17 + sm.top + 1];
			assert.ok(Math.abs(sm.w - traced) < 1e-6, `${sm.w} vs ${traced}`);
			assert.ok(Math.abs(sm.eTop - Math.exp(sm.sTop)) < 1e-9 * sm.eTop);
			const g = gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { mode: 'attention', block: 1, head: h } })!;
			assert.equal(sm.top, g.top, 'the narration and the dashed patch name the same patch');
			assert.ok(sm.w >= g.self, 'the largest weight (also above the [CLS] row itself)');
		},
		opts
	);
});

test('steps "qk" and "row": in the lesson head, a patch with ink gets more weight than an empty patch on every preset drawing and every line', async () => {
	const pkg = await pkgP;
	await withFixtures(
		pkg,
		(rt, fx) => {
			const g = gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { mode: 'attention', block: 1, head: rt.num('fact:head') } })!;
			const n = g.inked.filter(Boolean).length;
			assert.ok(rt.num('fact:headLean') > 1 && Number.isInteger(rt.num('fact:head')));
			if (n === 0 || n === 16) return;
			const perInk = g.inkShare / n;
			const perEmpty = (g.w.reduce((a, b) => a + b, 0) - g.inkShare) / (16 - n);
			if (fx.kind === 'preset' || LINES.includes(fx.id)) assert.ok(perInk > perEmpty, `${fx.id}: ${perInk} vs ${perEmpty}`);
		},
		opts
	);
});

test('review: step "row" task: a straight line inks only some of the patches (at most 10), they still get most of the attention, and the task is done; a preset is not', async () => {
	const pkg = await pkgP;
	const done = step('row').task!.done;
	const seen: string[] = [];
	await withFixtures(
		pkg,
		(rt, fx) => {
			const at = { index: 0 };
			if (fx.kind === 'preset') assert.equal(done(rt, at), false, `${fx.id}: a preset does not count`);
			if (!LINES.includes(fx.id)) return;
			seen.push(fx.id);
			const g = gridOf({ rt, source: 'fwd', at, props: { mode: 'attention', block: 1, head: rt.num('fact:head') } })!;
			const n = g.inked.filter(Boolean).length;
			assert.ok(n >= 1 && n <= 10 && n < 16, `${fx.id}: ${n} inked patches`);
			assert.ok(g.inkShare > 0.5, `${fx.id}: the inked patches get ${g.inkShare}`);
			assert.equal(done(rt, at), true, `${fx.id}: the task is done`);
		},
		opts
	);
	assert.deepEqual(seen.sort(), [...LINES].sort());
});

test('step "tokens": attention by itself cannot tell where a patch came from (without the position rows, shuffled patches give the same answer)', async () => {
	const pkg = await pkgP;
	const rt = await createRuntime(pkg, opts);
	const m = rt.model!;
	const x = rt.pre!.tensors.input as Tensor; // the first preset (apple)
	const shuffled = new Float32Array(x.data.length);
	const perm = Array.from({ length: GRID * GRID }, (_, k) => (k * 5 + 3) % 16); // patch k goes to place perm[k]
	for (let k = 0; k < 16; k++)
		for (let r = 0; r < PATCH; r++)
			for (let c = 0; c < PATCH; c++) {
				const src = (Math.floor(k / GRID) * PATCH + r) * SIDE + (k % GRID) * PATCH + c;
				const dst = (Math.floor(perm[k] / GRID) * PATCH + r) * SIDE + (perm[k] % GRID) * PATCH + c;
				shuffled[dst] = x.data[src];
			}
	const xs: Tensor = { shape: x.shape, data: shuffled };
	const pos = m.weights['pos.weight'];
	const noPos = fromParts(m.spec, { ...m.weights, 'pos.weight': { shape: pos.shape, data: new Float32Array(pos.data.length) } }, pkg.ops);
	assert.ok(maxDiff(noPos.forward(x).t.softmax.data, noPos.forward(xs).t.softmax.data) < 1e-5, 'no positions: same answer');
	assert.ok(maxDiff(m.forward(x).t.softmax.data, m.forward(xs).t.softmax.data) > 1e-3, 'with positions: the shuffle matters');
	rt.close();
});

test('review: step "embed": the 49 products plus the bias make the traced number, on every fixture', async () => {
	const pkg = await pkgP;
	const s = step('embed');
	await withFixtures(
		pkg,
		(rt) => {
			const sel = (s.select as (r: ExplainerRuntime) => { layer: string; unit: number[] })(rt);
			const view = Object.assign(Object.create(rt), { sel }) as ExplainerRuntime;
			const v = (s.narration as { vars: (r: ExplainerRuntime, at: unknown) => Record<string, number> }).vars(view, { index: 0 });
			assert.ok(Math.abs(v.dot + v.b - v.y) < 1e-5, `${v.dot} + ${v.b} vs ${v.y}`);
			assert.equal(v.p, v.pr * GRID + v.pc);
		},
		opts
	);
});

test('step "mlp": GELU lets large positive numbers through almost unchanged and shrinks negative ones toward 0; the MLP works row by row', async () => {
	const pkg = await pkgP;
	await withFixtures(
		pkg,
		(rt) => {
			const tr = rt.sources.fwd.trace!;
			const f = tr.t['B1.fc1'].data;
			const g = tr.t['B1.gelu'].data;
			for (let i = 0; i < f.length; i++) {
				if (f[i] < 0) assert.ok(g[i] <= 0 && Math.abs(g[i]) < Math.abs(f[i]));
				if (f[i] > 2) assert.ok(Math.abs(g[i] - f[i]) < 0.05 * f[i]);
			}
		},
		opts
	);
	const rt = await createRuntime(pkg, opts);
	assert.deepEqual(['fc1', 'gelu', 'fc2'].map((s) => rt.spec!.layers.find((l) => l.id === `B1.${s}`)!.op), ['linear', 'gelu', 'linear']);
	rt.close();
});

test('step "blocks": blocks 2 to 4 repeat block 1’s recipe with their own weights, and every row (each patch too) attends; step "classes": only the [CLS] row reaches the class head', async () => {
	const pkg = await pkgP;
	const rt = await createRuntime(pkg, opts);
	const L = rt.spec!.layers;
	const recipe = (b: number) => L.filter((l) => l.id.startsWith(`B${b}.`)).map((l) => `${l.id.slice(3)}:${l.op}`).join();
	for (const b of [2, 3, 4]) {
		assert.equal(recipe(b), recipe(1));
		assert.ok(maxDiff(rt.model!.weights[`B${b}.q.weight`].data, rt.model!.weights['B1.q.weight'].data) > 0.01);
	}
	for (let b = 1; b <= 4; b++) {
		const a = rt.sources.fwd.trace!.t[`B${b}.attn`];
		assert.deepEqual(a.shape, [4, 17, 17], 'a query for every row: each patch row attends too');
		for (let r = 0; r < 4 * 17; r++) assert.ok(Math.abs(Array.from(a.data.subarray(r * 17, r * 17 + 17)).reduce((x, y) => x + y, 0) - 1) < 1e-5);
	}
	const sel = L.find((l) => l.id === 'clsout')!;
	assert.deepEqual([sel.op, sel.in[0], (sel.params as { index: number }).index], ['select_token', 'ln', 0]);
	assert.deepEqual(L.find((l) => l.id === 'head')!.in, ['clsout']);
	rt.close();
});

test('review: steps "classes" and "compare": label smoothing 0.1 over 10 classes targets 91% for the right class and 1% for each other; clear drawings land near that; TinyCNN was trained without it, with another recipe, and can reach certainty', async () => {
	const pkg = await pkgP;
	const rt = await createRuntime(pkg, opts);
	const tr = rt.spec!.training as { optimizer?: string; epochs?: number };
	assert.ok(String(tr.optimizer).includes('label smoothing 0.1'));
	const eps = 0.1;
	const K = dims(rt).classes;
	assert.ok(Math.abs(rt.num('fact:smoothTarget') - (1 - eps + eps / K)) < 1e-12, 'PyTorch label_smoothing: (1 − ε) + ε/K');
	assert.ok(Math.abs((1 - rt.num('fact:smoothTarget')) / (K - 1) - eps / K) < 1e-12, 'each other class: ε/K = 1%');
	assert.ok(rt.num('fact:maxTopP') < 1 && rt.num('fact:maxTopP') > rt.num('fact:smoothTarget'));
	const cnnTr = rt.sources.cnn.model!.spec.training as { optimizer?: string; epochs?: number };
	assert.ok(!/smoothing/i.test(String(cnnTr.optimizer)), 'TinyCNN: no label smoothing');
	assert.notEqual(cnnTr.optimizer, tr.optimizer, 'two different training recipes');
	assert.notEqual(cnnTr.epochs, tr.epochs);
	rt.close();
	let cnnCertain = 0;
	let clear = 0;
	await withFixtures(
		pkg,
		(r, fx) => {
			const p = outputProbs(r.sources.fwd.trace!.t.softmax);
			const i = argmax(p);
			assert.ok(p[i] <= 0.95, `${fx.id}: TinyViT's top bar stays near the target (${p[i]})`);
			if (fx.kind === 'preset' && p[i] > 0.85) {
				clear++;
				assert.ok(Math.abs(p[i] - 0.91) < 0.03, `${fx.id}: ${p[i]} near 91%`);
				for (let j = 0; j < p.length; j++) if (j !== i) assert.ok(p[j] < 0.03, `${fx.id}: class ${j} at ${p[j]}, near 1%`);
			}
			const c = r.sources.cnn;
			if (Math.max(...outputProbs(c.trace!.t[c.model!.spec.outputs?.[0] ?? 'softmax'])) > 0.999) cnnCertain++;
		},
		opts
	);
	assert.ok(clear >= 5, `clear drawings: ${clear}`);
	assert.ok(cnnCertain >= 1, 'TinyCNN reaches certainty on some drawing');
});

test('step "compare", pre-question and misconception: the [CLS] row gives every row some weight in every block (it can read any patch), TinyCNN looks at 3×3 neighbours, and the ladder and envelope presets make the two models disagree', async () => {
	const pkg = await pkgP;
	const disagree: Record<string, boolean> = {};
	await withFixtures(
		pkg,
		(rt, fx) => {
			const tr = rt.sources.fwd.trace!;
			for (let b = 1; b <= 4; b++) {
				const a = tr.t[`B${b}.attn`];
				for (let h = 0; h < 4; h++) for (let j = 0; j < 17; j++) assert.ok(a.data[h * 289 + j] > 0, `${fx.id} B${b} h${h} key ${j}`);
			}
			const p = outputProbs(tr.t.softmax);
			assert.ok(Math.abs(p.reduce((s, v) => s + v, 0) - 1) < 1e-5, 'the shares add up to 1 over the 10 classes');
			const cnn = rt.sources.cnn;
			disagree[fx.id] = top(tr) !== top(cnn.trace!, cnn.model!.spec.outputs?.[0] ?? 'softmax');
		},
		opts
	);
	assert.ok(disagree['preset:heldout-ladder'] && disagree['preset:heldout-envelope'], 'the hint of the compare task');
	const rt = await createRuntime(pkg, opts);
	const conv = rt.sources.cnn.model!.spec.layers.filter((l) => l.op === 'conv2d');
	assert.ok(conv.length > 0 && conv.every((l) => (l.params as { kernel: number }).kernel === 3));
	assert.ok(inkedPatches(new Float32Array(SIDE * SIDE)).every((v) => !v));
	rt.close();
});
