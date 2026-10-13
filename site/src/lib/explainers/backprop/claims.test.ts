// backprop claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the starting leaves and
// the adapter's edge fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/backprop/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { backward, grad, input, param } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { EDGES, LEAVES, NET_P0, NET_PARAMS, NET_X, NODES, SAT, START, TASK_G, TASK_L, leavesOf, nodeGrad, nodeVal } from './data.ts';
import { graph, leafGrads, lossOf, net } from './process.ts';
import { STARTS } from './adapters/leaves/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('backprop', SITE)).pkg;
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
type T = { length: number; frame(i: number): Frame; state(i: number): unknown; find(c: unknown): number };
const at = (t: T, c: { phase: string; step?: number }) => {
	const i = t.find(c);
	return { m: t.frame(i).metrics ?? {}, s: t.state(i) };
};

test('step "forward": every node is the formula L = (tanh(w·x + b) − y)² computed step by step, on every fixture', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('graph', 'last');
			const { s } = at(rt.sources.graph.tape! as T, { phase: 'forward' });
			const l = leavesOf(rt.input);
			const a = Math.tanh(l.w * l.x + l.b);
			assert.ok(close(nodeVal(s, 'L'), (a - l.y) ** 2), `L ${nodeVal(s, 'L')}`);
			for (const n of NODES) assert.ok(Number.isNaN(nodeGrad(s, n.id)), `${n.id} already has a gradient in the forward frame`);
		},
		opts
	);
});

test('step "local": the engine\'s local derivatives are e, e, 1, −1, 1 − a², 1, 1, x and w (so ∂p/∂w = x)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('graph', 'last');
			const { s } = at(rt.sources.graph.tape! as T, { phase: 'forward' });
			const v = (id: string) => nodeVal(s, id);
			const want = [v('e'), v('e'), 1, -1, 1 - v('a') ** 2, 1, 1, v('x'), v('w')];
			const got = (s as { locals: { data: Float64Array } }).locals.data;
			want.forEach((w, k) => assert.ok(close(got[k], w), `edge ${k}: ${got[k]} vs ${w}`));
		},
		opts
	);
});

test('steps "predict", "twice": e receives one share of e along each edge, the second is added (grad +=), and dL/de = 2e; one share alone is half', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('graph', 'last');
			const t = rt.sources.graph.tape! as T;
			const e0 = at(t, { phase: 'edge', step: 0 });
			const e1 = at(t, { phase: 'edge', step: 1 });
			const e = nodeVal(e1.s, 'e');
			assert.equal(e0.m.acc, 0);
			assert.equal(e1.m.acc, 1);
			assert.ok(close(e0.m.share, e) && close(e1.m.share, e) && close(e1.m.before, e));
			assert.ok(close(e1.m.after, 2 * e), `dL/de ${e1.m.after} vs 2e ${2 * e}`);
			// only edge 1 adds to a gradient that was already there
			for (let k = 0; k < EDGES.length; k++) assert.equal(at(t, { phase: 'edge', step: k }).m.acc, k === 1 ? 1 : 0);
		},
		opts
	);
});

test('step "chain": each edge multiplies; through tanh the gradient is multiplied by 1 − a²; dL/dw = dL/dp × x; the walk equals backward()', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('graph', 'last');
			const t = rt.sources.graph.tape! as T;
			for (let k = 0; k < EDGES.length; k++) {
				const { m } = at(t, { phase: 'edge', step: k });
				assert.ok(close(m.share, m.up * m.local), `edge ${k}`);
			}
			const { s } = at(t, { phase: 'edge', step: EDGES.length - 1 });
			const g = (id: string) => nodeGrad(s, id);
			const a = nodeVal(s, 'a');
			assert.ok(close(g('z'), g('a') * (1 - a * a)));
			assert.ok(Math.abs(g('z')) <= Math.abs(g('a')) + 1e-15, 'tanh never makes the gradient larger');
			assert.ok(close(g('w'), g('p') * nodeVal(s, 'x')));
			const ref = leafGrads(leavesOf(rt.input));
			for (const l of LEAVES) assert.ok(close(g(l.id), ref[l.id]), `${l.id}: walk ${g(l.id)} vs backward ${ref[l.id]}`);
		},
		opts
	);
});

test('step "task": no starting setting already meets the goal, and it can be met (an input of 0, or a saturated tanh)', () => {
	const met = (l: typeof START) => lossOf(l) > TASK_L && Math.abs(leafGrads(l).w) < TASK_G;
	for (const s of STARTS) assert.equal(met(s.raw), false, s.id);
	assert.ok(met(SAT), 'the saturated leaves');
	assert.ok(met({ x: 0, w: 1, b: 0.3, y: -0.5 }), 'x = 0');
});

test('step "check": the numerical estimate agrees with backpropagation on every leaf, on every fixture', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('graph', 'last');
			const { m, s } = at(rt.sources.graph.tape! as T, { phase: 'check' });
			assert.ok(m.maxDiff < 1e-6, `largest difference ${m.maxDiff}`);
			for (const l of LEAVES) assert.ok(Math.abs(m[`n${l.id}`] - nodeGrad(s, l.id)) <= m.maxDiff + 1e-18);
		},
		opts
	);
});

test('steps "net", "cost", concept "real": nine parameters; the layer rules hold; the check needs 18 passes and agrees; one update lowers L', () => {
	assert.equal(NET_PARAMS.length, 9);
	assert.equal(NET_P0.length, 9);
	const t = record(net, null, { h: 1e-5, lr: 0.1 }) as unknown as T;
	const first = at(t, { phase: 'first' });
	const s = first.s as { pgrads: { data: Float64Array }; params: { data: Float64Array } };
	const h1 = nodeVal(first.s, 4);
	const v1 = NET_P0[6];
	assert.ok(close(first.m.gz1, first.m.gy * v1 * (1 - h1 * h1)), 'dL/dz₁ = dL/dŷ × v₁ × (1 − h₁²)');
	assert.ok(close(s.pgrads.data[0], first.m.gz1 * NET_X[0]) && close(s.pgrads.data[1], first.m.gz1 * NET_X[1]), 'dL/dW₁ⱼ = dL/dz₁ × xⱼ');
	// x₁ feeds both hidden units: its gradient is the sum of two shares
	assert.ok(close(nodeGrad(first.s, 0), first.m.gz1 * NET_P0[0] + first.m.gz2 * NET_P0[2]));
	const chk = at(t, { phase: 'check' }).m;
	assert.equal(chk.passes, 18);
	assert.ok(chk.maxDiff < 1e-6);
	const up = at(t, { phase: 'update' }).m;
	assert.ok(up.L1 < up.L, `L ${up.L} -> ${up.L1}`);
});

test('step "limits", compare, pitfall 3: a saturated tanh passes back almost nothing, and a chain of saturated layers passes back less with every layer', () => {
	const t = record(graph, SAT, { h: 1e-5 }) as unknown as T;
	const { s } = at(t, { phase: 'edge', step: EDGES.length - 1 });
	const a = nodeVal(s, 'a');
	assert.ok(1 - a * a < 1e-3 && nodeVal(s, 'L') > 1 && Math.abs(nodeGrad(s, 'w')) < 1e-3);
	// a deeper network: K saturated tanh layers after the first; the first weight's gradient shrinks with every layer
	let prev = Infinity;
	for (let K = 1; K <= 4; K++) {
		const w = param('w', { shape: [1], data: [SAT.w] });
		let h = op.tanh(op.add(op.mul(w, SAT.x), SAT.b));
		for (let k = 1; k < K; k++) h = op.tanh(op.add(op.mul(h, 2.5), 1));
		const e = op.sub(h, SAT.y);
		const g = Math.abs(grad(op.mul(e, e), w).item());
		assert.ok(g < prev / 10, `${K} layers: ${g} vs ${prev}`);
		prev = g;
	}
});

test('pitfall 1: the kit\'s autograd adds each backward pass into a stored gradient (so training loops clear it first)', () => {
	const once = leafGrads(START).w;
	const w = input({ shape: [1], data: [START.w] }, true);
	const loss = () => {
		const e = op.sub(op.tanh(op.add(op.mul(w, START.x), START.b)), START.y);
		return op.mul(e, e);
	};
	backward(loss());
	assert.ok(close(w.grad!.item(), once));
	backward(loss());
	assert.ok(close(w.grad!.item(), 2 * once), `${w.grad!.item()} vs ${2 * once}`);
});

test('quiz: 1 − 0.99² is about 2%', () => {
	assert.ok(Math.abs(1 - 0.99 * 0.99 - 0.0199) < 1e-12);
});

test('concept "real": the MLP playground trains a bigger network of the same tanh units', async () => {
	const { fit } = await import('../mlp-playground/trainer.ts');
	assert.equal(fit.defaults.act, 'tanh');
	assert.ok(fit.defaults.hidden.reduce((a: number, b: number) => a + b, 0) > 2, `hidden ${fit.defaults.hidden}`);
});
