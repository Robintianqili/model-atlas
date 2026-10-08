// W1-B1: modules (state_dict names, init ranges), the engine's edge cases, optimizers' exposed state, and toSpec.
// Run from site/:  node --test src/lib/atlas/autograd/nn.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, Module } from '../types/index.ts';
import { Var, backward, grad, input, noGrad, param, zeroGrad } from './var.ts';
import * as op from './ops.ts';
import { Conv1d, Embedding, LSTMCell, LayerNorm, Linear, MLP, lstmCell, loadParams } from './nn.ts';
import { Adam, Momentum, SGD, makeOptimizer } from './optim.ts';
import { stackToSpec } from './tospec.ts';
import { fromParts } from '../graph/executor.ts';
import { CORE_OPS } from '../ops/index.ts';
import { makeRng } from '../../../../tests/atlas/autograd/rng-stub.ts';
import { relErr } from '../../../../tests/atlas/autograd/support.ts';

test('MLP uses nn.Sequential state_dict names and PyTorch init ranges', () => {
	const m = new MLP('net', [2, 8, 8, 2], 'tanh', makeRng(1));
	assert.deepEqual(Object.keys(m.named()), ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias']);
	assert.deepEqual(m.named()['net.2.weight'].shape, [8, 8]);
	for (const [k, v] of Object.entries(m.named())) {
		const bound = 1 / Math.sqrt(k.startsWith('net.0') ? 2 : 8);
		assert.ok(v.data.every((x) => Math.abs(x) <= bound), `${k} within ±1/√fan_in`);
	}
	const bare = new MLP('', [3, 4, 1], 'silu', makeRng(1));
	assert.deepEqual(Object.keys(bare.named()), ['0.weight', '0.bias', '2.weight', '2.bias']);
	const seen: string[] = [];
	m.forward(input({ shape: [5, 2], data: new Float64Array(10) }), (id) => seen.push(id));
	assert.deepEqual(seen, ['net.0', 'net.1', 'net.2', 'net.3', 'net.4']);
});

test('other modules: names, shapes and PyTorch layouts', () => {
	const rng = makeRng(2);
	const cell = new LSTMCell('lstm', 3, 4, rng);
	assert.deepEqual(Object.keys(cell.named()), ['lstm.weight_ih', 'lstm.weight_hh', 'lstm.bias_ih', 'lstm.bias_hh']);
	assert.deepEqual(cell.weight_ih.shape, [16, 3]);
	const emb = new Embedding('tok', 10, 5, rng);
	assert.deepEqual(emb.forward([3, 3, 9]).shape, [3, 5]);
	const conv = new Conv1d('tcn.0', 2, 8, 2, rng, { dilation: 4, causal: true });
	assert.deepEqual(Object.keys(conv.named()), ['tcn.0.weight', 'tcn.0.bias']);
	assert.deepEqual(conv.forward(input({ shape: [3, 2, 24], data: new Float64Array(144) })).shape, [3, 8, 24], 'causal keeps the length');
	const ln = new LayerNorm('ln', 6);
	assert.deepEqual(Array.from(ln.weight.data), [1, 1, 1, 1, 1, 1]);
	const lin = new Linear('head', 4, 3, rng, { bias: false });
	assert.deepEqual(Object.keys(lin.named()), ['head.weight']);
});

test('LSTMCell module = lstmCell with its weights; run() unrolls and records gate sub-steps', () => {
	const rng = makeRng(3);
	const cell = new LSTMCell('lstm1', 3, 4, rng);
	const xs = input({ shape: [5, 3], data: Float64Array.from({ length: 15 }, (_, i) => Math.sin(i)) });
	const { hs, h } = cell.run(xs);
	assert.deepEqual(hs.shape, [5, 4]);
	let s = null as ReturnType<typeof lstmCell> | null;
	for (let t = 0; t < 5; t++) s = lstmCell(op.reshape(op.slice(xs, 0, t, t + 1), [3]), s, cell);
	assert.equal(relErr(h.data, s!.h.data), 0);
	const rec: string[] = [];
	cell.forward(op.reshape(op.slice(xs, 0, 0, 1), [3]), null, (id) => rec.push(id));
	assert.deepEqual(rec, ['lstm1#i', 'lstm1#f', 'lstm1#g', 'lstm1#o', 'lstm1#c', 'lstm1#h']);
	// batch-first [N, T, In]
	const xb = input({ shape: [2, 5, 3], data: Float64Array.from({ length: 30 }, (_, i) => Math.cos(i)) });
	assert.deepEqual(cell.run(xb).hs.shape, [2, 5, 4]);
	// a 200-step unroll differentiates without recursion limits
	const long = input({ shape: [200, 3], data: Float64Array.from({ length: 600 }, (_, i) => Math.sin(i / 7)) });
	const g = grad(op.sum(cell.run(long).h), cell.weight_hh);
	assert.ok(g.data.every(Number.isFinite));
});

test('engine: grad vs backward, accumulation, allowUnused, targets that are intermediates, noGrad', () => {
	const w = param('w', { shape: [3], data: [0.5, -1, 2] });
	const x = input({ shape: [3], data: [1, 2, 3] });
	const h = op.mul(w, x);
	const y = op.sum(op.mul(h, h));
	const [gw, gh] = grad(y, [w, h]);
	assert.deepEqual(Array.from(gh.data), [1, -4, 12]);
	assert.deepEqual(Array.from(gw.data), [1, -8, 36]);
	assert.equal(w.grad, null, 'grad() does not touch .grad');
	backward(y);
	backward(op.sum(op.mul(w, x)));
	assert.deepEqual(Array.from(w.grad!.data), [2, -6, 39], '.grad accumulates over backward calls');
	zeroGrad([w]);
	assert.equal(w.grad, null);
	const unused = param('u', { shape: [2], data: [1, 1] });
	assert.throws(() => grad(y, [w, unused]), /not used/);
	assert.deepEqual(Array.from(grad(y, [w, unused], { allowUnused: true })[1].data), [0, 0]);
	const z = noGrad(() => op.mul(w, w));
	assert.equal(z.requiresGrad, false);
	assert.equal(z.fn, null);
	// the same input used twice: d/dx (x·x) = 2x
	assert.deepEqual(Array.from(grad(op.sum(op.mul(w, w)), w).data), [1, -2, 4]);
	// createGraph keeps the graph: d²/dw² Σ w³ = 6w
	const g1 = grad(op.sum(op.pow(w, 3)), w, { createGraph: true });
	assert.ok(g1.requiresGrad);
	assert.deepEqual(Array.from(grad(op.sum(g1), w).data), [3, -6, 12]);
});

test('optimizers expose their moments and follow set()', () => {
	const mk = () => param('p', { shape: [2], data: [1, -1] });
	const p = mk();
	const adam = new Adam([p], { lr: 0.1 });
	p.grad = input({ shape: [2], data: [0.5, -0.25] });
	adam.step();
	const st = adam.state();
	assert.deepEqual(Object.keys(st), ['p']);
	assert.ok(st.p.m && st.p.v);
	assert.ok(Math.abs(p.data[0] - 0.9) < 1e-6, 'first Adam step moves by about lr');
	adam.set({ lr: 0.01 });
	assert.equal(adam.lr, 0.01);
	const q = mk();
	const mom = new Momentum([q], { lr: 0.1, momentum: 0.9 });
	q.grad = input({ shape: [2], data: [1, 1] });
	mom.step();
	mom.step();
	assert.deepEqual(Array.from(q.data).map((v) => +v.toFixed(12)), [0.71, -1.29]);
	assert.deepEqual(Object.keys(mom.state()), ['p']);
	const r = mk();
	const sgd = makeOptimizer('sgd', [r], { lr: 0.5 }) as SGD;
	sgd.step(); // no grad: skipped, as in PyTorch
	assert.deepEqual(Array.from(r.data), [1, -1]);
	assert.equal(makeOptimizer('adamw', [mk()], { lr: 1e-3 }).kind, 'adamw');
});

// A tiny reference executor for the spec toSpec writes (W1-A's graph/executor.ts runs the same spec at integration;
// design §8.1 integration check 3). f32 weights, f64 sums, in layer order.
function runSpec(spec: ReturnType<typeof stackToSpec>['spec'], weights: Float32Array, inputs: Record<string, number[]>): Float64Array {
	const vals: Record<string, Float64Array> = {};
	for (const [k, v] of Object.entries(inputs)) vals[k] = Float64Array.from(v);
	const tensor = (name: string) => {
		const e = spec.tensors.find((t) => t.name === name)!;
		return weights.subarray(e.offset, e.offset + e.length);
	};
	const slot = (l: AtlasLayer, s: string) => l.w?.[s] ?? `${l.id}.${s}`;
	const acts: Record<string, (v: number) => number> = { tanh: Math.tanh, relu: (v) => Math.max(0, v), sigmoid: (v) => 1 / (1 + Math.exp(-v)), silu: (v) => v / (1 + Math.exp(-v)) };
	for (const l of spec.layers) {
		const ins = l.in.map((n) => vals[n]);
		if (l.op === 'concat') vals[l.id] = Float64Array.from(ins.flatMap((a) => Array.from(a)));
		else if (l.op === 'linear') {
			const W = tensor(slot(l, 'weight'));
			const hasB = spec.tensors.some((t) => t.name === slot(l, 'bias'));
			const b = hasB ? tensor(slot(l, 'bias')) : null;
			const out = l.outShape[0] as number;
			const x = ins[0];
			const y = new Float64Array(out);
			for (let o = 0; o < out; o++) {
				let s = 0;
				for (let i = 0; i < x.length; i++) s += x[i] * W[o * x.length + i];
				y[o] = b ? s + b[o] : s;
			}
			vals[l.id] = y;
		} else vals[l.id] = ins[0].map(acts[l.op]);
	}
	return vals[spec.outputs![0]];
}

test('toSpec: an MLP becomes a spec-1.1 linear/activation stack whose forward equals the module (≤ 1e-6 relative)', () => {
	const m = new MLP('net', [2, 8, 8, 2], 'tanh', makeRng(5));
	const { spec, weights } = m.toSpec({ slug: 'mlp-playground', inputAxes: ['xy'], outputAxes: ['class'], adapter: 'points2d', inputKind: 'points2d' });
	assert.deepEqual(
		spec.layers.map((l) => [l.id, l.op, l.in.join('+'), l.outShape.join('x')]),
		[
			['net.0', 'linear', 'input', '8'],
			['net.1', 'tanh', 'net.0', '8'],
			['net.2', 'linear', 'net.1', '8'],
			['net.3', 'tanh', 'net.2', '8'],
			['net.4', 'linear', 'net.3', '2']
		]
	);
	assert.equal(spec.params, 2 * 8 + 8 + 8 * 8 + 8 + 8 * 2 + 2);
	assert.equal(weights.length, spec.params);
	assert.deepEqual(spec.inputs, [{ name: 'input', adapter: 'points2d', kind: 'points2d', shape: [2], dtype: 'f32', axes: ['xy'] }]);
	assert.deepEqual(spec.layers[4].axes, ['class']);
	assert.deepEqual(spec.parity, { against: 'atlas-autograd-f64', max_abs_diff: null, checked: null, scope: 'forward' });
	assert.equal(spec.atlas?.spec, '1.1');
	for (const l of spec.layers) assert.ok(!l.id.includes('#'));
	for (const pt of [[0.3, -0.7], [1.2, 0.4], [-1, -1]]) {
		const y = runSpec(spec, weights, { input: pt });
		const ref = noGrad(() => m.forward(input({ shape: [2], data: pt })));
		const e = relErr(y, ref.data);
		assert.ok(e <= 1e-6, `point ${pt}: rel err ${e}`);
	}
});

test('toSpec: flow-2d style inputs (xy, t) are joined by a concat layer; SiLU', () => {
	const m = new MLP('v', [3, 16, 16, 2], 'silu', makeRng(6));
	const { spec, weights } = m.toSpec({ slug: 'flow-2d', inputs: [{ name: 'xy', size: 2, axes: ['xy'] }, { name: 't', size: 1 }] });
	assert.equal(spec.layers[0].op, 'concat');
	assert.deepEqual(spec.layers[0].in, ['xy', 't']);
	assert.equal(spec.layers[1].in[0], 'concat');
	assert.equal(spec.input, undefined, 'no single v1 input for two inputs');
	const y = runSpec(spec, weights, { xy: [0.2, -0.4], t: [0.7] });
	const ref = noGrad(() => m.forward(input({ shape: [3], data: [0.2, -0.4, 0.7] })));
	assert.ok(relErr(y, ref.data) <= 1e-6);
	const lin = new Linear('head', 4, 3, makeRng(7), { bias: false });
	const s2 = lin.toSpec({ slug: 'head' });
	assert.deepEqual(s2.spec.tensors.map((t) => t.name), ['head.weight']);
	const asModule: Module = lin;
	assert.ok(asModule.toSpec);
	assert.throws(() => m.toSpec({ slug: 'x', inputs: [{ name: 'a', size: 2 }] }), /add up to 2 features/);
});

test('MLP outAct: an output activation after the last layer (sigmoid, softmax); logits() stops before it; toSpec traces it', () => {
	const m = new MLP('net', [2, 2, 1], 'tanh', makeRng(9), { outAct: 'sigmoid' });
	assert.deepEqual(Object.keys(m.named()), ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'], 'no parameters added');
	const x = input({ shape: [4, 2], data: [0, 0, 0, 1, 1, 0, 1, 1] });
	const seen: string[] = [];
	const y = noGrad(() => m.forward(x, (id) => seen.push(id)));
	assert.deepEqual(seen, ['net.0', 'net.1', 'net.2', 'net.3'], 'Sequential(Linear, Tanh, Linear, Sigmoid) names');
	const z = noGrad(() => m.logits(x));
	for (let i = 0; i < 4; i++) assert.ok(Math.abs(y.data[i] - 1 / (1 + Math.exp(-z.data[i]))) < 1e-15);
	const { spec, weights } = m.toSpec({ slug: 'xor-sigmoid' });
	assert.deepEqual(spec.layers.map((l) => [l.id, l.op]), [['net.0', 'linear'], ['net.1', 'tanh'], ['net.2', 'linear'], ['net.3', 'sigmoid']]);
	assert.deepEqual(spec.outputs, ['net.3']);
	const W: Record<string, { shape: number[]; data: Float32Array }> = {};
	for (const t of spec.tensors) W[t.name] = { shape: t.shape, data: weights.slice(t.offset, t.offset + t.length) };
	const g = fromParts(spec, W, CORE_OPS);
	for (const pt of [[1, 0], [0.3, 0.9]]) {
		const tr = g.forward({ input: { shape: [2], data: Float32Array.from(pt) } });
		const ref = noGrad(() => m.forward(input({ shape: [2], data: pt })));
		assert.ok(Math.abs(tr.t['net.3'].data[0] - ref.data[0]) < 1e-6, `executor = module at ${pt}`);
	}
	const sm = new MLP('net', [2, 4, 3], 'relu', makeRng(10), { outAct: 'softmax' });
	const p = noGrad(() => sm.forward(input({ shape: [2], data: [0.5, -0.2] })));
	assert.ok(Math.abs(p.data[0] + p.data[1] + p.data[2] - 1) < 1e-12, 'softmax output sums to 1');
	const s2 = sm.toSpec({ slug: 'sm', outputAxes: ['class'] });
	const last = s2.spec.layers[s2.spec.layers.length - 1];
	assert.deepEqual([last.id, last.op, last.params, last.axes], ['net.3', 'softmax', { axis: -1 }, ['class']]);
	assert.deepEqual(s2.spec.layers[s2.spec.layers.length - 2].axes, ['f']);
});

test('loadParams checks names and sizes', () => {
	const m = new MLP('net', [2, 3, 1], 'relu', makeRng(8));
	assert.throws(() => loadParams(m, { 'net.9.weight': { data: [1] } }), /no parameter/);
	assert.throws(() => loadParams(m, { 'net.0.weight': { data: [1] } }), /has 1 values/);
	loadParams(m, { 'net.2.bias': { data: [0.25] } });
	assert.equal(m.named()['net.2.bias'].data[0], 0.25);
	assert.ok(m.params().every((p: Var) => p.requiresGrad));
});
