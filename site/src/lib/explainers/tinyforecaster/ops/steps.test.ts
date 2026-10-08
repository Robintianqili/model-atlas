// Tests for the tinyforecaster local ops and the live-model graph (node --test; `atlas check` runs them, op-tests gate).
// The graph check is the gate parity of design §8.3 M8: the kit's shared `lstm` op, run by the executor on the live
// spec (model.ts toSpec), gives the same gate sub-steps i f g o c h, the same h and the same forecast as the autograd
// network the trainer uses (whose training matches PyTorch float64 to 3.89e-16: `atlas parity`, twin.py).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, AtlasSpec, OpCtx, OpDef, Tensor, Trace } from '../../../atlas/types/index.ts';
import { fromParts } from '../../../atlas/graph/executor.ts';
import { createRegistry } from '../../../atlas/ops/registry.ts';
import { createRng } from '../../../atlas/core/rng.ts';
import { input, noGrad } from '../../../atlas/autograd/var.ts';
import { lstm } from '../trainer.ts';
import { Forecaster } from '../model.ts';
import { L } from '../data.ts';
import ops from '../ops.ts';
import { last_step, relative } from './steps.ts';

const T = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });
const layer = (op: string, outShape: number[]): AtlasLayer => ({ id: 'L', op, label: { en: 'L', zh: 'L' }, in: ['x'], outShape });
const ctxOf = (l: AtlasLayer): OpCtx => ({ layer: l, dims: {}, w: () => T([0], () => 0), has: () => false, emit: () => {} });

function explainAll(def: OpDef<any>, l: AtlasLayer, x: Tensor) {
	const out = def.forward([x], {} as never, ctxOf(l));
	const trace: Trace = { order: ['x', 'L'], t: { x, L: out } };
	for (let f = 0; f < out.data.length; f++) {
		const unit = out.shape.length === 2 ? [Math.floor(f / out.shape[1]), f % out.shape[1]] : [f];
		const e = def.explain!({ layer: l, unit, ins: [x], out, w: () => x, has: () => false, p: {} as never, trace, spec: {} as AtlasSpec });
		assert.equal(e.result, e.traced, `${def.op} unit [${unit}]`);
		assert.equal(e.result, out.data[f]);
	}
	return out;
}

test('relative: every row minus the last row; the last row becomes 0', () => {
	const x = T([4, 2], (i) => [1, 10, 2, 20, 4, 40, 3, 30][i]);
	const y = explainAll(relative, layer('relative', [4, 2]), x);
	assert.deepEqual(Array.from(y.data), [-2, -20, -1, -10, 1, 10, 0, 0]);
	assert.deepEqual(relative.inferShape([[24, 1]], {}, {}), [24, 1]);
	assert.ok(y.data instanceof Float64Array, 'f64 in, f64 out');
});

test('last_step: the last row only', () => {
	const x = T([3, 2], (i) => i * 10);
	assert.deepEqual(Array.from(explainAll(last_step, layer('last_step', [2]), x).data), [40, 50]);
	assert.deepEqual(last_step.inferShape([[24, 16]], {}, {}), [16]);
	assert.throws(() => last_step.forward([T([0, 2], () => 0)], {} as never, ctxOf(layer('last_step', [2]))));
});

test('live graph = autograd network: gate sub-steps, h and the forecast (≤ 1e-12), on the start weights and on weights scaled ×3', () => {
	const reg = createRegistry(...ops);
	for (const seed of [1, 7]) {
		const m = new Forecaster(createRng(seed).fork('init'));
		// a second set of weights far from the start: scale them up, so the gates are not all near 0.5
		if (seed === 7) for (const v of m.params()) for (let i = 0; i < v.data.length; i++) v.data[i] *= 3;
		const { spec, weights } = m.toSpec({ slug: 'tinyforecaster' });
		const w: Record<string, Tensor> = {};
		for (const t of spec.tensors) w[t.name] = { shape: t.shape.slice(), data: (weights as unknown as Float64Array).slice(t.offset, t.offset + t.length) };
		const g = fromParts(spec, w, reg);
		const win = Float64Array.from({ length: L }, (_, i) => Math.sin(i / 3) + 0.05 * i);
		const trace = g.forward({ window: { shape: [L, 1], data: win } });
		const rec: Record<string, Tensor> = {};
		const y = noGrad(() => m.forward(input({ shape: [1, L], data: win }), (id, v) => (rec[id] = v.value)));
		const close = (a: number, b: number, what: string) => assert.ok(Math.abs(a - b) <= 1e-12, `${what}: graph ${a}, autograd ${b}`);
		close(trace.t.out.data[0], y.data[0], `seed ${seed} forecast`);
		close(trace.t.head.data[0], rec.head.data[0], `seed ${seed} predicted change`);
		for (const sub of ['i', 'f', 'g', 'o', 'c', 'h']) {
			const s = trace.t[`lstm#${sub}`];
			assert.deepEqual(s.shape, [L, 16]);
			for (let j = 0; j < 16; j++) close(s.data[(L - 1) * 16 + j], rec[`lstm#${sub}`].data[j], `seed ${seed} lstm#${sub} unit ${j}`);
		}
		// every unit's explanation (the zoom.gate numbers) equals the trace
		for (let t = 0; t < L; t += 7)
			for (let j = 0; j < 16; j += 5) {
				const e = g.explain(trace, 'lstm', [t, j]);
				close(e.result, e.traced, `explain lstm [${t}, ${j}]`);
			}
		assert.equal(trace.t.rel.data[L - 1], 0, 'the last value minus itself');
	}
	assert.equal(lstm.model(createRng(1).fork('init'), lstm.defaults).params().reduce((s, p) => s + p.size, 0), 1233, '1,233 parameters');
});
