// Tests of the tinymoe local op 'moe' (node --test): top-2 routing and the gate-weighted sum on a small input against hand
// arithmetic, experts not chosen are never run, and explain() equals the forward pass (what `atlas parity` checks on the
// real spec as AK202).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, OpCtx, Tensor, Trace } from '../../../atlas/types/index.ts';
import { moe, topK } from './moe.ts';

const f32 = (shape: number[], a: number[]): Tensor => ({ shape, data: Float32Array.from(a) });
const layer = { id: 'B1.moe', op: 'moe', in: ['B1.ln2', 'B1.route'], outShape: [2, 2], params: { k: 2 } } as unknown as AtlasLayer;
// 3 experts, 1 hidden unit each, D = 2: expert j: h = relu(x0 + x1 − j), out = [h, −h] + [j, 0]
const W: Record<string, Tensor> = {
	fc: f32([3, 1, 2], [1, 1, 1, 1, 1, 1]),
	fc_b: f32([3, 1], [0, -1, -2]),
	out: f32([3, 2, 1], [1, -1, 1, -1, 1, -1]),
	out_b: f32([3, 2], [0, 0, 1, 0, 2, 0])
};

test('moe: the top 2 chances rescaled to add up to 1; only the chosen experts run; y = Σ gate · expert(x)', () => {
	const x = f32([2, 2], [1, 2, 0.5, 0.5]);
	const p = f32([2, 3], [0.5, 0.3, 0.2, 0.1, 0.1, 0.8]);
	const subs: Record<string, Tensor> = {};
	const ctx: OpCtx = { layer, dims: {}, w: (s) => W[s], emit: (n, t) => (subs[n] = t) };
	const y = moe.forward([x, p], { k: 2 }, ctx);
	// token 0: experts 0 (0.5) and 1 (0.3): gates 0.625, 0.375; h0 = 3 -> [3, −3], h1 = 2 -> [3, −2]
	// token 1: experts 2 (0.8) and 0 (0.1, the lower of the tie): gates 8/9, 1/9; h2 = 0 -> [2, 0], h0 = 1 -> [1, −1]
	const want = [0.625 * 3 + 0.375 * 3, 0.625 * -3 + 0.375 * -2, (8 / 9) * 2 + (1 / 9) * 1, (1 / 9) * -1];
	want.forEach((v, i) => assert.ok(Math.abs(y.data[i] - v) < 1e-6, `y[${i}] = ${y.data[i]}, want ${v}`));
	assert.deepEqual(Array.from(subs.gate.data).map((v) => Math.round(v * 1e6) / 1e6), [0.625, 0.375, 0, 0.111111, 0, 0.888889]);
	assert.deepEqual(Array.from(subs.e.data.subarray(4, 6)), [0, 0], 'expert 2 does not run for token 0');
	const trace = { order: [], t: { 'B1.moe#gate': subs.gate, 'B1.moe#e': subs.e } } as unknown as Trace;
	for (let t = 0; t < 2; t++)
		for (let d = 0; d < 2; d++) {
			const e = moe.explain!({ layer, unit: [t, d], ins: [x, p], out: y, w: (s) => W[s], p: { k: 2 }, trace, spec: {} as never });
			assert.equal(e.result, e.traced);
			assert.equal(e.terms!.length, 2);
		}
	assert.deepEqual(topK([0.2, 0.4, 0.4], 2).i, [1, 2], 'ties go to the lower expert');
	assert.throws(() => moe.inferShape([[2, 2], [2, 3]], { k: 4 }, { fc: [3, 1, 2], out: [3, 2, 1] }));
});
