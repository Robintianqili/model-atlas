// Tests for the contrastive-learning local ops (node --test; `atlas check` runs them in the op-tests gate). PyTorch parity of the
// three ops on the exported model is in `atlas parity contrastive-learning` (reference.json); these check the arithmetic by hand.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AnyTensor, AtlasLayer, AtlasSpec, OpCtx, OpDef, Tensor } from '../../../atlas/types/index.ts';
import { l2norm, mean_pool, similarity } from './clip.ts';

const F = (shape: number[], d: number[]): Tensor => ({ shape, data: Float32Array.from(d) });
const layer = (op: string): AtlasLayer => ({ id: 'L', op, label: { en: 'L', zh: 'L' }, in: ['a', 'b'], outShape: [] });
const w = { logit_scale: F([1], [Math.log(2)]) };
const ctx = (l: AtlasLayer): OpCtx => ({ layer: l, dims: {}, w: (s) => w[s as 'logit_scale'], has: (s) => s in w, emit: () => {} });

function run(def: OpDef<any>, ins: AnyTensor[], p: Record<string, unknown> = {}) {
	const l = layer(def.op);
	const out = def.forward(ins, p as never, ctx(l));
	for (let f = 0; f < out.data.length; f++) {
		const unit = out.shape.length === 2 ? [Math.floor(f / out.shape[1]), f % out.shape[1]] : [f];
		const e = def.explain!({ layer: l, unit, ins, out, w: (s) => w[s as 'logit_scale'], has: (s) => s in w, p: p as never, trace: { order: [], t: {} }, spec: {} as AtlasSpec });
		assert.equal(e.result, e.traced, `${def.op} [${unit}]`);
	}
	return out;
}

test('mean_pool: the average of the word rows, pads (id 0) skipped, order ignored, no words -> 0', () => {
	const e = F([3, 3, 2], [1, 2, 3, 4, 9, 9, 3, 4, 1, 2, 9, 9, 9, 9, 9, 9, 9, 9]);
	const ids: AnyTensor = { shape: [3, 3], data: Int32Array.from([5, 6, 0, 6, 5, 0, 0, 0, 0]), dtype: 'i32' };
	const y = run(mean_pool, [e, ids]);
	assert.deepEqual(y.shape, [3, 2]);
	assert.deepEqual(Array.from(y.data), [2, 3, 2, 3, 0, 0]);
	assert.deepEqual(mean_pool.inferShape([['K', 'L', 32], ['K', 'L']], {}, {}), ['K', 32]);
});

test('l2norm: every row gets length 1 and keeps its direction', () => {
	const y = run(l2norm, [F([2, 2], [3, 4, 0, -2])]);
	assert.deepEqual(Array.from(y.data).map((v) => +v.toFixed(6)), [0.6, 0.8, 0, -1]);
});

test('similarity: s · (a · b) with s = e^t, for one image against K captions and N against M', () => {
	const a = F([2], [0.6, 0.8]);
	const b = F([3, 2], [1, 0, 0, 1, 0.6, 0.8]);
	assert.deepEqual(Array.from(run(similarity, [a, b], { maxScale: 100 }).data).map((v) => +v.toFixed(5)), [1.2, 1.6, 2]);
	const A = F([2, 2], [1, 0, 0, 1]);
	const y = run(similarity, [A, b], { maxScale: 100 });
	assert.deepEqual(y.shape, [2, 3]);
	assert.deepEqual(Array.from(y.data).map((v) => +v.toFixed(5)), [2, 0, 1.2, 0, 2, 1.6]);
	assert.deepEqual(Array.from(run(similarity, [a, b], { maxScale: 1.5 }).data).map((v) => +v.toFixed(5)), [0.9, 1.2, 1.5], 'the scale is capped');
});
