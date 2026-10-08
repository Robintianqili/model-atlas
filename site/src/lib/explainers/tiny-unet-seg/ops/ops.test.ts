// Tests of the tiny-unet-seg local op 'upsample' (node --test): forward against hand arithmetic (PyTorch's bilinear,
// align_corners=False), explain() equal to the trace for every unit, and the receptive box.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, OpCtx, Tensor, Trace } from '../../../atlas/types/index.ts';
import { upsample, taps } from './upsample.ts';

const layer = { id: 'up', op: 'upsample', in: ['b'], outShape: [1, 4, 4], params: { scale: 2 } } as unknown as AtlasLayer;
const ctx: OpCtx = { layer, dims: {}, w: () => { throw new Error('no weights'); }, emit: () => {} };
const x: Tensor = { shape: [1, 2, 2], data: Float32Array.of(1, 2, 3, 4) };

test('upsample: 2×2 -> 4×4 as F.interpolate(bilinear, align_corners=False)', () => {
	const y = upsample.forward([x], { scale: 2 }, ctx);
	assert.deepEqual(y.shape, [1, 4, 4]);
	// torch: [[1, 1.25, 1.75, 2], [1.5, 1.75, 2.25, 2.5], [2.5, 2.75, 3.25, 3.5], [3, 3.25, 3.75, 4]]
	assert.deepEqual(Array.from(y.data), [1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4]);
	assert.deepEqual(taps(0, 2, 2), [0, 1, 0]);
	assert.deepEqual(taps(1, 2, 2), [0, 1, 0.25]);
	assert.deepEqual(taps(3, 2, 2), [1, 1, 0.25]);
	for (let i = 0; i < 16; i++) {
		const unit = [0, Math.floor(i / 4), i % 4];
		const e = upsample.explain!({ layer, unit, ins: [x], out: y, p: { scale: 2 }, trace: { order: [], t: {} } as unknown as Trace, spec: {} as never, w: () => { throw new Error('no weights'); } });
		assert.equal(e.result, y.data[i]);
		assert.equal(e.traced, y.data[i]);
		assert.ok(Math.abs(e.terms!.reduce((s, q) => s + q.w, 0) - 1) < 1e-12);
	}
	assert.deepEqual(upsample.inferShape([[32, 8, 8]], { scale: 2 }, {}), [32, 16, 16]);
	assert.deepEqual(upsample.receptive!({ y0: 5, x0: 5, y1: 5, x1: 5 }, { scale: 2 }), { y0: 2, x0: 2, y1: 3, x1: 3 });
});
