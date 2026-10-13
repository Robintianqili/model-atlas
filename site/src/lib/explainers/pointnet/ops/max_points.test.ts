// Tests for the local op max_points (node --test): forward on a small table, explain() equals the trace for every
// feature, and the order of the points does not change the result.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxPoints, winner } from './max_points.ts';
import type { AtlasLayer, Tensor } from '../../../atlas/types/index.ts';

const x: Tensor = { shape: [4, 3], data: Float32Array.from([0, 2, -1, 5, 2, -3, 1, 0, -2, 5, 1, -4]) };
const layer = { id: 'pool', op: 'max_points', in: ['h'], outShape: [3] } as unknown as AtlasLayer;

test('max_points: the largest value of each feature, first winner on a tie', () => {
	const y = maxPoints.forward([x], {} as never, {} as never);
	assert.deepEqual(y.shape, [3]);
	assert.deepEqual(Array.from(y.data), [5, 2, -1]);
	assert.equal(winner(x.data, 4, 3, 0).at, 1);
	assert.equal(winner(x.data, 4, 3, 1).at, 0);
});

test('max_points: explain() equals the trace for every feature', () => {
	const out = maxPoints.forward([x], {} as never, {} as never);
	for (let f = 0; f < 3; f++) {
		const e = maxPoints.explain!({ layer, unit: [f], ins: [x], out, p: {} as never, w: () => null } as never);
		assert.equal(e.result, out.data[f]);
		assert.equal(e.terms!.reduce((s, q) => s + q.p, 0), out.data[f]);
		assert.equal(e.values!.argmax, winner(x.data, 4, 3, f).at);
	}
});

test('max_points: shuffling the points gives exactly the same output', () => {
	const order = [2, 0, 3, 1];
	const sh: Tensor = { shape: [4, 3], data: Float32Array.from(order.flatMap((i) => Array.from(x.data.subarray(i * 3, i * 3 + 3)))) };
	assert.deepEqual(Array.from(maxPoints.forward([sh], {} as never, {} as never).data), Array.from(maxPoints.forward([x], {} as never, {} as never).data));
});
