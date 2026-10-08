// Tests for the gcn local ops (node --test; `atlas check` runs them in the op-tests gate).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, AtlasSpec, Tensor } from '../../../atlas/types/index.ts';
import { gcn_conv, gcn_norm } from './gcn.ts';

const T = (shape: number[], v: number[]): Tensor => ({ shape, data: Float32Array.from(v) });
const L = (op: string, outShape: number[]): AtlasLayer => ({ id: 'L', op, label: { en: 'L', zh: 'L' }, in: ['a', 'b'], outShape });
const subs: Record<string, Tensor> = {};
const ctx = (w: Record<string, Tensor> = {}) => ({ layer: L('x', []), dims: {}, w: (s: string) => w[s], has: (s: string) => !!w[s], emit: (k: string, x: Tensor) => void (subs[k] = x) });

test('gcn_norm: a path 0-1-2 gives 1/√(d_i d_j) with self-loops', () => {
	const A = T([3, 3], [0, 1, 0, 1, 0, 1, 0, 1, 0]);
	const y = gcn_norm.forward([A], {}, ctx());
	assert.deepEqual(Array.from(subs.deg.data), [2, 3, 2]);
	const want = [1 / 2, 1 / Math.sqrt(6), 0, 1 / Math.sqrt(6), 1 / 3, 1 / Math.sqrt(6), 0, 1 / Math.sqrt(6), 1 / 2];
	want.forEach((v, k) => assert.ok(Math.abs(y.data[k] - v) < 1e-7, `cell ${k}`));
	for (let k = 0; k < 9; k++) {
		const e = gcn_norm.explain!({ layer: L('gcn_norm', [3, 3]), unit: [Math.floor(k / 3), k % 3], ins: [A], out: y, w: () => A, p: {}, trace: { order: [], t: {} }, spec: {} as AtlasSpec });
		assert.equal(e.result, e.traced);
	}
	// an isolated member listens only to itself: its row is 1 on the diagonal
	const iso = gcn_norm.forward([T([2, 2], [0, 0, 0, 0])], {}, ctx());
	assert.deepEqual(Array.from(iso.data), [1, 0, 0, 1]);
});

test('gcn_conv: Â (H Wᵀ) + b, explained as one message per neighbour', () => {
	const ah = T([2, 2], [0.5, 0.5, 0.5, 0.5]);
	const H = T([2, 3], [1, 0, 0, 0, 1, 0]);
	const w = { weight: T([2, 3], [1, 2, 3, 4, 5, 6]), bias: T([2], [0.25, -1]) };
	const y = gcn_conv.forward([H, ah], {}, ctx(w));
	assert.deepEqual(Array.from(subs.xw.data), [1, 4, 2, 5]);
	assert.deepEqual(Array.from(y.data), [1.75, 3.5, 1.75, 3.5]);
	const e = gcn_conv.explain!({ layer: L('gcn_conv', [2, 2]), unit: [1, 1], ins: [H, ah], out: y, w: (s) => w[s as 'weight'], p: {}, trace: { order: [], t: {} }, spec: {} as AtlasSpec });
	assert.equal(e.result, e.traced);
	assert.deepEqual(e.terms!.map((x) => [x.w, x.x]), [[0.5, 4], [0.5, 5]]);
	assert.deepEqual(gcn_conv.inferShape([[34, 34], [34, 34]], {}, { weight: [4, 34] }), [34, 4]);
});
