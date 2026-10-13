// Tests for the egnn local ops (node --test): forward on a three-atom example by hand, explain() equals forward for
// every unit of pair_dist2, vec_sum and nb_sum, and the diagonal (i = j) never enters a sum.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { edge_concat, nb_sum, pair_dist2, sum_atoms, vec_sum } from './egnn.ts';
import type { AtlasLayer, OpDef, Tensor } from '../../../atlas/types/index.ts';

const x: Tensor = { shape: [3, 2], data: Float32Array.from([0, 0, 3, 0, 0, 4]) };
const w: Tensor = { shape: [3, 3, 1], data: Float32Array.from([9, 1, 2, 0.5, 9, -1, 1, 1, 9]) };
const m: Tensor = { shape: [3, 3, 2], data: Float32Array.from({ length: 18 }, (_, k) => k) };
const fwd = (op: OpDef<never>, ins: Tensor[]) => op.forward(ins, {} as never, {} as never);
const layer = (id: string, op: string, n: number) => ({ id, op, in: Array.from({ length: n }, (_, k) => `in${k}`), outShape: [] }) as unknown as AtlasLayer;
const explainAll = (op: OpDef<never>, ins: Tensor[], out: Tensor) => {
	const [a, b] = out.shape;
	for (let i = 0; i < a; i++)
		for (let j = 0; j < b; j++) {
			const e = op.explain!({ layer: layer('l', op.op, ins.length), unit: [i, j], ins, out, p: {} as never, w: () => null } as never);
			assert.equal(e.result, out.data[i * b + j], `${op.op} [${i}, ${j}]`);
			assert.ok(Math.abs(e.terms!.reduce((s, q) => s + q.p, 0) - out.data[i * b + j]) <= 1e-5);
		}
};

test('pair_dist2: squared distances by hand, symmetric, 0 on the diagonal; explain() equals forward', () => {
	const d = fwd(pair_dist2 as OpDef<never>, [x]);
	assert.deepEqual(Array.from(d.data), [0, 9, 16, 9, 0, 25, 16, 25, 0]);
	explainAll(pair_dist2 as OpDef<never>, [x], d);
});

test('edge_concat: row (i, j) is h_i, h_j, d²_ij', () => {
	const h: Tensor = { shape: [3, 1], data: Float32Array.from([1, 2, 3]) };
	const e = fwd(edge_concat as OpDef<never>, [h, fwd(pair_dist2 as OpDef<never>, [x])]);
	assert.deepEqual(e.shape, [3, 3, 3]);
	assert.deepEqual(Array.from(e.data.subarray(3 * 5, 3 * 6)), [2, 3, 25]);
});

test('vec_sum: Σ_{j≠i} (x_i − x_j) w_ij / (n − 1) by hand (the diagonal weight 9 is never used); explain() equals forward', () => {
	const v = fwd(vec_sum as OpDef<never>, [x, w]);
	// atom 0: ((0 − 3) × 1 + (0 − 0) × 2, (0 − 0) × 1 + (0 − 4) × 2) / 2 = (−1.5, −4)
	assert.deepEqual(Array.from(v.data.subarray(0, 2)), [-1.5, -4]);
	explainAll(vec_sum as OpDef<never>, [x, w], v);
	const one = fwd(vec_sum as OpDef<never>, [{ shape: [1, 2], data: Float32Array.of(1, 2) }, { shape: [1, 1, 1], data: Float32Array.of(5) }]);
	assert.deepEqual(Array.from(one.data), [0, 0]);
});

test('nb_sum and sum_atoms: sums without the diagonal, and over the atoms; explain() equals forward', () => {
	const a = fwd(nb_sum as OpDef<never>, [m]);
	// atom 0 receives rows (0, 1) and (0, 2): [2, 3] + [4, 5]
	assert.deepEqual(Array.from(a.data.subarray(0, 2)), [6, 8]);
	explainAll(nb_sum as OpDef<never>, [m], a);
	assert.deepEqual(Array.from(fwd(sum_atoms as OpDef<never>, [{ shape: [3, 1], data: Float32Array.from([1, -2, 4]) }]).data), [3]);
});
