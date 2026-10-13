// Tests for the local ops 'columns' and 'flip' (node --test): forward on a small input, explain() equals the output for
// every unit. Parity against PyTorch runs in atlas parity (pipeline/crnn-ocr/ops_columns.py, ops_flip.py).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { OpDef, Tensor } from '../../../atlas/types/index.ts';
import { columns } from './columns.ts';
import { flip } from './flip.ts';

type Op = OpDef<Record<string, never>>;
const run = (op: Op, x: Tensor) => {
	const out = op.forward([x], {} as never, {} as never) as Tensor;
	const layer = { id: 'l', op: op.op, in: ['x'], outShape: out.shape } as never;
	return { out, at: (unit: number[]) => op.explain!({ layer, unit, ins: [x], out, p: {} } as never) };
};

test('columns: [C, H, W] -> [W, C·H], feature c·H + h of column w is x[c, h, w]', () => {
	const x = { shape: [2, 3, 4], data: Float32Array.from({ length: 24 }, (_, i) => i) };
	const { out, at } = run(columns, x);
	assert.deepEqual(out.shape, [4, 6]);
	for (let w = 0; w < 4; w++)
		for (let f = 0; f < 6; f++) {
			const [c, h] = [Math.floor(f / 3), f % 3];
			assert.equal(out.data[w * 6 + f], x.data[(c * 3 + h) * 4 + w]);
			assert.equal(at([w, f]).result, out.data[w * 6 + f]);
		}
	assert.deepEqual(columns.inferShape([[16, 8, 32]], {} as never, {}), [32, 128]);
});

test('flip: row t is row T − 1 − t; flipping twice gives the input back', () => {
	const x = { shape: [5, 2], data: Float32Array.from({ length: 10 }, (_, i) => i * 0.5 - 1) };
	const { out, at } = run(flip, x);
	for (let r = 0; r < 5; r++)
		for (let d = 0; d < 2; d++) {
			assert.equal(out.data[r * 2 + d], x.data[(4 - r) * 2 + d]);
			assert.equal(at([r, d]).result, out.data[r * 2 + d]);
		}
	assert.deepEqual(Array.from(run(flip, out).out.data), Array.from(x.data));
});
