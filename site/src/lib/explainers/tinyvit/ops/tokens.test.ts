// Tests for the tinyvit local ops (node --test; `atlas check` runs them in the op-tests gate).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, AtlasSpec, OpCtx, OpDef, Tensor, Trace } from '../../../atlas/types/index.ts';
import { cls_token, select_token, to_tokens } from './tokens.ts';

const T = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });
const layer = (op: string, outShape: number[], params = {}): AtlasLayer => ({ id: 'L', op, label: { en: 'L', zh: 'L' }, in: ['x'], outShape, params });
const ctxOf = (l: AtlasLayer, w: Record<string, Tensor> = {}): OpCtx => ({ layer: l, dims: {}, w: (s) => w[s], has: (s) => !!w[s], emit: () => {} });

function explainAll(def: OpDef<any>, l: AtlasLayer, x: Tensor, w: Record<string, Tensor>, p: Record<string, unknown>) {
	const out = def.forward([x], p as never, ctxOf(l, w));
	const trace: Trace = { order: ['x', 'L'], t: { x, L: out } };
	for (let f = 0; f < out.data.length; f++) {
		const unit = out.shape.length === 2 ? [Math.floor(f / out.shape[1]), f % out.shape[1]] : [f];
		const e = def.explain!({ layer: l, unit, ins: [x], out, w: (s) => w[s], has: (s) => !!w[s], p: p as never, trace, spec: {} as AtlasSpec });
		assert.equal(e.result, e.traced, `${def.op} unit [${unit}]`);
		assert.equal(e.result, out.data[f]);
	}
	return out;
}

test('to_tokens: [C, H, W] -> [H·W, C], patch t = row-major (y, x)', () => {
	const x = T([3, 2, 2], (i) => i); // channel c, cell (y, x) holds 4c + 2y + x
	const l = layer('to_tokens', [4, 3]);
	const y = explainAll(to_tokens, l, x, {}, {});
	assert.deepEqual(y.shape, [4, 3]);
	assert.deepEqual(Array.from(y.data), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
	assert.deepEqual(to_tokens.inferShape([[48, 4, 4]], {}, {}), [16, 48]);
	const e = to_tokens.explain!({ layer: l, unit: [3, 2], ins: [x], out: y, w: () => x, p: {}, trace: { order: [], t: {} }, spec: {} as AtlasSpec });
	assert.deepEqual([e.values.row, e.values.col, e.values.c], [1, 1, 2]);
});

test('cls_token: the learned row goes in front', () => {
	const x = T([2, 3], (i) => i + 1);
	const w = { weight: T([3], (i) => -(i + 1)) };
	const y = explainAll(cls_token, layer('cls_token', [3, 3]), x, w, {});
	assert.deepEqual(Array.from(y.data), [-1, -2, -3, 1, 2, 3, 4, 5, 6]);
	assert.deepEqual(cls_token.inferShape([[16, 48]], {}, { weight: [48] }), [17, 48]);
});

test('select_token: row `index` only', () => {
	const x = T([3, 2], (i) => i * 10);
	assert.deepEqual(Array.from(explainAll(select_token, layer('select_token', [2], { index: 0 }), x, {}, { index: 0 }).data), [0, 10]);
	assert.deepEqual(Array.from(explainAll(select_token, layer('select_token', [2], { index: 2 }), x, {}, { index: 2 }).data), [40, 50]);
	assert.throws(() => select_token.forward([x], { index: 3 }, ctxOf(layer('select_token', [2]))));
});
