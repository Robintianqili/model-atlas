// Tests for the llava-toy local ops (node --test; `atlas check` runs them in the op-tests gate).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, AtlasSpec, OpCtx, OpDef, Tensor, Trace } from '../../../atlas/types/index.ts';
import { cls_token, drop_cls, merge_image, to_tokens } from './tokens.ts';

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

test('drop_cls: every row but row 0', () => {
	const x = T([3, 2], (i) => i * 10);
	assert.deepEqual(Array.from(explainAll(drop_cls, layer('drop_cls', [2, 2]), x, {}, {}).data), [20, 30, 40, 50]);
	assert.deepEqual(drop_cls.inferShape([[17, 48]], {}, {}), [16, 48]);
	assert.throws(() => drop_cls.forward([T([1, 2], () => 0)], {}, ctxOf(layer('drop_cls', [0, 2]))));
});

test('merge_image: the first N rows come from the image tokens', () => {
	const img = T([2, 2], (i) => -(i + 1));
	const words = T([3, 2], (i) => i * 10);
	const l: AtlasLayer = { id: 'L', op: 'merge_image', label: { en: 'L', zh: 'L' }, in: ['img', 'w'], outShape: [3, 2], params: {} };
	const out = merge_image.forward([img, words], {} as never, ctxOf(l));
	assert.deepEqual(Array.from(out.data), [-1, -2, -3, -4, 40, 50]);
	const trace: Trace = { order: ['img', 'w', 'L'], t: { img, w: words, L: out } };
	for (let f = 0; f < 6; f++) {
		const e = merge_image.explain!({ layer: l, unit: [Math.floor(f / 2), f % 2], ins: [img, words], out, w: () => img, has: () => false, p: {} as never, trace, spec: {} as AtlasSpec });
		assert.equal(e.result, out.data[f]);
	}
	assert.deepEqual(merge_image.inferShape([[16, 64], ['T', 64]], {}, {}), ['T', 64]);
	assert.throws(() => merge_image.forward([words, img], {} as never, ctxOf(l)));
});
