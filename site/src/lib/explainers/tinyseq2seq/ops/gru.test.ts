// Tests for the local op 'gru' (node --test). The PyTorch check is `atlas parity tinyseq2seq` (every case of
// reference.json, the encoder and its gates <= 1e-4); here: shapes, sub-steps, and the GRU equations written out by hand.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, Tensor } from '../../../atlas/types/index.ts';
import { createRng } from '../../../atlas/core/rng.ts';
import { gru } from './gru.ts';

const In = 2;
const H = 3;
const T = 4;
const rng = createRng(7);
const ten = (shape: number[]): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rng.next() * 2 - 1) });
const W: Record<string, Tensor> = { weight_ih: ten([3 * H, In]), weight_hh: ten([3 * H, H]), bias_ih: ten([3 * H]), bias_hh: ten([3 * H]) };
const layer: AtlasLayer = { id: 'enc', op: 'gru', label: { en: 'GRU', zh: 'GRU' }, in: ['x'], outShape: ['T', H], params: { hidden: H } };
const x = ten([T, In]);

test('gru: shapes and the sub-steps r, z, n', () => {
	const subs: Record<string, Tensor> = {};
	const y = gru.forward([x], { hidden: H }, { layer, dims: {}, w: (s) => W[s], emit: (k, v) => (subs[k] = v) });
	assert.deepEqual(y.shape, [T, H]);
	assert.deepEqual(Object.keys(subs), ['r', 'z', 'n']);
	for (const v of Object.values(subs)) assert.deepEqual(v.shape, [T, H]);
	assert.deepEqual(gru.inferShape([['T', In]], { hidden: H }, { weight_hh: [3 * H, H] }), ['T', H]);
});

test('gru: h = (1 − z)·n + z·h_prev with r, z, n as nn.GRU computes them', () => {
	const y = gru.forward([x], { hidden: H }, { layer, dims: {}, w: (s) => W[s], emit: () => {} });
	const sig = (v: number) => 1 / (1 + Math.exp(-v));
	const lin = (w: Tensor, b: Tensor, r: number, v: ArrayLike<number>, n: number) => {
		let s = b.data[r];
		for (let k = 0; k < n; k++) s += w.data[r * n + k] * v[k];
		return s;
	};
	let h = [0, 0, 0];
	for (let t = 0; t < T; t++) {
		const xt = x.data.subarray(t * In, (t + 1) * In);
		const next = h.map((hp, j) => {
			const r = sig(lin(W.weight_ih, W.bias_ih, j, xt, In) + lin(W.weight_hh, W.bias_hh, j, h, H));
			const z = sig(lin(W.weight_ih, W.bias_ih, H + j, xt, In) + lin(W.weight_hh, W.bias_hh, H + j, h, H));
			const n = Math.tanh(lin(W.weight_ih, W.bias_ih, 2 * H + j, xt, In) + r * lin(W.weight_hh, W.bias_hh, 2 * H + j, h, H));
			return (1 - z) * n + z * hp;
		});
		for (let j = 0; j < H; j++) assert.ok(Math.abs(next[j] - y.data[t * H + j]) < 1e-6, `t ${t} unit ${j}`);
		h = next.map(Math.fround);
	}
});
