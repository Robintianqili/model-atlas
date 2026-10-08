// Tests for the local op 'attn_decoder' (node --test). The PyTorch check is `atlas parity tinyseq2seq` (every sub-step
// of every case <= 1e-4); here: shapes, weights that add up to 1, a context that is the weighted mix of the encoder
// states, a greedy choice that is the argmax, and agreement with the float64 decoder of seq.ts (the process's).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, Tensor } from '../../../atlas/types/index.ts';
import { createRng } from '../../../atlas/core/rng.ts';
import { attnDecoder } from './attn_decoder.ts';
import { argmax, decW, f64, greedy } from '../seq.ts';

const [T, H, A, E, V, L] = [5, 4, 3, 2, 6, 4];
const rng = createRng(11);
const ten = (shape: number[]): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rng.next() * 2 - 1) });
const W: Record<string, Tensor> = {
	emb: ten([V, E]), wq: ten([A, H]), wk: ten([A, H]), bk: ten([A]), v: ten([1, A]),
	weight_ih: ten([3 * H, E + H]), weight_hh: ten([3 * H, H]), bias_ih: ten([3 * H]), bias_hh: ten([3 * H]), wo: ten([V, 2 * H]), bo: ten([V])
};
const layer: AtlasLayer = { id: 'dec', op: 'attn_decoder', label: { en: 'Decoder', zh: '解码器' }, in: ['enc'], outShape: [L, V], params: { steps: L, bos: 0 } };
const enc = ten([T, H]);
const subs: Record<string, Tensor> = {};
const out = attnDecoder.forward([enc], { steps: L, bos: 0 }, { layer, dims: {}, w: (s) => W[s], emit: (k, v) => (subs[k] = v) });

test('attn_decoder: shapes', () => {
	assert.deepEqual(out.shape, [L, V]);
	const want: Record<string, number[]> = { keys: [T, A], query: [L, A], scores: [L, T], weights: [L, T], context: [L, H], state: [L, H], next: [L] };
	for (const [k, s] of Object.entries(want)) assert.deepEqual(subs[k].shape, s, k);
	assert.deepEqual(attnDecoder.inferShape([['T', H]], { steps: L, bos: 0 }, { wo: [V, 2 * H] }), [L, V]);
});

test('attn_decoder: weights add up to 1, the context mixes the encoder states, the choice is the argmax', () => {
	for (let t = 0; t < L; t++) {
		const a = subs.weights.data.subarray(t * T, (t + 1) * T);
		assert.ok(Math.abs(Array.from(a).reduce((s, x) => s + x, 0) - 1) < 1e-6);
		for (let k = 0; k < H; k++) {
			let c = 0;
			for (let j = 0; j < T; j++) c += a[j] * enc.data[j * H + k];
			assert.ok(Math.abs(c - subs.context.data[t * H + k]) < 1e-6);
		}
		assert.equal(subs.next.data[t], argmax(out.data.subarray(t * V, (t + 1) * V)));
	}
});

test('attn_decoder: float32 storage stays within 1e-5 of the float64 decoder (seq.ts, the process’s arithmetic)', () => {
	const w = decW((n) => W[n].data, (n) => W[n].shape, '');
	let t = 0;
	for (const r of greedy(w, enc.data, T, f64, L)) {
		for (let o = 0; o < V; o++) assert.ok(Math.abs(r.logits[o] - out.data[t * V + o]) < 1e-5);
		t++;
	}
});
