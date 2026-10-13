// Tests of the vq-vae local ops (node --test): forward on small inputs against hand arithmetic, and explain() equal to
// the forward pass for every unit (what `atlas parity` checks on the real spec as AK202).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, OpCtx, Tensor, Trace } from '../../../atlas/types/index.ts';
import { vq } from './vq.ts';
import { pixelShuffle } from './pixel_shuffle.ts';

const f32 = (shape: number[], a: number[]): Tensor => ({ shape, data: Float32Array.from(a) });
function ctxOf(layer: AtlasLayer, w: Record<string, Tensor> = {}) {
	const subs: Record<string, Tensor> = {};
	const ctx: OpCtx = { layer, dims: {}, w: (s) => w[s], emit: (s, x) => void (subs[s] = x) };
	return { ctx, subs };
}
const unitQuery = <P>(layer: AtlasLayer, ins: Tensor[], out: Tensor, unit: number[], p: P, w: Record<string, Tensor> = {}) => ({
	layer, unit, ins, out, p, trace: { order: [], t: {} } as unknown as Trace, spec: {} as never, w: (s: string) => w[s]
});

test('vq: each cell takes the nearest codebook row (ties to the smaller index); dist, idx and explain() agree', () => {
	const layer = { id: 'zq', op: 'vq', in: ['ze'], outShape: [2, 1, 3] } as unknown as AtlasLayer;
	// three cells of 2 numbers: (0, 0), (1, 1), (2, 0); codes (0, 0), (1, 1), (1, 1) (a tie with code 1), (2, 1)
	const ze = f32([2, 1, 3], [0, 1, 2, 0, 1, 0]);
	const codebook = f32([4, 2], [0, 0, 1, 1, 1, 1, 2, 1]);
	const { ctx, subs } = ctxOf(layer, { codebook });
	const zq = vq.forward([ze], {}, ctx);
	assert.deepEqual(Array.from(subs.idx.data), [0, 1, 3]);
	assert.deepEqual(Array.from(subs.dist.data.slice(0, 3)), [0, 2, 4]); // code 0 against the three cells
	assert.deepEqual(Array.from(subs.dist.data.slice(9, 12)), [5, 1, 1]); // code 3
	assert.deepEqual(Array.from(zq.data), [0, 1, 2, 0, 1, 1]);
	for (let d = 0; d < 2; d++)
		for (let c = 0; c < 3; c++) {
			const e = vq.explain!(unitQuery(layer, [ze], zq, [d, 0, c], {}, { codebook }));
			assert.equal(e.result, e.traced);
			assert.equal(e.values.k, subs.idx.data[c]);
		}
	assert.deepEqual(vq.inferShape([[2, 4, 4]], {}, { codebook: [32, 2] }), [2, 4, 4]);
	assert.throws(() => vq.inferShape([[3, 4, 4]], {}, { codebook: [32, 2] }));
});

test('pixel_shuffle: [r·r, H, W] -> [1, H·r, W·r] as PyTorch lays it out; explain() reads the same number', () => {
	const layer = { id: 'img', op: 'pixel_shuffle', in: ['dec3'], outShape: [1, 4, 6], params: { factor: 2 } } as unknown as AtlasLayer;
	const x = f32([4, 2, 3], Array.from({ length: 24 }, (_, i) => i));
	const y = pixelShuffle.forward([x], { factor: 2 }, ctxOf(layer).ctx);
	assert.deepEqual(y.shape, [1, 4, 6]);
	// out[i·2 + a, j·2 + b] = x[a·2 + b, i, j]: the first row reads channels 0 and 1 of cells (0, 0), (0, 1), (0, 2)
	assert.deepEqual(Array.from(y.data.slice(0, 6)), [0, 6, 1, 7, 2, 8]);
	assert.deepEqual(Array.from(y.data.slice(6, 12)), [12, 18, 13, 19, 14, 20]);
	for (const [r, c] of [[0, 0], [1, 3], [3, 5]]) {
		const e = pixelShuffle.explain!(unitQuery(layer, [x], y, [0, r, c], { factor: 2 }));
		assert.equal(e.result, y.data[r * 6 + c]);
		assert.equal(e.traced, e.result);
	}
	assert.deepEqual(pixelShuffle.inferShape([[49, 4, 4]], { factor: 7 }, {}), [1, 28, 28]);
	assert.throws(() => pixelShuffle.inferShape([[48, 4, 4]], { factor: 7 }, {}));
});
