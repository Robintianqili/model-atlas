// Tests of the vae-2d local ops (node --test): forward on small inputs against hand arithmetic, and explain() equal to
// the forward pass for every unit (what `atlas parity` checks on the real spec as AK202).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, OpCtx, Tensor, Trace } from '../../../atlas/types/index.ts';
import { reparam } from './reparam.ts';
import { toImage } from './to_image.ts';

const f32 = (a: number[]): Tensor => ({ shape: [a.length], data: Float32Array.from(a) });
function ctxOf(layer: AtlasLayer) {
	const subs: Record<string, Tensor> = {};
	const ctx: OpCtx = { layer, dims: {}, w: () => { throw new Error('no weights'); }, emit: (s, x) => void (subs[s] = x) };
	return { ctx, subs };
}
const unitQuery = (layer: AtlasLayer, ins: Tensor[], out: Tensor, unit: number[]) => ({
	layer, unit, ins, out, p: {}, trace: { order: [], t: {} } as unknown as Trace, spec: {} as never, w: () => { throw new Error('no weights'); }
});

test('reparam: z = mu + exp(logvar / 2) · eps, with sigma as a sub-step; explain() equals the trace', () => {
	const layer = { id: 'z', op: 'reparam', in: ['mu', 'logvar', 'eps'], outShape: [2] } as unknown as AtlasLayer;
	const mu = f32([0.5, -1.25]), lv = f32([-2, 0.4]), eps = f32([1.5, -0.3]);
	const { ctx, subs } = ctxOf(layer);
	const z = reparam.forward([mu, lv, eps], {}, ctx);
	for (let i = 0; i < 2; i++) {
		const s = Math.exp(0.5 * lv.data[i]);
		assert.ok(Math.abs(subs.sigma.data[i] - s) < 1e-6);
		assert.ok(Math.abs(z.data[i] - (mu.data[i] + s * eps.data[i])) < 1e-6);
		const e = reparam.explain!(unitQuery(layer, [mu, lv, eps], z, [i]));
		assert.equal(e.result, e.traced);
		assert.equal(e.values.z, z.data[i]);
		assert.ok(Math.abs(e.terms![0].p + e.terms![1].p - e.result) < 1e-6);
	}
	assert.deepEqual(reparam.inferShape([[2], [2], [2]], {}, {}), [2]);
	assert.throws(() => reparam.inferShape([[2], [3], [2]], {}, {}));
});

test('to_image: 784 numbers become [1, 28, 28] unchanged; explain() reads the same number', () => {
	const layer = { id: 'img', op: 'to_image', in: ['dec3'], outShape: [1, 28, 28] } as unknown as AtlasLayer;
	const x = f32(Array.from({ length: 784 }, (_, i) => Math.sin(i)));
	const y = toImage.forward([x], {}, ctxOf(layer).ctx);
	assert.deepEqual(y.shape, [1, 28, 28]);
	assert.deepEqual(Array.from(y.data), Array.from(x.data));
	for (const [r, c] of [[0, 0], [3, 7], [27, 27]]) {
		const e = toImage.explain!(unitQuery(layer, [x], y, [0, r, c]));
		assert.equal(e.result, x.data[r * 28 + c]);
		assert.equal(e.traced, e.result);
	}
	assert.deepEqual(toImage.inferShape([[784]], {}, {}), [1, 28, 28]);
	assert.throws(() => toImage.inferShape([[783]], {}, {}));
});
