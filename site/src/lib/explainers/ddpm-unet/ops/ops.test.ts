// Tests of the ddpm-unet local ops (node --test): forward on small inputs against hand arithmetic, and explain() equal to
// the forward pass (what `atlas parity` checks on the real spec as AK202).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, OpCtx, Tensor, Trace } from '../../../atlas/types/index.ts';
import { qSample } from './q_sample.ts';
import { timeEmbed } from './time_embed.ts';
import { upsample } from './upsample.ts';

const f32 = (shape: number[], a: number[]): Tensor => ({ shape, data: Float32Array.from(a) });
const layer = (id: string, op: string, inputs: string[], outShape: number[]) => ({ id, op, in: inputs, outShape }) as unknown as AtlasLayer;
const ctx = (l: AtlasLayer): OpCtx => ({ layer: l, dims: {}, w: () => { throw new Error('no weights'); }, emit: () => {} });
const query = <P>(l: AtlasLayer, ins: Tensor[], out: Tensor, unit: number[], p: P) => ({
	layer: l, unit, ins, out, p, trace: { order: [], t: {} } as unknown as Trace, spec: {} as never, w: () => { throw new Error('no weights'); }
});

test('q_sample: x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε with ᾱ_t read at index t − 1; explain() equals the forward pass', () => {
	const l = layer('xt', 'q_sample', ['input', 'eps', 't'], [1, 2, 2]);
	const p = { abar: [0.9, 0.5, 0.1] };
	const x0 = f32([1, 2, 2], [0, 1, 0.5, 0]), eps = f32([1, 2, 2], [0.3, -1, 2, 0]), t = f32([1], [2]);
	const y = qSample.forward([x0, eps, t], p, ctx(l));
	for (let i = 0; i < 4; i++) {
		assert.ok(Math.abs(y.data[i] - (Math.sqrt(0.5) * (2 * x0.data[i] - 1) + Math.sqrt(0.5) * eps.data[i])) < 1e-6);
		const e = qSample.explain!(query(l, [x0, eps, t], y, [0, i >> 1, i & 1], p));
		assert.equal(e.result, e.traced);
	}
	assert.throws(() => qSample.forward([x0, eps, f32([1], [4])], p, ctx(l)), /outside/);
	assert.throws(() => qSample.inferShape([[1, 2, 2], [1, 2, 2], [2]], p, {}));
});

test('time_embed: sin(t·f_k) then cos(t·f_k) with f_k = 10000^(−k/half), shape [dim, 1, 1]', () => {
	const l = layer('temb', 'time_embed', ['t'], [4, 1, 1]);
	const y = timeEmbed.forward([f32([1], [7])], { dim: 4 }, ctx(l));
	const f = [1, Math.exp(-Math.log(10000) / 2)];
	assert.deepEqual(Array.from(y.data), [Math.sin(7 * f[0]), Math.sin(7 * f[1]), Math.cos(7 * f[0]), Math.cos(7 * f[1])].map(Math.fround));
	for (let i = 0; i < 4; i++) assert.equal(timeEmbed.explain!(query(l, [f32([1], [7])], y, [i, 0, 0], { dim: 4 })).result, y.data[i]);
	assert.throws(() => timeEmbed.inferShape([[1]], { dim: 3 }, {}));
});

test('upsample: every number becomes a 2×2 block; explain() reads the source cell', () => {
	const l = layer('up', 'upsample', ['h'], [1, 4, 4]);
	const x = f32([1, 2, 2], [1, 2, 3, 4]);
	const y = upsample.forward([x], { scale: 2 }, ctx(l));
	assert.deepEqual(y.shape, [1, 4, 4]);
	assert.deepEqual(Array.from(y.data), [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
	const e = upsample.explain!(query(l, [x], y, [0, 3, 1], { scale: 2 }));
	assert.equal(e.result, 3);
	assert.deepEqual(e.reads[0].cells, [2]);
});
