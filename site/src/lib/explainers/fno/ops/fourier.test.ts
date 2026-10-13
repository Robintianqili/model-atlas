// fno local op tests: rfft and irfft invert each other on the kept modes, keep_modes zeroes the cut modes, spectral_mix
// is a complex multiply, and every explain() adds up to the forward value. Run: node --test src/lib/explainers/fno/ops/fourier.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AtlasLayer, Tensor } from '../../../atlas/types/index.ts';
import { irfft, keep_modes, rfft, spectral_mix } from './fourier.ts';
import { lowPass, presetY, spectrum } from '../heat.ts';

const f64 = (shape: number[], a: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(a) });
const layer = (id: string) => ({ id, in: ['x'] }) as unknown as AtlasLayer;
const u = presetY('square'), x = f64([64, 1], u);

test('rfft equals heat.ts spectrum(); irfft of the lowest 16 modes equals lowPass(u, 16)', () => {
	const X = rfft.forward([x], {}, {} as never), s = spectrum(u);
	assert.deepEqual(X.shape, [33, 1, 2]);
	for (let k = 0; k < 33; k++) assert.ok(Math.abs(X.data[2 * k] - s.re[k]) < 1e-12 && Math.abs(X.data[2 * k + 1] - s.im[k]) < 1e-12);
	const kept = keep_modes.forward([X, f64([1], [16])], { modes: 16 }, {} as never);
	const back = irfft.forward([kept], { n: 64 }, {} as never), ref = lowPass(u, 16);
	back.data.forEach((v, j) => assert.ok(Math.abs(v - ref[j]) < 1e-9, `point ${j}`));
});

test('keep_modes copies the modes below keep and zeroes the rest', () => {
	const X = rfft.forward([x], {}, {} as never), kept = keep_modes.forward([X, f64([1], [3])], { modes: 16 }, {} as never);
	assert.deepEqual(kept.shape, [16, 1, 2]);
	for (let k = 0; k < 16; k++) for (const p of [0, 1]) assert.equal(kept.data[2 * k + p], k < 3 ? X.data[2 * k + p] : 0);
});

test('spectral_mix is (a + bi)(c + di) summed over the input channels; explain() adds up to the forward value', () => {
	const X = f64([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]), wr = f64([2, 2, 1], [0.5, -1, 2, 0.25]), wi = f64([2, 2, 1], [1, 0.5, -0.5, 3]);
	const w = (s: string) => (s === 'wr' ? wr : wi), Y = spectral_mix.forward([X], {}, { w } as never);
	// mode 0: (1 + 2i)(0.5 + 1i) + (3 + 4i)(−1 + 0.5i) = (−1.5 + 2i) + (−5 − 2.5i) = −6.5 − 0.5i
	assert.deepEqual(Array.from(Y.data.slice(0, 2)), [-6.5, -0.5]);
	for (const p of [0, 1]) {
		const e = spectral_mix.explain!({ layer: layer('mix'), unit: [1, 0, p], ins: [X], out: Y, w, p: {}, trace: {} as never, spec: {} as never });
		assert.ok(Math.abs(e.result - Y.data[2 + p]) < 1e-12 && e.terms!.length === 4);
	}
	const Xf = rfft.forward([x], {}, {} as never), ef = rfft.explain!({ layer: layer('fft'), unit: [3, 0, 1], ins: [x], out: Xf, w, p: {}, trace: {} as never, spec: {} as never });
	assert.ok(Math.abs(ef.result - Xf.data[7]) < 1e-12 && ef.terms!.length === 64);
});
