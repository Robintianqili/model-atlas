// core/: rng test vectors, compare, io, format, tensor helpers, tex.
// Run from site/: node --test src/lib/atlas/core/core.test.ts   (part of npm run check:atlas)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng, fnv1a32, mulberry32, sampleIndices } from './rng.ts';
import { allclose, bitEqual, maxDiff } from './compare.ts';
import { b64ToBytes, bytesToB64, f32FromB64, f32ToB64, f64FromB64, f64ToB64, fsFetch, i32FromB64, i32ToB64, joinUrl, loadJson, parseJson } from './io.ts';
import { fmt3, fmtFull, fmtInt, fmtPct, fmtShape, fmtSigned, fmtValue, shapeText, sigDigits } from './format.ts';
import { argmax, argmaxAbs, argmin, flatIndex, numel, resolveShape, strides, unravel } from './tensor.ts';
import { sym, texNames } from './tex.ts';
import type { FetchLike } from '../types/index.ts';

test('rng: mulberry32 test vectors (types/core.ts) and the Rng contract', () => {
	const r42 = createRng(42);
	assert.deepEqual([r42.next(), r42.next(), r42.next(), r42.next()].map((v) => v.toFixed(10)), ['0.6011037519', '0.4482905590', '0.8524657935', '0.6697340414']);
	const r1 = createRng(1);
	assert.deepEqual([r1.next(), r1.next(), r1.next()].map((v) => v.toFixed(10)), ['0.6270739406', '0.0027357212', '0.5274470400']);
	// the plain closure is the same generator (tinygpt.ts copy)
	const m = mulberry32(42);
	const r = createRng(42);
	for (let i = 0; i < 100; i++) assert.equal(m(), r.next());
	// int(n) = floor(next()·n)
	const a = createRng(7);
	const b = createRng(7);
	for (let i = 0; i < 50; i++) assert.equal(a.int(10), Math.floor(b.next() * 10));
	// state round trip
	const s = createRng(3);
	s.next();
	const st = s.getState();
	const x = s.next();
	s.setState(st);
	assert.equal(s.next(), x);
});

test('rng: fork depends only on seed and label; fnv1a32 vectors', () => {
	assert.equal(fnv1a32(''), 0x811c9dc5);
	assert.equal(fnv1a32('a'), 0xe40c292c);
	assert.equal(fnv1a32('foobar'), 0xbf9cf968);
	const r = createRng(42);
	const f1 = r.fork('init');
	r.next();
	r.next();
	const f2 = r.fork('init');
	assert.equal(f1.seed, f2.seed);
	assert.equal(f1.seed, (42 ^ fnv1a32('init')) >>> 0);
	assert.equal(f1.next(), f2.next());
	assert.notEqual(r.fork('batch').seed, f1.seed);
	// documented vectors for the Python mirror (ak.Rng): FNV-1a over UTF-8 bytes
	assert.equal(fnv1a32('init'), 0x16b1d373);
	assert.equal(fnv1a32('batch'), 0x1a03dffb);
	assert.equal(fnv1a32('训练'), 0x2d61933d);
	const fi = createRng(42).fork('init');
	assert.equal(fi.seed, 380752729);
	assert.deepEqual([fi.next(), fi.next(), fi.next()].map((v) => v.toFixed(10)), ['0.3591537452', '0.3911991646', '0.0940016529']);
	const ri = createRng(42);
	assert.deepEqual([ri.int(10), ri.int(10), ri.int(10), ri.int(10), ri.int(10)], [6, 4, 8, 6, 1]);
	const n = createRng(5);
	for (let i = 0; i < 1000; i++) assert.ok(Number.isFinite(n.normal()));
	const idx = sampleIndices(createRng(9), 50, 50);
	assert.equal(new Set(idx).size, 50);
	assert.ok(idx.every((v) => v >= 0 && v < 50));
});

test('compare: maxDiff is NaN-safe; −∞ == −∞; relative mode', () => {
	assert.deepEqual(maxDiff([1, -Infinity, 3], [1, -Infinity, 3]), { max: 0, at: -1, nan: false });
	assert.deepEqual(maxDiff([1, 2], [1, NaN]), { max: Infinity, at: 1, nan: true });
	assert.deepEqual(maxDiff([1, -Infinity], [1, 0]), { max: Infinity, at: 1, nan: true });
	assert.deepEqual(maxDiff([Infinity], [-Infinity]), { max: Infinity, at: 0, nan: true });
	assert.equal(maxDiff([1, 2], [1]).max, Infinity);
	const r = maxDiff([100.5, 0.5], [100, 0.25], { relative: true });
	assert.equal(r.at, 1);
	assert.equal(r.max, 0.25);
	assert.equal(maxDiff([100.5], [100], { relative: true }).max, 0.005);
	const c = allclose([1 + 1e-6, 2], [1, 2], { atol: 1e-6, rtol: 1e-6 });
	assert.ok(c.ok);
	assert.ok(!allclose([1.1], [1], { atol: 1e-6, rtol: 1e-6 }).ok);
	assert.ok(!allclose([NaN], [1], { atol: 1, rtol: 1 }).ok);
	assert.ok(bitEqual(Float32Array.of(1, -0), Float32Array.of(1, -0)).equal);
	assert.equal(bitEqual([0], [-0]).at, 0);
});

test('io: base64 round trips, BOM-stripping loadJson, fsFetch', async () => {
	const f = Float32Array.of(1.5, -Infinity, 0, -0, 3.4028234663852886e38);
	assert.ok(bitEqual(f32FromB64(f32ToB64(f)), f).equal);
	const d = Float64Array.of(Math.PI, -1e-300);
	assert.ok(bitEqual(f64FromB64(f64ToB64(d)), d).equal);
	assert.deepEqual([...i32FromB64(i32ToB64([1, -2, 1023]))], [1, -2, 1023]);
	const big = new Uint8Array(100000).map((_, i) => i % 251);
	assert.deepEqual(b64ToBytes(bytesToB64(big)), big);
	assert.deepEqual(parseJson('﻿{"a":1}'), { a: 1 });
	const fake: FetchLike = async () => ({ ok: true, json: async () => ({}), arrayBuffer: async () => new TextEncoder().encode('﻿[1,2]').buffer as ArrayBuffer });
	assert.deepEqual(await loadJson(fake, 'x.json'), [1, 2]);
	assert.equal(joinUrl('models/x/', '/spec.json'), 'models/x/spec.json');
	const here = new URL('./core.test.ts', import.meta.url);
	const r = await fsFetch(here.href);
	assert.equal(r.ok, true);
	assert.match(await r.text!(), /core\/: rng test vectors/);
	const miss = await fsFetch(new URL('./does-not-exist.json', import.meta.url).href);
	assert.equal(miss.ok, false);
	assert.equal(miss.status, 404);
});

test('format and tex', () => {
	assert.equal(fmt3(-0.5), '−0.500');
	assert.equal(fmt3(12345), '12300');
	assert.equal(fmt3(123456), '1.23e5');
	assert.equal(fmt3(0.000123), '1.23e−4');
	// one minus sign (U+2212) everywhere, also in the exponent of a negative number (R0: "−7.70e-4")
	assert.equal(fmt3(-0.00077), '−7.70e−4');
	assert.equal(fmtFull(-1e-7), '−1e−7');
	assert.ok(![fmt3(-0.00077), fmt3(-1.5e-9, 4), fmtFull(-2.5e-12), fmtInt(-1234), fmtPct(-0.25)].some((s) => s.includes('-')), 'no ASCII hyphen-minus');
	// n2..n6: more (or fewer) significant digits, for narrating small changes
	assert.equal(fmtValue(0.69913, 'n4'), '0.6991');
	assert.equal(fmtValue(0.69874, 'n4'), '0.6987');
	assert.equal(fmtValue(0.69913, 'n3'), fmtValue(0.69874, 'n3'));
	assert.equal(fmtValue(-0.000376, 'n2'), '−3.8e−4');
	assert.equal(sigDigits('n5'), 5);
	assert.equal(sigDigits('n3x'), null);
	assert.equal(fmt3(-Infinity), '−∞');
	assert.equal(fmtInt(6722), '6,722');
	assert.equal(fmtPct(0.8734), '87.3%');
	assert.equal(fmtSigned(0.25), '+0.250');
	assert.equal(fmtShape([8, 26, 26]), '[8×26×26]');
	assert.equal(shapeText(['T', 76]), 'T×76');
	assert.equal(fmtValue(3, 'int'), '3');
	assert.equal(fmtValue('8×26×26', 'shape'), '8×26×26');
	assert.equal(fmtValue(' a', 'tok'), '␣a');
	assert.equal(sym('param', 'W'), '\\htmlClass{sym-param}{W}');
	assert.deepEqual(texNames('{{s}} + {{ b }} = {{y}} {{s}}'), ['s', 'b', 'y']);
});

test('tensor helpers', () => {
	assert.equal(numel([8, 26, 26]), 5408);
	assert.deepEqual(strides([2, 3, 4]), [12, 4, 1]);
	assert.equal(flatIndex([2, 3, 4], [1, 2, 3]), 23);
	assert.deepEqual(unravel([2, 3, 4], 23), [1, 2, 3]);
	assert.throws(() => flatIndex([2, 3], [2, 0]), RangeError);
	assert.deepEqual(resolveShape(['T', 76], { T: 9 }), [9, 76]);
	assert.throws(() => resolveShape(['N'], {}));
	assert.equal(argmax([1, 3, 3, 2]), 1);
	assert.equal(argmin([1, -3, -3]), 1);
	assert.equal(argmaxAbs([1, -3, 2]), 1);
});
