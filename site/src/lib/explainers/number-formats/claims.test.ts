// number-formats claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the lesson's fixtures, the
// adapter's edge fixtures and the start input) through withFixtures. `atlas check` runs this file (AK408 on failure).
// The casts themselves are checked against PyTorch's dtypes by `atlas parity` (pipeline/number-formats/reference.py);
// the PyTorch values pinned below were printed by torch 2.11 (x.to(dtype) from float32).
// Run alone: node --test src/lib/explainers/number-formats/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { BIG_X, cast, castArray, decode, encode, finiteCount, flipBit, FMTS, gapAtOne, HUGE_X, largest, NF4, rawOf, rowOf, smallest, smallestNormal, START, TINY_X, type FloatFmt } from './fmt.ts';
import { castIndex, CASTS, matrixNames } from './process.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'number-formats';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', SLUG, 'facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const fact = (k: string) => facts[k].value;
const fl = (id: 'fp32' | 'fp16' | 'bf16' | 'e4m3' | 'e5m2' | 'fp4') => FMTS[id] as FloatFmt;

/** Every frame of the sweep, by phase and step. */
async function sweepOf(rt: ExplainerRuntime) {
	await rt.ensure('sweep', 'last');
	const t = rt.sources.sweep.tape!;
	const at = (phase: string, step = 0): Frame => t.frame(t.find({ phase, step }));
	return { t, at, cast: (id: string) => at('cast', castIndex(id)).metrics! };
}

test('steps "fields", "predict", "fp8": the bit splits (fp32 1 + 8 + 23, fp16 1 + 5 + 10, bf16 1 + 8 + 7, e4m3 1 + 4 + 3, e5m2 1 + 5 + 2, FP4 1 + 2 + 1) and fp32’s bias 127', () => {
	const split = (id: Parameters<typeof fl>[0]) => [fl(id).e, fl(id).m];
	assert.deepEqual(split('fp32'), [8, 23]);
	assert.deepEqual(split('fp16'), [5, 10]);
	assert.deepEqual(split('bf16'), [8, 7]);
	assert.deepEqual(split('e4m3'), [4, 3]);
	assert.deepEqual(split('e5m2'), [5, 2]);
	assert.deepEqual(split('fp4'), [2, 1]);
	assert.equal(fl('fp32').bias, 127);
	// e5m2 "gives one fraction bit to the exponent": the same 7 bits after the sign
	assert.equal(fl('e5m2').e + fl('e5m2').m, fl('e4m3').e + fl('e4m3').m);
	assert.equal(fl('e5m2').m, fl('e4m3').m - 1);
});

test('step "fields": value = (−1)^sign · 2^(E − 127) · (1 + M / 2^23) for a normal fp32 number, and the fp32 bits agree with the browser’s own float32 (DataView)', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const x = rawOf(rt.input).x;
			const r = rowOf('fp32', x);
			assert.equal(r.value, Math.fround(x));
			if (r.E !== 0) assert.equal((r.groups[0].bits === '1' ? -1 : 1) * 2 ** (r.E! - 127) * (1 + r.M! / 2 ** 23), r.value);
		},
		opts
	);
});

test('steps "fields", "gap" (look): pressing a bit sets x to the value of the flipped pattern; every row then holds that x', () => {
	for (const x of [0.1, -2.5, 70000, 1e-9]) {
		for (const id of ['fp32', 'fp16', 'e4m3', 'int8'] as const) {
			const code = flipBit(id, encode(id, x), 1);
			const v = decode(id, code);
			if (!Number.isFinite(v)) continue;
			assert.equal(encode(id, v), code, `${id} bit 1 of ${x}`);
		}
	}
});

test('step "gap": each power of 2 holds 2^23 fp32 numbers, so the gap is between 2^−24 and 2^−23 of a normal number; rounding is off by at most half a gap', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const x = rawOf(rt.input).x;
			const r = rowOf('fp32', x);
			if (r.E !== 0 && Number.isFinite(r.gap)) {
				const rel = r.gap / Math.abs(r.value);
				assert.ok(rel > 2 ** -24 && rel <= 2 ** -23, `relative gap ${rel}`);
			}
			const [lo, hi] = [Math.fround(x) - (Number.isFinite(r.gap) ? r.gap : 0), Math.fround(x) + (Number.isFinite(r.gap) ? r.gap : 0)];
			assert.ok(lo <= x && x <= hi);
			if (Number.isFinite(r.gap)) assert.ok(Math.abs(r.value - x) <= r.gap / 2 + 1e-300);
		},
		opts
	);
	// 2^23 numbers per power of 2: from 1 (code 0x3f800000) to 2 (0x40000000)
	assert.equal(encode('fp32', 2) - encode('fp32', 1), 2 ** 23);
	assert.equal(smallest('fp32'), 2 ** -149);
});

test('step "predict" (explain), prequestion: fp16 ends at 65,504, so 70,000 becomes infinity; bf16 stores 70,144 with steps of 512 there; fp16’s steps are 2^3 = 8 times finer', () => {
	assert.equal(BIG_X, 70000);
	assert.equal(largest('fp16'), 65504);
	assert.equal(cast('fp16', 70000), Infinity);
	assert.equal(cast('fp16', 65519), 65504);
	assert.equal(cast('fp16', 65520), Infinity);
	assert.equal(cast('bf16', 70000), 70144);
	assert.equal(rowOf('bf16', 70000).gap, 512);
	assert.equal(gapAtOne('bf16') / gapAtOne('fp16'), 8);
	// bf16 reaches about as far as fp32
	assert.ok(largest('bf16') / largest('fp32') > 0.99);
	// pinned PyTorch values (torch.tensor([...], dtype=float32).to(dtype))
	assert.equal(cast('fp16', 0.1), 0.0999755859375);
	assert.equal(cast('bf16', 0.1), 0.10009765625);
});

test('step "range16": fp16’s steps are 8 times finer wherever both are normal; the task is reachable by flipping fp32 bit 1 (the highest exponent bit) of the start x', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const x = rawOf(rt.input).x;
			const a = Math.abs(Math.fround(x));
			if (a >= smallestNormal('fp16') && a < 32768) {
				const r16 = rowOf('fp16', x);
				const rbf = rowOf('bf16', x);
				if (Math.floor(Math.log2(Math.abs(r16.value))) === Math.floor(Math.log2(Math.abs(rbf.value)))) assert.equal(rbf.gap / r16.gap, 8);
			}
		},
		opts
	);
	const y = decode('fp32', flipBit('fp32', encode('fp32', START.x), 1));
	assert.equal(cast('fp16', y), Infinity);
	assert.ok(Number.isFinite(cast('bf16', y)));
	assert.equal(rowOf('fp32', START.x).groups[1].bits.length, 8); // bits 1..8 are the exponent
});

test('step "fp8", quiz, glossary: e4m3 gap 0.125 at 1 and largest 448 (no infinity: PyTorch gives NaN above 464); e5m2 gap 0.25, largest 57,344, infinity; FP4 has 15 values, largest 6', () => {
	assert.equal(gapAtOne('e4m3'), 0.125);
	assert.equal(largest('e4m3'), 448);
	assert.equal(cast('e4m3', 464), 448);
	assert.ok(Number.isNaN(cast('e4m3', 470)));
	assert.equal(gapAtOne('e5m2'), 0.25);
	assert.equal(largest('e5m2'), 57344);
	assert.equal(cast('e5m2', 61439), 57344);
	assert.equal(cast('e5m2', 61440), Infinity);
	assert.equal(finiteCount('fp4'), 15);
	assert.equal(largest('fp4'), 6);
	const fp4 = new Set(Array.from({ length: 16 }, (_, c) => decode('fp4', c)).map(Math.abs));
	assert.deepEqual([...fp4].sort((a, b) => a - b), [0, 0.5, 1, 1.5, 2, 3, 4, 6]);
	// pinned PyTorch values
	assert.equal(cast('e4m3', 0.1), 0.1015625);
	assert.equal(cast('e5m2', 0.1), 0.09375);
	assert.equal(cast('e5m2', 470), 448);
	// overflow (glossary): fp32, fp16, bf16 and e5m2 give infinity, e4m3 NaN, FP4 and integer codes their largest value
	for (const id of ['fp16', 'e5m2'] as const) assert.equal(cast(id, 1e6), Infinity);
	assert.equal(cast('bf16', 3.4e38), Infinity);
	assert.equal(cast('fp4', 1e6), 6);
	assert.equal(cast('int8', 5), 1);
});

test('step "tiny", pitfall 2: below e4m3’s smallest normal 2^−6 the exponent field is 0 (no leading 1); x = 0.001 is stored as 2^−9, almost twice too large; FP4 stores 0', () => {
	assert.equal(TINY_X, 0.001);
	assert.equal(smallestNormal('e4m3'), 2 ** -6);
	const r = rowOf('e4m3', TINY_X);
	assert.equal(r.E, 0);
	assert.equal(r.status, 'subnormal');
	assert.equal(r.value, (r.M ?? NaN) * 2 ** (1 - 7 - 3));
	assert.equal(r.value, 2 ** -9);
	const ratio = r.value / TINY_X;
	assert.ok(ratio > 1.9 && ratio < 2, `ratio ${ratio}`);
	assert.equal(cast('fp4', TINY_X), 0);
	assert.equal(rowOf('fp4', TINY_X).status, 'underflow');
});

test('step "ints", glossary "nf4": int8 s = 1/127, int4 s = 1/7, even steps; NF4 has 16 levels from −1 to 1, finer near 0 than near ±1; x outside −1..1 is clipped to the end', async () => {
	for (const v of [-0.9, -0.3, 0, 0.2, 0.7]) {
		assert.ok(Math.abs(rowOf('int8', v).gap - 1 / 127) < 1e-15);
		assert.ok(Math.abs(rowOf('int4', v).gap - 1 / 7) < 1e-15);
	}
	assert.equal(NF4.length, 16);
	assert.equal(NF4[0], -1);
	assert.equal(NF4[15], 1);
	assert.equal(NF4[7], 0);
	const gaps = NF4.slice(1).map((v, i) => v - NF4[i]);
	assert.ok(Math.max(gaps[7], gaps[8]) < Math.min(gaps[0], gaps[14]));
	for (const v of [-2.5, 3e38, 70000]) for (const id of ['int8', 'int4', 'nf4'] as const) assert.equal(Math.abs(cast(id, v)), 1);
	// "where most weights lie": TinyGPT's weights, each block of 64 divided by its largest size, mostly land in NF4's
	// central levels (|y| below the 12th level, 0.338)
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('sweep', 'first');
			const W = rt.sources.gpt.model!.weights;
			let inner = 0;
			let n = 0;
			for (const k of matrixNames(W)) {
				const [rows, cols] = W[k].shape as number[];
				const d = W[k].data;
				for (let r = 0; r < rows; r++)
					for (let c0 = 0; c0 < cols; c0 += 64) {
						let amax = 0;
						for (let c = c0; c < Math.min(cols, c0 + 64); c++) amax = Math.max(amax, Math.abs(d[r * cols + c]));
						for (let c = c0; c < Math.min(cols, c0 + 64); c++) {
							n++;
							if (amax > 0 && Math.abs(d[r * cols + c]) / amax < NF4[11]) inner++;
						}
					}
			}
			assert.ok(inner / n > 0.5, `share ${inner / n}`);
		},
		{ ...opts, only: ['start'] }
	);
});

test('steps "weights", "cast", pitfall 2: FP4 without a scale rounds most weights to 0 and wrecks TinyGPT; a scale per block brings it back; bf16 and fp16 barely change it; every scale adds bits per weight; concept "real": e4m3 with and without a per-matrix scale differ by noise', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const { at, cast: c } = await sweepOf(rt);
			const f0 = at('fp32').metrics!;
			assert.equal(f0.nm, 24);
			assert.ok(f0.zero_fp4 > 0.5, `FP4 zero share ${f0.zero_fp4}`);
			assert.ok(c('fp4').ppl > 10 * f0.ppl0, `FP4 ${c('fp4').ppl}`);
			assert.ok(c('fp4-b').ppl < 2 * f0.ppl0 && c('fp4-b').ppl < c('fp4').ppl / 10);
			assert.ok(Math.abs(c('bf16').ppl - f0.ppl0) < 0.01 * f0.ppl0);
			assert.ok(Math.abs(c('fp16').ppl - f0.ppl0) < 0.01 * f0.ppl0);
			for (const k of CASTS) if (k.scale !== 'none') assert.ok(c(k.id).bpw > c(k.id).bits);
			// the PyTorch run on 64,000 tokens agrees on the story
			assert.ok(fact('pplFp4') > 10 * fact('bigPpl0') && fact('pplFp4B') < 2 * fact('bigPpl0'));
			assert.ok(fact('pplNf4B') < fact('pplInt4B'));
			// concept "real": on 256 tokens the two e4m3 settings differ by a few hundredths; on 64,000 tokens by less than 0.01
			assert.ok(Math.abs(c('e4m3-t').ppl - c('e4m3').ppl) < 0.03 * f0.ppl0, `e4m3 ${c('e4m3').ppl} vs ${c('e4m3-t').ppl}`);
			assert.ok(Math.abs(fact('pplE4m3T') - fact('pplE4m3')) < 0.01);
		},
		{ ...opts, only: ['start'] }
	);
});

test('castArray: a per-block scale maps each block’s largest size to the format’s largest value, in float32', () => {
	const w = Float32Array.from([0.01, -0.02, 0.03, 0.5, -0.001, 0.002, 0.004, -0.008]);
	const out = castArray(w, { id: 't', fmt: 'fp4', scale: 'block', block: 4 }, 8);
	assert.equal(out[3], Math.fround(0.5));
	assert.equal(out[7], Math.fround(-0.008));
	const none = castArray(w, { id: 't', fmt: 'fp4', scale: 'none' }, 8);
	assert.ok(Array.from(none).every((v, i) => (i === 3 ? v === 0.5 : v === 0)));
});

test('step "mixed", quiz 2, pitfall 3: an update of 0.001 is less than half the bf16 gap at 1, so the bf16 weight never moves; the fp32 master collects the updates; fp32’s gaps are 65,536 times finer than bf16’s', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('sum', 'last');
			const t = rt.sources.sum.tape!;
			const first = t.frame(0).metrics!;
			const last = t.frame(t.length - 1).metrics!;
			assert.equal(first.fp32, 1);
			assert.ok(cast('bf16', 0.001) < gapAtOne('bf16') / 2);
			for (let i = 0; i < t.length; i++) assert.equal(t.frame(i).metrics!.bf16, 1);
			assert.ok(last.fp32 > 1.09 && last.mixed > 1.09 && last.fp16 > 1.09);
			assert.equal(gapAtOne('bf16') / gapAtOne('fp32'), 65536);
		},
		{ ...opts, only: ['start'] }
	);
});

test('step "limits", pitfall 1, glossary "loss-scaling": x = 500 gives NaN in e4m3 (and NaN spreads through a sum), FP4 stays at 6; loss scaling lowers the share of gradients lost in fp16 without overflowing; fp16 loses small gradients, bf16 none', () => {
	assert.equal(HUGE_X, 500);
	assert.ok(HUGE_X > largest('e4m3'));
	assert.ok(Number.isNaN(cast('e4m3', HUGE_X)));
	assert.ok(Number.isNaN(cast('e4m3', HUGE_X) + 1));
	assert.equal(cast('fp4', HUGE_X), 6);
	assert.ok(fact('gradZero_fp16') > 0);
	assert.equal(fact('gradZero_bf16'), 0);
	assert.ok(fact('gradZeroScaled_fp16') < fact('gradZero_fp16'));
	assert.equal(fact('gradScale'), 1024);
	assert.ok(fact('gradMaxScaled') < largest('fp16'));
});

test('glossary "ulp": a float’s gap doubles at every power of 2', () => {
	for (const id of ['fp32', 'fp16', 'bf16', 'e4m3', 'e5m2'] as const) assert.equal(rowOf(id, 2).gap, 2 * rowOf(id, 1).gap);
});
