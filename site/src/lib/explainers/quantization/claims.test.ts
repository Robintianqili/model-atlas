// quantization claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept page
// (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the lesson's fixtures,
// the adapter's edge fixtures and the start input) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/quantization/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { biggest, flipFp32, formatsOf, fp16Bits, fromFp16Bits, groupCount, hasOutlier, ledger, medianAbs, OUTLIER_MAKE, quantizeMatrix, qmaxOf, rawOf, START, withOutlier } from './quant.ts';
import { FOCUS, matrixNames } from './process.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'quantization';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', SLUG, 'facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const fact = (k: string) => facts[k].value;

/** Every frame of the sweep, by phase and step. */
async function sweepOf(rt: ExplainerRuntime) {
	await rt.ensure('sweep', 'last');
	const t = rt.sources.sweep.tape!;
	const at = (phase: string, step = 0): Frame => t.frame(t.find({ phase, step }));
	return { t, at };
}

test('step "formats": fp32 has 1 + 8 + 23 bits, fp16 1 + 5 + 10, bf16 1 + 8 + 7 (the eight exponent bits of fp32, seven fraction bits)', () => {
	const rows = formatsOf(0.1);
	const lens = (id: string) => rows.find((r) => r.id === id)!.groups.map((g) => g.bits.length);
	assert.deepEqual(lens('fp32'), [1, 8, 23]);
	assert.deepEqual(lens('fp16'), [1, 5, 10]);
	assert.deepEqual(lens('bf16'), [1, 8, 7]);
	assert.deepEqual(lens('int8'), [8]);
	assert.deepEqual(lens('int4'), [4]);
});

test('step "formats": fp16 rounds to nearest (ties to even), overflows to infinity above its largest value, so 70000 becomes infinity; bf16 holds it', () => {
	assert.equal(fromFp16Bits(fp16Bits(65504)), 65504);
	assert.equal(fromFp16Bits(fp16Bits(70000)), Infinity);
	assert.equal(fromFp16Bits(fp16Bits(1 + 2 ** -11)), 1); // a tie rounds to the even fraction
	assert.equal(fromFp16Bits(fp16Bits(1 + 3 * 2 ** -11)), 1 + 2 ** -9);
	assert.equal(fromFp16Bits(fp16Bits(2 ** -24)), 2 ** -24); // the smallest subnormal
	const f16round = (Math as unknown as { f16round?: (v: number) => number }).f16round;
	if (f16round) for (const v of [0.1, 1 / 3, 1e-6, 12345.678, -0.7, 65519, 3e-8]) assert.equal(fromFp16Bits(fp16Bits(Math.fround(v))), f16round(Math.fround(v)), `fp16 of ${v}`);
	const bf = formatsOf(70000).find((r) => r.id === 'bf16')!.value;
	assert.ok(Number.isFinite(bf) && Math.abs(bf - 70000) / 70000 < 2 ** -7, `bf16 of 70000 = ${bf}`);
});

test('step "formats" (look): flipping an exponent bit multiplies or divides x by a power of 2; a fraction bit changes it by less than a factor of 2', () => {
	for (const x of [0.1, 1 / 3, -2.5, 70000]) {
		for (let b = 2; b <= 8; b++) {
			const y = flipFp32(x, b);
			if (!Number.isFinite(y) || y === 0) continue;
			const r = Math.log2(Math.abs(y / Math.fround(x)));
			assert.ok(Math.abs(r - Math.round(r)) < 1e-9 && Math.round(r) !== 0, `exponent bit ${b} of ${x}: ratio 2^${r}`);
		}
		for (let b = 9; b < 32; b++) {
			const ratio = flipFp32(x, b) / Math.fround(x);
			assert.ok(ratio > 0.5 && ratio < 2, `fraction bit ${b} of ${x}: ratio ${ratio}`);
		}
	}
});

test('steps "int8", "predict", "int4", "outlier": absmax gives the largest value the top code, every unclipped value is at most half a step from its code, and the int4 step is 127 / 7 times the int8 step', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const vals = rawOf(rt.input).vals;
			for (const bits of [8, 4]) {
				const L = ledger(vals, { bits });
				const s = L.blocks[0].s;
				const k = vals.reduce((m, v, i) => (Math.abs(v) > Math.abs(vals[m]) ? i : m), 0);
				if (vals[k] !== 0) assert.equal(Math.abs(L.q[k]), qmaxOf(bits));
				L.err.forEach((e, i) => assert.ok(L.clipped.includes(i) || Math.abs(e) <= s / 2 + 1e-12, `value ${i}: error ${e}, s ${s}`));
			}
			if (vals.some((v) => v !== 0)) {
				const r = ledger(vals, { bits: 4 }).blocks[0].s / ledger(vals, { bits: 8 }).blocks[0].s;
				assert.ok(Math.abs(r - 127 / 7) < 1e-9, `ratio ${r}`);
			}
		},
		opts
	);
});

test('step "affine": on the sizes |x| no absmax code is negative (its negative codes are wasted); the range starts at 0, so the zero point is the lowest code, and it stands for 0', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const a = rawOf(rt.input).vals.map(Math.abs);
			assert.ok(ledger(a, { bits: 4 }).q.every((q) => q >= 0));
			const A = ledger(a, { bits: 4, scheme: 'affine' });
			assert.equal(A.blocks[0].lo, 0);
			assert.equal(A.blocks[0].z, A.blocks[0].qlo);
			assert.equal(A.blocks[0].qhi - A.blocks[0].qlo + 1, 16);
			// a value of 0 gets the code z and is read back as exactly 0
			const Z = ledger([...a.slice(0, 7), 0], { bits: 4, scheme: 'affine' });
			assert.equal(Z.q[7], Z.blocks[0].z);
			assert.equal(Z.xh[7], 0);
		},
		opts
	);
});

test('steps "outlier", "fixes": one scale is set by the largest value; with blocks of four, the block without the outlier keeps its own finer scale; the clipped outlier is stored at the clipping range; a second scale costs bits', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const vals = withOutlier(rawOf(rt.input).vals);
			assert.ok(hasOutlier(vals), `withOutlier gave no outlier: ${vals}`);
			const k = vals.reduce((m, v, i) => (Math.abs(v) > Math.abs(vals[m]) ? i : m), 0);
			const one = ledger(vals, { bits: 4 });
			assert.equal(one.blocks[0].s, Math.abs(vals[k]) / 7);
			const B = ledger(vals, { bits: 4, block: 4 });
			const other = B.blocks[k < 4 ? 1 : 0];
			const zeros = vals.slice(other.from, other.to).every((v) => v === 0); // a block of zeros needs no scale (it keeps s = 1)
			assert.ok(zeros || other.s <= B.blocks[k < 4 ? 0 : 1].s, 'the other block has the finer scale');
			const C = ledger(vals, { bits: 4, clip: 'second' });
			assert.equal(Math.abs(C.xh[k]), C.blocks[0].hi);
			// the price: one more 32-bit scale for the eight values
			assert.equal(B.bitsPerValue - C.bitsPerValue, 32 / 8);
			assert.equal(C.bitsPerValue, 4 + 32 / 8);
			// without an outlier of its own, the step moves the largest value out to OUTLIER_MAKE times the median size
			const raw = rawOf(rt.input).vals;
			if (!hasOutlier(raw)) {
				assert.equal(k, biggest(raw));
				vals.forEach((v, i) => i !== k && assert.equal(v, raw[i]));
				if (medianAbs(raw) > 0) assert.ok(Math.abs(Math.abs(vals[k]) - OUTLIER_MAKE * medianAbs(raw)) < 1e-12);
			} else assert.deepEqual(vals, raw);
		},
		opts
	);
	// the task's preset makes an outlier; the start values have none
	assert.ok(hasOutlier([0.31, -0.12, 0.05, -0.47, 0.22, 0.09, -0.26, 6]));
	assert.ok(!hasOutlier(START.vals));
});

test('steps "layer", "granularity", "whole", "limits", concept "real"/"compare" and the pitfalls: TinyGPT under quantization', async () => {
	const p = await pkg();
	let checked = 0;
	await withFixtures(
		p,
		async (rt) => {
			if (checked++) return; // the sweep ignores the learner's input: one runtime is enough
			const { t, at } = await sweepOf(rt);
			assert.equal(t.length, 11);
			const fp = at('fp32').metrics!;
			// "layer": most weights sit near 0 (over 90% within a quarter of the largest size)
			const w = rt.sources.gpt.model!.weights[FOCUS].data;
			let amax = 0;
			for (let i = 0; i < w.length; i++) amax = Math.max(amax, Math.abs(w[i]));
			let near = 0;
			for (let i = 0; i < w.length; i++) if (Math.abs(w[i]) < amax / 4) near++;
			assert.ok(near / w.length > 0.9, `near 0: ${near / w.length}`);
			// "layer": int8 on one layer moves perplexity by under 1%, too little to read on 256 tokens; PyTorch on 64,000 agrees
			const l8 = at('layer', 0).metrics!;
			assert.ok(Math.abs(l8.ppl - l8.ppl0) / l8.ppl0 < 0.01, `live int8 layer: ${l8.ppl} vs ${l8.ppl0}`);
			assert.ok(Math.abs(fact('pplLay8t') - fact('bigPpl0')) / fact('bigPpl0') < 0.01);
			// "granularity": per channel and per block beat one scale for the layer, in perplexity and KL
			const [lT, lC, lB] = [at('layer', 1), at('layer', 2), at('layer', 3)].map((f) => f.metrics!);
			assert.ok(lT.ppl > fp.ppl0 && lC.ppl < lT.ppl && lB.ppl < lT.ppl && lC.kl < lT.kl && lB.kl < lT.kl);
			assert.ok(lC.wRms < lT.wRms && lB.wRms < lT.wRms, 'smaller groups fit their weights more closely');
			// "whole": int4 per block costs more KL than int8 per channel; not eight times smaller; the embedding stays fp32
			const all = [0, 1, 2, 3, 4, 5].map((k) => at('all', k).metrics!);
			assert.ok(all[5].kl > all[1].kl);
			assert.ok(all[5].bytes0 / all[5].bytes < 8 && all[3].bytes0 / all[3].bytes < 8);
			const W = rt.sources.gpt.model!.weights;
			assert.equal(all[0].nq, matrixNames(W).reduce((a, n) => a + W[n].data.length, 0));
			assert.ok(!matrixNames(W).includes('tok_emb.weight'));
			assert.equal(all[0].groups, matrixNames(W).length);
			assert.equal(all[5].groups, matrixNames(W).reduce((a, n) => a + groupCount(W[n].shape[0], W[n].shape[1], 'block'), 0));
			// look of "whole": every smaller file in the table pays in KL
			const bySize = all.slice().sort((a, b) => a.bytes - b.bytes);
			for (let i = 1; i < bySize.length; i++) assert.ok(bySize[i].kl < bySize[i - 1].kl, `size order vs KL at ${i}`);
			// hook: several times smaller (the smallest setting is over twice as small)
			assert.ok(fp.bytes0 / bySize[0].bytes > 2);
			// "limits": int4 per tensor breaks the model (live and in PyTorch); the probe's top token changes
			assert.ok(all[3].ppl > 2 * fp.ppl0 && fact('pplAll4t') > 2 * fact('bigPpl0'));
			assert.notEqual(all[3].probeTop, all[3].probeTop0);
			// manifest prequestion: one scale per matrix is much worse than one per row or block
			assert.ok(all[3].kl > 3 * all[4].kl && all[4].kl > all[5].kl);
			// pitfall 3: on a short text, perplexity can even improve by chance while KL and top-1 show the change
			const lucky = [...[0, 1, 2, 3].map((k) => at('layer', k).metrics!), ...all].filter((m) => m.ppl < m.ppl0);
			assert.ok(lucky.length > 0 && lucky.every((m) => m.kl > 0 && m.agree < 1));
			// on 64,000 tokens, int8 per tensor is worse than fp32 (the improvement above is chance)
			assert.ok(fact('pplAll8t') > fact('bigPpl0'));
		},
		opts
	);
});

test('step "limits": each code is turned back into a float (q × s) before the forward pass multiplies it', () => {
	const w = Float32Array.from([0.5, -0.25, 0.125, 0.9, -0.01, 0.3]);
	const q = quantizeMatrix(w, 2, 3, 4, 'channel');
	for (let r = 0; r < 2; r++) {
		const row = Array.from(w.subarray(r * 3, r * 3 + 3));
		const s = Math.fround(Math.max(...row.map(Math.abs)) / 7);
		for (let c = 0; c < 3; c++) {
			const v = q[r * 3 + c] / s;
			assert.ok(Math.abs(v - Math.round(v)) < 1e-5 && Math.abs(Math.round(v)) <= 7, `code ${v}`);
		}
	}
});
