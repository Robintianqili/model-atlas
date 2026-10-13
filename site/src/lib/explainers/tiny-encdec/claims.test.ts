// tiny-encdec claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, over every fixture (the lesson's 8, the adapter's empty / one / long, the 5 presets)
// through withFixtures. `atlas check` runs this file (AK408). Alone: node --test src/lib/explainers/tiny-encdec/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { dataset, readDate } from './data.ts';
import { avgHeads, crossAt, head, look, rightward, ROW, tr } from './look.ts';
import { argmax, encode, f64, weightsOf } from './tf.ts';
import { TRIES } from './views/cross/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tiny-encdec', SITE).then((r) => r.pkg);
const facts = JSON.parse(readFileSync(join(SITE, 'static/models/tiny-encdec/facts.json'), 'utf8')).facts as Record<string, { value: number | string }>;
const fact = (k: string) => Number(facts[k].value);
/** a fixture written in one of the training formats */
const trained = (rt: ExplainerRuntime) => look(rt).fmt !== null;
const at = async (text: string) => createRuntime(await pkgP, { ...opts, input: text });

test('the data: Python trained on exactly the pairs data.ts generates (the same sha256 as tinyseq2seq’s)', () => {
	const pairs = dataset(1, fact('trainPairs'));
	const sha = createHash('sha256').update(pairs.map((p) => `${p.src}\t${p.tgt}\n`).join('')).digest('hex');
	assert.equal(sha, facts.dataSha.value);
	const gru = JSON.parse(readFileSync(join(SITE, 'static/models/tinyseq2seq/facts.json'), 'utf8')).facts;
	assert.equal(sha, gru.dataSha.value, 'the same training pairs as tinyseq2seq, so the two can be compared');
});

test('step "chars" and the misconception: attention by itself ignores order; only the position vectors tell positions apart', async () => {
	const rt = await at('March 5, 2021');
	const w = weightsOf(rt.model!);
	const { D, H, F } = w.dec;
	const ids = Array.from(rt.pre!.tensors.input.data);
	const rev = ids.slice().reverse();
	const still = { ...w.enc, pos: new Float64Array(w.enc.pos.length) };
	const a = encode(still, ids, D, H, F, f64).mem;
	const b = encode(still, rev, D, H, F, f64).mem;
	const T = ids.length;
	for (let i = 0; i < T; i++) for (let d = 0; d < D; d++) assert.ok(Math.abs(a[i * D + d] - b[(T - 1 - i) * D + d]) < 1e-9, 'without positions, reversing the text only reverses the rows');
	const c = encode(w.enc, rev, D, H, F, f64).mem;
	assert.ok(Math.abs(a[0] - c[(T - 1) * D]) > 1e-3, 'with positions it does not');
});

test('step "chars": a character the model does not know becomes ? (id 1)', async () => {
	assert.equal(look(await at('5 März 2021')).chars[3], '?');
});

test('step "self": self-attention reads in both directions (weight on later characters) on every fixture longer than one character', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (look(rt).T < 2) return;
		const R = rightward(rt);
		assert.ok(R.share > 0.05 && R.j > R.i, `${look(rt).text}: ${R.share}`);
	}, opts);
});

test('step "dself": the causal mask: decoder position t gives exactly 0 to later positions; the decoder reads 11 positions', async () => {
	await withFixtures(await pkgP, (rt) => {
		const A = tr(rt, 'dec.sattn');
		if (!A) return;
		const [H, L] = A.shape;
		assert.equal(L, 11);
		for (let h = 0; h < H; h++) for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) assert.equal(A.data[(h * L + i) * L + j], 0);
	}, opts);
});

test('step "memory": the decoder’s cross-attention covers every character of the memory at every step', async () => {
	await withFixtures(await pkgP, (rt) => {
		const C = tr(rt, 'dec.cattn');
		if (!C) return;
		assert.deepEqual(C.shape.slice(1), [11, look(rt).T]);
	}, opts);
});

test('step "cross" (predict): the row of the year’s last digit reads that very digit (fixtures and every test date)', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!trained(rt)) return;
		const L = look(rt);
		const y = /\d{4}/.exec(L.chars.join(''))!.index;
		const a = avgHeads(tr(rt, 'dec.cattn'));
		assert.equal(argmax(a.subarray(ROW * L.T, (ROW + 1) * L.T)), y + 3, L.text);
	}, opts);
	assert.equal(fact('year3CopyN'), fact('testN'), 'on every test date');
	assert.ok(fact('monthGapN') > 0, 'the month’s row may read a space or comma');
	const rt = await at('March 5, 2021');
	const a = avgHeads(tr(rt, 'dec.cattn'));
	const T = look(rt).T;
	assert.equal(look(rt).chars[argmax(a.subarray(6 * T, 7 * T))], ' ', 'on March 5, 2021 the month’s last digit reads a space');
});

test('step "score": the character with the largest weight has the largest score (the others score lower)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const C = crossAt(rt, ROW);
		const S = head(tr(rt, 'dec.cscores'), C.h, ROW);
		if (!S) return;
		assert.equal(argmax(S), C.j);
	}, opts);
});

test('steps "next" and "decode": greedy decoding feeds the chosen symbol back in; the parallel pass and the step-by-step process agree', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const L = look(rt);
		if (!L.T) return;
		for (let t = 0; t + 1 < L.yin.length; t++) assert.equal(L.yin[t + 1], L.ids[t], `${L.text}: position ${t + 1} reads what step ${t} wrote`);
		await rt.ensure('decode', 'last');
		const tape = rt.sources.decode.tape!;
		const last = tape.state(tape.length - 1) as { cross: Tensor; out: Tensor };
		const n = L.end >= 0 ? L.end + 1 : L.ids.length;
		assert.deepEqual(Array.from(last.out.data).slice(0, n), L.ids.slice(0, n));
		const W = tr(rt, 'dec.cattn')!;
		const [H, R, T] = W.shape;
		for (let h = 0; h < H; h++) for (let i = 0; i < n * T; i++) assert.ok(Math.abs(last.cross.data[h * R * T + i] - W.data[h * R * T + i]) < 1e-5);
	}, opts);
});

test('steps "yours" and "limits": right on every fixture in a training format; on formats it never saw it still answers, and fails where the GRU does not', async () => {
	await withFixtures(await pkgP, (rt) => {
		const L = look(rt);
		if (trained(rt)) assert.ok(L.right, `“${L.text}” -> ${L.written}`);
	}, opts);
	const bad = look(await at(TRIES[4]));
	assert.equal(bad.written, '2021-03-32', 'a day that does not exist: the digits are copied, no doubt expressed');
	assert.equal(readDate(TRIES[4]).valid, false);
	for (const k of ['offYearFirst', 'offDashes', 'offNoComma']) assert.ok(fact(k) < 0.9, `${k} is not mastered`);
	assert.ok(fact('offImpossible') > 0.9, 'it copies an impossible day nearly always');
	assert.ok(fact('gruOffNoComma') - fact('offNoComma') > 0.2 && fact('offImpossible') - fact('gruOffImpossible') > 0.2, 'the two fail in different places');
	assert.ok(fact('testExact') >= 0.99 && fact('gruTestExact') >= 0.99);
});

test('step "limits": the five buttons are the formats of off.json (make_data.ts), none of them a training format', () => {
	for (const x of TRIES) assert.equal(readDate(x).fmt, null, x);
});
