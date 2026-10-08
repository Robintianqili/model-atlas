// tinyseq2seq claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not
// computed live is asserted here, over every fixture (the lesson's 8, the adapter's empty / one / long, the 5 presets)
// through withFixtures. `atlas check` runs this file (AK408). Alone: node --test src/lib/explainers/tinyseq2seq/claims.test.ts
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
import { look, row, stepAt, tr } from './look.ts';
import { argmax, run64, weightsOf } from './seq.ts';
import { MONTH_ROW } from './lesson.ts';
import { TRIES } from './views/align/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinyseq2seq', SITE).then((r) => r.pkg);
const facts = JSON.parse(readFileSync(join(SITE, 'static/models/tinyseq2seq/facts.json'), 'utf8')).facts as Record<string, { value: number | string }>;
const fact = (k: string) => Number(facts[k].value);
/** a fixture written in one of the training formats */
const trained = (rt: ExplainerRuntime) => look(rt).fmt !== null;
const at = async (text: string) => {
	const rt = await createRuntime(await pkgP, { ...opts, input: text });
	return look(rt);
};

test('the data: Python trained on exactly the pairs data.ts generates (sha256 in facts.json, from make_data.ts)', () => {
	const pairs = dataset(1, fact('trainPairs'));
	const sha = createHash('sha256').update(pairs.map((p) => `${p.src}\t${p.tgt}\n`).join('')).digest('hex');
	assert.equal(sha, facts.dataSha.value);
	for (const p of pairs.slice(0, 500)) assert.ok(readDate(p.src).fmt, `readDate knows the format of “${p.src}”`);
});

test('step "chars": a character the model does not know becomes ? (id 1)', async () => {
	const L = await at('5 März 2021');
	assert.equal(L.chars[3], '?');
});

test('steps "start" and "read": the decoder starts from the encoder’s last state; every fixture gets 11 decoder steps', async () => {
	await withFixtures(await pkgP, (rt) => {
		const L = look(rt);
		if (!L.T) return;
		const enc = tr(rt, 'enc')!;
		const wq = rt.model!.weights['dec.wq'];
		const H = enc.shape[1];
		const q0 = row(tr(rt, 'dec#query'), 0)!;
		for (let a = 0; a < wq.shape[0]; a++) {
			let s = 0;
			for (let k = 0; k < H; k++) s += wq.data[a * H + k] * enc.data[(L.T - 1) * H + k];
			assert.ok(Math.abs(s - q0[a]) < 1e-5, 'query 0 = W_q · the last encoder state');
		}
		assert.equal(L.ids.length, 11);
	}, opts);
});

test('step "scores" (predict): for the year’s first digit, the top weight is on a year digit, usually the second', async () => {
	let n = 0;
	let second = 0;
	await withFixtures(await pkgP, (rt) => {
		if (!trained(rt)) return;
		const S = stepAt(rt, 0);
		assert.ok(S.year && S.j >= S.year[0] && S.j < S.year[1], `top weight on a year digit (“${S.text}”: position ${S.j})`);
		n++;
		if (S.j === S.year![0] + 1) second++;
	}, opts);
	assert.ok(n >= 10 && second / n > 0.5, `the year’s second digit on ${second} of ${n}`);
	assert.ok(fact('yearInside') > 0.99 && fact('yearSecond') > 0.5, 'on the test dates too (facts.json)');
	assert.equal(fact('yearSecondN'), Math.round(fact('yearSecond') * fact('testN')));
});

test('step "softmax": nearly all the weight sits on a few characters (the top three hold over 90%)', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!trained(rt)) return;
		const a = Array.from(stepAt(rt, 0).a!).sort((x, y) => y - x);
		assert.ok(a[0] + a[1] + a[2] > 0.9, `${look(rt).text}: ${a.slice(0, 3)}`);
	}, opts);
});

test('step "align": the decode process (float64) writes what the trace writes, with the same weights within 1e-5', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const L = look(rt);
		if (!L.T) return;
		await rt.ensure('decode', 'last');
		const tape = rt.sources.decode.tape!;
		const last = tape.state(tape.length - 1) as { align: Tensor; out: Tensor };
		const n = L.end >= 0 ? L.end + 1 : L.ids.length;
		assert.deepEqual(Array.from(last.out.data).slice(0, n), L.ids.slice(0, n));
		const W = tr(rt, 'dec#weights')!;
		for (let i = 0; i < n * L.T; i++) assert.ok(Math.abs(last.align.data[i] - W.data[i]) < 1e-5);
		const r = run64(weightsOf(rt.model!), Array.from(rt.pre!.tensors.input.data));
		assert.equal(argmax(r.steps[0].a), stepAt(rt, 0).j);
	}, opts);
});

test('step "align" and try-this "March or May?": the month’s last digit is read where the encoder has read (enough of) the month', async () => {
	await withFixtures(await pkgP, (rt) => {
		const L = look(rt);
		const d = readDate(L.text);
		if (!trained(rt) || !d.ymd) return;
		const S = stepAt(rt, MONTH_ROW);
		const low = L.text.toLowerCase();
		// the month's name and the character after it, or the month's number and the separator after it
		const name = ['us', 'eu'].includes(L.fmt!) ? null : /(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*/.exec(low);
		const span = name ? [name.index, name.index + name[0].length + 1] : L.fmt === 'us' ? [0, low.indexOf('/') + 1] : [low.indexOf('.') + 1, low.lastIndexOf('.') + 1];
		assert.ok(S.j >= span[0] && S.j < span[1], `“${L.text}”: position ${S.j}, the month ends at ${span[1] - 1}`);
	}, opts);
});

test('steps "yours" and "limits": right on every fixture in a training format; on formats it never saw it still answers', async () => {
	await withFixtures(await pkgP, (rt) => {
		const L = look(rt);
		if (trained(rt)) assert.ok(L.right, `“${L.text}” -> ${L.written}`);
	}, opts);
	const bad = await at('March 32, 2021');
	assert.match(bad.written, /^\d{4}-\d{2}-\d{2}$/, 'a day that does not exist still gets a date');
	assert.ok(fact('offNoComma') > 0.9, 'a missing comma hardly matters');
	for (const k of ['offYearFirst', 'offDashes', 'offSpelled']) assert.ok(fact(k) < 0.5, `${k} is low`);
	assert.equal(fact('testExact') >= 0.99, true);
});

test('step "limits" look: on “March fifth, 2021” most rows still put most of their weight on one character, yet the answer is wrong', async () => {
	const rt = await createRuntime(await pkgP, { ...opts, input: TRIES[3] });
	const L = look(rt);
	assert.ok(L.want && !L.right, `“${L.text}” -> ${L.written}, right: ${L.want}`);
	const n = L.end >= 0 ? L.end + 1 : L.ids.length;
	const sharp = Array.from({ length: n }, (_, t) => Math.max(...stepAt(rt, t).a!)).filter((w) => w >= 0.5).length;
	assert.ok(sharp > n / 2, `${sharp} of ${n} rows put at least half their weight on one character`);
});

test('step "softmax" (the stage): subtracting the largest score does not change the weights', () => {
	const e = [5.34, -2.34, -17, 0.5];
	const w = (x: number[], c: number) => x.map((v) => Math.exp(v - c) / x.reduce((s, u) => s + Math.exp(u - c), 0));
	w(e, 0).forEach((v, i) => assert.ok(Math.abs(v - w(e, Math.max(...e))[i]) < 1e-12));
});

test('step "bottleneck": without attention the same-size model still works on dates; squeezed to 8 numbers, attention wins by far', () => {
	assert.ok(fact('noAttnExact') > 0.95, 'still works');
	assert.ok(fact('h8AttnExact') > fact('h8NoAttnExact') + 0.5, 'only … with attention …');
});
