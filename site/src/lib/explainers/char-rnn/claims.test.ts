// char-rnn claim tests: every sentence of the lesson and the manifest that is not computed live is asserted here, on
// every fixture where it depends on the input (lesson fixtures, the adapter's edge fixtures and the presets).
// `atlas check` runs this file (AK408 on failure).   node --test src/lib/explainers/char-rnn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { sample } from './process.ts';
import { rnnInput, scoresAfter, softmax, weightsOf } from './rnn.ts';
import { REF_GATE, charsOf, isLetter, riskiest, timelineOf, unitOf, words } from './views/timeline/data.ts';
import { normalize } from './adapters/text-presets/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('char-rnn', SITE).then((l) => l.pkg);
const tr = (rt: AtlasRuntime, id: string): Tensor => rt.sources.fwd.trace!.t[id];
const each = async (fn: (rt: AtlasRuntime) => void | Promise<void>) => withFixtures(await pkgP, async (rt) => (charsOf(rt).length ? fn(rt) : undefined), opts);

test('manifest and steps 1, 2, 6, 7: the sizes in the text (64 numbers per list, 74 characters, 24-number embeddings, one LSTM layer)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const spec = rt.spec!;
		const lstm = spec.layers.filter((l) => l.op === 'lstm');
		assert.equal(lstm.length, 1, 'a single LSTM cell (step 8)');
		assert.equal((lstm[0].params as { hidden: number }).hidden, 64, 'prequestion: two lists of 64 numbers');
		assert.equal((spec as unknown as { alphabet: string[] }).alphabet.length, rt.num('fact:vocab'));
		assert.equal(spec.tensors.find((t) => t.name === 'emb.weight')!.shape[1], 24);
	}, { ...opts, only: ['preset:lily'] });
});

test('steps 3-6: every gate is between 0 and 1 and every candidate between −1 and 1; softmax rows add up to 1 (step 7)', async () => {
	await each((rt) => {
		for (const k of ['lstm#f', 'lstm#i', 'lstm#o']) for (const v of tr(rt, k).data) assert.ok(v >= 0 && v <= 1, `${k} ${v}`);
		for (const v of tr(rt, 'lstm#g').data) assert.ok(v >= -1 && v <= 1, `g ${v}`);
		const p = tr(rt, 'probs');
		const V = p.shape[1];
		for (let t = 0; t < p.shape[0]; t++) {
			let s = 0;
			for (let k = 0; k < V; k++) s += p.data[t * V + k];
			assert.ok(Math.abs(s - 1) < 1e-5, `row ${t} sums to ${s}`);
		}
		// step 7: the chances of any two characters differ by the factor e^(difference of their scores)
		const sc = tr(rt, 'head');
		const r = (p.shape[0] - 1) * V;
		for (const [a, b] of [[1, 2], [0, 5], [3, 40]]) assert.ok(Math.abs(p.data[r + a] / p.data[r + b] / Math.exp(sc.data[r + a] - sc.data[r + b]) - 1) < 1e-4);
	});
});

test('step 8: the unrolled run (one step at a time, float64) is the same LSTM as the trace, at every character', async () => {
	await each(async (rt) => {
		await rt.ensure('unroll', 'last');
		const tape = rt.sources.unroll.tape!;
		const h = tr(rt, 'lstm');
		const H = h.shape[1];
		assert.equal(tape.length, 4 * h.shape[0]);
		let max = 0;
		for (let t = 0; t < h.shape[0]; t++) {
			const s = tape.state(tape.find({ phase: 'hidden', step: t })) as { h: Tensor };
			for (let j = 0; j < H; j++) max = Math.max(max, Math.abs(s.h.data[j] - h.data[t * H + j]));
		}
		assert.ok(max < 1e-5, `max |diff| ${max}`);
	});
});

test('step 9: unit 51 usually rises inside a word and is mostly wiped at a space after a word (as its measured facts say)', async () => {
	await each((rt) => {
		const ch = charsOf(rt);
		const U = unitOf(rt);
		const H = tr(rt, 'lstm').shape[1];
		const c = (k: number) => tr(rt, 'lstm#c').data[k * H + U];
		const f = (k: number) => tr(rt, 'lstm#f').data[k * H + U];
		let rises = 0;
		let inword = 0;
		let wiped = 0;
		let spaces = 0;
		for (let k = 1; k < ch.length; k++) {
			if (isLetter(ch[k]) && isLetter(ch[k - 1])) (inword++, (rises += c(k) > c(k - 1) ? 1 : 0));
			if (ch[k] === ' ' && isLetter(ch[k - 1])) (spaces++, (wiped += f(k) < 0.5 ? 1 : 0));
		}
		if (inword >= 5) assert.ok(rises / inword >= 0.6, `rises at ${rises} of ${inword} letters`);
		if (spaces >= 3) assert.ok(wiped / spaces >= 2 / 3, `forget gate < 0.5 at ${wiped} of ${spaces} spaces`);
	});
	const pkg = await pkgP;
	await withFixtures(pkg, (rt) => {
		assert.equal(unitOf(rt), 51);
		assert.ok(rt.num('fact:counterR') >= 0.5, 'its value follows the letter count');
		assert.ok(rt.num('fact:counterResetF') < 0.1 && rt.num('fact:counterLetterF') > 0.9);
		assert.ok(rt.num('fact:counterRise') > 0.5);
	}, { ...opts, only: ['preset:lily'] });
});

test('step 9 task: a word of 10 letters or more completes it; the presets and the start text do not', async () => {
	const pkg = await pkgP;
	const step = pkg.lesson!.steps.find((s) => 'id' in s && s.id === 'counter') as unknown as { task: { done: (rt: AtlasRuntime) => boolean } };
	await withFixtures(pkg, (rt) => assert.equal(step.task.done(rt), (words(charsOf(rt)).reduce((m, w) => Math.max(m, w.word.length), 0)) >= 10), opts);
	await withFixtures(pkg, (rt) => assert.equal(step.task.done(rt), false), { ...opts, only: ['preset:lily', 'preset:dialog', 'preset:two-lines'] });
	await withFixtures(pkg, (rt) => assert.equal(step.task.done(rt), true), { ...opts, only: ['long-word', 'preset:long-words'] });
});

test('step 10: the drawn character is the one where u lands when the chances are laid end to end', async () => {
	await each(async (rt) => {
		await rt.ensure('sample', 'last');
		const tape = rt.sources.sample.tape!;
		const w = weightsOf(rt.model!);
		const prompt = (tape.state(0) as { ids: number[] }).ids;
		for (let i = 1; i < Math.min(tape.length, 12); i++) {
			const s = tape.state(i) as { ids: number[]; token: number; u: number };
			const p = softmax(scoresAfter(w, s.ids.slice(0, -1).slice(-128)));
			let lo = 0;
			for (let k = 0; k < s.token; k++) lo += p[k];
			assert.ok(lo <= s.u && s.u < lo + p[s.token], `frame ${i}: u ${s.u} outside [${lo}, ${lo + p[s.token]})`);
			assert.equal(s.ids.length, prompt.length + i);
		}
	});
});

test('step 10 stage: the strip lays the chances in the order the draw walks them, so u falls inside the highlighted piece on every frame', async () => {
	await each(async (rt) => {
		await rt.ensure('sample', 'last');
		const tape = rt.sources.sample.tape!;
		for (let k = 1; k < tape.length; k++) {
			const g = timelineOf({ rt, source: 'sample', at: { index: k, frame: tape.frame(k) }, props: { mode: 'sample' } } as never) as unknown as { a: { token: number; u: number; p: number }; draw: Float64Array; from: number[] };
			const { token, u, p } = g.a;
			assert.ok(g.from[token] <= u && u < g.from[token] + g.draw[token], `frame ${k}: u ${u} outside [${g.from[token]}, ${g.from[token] + g.draw[token]})`);
			assert.ok(Math.abs(g.draw[token] - p) < 1e-12, 'the piece is the chance the draw used');
		}
	});
});

test('step 10 predict: greedy writing gives the same text for any seed; drawing gives different texts for different seeds', async () => {
	await each((rt) => {
		const input = rnnInput(rt);
		const ids = (seed: number, temperature: number) => {
			const t = record(sample, input, { ...sample.defaults, temperature, maxNew: 30 }, { seed });
			return JSON.stringify((t.state(t.length - 1) as { ids: number[] }).ids);
		};
		assert.equal(ids(1, 0), ids(7, 0));
		assert.notEqual(ids(1, 1), ids(7, 1));
	});
});

test('step 11 predict and narration: 0.9^20 ≈ 12%, gates of 0.9 lose most within a few dozen characters, 0.99^20 ≈ 0.82, unit 51 inside a word keeps far more, and no unit follows open quotes', async () => {
	assert.equal(REF_GATE, 0.9);
	assert.equal(Math.round(REF_GATE ** 20 * 100), 12);
	assert.ok(REF_GATE ** 30 < 0.05);
	assert.equal(Math.round(0.99 ** 20 * 100), 82);
	await withFixtures(await pkgP, (rt) => {
		assert.ok(rt.num('fact:quoteR') < 0.1, 'best correlation with "a quote is open"');
		assert.ok(rt.num('fact:counterLetterF') >= 0.99 && rt.num('fact:counterLetterF') ** 20 > 0.8, 'a gate near 1 (unit 51 inside a word) keeps a memory far longer than 0.9');
	}, { ...opts, only: ['preset:lily'] });
});

test('step 10 predict explanation: the top-character writer soon writes “said, "I will help” on each of the four example texts', async () => {
	const pkg = await pkgP;
	let n = 0;
	await withFixtures(pkg, async (rt) => {
		await rt.ensure('greedy', 'last');
		const tape = rt.sources.greedy.tape!;
		const alpha = (rt.spec as unknown as { alphabet: string[] }).alphabet;
		const prompt = (tape.state(0) as { ids: number[] }).ids.length;
		const text = (tape.state(tape.length - 1) as { ids: number[] }).ids.slice(prompt).map((i) => alpha[i]).join('');
		assert.ok(text.includes('said, "I will help'), text);
		n++;
	}, { ...opts, only: ['preset:lily', 'preset:dialog', 'preset:long-words', 'preset:two-lines'] });
	assert.equal(n, 4);
});

test('step 10 narration: the least likely drawn character is one the run really drew, with that chance', async () => {
	await each(async (rt) => {
		await rt.ensure('sample', 'last');
		const tape = rt.sources.sample.tape!;
		const r = riskiest(rt, 'sample');
		assert.equal(r.n, tape.length - 1);
		const ps = Array.from({ length: tape.length - 1 }, (_, k) => (tape.state(k + 1) as { p: number; pModel: number }));
		assert.ok(ps.some((s) => s.p === r.p) && ps.every((s) => s.p >= r.p && Math.abs(s.p - s.pModel) < 1e-12), 'temperature 1: p is the model chance');
	});
});

test('input: curly quotes and dashes from a phone keyboard reach the model as the plain ones it was trained on', () => {
	assert.equal(normalize('“Hi,” she said—it’s…'), '"Hi," she said-it\'s...');
});
