// decoding claim tests (DEVKIT §8.4): every factual sentence of the lesson, the concept page (concept.ts) and the manifest
// that is not computed live is asserted here, over every fixture (the presets, the adapter's edge fixtures and the
// lesson's settings fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
// Run alone: node --test src/lib/explainers/decoding/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Tape } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { pickToken, samplingDistribution } from '../../atlas/process/library/sampling.ts';
import { tokenizerOf } from '../../atlas/adapters/text-bpe/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { CONTEXT, DEFAULTS, HIGH_T, K_DEMO, LOW_T, P_DEMO, ROWS, V, logitsOf, ranked, repeats4, restMass, settingsOf, shapeOf } from './decode.ts';
import { decode, type DecodeInput } from './process.ts';
import { PROMPTS } from './adapters/prompt/adapter.ts';
import { LOW_P, written } from './lesson.ts';
import { distOf } from './views/dist/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('decoding', SITE)).pkg;
const end = (t: Tape) => ({ index: t.length - 1, tape: t });
const ids = (t: Tape): number[] => written(end(t)).ids;
const PLAIN = { temperature: 1, topK: 0, topP: 1 };
/** a source's own input for this runtime (explainer.ts), to record the same run again with other options */
const inputOf = async (rt: ExplainerRuntime, src: string): Promise<DecodeInput> => {
	const d = (await pkg()).manifest.sources![src] as { input: (rt: ExplainerRuntime) => DecodeInput };
	return d.input(rt);
};

test('TinyGPT facts used by the page: 1,024 tokens, a 64-token context; the presets are its tokenizer’s ids', () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/spec.json'), 'utf8'));
	assert.equal(spec.vocabSize, V);
	assert.equal(spec.context, CONTEXT);
	const tok = tokenizerOf(JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/tokenizer.json'), 'utf8')))!;
	for (const p of PROMPTS) assert.deepEqual(tok.encode(p.raw.text), p.raw.ids, p.id);
	// the package's tokenizer.json is TinyGPT's, unchanged
	assert.equal(readFileSync(join(SITE, 'static/models/decoding/tokenizer.json'), 'utf8'), readFileSync(join(SITE, 'static/models/tinygpt/tokenizer.json'), 'utf8'));
});

test('steps "scores" and "pick": chances add up to 1; the runs start from the same chances the toy reshapes; greedy takes the top token; u lands in the picked token’s piece', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const l = logitsOf(rt);
			if (!l) return assert.ok(rt.pre?.empty, 'no logits only for the empty input');
			const x = shapeOf(l, PLAIN);
			assert.ok(Math.abs(x.p1.reduce((a, v) => a + v, 0) - 1) < 1e-12);
			for (const src of ['greedy', 'sample']) {
				await rt.ensure(src, { phase: 'sample', step: 0 });
				const t = rt.sources[src].tape!;
				const f = t.frame(1);
				const tok = (t.state(1) as { token: number }).token;
				assert.ok(Math.abs(f.metrics!.pModel - x.p1[tok]) < 1e-6, `${src}: ${f.metrics!.pModel} vs ${x.p1[tok]}`);
				if (src === 'greedy') assert.equal(tok, x.order[0]);
				else {
					// the strip of the view: the piece that holds u is the token the kit picked
					const pic = distOf({ rt, at: { index: 1, frame: f, state: t.state(1), tape: t }, props: { draw: true } })!;
					const seg = pic.draw!.strip.find((s) => f.metrics!.u >= s.from && f.metrics!.u < s.to)!;
					assert.equal(seg.id, tok);
					assert.equal(pickToken(samplingDistribution(l, 1), f.metrics!.u), tok);
				}
			}
		},
		opts
	);
});

test('steps "pick", "loops", quiz 2 and pitfall 1: the same seed gives the same text; greedy draws no random number and ignores the seed', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			if (!logitsOf(rt)) return;
			const inp = await inputOf(rt, 'mine');
			const s = settingsOf(rt.input);
			const a = record(decode, inp, { ...decode.defaults, maxNew: 12 });
			const b = record(decode, inp, { ...decode.defaults, maxNew: 12 });
			assert.deepEqual(ids(a), ids(b));
			const g1 = record(decode, { ...inp!, temperature: 0, seed: 1 }, { ...decode.defaults, maxNew: 12 });
			const g2 = record(decode, { ...inp!, temperature: 0, seed: 2 }, { ...decode.defaults, maxNew: 12 });
			assert.deepEqual(ids(g1), ids(g2));
			for (let i = 1; i < g1.length; i++) assert.ok(Number.isNaN(g1.frame(i).metrics!.u));
			// the generate step's seed note: a new seed changes the text unless only one token is ever left
			const other = record(decode, { ...inp!, seed: s.seed + 1 }, { ...decode.defaults, maxNew: 48 });
			const mine = record(decode, inp, { ...decode.defaults, maxNew: 48 });
			if (s.temperature <= 0 || s.topK === 1) assert.deepEqual(ids(other), ids(mine));
			else assert.notDeepEqual(ids(other), ids(mine));
		},
		opts
	);
});

test('steps "temperature" and "temp-numbers": T below 1 sharpens (the top token grows, the tail shrinks), above 1 flattens; toward 0 the top token takes everything; temperature keeps the order', async () => {
	await withFixtures(
		await pkg(),
		async (rt) => {
			const l = logitsOf(rt);
			if (!l) return;
			const one = shapeOf(l, PLAIN);
			const top = one.order[0];
			let prev = -1;
			for (const T of [HIGH_T, 1.5, 1, 0.8, LOW_T, 0.25, 0.1, 0.05]) {
				const x = shapeOf(l, { ...PLAIN, temperature: T });
				assert.ok(x.pT[top] >= prev, `T ${T}: ${x.pT[top]} < ${prev}`);
				prev = x.pT[top];
				assert.deepEqual(ranked(x.pT).slice(0, ROWS), one.order.slice(0, ROWS), 'temperature keeps the order');
			}
			const lo = shapeOf(l, { ...PLAIN, temperature: LOW_T });
			const hi = shapeOf(l, { ...PLAIN, temperature: HIGH_T });
			assert.ok(lo.pT[top] > one.p1[top] && hi.pT[top] < one.p1[top]);
			assert.ok(restMass(lo.pT, lo.order, ROWS) < restMass(one.p1, one.order, ROWS) && restMass(hi.pT, hi.order, ROWS) > restMass(one.p1, one.order, ROWS));
			const g = shapeOf(l, { ...PLAIN, temperature: 0 });
			assert.equal(g.dist[top], 1);
			assert.equal(g.kept, 1);
		},
		opts
	);
});

test('step "top-k" and quiz 1: the k kept chances are divided by their total, so they add up to 1; every other token gets 0', async () => {
	// quiz 1: kept chances 30%, 20%, 10% (60% together) -> the top one gets 50%
	const toy = shapeOf(Float64Array.from([0.3, 0.2, 0.1, 0.09, 0.08, 0.08, 0.08, 0.07].map(Math.log)), { temperature: 1, topK: 3, topP: 1 });
	assert.ok(Math.abs(toy.dist[0] - 0.5) < 1e-12);
	await withFixtures(
		await pkg(),
		async (rt) => {
			const l = logitsOf(rt);
			if (!l) return;
			const x = shapeOf(l, { ...PLAIN, topK: K_DEMO });
			assert.equal(x.kept, K_DEMO);
			assert.ok(Math.abs(x.dist.reduce((a, v) => a + v, 0) - 1) < 1e-12);
			for (const id of x.order.slice(0, K_DEMO)) assert.ok(Math.abs(x.dist[id] - x.p1[id] / x.keptMass) < 1e-12);
			for (const id of x.order.slice(K_DEMO)) assert.equal(x.dist[id], 0);
		},
		opts
	);
});

test('step "top-p", pitfall 2 and the task: top-p keeps the fewest top tokens whose chances reach p; fewer for “A big dog” than for “The cat sat”; the task can be done on every prompt', async () => {
	const counts: Record<string, number> = {};
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			const l = logitsOf(rt);
			if (!l) return;
			const x = shapeOf(l, { ...PLAIN, topP: P_DEMO });
			let cum = 0;
			for (let n = 0; n < x.kept; n++) cum += x.p1[x.order[n]];
			assert.ok(cum >= P_DEMO - 1e-12 && cum - x.p1[x.order[x.kept - 1]] < P_DEMO, `${x.kept} tokens hold ${cum}`);
			counts[fx.id] = x.kept;
			// the task: top-k off, some top-p (and a temperature) leaves 2 to 5 tokens
			let ok = false;
			for (const T of [1, 0.8, 0.5]) for (let p = 0.05; p <= 1 && !ok; p += 0.05) ok = (({ kept }) => kept >= 2 && kept <= 5)(shapeOf(l, { temperature: T, topK: 0, topP: Math.round(p * 100) / 100 }));
			assert.ok(ok, `${fx.id}: no setting leaves 2 to 5 tokens`);
		},
		opts
	);
	assert.ok(counts['preset:dog'] < counts['preset:cat'], JSON.stringify(counts));
});

test('steps "greedy-vs-sample", "too-hot", pitfall 3 and the misconception: greedy always takes the top token, sampling sometimes does not; T = 2 picks more unlikely tokens; sampling repeats less than greedy', async () => {
	let repGreedy = 0;
	let repSample = 0;
	await withFixtures(
		await pkg(),
		async (rt) => {
			if (!logitsOf(rt)) return;
			for (const s of ['greedy', 'sample', 'hot', 'mine']) await rt.ensure(s, 'last');
			const g = rt.sources.greedy.tape!;
			for (let i = 1; i < g.length; i++) {
				const st = g.state(i) as { token: number; top: { id: number }[] };
				assert.equal(st.token, st.top[0].id);
			}
			const sm = written(end(rt.sources.sample.tape!));
			assert.ok(sm.made.some((m, i) => (rt.sources.sample.tape!.state(i + 1) as { top: { id: number; p: number }[] }).top[0].p > m.p), 'sampling never left the top token');
			const hot = written(end(rt.sources.hot.tape!));
			const sm48 = sm.made.slice(0, hot.n).filter((m) => m.pModel < LOW_P).length;
			assert.ok(hot.rare > sm48, `T = 2: ${hot.rare} unlikely picks, T = 1: ${sm48}`);
			assert.notDeepEqual(ids(g), ids(rt.sources.sample.tape!), 'the same model writes different text with different rules');
		},
		opts
	);
	// the four presets: greedy against sampling at temperature 1, both 96 tokens
	await withFixtures(
		await pkg(),
		async (rt, fx) => {
			if (!fx.id.startsWith('preset:')) return;
			await rt.ensure('greedy', 'last');
			await rt.ensure('sample', 'last');
			repSample += repeats4(ids(rt.sources.sample.tape!));
			repGreedy += repeats4(ids(rt.sources.greedy.tape!));
		},
		opts
	);
	assert.ok(repSample < repGreedy, `sampling ${repSample} repeats, greedy ${repGreedy}`);
	assert.deepEqual(settingsOf(PROMPTS[0].raw), DEFAULTS);
});
