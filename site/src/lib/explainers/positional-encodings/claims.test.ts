// positional-encodings claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept
// page (concept.ts) and the manifest that is not computed live is asserted here, over every fixture (the lesson's
// fixtures, the adapter's presets and edge fixtures) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/positional-encodings/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { D, dot, freq, HD, rope, ROT, shuffleOf, sinTable, slope } from './pe.ts';
import { CTX, peOf } from './adapters/pe-text/adapter.ts';
import { ROWS } from './process.ts';
import { EVAL_CTX, LOSS, TRAIN_CTX, BIN } from './extrap.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'positional-encodings';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const facts = JSON.parse(readFileSync(join(SITE, 'static', 'models', SLUG, 'facts.json'), 'utf8')).facts as Record<string, { value: number }>;
const fact = (k: string) => facts[k].value;
const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinygpt', 'spec.json'), 'utf8'));
const IN = ['None', 'Learned', 'Sin', 'Rope', 'Alibi'].map((s) => fact('in' + s));

/** The frames of source 'pe' by phase, or null when the input is empty (the source does not run). */
async function peOfRt(rt: ExplainerRuntime) {
	if (!peOf(rt.input).ids.length) return null;
	await rt.ensure('pe', 'last');
	const t = rt.sources.pe.tape!;
	return (phase: string): Frame => t.frame(t.find({ phase }));
}
const each = async (fn: (at: (phase: string) => Frame, rt: ExplainerRuntime) => void) =>
	withFixtures(await pkg(), async (rt) => {
		const at = await peOfRt(rt);
		if (at) fn(at, rt);
	}, opts);

test('steps "order", "limits", the hook and the prequestion: without positions the last token’s layer-0 output and its weights per token do not change when the earlier tokens are shuffled', async () => {
	await each((at) => {
		const m = at('order').metrics!;
		assert.ok(m.d0 < 1e-6, `layer-0 output changed by ${m.d0}`);
		const s = at('order').state as Record<string, { data: ArrayLike<number> }>;
		const sorted = (k: string) => Array.from(s[k].data).sort((a, b) => a - b);
		sorted('w0').forEach((v, i) => assert.ok(Math.abs(v - sorted('w0s')[i]) < 1e-6, 'the weights are not a reordering'));
	});
});

test('step "positions": with TinyGPT’s position table, shuffling changes the last token’s output (whenever something moved)', async () => {
	await each((at) => {
		const m = at('order').metrics!;
		if (m.n >= 3 && m.same === 0) assert.ok(m.d1 > 1e-3, `only ${m.d1}`);
	});
});

test('step "limits" and pitfall 1: with no positions the causal mask still lets the final hidden state change when the tokens move', async () => {
	await each((at) => {
		const m = at('order').metrics!;
		if (m.n >= 3 && m.same === 0) assert.ok(m.dF0 > 1e-3, `only ${m.dF0}`);
	});
});

test('the lesson note: ten steps', async () => {
	assert.equal((await import('./lesson.ts')).default.steps.length, 10);
});

test('shuffleOf keeps the last token in place and never returns the typed order when two or more tokens can move', () => {
	for (let n = 1; n < 40; n++)
		for (let seed = 1; seed < 30; seed++) {
			const p = shuffleOf(n, seed);
			assert.deepEqual([...p].sort((a, b) => a - b), Array.from({ length: n }, (_, i) => i));
			assert.equal(p[n - 1], n - 1);
			if (n >= 3) assert.ok(p.some((v, i) => v !== i), `identity for n ${n} seed ${seed}`);
		}
});

test('step "learned", the real panel and the titles: TinyGPT has a learned table of 64 rows of 76 (no row 64, at most 64 tokens), neighbours alike, the ends almost unrelated', async () => {
	assert.ok(spec.tensors.some((t: { name: string; shape: number[] }) => t.name === 'pos_emb.weight' && t.shape[0] === ROWS && t.shape[1] === D));
	assert.equal(spec.context, ROWS);
	assert.equal(CTX, ROWS);
	assert.equal(ROWS, 64);
	await each((at) => {
		const m = at('learned').metrics!;
		assert.ok(m.mean1 > 0.9 && m.mean1 > m.far && m.far < 0.1, `mean1 ${m.mean1}, far ${m.far}`);
	});
});

test('step "sinusoidal": the table is the formula, pair 0 turns at 1 radian per position and every later pair slower; it matches the PyTorch sweep', () => {
	const t = sinTable(128, D);
	for (const [p, i] of [[0, 0], [5, 0], [10, 1], [70, 20], [127, 37]]) {
		assert.ok(Math.abs(t[p * D + 2 * i] - Math.sin(p * 10000 ** ((-2 * i) / D))) < 1e-12);
		assert.ok(Math.abs(t[p * D + 2 * i + 1] - Math.cos(p * 10000 ** ((-2 * i) / D))) < 1e-12);
	}
	assert.equal(freq(0, D), 1);
	for (let i = 1; i < D / 2; i++) assert.ok(freq(i, D) < freq(i - 1, D));
	assert.ok(Math.abs(t[10 * D + 3] - fact('chkSin')) < 1e-6, 'PE[10, 3] differs from extrap.py');
});

test('step "rope" and quiz 1: pair 0 turns 1 radian per position, 9 pairs of a 19-dim head turn, the slowest least; it matches the PyTorch sweep; TinyGPT itself uses a learned table', () => {
	assert.equal(freq(0, ROT), 1);
	assert.equal(ROT / 2, 9);
	assert.equal(HD, 19);
	assert.ok(Math.abs(100 * freq(1, ROT) - fact('chkRope')) < 1e-4, 'RoPE angle differs from extrap.py');
	const x = Array.from({ length: HD }, (_, i) => i + 1);
	assert.equal(rope(x, 7)[HD - 1], x[HD - 1], 'the last dim is not turned');
});

test('step "relative", the toy, quiz 1 and the misconception: with RoPE the query–key dot product depends only on the distance; with TinyGPT’s learned rows it changes with the shift', async () => {
	const rng = createRng(3);
	for (let r = 0; r < 50; r++) {
		const a = Array.from({ length: HD }, () => rng.normal());
		const b = Array.from({ length: HD }, () => rng.normal());
		const [m, n, s] = [rng.int(200), rng.int(200), rng.int(200)];
		assert.ok(Math.abs(dot(rope(a, m), rope(b, n)) - dot(rope(a, m + s), rope(b, n + s))) < 1e-9);
	}
	await each((at) => {
		const m = at('relative').metrics!;
		assert.ok(Math.abs(m.ropeP - m.rope0) < 1e-9, `${m.ropeP} vs ${m.rope0}`);
		assert.ok(Math.abs(m.ropeP - m.ropeRef) < 1e-9 && m.ref !== m.p, `${m.ropeP} vs ${m.ropeRef} at ${m.ref}`);
		// with k = 63 the table holds a single placement (key 0, query 63): nothing to compare
		if (m.k < ROWS - 1) assert.ok(m.absMax - m.absMin > 0.1, `learned rows: ${m.absMin}..${m.absMax}`);
	});
});

test('step "alibi", quiz 2 and pitfall 3: slopes 1/4, 1/16, 1/64, 1/256; 8 back at m = 1/4 is −2 (e^−2 ≈ 14%); head 0 subtracts 16 at 64 back', () => {
	assert.deepEqual([0, 1, 2, 3].map((h) => slope(h)), [1 / 4, 1 / 16, 1 / 64, 1 / 256]);
	assert.equal(-slope(0) * 8, -2);
	assert.equal(Math.round(Math.exp(-2) * 100), 14);
	assert.equal(slope(0) * 64, 16);
});

test('steps "extrapolate", "results", the compare panel, the prequestion and pitfalls 1 and 2: the sweep’s numbers', () => {
	assert.equal(fact('trainCtx'), TRAIN_CTX);
	assert.equal(fact('evalCtx'), EVAL_CTX);
	assert.equal(TRAIN_CTX, 64);
	assert.equal(EVAL_CTX, 256);
	assert.equal(EVAL_CTX / TRAIN_CTX, 4); // "four times longer"
	assert.equal(fact('windows'), 500);
	// within 64 tokens they score alike, RoPE lowest, no positions highest
	assert.ok(Math.max(...IN) - Math.min(...IN) < 0.2, `in: ${IN}`);
	assert.equal(Math.min(...IN), fact('inRope'));
	assert.equal(Math.max(...IN), fact('inNone'));
	// ALiBi keeps improving and is the lowest past 64 among sinusoidal, RoPE and ALiBi (and no positions)
	assert.ok(fact('farAlibi') < fact('inAlibi'));
	assert.ok(fact('farAlibi') < Math.min(fact('farRope'), fact('farSin'), fact('farNone')));
	// sinusoidal and RoPE much worse, no positions somewhat worse; the sinusoidal loss more than doubles
	assert.ok(fact('farSin') > fact('inSin') + 1 && fact('farRope') > fact('inRope') + 1);
	assert.ok(fact('farNone') > fact('inNone') && fact('farNone') < Math.min(fact('farSin'), fact('farRope')));
	assert.ok(fact('farSin') > 2 * fact('inSin'));
	// the learned table has no rows past its trained length
	LOSS.learned.forEach((v, b) => assert.equal(v === null, b * BIN >= TRAIN_CTX));
	// RoPE's slowest pair and the sinusoidal slowest pair never wrap within the trained length: later angles are new
	assert.ok((TRAIN_CTX - 1) * freq(ROT / 2 - 1, ROT) < 2 * Math.PI && (TRAIN_CTX - 1) * freq(D / 2 - 1, D) < 2 * Math.PI);
});
