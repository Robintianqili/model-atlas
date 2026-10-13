// init-normalization claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept
// page (concept.ts) and the manifest that is not computed live is asserted here, on the sources the sentences describe
// and, for the learner's own network, over every fixture through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/init-normalization/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime, Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import {
	BATCH, DEPTH, EPS, INITS, INIT_STD, NB, NF, N_BIASES, N_WEIGHTS, PROBE_IDS, PROBE_TOKENS, SAT, WIDTH,
	batchNorm, layerNorm, meanStd, setupOf, type Setup
} from './data.ts';
import { deepRun, draws } from './net.ts';
import { inits, mlp } from './process.ts';
import { STARTS } from './adapters/setup/adapter.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'init-normalization';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const SEED = 1;
const frameOf = (s: Setup, phase: 'forward' | 'backward' = 'backward'): Record<string, number> => {
	const tape = record(mlp, s, {}, { seed: SEED });
	for (let i = 0; i < tape.length; i++) if (tape.frame(i).phase === phase) return tape.frame(i).metrics!;
	throw new Error('no frame');
};
const initsOf = (act: Setup['act'], norm: Setup['norm']) => record(inits, { act, norm }, {}, { seed: SEED }).frame(0).metrics!;
const tanh = (std: number): Setup => ({ act: 'tanh', std, norm: 'none' });
const relu = (std: number, norm: Setup['norm'] = 'none'): Setup => ({ act: 'relu', std, norm });
const spreads = (m: Record<string, number>, p = 's') => Array.from({ length: DEPTH }, (_, l) => m[`${p}${l + 1}`]);

test('the sources the lesson names run on seed 1, the default of a process source', async () => {
	const p = await pkg();
	await withFixtures(
		p,
		async (rt: ExplainerRuntime) => {
			const i = await rt.ensure('small', { phase: 'backward' });
			const m = rt.sources.small.tape!.frame(i).metrics!;
			assert.equal(m.s10, frameOf(tanh(0.01)).s10);
		},
		opts
	);
});

test('the network: 41,024 weights and 641 biases; Xavier is √(2/(64 + 64)) = √(1/64) and He √(2/64) for these layers (step "deep", "relu", glossary)', () => {
	assert.equal(N_WEIGHTS, 41024);
	assert.equal(N_BIASES, 641);
	assert.equal(N_WEIGHTS, DEPTH * WIDTH * WIDTH + WIDTH);
	assert.ok(Math.abs(INIT_STD.xavier - Math.sqrt(1 / 64)) < 1e-15);
	assert.ok(Math.abs(INIT_STD.he - Math.sqrt(2 / 64)) < 1e-15);
	assert.equal(BATCH, 256);
});

test('autograd cross-check: the forward pass equals plain loops, and dL/dz of layer 10 equals the chain rule written out', () => {
	for (const s of [tanh(INIT_STD.xavier), relu(INIT_STD.he)]) {
		const r = deepRun(s, 1);
		const d = draws(1);
		let h = Array.from(d.x);
		const zs: number[][] = [];
		for (let l = 0; l < DEPTH; l++) {
			const z = new Array(BATCH * WIDTH).fill(0);
			for (let i = 0; i < BATCH; i++)
				for (let o = 0; o < WIDTH; o++) {
					let acc = 0;
					for (let k = 0; k < WIDTH; k++) acc += h[i * WIDTH + k] * s.std * d.w[l][o * WIDTH + k];
					z[i * WIDTH + o] = acc;
				}
			zs.push(z);
			h = z.map((v) => (s.act === 'tanh' ? Math.tanh(v) : Math.max(0, v)));
		}
		h.forEach((v, k) => assert.ok(Math.abs(v - r.h[DEPTH - 1][k]) < 1e-9 * Math.max(1, Math.abs(v))));
		// L = mean((ŷ − y)²): dL/dŷ_i = 2(ŷ_i − y_i)/B; dL/dz10 = dL/dŷ · w_out · act'(z10)
		const yhat = Array.from({ length: BATCH }, (_, i) => h.slice(i * WIDTH, (i + 1) * WIDTH).reduce((a, v, k) => a + v * s.std * d.wOut[k], 0));
		const z10 = zs[DEPTH - 1];
		for (let i = 0; i < BATCH; i += 37)
			for (let o = 0; o < WIDTH; o += 7) {
				const dy = (2 * (yhat[i] - d.y[i])) / BATCH;
				const da = s.act === 'tanh' ? 1 - Math.tanh(z10[i * WIDTH + o]) ** 2 : z10[i * WIDTH + o] > 0 ? 1 : 0;
				const g = dy * s.std * d.wOut[o] * da;
				assert.ok(Math.abs(g - r.gz[DEPTH - 1][i * WIDTH + o]) < 1e-9 * Math.max(1e-6, Math.abs(g)), `${g} vs ${r.gz[DEPTH - 1][i * WIDTH + o]}`);
			}
	}
});

test('step "deep": with Xavier and tanh the spread shrinks slowly but stays in the same range', () => {
	const m = frameOf(tanh(INIT_STD.xavier));
	const s = spreads(m);
	for (let l = 1; l < DEPTH; l++) assert.ok(s[l] < s[l - 1], `layer ${l + 1}`);
	assert.ok(m.s10 / m.s1 > 0.25, `ratio ${m.s10 / m.s1}`);
	assert.ok(Math.abs(m.s0 - 1) < 0.02, `input spread ${m.s0}`);
});

test('step "zero" and the misconception: every activation and gradient is 0, no weight gets a gradient, only the output bias does', () => {
	for (const act of ['tanh', 'relu'] as const) {
		const m = frameOf({ act, std: 0, norm: 'none' });
		spreads(m).forEach((v) => assert.equal(v, 0));
		spreads(m, 'g').forEach((v) => assert.equal(v, 0));
		assert.equal(m.nz, 0);
		assert.equal(m.nzb, 1);
		assert.ok(m.gb !== 0);
	}
	assert.equal(Math.tanh(0), 0);
});

test('steps "predict", "small": each layer multiplies the spread by about 0.01 × √64 = 0.08, ten layers by about 10⁻¹¹; the gradient fades the same way and is almost 0 at layer 1', () => {
	assert.ok(Math.abs(0.01 * Math.sqrt(64) - 0.08) < 1e-15);
	assert.ok(0.08 ** 10 > 1e-12 && 0.08 ** 10 < 1e-10);
	const m = frameOf(tanh(0.01));
	assert.ok(Math.abs(m.f - 0.08) < 0.008, `per-layer factor ${m.f}`);
	assert.ok(m.s10 / m.s0 > 1e-12 && m.s10 / m.s0 < 1e-10, `layer 10 ${m.s10}`);
	assert.ok(Math.abs(m.gf - 0.08) < 0.008, `gradient factor ${m.gf}`);
	assert.ok(m.g1 < 1e-10 && m.g1 < m.g10 * 1e-6, `g1 ${m.g1}, g10 ${m.g10}`);
	const s = spreads(m);
	for (let l = 1; l < DEPTH; l++) assert.ok(s[l] < s[l - 1]);
});

test('step "large" and pitfall 1: tanh saturates (most of layer 10 beyond ±0.99) but the gradient grows toward layer 1, about twice per layer', () => {
	const m = frameOf(tanh(INIT_STD.large));
	assert.equal(SAT, 0.99);
	assert.ok(m.sat > 0.5, `saturated ${m.sat}`);
	assert.ok(m.g1 > m.g10 * 100, `g1 ${m.g1} g10 ${m.g10}`);
	assert.ok(m.gf > 1.5 && m.gf < 3, `per layer ${m.gf}`);
	const g = spreads(m, 'g');
	for (let l = 1; l < DEPTH; l++) assert.ok(g[l] < g[l - 1], `layer ${l + 1}`);
});

test('step "relu" and pitfall 2: with Xavier, ReLU loses about half the variance per layer; He doubles σ² and the spread holds', () => {
	const m = initsOf('relu', 'none');
	const perLayerVar = (m.xavier10 / m.xavier1) ** (2 / (DEPTH - 1));
	assert.ok(perLayerVar > 0.35 && perLayerVar < 0.65, `variance factor ${perLayerVar}`);
	assert.ok(Math.abs(m.heStd ** 2 - 2 * m.xavierStd ** 2) < 1e-15);
	assert.ok(m.he10 / m.he1 > 0.5 && m.he10 / m.he1 < 2, `He ratio ${m.he10 / m.he1}`);
	// tanh with Xavier keeps far more than ReLU with Xavier
	const t = initsOf('tanh', 'none');
	assert.ok(t.xavier10 / t.xavier1 > 5 * (m.xavier10 / m.xavier1));
});

test('steps "batchnorm", "layernorm", "limits": every column (row) comes out with mean 0 and spread 1; the worked cells; LayerNorm of one example alone equals its row; BatchNorm of a batch of one is all 0', () => {
	const tape = record(mlp, tanh(1), {}, { seed: SEED });
	const t = Array.from((tape.state(0) as { t: { data: ArrayLike<number> } }).t.data);
	assert.equal(t.length, NB * NF);
	const bn = batchNorm(t, NB, NF);
	const ln = layerNorm(t, NB, NF);
	for (let j = 0; j < NF; j++) {
		const s = meanStd(bn, j, NB, NF);
		assert.ok(Math.abs(s.mean) < 1e-9 && Math.abs(s.std - 1) < 1e-3, `column ${j}: ${s.mean}, ${s.std}`);
	}
	for (let i = 0; i < NB; i++) {
		const s = meanStd(ln, i * NF, NF);
		assert.ok(Math.abs(s.mean) < 1e-9 && Math.abs(s.std - 1) < 1e-3, `row ${i}: ${s.mean}, ${s.std}`);
		const alone = layerNorm(t.slice(i * NF, (i + 1) * NF), 1, NF);
		alone.forEach((v, j) => assert.equal(v, ln[i * NF + j]));
		batchNorm(t.slice(i * NF, (i + 1) * NF), 1, NF).forEach((v) => assert.equal(v, 0));
	}
	// "(x − m) / sd ≈ y": ε changes the result by less than 0.1%
	const m = tape.frame(0).metrics!;
	const k = m.cr * NF + m.cc;
	const col = meanStd(t, m.cc, NB, NF);
	assert.ok(Math.abs((t[k] - col.mean) / col.std - bn[k]) < 1e-3 * Math.abs(bn[k]) + 1e-9);
	assert.ok(Math.abs(bn[k] - ln[k]) > 1e-3, 'the two normalizations give different numbers for the worked cell');
	assert.equal(EPS, 1e-5);
});

test('step "task": ReLU with σ = 1 grows at every layer without normalization; with BatchNorm or LayerNorm (σ = 1 or 2) layer 10 stays within a factor of 2 of layer 1', () => {
	const plain = spreads(frameOf(relu(1)));
	for (let l = 1; l < DEPTH; l++) assert.ok(plain[l] > plain[l - 1]);
	for (const norm of ['batch', 'layer'] as const)
		for (const std of [1, 2]) {
			const m = frameOf(relu(std, norm));
			assert.ok(m.s10 / m.s1 > 0.5 && m.s10 / m.s1 < 2, `${norm} ${std}: ${m.s1} -> ${m.s10}`);
		}
	// the task's predicate accepts exactly these setups
	assert.equal(setupOf({ act: 'relu', std: 1, norm: 'batch' }).std, INIT_STD.large);
});

test('concept "compare": with a normalization on, the four non-zero starts end within a factor of 3 of each other at layer 10, for both activations', () => {
	for (const act of ['tanh', 'relu'] as const)
		for (const norm of ['batch', 'layer'] as const) {
			const m = initsOf(act, norm);
			const v = INITS.filter((id) => id !== 'zero').map((id) => m[`${id}10`]);
			assert.ok(Math.max(...v) / Math.min(...v) < 3, `${act} ${norm}: ${v}`);
		}
});

test('the learner\'s network on every fixture: the frames have every number the toy and the task narrate', async () => {
	await withFixtures(
		await pkg(),
		async (rt: ExplainerRuntime) => {
			const i = await rt.ensure('mine', { phase: 'backward' });
			const m = rt.sources.mine.tape!.frame(i).metrics!;
			for (const k of ['s1', 's10', 'g1', 'g10']) assert.ok(Number.isFinite(m[k]), k);
		},
		opts
	);
	assert.ok(STARTS.length >= 3);
});

test('concept "real" and step "layernorm", "limits": TinyGPT has a LayerNorm before every attention and every MLP and no BatchNorm; the probe is replay probe 0 and its tokens; our LayerNorm matches L1.ln1; the chosen dimension is near its mean for every token and LayerNorm keeps it large', async () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/spec.json'), 'utf8')) as { nLayer: number; layers: { id: string; op: string; in: string[] }[] };
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	for (let k = 0; k < spec.nLayer; k++) {
		for (const p of ['q', 'k', 'v']) assert.deepEqual(byId.get(`L${k}.attn.${p}`)!.in, [`L${k}.ln1`]);
		assert.deepEqual(byId.get(`L${k}.mlp.fc`)!.in, [`L${k}.ln2`]);
		assert.equal(byId.get(`L${k}.ln1`)!.op, 'layernorm');
		assert.equal(byId.get(`L${k}.ln2`)!.op, 'layernorm');
	}
	assert.ok(!spec.layers.some((l) => /batch/i.test(l.op)));
	const replay = JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/replay.json'), 'utf8')) as { probes: string[]; probeIds: number[][] };
	assert.deepEqual(replay.probeIds[0], PROBE_IDS);
	assert.equal(replay.probes[0], PROBE_TOKENS.join(''));
	const tok = JSON.parse(readFileSync(join(SITE, 'static/models/tinygpt/tokenizer.json'), 'utf8')) as { tokenBytes: (number[] | string)[] };
	PROBE_IDS.forEach((id, i) => {
		const b = tok.tokenBytes[id];
		const text = Array.isArray(b) ? Buffer.from(b).toString('utf8') : Buffer.from(String(b), 'hex').toString('utf8');
		assert.equal(text, PROBE_TOKENS[i], `token ${id}`);
	});
	await withFixtures(
		await pkg(),
		async (rt: ExplainerRuntime) => {
			const i = await rt.ensure('tgpt', 'last');
			const f: Frame = rt.sources.tgpt.tape!.frame(i);
			const m = f.metrics!;
			assert.equal(m.tokens, PROBE_IDS.length);
			assert.equal(m.dims, 76);
			assert.ok(m.diff < 1e-4, `LayerNorm gap ${m.diff}`);
			assert.equal(m.row, PROBE_IDS.length - 1);
			// 'dimension c is near its mean for every token': its spread over the tokens is small next to its mean
			assert.ok(m.colStd < 0.25 * Math.abs(m.colMean), `column ${m.col}: mean ${m.colMean}, spread ${m.colStd}`);
			// 'LayerNorm keeps it large' while BatchNorm makes it small
			assert.ok(Math.abs(m.ln) > 2 && Math.abs(m.ln) > 10 * Math.abs(m.bn), `ln ${m.ln}, bn ${m.bn}`);
		},
		opts
	);
});

test('quiz: He for 256 inputs is √(2/256) ≈ 0.088 and Xavier for a square 256 layer √(1/256) = 0.0625; 32 × 10 = 320 tokens, 320 × 76 = 24,320 numbers', () => {
	assert.equal(Math.sqrt(2 / 256).toFixed(3), '0.088');
	assert.equal(Math.sqrt(2 / (256 + 256)), 0.0625);
	assert.equal(32 * 10, 320);
	assert.equal(320 * 76, 24320);
});
