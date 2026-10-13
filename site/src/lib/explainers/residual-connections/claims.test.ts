// residual-connections claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept
// page (concept.ts) and the manifest that is not computed live is asserted here; the claims about the learner's own
// setup run over every fixture (the starting setups and the adapter's edge fixtures) through withFixtures. `atlas check`
// runs this file (AK408 on failure). Run alone: node --test src/lib/explainers/residual-connections/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor, TrainState } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { grad, noGrad } from '../../atlas/autograd/var.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { DEPTHS, LR, MAX_DEPTH, MIN_DEPTH, MOONS, WIDTH, midLayer, setupOf } from './data.ts';
import { analyse, depths, grads, race } from './process.ts';
import { DeepNet, deep, pointsVar } from './net.ts';
import { LOG_HI, LOG_LO } from './views/bars/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'residual-connections';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
const json = (p: string) => JSON.parse(readFileSync(join(SITE, 'static', p), 'utf8'));
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const data = (t: unknown) => Array.from((t as Tensor).data);

// the fixed runs the lesson narrates (seed 1, as the sources)
const plain = record(grads, { depth: MAX_DEPTH, skip: 0 }, {}, { seed: 1 });
const res = record(grads, { depth: MAX_DEPTH, skip: 1 }, {}, { seed: 1 });
let raceTape: Tape | null = null;
let depthTape: Tape | null = null;
const raced = () => (raceTape ??= record(race, null, {}, { seed: 1 }));
const swept = () => (depthTape ??= record(depths, null, {}, { seed: 1 }));
const lastM = (t: Tape) => t.frame(t.length - 1).metrics!;

test('net: the plain and the skip network start from the very same weights, and so does the training run (steps "flat", "race")', () => {
	const a = new DeepNet(MAX_DEPTH, false, createRng(1).fork('init'));
	const b = new DeepNet(MAX_DEPTH, true, createRng(1).fork('init'));
	for (const [k, v] of Object.entries(a.named())) assert.deepEqual(Array.from(v.data), Array.from(b.named()[k].data), k);
	const run = trainingProcess(deep).run(null, { rng: createRng(1), opt: { ...deep.defaults, skip: 0 }, detail: 'coarse', signal: { cancelled: false } } as never);
	const init = run.next().value as { state: Partial<TrainState> };
	for (const [k, v] of Object.entries(a.named())) assert.deepEqual(data(init.state.params![k]), Array.from(v.data), k);
	// 20 linear layers: 2 -> 8, 18 of 8 -> 8, 8 -> 2; skips on the 18 middle ones
	assert.equal(a.layers.length, MAX_DEPTH);
	assert.equal(b.layers.filter((_, l) => b.skips(l)).length, MAX_DEPTH - 2);
	assert.equal(a.layers[1].weight.shape.join('x'), `${WIDTH}x${WIDTH}`);
});

test('backward: autograd\'s layer-0 gradient of the plain 20-layer network matches central differences (the tiny numbers are real)', () => {
	const net = new DeepNet(MAX_DEPTH, false, createRng(1).fork('init'));
	const w = net.layers[0].weight;
	const g = plain.state(1).gW as Tensor;
	const lossAt = () => noGrad(() => crossEntropy(net.forward(pointsVar()), MOONS.label).item());
	const h = 1e-4;
	// the gradient norm from the frame equals the norm of the central differences over all 16 weights of layer 0
	let s = 0;
	for (let i = 0; i < w.data.length; i++) {
		const v = w.data[i];
		w.data[i] = v + h;
		const up = lossAt();
		w.data[i] = v - h;
		const dn = lossAt();
		w.data[i] = v;
		s += ((up - dn) / (2 * h)) ** 2;
	}
	assert.ok(close(Math.sqrt(s), g.data[0], 1e-4), `numerical ${Math.sqrt(s)} vs autograd ${g.data[0]}`);
});

test('steps "deeper", hook, prequestion: 3 layers learn, 20 plain layers stay at the loss of a coin flip; deeper plain is worse, deeper with skips is better', () => {
	const t = swept();
	const f = DEPTHS.map((_, i) => t.frame(i).metrics!);
	assert.deepEqual(f.map((m) => m.depth), DEPTHS);
	assert.equal(f[0].depth, MIN_DEPTH);
	assert.ok(f[0].lossPlain < 0.4, `3 plain layers: ${f[0].lossPlain}`);
	const z = f[f.length - 1];
	assert.ok(Math.abs(z.lossPlain - Math.LN2) < 0.005, `20 plain layers: ${z.lossPlain} vs ln 2`);
	assert.ok(z.accPlain < 0.65, `20 plain layers sort ${z.accPlain}`);
	// "Without skips, more layers do worse"; "with skips, more layers do better"
	assert.ok(z.lossPlain > f[0].lossPlain);
	for (let i = 1; i < f.length; i++) assert.ok(f[i].lossPlain > f[i - 1].lossPlain - 1e-3, `plain, ${f[i].depth} layers ${f[i].lossPlain} vs ${f[i - 1].depth} layers ${f[i - 1].lossPlain}`);
	for (let i = 1; i < f.length; i++) assert.ok(f[i].loss < f[i - 1].loss, `with skips, ${f[i].depth} layers ${f[i].loss} vs ${f[i - 1].depth} layers ${f[i - 1].loss}`);
	// "With only 3 layers there is little to skip, and the two end close"
	assert.ok(Math.abs(f[0].loss - f[0].lossPlain) < 0.02);
	// the comparison's 20-layer runs are the race's runs
	const r = lastM(raced());
	assert.ok(close(r.loss, z.loss) && close(r.lossPlain, z.lossPlain), `race ${r.loss}, ${r.lossPlain} vs sweep ${z.loss}, ${z.lossPlain}`);
});

test('steps "predict", "vanish", goals, pitfall 1: layer 0 of the plain network gets a gradient thousands of times smaller than the last layer, but not zero', () => {
	const m = plain.frame(1).metrics!;
	assert.equal(plain.frame(0).phase, 'forward');
	assert.equal(plain.frame(1).phase, 'backward');
	assert.ok(m.g0 > 0, 'not zero');
	assert.ok(m.ratio > 10_000, `ratio ${m.ratio}`);
	// "Here those factors are mostly below 1" (predict.explain): the factor of most plain layers
	const gH = data(plain.state(1).gH);
	const below = gH.slice(1).filter((v, i) => gH[i] / v < 1).length;
	assert.ok(below > (gH.length - 1) / 2, `${below} of ${gH.length - 1} factors below 1`);
});

test('step "factor": the plain middle layer shrinks the gradient; the block\'s share is Wᵀ(tanh′(z) ⊙ g), written out by hand', () => {
	const m = plain.frame(2).metrics!;
	assert.equal(plain.frame(2).phase, 'block');
	assert.equal(m.layer, midLayer(MAX_DEPTH));
	assert.ok(m.factor < 1, `factor ${m.factor}`);
	assert.equal(m.skipPart, 0);
	assert.ok(close(m.total, m.block), 'without a skip only the block sends the gradient back');
	// by hand: dL/dh_{l−1}[p, j] = Σ_i W[i, j] · (1 − a_l[p, i]²) · dL/dh_l[p, i]  (tanh′ = 1 − tanh², at most 1)
	const net = new DeepNet(MAX_DEPTH, false, createRng(1).fork('init'));
	const tr = net.trace(pointsVar());
	const l = m.layer;
	const W = net.layers[l].weight.data;
	const a = tr.a[l].data;
	const g = grad(crossEntropy(tr.scores, MOONS.label), tr.h[l]).data;
	const [p, j] = [m.point, m.unit];
	let byHand = 0;
	for (let i = 0; i < WIDTH; i++) {
		const slope = 1 - a[p * WIDTH + i] ** 2;
		assert.ok(slope > 0 && slope <= 1, 'the slope of tanh is never above 1');
		byHand += W[i * WIDTH + j] * slope * g[p * WIDTH + i];
	}
	assert.ok(close(byHand, m.block, 1e-9), `by hand ${byHand} vs ${m.block}`);
	assert.ok(close(g[p * WIDTH + j], m.up, 1e-9));
	// "these factors multiply to {prod}": the product of the per-layer factors is h0 / hLast
	const gH = data(plain.state(1).gH);
	let prod = 1;
	for (let k = 1; k < gH.length; k++) prod *= gH[k - 1] / gH[k];
	assert.ok(close(prod, m.h0 / m.hLast, 1e-9));
});

test('step "skip", quiz 2: through a skip, the gradient that leaves is the arriving one plus the block\'s share (the derivative of the addition is 1)', async () => {
	const m = res.frame(2).metrics!;
	assert.equal(res.frame(2).phase, 'skip');
	assert.ok(close(m.skipPart, m.up));
	// "Take the largest number arriving at layer {l}'s output": the chosen entry is the largest of all, and positive
	const net = new DeepNet(MAX_DEPTH, true, createRng(1).fork('init'));
	const tr = net.trace(pointsVar());
	const arriving = Array.from(grad(crossEntropy(tr.scores, MOONS.label), tr.h[m.layer]).data);
	assert.ok(m.up > 0 && close(m.up, Math.max(...arriving), 1e-9), `up ${m.up} vs largest ${Math.max(...arriving)}`);
	assert.ok(close(m.total, m.up + m.block, 1e-10), `${m.total} vs ${m.up} + ${m.block}`);
	assert.ok(close(0.2 + -0.05, 0.15, 1e-12));
	// the same holds on every fixture with skips
	await withFixtures(
		await pkg(),
		async (rt) => {
			const s = setupOf(rt.input);
			const i = await rt.ensure('bars', 'last');
			const f = rt.sources.bars.tape!.frame(i);
			assert.equal(f.phase, s.skip ? 'skip' : 'block');
			const q = f.metrics!;
			assert.ok(close(q.total, q.skipPart + q.block, 1e-10), `${q.total} vs ${q.skipPart} + ${q.block}`);
		},
		opts
	);
});

test('steps "flat", "task", toy: with skips every layer gets a gradient of a similar size; the bars fit the view\'s log axis for every setup', () => {
	const m = res.frame(1).metrics!;
	assert.ok(m.ratio < 10, `ratio ${m.ratio}`);
	assert.ok(close(m.oRatio, plain.frame(1).metrics!.ratio), 'the flipped switch is the plain network');
	for (let d = MIN_DEPTH; d <= MAX_DEPTH; d++)
		for (const skip of [0, 1]) {
			const a = analyse(d, skip, createRng(1).fork('init'));
			for (const v of [...a.gW, ...a.gH]) assert.ok(v >= LOG_LO && v <= LOG_HI, `depth ${d} skip ${skip}: ${v} outside the axis`);
		}
});

test('step "race", concept compare: trained alike, the plain network stays at ln 2 while the skip network learns; layer 0 barely moves without skips', () => {
	const m = lastM(raced());
	assert.ok(Math.abs(m.lossPlain - Math.LN2) < 0.005, `plain ${m.lossPlain}`);
	assert.ok(m.loss < 0.05 && m.accuracy > 0.95, `skips ${m.loss}, ${m.accuracy}`);
	assert.ok(m.movePlain < 1e-3 && m.move > 100 * m.movePlain, `moved ${m.move} vs ${m.movePlain}`);
	assert.equal(m.lr, LR);
});

test('step "limits", pitfall 3: with skips the signal grows with depth and the starting loss is higher', () => {
	const m = res.frame(1).metrics!;
	assert.ok(m.rmsLast > 3 * m.rms0, `${m.rms0} -> ${m.rmsLast}`);
	assert.ok(m.rmsLast > m.oRmsLast);
	assert.ok(m.loss > m.oLoss, `${m.loss} vs ${m.oLoss}`);
});

test('facts, step "limits", hook, pitfall 3: TinyGPT\'s residual additions and LayerNorms, one LayerNorm in front of every attention and MLP block', () => {
	const spec = json('models/tinygpt/spec.json') as { nLayer: number; layers: { id: string; op: string; in: string[] }[] };
	const facts = json('models/residual-connections/facts.json').facts as Record<string, { value: number }>;
	const adds = spec.layers.filter((l) => l.op === 'add' && /resid/.test(l.id));
	assert.equal(facts.gptAdds.value, adds.length);
	assert.equal(facts.gptAdds.value, 2 * spec.nLayer);
	assert.equal(facts.gptBlocks.value, spec.nLayer);
	assert.equal(facts.gptNorms.value, spec.layers.filter((l) => l.op === 'layernorm').length);
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	for (let b = 0; b < spec.nLayer; b++) {
		assert.equal(byId.get(`L${b}.attn.q`)!.in[0], `L${b}.ln1`);
		assert.equal(byId.get(`L${b}.mlp.fc`)!.in[0], `L${b}.ln2`);
		// each addition takes the stream and the block's output: x + f(x)
		assert.ok(byId.get(`L${b}.resid1`)!.in.includes(`L${b}.attn.proj`));
		assert.ok(byId.get(`L${b}.resid2`)!.in.includes(`L${b}.mlp.proj`));
	}
});

test('pitfall 2: the U-Net\'s skip connections concatenate maps instead of adding them', () => {
	const spec = json('models/tiny-unet-seg/spec.json') as { layers: { id: string; op: string; in: string[] }[] };
	const cats = spec.layers.filter((l) => l.op === 'concat');
	assert.ok(cats.length >= 1);
	assert.ok(cats.every((c) => c.in.some((i) => i.startsWith('enc'))), 'each joins a down-path map');
});

test('quiz 1: 0.5¹⁸ is about four millionths', () => {
	assert.ok(Math.abs(0.5 ** 18 - 3.8e-6) < 1e-7);
	assert.equal(2 ** 18, 262_144);
});
