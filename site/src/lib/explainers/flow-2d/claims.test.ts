// flow-2d claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here. The fixtures (lesson fixtures, the adapter's edge fixtures, the presets) run once, in `runs`,
// and every test reads the numbers it needs from there. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/flow-2d/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { record } from '../../atlas/process/record.ts';
import { loadTape } from '../../atlas/process/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { fit } from './trainer.ts';
import flow from './process.ts';
import { dataOf, showcase, velocity, type Params } from './flow.ts';
import { presetOf } from './adapters/points2d/adapter.ts';
import { middleOf, pairsOf, predOf, runEnd, spreadAround } from './views/plane/meta.ts';
import { wireOf } from './lesson.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
type St = { params?: Params; batch?: Record<string, Tensor>; x?: Tensor };
const stateAt = (t: Tape, c: Parameters<Tape['find']>[0]) => t.state(t.find(c)) as St;
const at = (t: Tape, c: Parameters<Tape['find']>[0]) => ({ index: t.find(c), frame: t.frame(t.find(c)), state: t.state(t.find(c)), tape: t });
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
/** training losses before the final frame, in order */
const losses = (t: Tape) => {
	const out: number[] = [];
	for (let i = 0; i < t.length; i++) {
		const f = t.frame(i);
		if ((f.phase === 'step' || f.phase === 'snapshot') && typeof f.metrics?.loss === 'number' && t.find({ phase: 'snapshot', last: true }) !== i) out.push(f.metrics.loss);
	}
	return out;
};
const mse = (p: Params, b: Record<string, Tensor>) => {
	const n = b.inp.shape[0];
	const xy = Float64Array.from({ length: 2 * n }, (_, k) => b.inp.data[3 * (k >> 1) + (k & 1)]);
	const v = velocity(p, xy, Float64Array.from({ length: n }, (_, i) => b.inp.data[3 * i + 2]), n);
	return v.reduce((s, x, k) => s + (x - b.target.data[k]) ** 2, 0) / (2 * n);
};

/** every fixture once (the cheap sources), and the presets with the 1,500-step run: the tapes the tests read */
type Run = { id: string; raw: unknown; input?: Float64Array; sgd1: Tape; t0: Tape; flow: Tape; job?: Record<string, { data: ArrayLike<number> }> };
const PRESETS = ['preset:smiley', 'preset:moons', 'preset:ring'];
const collect = async (sources: string[], only?: string[]) => {
	const pkg = (await loadPackage('flow-2d', SITE)).pkg;
	const out: Run[] = [];
	await withFixtures(
		pkg,
		async (rt, fx) => {
			for (const s of sources) await rt.ensure(s, 'last');
			const tape = (s: string) => rt.sources[s]?.tape as Tape;
			out.push({ id: fx.id, raw: rt.input, input: rt.pre?.tensors.input?.data as Float64Array | undefined, sgd1: tape('sgd1'), t0: tape('t0'), flow: tape('flow'), job: rt.sources.job?.trace?.t });
		},
		{ ...opts, only }
	);
	return out;
};
const runs = collect(['sgd1', 't0', 'job']);
const flows = collect(['flow'], PRESETS);
const full = async () => (await runs).filter((r) => r.sgd1.length > 0);

test('empty input: no source records a frame (the stage shows the adapter note instead)', async () => {
	const pkg = (await loadPackage('flow-2d', SITE)).pkg;
	await withFixtures(pkg, async (rt) => {
		for (const s of ['sgd1', 't0', 'flow']) await rt.ensure(s, 'last');
		assert.deepEqual(['sgd1', 't0', 'flow'].map((s) => rt.sources[s].tape?.length ?? 0), [0, 0, 0]);
	}, { ...opts, only: ['edge:empty'] });
});

test('network and quiz: v reads three numbers (x, y, t), has 3 -> 64 -> 64 -> 64 -> 2 layers and 8,706 weights and biases', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.equal(m.params().reduce((s, p) => s + p.size, 0), 8706);
	assert.deepEqual(m.sizes, [3, 64, 64, 64, 2]);
});

test('pairs and the job: the followed pair of the traced input is its row of the first batch, and the inputs are only (x_t, y_t, t)', async () => {
	for (const r of await full()) {
		const b = stateAt(r.sgd1, { phase: 'forward', step: 0 }).batch!;
		assert.deepEqual(b.inp.shape, [256, 3], r.id);
		const k = showcase(r.raw)!.i;
		assert.deepEqual(Array.from(r.input!), Array.from(b.inp.data.slice(3 * k, 3 * k + 3)), r.id);
		const p = pairsOf({ at: at(r.sgd1, { phase: 'forward', step: 0 }) })!;
		// x_t = (1 − t)·x0 + t·x1 and target = x1 − x0 for every pair (to rounding)
		for (let i = 0; i < p.n; i++) for (let c = 0; c < 2; c++) assert.ok(Math.abs((1 - p.t[i]) * p.x0[2 * i + c] + p.t[i] * p.x1[2 * i + c] - b.inp.data[3 * i + c]) < 1e-12, r.id);
	}
});

test('noise: the share of x and y values between −2 and 2 is near the bell curve’s 95% ("as the bell curve predicts", glossary)', async () => {
	for (const r of await full()) {
		const p = pairsOf({ at: at(r.sgd1, { phase: 'forward', step: 0 }) })!;
		const inside = Array.from(p.x0).filter((v) => Math.abs(v) < 2).length / p.x0.length;
		assert.ok(inside > 0.9 && inside < 0.99, `${r.id}: ${inside}`);
	}
});

test('pairs: one data point may be picked more than once (it happens in the first batch of every preset)', async () => {
	for (const r of (await full()).filter((x) => x.id.startsWith('preset:'))) {
		const p = pairsOf({ at: at(r.sgd1, { phase: 'forward', step: 0 }) })!;
		const keys = new Set<string>();
		for (let i = 0; i < p.n; i++) keys.add(`${p.x1[2 * i].toFixed(9)},${p.x1[2 * i + 1].toFixed(9)}`);
		assert.ok(keys.size < p.n, r.id);
	}
});

test('slider: training gives each pair its own random t', async () => {
	for (const r of await full()) {
		const p = pairsOf({ at: at(r.sgd1, { phase: 'forward', step: 0 }) })!;
		assert.equal(new Set(p.t).size, p.n, r.id);
	}
});

test('network: right now every weight is random (t0 holds the starting weights of the training runs)', async () => {
	for (const r of await full()) {
		const a = stateAt(r.t0, 'last').params!;
		for (const k of Object.keys(a)) {
			assert.deepEqual(Array.from(a[k].data), Array.from(stateAt(r.sgd1, { phase: 'init' }).params![k].data), `${r.id} ${k}`);
		}
		// the live-model trace (f32 weights) gives the arrow these f64 weights give, to f32 precision
		const v = velocity(a, r.input!.slice(0, 2), r.input![2], 1);
		assert.ok(Math.abs(r.job!['net.6'].data[0] - v[0]) < 1e-5 && Math.abs(r.job!['net.6'].data[1] - v[1]) < 1e-5, r.id);
	}
});

test('arrow (predict): the untrained arrow is less than a tenth as long as the target, but not zero', async () => {
	for (const r of await full()) {
		const a = at(r.sgd1, { phase: 'forward', step: 0 });
		const k = showcase(r.raw)!.i;
		const [v, p] = [predOf({ at: a }, k)!, pairsOf({ at: a })!];
		const [len, target] = [Math.hypot(v[0], v[1]), Math.hypot(p.target[2 * k], p.target[2 * k + 1])];
		assert.ok(len < 0.1 * target && len > 1e-3, `${r.id}: ${len} vs ${target}`);
	}
});

test('update: the weight with the largest gradient is described by what it connects (net.6 is the last layer, rows = the arrow’s x and y)', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.deepEqual(Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, v.shape])), { 'net.0.weight': [64, 3], 'net.0.bias': [64], 'net.2.weight': [64, 64], 'net.2.bias': [64], 'net.4.weight': [64, 64], 'net.4.bias': [64], 'net.6.weight': [2, 64], 'net.6.bias': [2] });
	assert.deepEqual(wireOf('net.6.weight', [0, 31]), { en: 'neuron 31 of the third layer of 64 to the arrow’s x (neurons counted from 0)', zh: '第 3 层（64 个神经元）的 31 号神经元和箭头的 x（神经元从 0 开始数）' });
	assert.deepEqual(wireOf('net.0.weight', [5, 2]).en, 'the input t to neuron 5 of the first layer of 64 (neurons counted from 0)');
});

test('update: one gradient-descent step lowers the loss on the same batch', async () => {
	for (const r of await full()) {
		const b = stateAt(r.sgd1, { phase: 'forward', step: 0 }).batch!;
		const before = r.sgd1.frame(r.sgd1.find({ phase: 'loss', step: 0 })).metrics!.loss;
		assert.ok(Math.abs(mse(stateAt(r.sgd1, { phase: 'init' }).params!, b) - before) < 1e-12, r.id);
		assert.ok(mse(stateAt(r.sgd1, { phase: 'update', step: 0 }).params!, b) < before, r.id);
	}
});

test('watch (presets): the loss falls; where it stays above 0.25 it has levelled off ("and no lower"); training starts from the weights of t0', async () => {
	for (const r of await flows) {
		const t0 = (await runs).find((x) => x.id === r.id)!.t0;
		for (const [k, v] of Object.entries(stateAt(t0, 'last').params!)) assert.deepEqual(Array.from(v.data), Array.from(stateAt(r.flow, { phase: 'init' }).params![k].data), `${r.id} ${k}`);
		const l = losses(r.flow);
		const [l0, l1] = [mean(l.slice(0, 100)), mean(l.slice(-100))];
		assert.ok(l1 < l0, `${r.id}: ${l0} -> ${l1}`);
		if (l1 > 0.25) assert.ok(Math.abs(mean(l.slice(-300, -200)) - l1) < 0.05, `${r.id}: ${mean(l.slice(-300, -200))} vs ${l1}`);
	}
});

test('euler (presets): x <- x + h·v(x, t) is the rule that ran, and the points end closer to the data than the noise was', async () => {
	for (const r of await flows) {
		const [x0, x1] = [stateAt(r.flow, { phase: 'noise' }).x!.data, stateAt(r.flow, { phase: 'move', step: 0 }).x!.data];
		const v = velocity(stateAt(r.flow, { phase: 'noise' }).params!, x0, 0, x0.length / 2);
		for (let i = 0; i < x0.length; i++) assert.equal(x1[i], x0[i] + 0.02 * v[i], r.id);
		const e = runEnd({ at: at(r.flow, 'last') }, 'move')!;
		assert.ok(e.dist < e.dist0, `${r.id}: ${e.dist} vs ${e.dist0}`);
	}
});

test('misconception (presets): samples are new points, not copies of the data', async () => {
	for (const r of await flows) {
		const x = runEnd({ at: at(r.flow, 'last') }, 'move')!.x.data;
		const d = dataOf(r.raw);
		let m = Infinity;
		for (let i = 0; i < x.length / 2; i++) for (let j = 0; j < d.n; j++) m = Math.min(m, Math.hypot(x[2 * i] - d.xy[2 * j], x[2 * i + 1] - d.xy[2 * j + 1]));
		assert.ok(m > 1e-4, `${r.id}: ${m}`);
	}
});

test('try this (ring): the samples spread around the whole ring (every eighth of the circle gets at least 5%); noise from outside moves in, from the middle out', async () => {
	const r = (await flows).find((x) => x.id === 'preset:ring')!;
	const x = runEnd({ at: at(r.flow, 'last') }, 'move')!.x.data;
	const n = x.length / 2;
	const bins = new Array(8).fill(0);
	for (let i = 0; i < n; i++) bins[Math.floor(((Math.atan2(x[2 * i + 1], x[2 * i]) + Math.PI) / (2 * Math.PI)) * 8) % 8]++;
	assert.ok(bins.every((b) => b >= 0.05 * n), bins.join(','));
	// "noise from outside moves in; noise from the middle moves out": radius before and after, around the ring's middle
	const m = middleOf(dataOf(r.raw));
	const x0 = stateAt(r.flow, { phase: 'noise' }).x!.data;
	const rad = (a: ArrayLike<number>, i: number) => Math.hypot(a[2 * i] - m[0], a[2 * i + 1] - m[1]);
	const [inner, outer] = [[] as number[], [] as number[]];
	for (let i = 0; i < n; i++) (rad(x0, i) < 0.5 ? inner : rad(x0, i) > 1 ? outer : []).push(i);
	const share = (ids: number[], moved: (i: number) => boolean) => ids.filter(moved).length / ids.length;
	assert.ok(inner.length > 10 && share(inner, (i) => rad(x, i) > rad(x0, i)) >= 0.9, `inner ${inner.length}: ${share(inner, (i) => rad(x, i) > rad(x0, i))}`);
	assert.ok(outer.length > 100 && share(outer, (i) => rad(x, i) < rad(x0, i)) >= 0.9, `outer ${outer.length}: ${share(outer, (i) => rad(x, i) < rad(x0, i))}`);
});

test('few (presets, at 1,500 and at 200 steps): at t = 0 almost every arrow points to the middle; big steps pull the points inward; small steps follow the arrows as they turn (the full run a learner gets)', async (tc) => {
	const runs = [...(await flows).map((r) => ({ id: r.id, raw: r.raw, t: r.flow })), ...['smiley', 'moons', 'ring'].map((id) => ({ id: id + '@200', raw: presetOf(id), t: record(flow, presetOf(id), { steps: 200 }, { seed: 1 }) }))];
	for (const { id, raw, t } of runs) {
		const m = middleOf(dataOf(raw));
		const x0 = stateAt(t, { phase: 'noise' }).x!.data;
		const n = x0.length / 2;
		const p = stateAt(t, { phase: 'noise' }).params!;
		// every Euler step, big or small, moves along the arrow where it starts: the first big step is x0 + h·v(x0, 0)
		const [v, f1] = [velocity(p, x0, 0, n), stateAt(t, { phase: 'moveFew', step: 0 }).x!.data];
		for (let i = 0; i < 2 * n; i++) assert.equal(f1[i], x0[i] + 0.25 * v[i], id);
		// at t = 0 at least 98% of the arrows at the noise points are within cos 0.9 of the direction to the middle
		let to = 0;
		for (let i = 0; i < n; i++) {
			const [ex, ey] = [m[0] - x0[2 * i], m[1] - x0[2 * i + 1]];
			if ((v[2 * i] * ex + v[2 * i + 1] * ey) / (Math.hypot(v[2 * i], v[2 * i + 1]) * Math.hypot(ex, ey)) > 0.9) to++;
		}
		assert.ok(to / n >= 0.98, `${id}: ${to / n}`);
		// 4 big steps end nearer the middle than 50 small ones (the narration's "pull the points inward" branch)
		const [a, b] = [runEnd({ at: at(t, 'last') }, 'move')!, runEnd({ at: at(t, 'last') }, 'moveFew')!];
		const [r4, r50] = [spreadAround(b.x.data, n, m), spreadAround(a.x.data, n, m)];
		assert.ok(r4 < 0.9 * r50, `${id}: ${r4} vs ${r50}`);
		// the small steps turn (full runs; the e2e cap's 200 steps leave the paths straighter): for most points the
		// last of the 50 moves points more than 15° away from the first
		const [s0, s1, e0, e1] = [stateAt(t, { phase: 'noise' }).x!.data, stateAt(t, { phase: 'move', step: 0 }).x!.data, stateAt(t, { phase: 'move', step: 48 }).x!.data, a.x.data];
		let bent = 0;
		for (let i = 0; i < n; i++) {
			const [ux, uy, wx, wy] = [s1[2 * i] - s0[2 * i], s1[2 * i + 1] - s0[2 * i + 1], e1[2 * i] - e0[2 * i], e1[2 * i + 1] - e0[2 * i + 1]];
			if ((ux * wx + uy * wy) / (Math.hypot(ux, uy) * Math.hypot(wx, wy)) < Math.cos((15 * Math.PI) / 180)) bent++;
		}
		tc.diagnostic(`${id}: arrows to the middle at t = 0 ${to / n}; middle distance 4 steps ${r4.toFixed(3)}, 50 steps ${r50.toFixed(3)}; bent ${bent / n}`);
		if (!id.endsWith('@200')) assert.ok(bent / n > 0.5, `${id}: ${bent / n}`);
	}
});

test('phones: the precomputed tape is the live run on the start input (same frames)', async () => {
	const { readFileSync } = await import('node:fs');
	const dir = join(SITE, 'static', 'models', 'flow-2d', 'tapes');
	const header = JSON.parse(readFileSync(join(dir, 'flow.tape.json'), 'utf8'));
	const bin = readFileSync(join(dir, 'flow.tape.bin'));
	const pre = loadTape(header, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer, flow);
	const live = (await flows).find((x) => x.id === 'preset:smiley')!.flow;
	assert.equal(pre.length, live.length);
	for (let i = 0; i < live.length; i++) {
		const [a, b] = [pre.frame(i), live.frame(i)];
		assert.deepEqual([a.phase, a.step, a.metrics], [b.phase, b.step, b.metrics], `frame ${i}`);
	}
	assert.deepEqual(Array.from(stateAt(pre, 'last').x!.data), Array.from(stateAt(live, 'last').x!.data));
	const [pa, pb] = [stateAt(pre, { phase: 'snapshot', last: true }).params!, stateAt(live, { phase: 'snapshot', last: true }).params!];
	for (const k of Object.keys(pb)) assert.deepEqual(Array.from(pa[k].data), Array.from(pb[k].data), k);
});
