// realnvp-2d claim tests (design §4.6 "Claims are tests"): every sentence of the lesson that is not computed live is
// asserted here, on every fixture (the lesson fixtures and the adapter's edge fixtures), at full training length.
// `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/realnvp-2d/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tape, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { fit } from './trainer.ts';
import { bandShare, dataOf, forward, grid, inverse, LAYERS, lines, logp, momentsOf, net, type Params } from './nvp.ts';
import { GAP } from './process.ts';
import { cellOf } from './views/plane/meta.ts';
import manifest, { RINGS } from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const SOURCES = ['train', 'sgd1', 'flow', 'sample', 'rings', 'ringSample'];
type Run = { id: string; data: Points2D; tapes: Record<string, Tape> };
const runs = (async () => {
	const pkg = (await loadPackage('realnvp-2d', SITE)).pkg;
	const out: Run[] = [];
	await withFixtures(pkg, async (rt, fx) => {
		for (const s of SOURCES) await rt.ensure(s, 'last');
		out.push({ id: fx.id, data: dataOf(rt.input), tapes: Object.fromEntries(SOURCES.map((s) => [s, rt.sources[s]?.tape as Tape])) });
	}, { fetch: fsFetch, base: join(SITE, 'static') });
	return out;
})();
const full = async () => (await runs).filter((r) => r.data.n > 0);
const params = (t: Tape, i = t.length - 1) => (t.state(i) as { params: Params }).params;
const state = (t: Tape, i: number) => t.state(i) as { pts: Tensor; grid: Tensor; x: Tensor };
/** the probability mass of the flow's density within r of a data point, on a fine grid over ±3 */
function massNear(p: Params, d: Points2D, r: number) {
	const G = 240;
	const g = grid(G, 3);
	const lp = logp(p, g, G * G);
	const cell = (6 / G) ** 2;
	let [near, all] = [0, 0];
	for (let i = 0; i < G * G; i++) {
		const m = Math.exp(lp[i]) * cell;
		all += m;
		let j = 0;
		while (j < d.n && (g[2 * i] - d.xy[2 * j]) ** 2 + (g[2 * i + 1] - d.xy[2 * j + 1]) ** 2 >= r * r) j++;
		if (j < d.n) near += m;
	}
	return { near, all };
}

test('catalog: 8 coupling layers, each 1 -> 24 -> 24 -> 2, 5,584 weights and biases, as the params text says', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.equal(m.nets.length, LAYERS);
	assert.equal(LAYERS, 8);
	for (const n of m.nets) assert.deepEqual(n.sizes, [1, 24, 24, 2]);
	assert.equal(m.params().reduce((s, p) => s + p.size, 0), 5584);
	assert.equal(manifest.catalog?.params?.en, '8 coupling layers, each 1 → 24 → 24 → 2 (5,584 parameters)');
});

test('empty input: the sources of your points record no frame; the two-rings sources still run', async () => {
	for (const r of (await runs).filter((r) => r.data.n === 0)) {
		assert.deepEqual(['train', 'sgd1', 'flow', 'sample'].map((s) => r.tapes[s]?.length ?? 0), [0, 0, 0, 0], r.id);
		assert.ok(r.tapes.ringSample.length > 0, r.id);
	}
});

test('steps "data" and "compare": the noise is normal draws with mean near 0 and spread near 1; one sample takes one pass through the 8 layers', async () => {
	for (const r of await full()) {
		const t = r.tapes.sample;
		const z = state(t, t.find({ phase: 'noise' })).x;
		const m = momentsOf(z.data, z.shape[0]);
		assert.ok(Math.abs(m.mx) < 0.15 && Math.abs(m.my) < 0.15 && Math.abs(m.sx - 1) < 0.15 && Math.abs(m.sy - 1) < 0.15, `${r.id}: ${JSON.stringify(m)}`);
		let k = 0;
		for (let i = 0; i < t.length; i++) if (t.frame(i).phase === 'uncouple') k++;
		assert.equal(k, LAYERS, r.id);
	}
});

test('step "couple": layer 0 keeps x, and every vertical line of the grid slides and stretches as one piece (one s and one t per line)', async () => {
	for (const r of await full()) {
		const t = r.tapes.flow;
		const [a, b] = [state(t, t.find({ phase: 'start' })).grid.data, state(t, t.find({ phase: 'couple', step: 0 })).grid.data];
		const PER = 49;
		for (let l = 0; l < 13; l++) {
			const i0 = 2 * l * PER;
			const scale = (b[i0 + 3] - b[i0 + 1]) / (a[i0 + 3] - a[i0 + 1]);
			for (let q = 0; q < PER; q++) {
				const i = 2 * (l * PER + q);
				assert.equal(b[i], a[i], `${r.id}: x kept`);
				assert.ok(Math.abs(b[i + 1] - (b[i0 + 1] + scale * (a[i + 1] - a[i0 + 1]))) < 1e-9, `${r.id}: line ${l} is moved as one piece`);
			}
		}
	}
});

test('steps "invert" and "sample": (y′ − t)·e^−s gives y back exactly, layer by layer, without inverting the network', async () => {
	for (const r of await full()) {
		const p = params(r.tapes.train);
		const { z } = forward(p, r.data.xy, r.data.n);
		const back = inverse(p, z, r.data.n);
		for (let i = 0; i < back.length; i++) assert.ok(Math.abs(back[i] - r.data.xy[i]) < 1e-9, r.id);
		const { s, t } = net(p, 0, [r.data.xy[0]], 1);
		const f = r.tapes.flow.frame(r.tapes.flow.find({ phase: 'couple', step: 0 })).metrics!;
		assert.ok(Math.abs(s[0] - f.s) < 1e-15 && Math.abs(t[0] - f.t) < 1e-15 && Math.abs((f.v2 - f.t) * Math.exp(-f.s) - f.v) < 1e-12, r.id);
	}
});

test('step "logdet": the small square of the stage grows close to e^Σs times (within 10%); with 100 or more points the density’s total mass on a fine grid is near 1', async () => {
	for (const r of await full()) {
		const p = params(r.tapes.train);
		const { ld } = forward(p, r.data.xy, 1);
		const c = cellOf(p, r.data.xy[0], r.data.xy[1], LAYERS);
		assert.ok(Math.abs(c.ratio / Math.exp(ld[0]) - 1) < 0.1, `${r.id}: ${c.ratio} vs ${Math.exp(ld[0])}`);
		// a few points pull the density into spikes narrower than the grid's cells, so the sum cannot measure it there
		if (r.data.n < 100) continue;
		const { all } = massNear(p, r.data, 0);
		assert.ok(Math.abs(all - 1) < 0.05, `${r.id}: total mass ${all}`);
	}
});

test('step "step": the untrained flow is the identity (z = x, Σs = 0), so the first loss is the noise’s −log N(x) of the batch', async () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	const p = Object.fromEntries(Object.entries(m.named()).map(([k, v]) => [k, { shape: v.shape, data: v.data }]));
	for (const r of await full()) {
		const { z, ld } = forward(p, r.data.xy, r.data.n);
		assert.deepEqual(Array.from(z), Array.from(r.data.xy), r.id);
		assert.ok(ld.every((v) => v === 0), r.id);
		const t = r.tapes.sgd1;
		const b = (t.state(t.find({ phase: 'forward', step: 0 })) as { batch: { x: Tensor } }).batch.x;
		let want = 0;
		for (let i = 0; i < b.shape[0]; i++) want += (0.5 * (b.data[2 * i] ** 2 + b.data[2 * i + 1] ** 2) + Math.log(2 * Math.PI)) / b.shape[0];
		assert.ok(Math.abs(t.frame(t.find({ phase: 'loss', step: 0 })).metrics!.loss - want) < 1e-12, r.id);
	}
});

test('step "step": the first gradient-descent step moves only weights and biases of the last layers (the zeros there block the gradient to the earlier layers)', async () => {
	for (const r of await full()) {
		const t = r.tapes.sgd1;
		const grads = (t.state(t.find({ phase: 'update', step: 0 })) as { grads: Params }).grads;
		let moved = 0;
		for (const [key, g] of Object.entries(grads)) {
			const nz = g.data.filter((v) => v !== 0).length;
			if (!/^c\d+\.4\./.test(key)) assert.equal(nz, 0, `${r.id}: ${key} has a gradient`);
			moved += nz;
		}
		assert.ok(moved > 0 && moved <= LAYERS * (24 * 2 + 2), `${r.id}: ${moved}`);
	}
});

test('step "train": the mean −log p falls, and the density gathers onto your shape (more of its mass within 0.15 of your points)', async () => {
	for (const r of await full()) {
		const t = r.tapes.train;
		assert.ok(t.frame(t.length - 1).metrics!.nllAll < t.frame(0).metrics!.nllAll, r.id);
		const [a, b] = [massNear(params(t, 0), r.data, 0.15).near, massNear(params(t), r.data, 0.15).near];
		assert.ok(b > a + 0.1, `${r.id}: ${a} -> ${b}`);
	}
});

test('step "limits": far more samples than ring points land in the gap, and the flow’s density there is never 0 (thin bridges)', async () => {
	const d = dataOf(RINGS);
	for (const r of await runs) {
		const t = r.tapes.ringSample;
		const f = t.frame(t.length - 1).metrics!;
		const dgap = bandShare(d.xy, d.n, GAP[0], GAP[1]);
		assert.ok(dgap < 0.01 && f.gap > 5 * dgap, `${r.id}: samples ${f.gap}, points ${dgap}`);
		const p = params(r.tapes.rings);
		const ring = Float64Array.from({ length: 720 }, (_, i) => (i % 2 ? Math.sin : Math.cos)((Math.floor(i / 2) * 2 * Math.PI) / 360) * ((GAP[0] + GAP[1]) / 2));
		assert.ok(logp(p, ring, 360).every((v) => Number.isFinite(v)), r.id);
	}
});

test('view "warp": the grid lines and the points of a flow frame are the same function the lesson narrates', async () => {
	for (const r of await full()) {
		const p = params(r.tapes.train);
		const t = r.tapes.flow;
		const g = state(t, t.find({ phase: 'couple', last: true })).grid.data;
		const want = forward(p, lines(1.5), g.length / 2).z;
		for (let i = 0; i < g.length; i += 97) assert.ok(Math.abs(g[i] - want[i]) < 1e-12, r.id);
	}
});
