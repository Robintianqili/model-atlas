// score-2d claim tests (design §4.6 "Claims are tests"): every sentence of the lesson that is not computed live is
// asserted here, on every fixture (the lesson fixtures and the adapter's edge fixtures), at full training length.
// `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/score-2d/claims.test.ts
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
import { chains, dataOf, drawPair, exact, grid, limitsOf, probeOf, score, SHOW, SIGMAS, SMAX, SMIN, SPAN, type Params } from './score.ts';
import { pairsOf } from './views/plane/meta.ts';
import manifest, { ALPHA } from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const SOURCES = ['train', 'sgd1', 'walk', 'plain', 'annealed', 'still'];
type Run = { id: string; data: Points2D; tapes: Record<string, Tape> };
const runs = (async () => {
	const pkg = (await loadPackage('score-2d', SITE)).pkg;
	const out: Run[] = [];
	await withFixtures(pkg, async (rt, fx) => {
		for (const s of SOURCES) await rt.ensure(s, 'last');
		out.push({ id: fx.id, data: dataOf(rt.input), tapes: Object.fromEntries(SOURCES.map((s) => [s, rt.sources[s]?.tape as Tape])) });
	}, { fetch: fsFetch, base: join(SITE, 'static') });
	return out;
})();
const full = async () => (await runs).filter((r) => r.data.n > 0);
const last = (t: Tape) => t.state(t.length - 1) as { params: Params; x: Tensor };
const end = (t: Tape) => t.frame(t.find({ phase: 'move', last: true })).metrics!;
const cos = (a: number, b: number, c: number, d: number) => (a * c + b * d) / (Math.hypot(a, b) * Math.hypot(c, d));
/** the nearest data point to (x, y) and its distance */
function nearestOf(d: Points2D, x: number, y: number) {
	let [j, best] = [0, Infinity];
	for (let k = 0; k < d.n; k++) {
		const r = Math.hypot(x - d.xy[2 * k], y - d.xy[2 * k + 1]);
		if (r < best) [j, best] = [k, r];
	}
	return { x: d.xy[2 * j], y: d.xy[2 * j + 1], r: best };
}
/** the mean cosine between the learned score at σ and `want(x, y)` over the grid places where `keep(r)` holds */
function agree(p: Params, d: Points2D, s: number, keep: (r: number) => boolean, want: (x: number, y: number) => [number, number]) {
	const g = grid(30, SPAN);
	const v = score(p, g, s, 900);
	const cs: number[] = [];
	for (let i = 0; i < 900; i++) if (keep(nearestOf(d, g[2 * i], g[2 * i + 1]).r)) cs.push(cos(v[2 * i], v[2 * i + 1], ...want(g[2 * i], g[2 * i + 1])));
	return cs.reduce((a, b) => a + b, 0) / cs.length;
}

test('catalog: the network is 3 -> 64 -> 64 -> 2 with 4,546 weights and biases, as the params text says', () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.deepEqual(m.sizes, [3, 64, 64, 2]);
	assert.equal(m.params().reduce((s, p) => s + p.size, 0), 4546);
	assert.equal(manifest.catalog?.params?.en, '3 → 64 → 64 → 2 (4,546 parameters)');
});

test('empty input: no source records a frame (the stage shows the adapter note instead)', async () => {
	for (const r of (await runs).filter((r) => r.data.n === 0)) assert.deepEqual(SOURCES.map((s) => r.tapes[s]?.length ?? 0), [0, 0, 0, 0, 0, 0], r.id);
});

test('step "step": d reads (x̃, ỹ, log σ), its target is −ε, so x̃ + σ·target is one of your points (the copy came from it)', async () => {
	const m = fit.model(createRng(1).fork('init'), fit.defaults);
	assert.deepEqual(m.sizes, [3, 64, 64, 2]);
	for (const r of await full()) {
		const t = r.tapes.sgd1;
		const q = pairsOf({ at: { index: 0, state: t.state(t.find({ phase: 'forward', step: 0 })) } })!;
		for (let i = 0; i < q.n; i++) {
			assert.ok(q.s[i] >= SMIN - 1e-12 && q.s[i] <= SMAX + 1e-12, `${r.id}: σ ${q.s[i]} in [${SMIN}, ${SMAX}]`);
			const d = nearestOf(r.data, q.x[2 * i], q.x[2 * i + 1]);
			assert.ok(d.r < 1e-9, `${r.id}: copy ${i} came from a data point`);
		}
	}
});

test('step "score": the arrow at the ringed point points toward the nearest part of your shape, the nearest point has the largest weight, and farther out it is longer', async () => {
	for (const r of await full()) {
		const [px, py] = probeOf(r.data);
		const e = exact(r.data, px, py, SHOW);
		const n = nearestOf(r.data, px, py);
		assert.ok(cos(e.sx, e.sy, n.x - px, n.y - py) > 0.8, `${r.id}: toward the nearest part`);
		assert.ok(Math.hypot(r.data.xy[2 * e.j] - n.x, r.data.xy[2 * e.j + 1] - n.y) < 1e-12, `${r.id}: the nearest point weighs most`);
		const f = exact(r.data, px + 0.7, py, SHOW);
		assert.ok(Math.hypot(f.sx, f.sy) > Math.hypot(e.sx, e.sy), `${r.id}: longer farther out`);
		assert.ok(f.density > 0, `${r.id}: the density far out is not exactly 0`);
	}
});

test('step "noise": the best guess of the way back, (x − x̃) / σ² averaged over the points a copy could come from, is the gradient of log p (finite differences)', async () => {
	for (const r of await full())
		for (const [x, y] of [probeOf(r.data), [0.1, -0.2], [-1.5, 2]] as [number, number][]) {
			const h = 1e-5;
			const lp = (u: number, v: number) => Math.log(exact(r.data, u, v, SHOW).density);
			const [gx, gy] = [(lp(x + h, y) - lp(x - h, y)) / (2 * h), (lp(x, y + h) - lp(x, y - h)) / (2 * h)];
			const e = exact(r.data, x, y, SHOW);
			assert.ok(Math.hypot(gx - e.sx, gy - e.sy) <= 1e-5 * Math.max(1, Math.hypot(e.sx, e.sy)), `${r.id} at (${x}, ${y}): ${gx}, ${gy} vs ${e.sx}, ${e.sy}`);
		}
});

test('step "field": the loss stays above 0 with more than one point; at the largest σ the arrows point to the middle, at the smallest to the nearest part', async () => {
	for (const r of await full()) {
		const p = last(r.tapes.train).params;
		const [mx, my] = [0, 1].map((c) => Array.from({ length: r.data.n }, (_, j) => r.data.xy[2 * j + c]).reduce((a, b) => a + b, 0) / r.data.n);
		const big = agree(p, r.data, SMAX, (d) => d > 1, (x, y) => [mx - x, my - y]);
		const small = agree(p, r.data, SMIN, (d) => d > 0.3, (x, y) => { const n = nearestOf(r.data, x, y); return [n.x - x, n.y - y]; });
		const t = r.tapes.train;
		const ls: number[] = [];
		for (let i = 0; i < t.length; i++) if (t.frame(i).phase === 'step') ls.push(t.frame(i).metrics!.loss);
		const tail = ls.slice(-50).reduce((a, b) => a + b, 0) / 50;
		console.log(`  ${r.id}: n ${r.data.n}, cos σmax ${big.toFixed(3)}, cos σmin ${small.toFixed(3)}, last-50 loss ${tail.toFixed(3)}`);
		assert.ok(big > 0.9 && small > 0.9, `${r.id}: ${big} ${small}`);
		if (r.data.n > 1) assert.ok(tail > 0.2, `${r.id}: loss floor ${tail}`);
	}
});

test('steps "anneal" and "limits": at σ = 1.28 the arrows are about as strong as the true score everywhere; at σ = 0.04 far out at least twice too short but roughly the right way', async () => {
	for (const r of await full()) {
		const p = last(r.tapes.train).params;
		const g = grid(30, SPAN);
		const v = score(p, g, SMAX, 900);
		const ratios = Array.from({ length: 900 }, (_, i) => { const e = exact(r.data, g[2 * i], g[2 * i + 1], SMAX); return Math.hypot(e.sx, e.sy) / Math.hypot(v[2 * i], v[2 * i + 1]); }).sort((a, b) => a - b);
		const big = ratios[450];
		const s = limitsOf(p, r.data, SMIN);
		console.log(`  ${r.id}: σmax true/network ${big.toFixed(2)}; σmin ${s.times.toFixed(2)}x ${s.angle.toFixed(1)}° share ${s.share.toFixed(3)}`);
		assert.ok(big > 0.5 && big < 2 && s.times > 2 && s.angle < 30, `${r.id}`);
	}
});

test('step "limits": hardly any noisy copy at σ ≤ 0.05 lands more than 3σ from your points', async () => {
	for (const r of await full()) {
		const rng = createRng(1).fork('batch');
		let [far, all] = [0, 0];
		for (let k = 0; k < 20 * 256; k++) {
			const q = drawPair(r.data, rng);
			if (q.s > 0.05) continue;
			all++;
			if (nearestOf(r.data, q.x + q.s * q.ex, q.y + q.s * q.ey).r > 3 * q.s) far++;
		}
		assert.ok(all > 100 && far / all < 0.03, `${r.id}: ${far} of ${all}`);
	}
});

test('step "flow": without the noise term the annealed chains only climb: they draw together at σ = 1.28 and end in a few tight bunches', async () => {
	for (const r of await full()) {
		const t = r.tapes.still;
		const T = (t.frame(t.find({ phase: 'move', last: true })).step! + 1) / SIGMAS.length;
		const s0 = t.frame(t.find({ phase: 'start' })).metrics!.spread;
		const s1 = t.frame(t.find({ phase: 'move', step: T - 1 })).metrics!.spread;
		const x = last(t).x;
		const spots = new Set(Array.from({ length: x.shape[0] }, (_, i) => `${Math.round(x.data[2 * i] / 0.05)},${Math.round(x.data[2 * i + 1] / 0.05)}`)).size;
		const a = last(r.tapes.annealed).x;
		const spotsA = new Set(Array.from({ length: a.shape[0] }, (_, i) => `${Math.round(a.data[2 * i] / 0.05)},${Math.round(a.data[2 * i + 1] / 0.05)}`)).size;
		console.log(`  ${r.id}: spread ${s0.toFixed(3)} -> ${s1.toFixed(3)}; places (0.05 cells) without noise ${spots}, with ${spotsA}`);
		assert.ok(s1 < 0.5 * s0 && spots <= 0.2 * spotsA, `${r.id}`);
		assert.ok(end(t).cover <= end(r.tapes.annealed).cover, `${r.id}: cover`);
	}
});

test('view "chains": its sliders re-run the same function as the walk source (with the source’s options: the same frames)', async () => {
	const r = (await full())[0];
	const t = r.tapes.walk;
	const it = chains(last(r.tapes.train).params, { sigmas: [SHOW], T: 100, alpha: ALPHA, noise: 1, chains: 300 }, createRng(1));
	let k = 0;
	for (const m of it) assert.deepEqual(Array.from(m.x), Array.from((t.state(k++) as { x: Tensor }).x.data));
	assert.equal(k, t.length);
});
