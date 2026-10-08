// adversarial-tinycnn claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is
// not computed live is asserted here, over every fixture (the lesson's ε fixtures, the adapter's blank / dot / small, the
// 10 presets) through withFixtures. `atlas check` runs this file (AK408).
// Run alone: node --test src/lib/explainers/adversarial-tinycnn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime, type AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import type { Tape } from '../../atlas/types/index.ts';
import manifest from './explainer.ts';
import { CONCEPT_BY_SLUG } from '../../catalog/concepts.ts';
import { EPS, attack } from './process.ts';
import { N, lossGrad, type Weights } from './net.ts';
import { TABLE_EPS, stateOf, type St } from './read.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('adversarial-tinycnn', SITE).then((r) => r.pkg);

async function tapeOf(rt: AtlasRuntime): Promise<Tape> {
	await rt.ensure('attack', 'last');
	return rt.sources.attack.tape as Tape;
}
const frames = (t: Tape) => Array.from({ length: t.length }, (_, i) => ({ f: t.frame(i), s: t.state(i) as St }));
const weights = (rt: AtlasRuntime) => rt.sources.fwd.model!.weights as Weights;

test('step "drawing": TinyCNN reads 28×28 numbers; the page reuses its shipped model', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const spec = rt.sources.fwd.model!.spec;
	assert.deepEqual(spec.inputs![0].shape, [1, 28, 28]);
	assert.equal(spec.slug, 'tinycnn');
	assert.equal(spec.params, 6722);
	assert.equal(manifest.catalog!.params!.en.startsWith(spec.params.toLocaleString('en-US')), true);
	rt.close();
});

test('pre-question: on most test drawings, PGD with every pixel moved by at most a tenth of the ink scale flips the answer', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.equal(TABLE_EPS, 0.1);
	assert.ok(rt.num('fact:pgd10') > 0.5, 'most');
	assert.ok(rt.num('fact:cleanAcc') > 0.95, '"almost all of its test drawings right"');
	assert.equal(manifest.page.prequestion.answer, 1);
	rt.close();
});

test('predict "grad": empty paper has a gradient, on every fixture; the blank pad: most pixels can be pushed', async () => {
	await withFixtures(await pkgP, async (rt, fx) => {
		const g = stateOf(await tapeOf(rt), 'grad')!, x = g.x!.data, gr = g.g!.data;
		let paper = 0, pushed = 0, nonzero = 0;
		for (let i = 0; i < N; i++) {
			if (gr[i] !== 0) nonzero++;
			if (x[i] === 0) (paper++, gr[i] !== 0 && pushed++);
		}
		assert.ok(paper > 0 && pushed > 0, 'some pixel of empty paper has a non-zero gradient');
		if (fx.id === 'blank') assert.ok(nonzero > N / 2, 'try-this "blank": most pixels can still be pushed');
	}, opts);
});

test('step "grad": g is how fast the loss grows with a little more ink (a finite difference agrees)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt), c = t.frame(0).metrics!, s = stateOf(t, 'grad')!, x = Float64Array.from(s.x!.data);
		const gs = Array.from(s.g!.data), i = gs.reduce((b, v, k) => (Math.abs(v) > Math.abs(gs[b]) ? k : b), 0), h = 1e-6;
		const up = Float64Array.from(x), dn = Float64Array.from(x);
		up[i] += h;
		dn[i] -= h;
		const fd = (lossGrad(up, c.y, weights(rt)).loss - lossGrad(dn, c.y, weights(rt)).loss) / (2 * h);
		assert.ok(Math.abs(fd - s.g!.data[i]) <= 1e-3 * Math.abs(s.g!.data[i]) + 1e-12, `finite difference ${fd} vs ${s.g!.data[i]}`);
	}, { ...opts, only: ['blank', 'dot', 'small', 'tree', 'car', 'sun'] });
});

test('step "sign": to first order, moving every pixel ε along its sign gains the most of any change within ε', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const s = stateOf(await tapeOf(rt), 'grad')!, g = Array.from(s.g!.data);
		const best = g.reduce((a, v) => a + Math.abs(v), 0);
		let seed = 7;
		for (let k = 0; k < 50; k++) {
			let d = 0;
			for (let i = 0; i < N; i++) {
				seed = (seed * 1103515245 + 12345) % 2147483648;
				d += g[i] * (2 * (seed / 2147483648) - 1); // any change with |δᵢ| ≤ 1
			}
			assert.ok(d <= best + 1e-15, 'no change in the box beats ε·sign(g) to first order');
		}
	}, { ...opts, only: ['blank', 'tree', 'sun'] });
});

test('steps "step", "flip", "pgd", quiz 2: paper only goes up, ink only down, no pixel moves more than ε, PGD stays in the box', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt), x = t.state(0) as St;
		for (const { f, s } of frames(t)) {
			if (f.phase !== 'fgsm' && f.phase !== 'pgd') continue;
			const eps = f.metrics!.eps, a = s.adv!.data, x0 = x.x!.data;
			for (let i = 0; i < N; i++) {
				assert.ok(a[i] >= 0 && a[i] <= 1, '0…1');
				assert.ok(Math.abs(a[i] - x0[i]) <= eps + 1e-12, 'within ε');
				if (f.phase === 'fgsm' && x0[i] === 0) assert.ok(a[i] === 0 || Math.abs(a[i] - eps) < 1e-15, 'paper only goes up, to ε');
				if (f.phase === 'fgsm' && x0[i] === 1) assert.ok(a[i] <= 1, 'ink only down');
			}
		}
	}, opts);
});

test('step "flip": the change is spread over the whole grid (changed pixels in all four quarters), not a new stroke', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt), x = (t.state(0) as St).x!.data, f = stateOf(t, 'fgsm')!, a = f.adv!.data;
		const quarters = new Set<number>();
		for (let i = 0; i < N; i++) if (a[i] !== x[i]) quarters.add((Math.floor(i / 28) < 14 ? 0 : 2) + ((i % 28) < 14 ? 0 : 1));
		assert.equal(quarters.size, 4);
	}, { ...opts, only: ['blank', 'dot', 'small', 'tree', 'car', 'apple', 'sun', 'big-eps'] });
});

test('the start drawing: one FGSM step at the default ε flips it (the "flip" step and the misconception)', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const t = await tapeOf(rt);
	const f = Array.from({ length: t.length }, (_, i) => t.frame(i)).find((q) => q.phase === 'fgsm')!;
	assert.equal(f.metrics!.eps, EPS);
	assert.equal(f.metrics!.flipped, 1);
	rt.close();
});

test('misconception: one FGSM step moves hundreds of pixels on every preset drawing', async () => {
	const pkg = await pkgP;
	const probe = await createRuntime(pkg, opts);
	const ids = probe.presets.map((p) => p.id);
	probe.close();
	await withFixtures(pkg, async (rt) => {
		const t = await tapeOf(rt), x = (t.state(0) as St).x!.data, a = stateOf(t, 'fgsm')!.adv!.data;
		let moved = 0;
		for (let i = 0; i < N; i++) if (a[i] !== x[i]) moved++;
		assert.ok(moved >= 200, `${moved} pixels moved`);
	}, { ...opts, only: ids });
});

test('try-this "sure": TinyCNN is almost completely sure of the sun, its gradient is tiny, FGSM never flips it, PGD does', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const t = await tapeOf(rt), scan = t.frame(t.length - 1).metrics!;
		assert.ok(t.frame(0).metrics!.py > 0.999, 'almost completely sure');
		assert.ok(t.frame(1).metrics!.gmax < 1e-6, 'tiny gradient');
		assert.equal(scan.found, 0, 'one FGSM step never flips it anywhere on the slider');
		assert.ok(scan.pgdFlip > 0, 'PGD does');
	}, { ...opts, only: ['sun'] });
});

test('step "defend": the planned concept page shows adversarial training on a small 2-D network, and links back here', () => {
	const c = CONCEPT_BY_SLUG['adversarial-robustness'];
	assert.ok(c, 'the concept exists (the view links /train/adversarial-robustness/)');
	assert.equal(c.status, 'planned');
	assert.ok(/adversarial-training toggle on the 2D MLP/.test(c.interaction.en));
	assert.ok(c.links.includes('adversarial-tinycnn'));
});

test('step "defend": TinyCNN was trained without adversarial training', () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'tinycnn', 'spec.json'), 'utf8'));
	const recipe = JSON.stringify(spec.training).toLowerCase();
	for (const w of ['adversarial', 'fgsm', 'pgd']) assert.ok(!recipe.includes(w), w);
});

test('the process: PGD defaults are 10 steps of ε/4, and the scan grid is the slider (0.005 to 0.3)', () => {
	assert.deepEqual(attack.defaults, { steps: 10, alphaFrac: 0.25, scanStep: 0.005, scanMax: 0.3 });
});
