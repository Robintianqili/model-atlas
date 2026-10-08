// vae-2d claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the border fixture, the adapter's blank / dot / small, the 7 presets) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/vae-2d/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { dims } from './lesson.ts';
import { RANGE, T, atBoundary, codeOf, decode, epsOf, extraOf, greyShare, klFromLog, latentOf, lossOf, muOf, near, withExtra, type XY } from './vae.ts';
import { targetOf } from './views/latent/meta.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('vae-2d', SITE).then((r) => r.pkg);
/** pipeline/vae-2d/train.py from the site folder (a lane's or the main one) */
const trainPy = () => {
	let d = SITE;
	for (let i = 0; i < 6 && !existsSync(join(d, 'pipeline', 'vae-2d', 'train.py')); i++) d = join(d, '..');
	return readFileSync(join(d, 'pipeline', 'vae-2d', 'train.py'), 'utf8');
};
const KINDS = ['star', 'ladder', 'umbrella', 'eyeglasses', 't-shirt'];
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

test('step "input" and the page texts: 784 → 2 numbers, five kinds, 60,000 training drawings, the parameter count', async () => {
	const rt = await createRuntime(await pkgP, opts);
	assert.deepEqual(dims(rt), { n: 784, h1: 256, h2: 64, zdim: 2, k: 10, side: 28 });
	assert.deepEqual((rt.spec!.kinds as { id: string }[]).map((k) => k.id), KINDS, 'the narration names stars, ladders, umbrellas, glasses and T-shirts');
	assert.equal(rt.num('fact:classesN'), KINDS.length);
	assert.equal(rt.num('fact:trainN'), 12000 * KINDS.length);
	assert.equal(rt.num('fact:params'), rt.spec!.params);
	assert.equal(manifest.catalog!.params!.en, rt.spec!.params.toLocaleString('en-US'));
	assert.equal(rt.spec!.tensors.reduce((s, t) => s + t.length, 0), rt.spec!.params, 'misconception: the model folder stores numbers, no drawings');
	// quiz 1: only z (2 numbers) reaches the decoder; mu and logvar are 4
	const dec1 = rt.spec!.layers.find((l) => l.id === 'dec1')!;
	assert.deepEqual(dec1.in, ['z']);
	assert.equal(manifest.page.quiz![0].answer, 0);
	const text = JSON.stringify(manifest.page);
	for (const s of ['784 numbers', 'just 2 numbers', '2 numbers per point', 'five kinds']) assert.ok(text.includes(s), s);
	rt.close();
});

test('steps "squeeze", "spread", "sample": σ = exp(½ log σ²), ε is the adapter’s draw, z = μ + σ·ε, and z stays within 2σ', async () => {
	await withFixtures(await pkgP, (rt) => {
		const c = codeOf(rt);
		const e = epsOf(rt.input);
		for (let i = 0; i < 2; i++) {
			assert.ok(Math.abs(c.sigma[i] - Math.exp(0.5 * c.logvar[i])) < 1e-6, 'sigma');
			assert.equal(c.eps[i], Math.fround(e[i]), 'eps from the draw');
			assert.ok(Math.abs(c.z[i] - (c.mu[i] + c.sigma[i] * c.eps[i])) < 1e-6, 'z');
			assert.ok(Math.abs(c.z[i] - c.mu[i]) <= 2 * c.sigma[i], 'this draw lands inside the dashed (2σ) ring');
		}
	}, opts);
	let beyond = 0;
	for (let k = 0; k < 1000; k++) for (const v of epsOf({ draw: k })) if (Math.abs(v) > 2) beyond++;
	assert.ok(beyond / 2000 < 0.06, `ε is rarely beyond ±2 (${beyond} of 2000)`);
	let inside = 0;
	for (let k = 0; k < 20000; k++) {
		const e = epsOf({ draw: k });
		if (e[0] ** 2 + e[1] ** 2 <= 4) inside++;
	}
	assert.ok(Math.abs(inside / 20000 - 0.86) < 0.015, `predict: about 86 of 100 draws fall inside the dashed ring (${inside} of 20000; 1 − e^−2 = 0.865)`);
	const rt = await createRuntime(await pkgP, opts);
	assert.ok(rt.num('fact:sigmaMedian') < rt.num('fact:muStd') / 10, 'the median σ is small next to the spread of the map');
	rt.close();
});

test('step "decode": the decoder alone redraws exactly the traced picture, with ink between 0 and 1', async () => {
	await withFixtures(await pkgP, (rt) => {
		const out = T(rt, 'out')!.data;
		assert.equal(maxDiff(decode(rt.model!, codeOf(rt).z), out), 0);
		assert.ok(Array.from(out).every((v) => v >= 0 && v <= 1));
	}, opts);
});

test('step "loss": the BCE of the 784 pixels and the KL, computed two ways; KL grows away from μ = 0, σ = 1', async () => {
	await withFixtures(await pkgP, (rt) => {
		const L = lossOf(rt)!;
		const x = T(rt, 'input')!.data, y = T(rt, 'out')!.data;
		let bce = 0;
		for (let i = 0; i < x.length; i++) bce -= x[i] * Math.log(Math.max(y[i], 1e-12)) + (1 - x[i]) * Math.log(Math.max(1 - y[i], 1e-12));
		assert.ok(Math.abs(bce - L.rec) <= 1e-3 * L.rec, `BCE ${bce} vs ${L.rec}`);
		assert.ok(L.kl >= 0 && Math.abs(L.total - L.rec - L.kl) < 1e-9);
		const c = codeOf(rt);
		if (klFromLog(c.mu, c.logvar)) assert.ok(Math.max(...c.sigma) < 0.3, 'step 6: "mostly from the −log σ² term, because σ is so small"');
		else assert.ok(Math.hypot(...c.mu) > 1, 'step 6: "because μ is far from 0"');
	}, opts);
	const kl = (mu: number, lv: number) => 0.5 * (mu * mu + Math.exp(lv) - lv - 1);
	assert.equal(kl(0, 0), 0);
	for (const [a, b] of [[0.5, 1], [-1, -2]]) assert.ok(kl(b, 0) > kl(a, 0), 'grows as μ moves away from 0');
	for (const [a, b] of [[-1, -3], [1, 2]]) assert.ok(kl(0, b) > kl(0, a), 'grows as σ moves away from 1');
});

test('step "map": the dots are this model’s (precomputed μ = the live encoder), kinds gather, and labels were never used', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const lat = latentOf(rt)!;
	assert.deepEqual(lat.classes, KINDS);
	assert.equal(lat.x.length, KINDS.length * rt.num('fact:mapPerClass'));
	for (const p of rt.presets) {
		const mu = muOf(rt.model!, p.raw);
		assert.ok(maxDiff(mu, lat.presets[p.id]) < 1e-3, `${p.id}: live μ ${mu} vs latent.json ${lat.presets[p.id]}`);
		const k = KINDS.indexOf(p.id);
		if (k >= 0) assert.equal(near(lat, mu).order[0], k, `${p.id} lands in its own kind's region`);
	}
	assert.ok(rt.num('fact:knnAgree') > 0.8);
	assert.match(trainPy(), /Xtr, _ = tensors\(classes, "train"\)/, 'train.py drops the labels: the VAE is never told the kinds');
	rt.close();
});

test('step "drag": the task is done exactly on a border; every point of the map decodes to a drawing', async () => {
	const pkg = await pkgP;
	const step = lesson.steps.find((s) => 'id' in s && s.id === 'drag') as { task: { done: (rt: unknown, at: unknown) => boolean } };
	const rt = await createRuntime(pkg, { ...opts, input: lesson.fixtures![0].raw });
	const lat = latentOf(rt)!;
	assert.ok(step.task.done(rt, {}), 'the border fixture');
	const p = extraOf(rt.input).probe!;
	const n = near(lat, p);
	assert.ok(n.counts[n.order[0]] >= 3 && n.counts[n.order[1]] >= 3);
	for (const c of lat.centres) {
		rt.setInput(withExtra(rt.input, { probe: c }));
		assert.equal(step.task.done(rt, {}), false, `a kind's centre ${c} is not a border`);
	}
	for (let i = 0; i < 25; i++) {
		const z: XY = [-RANGE + (i % 5) * 1.5, -RANGE + Math.floor(i / 5) * 1.5];
		const img = decode(rt.model!, z);
		assert.ok(Math.max(...img) > 0.3 && Array.from(img).every(Number.isFinite), `z = ${z} decodes to a drawing`);
	}
	rt.close();
});

test('step "walk": every point on the walk to every kind gives a full drawing; the default target is another kind', async () => {
	await withFixtures(await pkgP, (rt) => {
		const lat = latentOf(rt)!, mu = codeOf(rt).mu;
		const own = near(lat, mu).order[0];
		if (!extraOf(rt.input).target) assert.notEqual(targetOf(rt, lat, mu).c, own);
		for (const kind of KINDS) {
			rt.setInput(withExtra(rt.input, { target: kind }));
			const tg = targetOf(rt, lat, mu);
			for (let k = 0; k <= 6; k++) {
				const img = decode(rt.model!, [mu[0] + ((tg.mu[0] - mu[0]) * k) / 6, mu[1] + ((tg.mu[1] - mu[1]) * k) / 6]);
				assert.ok(Math.max(...img) > 0.4, `walk to ${kind}, point ${k + 1}: no blank gap`);
			}
		}
	}, opts);
});

test('step "limits", the pre-question and try-this: redrawings are greyer; 16 numbers fix detail, not blur; a house lands among the known kinds', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const f = (k: string) => rt.num(`fact:${k}`);
	assert.ok(f('greyOut') > f('greyIn') + 0.1, 'redrawings are blurrier than drawings');
	assert.equal(f('z16Dim'), 16);
	assert.ok(f('z16Rec') < f('testRec') - 20, '16 numbers redraw clearly better');
	assert.ok(f('z16GreyOut') > f('greyOut') - 0.02, '… but just as grey');
	const lat = latentOf(rt)!;
	for (const p of rt.presets) {
		rt.setInput(p.raw);
		assert.ok(greyShare(T(rt, 'out')!.data) > greyShare(T(rt, 'input')!.data), `${p.id}: the redrawing is greyer (pre-question: blurry, typical)`);
		const x = T(rt, 'input')!.data;
		let grey = 0, edge = 0;
		for (let i = 0; i < 784; i++) {
			if (!(x[i] > 0.2 && x[i] < 0.8)) continue;
			grey++;
			const r = Math.floor(i / 28), q = i % 28;
			if ([[r - 1, q], [r + 1, q], [r, q - 1], [r, q + 1]].some(([a, b]) => a < 0 || a > 27 || b < 0 || b > 27 || x[a * 28 + b] < 0.2)) edge++;
		}
		assert.ok(edge >= 0.85 * grey, `${p.id}: the grey pixels of a drawing are stroke edges (${edge} of ${grey} touch paper)`);
		const mu = codeOf(rt).mu;
		assert.ok(mu.every((v) => Math.abs(v) < RANGE));
		const d = Math.min(...lat.x.map((x, i) => Math.hypot(x - mu[0], lat.y[i] - mu[1])));
		assert.ok(d < 0.25, `${p.id} lands among the test drawings of the five kinds (nearest at ${d})`);
	}
	rt.setInput({ strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] });
	assert.ok(codeOf(rt).mu.every((v) => Math.abs(v) < RANGE), 'try-this "one dot": still a place on the map');
	assert.ok(codeOf(rt).sigma.every((s) => s > 3 * rt.num('fact:sigmaMedian')), 'try-this "one dot": its σ is several times the usual one');
	const house = rt.presets.find((p) => p.id === 'heldout-house')!;
	rt.setInput(house.raw);
	const inked = (a: ArrayLike<number>) => Array.from(a).filter((v) => v > 0.5).length;
	assert.ok(atBoundary(lat, codeOf(rt).mu), 'try-this "a house": it lands between kinds it knows (a border)');
	assert.ok(inked(T(rt, 'out')!.data) < inked(T(rt, 'input')!.data) / 2, 'try-this "a house" and the limits look: the redrawing is faint');
	rt.close();
});
