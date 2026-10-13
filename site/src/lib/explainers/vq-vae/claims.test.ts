// vq-vae claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the swap and cell fixtures, the adapter's blank / dot / small, the 7 presets)
// through withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/vq-vae/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ExplainerRuntime } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import drawing28 from '../../atlas/adapters/drawing28/adapter.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import manifest from './explainer.ts';
import lesson, { ADD } from './lesson.ts';
import { T, blankOf, blockInk, cellOf, codesOf, decodeIdx, decodeZq, dims, distOf, extraOf, gradAt, inkOf, rankOf, swapOf, withExtra } from './vq.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('vq-vae', SITE).then((r) => r.pkg);
const KINDS = ['star', 'ladder', 'umbrella', 'eyeglasses', 't-shirt'];
const ran = (rt: ExplainerRuntime) => !!T(rt, 'out') && codesOf(rt).length > 0;
/** the presets of the page (the adapter's first drawing of each kind and the held-out kinds) */
const presets = async () => drawing28.presets!({ 'presets.json': JSON.parse(readFileSync(join(SITE, 'static', 'models', 'vq-vae', 'presets.json'), 'utf8')) });
async function on(raw: unknown) {
	const rt = await createRuntime(await pkgP, opts);
	rt.setInput(raw);
	await rt.ensure('fwd', 'last');
	return rt;
}

test('step "patches" and the page texts: 784 numbers, a 4×4 grid of 7×7 patches, 32 codes of 8 numbers, five kinds', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const d = dims(rt);
	assert.deepEqual([d.n, d.G, d.P, d.K, d.D, d.cells, d.bits, d.bitsPer], [784, 4, 7, 32, 8, 16, 80, 5]);
	assert.deepEqual((rt.spec!.kinds as { id: string }[]).map((k) => k.id), KINDS);
	assert.equal(rt.num('fact:classesN'), KINDS.length);
	assert.equal(rt.num('fact:trainN'), 12000 * KINDS.length);
	assert.equal(rt.num('fact:params'), rt.spec!.params);
	assert.equal(manifest.catalog!.params!.en, rt.spec!.params.toLocaleString('en-US'));
	const text = JSON.stringify(manifest.page);
	for (const s of ['16 small whole numbers', 'list of 32', '16 patches', '7×7 patch', 'which of 32']) assert.ok(text.includes(s), s);
	// quiz 1 and step "decode": the decoder reads the codebook rows (zq), never z_e
	assert.deepEqual(rt.spec!.layers.find((l) => l.id === 'dec1')!.in, ['zq']);
	rt.close();
});

test('step "decode", quiz 1: the redrawing is a function of the 16 indices alone (decoding the indices gives the same picture)', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const again = decodeIdx(rt.model!, codesOf(rt)), out = T(rt, 'out')!.data;
		assert.ok(Array.from(out).every((v, i) => Math.abs(v - again[i]) < 1e-6));
	}, opts);
});

test('step "snap": the code is the nearest by squared distance; "train": a small nudge of z_e keeps the same code', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const codes = codesOf(rt);
		for (let c = 0; c < codes.length; c++) {
			const r = rankOf(distOf(rt, c));
			assert.equal(r[0].k, codes[c]);
			assert.ok(r[1].d - r[0].d > 1e-3, `patch ${c}: the winner is not a near tie, so nudging z_e keeps it`);
		}
	}, opts);
});

test('step "codebook", tryThis "dot": every empty patch takes the blank code; only patches with ink take others', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const { G, P } = dims(rt), ink = inkOf(T(rt, 'input')!.data, G, P), blank = blankOf(rt.model!);
		codesOf(rt).forEach((k, c) => {
			if (ink[c] === 0) assert.equal(k, blank, `empty patch ${c}`);
		});
		if (extraOf(rt.input) && JSON.stringify(rt.input).includes('"x":0.5,"y":0.5')) assert.ok(codesOf(rt).some((k) => k !== blank), 'the dot takes a non-blank code');
	}, opts);
});

test('step "codebook" and the misconception: codes are pieces shared across kinds (every kind shares a non-blank code with another)', async () => {
	const ps = await presets();
	const sets: Record<string, Set<number>> = {};
	for (const p of ps.filter((q) => KINDS.includes(q.id))) {
		const rt = await on(p.raw);
		const blank = blankOf(rt.model!);
		sets[p.id] = new Set(codesOf(rt).filter((k) => k !== blank));
		rt.close();
	}
	const shared = (a: string, b: string) => [...sets[a]].some((k) => sets[b].has(k));
	for (const k of KINDS) assert.ok(KINDS.some((o) => o !== k && shared(k, o)), `${k} shares a code with another kind`);
});

test('step "swap": a swap changes only the 3×3 neighbourhood of its patch; the task can be done on every preset', async () => {
	for (const p of await presets()) {
		const rt = await on(p.raw);
		const codes = codesOf(rt), blank = blankOf(rt.model!), { G, P } = dims(rt), base = T(rt, 'out')!.data;
		let best = -Infinity;
		for (let c = 0; c < codes.length; c++) {
			if (codes[c] !== blank) continue;
			for (let k = 0; k < 32; k++) {
				const idx = codes.slice();
				idx[c] = k;
				const out = decodeIdx(rt.model!, idx);
				best = Math.max(best, blockInk(out, c, G, P) - blockInk(base, c, G, P));
				for (let i = 0; i < out.length; i++) {
					const r = Math.floor(Math.floor(i / (G * P)) / P), q = Math.floor((i % (G * P)) / P);
					if (Math.abs(r - Math.floor(c / G)) > 1 || Math.abs(q - (c % G)) > 1) assert.equal(out[i], base[i], 'outside the 3×3 neighbourhood nothing changes');
				}
			}
		}
		assert.ok(best >= ADD, `${p.id}: some swap adds ${ADD} inked pixels (best ${best})`);
		assert.ok(JSON.stringify(lesson.steps).includes(`at least ${ADD} pixels`), 'the task prompt says ADD');
		rt.close();
	}
	await withFixtures(await pkgP, (rt) => {
		const s = swapOf(rt);
		if (s) assert.ok(s.outside > 0, 'the neighbours change too');
	}, opts);
});

test('step "train": the gradient at z_q is the gradient of the training loss (finite differences through the decoder)', async () => {
	await withFixtures(await pkgP, (rt) => {
		if (!ran(rt)) return;
		const c = cellOf(rt), g = gradAt(rt, c), zq = Float32Array.from(T(rt, 'zq')!.data), x = T(rt, 'input')!.data, n = 16, h = 2e-3;
		const loss = (z: Float32Array) => {
			const s = decodeZq(rt.model!, z).img;
			let L = 0;
			for (let i = 0; i < s.length; i++) L += Math.max(s[i], 0) + Math.log1p(Math.exp(-Math.abs(s[i]))) - s[i] * x[i];
			return L / s.length;
		};
		const scale = Math.max(...g.map(Math.abs));
		g.forEach((gd, d) => {
			const a = zq.slice(), b = zq.slice();
			a[d * n + c] += h;
			b[d * n + c] -= h;
			const fd = (loss(a) - loss(b)) / (2 * h);
			assert.ok(Math.abs(fd - gd) <= 0.05 * scale + 2e-5, `number ${d}: backprop ${gd}, finite difference ${fd}`);
		});
	}, opts);
});

test('step "limits": more codes help only a little; the textbook recipe leaves most of a big codebook dead', async () => {
	const rt = await createRuntime(await pkgP, opts);
	const f = (k: string) => rt.num(`fact:${k}`);
	assert.ok(f('k128RecPx') < f('testRecPx') && (f('testRecPx') - f('k128RecPx')) / f('testRecPx') < 0.25, 'a little');
	assert.equal(f('k128Used'), f('k128Codes'), 'this page’s recipe uses all 128');
	assert.ok(f('plainK128Used') < f('k128Codes') / 2, 'the textbook recipe: most codes dead');
	assert.ok(f('plainK128RecPx') >= f('testRecPx') - 0.002, 'and it redraws no better than 32 codes');
	assert.equal(f('usedN'), 32);
	rt.close();
});

test('prequestion and tryThis "house": strokes come back roughly in place, also for a kind the model never saw', async () => {
	for (const p of await presets()) {
		const rt = await on(p.raw);
		const x = T(rt, 'input')!.data, y = T(rt, 'out')!.data;
		// share of the redrawing's inked pixels (> 0.5) within 1 pixel of the drawing's ink (> 0.2)
		let on1 = 0, near = 0;
		for (let i = 0; i < 784; i++) {
			if (y[i] <= 0.5) continue;
			on1++;
			const r = Math.floor(i / 28), q = i % 28;
			let hit = false;
			for (let a = -1; a <= 1 && !hit; a++) for (let b = -1; b <= 1 && !hit; b++) hit = x[(r + a) * 28 + q + b] > 0.2;
			if (hit) near++;
		}
		assert.ok(near / on1 > 0.8, `${p.id}: ${near} of ${on1} redrawn ink pixels lie on the strokes`);
		rt.close();
	}
});

test('the swap and cell extras keep the drawing: withExtra only adds fields', () => {
	const raw = { b64: 'AAAA' };
	assert.deepEqual(withExtra(raw, { cell: 3 }), { b64: 'AAAA', cell: 3 });
	assert.deepEqual(extraOf(withExtra(null, { swap: { cell: 1, code: 2 } })).swap, { cell: 1, code: 2 });
});
