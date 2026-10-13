// egnn claim tests: every sentence of the lesson and the page that is not computed live is asserted here, on every
// fixture (presets and edge fixtures) through withFixtures. `atlas check` runs this file (AK408 on failure).
//   node --test src/lib/explainers/egnn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRuntime } from '../../atlas/runtime/runtime.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import OPS from './ops.ts';
import { focus, mol, run, T, variant, wrap } from './look.ts';
import { energy, mirror, molOf, presetRaw, turn, turnVec, PRESET_IDS } from './mol.ts';
import { mirrorDiffers } from './lesson.ts';
import { spin, spinInput } from './process.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkg = async () => (await loadPackage('egnn', SITE)).pkg;
const ready = async (rt: ExplainerRuntime) => (await rt.ensure('fwd', 'last'), rt);
const at = async (raw: unknown) => ready(await createRuntime(await pkg(), { ...opts, input: raw }));
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, k) => Math.max(m, Math.abs(v - b[k])), 0);
/** the largest |a − b| relative to max(1, |b|) */
const maxRel = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, k) => Math.max(m, Math.abs(v - b[k]) / Math.max(1, Math.abs(b[k]))), 0);
const tol = (E: number) => 1e-5 * Math.max(1, Math.abs(E));

test('distances, spin, turn: turning or shifting leaves every d² (to float32 rounding) and the energy the same; forces turn with it', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const m = mol(rt);
		if (!m) return;
		for (const deg of [30, 90, 200]) {
			const v = variant(rt, deg)!;
			assert.ok(Math.abs(v.dE) <= tol(m.E), `turn ${deg}: dE ${v.dE}`);
			assert.ok(v.df <= 1e-5, `turn ${deg}: df ${v.df}`);
			const t = rt.model!.forward({ pos: { shape: [m.n, 2], data: v.pos }, kind: T(rt, 'kind')! }, { record: 'layers' }).t;
			assert.ok(maxAbs(t['L1.d2'].data, T(rt, 'L1.d2')!.data) <= 1e-4, 'd² after a turn');
		}
		const shifted = Float32Array.from(m.pos, (v, k) => v + (k % 2 ? -0.75 : 1.5));
		const o = run(rt.model!, shifted, T(rt, 'kind')!.data);
		assert.ok(Math.abs(o.E - m.E) <= tol(m.E) && maxAbs(o.f, m.f) <= 1e-5, 'a shift changes nothing');
	}, opts);
});

test('push and output: each force is the sum of its two pushes; the energy is the sum of one share per atom; pushes turn with the molecule', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const m = mol(rt);
		if (!m) return;
		const p1 = T(rt, 'L1.push')!, p2 = T(rt, 'L2.push')!, sh = T(rt, 'share')!;
		assert.ok(maxRel(T(rt, 'force')!.data, Array.from(p1.data, (v, k) => v + p2.data[k])) <= 1e-6);
		assert.ok(Math.abs(Array.from(sh.data).reduce((a, b) => a + b, 0) - m.E) <= 1e-5 * Math.max(1, Math.abs(m.E)));
		const t = rt.model!.forward({ pos: { shape: [m.n, 2], data: Float32Array.from(turn(m.pos, 90)) }, kind: T(rt, 'kind')! }, { record: 'layers' }).t;
		for (let i = 0; i < m.n; i++) {
			const [x, y] = turnVec(p1.data[2 * i], p1.data[2 * i + 1], 90);
			assert.ok(maxRel([t['L1.push'].data[2 * i], t['L1.push'].data[2 * i + 1]], [x, y]) <= 1e-5);
		}
	}, opts);
});

test('atoms, message, sum: every atom is joined to every other; a message row is h_i, h_j, d²; atom order does not change the energy', async () => {
	const rt = await at(presetRaw('star').raw);
	const m = mol(rt)!, n = m.n, h = T(rt, 'embed')!, e = T(rt, 'L1.edge')!, d = T(rt, 'L1.d2')!;
	const F = h.shape[1], C = e.shape[2];
	assert.equal(C, 2 * F + 1);
	for (const [i, j] of [[0, 3], [2, 1]]) {
		const row = e.data.subarray((i * n + j) * C, (i * n + j + 1) * C);
		assert.deepEqual(Array.from(row.subarray(0, F)), Array.from(h.data.subarray(i * F, (i + 1) * F)));
		assert.deepEqual(Array.from(row.subarray(F, 2 * F)), Array.from(h.data.subarray(j * F, (j + 1) * F)));
		assert.equal(row[2 * F], d.data[i * n + j]);
	}
	// the summed messages of atom i hold all n − 1 other atoms (each a full row of the message table)
	const msg = T(rt, 'L1.msg')!, agg = T(rt, 'L1.agg')!, M = agg.shape[1];
	let s = 0;
	for (let j = 1; j < n; j++) s += msg.data[(0 * n + j) * M];
	assert.ok(Math.abs(s - agg.data[0]) <= 1e-5);
	// reversed atom order: the same energy, the forces reversed with the atoms
	const order = Array.from({ length: n }, (_, i) => n - 1 - i);
	const k = T(rt, 'kind')!.data;
	const o = run(rt.model!, Float32Array.from(order.flatMap((i) => [m.pos[2 * i], m.pos[2 * i + 1]])), Float32Array.from(order.flatMap((i) => Array.from(k.subarray(3 * i, 3 * i + 3)))));
	assert.ok(Math.abs(o.E - m.E) <= 1e-5);
	order.forEach((i, a) => assert.ok(Math.abs(o.f[2 * a] - m.f[2 * i]) <= 1e-5));
});

test('mirror (limits): the mirror image gets the same energy and mirrored forces on every fixture; a 2-D cross product changes sign in a mirror', async () => {
	await withFixtures(await pkg(), async (rt) => {
		await ready(rt);
		const m = mol(rt);
		if (!m) return;
		const v = variant(rt, 'mirror')!;
		assert.ok(Math.abs(v.dE) <= tol(m.E) && v.df <= 1e-5);
	}, opts);
	const q = mirror([0, 0, 1, 0, 0.3, 1]);
	const cross = (p: ArrayLike<number>) => (p[2] - p[0]) * (p[5] - p[1]) - (p[3] - p[1]) * (p[4] - p[0]);
	assert.ok(cross([0, 0, 1, 0, 0.3, 1]) * cross(q) < 0);
	// the preset 'bent' cannot be turned onto its mirror image; two atoms always can
	assert.equal(mirrorDiffers(molOf(presetRaw('bent').raw).pos, molOf(presetRaw('bent').raw).kind), true);
	assert.equal(mirrorDiffers(molOf(presetRaw('pair').raw).pos, molOf(presetRaw('pair').raw).kind), false);
});

test('spin: 12 frames; the energy stays (float32 rounding) and the followed force turns by the same angle as the molecule', async () => {
	for (const id of PRESET_IDS) {
		const rt = await at(presetRaw(id).raw);
		const frames = [...spin.run(spinInput(rt), { rng: createRng(1), opt: {}, detail: 'coarse', signal: { cancelled: false } })];
		assert.equal(frames.length, 12);
		const m = mol(rt)!, i = focus(rt);
		for (const f of frames) {
			assert.ok(Math.abs(f.metrics!.dE) <= tol(m.E), `${id} dE`);
			if (Math.hypot(m.f[2 * i], m.f[2 * i + 1]) > 1e-3) assert.ok(Math.abs(wrap(f.metrics!.turned - f.metrics!.deg)) <= 0.01, `${id} turned ${f.metrics!.turned} at ${f.metrics!.deg}`);
		}
	}
});

test('prequestion and misconception: equivariance holds by construction, also for weights the network was never trained to', async () => {
	const rt = await at(presetRaw('bent').raw);
	const rng = createRng(7);
	const w: Record<string, Tensor> = {};
	for (const [k, t] of Object.entries(rt.model!.weights)) w[k] = { shape: t.shape, data: Float32Array.from(t.data, () => rng.normal()) };
	const g = fromParts(rt.model!.spec, w, OPS);
	const m = mol(rt)!, kind = T(rt, 'kind')!.data;
	const a = run(g, m.pos, kind), b = run(g, turn(m.pos, 77), kind);
	assert.ok(Math.abs(a.E - b.E) <= 1e-4 * Math.max(1, Math.abs(a.E)));
	for (let i = 0; i < m.n; i++) {
		const [x, y] = turnVec(a.f[2 * i], a.f[2 * i + 1], 77);
		assert.ok(Math.abs(b.f[2 * i] - x) <= 1e-4 * Math.max(1, Math.abs(x)) && Math.abs(b.f[2 * i + 1] - y) <= 1e-4 * Math.max(1, Math.abs(y)));
	}
});

test('gcn: the gcn page’s network reads only the adjacency matrix and one-hot member features', () => {
	const spec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'gcn', 'spec.json'), 'utf8'));
	assert.deepEqual(spec.inputs.map((x: { name: string }) => x.name).sort(), ['adj', 'x']);
});

test('output: the toy formula is a Morse pair sum whose forces are its exact negative gradient (finite differences)', () => {
	const { pos, kind } = molOf(presetRaw('ring').raw);
	const e = energy(pos, kind), h = 1e-6;
	for (const k of [0, 3, 7]) {
		const p = Float64Array.from(pos), q = Float64Array.from(pos);
		(p[k] += h), (q[k] -= h);
		assert.ok(Math.abs(-(energy(p, kind).E - energy(q, kind).E) / (2 * h) - e.F[k]) <= 1e-6);
	}
});

test('turn (task): for every preset some slider position (a multiple of 5°) makes the ringed force point up, and the task sees it', async () => {
	const step = (await import('./lesson.ts')).default.steps.find((s) => 'id' in s && s.id === 'turn') as { task: { done: (rt: ExplainerRuntime, at: never) => boolean } };
	for (const id of PRESET_IDS) {
		const raw = presetRaw(id).raw;
		const rt0 = await at(raw), m = mol(rt0)!, i = focus(rt0);
		if (Math.hypot(m.f[2 * i], m.f[2 * i + 1]) < 1e-3) continue;
		const a0 = Math.atan2(m.f[2 * i + 1], m.f[2 * i]) * (180 / Math.PI);
		const t = (((Math.round((90 - a0) / 5) * 5) % 360) + 360) % 360;
		if (!t) continue; // it already points up: the task asks for a turn
		const rt = await at({ ...raw, turn: t });
		assert.equal(focus(rt), i, id);
		assert.ok(step.task.done(rt, undefined as never), `${id}: turn ${t}`);
	}
});

test('output (warning): every training molecule of make_data.ts has NMIN to NMAX atoms and keeps every pair at least CLOSE bond lengths apart', async () => {
	const { fnv1a32 } = await import('../../atlas/core/rng.ts');
	const { NMIN, NMAX, sample } = await import('./mol.ts');
	const { unlike } = await import('./lesson.ts');
	const root = createRng(fnv1a32('egnn data v1')); // the same forks as pipeline/egnn/make_data.ts
	for (let i = 0; i < 40000; i++) {
		const rng = root.fork(`train:${i}`);
		const m = sample(NMIN + rng.int(NMAX - NMIN + 1), rng);
		const u = unlike(m);
		assert.ok(!u.count && !u.close, `train ${i}: ${m.n} atoms, closest ${u.q}`);
	}
	// and the crowded edge fixture does trip the warning
	const fx = (await import('./adapters/points2d/adapter.ts')).default.edgeFixtures!().find((f) => f.id === 'crowded')!;
	const crowded = molOf(fx.raw);
	assert.ok(unlike(crowded).count && unlike(crowded).close);
});
