// ppo-reacher claim tests: every sentence of the lesson (and the page) that is not computed live is asserted here, on
// every fixture (the presets and the adapter's edge fixtures) where it depends on the input. `atlas check` runs this file
// (AK408 on failure).   node --test src/lib/explainers/ppo-reacher/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Tape, Tensor } from '../../atlas/types/index.ts';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { CTRL, DEFAULT_EPS, EVAL_DOWN, EVAL_UP, L1, L2, gae, obs, randomTarget, START, step, targetOf, tip } from './sim.ts';
import { ActorCritic, collect, paramsOf, ppo, ppoLoss } from './trainer.ts';
import { backward } from '../../atlas/autograd/var.ts';
import { pair } from './process.ts';
import { mbOf } from './views/clip/meta.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('ppo-reacher', SITE).then((l) => l.pkg);
const all = (t: Tape): Frame[] => Array.from({ length: t.length }, (_, i) => t.frame(i));
const D = ppo.defaults, U = D.epochs * D.minibatches;
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
async function run(rt: AtlasRuntime, source: string): Promise<Tape> {
	await rt.ensure(source, 'last');
	const t = rt.sources[source].tape;
	assert.ok(t?.length, `${source} has frames`);
	return t!;
}
const batches = (t: Tape) => all(t).flatMap((f, i) => (f.phase === 'forward' ? [(t.state(i) as { batch: Record<string, Tensor> }).batch] : []));

test('manifest: every source has seed 1; the parameter counts in the catalog row are the networks’ sizes', () => {
	for (const [k, s] of Object.entries(manifest.sources!)) assert.equal((s as { seed?: number }).seed ?? (s as { opt?: { seed?: number } }).opt?.seed ?? 1, 1, k);
	const p = paramsOf(ppo.model(createRng(1).fork('init'), D)), n = (pre: string) => Object.entries(p).filter(([k]) => k.startsWith(pre)).reduce((s, [, t]) => s + t.data.length, 0);
	assert.equal(n('pi.') + n('logstd'), 292);
	assert.equal(n('v.'), 257);
	assert.ok(manifest.catalog?.params?.en.includes('292') && manifest.catalog.params.en.includes('257'));
});

test('arm: two links of the same length; the policy sees dx, dy, q1, q2, ω1, ω2; reward = −distance − CTRL·|a|²', () => {
	assert.equal(L1, L2);
	const g: [number, number] = [-0.3, 0.6], o = obs(START, g), p = tip(START);
	assert.deepEqual(o, [g[0] - p[0], g[1] - p[1], ...START]);
	const s = step(START, [0.7, -1.2], g);
	assert.equal(s.r, -Math.hypot(tip(s.s2)[0] - g[0], tip(s.s2)[1] - g[1]) - CTRL * (0.49 + 1.44));
});

test('policy, episode, value, gae (every fixture): a = μ + σ·z, σ = e^logstd; one episode is T steps; δ = r + γ·V(next) − V; A = δ + γλ·A(next); the episode is episode 0 of the first batch', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const fs = all(await run(rt, 'episode')), acts = fs.filter((f) => f.phase === 'act').map((f) => f.metrics!), g = fs.filter((f) => f.phase === 'gae').map((f) => f.metrics!).sort((a, b) => a.t - b.t);
		assert.equal(acts.length, D.T);
		for (const a of acts) {
			assert.equal(a.s1, Math.exp(a.ls1));
			assert.equal(a.a1, a.mu1 + a.s1 * a.z1);
			assert.equal(a.a2, a.mu2 + a.s2 * a.z2);
		}
		const end = fs.find((f) => f.phase === 'end')!.metrics!;
		assert.equal(g[D.T - 1].v1, end.vLast, 'the last step uses V of the state after it (the episode is cut off by time)');
		for (const x of g) {
			assert.ok(Math.abs(x.delta - (x.r + x.gamma * x.v1 - x.v)) < 1e-12);
			assert.ok(Math.abs(x.A - (x.delta + x.gl * x.next)) < 1e-12);
		}
		const b0 = batches(await run(rt, 'iter'))[0], ro = Array.from(b0.allAdv.data);
		assert.equal(ro.length, D.episodes * D.T);
		const raw = g.map((x) => x.A), { mean: m, sd } = { mean: b0.advMean.data[0], sd: b0.advSd.data[0] };
		raw.forEach((a, t) => assert.ok(Math.abs((a - m) / sd - ro[t]) < 1e-12, `episode 0, step ${t}: the batch's normalized A is this episode's A`));
	}, opts);
});

test('gae: λ = 0 leaves only the one-step δ (all trust in the critic); λ = 1 makes A the discounted rewards minus V (the critic only as a baseline)', () => {
	const r = [-1, -0.5, -0.2], v = [-0.8, -0.6, -0.1], vLast = -0.05, gm = 0.9;
	const g0 = gae(r, v, vLast, gm, 0), g1 = gae(r, v, vLast, gm, 1);
	g0.A.forEach((a, t) => assert.equal(a, g0.delta[t]));
	const G0 = r[0] + gm * r[1] + gm * gm * r[2] + gm ** 3 * vLast;
	assert.ok(Math.abs(g1.A[0] - (G0 - v[0])) < 1e-12);
});

test('batch, ratio, clip (every fixture): the other episodes go to random targets above the shoulder line; each epoch uses every step once; r = 1 at update 0; a sample is clipped exactly when its ratio passed 1 ± ε the way its advantage points; the advantages stay the old policy’s; some ratio leaves [1 − ε, 1 + ε] anyway (the misconception)', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const bs = batches(await run(rt, 'iter'));
		assert.equal(bs.length, U);
		const T = bs[0].targets.data;
		for (let e = 1; e < D.episodes; e++) assert.ok(T[2 * e + 1] >= 0, 'a target above the shoulder line');
		for (let k = 0; k < D.epochs; k++) {
			const idx = bs.slice(k * D.minibatches, (k + 1) * D.minibatches).flatMap((b) => Array.from(b.idx.data));
			assert.deepEqual(idx.slice().sort((a, b) => a - b), Array.from({ length: D.episodes * D.T }, (_, i) => i));
		}
		let outside = false;
		bs.forEach((b, j) => {
			const m = mbOf(b)!, eps = m.eps;
			if (j === 0) assert.ok(m.r.every((r) => Math.abs(r - 1) < 1e-12));
			m.r.forEach((r, i) => {
				const A = m.A[i];
				assert.equal(m.clipped[i] === 1, (A > 0 && r > 1 + eps) || (A < 0 && r < 1 - eps));
				if (r > 1 + eps || r < 1 - eps) outside = true;
				assert.equal(A, bs[0].allAdv.data[b.idx.data[i]], 'the minibatch’s A is the batch’s');
			});
		});
		assert.ok(outside || epsOfRaw(rt) >= 100);
	}, opts);
});
const epsOfRaw = (rt: AtlasRuntime) => Number((rt.input as { eps?: number }).eps ?? DEFAULT_EPS);

test('clip (every fixture): the clipped samples add nothing to the gradient (setting their advantage to 0 changes no gradient)', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const t = await run(rt, 'iter'), fs = all(t), i = fs.findIndex((f) => f.phase === 'forward' && f.step === U - 1);
		const b = (t.state(i) as { batch: Record<string, Tensor> }).batch, prev = t.state(fs.findIndex((f) => f.phase === 'update' && f.step === U - 2)) as { params: Record<string, Tensor> };
		const grads = (adv: Float64Array) => {
			const m = new ActorCritic(createRng(1).fork('init'), D);
			for (const [k, v] of Object.entries(m.named())) v.value.data.set(prev.params[k].data);
			const mb = { obs: b.obs, act: b.act, logpOld: Float64Array.from(b.logpOld.data), adv, ret: Float64Array.from(b.ret.data), eps: [b.eps.data[0]], idx: [], j: [], ro: null as never };
			backward(ppoLoss(m, mb, D).pg);
			return Object.values(m.named()).flatMap((v) => Array.from(v.grad?.data ?? []));
		};
		const A = Float64Array.from(b.adv.data), g1 = grads(A), g2 = grads(A.map((a, k) => (b.clipped.data[k] ? 0 : a)));
		g1.forEach((g, k) => assert.ok(Math.abs(g - g2[k]) <= 1e-14 * Math.max(1, Math.abs(g)), `gradient ${k}: ${g} vs ${g2[k]}`));
	}, opts);
});

test('train (critic): the value target is R = A + V, the advantage before normalization plus the critic’s guess', () => {
	const ro = collect(paramsOf(ppo.model(createRng(1).fork('init'), D)), [-0.3, 0.6], createRng(1).fork('batch'), D);
	ro.ret.forEach((R, i) => assert.equal(R, ro.advRaw[i] + ro.v[i]));
});

test('train (every fixture, full length): the average episode return climbs; σ shrinks', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const s = all(await run(rt, 'train')).filter((f) => f.phase === 'snapshot' && (f.step ?? 0) % U === 0).map((f) => f.metrics!);
		assert.equal(s.length, D.steps / U + 1);
		assert.ok(mean(s.slice(-10).map((m) => m.epRet)) > s[0].epRet + 10, 'the return climbs');
		assert.ok(s[s.length - 1].sigma < s[0].sigma, 'σ shrinks');
	}, opts);
});

test('epsilon (every fixture, full length): the pair’s first run is the train source, one frame per iteration at its last update; with the clip off the average KL per iteration is ten times larger or more, the return swings harder and ends below the ε = 0.2 run', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		const p = all(await run(rt, 'pair')).filter((f) => f.phase === 'step'), tr = all(await run(rt, 'train'));
		assert.equal(p.length, D.steps / U);
		p.forEach((f, i) => {
			const k = tr.find((x) => x.phase !== 'init' && x.step === U * i + U - 1)!.metrics!;
			assert.ok(f.metrics!.epRet === k.epRet && f.metrics!.kl === k.kl, `iteration ${i}`);
		});
		const off = all(record(pair, { target: targetOf(rt.input), eps: 100 }, {}, { seed: 1 })).filter((f) => f.phase === 'step').map((f) => f.metrics!);
		assert.ok(mean(off.map((m) => m.kl)) >= 10 * mean(off.map((m) => m.kl0)));
		const r = off.map((m) => m.epRet), r0 = off.map((m) => m.epRet0);
		const drop = (v: number[]) => Math.max(...v.slice(1).map((x, i) => v[i] - x));
		assert.ok(drop(r) > drop(r0), 'swings harder');
		assert.ok(mean(r.slice(-10)) < mean(r0.slice(-10)), 'ends below the ε = 0.2 run');
	}, opts);
});

test('limits: training targets (random ones and the learner’s) are never below the shoulder line; the test targets below it are the mirror images of those above', () => {
	const rng = createRng(7);
	for (let i = 0; i < 5000; i++) assert.ok(randomTarget(rng)[1] >= 0);
	for (const g of [[0.3, -0.5], [-2, -1], [0, 0], [5, 5]]) assert.ok(targetOf({ target: g })[1] >= 0);
	EVAL_UP.forEach((g, i) => assert.deepEqual(EVAL_DOWN[i], [g[0], -g[1]]));
});

test('train: cart-pole’s REINFORCE has no critic (its only network is the policy) and plays a new round for every update', async () => {
	const cp = (await loadPackage('cart-pole', SITE)).pkg, t = record(cp.processes['train:reinforce'], { start: [0, 0, 0.05, 0] }, { steps: 2 }, { seed: 1, detail: 'fine' });
	const fw = all(t).flatMap((f, i) => (f.phase === 'forward' ? [(t.state(i) as { batch: Record<string, Tensor> }).batch.x.data] : []));
	assert.ok(Object.keys((t.state(0) as { params: Record<string, Tensor> }).params).every((k) => k.startsWith('net.')));
	assert.notDeepEqual(Array.from(fw[0]), Array.from(fw[1]));
});

test('quiz: δ = −1 and then −1 with γλ = 0.9 give the first step A = −1.9', () => {
	const g = gae([-1, -1], [0, 0], 0, 1, 0.9);
	assert.ok(Math.abs(g.A[0] - -1.9) < 1e-12);
});
