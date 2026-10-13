// catastrophic-forgetting claim tests (design §4.6 "Claims are tests"): every factual sentence of the lesson, the concept
// page (concept.ts) and the manifest that is not computed live is asserted here. The fixed-input sources (plain, pullRun,
// remedies, big) are recorded once; the learner's source (mine) runs on every fixture through withFixtures.
// `atlas check` runs this file (AK408 on failure). Run alone: node --test src/lib/explainers/catastrophic-forgetting/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame } from '../../atlas/types/index.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { GAP, N_TASK, TASK_A, TASK_B, replayCount, settingsOf } from './data.ts';
import { Learner, REMEDIES, batchOf, paramsOf, remedies, sequence } from './process.ts';
import manifest from './explainer.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const SLUG = 'catastrophic-forgetting';
const pkg = async () => (await loadPackage(SLUG, SITE)).pkg;
type T = { length: number; frame(i: number): Frame; find(c: unknown): number };
const last = (t: T) => t.frame(t.length - 1).metrics!;
const at = (t: T, c: unknown) => t.frame(t.find(c)).metrics!;
const src = manifest.sources!;
const run = (name: string): T => {
	const s = src[name] as { process: string; opt?: Record<string, unknown> };
	const t = s.process === 'remedies' ? record(remedies, null, s.opt ?? {}, { detail: 'coarse' }) : record(sequence, null, s.opt ?? {}, { detail: 'coarse' });
	return t as unknown as T;
};
const plain = run('plain');
const pullRun = run('pullRun');
const rem = run('remedies');
const big = run('big');

test('steps "learn-a", "predict": task A is class 1 above y = 0 at negative x, task B the other way round at positive x, with a gap', () => {
	for (const [t, above, sign] of [
		[TASK_A, 1, -1],
		[TASK_B, 0, 1]
	] as const) {
		let ones = 0;
		for (let i = 0; i < t.n; i++) {
			const x = t.xy[2 * i];
			const y = t.xy[2 * i + 1];
			assert.ok(Math.sign(x) === sign, `x ${x}`);
			assert.ok(Math.abs(y) >= GAP, `y ${y}`);
			assert.equal(t.label[i], y > 0 ? above : 1 - above);
			ones += t.label[i];
		}
		assert.equal(t.n, N_TASK);
		assert.equal(ones, N_TASK / 2, 'balanced');
	}
	assert.equal(paramsOf(16), 354);
});

test('prequestion, "predict" and "forget": plain fine-tuning trains on B\'s points only, and task A goes from all right to all wrong (B\'s rule: every A point gets the other class)', () => {
	assert.equal(batchOf(replayCount(0)), TASK_B);
	const sw = at(plain, { phase: 'switch' });
	assert.equal(sw.k, 0);
	assert.equal(sw.accA, 1);
	assert.equal(sw.accB, 0, 'after A, the network sorts B the wrong way: A\'s rule');
	const end = last(plain);
	assert.equal(end.accA, 0);
	assert.equal(end.accB, 1);
});

test('step "curves": task A ends below a coin flip and is sure of its wrong answers (its loss is above ln 2, a coin flip\'s)', () => {
	const end = last(plain);
	assert.ok(end.accA < 0.5 && end.lossA > Math.log(2), `accA ${end.accA}, lossA ${end.lossA}`);
});

test('step "why" and goal 1: at the end of A the two gradients point in opposing directions; B\'s first update lowers B\'s loss and raises A\'s', () => {
	const m = at(plain, { phase: 'switch' });
	assert.ok(m.cos < -0.5 && m.cos >= -1, `cos ${m.cos}`);
	assert.ok(m.lossB1 < m.lossB && m.lossA1 > m.lossA, `B ${m.lossB} -> ${m.lossB1}, A ${m.lossA} -> ${m.lossA1}`);
});

test('step "replay" and pitfall 3: replay adds the first k task-A points to B\'s batch, k / (60 + k) is the replay share, and the batch grows by k', () => {
	for (const r of [0.05, 0.1, 0.25, 0.5]) {
		const k = replayCount(r);
		const b = batchOf(k);
		assert.equal(b.n, N_TASK + k);
		assert.ok(Math.abs(k / b.n - r) <= 1 / b.n, `share ${k / b.n} vs ${r}`);
		for (let i = 0; i < k; i++) {
			assert.equal(b.xy[2 * (N_TASK + i)], TASK_A.xy[2 * i]);
			assert.equal(b.label[N_TASK + i], TASK_A.label[i]);
		}
	}
});

test('step "pull": the update is w − lr · (B\'s slope + λ · (w − w_A)), the pull points back toward w_A, and this remedy uses no task-A data', () => {
	let seen = 0;
	for (let i = 0; i < pullRun.length; i++) {
		const m = pullRun.frame(i).metrics!;
		if (pullRun.frame(i).phase !== 'learnB') continue;
		seen++;
		assert.equal(m.k, 0);
		assert.ok(Math.abs(m.pp - m.pull * (m.pw - m.pa)) < 1e-12);
		assert.ok(Math.abs(m.pw1 - (m.pw - m.lrB * (m.pg + m.pp))) < 1e-9, `frame ${i}: ${m.pw1} vs ${m.pw - m.lrB * (m.pg + m.pp)}`);
		if (m.pw !== m.pa) assert.equal(Math.sign(-m.pp), Math.sign(m.pa - m.pw));
	}
	assert.ok(seen > 20);
	assert.ok(pullRun.find({ phase: 'learnB', step: 350 }) >= 0, 'the step\'s frame exists');
});

test('step "tradeoff", the replay predict, quiz 2 and goal 3: replay of a handful of points (10%) keeps both tasks; the pull is a seesaw (weak: A lost; strong: A kept, B near chance or below)', () => {
	assert.equal(REMEDIES.length, 6);
	const m = last(rem);
	assert.ok(m.accA_replay >= 0.95 && m.accB_replay >= 0.95, `replay ${m.accA_replay}, ${m.accB_replay}`);
	assert.ok(m.accA_pullWeak < 0.5, `weak pull keeps A at ${m.accA_pullWeak}`);
	assert.ok(m.accA_pull > 0.9 && m.accB_pull <= 0.6 && m.accB_pull < m.accB_pullWeak, `strong pull ${m.accA_pull}, ${m.accB_pull}`);
	// the quiz's preset (λ = 0.8) too
	const p8 = last(record(sequence, { pull: 0.8 }, {}, { detail: 'coarse' }) as unknown as T);
	assert.ok(p8.accA > 0.9 && p8.accB <= 0.6, `λ 0.8: ${p8.accA}, ${p8.accB}`);
	// the comparison's copies start from the same network as the lesson's runs: same ends as the single runs
	assert.equal(m.accA_plain, last(plain).accA);
	assert.equal(m.accA_pull, last(pullRun).accA);
});

test('step "limits", pitfalls 1 and 2: a frozen first layer keeps task A\'s first layer exactly, yet A is lost; a ten times smaller rate only delays it, ends nearly as low and learns B more slowly', () => {
	const L = new Learner(16, 7);
	for (let s = 0; s < 300; s++) {
		L.grad(TASK_A);
		L.update(0.5, false);
	}
	const first = L.ps.slice(0, 2).map((p) => Float64Array.from(p.data));
	for (let s = 0; s < 300; s++) {
		L.grad(TASK_B);
		L.update(0.5, true);
	}
	L.ps.slice(0, 2).forEach((p, j) => assert.deepEqual(Array.from(p.data), Array.from(first[j])));
	assert.ok(L.measure(TASK_A).acc < 0.5);
	const m = last(rem);
	const mid = at(rem, { step: 50 });
	assert.ok(m.accA_freeze < 0.5, `freeze ${m.accA_freeze}`);
	assert.ok(mid.accA_slow > mid.accA_plain + 0.5, `at 50: slow ${mid.accA_slow}, plain ${mid.accA_plain}`);
	assert.ok(m.accA_slow < 0.1, `slow ends at ${m.accA_slow}`);
	assert.ok(mid.accB_slow < mid.accB_plain, `B at 50: slow ${mid.accB_slow}, plain ${mid.accB_plain}`);
});

test('concept "real": a network with 48 units per layer (more than five times the weights) forgets task A just the same', () => {
	assert.ok(paramsOf(48) > 5 * paramsOf(16));
	assert.equal(at(big, { phase: 'switch' }).accA, 1);
	const m = last(big);
	assert.ok(m.accA < 0.1 && m.accB > 0.95, `big: A ${m.accA}, B ${m.accB}`);
});

test('every fixture: the learner\'s source runs task A exactly like the plain run (same network after A), then trains B with the fixture\'s settings', async () => {
	const swPlain = at(plain, { phase: 'switch' });
	await withFixtures(
		await pkg(),
		async (rt) => {
			await rt.ensure('mine', 'last');
			const t = rt.sources.mine.tape! as unknown as T;
			const sw = at(t, { phase: 'switch' });
			assert.equal(sw.lossA, swPlain.lossA);
			const s = settingsOf(rt.input);
			assert.equal(sw.k, replayCount(s.replay));
			assert.equal(sw.lrB, s.slow ? 0.05 : 0.5);
		},
		opts
	);
});
