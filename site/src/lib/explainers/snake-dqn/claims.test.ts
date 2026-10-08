// snake-dqn claim tests: every sentence of the lesson and the manifest that is not computed live is asserted here, on
// every fixture where it depends on the input (lesson fixtures, the adapter's edge fixtures and the presets).
// `atlas check` runs this file (AK408 on failure).   node --test src/lib/explainers/snake-dqn/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { record } from '../../atlas/process/record.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { posOf } from './adapters/snake-board/adapter.ts';
import { logOf, qOf } from './picture.ts';
import { agent, dqn, replay, withWeights } from './process.ts';
import { qvalues, weightsOf } from './qnet.ts';
import { HUNGER, encode, newGame, nextCell, replay as play, safe, step } from './rules.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('snake-dqn', SITE).then((l) => l.pkg);
const each = async (fn: (rt: AtlasRuntime) => void | Promise<void>) => withFixtures(await pkgP, async (rt) => (posOf(rt.input).over ? undefined : fn(rt)), opts);
const fact = (rt: AtlasRuntime, k: string) => rt.num(`fact:${k}`);
const check = (rt: AtlasRuntime) => (rt.spec as unknown as { agentCheck: { seeds: number[]; scores: number[] } }).agentCheck;

test('manifest and catalog: 171,235 parameters; conv 3 → 16 → 32, then 2,592 → 64 → 3; 243 input numbers', async () => {
	await withFixtures(await pkgP, (rt) => {
		const sp = rt.spec!, sh = (id: string) => sp.layers.find((l) => l.id === id)!.outShape;
		assert.equal(sp.params, 171235);
		assert.deepEqual([sh('board'), sh('conv1'), sh('conv2'), sh('flat'), sh('fc1'), sh('out')], [[3, 9, 9], [16, 9, 9], [32, 9, 9], [2592], [64], [3]]);
		assert.equal(fact(rt, 'params'), 171235);
	}, { ...opts, only: ['preset:start'] });
});

test('step 1: one 1 on the head and on the food; the body falls from (L − 1) / L behind the head to 1 / L at the tail; the brightest body cell is next to the head', async () => {
	await each((rt) => {
		const g = posOf(rt.input), x = Array.from(rt.pre!.tensors.board.data), n = g.snake.length;
		assert.deepEqual(Array.from(rt.sources.fwd.trace!.t.board.data), x, 'the trace reads the same numbers');
		assert.equal(x.slice(0, 81).filter((v) => v === 1).length, 1);
		assert.equal(x.slice(162).filter((v) => v === 1).length, 1);
		const body = g.snake.slice(1).map((c) => x[81 + c]);
		assert.ok(Math.abs(body[0] - (n - 1) / n) < 1e-7 && Math.abs(body[n - 2] - 1 / n) < 1e-7);
		for (let i = 1; i < body.length; i++) assert.ok(body[i] < body[i - 1]);
		const [h, k] = [g.snake[0], g.snake[1]];
		assert.equal(Math.abs(h - k) === 9 || (Math.abs(h - k) === 1 && Math.floor(h / 9) === Math.floor(k / 9)), true, 'the neck touches the head');
	});
});

test('step 2: the float64 network of the processes equals the float32 trace; a move that crashes at once is rated below every move that does not', async () => {
	await each((rt) => {
		const g = posOf(rt.input), q = qOf(rt)!, e = qvalues(weightsOf(rt.model)!, encode(g));
		for (let a = 0; a < 3; a++) assert.ok(Math.abs(q[a] - e[a]) < 1e-5);
		for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (!safe(g, a) && safe(g, b)) assert.ok(q[a] < q[b], `move ${a} (crash) vs ${b}: ${q[a]} ${q[b]}`);
		const t = qOf(rt, 'tgt')!, et = qvalues(weightsOf(rt.sources.tgt.model)!, encode(g));
		for (let a = 0; a < 3; a++) assert.ok(Math.abs(t[a] - et[a]) < 1e-5, 'the target network: trace and float64 agree');
	});
});

test('steps 3 to 6: ε-greedy (u < ε explores), rewards +1 / −1 / 0 only, y = r at the end of a game else r + γ · max Q_target(s′), loss = δ²', async () => {
	await each(async (rt) => {
		await rt.ensure('dqn', 'last');
		const tp = rt.sources.dqn.tape!, f = (ph: string) => tp.frame(tp.find({ phase: ph })).metrics!;
		const a = f('act'), r = f('reward'), t = f('target'), l = f('loss');
		assert.equal(a.explore, a.u < a.eps ? 1 : 0);
		if (!a.explore) assert.equal(a.a, a.best);
		assert.ok([-1, 0, 1].includes(r.r));
		assert.equal(r.r, r.ate ? 1 : r.end === 1 || r.end === 2 ? -1 : 0);
		const g = play(posOf(rt.input).seed, (rt.input as { moves: number[] }).moves), s2 = step(g, a.a).g;
		if (t.terminal) assert.equal(t.y, t.r);
		else {
			const q2 = qvalues(weightsOf(rt.sources.tgt.model)!, encode(s2));
			assert.equal(t.tmax, Math.max(...q2));
			assert.equal(t.y, t.r + t.gamma * t.tmax);
		}
		assert.equal(l.loss, l.delta * l.delta);
		assert.equal(Math.sign(l.delta), Math.sign(l.y - l.qa), 'a step down the loss moves Q(s, a) towards y');
	});
});

test('step 3: the explore fixture explores; the default draw does not', async () => {
	const seen: Record<string, number> = {};
	await withFixtures(await pkgP, async (rt, fx) => { await rt.ensure('dqn', 'last'); const tp = rt.sources.dqn.tape!; seen[fx.id] = tp.frame(tp.find({ phase: 'act' })).metrics!.explore; }, { ...opts, only: ['explore', 'preset:start'] });
	assert.deepEqual(seen, { explore: 1, 'preset:start': 0 });
});

test('the game: only +1 (food), −1 (crash) and 0; a crash ends the game; the k-th food depends only on the seed and k', () => {
	let s = 7;
	const r = () => (s = (s * 1103515245 + 12345) % 2147483648);
	for (let k = 0; k < 200; k++) {
		let g = newGame(r() % 500);
		while (!g.over && g.moves < 300) {
			const o = step(g, r() % 3);
			assert.ok(o.r === (o.ate ? 1 : o.g.over ? -1 : 0));
			g = o.g;
		}
	}
	assert.equal(newGame(42).food, newGame(42).food);
});

test('step 7: the buffer keeps the newest cap moves and the batch is drawn from it', async () => {
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('replay', 'last');
		const tp = rt.sources.replay.tape!, o = tp.opt as { fill: number; cap: number; batch: number }, last = tp.state(tp.length - 1) as { buf: { t: number }[]; pick: number[] };
		assert.equal(last.buf.length, o.cap);
		assert.deepEqual(last.buf.map((x) => x.t).sort((a, b) => a - b), Array.from({ length: o.cap }, (_, i) => o.fill - o.cap + i));
		assert.equal(last.pick.length, o.batch);
	}, { ...opts, only: ['preset:start'] });
});

test('step 8: the target network stays flat between copies and equals the online network right after each copy; the checkpoint is ckSinceSync updates after one', async () => {
	await withFixtures(await pkgP, (rt) => {
		const p = logOf(rt)!.probe, sync = fact(rt, 'sync');
		for (let i = 1; i < p.updates.length; i++) if (Math.floor((p.updates[i] - 1) / sync) === Math.floor((p.updates[i - 1] - 1) / sync) && p.updates[i] % sync !== 0) assert.equal(p.target[i], p.target[i - 1]);
		for (let i = 0; i < p.updates.length; i++) if (p.updates[i] % sync === 0) assert.ok(Math.abs(p.target[i] - p.online[i]) < 1e-6);
		assert.equal(fact(rt, 'ckUpdates') % sync, fact(rt, 'ckSinceSync'));
		const tspec = JSON.parse(readFileSync(join(SITE, 'static', 'models', 'snake-dqn', 'target', 'spec.json'), 'utf8'));
		assert.equal(tspec.copiedAtUpdate, fact(rt, 'ckUpdates') - fact(rt, 'ckSinceSync'));
	}, { ...opts, only: ['preset:start'] });
});

test('step 9: the test score levels off above twenty; training games (with ε) score lower than the test games at the end', async () => {
	await withFixtures(await pkgP, (rt) => {
		const lg = logOf(rt)!, n = lg.score.length, mean = (xs: number[]) => xs.reduce((a, x) => a + x, 0) / xs.length;
		const tail = lg.score.slice(Math.floor(n / 2));
		assert.ok(fact(rt, 'evalBest') > 20 && fact(rt, 'evalBest') < 25);
		for (const v of tail) assert.ok(v > 20 && v < 25, `test score ${v}`);
		assert.ok(mean(lg.train.slice(Math.floor(lg.train.length / 2))) < mean(tail));
		assert.ok(fact(rt, 'randomMean') < 1);
		// "passed half its best (evalHalf) by move halfMoves": the first test score at or above half the best
		assert.ok(Math.abs(fact(rt, 'evalHalf') - fact(rt, 'evalBest') / 2) < 0.05, 'evalHalf is half the best');
		const first = lg.score.findIndex((v) => v >= fact(rt, 'evalBest') / 2);
		assert.equal(lg.moves[first], fact(rt, 'halfMoves'));
		assert.equal(Math.max(...lg.score), fact(rt, 'evalBest'), 'the best test score is the checkpoint’s');
		assert.equal(lg.moves[lg.score.indexOf(fact(rt, 'evalBest'))], fact(rt, 'ckMoves'));
	}, { ...opts, only: ['preset:start'] });
});

test('step 10: the page’s TS agent reproduces evaluate.py’s first 20 games (seeds 1 to 20; spec.agentCheck); the first food is the same for every game on a seed', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, (rt) => {
		const w = weightsOf(rt.model)!, want = check(rt).scores;
		const got = want.map((_: number, i: number) => { const t = record(agent, withWeights({ seed: i + 1 }, w), { cap: 1500 }, { seed: 1 }); return t.frame(t.length - 1).metrics!.score; });
		assert.deepEqual(got, want);
		assert.deepEqual(check(rt).seeds, [1, fact(rt, 'agentGames')], 'one game on each seed from 1 up');
		assert.equal(newGame(7).food, play(7, []).food);
		// "step 9's evalBest came from only evalGames games"
		assert.ok(fact(rt, 'evalGames') < fact(rt, 'agentGames'));
	}, { ...opts, only: ['preset:start'] });
});

test('step 11: the network only reads a 9 × 9 board; it never filled the board; a step penalty lowered its score; tryThis and preset texts hold', async () => {
	await withFixtures(await pkgP, (rt) => {
		const fc1 = rt.spec!.tensors.find((t) => t.name === 'fc1.weight')!;
		assert.deepEqual(fc1.shape, [64, 2592]);
		assert.equal(fact(rt, 'agentFull'), 0);
		assert.equal(fact(rt, 'maxFood'), 78);
		assert.ok(fact(rt, 'hurryMean') < fact(rt, 'agentMean'));
		// "in hurryMoves moves instead of agentMoves"; "it risks more: on hurryAvoidable of its crashes another move was safe, against agentAvoidable"
		assert.ok(fact(rt, 'hurryMoves') < fact(rt, 'agentMoves'));
		assert.ok(fact(rt, 'hurryAvoidable') > fact(rt, 'agentAvoidable'));
		assert.equal(HUNGER, fact(rt, 'hunger'));
		const P = Object.fromEntries((rt.presets ?? []).map((p) => [p.id, posOf(p.raw)]));
		assert.ok(nextCell(P.wall, 0).c < 0, 'next to the wall: straight on hits it');
		assert.equal(nextCell(P.food, 0).c, P.food.food, 'food right ahead');
		assert.equal([0, 1, 2].filter((a) => safe(P.tight, a)).length, 1, 'only one way out');
		assert.ok(P.long.snake.length >= 15);
	}, { ...opts, only: ['preset:start'] });
});
