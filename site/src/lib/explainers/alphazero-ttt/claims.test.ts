// alphazero-ttt claim tests: every sentence of the lesson and the manifest that is not computed live is asserted here,
// on every fixture where it depends on the input (lesson fixtures, the adapter's edge fixtures and the presets).
// `atlas check` runs this file (AK408 on failure).   node --test src/lib/explainers/alphazero-ttt/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AtlasRuntime } from '../../atlas/runtime/runtime.ts';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { rawOf } from './adapters/ttt-board/adapter.ts';
import { boardAt, posOf, stateOf, kidsOf, gameOf } from './picture.ts';
import { search, game } from './process.ts';
import { EMPTY, type Board, boardOf, isOver, legal, lines, moveValue, play, solve, toMove, treeSize, urgent, view, winner } from './rules.ts';
import { bestMove, evaluate, score, weightsOf } from './search.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('alphazero-ttt', SITE).then((l) => l.pkg);
const open = (rt: AtlasRuntime) => !isOver(posOf(rt));
const each = async (fn: (rt: AtlasRuntime) => void | Promise<void>) => withFixtures(await pkgP, async (rt) => (open(rt) ? fn(rt) : undefined), opts);
const facts = (rt: AtlasRuntime) => (k: string) => rt.num(`fact:${k}`);

test('manifest, catalog, summary: 11,242 parameters, 9 → 96 → 96 → 9 + 1, 50 simulations by default', async () => {
	await withFixtures(await pkgP, (rt) => {
		const sp = rt.spec!, sh = (id: string) => sp.layers.find((l) => l.id === id)!.outShape;
		assert.equal(sp.params, 11242);
		assert.deepEqual([sh('board'), sh('fc1'), sh('fc2'), sh('prior'), sh('value')], [[9], [96], [96], [9], [1]]);
		assert.equal(search.defaults.sims, 50);
		assert.equal(game.defaults.sims, 50);
		assert.equal(search.defaults.c, game.defaults.c);
	}, { ...opts, only: ['preset:start'] });
});

test('step 1: nine numbers from the side of the player to move (+1 mine, −1 theirs, 0 empty); the other side sees every sign flipped; the mask is 0 or −∞', async () => {
	await each((rt) => {
		const b = posOf(rt), x = Array.from(rt.pre!.tensors.board.data), mask = Array.from(rt.pre!.tensors.mask.data);
		assert.deepEqual(x, b.map((v) => (v === 0 ? 0 : v === toMove(b) ? 1 : -1)));
		assert.deepEqual(b.map((v) => v * -toMove(b) || 0), view(b).map((v) => -v || 0), 'the other side: every sign flipped');
		assert.deepEqual(Array.from(rt.sources.fwd.trace!.t.board.data), x, 'the trace reads the same numbers');
		assert.deepEqual(mask, b.map((v) => (v === 0 ? 0 : -Infinity)));
	});
});

test('step 2: the priors add up to 1 over the empty cells and a taken cell gets exactly 0; the search starts from the same priors', async () => {
	await each(async (rt) => {
		const b = posOf(rt), p = Array.from(rt.sources.fwd.trace!.t.prior.data);
		assert.ok(Math.abs(p.reduce((a, x) => a + x, 0) - 1) < 1e-6);
		for (let i = 0; i < 9; i++) if (b[i] !== 0) assert.equal(p[i], 0);
		const e = evaluate(weightsOf(rt.model!), b);
		for (let i = 0; i < 9; i++) assert.ok(Math.abs(e.p[i] - p[i]) < 1e-6, `float64 search vs float32 trace, cell ${i}`);
		assert.ok(Math.abs(e.v - rt.sources.fwd.trace!.t.value.data[0]) < 1e-6);
	});
});

test('step 3: the value is between −1 and +1', async () => {
	await each((rt) => { const v = rt.sources.fwd.trace!.t.value.data[0]; assert.ok(v >= -1 && v <= 1); });
});

test('steps 4-7: every choice takes the largest U (ties: lowest cell); new branches get the network’s priors; the value flips sign at every level; every simulation starts at the board', async () => {
	await each(async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, root = posOf(rt), w = weightsOf(rt.model!);
		for (let i = 0; i < tp.length; i++) {
			const f = tp.frame(i), m = f.metrics!, s = stateOf({ tape: tp, index: i })!;
			if (f.phase === 'select') {
				const us = kidsOf(s, m.node).map((k) => ({ cell: s.move.data[k], u: score(s.N.data, s.W.data, s.P.data, m.node, k, m.c).u }));
				const best = us.reduce((a, x) => (x.u > a.u ? x : a));
				assert.equal(m.cell, best.cell);
			} else if (f.phase === 'expand' && !m.terminal) {
				const s0 = stateOf({ tape: tp, index: i })!, nb = boardAtNode(s0, root, m.node), e = evaluate(w, nb);
				for (const k of kidsOf(s0, m.node)) assert.equal(s0.P.data[k], e.p[s0.move.data[k]]);
				assert.equal(m.v, e.v);
			} else if (f.phase === 'backup') {
				const path = s.path, before = stateOf({ tape: tp, index: i - 1 })!;
				assert.equal(path[0], 0, 'every simulation starts at the board');
				path.forEach((x, j) => {
					const sign = (path.length - 1 - j) % 2 === 0 ? -1 : 1;
					assert.ok(Math.abs(s.W.data[x] - before.W.data[x] - sign * m.v) < 1e-12, `node ${x}`);
					assert.equal(s.N.data[x], before.N.data[x] + 1);
				});
			}
		}
	});
});
function boardAtNode(s: NonNullable<ReturnType<typeof stateOf>>, root: Board, node: number): Board {
	const moves: number[] = [];
	for (let x = node; x > 0; x = s.parent.data[x]) moves.unshift(s.move.data[x]);
	return moves.reduce(play, root);
}

test('step 8 and the misconception: it plays the most-visited move; the network looks at no more than sims + 1 positions', async () => {
	await each(async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, f = tp.frame(tp.length - 1), s = stateOf({ tape: tp, index: tp.length - 1 })!;
		assert.equal(f.phase, 'choose');
		const most = kidsOf(s, 0).reduce((a, k) => (s.N.data[k] > s.N.data[a] ? k : a));
		assert.equal(f.metrics!.cell, s.move.data[most]);
		assert.ok(f.metrics!.nets <= f.metrics!.sims + 1);
	});
});

test('step 9: each reply of the search is a fresh search from that position; in every possible game it never loses (as X and as O), as facts.json says', async () => {
	const pkg = await pkgP;
	await withFixtures(pkg, async (rt) => {
		await rt.ensure('game', 'last');
		const tp = rt.sources.game.tape!, w = weightsOf(rt.model!);
		for (let i = 1; i < tp.length; i++) {
			const f = tp.frame(i);
			if (f.phase !== 'ai') continue;
			const before = Array.from((tp.state(i - 1) as { board: { data: Float64Array } }).board.data);
			assert.equal(f.metrics!.cell, bestMove(w, before, 50, 1.5).cell);
		}
		const g = gameOf(rt)!;
		if (rawOf(rt.input).game!.length >= 9) assert.ok(g.over, 'a game with enough moves ends');
		// every opponent line, the search on its own turns (memoized by position)
		const memo = new Map<string, number>();
		let games = 0, losses = 0;
		const rec = (b: Board, ai: number) => {
			if (isOver(b)) { games++; if (winner(b) === -ai) losses++; return; }
			if (toMove(b) === ai) {
				const k = b.join();
				if (!memo.has(k)) memo.set(k, bestMove(w, b, 50, 1.5).cell);
				rec(play(b, memo.get(k)!), ai);
			} else for (const c of legal(b)) rec(play(b, c), ai);
		};
		rec(EMPTY, 1);
		rec(EMPTY, -1);
		assert.equal(games, facts(rt)('searchGames'));
		assert.equal(losses, facts(rt)('searchLosses'));
	}, { ...opts, only: ['game-over'] });
});

test('steps 10-11 and the prequestion: the measured facts agree with the page’s own rules; the self-play curve peaks before the last round', async () => {
	await withFixtures(await pkgP, (rt) => {
		const f = facts(rt);
		const seen = new Set<string>(), stack = [EMPTY];
		while (stack.length) { const b = stack.pop()!; if (seen.has(b.join())) continue; seen.add(b.join()); if (!isOver(b)) for (const c of legal(b)) stack.push(play(b, c)); }
		assert.equal(seen.size, f('positions'));
		assert.equal(lines(EMPTY), f('games'));
		assert.equal(solve(EMPTY), 0, 'perfect play from the empty board is a draw');
		assert.ok(f('priorPerfect') > 0.5 && f('priorPerfect') < 1, 'on most positions, not all');
		assert.ok(f('searchPerfect') > f('priorPerfect') && f('fixedBySearch') > 0);
		const acc = (rt.spec as unknown as { selfplay: { acc: number[] } }).selfplay.acc, top = acc.indexOf(Math.max(...acc));
		assert.ok(top < acc.length - 1 && acc[acc.length - 1] < acc[top], 'peaks early, then sags');
		assert.ok(acc[1] > acc[0], 'rises fast');
		assert.equal(acc[0], f('accStart'));
		assert.equal(acc[acc.length - 1], f('accEnd'));
		assert.equal(f('selfplayGames'), f('rounds') * f('gamesPerRound'));
	}, { ...opts, only: ['preset:start'] });
});

test('the start position: the network’s favourite is not perfect, the search fixes it (steps 4 and 8 say so on the start board)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, m = tp.frame(tp.length - 1).metrics!, b = posOf(rt);
		assert.equal(m.agree, 0);
		assert.ok(moveValue(b, m.top) < solve(b), 'the favourite loses');
		assert.equal(moveValue(b, m.cell), solve(b), 'the search keeps the draw');
	}, { ...opts, only: ['preset:start'] });
});

// ---- added by the teaching review (notes/review-alphazero-ttt.md) ----

/** Every position reachable from the empty board (5,478), each once. */
function allPositions(): Board[] {
	const seen = new Map<string, Board>(), stack = [EMPTY];
	while (stack.length) {
		const b = stack.pop()!;
		if (seen.has(b.join())) continue;
		seen.set(b.join(), b);
		if (!isOver(b)) for (const c of legal(b)) stack.push(play(b, c));
	}
	return [...seen.values()];
}

test('step 4: simulation 1 tries the network’s favourite (every Q is still 0), so at simulation 2 the favourite has one visit; the trace and the search agree on the favourite', async () => {
	await each(async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, b = posOf(rt), p = Array.from(rt.sources.fwd.trace!.t.prior.data);
		const i1 = tp.find({ phase: 'select', step: 1, sub: 0 }), i2 = tp.find({ phase: 'select', step: 2, sub: 0 });
		const s1 = stateOf({ tape: tp, index: i1 })!, s2 = stateOf({ tape: tp, index: i2 })!;
		const fav = kidsOf(s1, 0).reduce((a, k) => (s1.P.data[k] > s1.P.data[a] ? k : a));
		assert.equal(tp.frame(i1).metrics!.cell, s1.move.data[fav], 'simulation 1 picks the largest prior');
		assert.equal(tp.frame(i1).metrics!.q, 0);
		assert.equal(s1.move.data[fav], legal(b).reduce((a, c) => (p[c] > p[a] ? c : a)), 'the search’s favourite is the trace’s favourite (step 2)');
		assert.equal(s2.N.data[fav], 1);
		assert.equal(tp.frame(i2).metrics!.N, 2, 'N = 2 at simulation 2: the first look counts as a visit');
	});
});

test('step 5: “wins at once” and “blocks the three in a row” mean what they say; on the start board the new position’s top prior blocks O', async () => {
	const after = boardOf([2, 3, 7, 4]);
	assert.equal(urgent(after, 5), -1, 'X must block O’s middle row');
	assert.equal(urgent(boardOf([0, 3, 1, 4]), 2), 1, 'X completes the top row');
	assert.equal(urgent(boardOf([0, 3, 1, 4]), 8), 0);
	for (const b of allPositions()) {
		if (isOver(b)) continue;
		for (const c of legal(b)) {
			const u = urgent(b, c), o = b.slice();
			o[c] = -toMove(b);
			assert.equal(u === 1, winner(play(b, c)) === toMove(b));
			if (u === -1) assert.ok(winner(o) === -toMove(b) && winner(play(b, c)) === 0);
		}
	}
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, i = tp.find({ phase: 'expand', step: 2 }), m = tp.frame(i).metrics!;
		const nb = boardAt(stateOf({ tape: tp, index: i })!, posOf(rt), m.node);
		assert.equal(m.cell, 4, 'simulation 2 goes to the centre');
		assert.equal(m.top, 5);
		assert.equal(urgent(nb, m.top), -1);
	}, { ...opts, only: ['preset:start'] });
});

test('steps 7 and 11: a box appears when a simulation first reaches its position; the network looked at exactly the visited positions that are not finished games', async () => {
	await each(async (rt) => {
		await rt.ensure('search', 'last');
		const tp = rt.sources.search.tape!, last = tp.length - 1, s = stateOf({ tape: tp, index: last })!, root = posOf(rt);
		let looked = 0;
		for (let k = 0; k < s.count; k++) if (s.N.data[k] > 0 && !isOver(boardAt(s, root, k))) looked++;
		assert.equal(looked, tp.frame(last).metrics!.nets);
		for (let k = 1; k < s.count; k++) if (s.N.data[k] === 0) assert.equal(kidsOf(s, k).length, 0, 'a position never reached has no branches yet');
	});
});

test('step 8: the page’s own network and search reproduce evaluate.py on all 4,520 open positions (favourite 94.0%, search 98.6%, 206 fixed, 0 spoiled)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const w = weightsOf(rt.model!), f = facts(rt);
		let open = 0, pp = 0, sp = 0, fixed = 0, broken = 0;
		for (const b of allPositions()) {
			if (isOver(b)) continue;
			open++;
			const best = solve(b), e = evaluate(w, b), top = legal(b).reduce((a, c) => (e.p[c] > e.p[a] ? c : a));
			const okP = moveValue(b, top) === best, okS = moveValue(b, bestMove(w, b, 50, 1.5).cell) === best;
			pp += +okP;
			sp += +okS;
			if (okS && !okP) fixed++;
			if (okP && !okS) broken++;
		}
		assert.equal(open, f('openPositions'));
		assert.equal(pp / open, f('priorPerfect'));
		assert.equal(sp / open, f('searchPerfect'));
		assert.equal(fixed, f('fixedBySearch'));
		assert.equal(broken, f('brokenBySearch'));
	}, { ...opts, only: ['preset:start'] });
});

test('step 10: many positions have several perfect moves, so a random pick is perfect on more than half of the open positions; the untrained network started near that', async () => {
	await withFixtures(await pkgP, (rt) => {
		let open = 0, share = 0;
		for (const b of allPositions()) {
			if (isOver(b)) continue;
			open++;
			const best = solve(b), L = legal(b);
			share += L.filter((c) => moveValue(b, c) === best).length / L.length;
		}
		assert.ok(share / open > 0.5, `random pick ${share / open}`);
		assert.ok(Math.abs(facts(rt)('accStart') - share / open) < 0.1);
	}, { ...opts, only: ['preset:start'] });
});

test('step 11: 5,478 different positions, but 549,946 in the game tree from the empty board (a position counted once per order of moves)', async () => {
	await withFixtures(await pkgP, (rt) => {
		assert.equal(allPositions().length, facts(rt)('positions'));
		assert.equal(treeSize(EMPTY), facts(rt)('treeNodes'));
		assert.ok(treeSize(EMPTY) > facts(rt)('positions'));
	}, { ...opts, only: ['preset:start'] });
});

test('try this: with perfect play every first move draws; on “A chance for O” O can force a win, the network’s favourite does not keep it and the search’s move does', async () => {
	for (const c of legal(EMPTY)) assert.equal(moveValue(EMPTY, c), 0);
	await withFixtures(await pkgP, async (rt) => {
		await rt.ensure('search', 'last');
		const b = posOf(rt), tp = rt.sources.search.tape!, m = tp.frame(tp.length - 1).metrics!;
		assert.equal(toMove(b), -1, 'O to move');
		assert.equal(solve(b), 1, 'O can force a win');
		assert.ok(moveValue(b, m.top) < 1, 'the favourite does not keep the win');
		assert.equal(moveValue(b, m.cell), 1, 'the search’s move keeps it');
	}, { ...opts, only: ['preset:chance'] });
});
