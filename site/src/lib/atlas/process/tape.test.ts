// W1-C acceptance 1 (determinism and state(i) replay) plus the tape layer's own contract:
// TapeStore, record/recorder (LiveTape), saveTape/loadTape, the tape validator.
// Run from site/:  node --test src/lib/atlas/process/tape.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkDeterminism } from './determinism.ts';
import { record, recorder, stepper } from './record.ts';
import { loadTape, saveTape, headerText } from './io.ts';
import { TapeStore } from './tape.ts';
import { validateTapeHeader } from '../schema/tape.ts';
import { createRng } from '../runtime/seams.ts';
import { walker, type WalkerState } from './fixtures/walker.ts';
import type { TapeHeader } from '../types/process.ts';

test('acceptance 1: same seed -> identical frame hash; state(i) equals a replay for 200 random i (changes + recordDraws)', () => {
	const r = checkDeterminism(walker, { start: 2 }, { steps: 400 }, { seed: 7, recordDraws: true, samples: 200 });
	assert.ok(r.frames > 200, `frames ${r.frames}`);
	assert.equal(r.sameFrames, true, r.problems.join('\n'));
	assert.equal(r.sameStates, true, r.problems.join('\n'));
	assert.equal(r.replayChecked, 200);
	assert.deepEqual(r.replayMismatches, []);
	assert.equal(r.ok, true, r.problems.join('\n'));
	// a different seed gives different frames
	const other = checkDeterminism(walker, { start: 2 }, { steps: 400 }, { seed: 8, recordDraws: true, samples: 20 });
	assert.notEqual(other.frameHash[0], r.frameHash[0]);
});

test('the harness catches an in-place mutation that was not reported in frame.changes', () => {
	const r = checkDeterminism(walker, { start: 2 }, { steps: 400, forgetChange: true }, { seed: 7, samples: 50 });
	assert.equal(r.sameFrames, true);
	assert.equal(r.sameStates, false);
	assert.ok(r.firstStateMismatch > 0);
	assert.match(r.problems.join('\n'), /Q/);
	assert.equal(r.ok, false);
});

test('recordDraws: every value the rng and its forks return is on the frame that consumed it, in order', () => {
	const t = record(walker, { start: 0 }, { steps: 30 }, { seed: 99, recordDraws: true });
	const draws = Array.from({ length: t.length }, (_, i) => t.frame(i).draws ?? []);
	assert.equal(draws[0].length, 0);
	// per step: int(2), normal(), fork('env').next() -> 3 draws on the 'act' frame, none on 'update'
	for (let i = 1; i < t.length; i++) assert.equal(draws[i].length, t.frame(i).phase === 'act' ? 3 : 0, `frame ${i}`);
	// replaying the documented semantics reproduces them
	const rng = createRng(99);
	const env = rng.fork('env');
	const flat = draws.flat();
	const expect: number[] = [];
	for (let s = 0; s < 30; s++) expect.push(rng.int(2), rng.normal(), env.next());
	assert.deepEqual(flat, expect);
	// without recordDraws nothing is stored and the frames are otherwise identical
	const plain = record(walker, { start: 0 }, { steps: 30 }, { seed: 99 });
	assert.equal(plain.frame(1).draws, undefined);
	assert.equal(plain.stateHash(), t.stateHash());
});

test('TapeStore: snapshot vs changes fields, keyframes, derived getters, find, segments, metric', () => {
	const t = record(walker, { start: 0 }, { steps: 100 }, { seed: 3, keyEvery: 16 });
	assert.equal(t.length, 201);
	const last = t.state(t.length - 1) as WalkerState;
	// visits sum = 1 (start) + one per step
	assert.equal(Array.from(last.visits.data).reduce((a, b) => a + b, 0), 101);
	assert.equal(last.best, Math.max(...last.Q.data));
	assert.equal(Object.keys(last).includes('best'), false, 'derived getters are not enumerable');
	// the stored frames never contain the derived field and only anchor Q at keyframes
	for (let i = 0; i < t.length; i++) {
		assert.equal('best' in (t.frame(i).state as object), false);
		if ('Q' in (t.frame(i).state as object)) assert.equal(i % 16, 0, `Q stored at ${i}`);
	}
	// state(i) is pure: asking twice, or out of order, gives the same values
	const a = JSON.stringify(Array.from((t.state(57) as WalkerState).Q.data));
	t.state(3);
	t.state(190);
	assert.equal(JSON.stringify(Array.from((t.state(57) as WalkerState).Q.data)), a);
	// find
	assert.equal(t.find('first'), 0);
	assert.equal(t.find('last'), 200);
	assert.equal(t.find(500), -1);
	assert.equal(t.find({ phase: 'update', step: 10 }), 22);
	assert.equal(t.find({ phase: 'update', last: true }), 200);
	const firstReward = t.find({ phase: 'update', where: (f) => f.metrics!.r === 1 });
	assert.ok(firstReward > 0 && t.frame(firstReward).metrics!.r === 1);
	const withState = t.find({ where: (_f, s: WalkerState) => s.visits.data[8] >= 3 });
	assert.ok(withState > 0 && (t.state(withState) as WalkerState).visits.data[8] >= 3 && (t.state(withState - 1) as WalkerState).visits.data[8] < 3);
	// segments and metrics
	const segs = t.segments();
	assert.equal(segs.length, 200); // [act 0..1] then alternating update/act
	assert.deepEqual(segs[0], { phase: 'act', from: 0, to: 1 });
	const r = t.metric('r');
	assert.ok(Number.isNaN(r[0]) && (r[2] === 0 || r[2] === 1));
	assert.throws(() => t.state(201), RangeError);
});

test("keep: { every } merges dropped frames' changes, so states at kept frames are exact", () => {
	const full = record(walker, { start: 0 }, { steps: 60 }, { seed: 5 });
	const thin = record(walker, { start: 0 }, { steps: 60 }, { seed: 5, keep: { every: 4 } });
	assert.ok(thin.length < full.length / 3);
	// kept frame k corresponds to raw frame 4k; the last kept frame flushes the tail
	for (let k = 0; k < thin.length - 1; k++) {
		const s = thin.state(k) as WalkerState;
		const f = full.state(4 * k) as WalkerState;
		assert.deepEqual(Array.from(s.Q.data), Array.from(f.Q.data), `kept ${k}`);
		assert.deepEqual(Array.from(s.visits.data), Array.from(f.visits.data));
	}
	const endThin = thin.state(thin.length - 1) as WalkerState;
	const endFull = full.state(full.length - 1) as WalkerState;
	assert.deepEqual(Array.from(endThin.Q.data), Array.from(endFull.Q.data));
	assert.equal(thin.frame(1).draws, undefined);
});

test('maxFrames stops the run; set controls reach the generator through the stepper', () => {
	const t = record(walker, { start: 0 }, { steps: 1000 }, { seed: 1, maxFrames: 50 });
	assert.equal(t.length, 50);
	const s = stepper(walker, { start: 0 }, { steps: 5, alpha: 0.5 }, { seed: 1 });
	const frames = [];
	frames.push(s.next()!); // start
	frames.push(s.next()!); // act, step 0: the generator is now paused at this yield
	frames.push(s.next({ type: 'set', opt: { alpha: 0.1 } })!); // the control is the value of that yield
	for (let f = s.next(); f; f = s.next()) frames.push(f);
	assert.equal(frames[2].phase, 'update');
	assert.equal(frames[2].metrics!.alpha, 0.1);
	assert.equal(s.done, true);
});

test('recorder: LiveTape ensure() resolves cursors as frames arrive; pause, step, reset, close', async () => {
	const live = recorder(walker, { start: 0 }, { steps: 300 }, { seed: 4, autorun: false });
	assert.equal(live.length, 0);
	const i = await live.ensure({ phase: 'update', step: 20 });
	assert.equal(live.frame(i).step, 20);
	assert.equal(live.frame(i).phase, 'update');
	assert.ok(live.length > i && !live.done);
	const j = await live.ensure(10);
	assert.equal(j, 10);
	// a cursor that never matches resolves with the last index once the run ends
	const k = await live.ensure({ phase: 'nope' });
	assert.equal(live.done, true);
	assert.equal(k, live.length - 1);
	assert.equal(live.length, 601);
	const done = record(walker, { start: 0 }, { steps: 300 }, { seed: 4 });
	assert.equal(live.store.stateHash(), done.stateHash());
	// subscribe + autorun + pause/step (a long run, so it is still going when paused)
	const auto = recorder(walker, { start: 0 }, { steps: 100000 }, { seed: 4, maxFrames: 1e6 });
	let notified = 0;
	const off = auto.subscribe(() => notified++);
	await auto.ensure({ phase: 'act', step: 50 });
	auto.control({ type: 'pause' });
	assert.equal(auto.done, false);
	const n0 = auto.length;
	await new Promise((r) => setTimeout(r, 30));
	assert.equal(auto.length, n0, 'paused: no background progress');
	auto.control({ type: 'step', n: 5 });
	// K2: wait for the 5 steps (up to 10 s), not a fixed 30 ms: a loaded machine runs the slice later
	for (const t0 = Date.now(); auto.length < n0 + 5 && Date.now() - t0 < 10_000; ) await new Promise((r) => setTimeout(r, 10));
	await new Promise((r) => setTimeout(r, 30));
	assert.equal(auto.length, n0 + 5);
	assert.ok(notified > 0);
	off();
	auto.control({ type: 'reset', seed: 9 });
	assert.equal(auto.length, 0);
	const u = await auto.ensure({ phase: 'update', step: 100 });
	assert.equal(auto.frame(u).step, 100);
	const ref = record(walker, { start: 0 }, { steps: 100000 }, { seed: 9, maxFrames: u + 1 });
	assert.equal(JSON.stringify(Array.from((auto.state(u) as WalkerState).Q.data)), JSON.stringify(Array.from((ref.state(u) as WalkerState).Q.data)));
	auto.close();
	live.close();
});

test('saveTape -> JSON -> loadTape round trip keeps every state; derived fields come back only with the process', () => {
	const t = record(walker, { start: 1 }, { steps: 80 }, { seed: 11, recordDraws: true });
	const { header, blob } = saveTape(t);
	assert.deepEqual(validateTapeHeader(header, blob.byteLength), []);
	// keyframes carry the tracked fields and the key flag (shared layout with pipeline/atlas_kit/tape.py)
	const f64 = header.frames[64] as { key?: boolean; state: Record<string, unknown> };
	assert.equal(f64.key, true);
	assert.ok('Q' in f64.state && 'visits' in f64.state);
	assert.equal((header.frames[65] as { key?: boolean }).key, undefined);
	assert.equal(header.keyEvery, 64);
	const text = headerText(header);
	assert.ok(text.endsWith('\n') && !text.startsWith('﻿'));
	const parsed = JSON.parse(text) as TapeHeader;
	const back = loadTape(parsed, blob, walker);
	assert.equal(back.length, t.length);
	assert.equal(back.stateHash(), t.stateHash());
	assert.equal(back.frameHash(), t.frameHash());
	assert.equal((back.state(back.length - 1) as WalkerState).best, (t.state(t.length - 1) as WalkerState).best);
	const bare = loadTape(parsed, blob);
	assert.equal(bare.stateHash(), t.stateHash());
	assert.equal((bare.state(bare.length - 1) as WalkerState).best, undefined);
	// non-finite numbers survive
	const odd = new TapeStore({ process: 'odd', schema: { x: { kind: 'scalar', label: { en: 'x', zh: 'x' } }, ids: { kind: 'tensor', label: { en: 'ids', zh: 'ids' } } }, phases: { p: { label: { en: 'p', zh: 'p' }, kind: 'other' } } });
	odd.push({ phase: 'p', state: { x: NaN, ids: { shape: [3], data: Int32Array.from([1, 2, 3]), dtype: 'i32' } }, metrics: { a: -Infinity, b: Infinity }, draws: [0.25] });
	const o2 = saveTape(odd, { binName: 'odd.tape.bin' });
	assert.deepEqual(o2.header.bin, { file: 'odd.tape.bin', bytes: 12 }); // no trailing padding
	assert.deepEqual((o2.header.frames[0] as { state: { ids: unknown } }).state.ids, { $t: 'i32', shape: [3], o: 0, n: 3 });
	assert.deepEqual((o2.header.frames[0] as { metrics: unknown }).metrics, { a: { $f: '-Infinity' }, b: { $f: 'Infinity' } });
	const odd2 = loadTape(JSON.parse(JSON.stringify(o2.header)), o2.blob);
	assert.ok(Number.isNaN((odd2.state(0) as { x: number }).x));
	assert.equal(odd2.frame(0).metrics!.a, -Infinity);
	assert.equal(odd2.frameHash(), odd.frameHash());
});

test('validateTapeHeader reports malformed headers', () => {
	const t = record(walker, { start: 1 }, { steps: 3 }, { seed: 11 });
	const { header, blob } = saveTape(t);
	const bad = JSON.parse(JSON.stringify(header));
	bad.format = 'atlas-tape/2';
	bad.provenance = 'made-up';
	bad.schema.pos.kind = 'float';
	bad.frames[1].phase = 'teleport';
	bad.frames[0].state.Q.n = 1e9;
	const problems = validateTapeHeader(bad, blob.byteLength);
	for (const want of [/format/, /provenance/, /schema\.pos\.kind/, /teleport/, /outside|ends at byte/]) assert.ok(problems.some((p) => want.test(p)), `${want} in ${problems.join(' | ')}`);
	assert.throws(() => loadTape(bad, blob), /atlas-tape\/1/);
});
