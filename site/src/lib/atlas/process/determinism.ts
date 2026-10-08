// The process determinism harness (design §6.3 "Process determinism"; W1-B1 acceptance 6 runs trainingProcess through it).
//  1. same seed twice -> identical frame hash;
//  2. keyEvery = 1 (every frame anchored from the process's live objects) and keyEvery = 64 give the same state(i) for
//     every i: this catches in-place mutations of 'changes' fields that were not reported in frame.changes;
//  3. state(i), asked in random order, equals a straight sequential replay of frames 0..i (and the keyEvery = 1 truth)
//     for `samples` random i (default 200).
import type { ProcessDef, RecordOpts } from '../types/process.ts';
import { createRng } from '../runtime/seams.ts';
import { record } from './record.ts';
import type { TapeStore } from './tape.ts';
import { applyChange, clone, hashValue } from './values.ts';

export interface DeterminismReport {
	ok: boolean;
	frames: number;
	frameHash: [string, string];
	stateHash: { key64: string; key1: string };
	sameFrames: boolean;
	sameStates: boolean;
	replayChecked: number;
	replayMismatches: number[];
	/** first frame where keyEvery 1 and 64 disagree (-1 if none) */
	firstStateMismatch: number;
	problems: string[];
}

/** State after frames 0..i, rebuilt from scratch by applying every stored frame in order (no keyframes, no cache). */
export function straightReplay(t: TapeStore, upTo: number, out: Record<string, unknown>[] = []): Record<string, unknown>[] {
	const changes = t.changeTracked;
	const s: Record<string, unknown> = {};
	for (let j = 0; j <= upTo; j++) {
		const f = t.frame(j);
		for (const [k, v] of Object.entries(f.state as Record<string, unknown>)) s[k] = changes.has(k) ? clone(v) : v;
		for (const c of f.changes ?? []) if (changes.has(c.field)) applyChange(s, c);
		out[j] = { ...s, ...Object.fromEntries([...changes].filter((k) => k in s).map((k) => [k, clone(s[k])])) };
	}
	return out;
}

export function checkDeterminism<I, O extends Record<string, unknown>, S>(
	p: ProcessDef<I, O, S>,
	input: I,
	opt?: Partial<O>,
	ro?: RecordOpts & { samples?: number; sampleSeed?: number }
): DeterminismReport {
	const problems: string[] = [];
	const base = { ...(ro ?? {}) };
	const a = record(p, input, opt, { ...base, keyEvery: 64 });
	const b = record(p, input, opt, { ...base, keyEvery: 64 });
	const k1 = record(p, input, opt, { ...base, keyEvery: 1 });
	const frameHash: [string, string] = [a.frameHash(), b.frameHash()];
	const sameFrames = frameHash[0] === frameHash[1];
	if (!sameFrames) problems.push('two runs with the same seed recorded different frames');
	if (a.length !== k1.length) problems.push(`keyEvery 1 recorded ${k1.length} frames, keyEvery 64 recorded ${a.length}`);
	let firstStateMismatch = -1;
	const n = Math.min(a.length, k1.length);
	for (let i = 0; i < n && firstStateMismatch < 0; i++) if (hashValue(a.rawState(i)) !== hashValue(k1.rawState(i))) firstStateMismatch = i;
	const stateHash = { key64: a.stateHash(), key1: k1.stateHash() };
	const sameStates = firstStateMismatch < 0 && stateHash.key64 === stateHash.key1;
	if (!sameStates) {
		const i = Math.max(0, firstStateMismatch);
		const s64 = a.rawState(i);
		const s1 = k1.rawState(i);
		const fields = Object.keys({ ...s64, ...s1 }).filter((k) => hashValue(s64[k]) !== hashValue(s1[k]));
		problems.push(`state(${i}) differs between keyEvery 1 and 64 in ${fields.join(', ') || '?'}: a 'changes' field was mutated without a matching frame.changes entry`);
	}
	if (a.problems.length) problems.push(...a.problems.slice(0, 5));
	// random-order state(i) vs a straight replay and vs the keyEvery = 1 truth
	const samples = Math.min(ro?.samples ?? 200, a.length);
	const rng = createRng(ro?.sampleSeed ?? 12345);
	const picks = a.length <= samples ? Array.from({ length: a.length }, (_, i) => i) : Array.from({ length: samples }, () => rng.int(a.length));
	for (let k = picks.length - 1; k > 0; k--) {
		const j = rng.int(k + 1);
		[picks[k], picks[j]] = [picks[j], picks[k]];
	}
	const replay = a.length ? straightReplay(a, a.length - 1) : [];
	const replayMismatches: number[] = [];
	for (const i of picks) {
		const got = hashValue(a.rawState(i));
		if (got !== hashValue(replay[i]) || (i < k1.length && got !== hashValue(k1.rawState(i)))) replayMismatches.push(i);
	}
	if (replayMismatches.length) problems.push(`state(i) differs from a straight replay at ${replayMismatches.length} of ${picks.length} sampled frames (first ${replayMismatches[0]})`);
	return {
		ok: problems.length === 0,
		frames: a.length,
		frameHash,
		stateHash,
		sameFrames,
		sameStates,
		replayChecked: picks.length,
		replayMismatches,
		firstStateMismatch,
		problems
	};
}
