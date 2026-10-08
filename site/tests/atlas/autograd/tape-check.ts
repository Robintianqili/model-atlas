// A local copy of W1-C's process determinism checks (design §6.3 "Process determinism", W1-C acceptance 1), used by
// W1-B1 before integration because W1-C's TapeStore lives in another lane. At integration, swap MiniTape for
// process/tape.ts TapeStore and record() for process/record.ts.
//  1. the same seed twice gives an identical frame hash;
//  2. keyEvery = 1 equals keyEvery = 64 (catches unreported in-place mutations of tensors already in frames);
//  3. state(i) equals a straight replay for 200 random i.
import { createHash } from 'node:crypto';
import type { Control, Frame, ProcessDef, Rng } from '../../../src/lib/atlas/types/index.ts';
import { makeRng } from './rng-stub.ts';

/** Run a process to the end (or maxFrames), feeding controls from `controlAt(frameIndex)`. */
export function runProcess<I, O, S>(
	p: ProcessDef<I, O, S>,
	input: I,
	o: { opt?: Partial<O>; seed: number; detail?: 'coarse' | 'fine'; maxFrames?: number; controlAt?: (i: number, f: Frame<S>) => Control<O> | undefined; rng?: (seed: number) => Rng }
): Frame<S>[] {
	const rng = (o.rng ?? makeRng)(o.seed);
	const gen = p.run(input, { rng, opt: { ...p.defaults, ...(o.opt ?? {}) } as O, detail: o.detail ?? 'coarse', signal: { cancelled: false } });
	const frames: Frame<S>[] = [];
	let r = gen.next();
	while (!r.done) {
		frames.push(r.value);
		if (frames.length >= (o.maxFrames ?? 20000)) break;
		r = gen.next(o.controlAt ? o.controlAt(frames.length - 1, r.value) : undefined);
	}
	return frames;
}

function deepClone<T>(v: T): T {
	return structuredClone(v);
}

/** The minimum TapeStore behaviour the checks need: snapshot fields carry forward; keyframes are deep copies taken at push time. */
export class MiniTape<S = Record<string, unknown>> {
	readonly frames: Frame<S>[] = [];
	private readonly keys = new Map<number, Partial<S>>();
	private acc: Partial<S> = {};
	private readonly keyEvery: number;
	constructor(keyEvery = 64) {
		this.keyEvery = keyEvery;
	}
	push(f: Frame<S>): void {
		const i = this.frames.length;
		this.frames.push(f);
		this.acc = { ...this.acc, ...f.state };
		if (i % this.keyEvery === 0) this.keys.set(i, deepClone(this.acc));
	}
	state(i: number): Partial<S> {
		const k = i - (i % this.keyEvery);
		const s = deepClone(this.keys.get(k)!);
		for (let j = k + 1; j <= i; j++) Object.assign(s as object, this.frames[j].state);
		return s;
	}
}

function feed(h: ReturnType<typeof createHash>, v: unknown): void {
	if (v === null || v === undefined) h.update(v === null ? 'N' : 'U');
	else if (typeof v === 'number') {
		const b = new Float64Array([v]);
		h.update('n');
		h.update(new Uint8Array(b.buffer));
	} else if (typeof v === 'string') h.update(`s${v.length}:${v}`);
	else if (typeof v === 'boolean') h.update(v ? 'T' : 'F');
	else if (ArrayBuffer.isView(v)) {
		const arr = Float64Array.from(v as unknown as ArrayLike<number>);
		h.update(`a${arr.length}:`);
		h.update(new Uint8Array(arr.buffer));
	} else if (Array.isArray(v)) {
		h.update(`[${v.length}`);
		for (const x of v) feed(h, x);
	} else if (typeof v === 'object') {
		const keys = Object.keys(v as object).sort();
		h.update(`{${keys.length}`);
		for (const k of keys) {
			h.update(`k${k}`);
			feed(h, (v as Record<string, unknown>)[k]);
		}
	} else h.update(`?${String(v)}`);
}

export function hashOf(v: unknown): string {
	const h = createHash('sha256');
	feed(h, v);
	return h.digest('hex');
}

export interface DeterminismReport {
	frames: number;
	sameSeed: boolean;
	keyEvery: boolean;
	replay: boolean;
	checkedStates: number;
	hash: string;
}

export function checkDeterminism<I, O, S>(
	p: ProcessDef<I, O, S>,
	input: () => I,
	o: { opt?: Partial<O>; seed: number; detail?: 'coarse' | 'fine'; samples?: number; controlAt?: (i: number, f: Frame<S>) => Control<O> | undefined }
): DeterminismReport {
	const a = runProcess(p, input(), o);
	const b = runProcess(p, input(), o);
	const ha = hashOf(a);
	const sameSeed = ha === hashOf(b);

	// tapes built while the run is in progress, as a live recorder would
	const t1 = new MiniTape<S>(1);
	const t64 = new MiniTape<S>(64);
	{
		const rng = makeRng(o.seed);
		const gen = p.run(input(), { rng, opt: { ...p.defaults, ...(o.opt ?? {}) } as O, detail: o.detail ?? 'coarse', signal: { cancelled: false } });
		let r = gen.next();
		let i = 0;
		while (!r.done) {
			t1.push(r.value);
			t64.push(r.value);
			r = gen.next(o.controlAt ? o.controlAt(i, r.value) : undefined);
			i++;
		}
	}
	let keyEvery = t1.frames.length === t64.frames.length;
	for (let i = 0; i < t1.frames.length && keyEvery; i++) if (hashOf(t1.state(i)) !== hashOf(t64.state(i))) keyEvery = false;

	// straight replay: a fresh run, each accumulated state hashed the moment its frame is produced
	const replayStates: string[] = [];
	{
		const rng = makeRng(o.seed);
		const gen = p.run(input(), { rng, opt: { ...p.defaults, ...(o.opt ?? {}) } as O, detail: o.detail ?? 'coarse', signal: { cancelled: false } });
		let acc: Partial<S> = {};
		let r = gen.next();
		let i = 0;
		while (!r.done) {
			acc = { ...acc, ...r.value.state };
			replayStates.push(hashOf(acc));
			r = gen.next(o.controlAt ? o.controlAt(i, r.value) : undefined);
			i++;
		}
	}
	const pick = makeRng(o.seed ^ 0x5eed);
	const samples = Math.min(o.samples ?? 200, t64.frames.length);
	let replay = replayStates.length === t64.frames.length;
	for (let s = 0; s < samples && replay; s++) {
		const i = samples >= t64.frames.length ? s : pick.int(t64.frames.length);
		if (hashOf(t64.state(i)) !== replayStates[i]) replay = false;
	}
	return { frames: a.length, sameSeed, keyEvery, replay, checkedStates: samples, hash: ha };
}
