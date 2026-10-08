// Recording a ProcessDef into a tape (design §4.4):
//   record(p, input, opt?, ro?)   -> Tape      (runs to the end, synchronously)
//   recorder(p, input, opt?, ro?) -> LiveTape  (advances in <= 8 ms slices; async ensure())
//   stepper(p, input, opt?, ro?)               the frame source both use. W1-B2's engine host should use it too, so
//                                              worker frames and local frames are identical (draws, anchors, keep).
import type { Rng } from '../types/core.ts';
import type { Control, Cursor, Frame, LiveTape, ProcessDef, RecordOpts, RunCtx } from '../types/process.ts';
import { createRng } from '../runtime/seams.ts';
import { TapeStore } from './tape.ts';
import { clone } from './values.ts';

export interface RecorderOpts extends RecordOpts {
	/** recorder(): keep advancing in the background (default true). false: advance only for ensure()/step controls. */
	autorun?: boolean;
	/** recorder(): time slice in ms (default 8) */
	sliceMs?: number;
	provenance?: 'live' | 'precomputed';
}

const DEFAULTS = { keyEvery: 64, maxFrames: 20000, seed: 1 };

/** Wrap an Rng so every value its public methods return (and its forks') is appended to `sink`. */
export function recordingRng(base: Rng, sink: { draws: number[] }): Rng {
	const w: Rng = {
		get seed() {
			return base.seed;
		},
		next: () => {
			const v = base.next();
			sink.draws.push(v);
			return v;
		},
		int: (n: number) => {
			const v = base.int(n);
			sink.draws.push(v);
			return v;
		},
		normal: () => {
			const v = base.normal();
			sink.draws.push(v);
			return v;
		},
		fork: (label: string) => recordingRng(base.fork(label), sink),
		getState: () => base.getState(),
		setState: (s: number) => base.setState(s)
	};
	return w;
}

export interface Stepper<S = any> {
	readonly done: boolean;
	/** frames handed out so far (after `keep` merging) */
	readonly count: number;
	/** true when maxFrames stopped the run early */
	readonly truncated: boolean;
	readonly opt: Record<string, unknown>;
	readonly seed: number;
	/** The next frame for the tape, or null when the run is over. `index` is the tape index the frame will get, used to
	 *  anchor 'changes' fields at keyframes. Controls 'set' and 'input' go to the generator; the others are the caller's. */
	next(control?: Control): Frame<S> | null;
	cancel(): void;
}

export function stepper<I, O extends Record<string, unknown>, S>(p: ProcessDef<I, O, S>, input: I, opt?: Partial<O>, ro?: RecordOpts): Stepper<S> {
	const keyEvery = Math.max(1, Math.floor(ro?.keyEvery ?? DEFAULTS.keyEvery));
	const maxFrames = ro?.maxFrames ?? DEFAULTS.maxFrames;
	const every = ro?.keep && ro.keep !== 'all' ? Math.max(1, Math.floor(ro.keep.every)) : 1;
	const seed = (ro?.seed ?? DEFAULTS.seed) >>> 0;
	const sink = { draws: [] as number[] };
	const base = createRng(seed);
	const signal = { cancelled: false };
	const merged = { ...(p.defaults as object), ...((opt ?? {}) as object) } as O;
	const ctx: RunCtx<O> = { rng: ro?.recordDraws ? recordingRng(base, sink) : base, opt: merged, detail: ro?.detail ?? 'coarse', signal };
	const changeFields = Object.entries(p.schema)
		.filter(([, d]) => d.track === 'changes')
		.map(([k]) => k);
	const live: Record<string, unknown> = {};
	let gen: Generator<Frame<S>, void, Control<O> | undefined> | null = null;
	let done = false;
	let truncated = false;
	let count = 0;
	let raw = 0; // frames yielded by the generator
	let pending: Frame<S> | null = null; // merged frames dropped by `keep`

	const merge = (into: Frame<S> | null, f: Frame<S>): Frame<S> => {
		if (!into) return { ...f, state: { ...(f.state as object) } as Partial<S>, changes: f.changes?.slice(), draws: f.draws?.slice() };
		return {
			...f,
			state: { ...(into.state as object), ...(f.state as object) } as Partial<S>,
			changes: [...(into.changes ?? []), ...(f.changes ?? [])],
			draws: [...(into.draws ?? []), ...(f.draws ?? [])]
		};
	};

	const self: Stepper<S> = {
		get done() {
			return done;
		},
		get count() {
			return count;
		},
		get truncated() {
			return truncated;
		},
		opt: merged as Record<string, unknown>,
		seed,
		cancel() {
			signal.cancelled = true;
			done = true;
		},
		next(control?: Control): Frame<S> | null {
			while (!done) {
				if (count >= maxFrames) {
					truncated = true;
					done = true;
					break;
				}
				let r: IteratorResult<Frame<S>, void>;
				if (!gen) {
					if (control?.type === 'set') Object.assign(merged, control.opt);
					gen = p.run(input, ctx);
					r = gen.next();
				} else {
					const c = control && (control.type === 'set' || control.type === 'input') ? (control as Control<O>) : undefined;
					r = gen.next(c);
				}
				control = undefined;
				if (r.done) {
					done = true;
					break;
				}
				let f = r.value;
				if (sink.draws.length) {
					f = { ...f, draws: [...(f.draws ?? []), ...sink.draws] };
					sink.draws = [];
				}
				for (const k of changeFields) if (f.state && k in (f.state as object)) live[k] = (f.state as Record<string, unknown>)[k];
				raw++;
				pending = merge(pending, f);
				if ((raw - 1) % every !== 0 && raw > 1) continue;
				let out = pending;
				pending = null;
				if (count % keyEvery === 0) {
					// anchor every 'changes' field at a keyframe from the process's live object
					const st = { ...(out.state as object) } as Record<string, unknown>;
					for (const k of changeFields) if (k in live && !(k in st)) st[k] = live[k];
					out = { ...out, state: st as Partial<S> };
				}
				if (out.changes && out.changes.length === 0) delete out.changes;
				if (out.draws && out.draws.length === 0) delete out.draws;
				count++;
				return out;
			}
			if (pending) {
				// frames dropped by `keep` at the end of the run: flush them as the final frame
				const out = pending;
				pending = null;
				if (out.changes && out.changes.length === 0) delete out.changes;
				if (out.draws && out.draws.length === 0) delete out.draws;
				count++;
				return out;
			}
			return null;
		}
	};
	return self;
}

function storeFor<S>(p: ProcessDef<any, any, S>, s: Stepper<S>, ro?: RecorderOpts): TapeStore<S> {
	return new TapeStore<S>({
		process: p.id,
		seed: s.seed,
		opt: clone(s.opt),
		provenance: ro?.provenance ?? 'live',
		version: String((p as { version?: unknown }).version ?? '1'),
		schema: p.schema,
		phases: p.phases,
		keyEvery: ro?.keyEvery ?? DEFAULTS.keyEvery
	});
}

/** Run a process to the end (or maxFrames) and return its tape. */
export function record<I, O extends Record<string, unknown>, S>(p: ProcessDef<I, O, S>, input: I, opt?: Partial<O>, ro?: RecorderOpts): TapeStore<S> {
	const s = stepper(p, input, opt, ro);
	const store = storeFor(p, s, ro);
	for (let f = s.next(); f; f = s.next()) store.push(f, { owned: false });
	return store;
}

// ------------------------------------------------------------------ LiveTape
type Waiter = { c: Cursor; from: number; resolve: (i: number) => void };

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const later = (fn: () => void) => setTimeout(fn, 0);
const onFrame = (fn: () => void) => {
	const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => number }).requestAnimationFrame;
	if (raf) raf(fn);
	else setTimeout(fn, 0);
};

export interface Recorder<S = any> extends LiveTape<S> {
	readonly store: TapeStore<S>;
	readonly paused: boolean;
	readonly truncated: boolean;
	readonly error: string | null;
}

/** A LiveTape that records in the background in slices of <= sliceMs (default 8 ms).
 *  ensure(c) also drives the run while paused. 'reset' restarts with a fresh tape store (the old one is left as is), on the
 *  latest input ('input' controls) and options ('set' controls) the tape was given (K0: it used to restart on the open-time
 *  input and options, so a finished train source ignored a new input). */
export function recorder<I, O extends Record<string, unknown>, S>(p: ProcessDef<I, O, S>, input: I, opt?: Partial<O>, ro?: RecorderOpts): Recorder<S> {
	let roNow: RecorderOpts = { ...(ro ?? {}) };
	// what a 'reset' restarts with: the latest input and the options with every 'set' so far
	let inputNow: I = input;
	let optNow: Partial<O> = { ...((opt ?? {}) as object) } as Partial<O>;
	let s = stepper(p, input, opt, roNow);
	let store = storeFor(p, s, roNow);
	let closed = false;
	let error: string | null = null;
	let scheduled = false;
	let stepBudget = 0; // frames requested by { type: 'step' } while paused
	// 'set' / 'input' controls wait for the next frame request, one per frame, in order; 'set's that arrive back to back
	// are merged (the same rule as engine/protocol.ts mergeControls, so local and worker tapes agree; integration 1b)
	let queued: Control[] = [];
	const waiters: Waiter[] = [];
	const subs = new Set<(n: number) => void>();
	let notifyPending = false;
	const slice = roNow.sliceMs ?? 8;
	// running: advance in the background. Starts as `autorun`; 'pause' and 'resume' switch it.
	let running = roNow.autorun ?? true;

	const notify = () => {
		if (notifyPending || subs.size === 0) return;
		notifyPending = true;
		onFrame(() => {
			notifyPending = false;
			for (const fn of subs) fn(store.length);
		});
	};
	const isDone = () => s.done || closed || error !== null;

	const settle = () => {
		for (let k = waiters.length - 1; k >= 0; k--) {
			const w = waiters[k];
			const needsEnd = w.c === 'last' || (typeof w.c === 'object' && w.c !== null && w.c.last === true);
			let idx = -1;
			if (!needsEnd) {
				idx = store.findFrom(w.c, w.from);
				if (typeof w.c === 'object' && w.c !== null) w.from = store.length;
			} else if (isDone()) idx = store.find(w.c);
			if (idx >= 0) {
				waiters.splice(k, 1);
				w.resolve(idx);
			} else if (isDone()) {
				waiters.splice(k, 1);
				w.resolve(store.length - 1);
			}
		}
	};

	const pump = (budget: number) => {
		const t0 = now();
		try {
			while (!isDone()) {
				const want = running || waiters.length > 0 || stepBudget > 0;
				if (!want) break;
				const f = s.next(queued.shift());
				if (!f) break;
				store.push(f);
				if (stepBudget > 0) stepBudget--;
				if (!running) {
					// on demand only: stop as soon as every waiter is satisfied and no steps are owed
					settle();
					if (waiters.length === 0 && stepBudget === 0) break;
				}
				if (now() - t0 >= budget) break;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		settle();
		notify();
	};

	const schedule = () => {
		if (scheduled || isDone()) return;
		const want = running || waiters.length > 0 || stepBudget > 0;
		if (!want) return;
		scheduled = true;
		later(() => {
			scheduled = false;
			pump(slice);
			schedule();
		});
	};

	const tape: Recorder<S> = {
		get store() {
			return store;
		},
		get phases() {
			return store.phases;
		},
		get process() {
			return store.process;
		},
		get seed() {
			return store.seed;
		},
		get opt() {
			return store.opt;
		},
		get provenance() {
			return store.provenance;
		},
		get version() {
			return store.version;
		},
		get length() {
			return store.length;
		},
		get done() {
			return isDone();
		},
		get paused() {
			return !running;
		},
		get truncated() {
			return s.truncated;
		},
		get error() {
			return error;
		},
		frame: (i) => store.frame(i),
		state: (i) => store.state(i),
		find: (c) => store.find(c),
		segments: () => store.segments(),
		metric: (n) => store.metric(n),
		ensure(c: Cursor, o?: { budgetMs?: number }): Promise<number> {
			return new Promise<number>((resolve) => {
				waiters.push({ c, from: 0, resolve });
				settle();
				if (waiters.length && !isDone()) {
					if (o?.budgetMs !== undefined) pump(o.budgetMs);
					schedule();
				}
			});
		},
		control(c: Control) {
			if (closed) return;
			switch (c.type) {
				case 'pause':
					running = false;
					return;
				case 'resume':
					running = true;
					return schedule();
				case 'step':
					stepBudget += Math.max(0, Math.floor(c.n));
					return schedule();
				case 'reset': {
					s.cancel();
					roNow = { ...roNow, seed: c.seed ?? roNow.seed };
					s = stepper(p, inputNow, optNow, roNow);
					store = storeFor(p, s, roNow);
					error = null;
					stepBudget = 0;
					queued = [];
					notify();
					return schedule();
				}
				default:
					if (c.type === 'input') inputNow = c.input as I;
					else if (c.type === 'set') optNow = { ...(optNow as object), ...(c.opt as object) } as Partial<O>;
					// 'set' and 'input' reach the generator with the next frame requests (one per frame, sets merged)
					if (c.type === 'set' && queued[queued.length - 1]?.type === 'set') {
						const last = queued[queued.length - 1] as Extract<Control, { type: 'set' }>;
						queued[queued.length - 1] = { type: 'set', opt: { ...last.opt, ...c.opt } };
					} else queued.push(c);
					return schedule();
			}
		},
		subscribe(fn: (n: number) => void) {
			subs.add(fn);
			return () => subs.delete(fn);
		},
		close() {
			closed = true;
			s.cancel();
			settle();
			subs.clear();
		}
	};
	schedule();
	return tape;
}
