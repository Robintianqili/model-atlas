// engineRunner (design §4.7, §4.10): a SourceRunner that runs 'train' and 'process' sources in the package's own worker
// (pkg.worker()) and streams their frames into a TapeStore on the main thread. The returned LiveTape behaves like
// localRunner's recorder: on demand by default (ensure() runs the source as far as a cursor needs; 'resume' runs it
// freely), the same controls, the same frames (the worker uses process/record.ts stepper(), so the stores hash equal).
//
// Main-thread fallback: when the package has no worker, or it cannot be created, fails to start, reports an error,
// stays silent for readyTimeoutMs, or does not know the process, the tape switches to localRunner's recorder (8 ms
// slices on the main thread) and replays the controls it was given. A worker that dies mid-run restarts the run on the
// main thread with a notice (a fresh tape: frames already shown are not spliced).
// Hidden tab: a running source pauses while document.hidden and resumes when the tab is visible again (with a notice).
// Cursors (and CursorQuery.where) are evaluated here, on the main thread; they never cross to the worker.
import type { I18n } from '../types/core.ts';
import type { Control, Cursor, Frame, LiveTape, ProcessDef } from '../types/process.ts';
import type { OpenSource, PackageModules, SourceRunner } from '../types/runtime.ts';
import { TapeStore } from '../process/tape.ts';
import { clone } from '../process/values.ts';
import type { Recorder } from '../process/record.ts';
import { makeLocalRunner } from '../runtime/local.ts';
import { EngineClient, type EngineMode, type StreamHandle, type WorkerLike } from './client.ts';
import { ENGINE_ERRORS, type OpenArgs } from './protocol.ts';

export const NOTICES = {
	hidden: { en: 'Paused while this tab is hidden.', zh: '标签页隐藏时已暂停。' },
	fallback: { en: 'Running on the main thread (the worker is not available).', zh: '在主线程上运行（后台线程不可用）。' },
	restarted: { en: 'The worker stopped, so the run restarted on the main thread.', zh: '后台线程停止了，所以这次运行在主线程上重新开始。' }
} satisfies Record<string, I18n>;

export interface EngineTapeStats {
	/** 'frames' messages received from the worker */
	messages: number;
	/** frames received from the worker */
	frames: number;
	/** performance.now() of the last frames message */
	lastAt: number;
	/** main-thread time spent storing worker frames: the longest message and the total (ms) */
	maxHandleMs: number;
	totalHandleMs: number;
}

/** The LiveTape engineRunner returns (Recorder-compatible extras for the UI and TrainSession). */
export interface EngineTape<S = any> extends LiveTape<S> {
	readonly store: TapeStore<S>;
	/** the learner's intent: true unless the source was resumed (or opened with autorun) */
	readonly paused: boolean;
	readonly where: EngineMode;
	readonly error: string | null;
	readonly notice: I18n | null;
	readonly truncated: boolean;
	readonly stats: EngineTapeStats;
}

export interface EngineRunnerOpts {
	/** start running when opened (default false: on demand, as localRunner) */
	autorun?: boolean;
	/** frame batches per second from the worker (default 30) */
	maxFps?: number;
	keyEvery?: number;
	maxFrames?: number;
	keep?: 'all' | { every: number };
	/** 'auto' (default): the package worker when there is one; 'main': always the main-thread recorder */
	mode?: 'auto' | 'main';
	readyTimeoutMs?: number;
	/** a worker factory that wins over pkg.worker (tests, the performance page) */
	worker?: (pkg: PackageModules) => (() => WorkerLike) | null | undefined;
	/** pause running sources while the tab is hidden (default true in browsers) */
	pauseWhenHidden?: boolean;
	/** cap `opt.steps` of training sources (design §6.3: e2e runs use ?atlas-e2e=1, which caps training at 200 steps).
	 *  Default: 200 when the page URL has atlas-e2e=1, else no cap. */
	capSteps?: number;
}

/** The step cap for e2e runs: ro.capSteps, else under ?atlas-e2e=1 (browser) 200, or ?atlas-e2e-cap=n (0: no cap; K0,
 *  atlas e2e --cap), else none. */
export function stepCap(ro: EngineRunnerOpts): number | null {
	if (typeof ro.capSteps === 'number' && ro.capSteps > 0) return Math.floor(ro.capSteps);
	const search = (globalThis as { location?: { search?: string } }).location?.search ?? '';
	if (!/[?&]atlas-e2e=1(&|$)/.test(search)) return null;
	const m = /[?&]atlas-e2e-cap=(\d+)(&|$)/.exec(search);
	if (m) return Number(m[1]) > 0 ? Number(m[1]) : null;
	return 200;
}

type Waiter = { c: Cursor; from: number; resolve: (i: number) => void };

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const onFrame = (fn: () => void) => {
	const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => number }).requestAnimationFrame;
	if (raf) raf(fn);
	else setTimeout(fn, 0);
};

class RemoteTape<S> implements EngineTape<S> {
	private storeNow: TapeStore<S>;
	private local: Recorder<S> | null = null;
	private stream: StreamHandle | null = null;
	private readonly waiters: Waiter[] = [];
	private readonly subs = new Set<(n: number) => void>();
	private notifyPending = false;
	private userRunning: boolean;
	private sentRunning = false;
	/** frames asked for with 'step' (by the learner or by ensure) and not received yet */
	private outstanding = 0;
	private chunk = 16;
	private hidden = false;
	private doneNow = false;
	private closed = false;
	private awaitingSchema = true;
	private heardFrames = false;
	private seedNow: number;
	/** the source options with every 'set' so far: a 'reset' starts a run (and its store header) with them (K0) */
	private optNow: Record<string, unknown>;
	private readonly log: Control[] = [];
	private unsubLocal: (() => void) | null = null;
	private unsubVisibility: (() => void) | null = null;
	errorNow: string | null = null;
	noticeNow: I18n | null = null;
	readonly stats: EngineTapeStats = { messages: 0, frames: 0, lastAt: 0, maxHandleMs: 0, totalHandleMs: 0 };

	private readonly client: EngineClient;
	private readonly p: ProcessDef<any, any, S>;
	private readonly o: OpenSource;
	private readonly pkg: PackageModules;
	private readonly ro: EngineRunnerOpts;

	constructor(client: EngineClient, p: ProcessDef<any, any, S>, o: OpenSource, pkg: PackageModules, ro: EngineRunnerOpts) {
		this.client = client;
		this.p = p;
		this.o = o;
		this.pkg = pkg;
		this.ro = ro;
		this.seedNow = o.seed >>> 0;
		this.optNow = { ...(o.opt ?? {}) };
		this.userRunning = ro.autorun === true;
		this.storeNow = this.makeStore(this.seedNow);
		if (ro.pauseWhenHidden !== false) this.watchVisibility();
		const args: OpenArgs = {
			process: o.process,
			input: o.input,
			opt: o.opt,
			seed: o.seed,
			detail: o.detail,
			recordDraws: o.recordDraws,
			keyEvery: ro.keyEvery,
			maxFrames: ro.maxFrames,
			keep: ro.keep,
			autorun: this.userRunning && !this.hidden
		};
		this.sentRunning = args.autorun === true;
		this.stream = client.stream(args, o.source, {
			onSchema: () => {
				this.awaitingSchema = false;
			},
			onFrames: (frames, done) => this.onFrames(frames, done),
			onError: (code, message) => {
				if (code === ENGINE_ERRORS.unknownProcess) return this.goLocal(message);
				this.errorNow = message;
				this.doneNow = true;
				this.settle();
				this.notify();
			},
			onFallback: (reason) => this.goLocal(reason),
			onSuperseded: () => this.close()
		});
	}

	private makeStore(seed: number): TapeStore<S> {
		const p = this.p as ProcessDef<any, any, S> & { version?: unknown };
		return new TapeStore<S>({
			process: p.id,
			seed,
			opt: clone({ ...(p.defaults as object), ...this.optNow }),
			provenance: 'live',
			version: String(p.version ?? '1'),
			schema: p.schema,
			phases: p.phases,
			keyEvery: this.ro.keyEvery ?? 64
		});
	}

	private onFrames(frames: Frame[], done: boolean) {
		if (this.closed || this.local) return;
		const t0 = now();
		this.stats.messages++;
		this.stats.lastAt = t0;
		if (this.awaitingSchema) return; // frames of a run that was reset; the new run starts with its schema
		for (const f of frames) this.storeNow.push(f as Frame<S>, { owned: true });
		this.stats.frames += frames.length;
		this.outstanding = Math.max(0, this.outstanding - frames.length);
		if (frames.length) this.heardFrames = true;
		if (done) this.doneNow = true;
		this.settle();
		this.notify();
		const ms = now() - t0;
		this.stats.maxHandleMs = Math.max(this.stats.maxHandleMs, ms);
		this.stats.totalHandleMs += ms;
	}

	/** Continue on the main thread with localRunner's recorder. */
	private goLocal(reason: string) {
		if (this.closed || this.local) return;
		this.stream?.cancel();
		this.stream = null;
		const restart = this.heardFrames;
		const runner = makeLocalRunner({ autorun: false, keyEvery: this.ro.keyEvery, maxFrames: this.ro.maxFrames, keep: this.ro.keep });
		const rec = runner.open({ ...this.o, seed: this.seedNow }, this.pkg) as Recorder<S>;
		this.local = rec;
		this.noticeNow = restart ? NOTICES.restarted : null;
		this.errorNow = null;
		this.doneNow = false;
		void reason;
		// replay what the learner did before the switch (sets, inputs, steps); a restart keeps only the settings
		for (const c of this.log) if (!restart || c.type === 'set' || c.type === 'input') rec.control(c);
		if (this.userRunning && !this.hidden) rec.control({ type: 'resume' });
		this.unsubLocal = rec.subscribe(() => this.notify());
		const ws = this.waiters.splice(0);
		for (const w of ws) void rec.ensure(w.c).then(w.resolve);
		this.notify();
	}

	private watchVisibility() {
		const d = (globalThis as { document?: Document }).document;
		if (!d || typeof d.addEventListener !== 'function') return;
		const fn = () => this.setHidden(d.visibilityState === 'hidden');
		d.addEventListener('visibilitychange', fn);
		this.unsubVisibility = () => d.removeEventListener('visibilitychange', fn);
		this.hidden = d.visibilityState === 'hidden';
	}

	/** Pause a running source while the tab is hidden; resume when it is visible again. */
	setHidden(hidden: boolean) {
		if (this.hidden === hidden || this.closed) return;
		this.hidden = hidden;
		if (this.local) {
			if (this.userRunning) this.local.control({ type: hidden ? 'pause' : 'resume' });
		} else this.sync();
		this.noticeNow = hidden && (this.userRunning || this.waiters.length > 0) ? NOTICES.hidden : this.noticeNow === NOTICES.hidden ? null : this.noticeNow;
		this.notify();
	}

	/** Tell the worker what to do. Free running while the learner resumed the source (or a waiter needs the end of the
	 *  run); otherwise on demand: exactly the frames numeric cursors need, or a growing chunk (16, 32, … 1024) while a
	 *  query cursor is unanswered. The local recorder stops exactly at a query's match; a worker source may record up to
	 *  one chunk beyond it (the same frames, a longer tape). */
	private sync() {
		if (!this.stream || this.local) return;
		const needsEnd = this.waiters.some((w) => w.c === 'last' || (typeof w.c === 'object' && w.c !== null && w.c.last === true));
		const free = !this.hidden && !this.doneNow && (this.userRunning || needsEnd);
		if (free !== this.sentRunning) {
			this.sentRunning = free;
			this.stream.control({ type: free ? 'resume' : 'pause' });
		}
		if (free || this.hidden || this.doneNow || this.waiters.length === 0) return;
		const len = this.storeNow.length;
		let exact = 0;
		let query = false;
		for (const w of this.waiters) {
			if (typeof w.c === 'number') exact = Math.max(exact, w.c + 1 - len);
			else if (w.c === 'first') exact = Math.max(exact, 1 - len);
			else query = true;
		}
		let n = exact - this.outstanding;
		if (query && this.outstanding === 0) {
			n = Math.max(n, this.chunk);
			this.chunk = Math.min(1024, this.chunk * 2);
		}
		if (n > 0) {
			this.outstanding += n;
			this.stream.control({ type: 'step', n });
		}
	}

	private settle() {
		const store = this.storeNow;
		const over = this.doneNow || this.closed;
		for (let k = this.waiters.length - 1; k >= 0; k--) {
			const w = this.waiters[k];
			const needsEnd = w.c === 'last' || (typeof w.c === 'object' && w.c !== null && w.c.last === true);
			let idx = -1;
			if (!needsEnd) {
				idx = store.findFrom(w.c, w.from);
				if (typeof w.c === 'object' && w.c !== null) w.from = store.length;
			} else if (over) idx = store.find(w.c);
			if (idx >= 0) {
				this.waiters.splice(k, 1);
				w.resolve(idx);
			} else if (over) {
				this.waiters.splice(k, 1);
				w.resolve(store.length - 1);
			}
		}
		if (this.waiters.length === 0) this.chunk = 16;
		this.sync();
	}

	private notify() {
		if (this.notifyPending || this.subs.size === 0) return;
		this.notifyPending = true;
		onFrame(() => {
			this.notifyPending = false;
			const n = this.length;
			for (const fn of this.subs) fn(n);
		});
	}

	// ---------------------------------------------------------------- LiveTape
	get store(): TapeStore<S> {
		return this.local ? this.local.store : this.storeNow;
	}
	get phases() {
		return this.store.phases;
	}
	get process() {
		return this.store.process;
	}
	get seed() {
		return this.store.seed;
	}
	get opt() {
		return this.store.opt;
	}
	get provenance() {
		return this.store.provenance;
	}
	get version() {
		return this.store.version;
	}
	get length() {
		return this.store.length;
	}
	get done() {
		return this.local ? this.local.done : this.doneNow || this.closed;
	}
	get paused() {
		return !this.userRunning;
	}
	get where(): EngineMode {
		return this.local ? 'main' : this.client.mode;
	}
	get error() {
		return this.local ? this.local.error : this.errorNow;
	}
	get notice() {
		return this.noticeNow ?? (this.local && this.client.mode === 'main' && this.pkg.worker && this.ro.mode !== 'main' ? NOTICES.fallback : null);
	}
	get truncated() {
		if (this.local) return this.local.truncated;
		return this.ro.maxFrames !== undefined && this.doneNow && this.storeNow.length >= this.ro.maxFrames;
	}
	frame(i: number) {
		return this.store.frame(i);
	}
	state(i: number) {
		return this.store.state(i);
	}
	find(c: Cursor) {
		return this.store.find(c);
	}
	segments() {
		return this.store.segments();
	}
	metric(name: string) {
		return this.store.metric(name);
	}
	ensure(c: Cursor, o?: { budgetMs?: number }): Promise<number> {
		if (this.local) return this.local.ensure(c, o);
		return new Promise<number>((resolve) => {
			this.waiters.push({ c, from: 0, resolve });
			this.settle();
		});
	}
	control(c: Control) {
		if (this.closed) return;
		if (c.type === 'pause') this.userRunning = false;
		else if (c.type === 'resume') this.userRunning = true;
		if (this.local) {
			if (c.type === 'resume' && this.hidden) return;
			this.local.control(c);
			if (c.type === 'reset') this.notify();
			return;
		}
		this.log.push(c);
		if (c.type === 'set') this.optNow = { ...this.optNow, ...(c.opt as Record<string, unknown>) };
		switch (c.type) {
			case 'pause':
			case 'resume':
				this.sync();
				return this.notify();
			case 'step':
				this.outstanding += Math.max(0, Math.floor(c.n));
				this.stream?.control(c);
				return;
			case 'reset':
				this.outstanding = 0;
				this.chunk = 16;
				this.seedNow = c.seed !== undefined ? c.seed >>> 0 : this.seedNow;
				this.storeNow = this.makeStore(this.seedNow);
				this.awaitingSchema = true;
				this.doneNow = false;
				this.errorNow = null;
				this.heardFrames = false;
				this.stream?.control(c);
				this.settle();
				return this.notify();
			default:
				this.stream?.control(c);
		}
	}
	subscribe(fn: (n: number) => void) {
		this.subs.add(fn);
		return () => {
			this.subs.delete(fn);
		};
	}
	close() {
		if (this.closed) return;
		this.closed = true;
		this.stream?.cancel();
		this.stream = null;
		this.local?.close();
		this.unsubLocal?.();
		this.unsubVisibility?.();
		this.settle();
		this.subs.clear();
	}
}

export interface EngineRunner extends SourceRunner {
	open(o: OpenSource, pkg: PackageModules): EngineTape;
	/** the client of a package (null until a source of it was opened) */
	client(pkg: PackageModules): EngineClient | null;
	/** stop every worker this runner started */
	dispose(): void;
}

export function makeEngineRunner(ro: EngineRunnerOpts = {}): EngineRunner {
	const clients = new Map<PackageModules, EngineClient>();
	// K1: sources with run: 'main' record on the main thread (a client without a worker), without the fallback notice
	const mainClients = new Map<PackageModules, EngineClient>();
	const mainClientFor = (pkg: PackageModules): EngineClient => {
		let c = mainClients.get(pkg);
		if (!c) {
			c = new EngineClient(null, { readyTimeoutMs: ro.readyTimeoutMs });
			mainClients.set(pkg, c);
		}
		return c;
	};
	const clientFor = (pkg: PackageModules): EngineClient => {
		let c = clients.get(pkg);
		if (!c) {
			const factory = ro.mode === 'main' ? null : (ro.worker?.(pkg) ?? (pkg.worker ? (pkg.worker as () => WorkerLike) : null));
			c = new EngineClient(factory, { readyTimeoutMs: ro.readyTimeoutMs });
			clients.set(pkg, c);
		}
		return c;
	};
	return {
		open(o: OpenSource, pkg: PackageModules): EngineTape {
			const p = pkg.processes[o.process];
			if (!p) throw new Error(`${pkg.slug}: no process '${o.process}' (have ${Object.keys(pkg.processes).join(', ') || 'none'})`);
			const cap = stepCap(ro);
			const steps = Number((o.opt as { steps?: unknown })?.steps ?? (p.defaults as { steps?: unknown })?.steps);
			const opened = cap !== null && p.kind === 'training' && Number.isFinite(steps) && steps > cap ? { ...o, opt: { ...o.opt, steps: cap } } : o;
			if (o.run === 'main') return new RemoteTape(mainClientFor(pkg), p, opened, pkg, { ...ro, mode: 'main' });
			return new RemoteTape(clientFor(pkg), p, opened, pkg, ro);
		},
		client: (pkg) => clients.get(pkg) ?? null,
		dispose() {
			for (const c of clients.values()) c.dispose();
			for (const c of mainClients.values()) c.dispose();
			clients.clear();
			mainClients.clear();
		}
	};
}

/** On demand (as localRunner), in the package worker when there is one. */
export const engineRunner: EngineRunner = makeEngineRunner();
