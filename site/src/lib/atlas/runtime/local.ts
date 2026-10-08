// localRunner (design §4.7): runs 'train' and 'process' sources in-process with recorder(). Used in Node, in tests, for
// run: 'main' sources, and as the main-thread fallback when a worker fails (W1-B2's engineRunner).
import type { LiveTape } from '../types/process.ts';
// K1: sharedRunner() (below) shares identical runs between the runtimes of Node tools (lint fixtures, claim tests)
import type { OpenSource, PackageModules, SourceRunner } from '../types/runtime.ts';
import { recorder, type RecorderOpts } from '../process/record.ts';

export function makeLocalRunner(extra?: Pick<RecorderOpts, 'autorun' | 'sliceMs' | 'keyEvery' | 'maxFrames' | 'keep'>): SourceRunner {
	return {
		open(o: OpenSource, pkg: PackageModules): LiveTape {
			const p = pkg.processes[o.process];
			if (!p) throw new Error(`${pkg.slug}: no process '${o.process}' (have ${Object.keys(pkg.processes).join(', ') || 'none'})`);
			return recorder(p, o.input, o.opt, { ...(extra ?? {}), seed: o.seed, detail: o.detail, recordDraws: o.recordDraws });
		}
	};
}

/** Records on demand only (autorun off): a lesson step runs a process exactly as far as its cursor needs. */
export const localRunner: SourceRunner = makeLocalRunner({ autorun: false });

// ------------------------------------------------------------------ shared runs (K1; Node tools)
/** A key for one run: package, process, seed, detail, draws, options and input. null when the input or the options hold
 *  something that is not plain data (a model object, a function): such runs are never shared. */
export function runKey(slug: string, o: OpenSource): string | null {
	let body: string;
	try {
		body = stable({ p: o.process, s: o.seed, d: o.detail, r: !!o.recordDraws, o: o.opt, i: o.input });
	} catch {
		return null;
	}
	// two independent 32-bit FNV-1a hashes and the length: the key stays short for large inputs
	let h1 = 0x811c9dc5;
	let h2 = 0x01000193 ^ 0x5bd1e995;
	for (let i = 0; i < body.length; i++) {
		const c = body.charCodeAt(i);
		h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
		h2 = Math.imul(h2 ^ c, 0x5bd1e995) >>> 0;
		h2 ^= h2 >>> 15;
	}
	return `${slug}|${o.process}|${h1.toString(36)}.${h2.toString(36)}.${body.length.toString(36)}`;
}

function stable(v: unknown, depth = 0): string {
	if (depth > 40) throw new Error('too deep');
	if (v === undefined) return 'u';
	if (v === null || typeof v === 'boolean' || typeof v === 'string') return JSON.stringify(v);
	if (typeof v === 'number') return Number.isFinite(v) ? String(v) : `"${String(v)}"`;
	if (typeof v !== 'object') throw new Error(`not plain data: ${typeof v}`);
	if (ArrayBuffer.isView(v)) return `${v.constructor.name}[${Array.from(v as unknown as ArrayLike<number>).join(',')}]`;
	if (Array.isArray(v)) return `[${v.map((x) => stable(x, depth + 1)).join(',')}]`;
	const pr = Object.getPrototypeOf(v);
	if (pr !== Object.prototype && pr !== null) throw new Error('not plain data: a class instance');
	return `{${Object.keys(v)
		.sort()
		.map((k) => `${JSON.stringify(k)}:${stable((v as Record<string, unknown>)[k], depth + 1)}`)
		.join(',')}}`;
}

export interface SharedRunner extends SourceRunner {
	/** runs served from an earlier identical run, and runs recorded */
	readonly stats: { shared: number; recorded: number };
	/** forget every run */
	clear(): void;
}

/** K1: a runner for Node tools (lint, claim tests) that runs every source once per distinct run. Two runtimes (two lint
 *  fixtures, two claim tests) that open the same process with the same input, options, seed and detail read one
 *  recording, so a source that ignores the learner's input, or a fixture equal to the start input, is not run again.
 *  Each runtime gets its own view of the shared recording: closing it leaves the recording alone. A control ('set',
 *  'input', 'reset', 'step', ...) gives that view a private run of its own first (from frame 0), so nothing a test does
 *  reaches another runtime. `cap` limits training runs (kind 'training') to that many steps, as e2e's ?atlas-e2e=1 does. */
export function sharedRunner(o: { max?: number; cap?: number } = {}): SharedRunner {
	const max = o.max ?? 64;
	const runs = new Map<string, LiveTape>();
	const base = makeLocalRunner({ autorun: false });
	const stats = { shared: 0, recorded: 0 };
	return {
		stats,
		clear() {
			for (const t of runs.values()) t.close();
			runs.clear();
		},
		open(os: OpenSource, pkg: PackageModules): LiveTape {
			const p = pkg.processes[os.process];
			if (!p) throw new Error(`${pkg.slug}: no process '${os.process}' (have ${Object.keys(pkg.processes).join(', ') || 'none'})`);
			let oo = os;
			const steps = Number((os.opt as { steps?: unknown })?.steps ?? (p.defaults as { steps?: unknown })?.steps);
			if (o.cap && p.kind === 'training' && Number.isFinite(steps) && steps > o.cap) oo = { ...os, opt: { ...os.opt, steps: o.cap } };
			const key = runKey(pkg.slug, oo);
			if (key === null) {
				stats.recorded++;
				return base.open(oo, pkg);
			}
			let shared = runs.get(key);
			if (shared) {
				stats.shared++;
				runs.delete(key); // most recently used last
				runs.set(key, shared);
			} else {
				stats.recorded++;
				shared = base.open(oo, pkg);
				runs.set(key, shared);
				while (runs.size > max) {
					const [k0, t0] = runs.entries().next().value as [string, LiveTape];
					runs.delete(k0);
					t0.close();
				}
			}
			return viewOf(shared, () => base.open(oo, pkg));
		}
	};
}

/** A runtime's own view of a shared recording (see sharedRunner). */
function viewOf(shared: LiveTape, fresh: () => LiveTape): LiveTape {
	let own: LiveTape | null = null;
	const cur = () => own ?? shared;
	const subs = new Map<(n: number) => void, () => void>();
	const view: LiveTape = {
		get process() {
			return cur().process;
		},
		get seed() {
			return cur().seed;
		},
		get opt() {
			return cur().opt;
		},
		get provenance() {
			return cur().provenance;
		},
		get version() {
			return cur().version;
		},
		get phases() {
			return cur().phases;
		},
		get length() {
			return cur().length;
		},
		get done() {
			return cur().done;
		},
		get paused() {
			return cur().paused;
		},
		get truncated() {
			return cur().truncated;
		},
		get error() {
			return cur().error;
		},
		frame: (i) => cur().frame(i),
		state: (i) => cur().state(i),
		find: (c) => cur().find(c),
		segments: () => cur().segments(),
		metric: (n) => cur().metric(n),
		ensure: (c, x) => cur().ensure(c, x),
		control(c) {
			if (!own) {
				own = fresh();
				for (const [fn, off] of subs) {
					off();
					subs.set(fn, own.subscribe(fn));
				}
			}
			own.control(c);
		},
		subscribe(fn) {
			subs.set(fn, cur().subscribe(fn));
			return () => {
				subs.get(fn)?.();
				subs.delete(fn);
			};
		},
		close() {
			for (const off of subs.values()) off();
			subs.clear();
			own?.close();
		}
	};
	// the recorder's extras (store, ...) for code that reads them
	Object.defineProperty(view, 'store', { get: () => (cur() as { store?: unknown }).store, enumerable: false });
	return view;
}
