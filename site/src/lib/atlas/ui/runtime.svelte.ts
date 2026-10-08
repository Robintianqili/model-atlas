// The browser wrapper around the pure runtime (design §4.7). createRuntime() replaces rt.sources / rt.sel / rt.pre with
// new objects and calls onChange(); AtlasStore turns that into Svelte reactivity with one counter, so components read
// `store.r` (the runtime, tracked) and redraw when it changes. Traces and tapes are never copied or proxied (KIT rule 5).
import { createRuntime, type AtlasRuntime } from '../runtime/runtime.ts';
import type { FetchLike } from '../types/core.ts';
import type { PackageModules, SourceRunner } from '../types/runtime.ts';
import { engineRunnerFor } from './packages.ts';

export class AtlasStore {
	readonly rt: AtlasRuntime;
	/** bumped on every runtime change (sources, selection, input) */
	version = $state(0);
	/** bumped on every new input: lesson steps re-resolve their cursors */
	inputGen = $state(0);
	private off: () => void;
	private pending: { raw: unknown } | null = null;
	private raf = 0;

	constructor(rt: AtlasRuntime) {
		this.rt = rt;
		this.off = rt.onChange(() => this.version++);
		// every new input reaches the lesson, also one set by a view or a Capture through rt.setInput() (K0: only
		// store.setInput() bumped inputGen, so a view that edited the input left the lesson on its old frames)
		const own = rt.setInput.bind(rt);
		rt.setInput = (raw: unknown) => {
			this.pending = null;
			own(raw);
			this.inputGen++;
			this.version++;
		};
	}

	/** The runtime, read reactively: anything derived from `store.r` re-runs after a change. */
	get r(): AtlasRuntime {
		void this.version;
		return this.rt;
	}

	setInput(raw: unknown): void {
		this.rt.setInput(raw); // the wrapped runtime method: it bumps inputGen and version
	}

	/** Coalesce calls into one setInput per animation frame (a pad that reports every pointer move). */
	setInputSoon(raw: unknown): void {
		this.pending = { raw };
		if (typeof requestAnimationFrame === 'undefined') {
			this.setInput(raw);
			return;
		}
		if (this.raf) return;
		this.raf = requestAnimationFrame(() => {
			this.raf = 0;
			const p = this.pending;
			if (p) this.setInput(p.raw);
		});
	}

	/** Tell components that the runtime changed outside onChange (stepAt() replaces rt.sel). */
	touch(): void {
		this.version++;
	}

	close(): void {
		if (this.raf && typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(this.raf);
		this.off();
		this.rt.close();
	}
}

export interface OpenOpts {
	/** URL of the static folder ('' or `${base}`) */
	base: string;
	input?: unknown;
	runner?: SourceRunner;
	fetch?: FetchLike;
	/** use precomputed tapes instead of training live (default: on small touch screens) */
	preferPrecomputed?: boolean;
}

/** The static-folder prefix for the runtime: `base` from $app/paths, or '/' at the site root (the runtime joins
 *  paths and drops an empty base, which would make 'models/x' relative to the page URL). */
export function staticBase(base: string): string {
	return base === '' ? '/' : base;
}

function smallTouchScreen(): boolean {
	try {
		return typeof matchMedia === 'function' && matchMedia('(pointer: coarse) and (max-width: 700px)').matches;
	} catch {
		return false;
	}
}

/** Create the runtime for a package (worker runner when the engine and a package worker exist) and wrap it. */
export async function openAtlas(pkg: PackageModules, o: OpenOpts): Promise<AtlasStore> {
	const fetchFn: FetchLike = o.fetch ?? ((url: string) => fetch(url));
	const runner = o.runner ?? (await engineRunnerFor(pkg));
	const rt = await createRuntime(pkg, { fetch: fetchFn, base: staticBase(o.base), input: o.input, runner, preferPrecomputed: o.preferPrecomputed ?? smallTouchScreen() });
	return new AtlasStore(rt);
}
