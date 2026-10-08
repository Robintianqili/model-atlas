// TapeStore: the one Tape implementation behind every recorded source (local recorder, W1-B2's engine client,
// loadTape). Design §2.1/§4.4: keyframes + changes, derived fields recomputed on demand, state(i) pure in i.
//
// Storage rules (per schema field; a field missing from the schema is treated as 'snapshot'):
//  - track 'snapshot' (default): a value present in frame.state is copied (unless push(..., { owned: true })) and is the
//    field's value from that frame on.
//  - track 'changes': a value present in frame.state is an ANCHOR: it is copied and replaces the running value. Then every
//    frame.changes entry with a `to` value is applied to the running value (set, so re-applying is harmless).
//    A value that the running value plus this frame's changes already explain is not stored again (integration 1a; the
//    atlas-tape/1 writer rule), so yielding the live table in every frame costs nothing between keyframes.
//    The local recorder anchors every 'changes' field at each keyframe from the process's live object, so a mutation the
//    process forgot to report shows up as a difference between keyEvery = 1 and keyEvery = 64 (process/determinism.ts).
//  - derived fields (FieldDef.derive) are never stored: a value yielded for one is dropped. state(i) exposes them as
//    lazy, memoized getters.
// state(i) returns a cached object: treat it as read-only.
import type { I18n, Provenance } from '../types/core.ts';
import type { Cursor, CursorQuery, FieldDef, Frame, PhaseDef, Schema, SchemaJson, Tape, TapeHeader } from '../types/process.ts';
import { applyChange, clone, Hasher, sameValue } from './values.ts';

export type PhaseInfo = PhaseDef | { label: I18n; kind: PhaseDef['kind']; tex?: string };

export interface TapeStoreInit {
	process: string;
	seed?: number;
	opt?: unknown;
	provenance?: Provenance;
	version?: string;
	schema?: Schema | SchemaJson;
	phases?: Record<string, PhaseInfo>;
	/** a keyframe every N frames (default 64) */
	keyEvery?: number;
	/** how many reconstructed states to keep (default 32) */
	cacheSize?: number;
}

type Raw = Record<string, unknown>;

export class TapeStore<S = any> implements Tape<S> {
	readonly process: string;
	readonly seed: number;
	readonly opt: unknown;
	readonly provenance: Provenance;
	readonly version: string;
	readonly keyEvery: number;
	readonly schema: Schema;
	readonly phases: Record<string, PhaseInfo>;
	/** anchors and changes the store could not apply (a change aimed at a missing target) */
	readonly problems: string[] = [];

	private readonly frames: Frame<S>[] = [];
	private readonly keys: Raw[] = [];
	private run: Raw = {};
	private readonly changesFields = new Set<string>();
	private readonly derived = new Map<string, (s: any, opt?: any) => unknown>();
	private readonly cache = new Map<number, { raw: Raw; view: S }>();
	private readonly cacheSize: number;
	private readonly listeners = new Set<(length: number) => void>();
	private segCache: { length: number; segs: { phase: string; from: number; to: number }[] } | null = null;

	constructor(init: TapeStoreInit) {
		this.process = init.process;
		this.seed = init.seed ?? 0;
		this.opt = init.opt ?? {};
		this.provenance = init.provenance ?? 'live';
		this.version = init.version ?? '1';
		this.keyEvery = Math.max(1, Math.floor(init.keyEvery ?? 64));
		this.cacheSize = Math.max(2, init.cacheSize ?? 32);
		this.schema = (init.schema ?? {}) as Schema;
		this.phases = init.phases ?? {};
		for (const [name, def] of Object.entries(this.schema)) {
			if (def.track === 'changes') this.changesFields.add(name);
			if (typeof (def as FieldDef).derive === 'function') this.derived.set(name, (def as FieldDef).derive!);
		}
	}

	get length(): number {
		return this.frames.length;
	}

	/** Field names tracked by changes (anchored at keyframes by the recorder). */
	get changeTracked(): ReadonlySet<string> {
		return this.changesFields;
	}

	isDerived(field: string): boolean {
		return this.derived.has(field) || (this.schema[field] as { derived?: boolean } | undefined)?.derived === true;
	}

	/** Append a frame. Returns its index. `owned`: the caller hands over fresh copies (e.g. frames decoded from a worker
	 *  message or a tape file), so nothing is copied again. */
	push(frame: Frame<S>, o?: { owned?: boolean }): number {
		if (!frame || typeof frame.phase !== 'string') throw new Error(`${this.process}: a frame needs a string phase`);
		const owned = o?.owned === true;
		const i = this.frames.length;
		const state: Raw = {};
		for (const [k, v] of Object.entries((frame.state ?? {}) as Raw)) {
			if (v === undefined || this.isDerived(k)) continue;
			if (this.changesFields.has(k) && k in this.run && this.explained(k, v, frame)) continue;
			const copy = owned ? v : clone(v);
			state[k] = copy;
			this.run[k] = this.changesFields.has(k) ? clone(copy) : copy;
		}
		const stored: Frame<S> = { phase: frame.phase, state: state as Partial<S> };
		if (frame.step !== undefined) stored.step = frame.step;
		if (frame.sub !== undefined) stored.sub = frame.sub;
		if (frame.changes?.length) stored.changes = owned ? frame.changes : clone(frame.changes);
		if (frame.focus?.length) stored.focus = owned ? frame.focus : clone(frame.focus);
		if (frame.metrics) stored.metrics = owned ? frame.metrics : { ...frame.metrics };
		if (frame.caption) stored.caption = frame.caption;
		if (frame.draws?.length) stored.draws = owned ? frame.draws : frame.draws.slice();
		this.applyChanges(this.run, stored, i);
		this.frames.push(stored);
		if (i % this.keyEvery === 0) this.keys.push(this.copyRaw(this.run));
		this.segCache = null;
		for (const fn of this.listeners) fn(this.frames.length);
		return i;
	}

	/** A 'changes' field given in full is not stored when the running value plus this frame's changes already equal it
	 *  (the same rule as the atlas-tape/1 writers in scripts/atlas/lib/tapefile.ts and atlas_kit.TapeWriter). A process
	 *  that yields its live table in every frame then stores it only at keyframes, not once per frame. */
	private explained(k: string, given: unknown, frame: Frame<S>): boolean {
		const ch = frame.changes?.filter((c) => c.field === k) ?? [];
		if (!ch.length) return sameValue(this.run[k], given);
		const next: Raw = { [k]: clone(this.run[k]) };
		for (const c of ch) if (c.to === undefined || !applyChange(next, c)) return false;
		return sameValue(next[k], given);
	}

	/** Called synchronously after every push (the LiveTape wrappers throttle their own subscribers). */
	onPush(fn: (length: number) => void): () => void {
		this.listeners.add(fn);
		return () => this.listeners.delete(fn);
	}

	frame(i: number): Frame<S> {
		this.check(i);
		return this.frames[i];
	}

	state(i: number): S {
		this.check(i);
		const hit = this.cache.get(i);
		if (hit) {
			this.cache.delete(i);
			this.cache.set(i, hit);
			return hit.view;
		}
		const k = Math.floor(i / this.keyEvery);
		const from = k * this.keyEvery;
		let raw: Raw | null = null;
		let start = from;
		for (let j = i - 1; j > from; j--) {
			const c = this.cache.get(j);
			if (c) {
				raw = this.copyRaw(c.raw);
				start = j;
				break;
			}
		}
		if (!raw) raw = this.copyRaw(this.keys[k]);
		for (let j = start + 1; j <= i; j++) this.applyStored(raw, this.frames[j], j);
		const view = this.makeView(raw);
		this.cache.set(i, { raw, view });
		while (this.cache.size > this.cacheSize) this.cache.delete(this.cache.keys().next().value as number);
		return view;
	}

	/** One derived field at frame i (memoized with the state). */
	derive(i: number, field: string): unknown {
		return (this.state(i) as Raw)[field];
	}

	find(c: Cursor): number {
		return this.findFrom(c, 0);
	}

	/** find() starting at frame `from` (forward queries only; 'last' and last: true always search the whole tape). */
	findFrom(c: Cursor, from: number): number {
		const n = this.frames.length;
		if (c === 'first') return n > 0 ? 0 : -1;
		if (c === 'last') return n - 1;
		if (typeof c === 'number') return Number.isInteger(c) && c >= 0 && c < n ? c : -1;
		if (!c || typeof c !== 'object') return -1;
		if (c.last) {
			for (let i = n - 1; i >= 0; i--) if (this.matches(c, i)) return i;
			return -1;
		}
		for (let i = Math.max(0, from); i < n; i++) if (this.matches(c, i)) return i;
		return -1;
	}

	matches(c: CursorQuery, i: number): boolean {
		const f = this.frames[i];
		if (c.phase !== undefined && f.phase !== c.phase) return false;
		if (c.step !== undefined && f.step !== c.step) return false;
		if (c.sub !== undefined && f.sub !== c.sub) return false;
		if (c.layer !== undefined && (f.state as Raw).layer !== c.layer) return false;
		if (c.where) return c.where(f, c.where.length >= 2 ? this.state(i) : undefined);
		return true;
	}

	segments(): { phase: string; from: number; to: number }[] {
		if (this.segCache && this.segCache.length === this.frames.length) return this.segCache.segs;
		const segs: { phase: string; from: number; to: number }[] = [];
		this.frames.forEach((f, i) => {
			const last = segs[segs.length - 1];
			if (last && last.phase === f.phase) last.to = i;
			else segs.push({ phase: f.phase, from: i, to: i });
		});
		this.segCache = { length: this.frames.length, segs };
		return segs;
	}

	metric(name: string): Float64Array {
		const out = new Float64Array(this.frames.length);
		this.frames.forEach((f, i) => {
			const v = f.metrics?.[name];
			out[i] = typeof v === 'number' ? v : NaN;
		});
		return out;
	}

	/** Schema without functions (tape files, worker messages). */
	schemaJson(): SchemaJson {
		const out: SchemaJson = {};
		for (const [k, d] of Object.entries(this.schema)) {
			const { derive, ...rest } = d as FieldDef;
			out[k] = { ...rest, ...(derive || (d as { derived?: boolean }).derived ? { derived: true } : {}) };
		}
		return out;
	}

	phasesJson(): TapeHeader['phases'] {
		const out: TapeHeader['phases'] = {};
		for (const [k, p] of Object.entries(this.phases)) {
			const tex = 'formula' in p && p.formula ? p.formula.tex : (p as { tex?: string }).tex;
			out[k] = { label: p.label, kind: p.kind, ...(tex ? { tex } : {}) };
		}
		return out;
	}

	/** Hash of every stored frame (what was recorded). Equal for two runs of a deterministic process. */
	frameHash(): string {
		const h = new Hasher();
		h.value(this.frames.length);
		for (const f of this.frames) h.value(f);
		return h.hex();
	}

	/** Hash of state(i) for every i (stored fields only, derived fields excluded). Independent of keyEvery. */
	stateHash(): string {
		const h = new Hasher();
		h.value(this.frames.length);
		for (let i = 0; i < this.frames.length; i++) h.value(this.rawState(i));
		return h.hex();
	}

	/** state(i) without derived getters (a fresh object; stored fields only). */
	rawState(i: number): Raw {
		this.state(i);
		return { ...this.cache.get(i)!.raw };
	}

	// ---------------------------------------------------------------- internals
	private check(i: number): void {
		if (!Number.isInteger(i) || i < 0 || i >= this.frames.length) throw new RangeError(`${this.process}: frame ${i} is not recorded (length ${this.frames.length})`);
	}

	private copyRaw(r: Raw): Raw {
		const out: Raw = { ...r };
		for (const k of this.changesFields) if (k in out) out[k] = clone(out[k]);
		return out;
	}

	private applyStored(raw: Raw, f: Frame<S>, i: number): void {
		for (const [k, v] of Object.entries(f.state as Raw)) raw[k] = this.changesFields.has(k) ? clone(v) : v;
		this.applyChanges(raw, f, i, true);
	}

	private applyChanges(raw: Raw, f: Frame<S>, i: number, quiet = false): void {
		if (!f.changes) return;
		for (const c of f.changes) {
			if (!this.changesFields.has(c.field)) continue; // a note about a snapshot field: the new value is in state
			if (!applyChange(raw, c) && !quiet) this.problems.push(`frame ${i}: change ${JSON.stringify({ field: c.field, key: c.key, index: c.index })} has no target`);
		}
	}

	private makeView(raw: Raw): S {
		if (this.derived.size === 0) return raw as S;
		const view: Raw = { ...raw };
		for (const [name, fn] of this.derived) {
			let done = false;
			let memo: unknown;
			Object.defineProperty(view, name, {
				enumerable: false,
				configurable: true,
				get: () => {
					if (!done) {
						memo = fn(view, this.opt);
						done = true;
					}
					return memo;
				}
			});
		}
		return view as S;
	}
}
