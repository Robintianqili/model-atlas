// FALLBACK recorder for `atlas run`, used only while this tree has no W1-C recorder (src/lib/atlas/process/record.ts).
// After integration 1a, `run` uses W1-C's record(); this file then serves only as the reference implementation of the
// tape layout shared with pipeline/atlas_kit/tape.py (see lib/tapefile.ts) and can be deleted by W1-E2.
//
// Semantics follow design §4.1/§4.4: mulberry32 Rng with fork(); frames keep every untracked field they carry
// (copied), fields with track: 'changes' are copied in full at keyframes (index % keyEvery == 0) and otherwise carried
// by frame.changes (an edit the changes do not explain fails); recordDraws stores the value of every top-level
// rng call (next, int, normal, on the rng and its forks) made while producing a frame.
import type { Change, Control, Cursor, Frame, ProcessDef, RecordOpts, Rng, Schema, Tape } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';

// ------------------------------------------------------------------------------------------ rng (same as core/rng.ts)
export function fnv1a32(label: string): number {
	let h = 0x811c9dc5;
	for (const b of new TextEncoder().encode(label)) {
		h ^= b;
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h >>> 0;
}

export function createRng(seed: number, onDraw?: (v: number) => void): Rng {
	const s0 = seed >>> 0;
	let a = s0;
	let depth = 0;
	const raw = () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
	const top = <T extends number>(v: T): T => {
		if (depth === 0) onDraw?.(v);
		return v;
	};
	const rng: Rng = {
		seed: s0,
		next: () => top(raw()),
		int: (n) => {
			depth++;
			const v = Math.floor(raw() * n);
			depth--;
			return top(v);
		},
		normal: () => {
			depth++;
			const u1 = 1 - raw();
			const u2 = raw();
			depth--;
			return top(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
		},
		fork: (label) => createRng((s0 ^ fnv1a32(label)) >>> 0, onDraw),
		getState: () => a >>> 0,
		setState: (s) => {
			a = s | 0;
		}
	};
	return rng;
}

// ------------------------------------------------------------------------------------------ values
type TypedArray = Float32Array | Float64Array | Int32Array | Uint8Array | Int16Array | Uint16Array | Uint32Array | Int8Array;
const isTyped = (v: unknown): v is TypedArray => ArrayBuffer.isView(v) && !(v instanceof DataView);
export const isTensorLike = (v: unknown): v is { shape: number[]; data: TypedArray; dtype?: string } =>
	!!v && typeof v === 'object' && Array.isArray((v as { shape?: unknown }).shape) && isTyped((v as { data?: unknown }).data);

export function clone<T>(v: T): T {
	return structuredClone(v);
}

export function deepEqual(a: unknown, b: unknown): boolean {
	if (Object.is(a, b)) return true;
	if (typeof a === 'number' && typeof b === 'number') return a === b || (Number.isNaN(a) && Number.isNaN(b));
	if (isTyped(a) || isTyped(b)) {
		if (!isTyped(a) || !isTyped(b) || a.length !== b.length || a.constructor !== b.constructor) return false;
		for (let i = 0; i < a.length; i++) if (!(a[i] === b[i] || (Number.isNaN(a[i]) && Number.isNaN(b[i])))) return false;
		return true;
	}
	if (Array.isArray(a) || Array.isArray(b)) {
		if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
		return a.every((x, i) => deepEqual(x, b[i]));
	}
	if (a && b && typeof a === 'object' && typeof b === 'object') {
		const ka = Object.keys(a as object);
		const kb = Object.keys(b as object);
		if (ka.length !== kb.length) return false;
		return ka.every((k) => k in (b as object) && deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
	}
	return false;
}

function flatIndex(shape: number[], index: number[], where: string): number {
	if (index.length !== shape.length) throw new AtlasError('AK301', `${where}: index [${index}] does not match shape [${shape}]`);
	let k = 0;
	for (let d = 0; d < shape.length; d++) {
		if (index[d] < 0 || index[d] >= shape[d]) throw new AtlasError('AK301', `${where}: index [${index}] is outside shape [${shape}]`);
		k = k * shape[d] + index[d];
	}
	return k;
}

/** Apply one change to a tracked value in place (tensor, record of tensors, nested array or scalar); returns the value. */
export function applyChange(cur: unknown, c: Change, where: string): unknown {
	if (c.to === undefined) throw new AtlasError('AK301', `${where}: a change to '${c.field}' needs 'to'`);
	let target = cur;
	if (c.key !== undefined) {
		if (!target || typeof target !== 'object') throw new AtlasError('AK301', `${where}: change with key '${c.key}' on a field that is not a record`);
		if (!c.index) {
			(target as Record<string, unknown>)[c.key] = c.to;
			return cur;
		}
		target = (target as Record<string, unknown>)[c.key];
	}
	if (isTensorLike(target)) {
		if (!c.index) throw new AtlasError('AK301', `${where}: a change to tensor field '${c.field}' needs an index`);
		target.data[flatIndex(target.shape, c.index, where)] = c.to;
		return cur;
	}
	if (Array.isArray(target) && c.index) {
		let t: unknown[] = target;
		for (let d = 0; d < c.index.length - 1; d++) t = t[c.index[d]] as unknown[];
		t[c.index[c.index.length - 1]] = c.to;
		return cur;
	}
	if (c.key !== undefined) throw new AtlasError('AK301', `${where}: cannot apply a change to '${c.field}'`);
	return c.to;
}

// ------------------------------------------------------------------------------------------ store
export interface StoredFrame extends Frame {
	key?: boolean;
}

export class MiniTape implements Tape {
	process: string;
	seed: number;
	opt: unknown;
	provenance: 'live' | 'precomputed' | 'illustrative' = 'live';
	version = '1';
	readonly frames: StoredFrame[] = [];
	readonly schema: Schema;
	readonly keyEvery: number;
	readonly notes: string[] = [];
	private tracked: string[];
	private current: Record<string, unknown> = {};

	constructor(p: ProcessDef<any, any, any>, seed: number, opt: unknown, keyEvery = 64) {
		this.process = p.id;
		this.seed = seed;
		this.opt = opt;
		this.schema = p.schema;
		this.keyEvery = keyEvery;
		this.tracked = Object.keys(p.schema).filter((k) => p.schema[k].track === 'changes');
	}

	get length(): number {
		return this.frames.length;
	}

	push(f: Frame): void {
		const i = this.frames.length;
		const where = `frame ${i} (${f.phase})`;
		const changes = f.changes ?? [];
		for (const c of changes)
			if (!this.tracked.includes(c.field)) throw new AtlasError('AK301', `${where}: change to '${c.field}', which is not a track: 'changes' field`);
		const store = new Set<string>();
		for (const k of this.tracked) {
			const prev = this.current[k];
			const ch = changes.filter((c) => c.field === k);
			let next: unknown;
			if (prev !== undefined) {
				next = clone(prev);
				for (const c of ch) next = applyChange(next, c, where);
			} else if (ch.length) {
				throw new AtlasError('AK301', `${where}: change to '${k}' before its first full value`, { fix: "yield the field's full value in state on the first frame" });
			}
			if (k in f.state) {
				const given = clone((f.state as Record<string, unknown>)[k]);
				if (prev !== undefined && ch.length && !deepEqual(next, given))
					throw new AtlasError('AK301', `${where}: the changes to '${k}' do not explain its new value`, {
						likely: 'the process mutated the field without reporting every edit in frame.changes'
					});
				if (prev === undefined || (!ch.length && !deepEqual(prev, given))) store.add(k);
				this.current[k] = given;
			} else if (next !== undefined) {
				this.current[k] = next;
			}
		}
		const key = i % this.keyEvery === 0;
		const state: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(f.state)) {
			if (this.tracked.includes(k)) continue;
			if (!(k in this.schema)) throw new AtlasError('AK301', `${where}: state field '${k}' is not in the process schema`);
			if (this.schema[k].derive) {
				if (!this.notes.includes(k)) this.notes.push(`derived field '${k}' was yielded in a frame; derived fields are never stored`);
				continue;
			}
			state[k] = clone(v);
		}
		for (const k of this.tracked) {
			if (key || store.has(k)) {
				if (this.current[k] === undefined) throw new AtlasError('AK301', `${where}: keyframe needs the full value of tracked field '${k}'`);
				state[k] = clone(this.current[k]);
			}
		}
		const out = { phase: f.phase } as StoredFrame; // keys in FrameJson order: phase, step, sub, key, state, ...
		if (f.step !== undefined) out.step = f.step;
		if (f.sub !== undefined) out.sub = f.sub;
		if (key) out.key = true;
		out.state = state;
		if (changes.length) out.changes = clone(changes);
		if (f.focus?.length) out.focus = clone(f.focus);
		if (f.metrics && Object.keys(f.metrics).length) out.metrics = { ...f.metrics };
		if (f.caption) out.caption = f.caption;
		if (f.draws) out.draws = [...f.draws];
		this.frames.push(out);
	}

	frame(i: number): StoredFrame {
		return this.frames[Math.max(0, Math.min(this.frames.length - 1, i))];
	}

	state(i: number): any {
		i = Math.max(0, Math.min(this.frames.length - 1, i));
		let k = i;
		while (k > 0 && !this.frames[k].key) k--;
		const s: Record<string, unknown> = {};
		for (const t of this.tracked) if (t in this.frames[k].state) s[t] = clone((this.frames[k].state as Record<string, unknown>)[t]);
		for (let j = k + 1; j <= i; j++) {
			const fr = this.frames[j];
			for (const t of this.tracked) {
				if (t in fr.state) s[t] = clone((fr.state as Record<string, unknown>)[t]);
				else for (const c of fr.changes ?? []) if (c.field === t) s[t] = applyChange(s[t], c, `frame ${j}`);
			}
		}
		for (const name of Object.keys(this.schema)) {
			if (this.tracked.includes(name) || this.schema[name].derive) continue;
			for (let j = i; j >= 0; j--)
				if (name in this.frames[j].state) {
					s[name] = (this.frames[j].state as Record<string, unknown>)[name];
					break;
				}
		}
		return s;
	}

	find(c: Cursor): number {
		const n = this.frames.length;
		if (c === 'first') return n ? 0 : -1;
		if (c === 'last') return n - 1;
		if (typeof c === 'number') return c < n ? c : -1;
		const hits: number[] = [];
		for (let i = 0; i < n; i++) {
			const f = this.frames[i];
			if (c.phase !== undefined && f.phase !== c.phase) continue;
			if (c.step !== undefined && f.step !== c.step) continue;
			if (c.sub !== undefined && f.sub !== c.sub) continue;
			if (c.where && !c.where(f, this.state(i))) continue;
			if (!c.last) return i;
			hits.push(i);
		}
		return hits.length ? hits[hits.length - 1] : -1;
	}

	segments(): { phase: string; from: number; to: number }[] {
		const out: { phase: string; from: number; to: number }[] = [];
		this.frames.forEach((f, i) => {
			const last = out[out.length - 1];
			if (last && last.phase === f.phase) last.to = i;
			else out.push({ phase: f.phase, from: i, to: i });
		});
		return out;
	}

	metric(name: string): Float64Array {
		return Float64Array.from(this.frames, (f) => f.metrics?.[name] ?? NaN);
	}
}

/** record(p, input, opt, ro) with the fallback store (all frames, synchronous). */
export function recordFallback(p: ProcessDef<any, any, any>, input: unknown, opt: Record<string, unknown>, ro: RecordOpts = {}): MiniTape {
	const seed = ro.seed ?? 1;
	const tape = new MiniTape(p, seed, opt, ro.keyEvery ?? 64);
	let draws: number[] | null = ro.recordDraws ? [] : null;
	const rng = createRng(seed, draws ? (v) => draws!.push(v) : undefined);
	const ctx = { rng, opt: { ...opt }, detail: ro.detail ?? 'coarse', signal: { cancelled: false } };
	const gen = p.run(input, ctx);
	const max = ro.maxFrames ?? 20000;
	let r = gen.next(undefined as Control | undefined);
	while (!r.done) {
		const f = r.value as Frame;
		if (draws) {
			f.draws = draws;
			draws = [];
		}
		tape.push(f);
		if (tape.length >= max) {
			tape.notes.push(`stopped at maxFrames = ${max}`);
			ctx.signal.cancelled = true;
			gen.return(undefined);
			break;
		}
		r = gen.next(undefined);
	}
	return tape;
}
