// atlas-tape/1 files: <stem>.tape.json (TapeHeader + keyEvery + bin) and <stem>.tape.bin. Design §4.2/§4.4.
//
// The layout is shared with the Python writer (pipeline/atlas_kit/tape.py, ak.TapeWriter) and the CLI's writer
// (scripts/atlas/lib/tapefile.ts); all three must produce the same bytes for the same stored frames:
//  - header: { format: 'atlas-tape/1', process, version, seed, opt, provenance, schema (SchemaJson), phases
//    ({ label, kind, tex? }), keyEvery, bin: { file, bytes }, frames }. JSON, UTF-8 without BOM, LF.
//  - frames[i]: { phase, step?, sub?, key?: true, state, changes?, focus?, metrics?, caption?, draws? }.
//    key: true on keyframes (i % keyEvery == 0); a keyframe's state holds the full value of every field with
//    track: 'changes'. Other frames hold a tracked field only when the process replaced it as a whole.
//    Snapshot fields are stored where they were present. Derived fields are never stored.
//  - values: a tensor { shape, data } or a bare typed array -> { "$t": "f32" | "f64" | "i32", "shape", "o", "n" }
//    (byte offset o into the blob, element count n; other integer arrays are stored as i32); non-finite numbers ->
//    { "$f": "NaN" | "Infinity" | "-Infinity" } (−0 is stored as 0); everything else is plain JSON. `opt` is plain JSON.
//  - blob: little-endian, each array starting on an 8-byte boundary (zero padding). Order: per frame, the untracked
//    state fields in frame order, then the tracked fields in schema order, then changes, focus, metrics, draws.
//  - reading: a "$t" ref decodes to a tensor { shape, data } ({ shape, data, dtype: 'i32' } for i32).
import type { Tape, TapeHeader, ProcessDef, Schema, SchemaJson, Frame } from '../types/process.ts';
import { validateTapeHeader } from '../schema/tape.ts';
import { TapeStore, type PhaseInfo } from './tape.ts';
import { isTypedArray } from './values.ts';

export interface TapeFileHeader extends TapeHeader {
	keyEvery: number;
	bin: { file: string; bytes: number };
}

const FRAME_KEYS = ['phase', 'step', 'sub', 'key', 'state', 'changes', 'focus', 'metrics', 'caption', 'draws'] as const;
const RAW_KEYS = new Set(['phase', 'step', 'sub', 'key', 'caption']);
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

type Ref = { $t: 'f32' | 'f64' | 'i32'; shape: number[]; o: number; n: number };

class BlobWriter {
	parts: Uint8Array[] = [];
	size = 0;
	add(data: ArrayBufferView, shape: number[], where: string): Ref {
		if (!littleEndian) throw new Error('atlas-tape/1 needs a little-endian platform');
		let $t: Ref['$t'];
		let bytes: Uint8Array;
		if (data instanceof Float32Array || data instanceof Float64Array) {
			$t = data instanceof Float32Array ? 'f32' : 'f64';
			bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
		} else if (data instanceof Int32Array || data instanceof Int16Array || data instanceof Int8Array || data instanceof Uint8Array || data instanceof Uint16Array || data instanceof Uint8ClampedArray) {
			$t = 'i32';
			bytes = new Uint8Array(Int32Array.from(data as ArrayLike<number>).buffer);
		} else throw new Error(`atlas-tape/1 at ${where}: tapes store Float32Array, Float64Array and integer arrays (got ${data.constructor.name})`);
		const pad = (8 - (this.size % 8)) % 8;
		if (pad) {
			this.parts.push(new Uint8Array(pad));
			this.size += pad;
		}
		const ref: Ref = { $t, shape: [...shape], o: this.size, n: (data as unknown as ArrayLike<number>).length };
		this.parts.push(bytes);
		this.size += bytes.byteLength;
		return ref;
	}
	finish(): ArrayBuffer {
		const out = new Uint8Array(this.size);
		let o = 0;
		for (const p of this.parts) {
			out.set(p, o);
			o += p.byteLength;
		}
		return out.buffer;
	}
}

const isTensorLike = (v: unknown): v is { shape: number[]; data: ArrayBufferView } =>
	!!v && typeof v === 'object' && Array.isArray((v as { shape?: unknown }).shape) && isTypedArray((v as { data?: unknown }).data);

export function encodeValue(v: unknown, w: BlobWriter, where = 'value'): unknown {
	if (isTensorLike(v)) return w.add(v.data, v.shape, where);
	if (isTypedArray(v)) return w.add(v, [(v as unknown as ArrayLike<number>).length], where);
	if (typeof v === 'number') {
		if (Number.isNaN(v)) return { $f: 'NaN' };
		if (!Number.isFinite(v)) return { $f: v > 0 ? 'Infinity' : '-Infinity' };
		return v;
	}
	if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
	if (v === undefined || typeof v === 'function') return null;
	if (Array.isArray(v)) return v.map((x, i) => encodeValue(x, w, `${where}[${i}]`));
	const out: Record<string, unknown> = {};
	for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
		if (x === undefined || typeof x === 'function') continue;
		out[k] = encodeValue(x, w, `${where}.${k}`);
	}
	return out;
}

export function decodeValue(v: unknown, blob: ArrayBuffer): unknown {
	if (v === null || typeof v !== 'object') return v;
	if (Array.isArray(v)) return v.map((x) => decodeValue(x, blob));
	const o = v as Record<string, unknown>;
	if (typeof o.$t === 'string') {
		const bytes = o.$t === 'f64' ? 8 : 4;
		const off = o.o as number;
		const end = off + (o.n as number) * bytes;
		if (off < 0 || end > blob.byteLength) throw new Error(`atlas-tape/1: array [${off}, ${end}) is outside the blob (${blob.byteLength} bytes)`);
		const buf = blob.slice(off, end);
		if (o.$t === 'f32') return { shape: o.shape, data: new Float32Array(buf) };
		if (o.$t === 'f64') return { shape: o.shape, data: new Float64Array(buf) };
		if (o.$t === 'i32') return { shape: o.shape, data: new Int32Array(buf), dtype: 'i32' };
		throw new Error(`atlas-tape/1: unknown dtype ${o.$t}`);
	}
	if (typeof o.$f === 'string' && Object.keys(o).length === 1) return o.$f === 'NaN' ? NaN : o.$f === 'Infinity' ? Infinity : -Infinity;
	const out: Record<string, unknown> = {};
	for (const [k, x] of Object.entries(o)) out[k] = decodeValue(x, blob);
	return out;
}

/** Serialize a tape. Keyframes (i % keyEvery == 0) carry the full value of every 'changes' field. */
export function saveTape(t: Tape, o?: { binName?: string; keyEvery?: number }): { header: TapeFileHeader; blob: ArrayBuffer } {
	const w = new BlobWriter();
	const store = t instanceof TapeStore ? t : null;
	const keyEvery = o?.keyEvery ?? store?.keyEvery ?? 64;
	const any = t as { schemaJson?: () => SchemaJson; phasesJson?: () => TapeHeader['phases'] };
	const schema = store ? store.schemaJson() : (any.schemaJson?.() ?? {});
	const phases = store ? store.phasesJson() : (any.phasesJson?.() ?? {});
	const tracked = Object.keys(schema).filter((k) => schema[k].track === 'changes');
	const frames: unknown[] = [];
	for (let i = 0; i < t.length; i++) {
		const f = t.frame(i) as Frame & { key?: boolean };
		const key = i % keyEvery === 0;
		const s = { ...(f.state as Record<string, unknown>) };
		if (key && store) {
			const raw = store.rawState(i);
			for (const k of tracked) if (k in raw) s[k] = raw[k];
		}
		const where = `frame ${i} (${f.phase})`;
		const st: Record<string, unknown> = {};
		for (const k of Object.keys(s)) if (!tracked.includes(k)) st[k] = encodeValue(s[k], w, `${where}.state.${k}`);
		for (const k of tracked) if (k in s) st[k] = encodeValue(s[k], w, `${where}.state.${k}`);
		const out: Record<string, unknown> = {};
		const src = { ...f, key: key ? true : undefined } as Record<string, unknown>;
		for (const k of FRAME_KEYS) {
			if (k === 'state') out.state = st;
			else if (src[k] !== undefined) out[k] = RAW_KEYS.has(k) ? src[k] : encodeValue(src[k], w, `${where}.${k}`);
		}
		frames.push(out);
	}
	const blob = w.finish();
	const header: TapeFileHeader = {
		format: 'atlas-tape/1',
		process: t.process,
		version: t.version,
		seed: t.seed,
		opt: t.opt,
		provenance: t.provenance,
		schema,
		phases,
		keyEvery,
		bin: { file: o?.binName ?? `${t.process.replace(/[^A-Za-z0-9_.-]+/g, '-')}.tape.bin`, bytes: blob.byteLength },
		frames
	};
	return { header, blob };
}

/** Rebuild a tape. `p` re-attaches derive functions and phase formulas; without it (Python-written tapes) derived
 *  fields are unavailable. Throws with the validation problems listed when the header is malformed. */
export function loadTape(h: TapeHeader | TapeFileHeader, blob: ArrayBuffer, p?: ProcessDef<any, any, any>, o?: { keyEvery?: number; provenance?: TapeHeader['provenance'] }): TapeStore {
	const problems = validateTapeHeader(h, blob.byteLength);
	if (problems.length) throw new Error(`atlas-tape/1 (${(h as { process?: string })?.process ?? '?'}): ${problems.slice(0, 8).join('; ')}`);
	const schema: Schema = {};
	for (const [k, d] of Object.entries(h.schema)) {
		const live = p?.schema[k];
		const { derived, ...rest } = d;
		schema[k] = live?.derive ? { ...rest, derive: live.derive } : derived ? ({ ...rest, derived: true } as Schema[string]) : rest;
	}
	const phases: Record<string, PhaseInfo> = {};
	for (const [k, ph] of Object.entries(h.phases)) phases[k] = p?.phases[k] ?? ph;
	const store = new TapeStore({
		process: h.process,
		seed: h.seed,
		opt: h.opt,
		provenance: o?.provenance ?? h.provenance,
		version: h.version,
		schema,
		phases,
		keyEvery: o?.keyEvery ?? (h as TapeFileHeader).keyEvery
	});
	for (const f of h.frames) {
		const fr = decodeValue(f, blob) as Frame & { key?: boolean };
		delete fr.key;
		store.push(fr, { owned: true });
	}
	return store;
}

/** JSON text for the .tape.json file (LF, no BOM). */
export function headerText(h: TapeHeader): string {
	return JSON.stringify(h) + '\n';
}
