// atlas-tape/1 files: the layout shared with pipeline/atlas_kit/tape.py (TapeWriter). See that file's docstring.
//   <stem>.tape.json  TapeHeader + keyEvery + bin { file, bytes }; frames as stored (keyframes carry tracked fields)
//   <stem>.tape.bin   tensors, little-endian, each at a multiple of 8 bytes
// Tensors in frames are { "$t": "f32" | "f64" | "i32", shape, o (byte offset), n (count) }; non-finite numbers are
// { "$f": "NaN" | "Infinity" | "-Infinity" }. Bin order per frame: untracked state fields in frame order, tracked
// fields in schema order, then changes, focus, metrics, draws (so the TS and Python writers give identical bytes).
// W1-C's io.ts saveTape/loadTape must read and write this layout (or the two writers change together).
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import type { PhaseDef, Schema, SchemaJson, TapeHeader } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import type { StoredFrame } from './fallback.ts';
import { isTensorLike, type MiniTape } from './fallback.ts';
import { readJson, writeBytes, writeText } from './io.ts';

const FRAME_KEYS = ['phase', 'step', 'sub', 'key', 'state', 'changes', 'focus', 'metrics', 'caption', 'draws'] as const;

class Blob {
	parts: Uint8Array[] = [];
	size = 0;
	add(data: ArrayBufferView, shape: number[], where: string): { $t: string; shape: number[]; o: number; n: number } {
		let code: 'f32' | 'f64' | 'i32';
		let bytes: Uint8Array;
		if (data instanceof Float32Array) {
			code = 'f32';
			bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
		} else if (data instanceof Float64Array) {
			code = 'f64';
			bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
		} else if (data instanceof Int32Array || data instanceof Int16Array || data instanceof Int8Array || data instanceof Uint8Array || data instanceof Uint16Array) {
			code = 'i32';
			const a = Int32Array.from(data as ArrayLike<number>);
			bytes = new Uint8Array(a.buffer);
		} else {
			throw new AtlasError('AK301', `tensor at ${where}: tapes store Float32Array, Float64Array and Int32Array`);
		}
		const pad = (8 - (this.size % 8)) % 8;
		if (pad) {
			this.parts.push(new Uint8Array(pad));
			this.size += pad;
		}
		const ref = { $t: code, shape: [...shape], o: this.size, n: (data as unknown as ArrayLike<number>).length };
		this.parts.push(bytes);
		this.size += bytes.byteLength;
		return ref;
	}
	bytes(): Uint8Array {
		const out = new Uint8Array(this.size);
		let o = 0;
		for (const p of this.parts) {
			out.set(p, o);
			o += p.byteLength;
		}
		return out;
	}
}

function encode(v: unknown, blob: Blob, where: string): unknown {
	if (isTensorLike(v)) return blob.add(v.data, v.shape, where);
	if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return blob.add(v, [(v as unknown as ArrayLike<number>).length], where);
	if (typeof v === 'number') {
		if (Number.isNaN(v)) return { $f: 'NaN' };
		if (!Number.isFinite(v)) return { $f: v > 0 ? 'Infinity' : '-Infinity' };
		return v;
	}
	if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
	if (v === undefined) return null;
	if (Array.isArray(v)) return v.map((x, i) => encode(x, blob, `${where}[${i}]`));
	if (typeof v === 'object') {
		const o: Record<string, unknown> = {};
		for (const [k, x] of Object.entries(v)) if (x !== undefined) o[k] = encode(x, blob, `${where}.${k}`);
		return o;
	}
	throw new AtlasError('AK301', `cannot store a ${typeof v} in a tape (at ${where})`);
}

export function schemaJson(s: Schema): SchemaJson {
	const out: SchemaJson = {};
	for (const [k, f] of Object.entries(s)) {
		const { derive, ...rest } = f;
		out[k] = derive ? { ...rest, derived: true } : rest;
	}
	return out;
}

export function phasesJson(p: Record<string, PhaseDef>): TapeHeader['phases'] {
	const out: TapeHeader['phases'] = {};
	for (const [k, ph] of Object.entries(p)) out[k] = { label: ph.label, kind: ph.kind, ...(ph.formula ? { tex: ph.formula.tex } : {}) };
	return out;
}

export interface TapeFileHeader extends TapeHeader {
	keyEvery: number;
	bin: { file: string; bytes: number };
}

/** Encode stored frames (keyframes already carry the tracked fields) into a header + blob. */
export function encodeTape(t: { process: string; version: string; seed: number; opt: unknown; provenance: TapeHeader['provenance']; frames: StoredFrame[]; schema: Schema; keyEvery: number }, phases: Record<string, PhaseDef>, binName: string): { header: TapeFileHeader; blob: Uint8Array } {
	const blob = new Blob();
	const tracked = Object.keys(t.schema).filter((k) => t.schema[k].track === 'changes');
	const frames = t.frames.map((f, i) => {
		const where = `frame ${i} (${f.phase})`;
		const st: Record<string, unknown> = {};
		const s = f.state as Record<string, unknown>;
		for (const k of Object.keys(s)) if (!tracked.includes(k)) st[k] = encode(s[k], blob, `${where}.state.${k}`);
		for (const k of tracked) if (k in s) st[k] = encode(s[k], blob, `${where}.state.${k}`);
		const out: Record<string, unknown> = {};
		const fr = f as unknown as Record<string, unknown>;
		for (const key of FRAME_KEYS) {
			if (key === 'state') out.state = st;
			else if (key === 'draws' && Array.isArray(fr.draws) && fr.draws.length === 0) continue; // empty draws are omitted, as TapeStore does
			else if (fr[key] !== undefined) out[key] = key === 'phase' || key === 'step' || key === 'sub' || key === 'key' || key === 'caption' ? fr[key] : encode(fr[key], blob, `${where}.${key}`);
		}
		return out;
	});
	const header: TapeFileHeader = {
		format: 'atlas-tape/1',
		process: t.process,
		version: t.version,
		seed: t.seed,
		opt: t.opt,
		provenance: t.provenance,
		schema: schemaJson(t.schema),
		phases: phasesJson(phases),
		keyEvery: t.keyEvery,
		bin: { file: binName, bytes: blob.size },
		frames
	};
	return { header, blob: blob.bytes() };
}

/** Write <stem>.tape.json and <stem>.tape.bin for a fallback tape. */
export function writeTapeFiles(stem: string, t: MiniTape, phases: Record<string, PhaseDef>, provenance: TapeHeader['provenance']): { json: string; bin: string; bytes: number } {
	const bin = `${stem}.tape.bin`;
	const json = `${stem}.tape.json`;
	const { header, blob } = encodeTape({ ...t, frames: t.frames, provenance, schema: t.schema, keyEvery: t.keyEvery, process: t.process, version: t.version, seed: t.seed, opt: t.opt }, phases, basename(bin));
	writeBytes(bin, blob);
	const bytes = writeText(json, JSON.stringify(header) + '\n');
	return { json, bin, bytes: bytes + blob.byteLength };
}

/** Read a tape written by either writer: header + decoded frames (tensors as { shape, data }). */
export function readTapeFiles(stem: string): { header: TapeFileHeader; frames: StoredFrame[]; blob: Uint8Array } {
	const header = readJson<TapeFileHeader>(`${stem}.tape.json`);
	if (header.format !== 'atlas-tape/1') throw new AtlasError('AK301', `${stem}.tape.json: not an atlas-tape/1 file`);
	const blob = new Uint8Array(readFileSync(`${stem}.tape.bin`));
	const dec = (v: unknown): unknown => {
		if (Array.isArray(v)) return v.map(dec);
		if (v && typeof v === 'object') {
			const o = v as Record<string, unknown>;
			if (typeof o.$t === 'string') {
				const n = o.n as number;
				const off = o.o as number;
				const buf = blob.buffer.slice(blob.byteOffset + off, blob.byteOffset + off + n * (o.$t === 'f64' ? 8 : 4));
				const data = o.$t === 'f32' ? new Float32Array(buf) : o.$t === 'f64' ? new Float64Array(buf) : new Int32Array(buf);
				return o.$t === 'i32' ? { shape: o.shape, data, dtype: 'i32' } : { shape: o.shape, data };
			}
			if (typeof o.$f === 'string' && Object.keys(o).length === 1) return o.$f === 'NaN' ? NaN : o.$f === 'Infinity' ? Infinity : -Infinity;
			const out: Record<string, unknown> = {};
			for (const [k, x] of Object.entries(o)) out[k] = dec(x);
			return out;
		}
		return v;
	};
	return { header, frames: (header.frames as unknown[]).map((f) => dec(f) as StoredFrame), blob };
}
