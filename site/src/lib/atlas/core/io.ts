// Loading helpers shared by the browser and Node (design §3.1 core/io.ts).
//   loadJson(fetch, url)   JSON with a UTF-8 BOM stripped (Windows tools sometimes write one)
//   fsFetch                a FetchLike that reads local files (Node only; node:fs is imported lazily,
//                          so bundling this module for the browser pulls in nothing)
//   base64 <-> typed arrays (little-endian), for atlas-ref-1 references and goldens
import type { FetchLike } from '../types/index.ts';

const BOM = '﻿';

/** Parse JSON text, ignoring a leading BOM. */
export function parseJson(text: string): unknown {
	return JSON.parse(text.startsWith(BOM) ? text.slice(1) : text);
}

/** Join a base URL (or folder) and a file name with exactly one slash. */
export function joinUrl(base: string, file: string): string {
	if (!base) return file;
	return base.replace(/\/+$/, '') + '/' + file.replace(/^\/+/, '');
}

async function ok(fetch: FetchLike, url: string) {
	const r = await fetch(url);
	if (r.ok === false) throw new Error(`could not load ${url} (HTTP ${r.status ?? 'error'})`);
	return r;
}

/** Fetch and parse JSON; a BOM is stripped. Uses text() when the response has it, else bytes. */
export async function loadJson<T = unknown>(fetch: FetchLike, url: string): Promise<T> {
	const r = await ok(fetch, url);
	if (typeof r.text === 'function') return parseJson(await r.text()) as T;
	const buf = await r.arrayBuffer();
	return parseJson(new TextDecoder('utf-8').decode(buf)) as T;
}

/** Fetch raw bytes. */
export async function loadBytes(fetch: FetchLike, url: string): Promise<ArrayBuffer> {
	const r = await ok(fetch, url);
	return r.arrayBuffer();
}

type ReadFile = (path: string) => Promise<Uint8Array>;
let readFileP: Promise<ReadFile> | null = null;
function nodeReadFile(): Promise<ReadFile> {
	if (!readFileP) {
		// Hidden from bundlers: this branch only runs in Node.
		const spec = 'node:fs/promises';
		readFileP = (import(/* @vite-ignore */ spec) as Promise<{ readFile: ReadFile }>).then((m) => m.readFile);
	}
	return readFileP;
}

/** A FetchLike over the local file system (Node). `url` is a path; a `file://` URL also works. Missing files give ok: false. */
export const fsFetch: FetchLike = async (url: string) => {
	const readFile = await nodeReadFile();
	let path = url;
	if (path.startsWith('file://')) {
		path = decodeURIComponent(new URL(path).pathname);
		if (/^\/[A-Za-z]:\//.test(path)) path = path.slice(1); // Windows drive
	}
	let buf: Uint8Array;
	try {
		buf = await readFile(path);
	} catch (e) {
		const code = (e as { code?: string }).code;
		return {
			ok: false,
			status: code === 'ENOENT' ? 404 : 500,
			json: async () => {
				throw e;
			},
			arrayBuffer: async () => {
				throw e;
			},
			text: async () => {
				throw e;
			}
		};
	}
	const text = () => new TextDecoder('utf-8').decode(buf);
	return {
		ok: true,
		status: 200,
		json: async () => parseJson(text()),
		arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
		text: async () => text()
	};
};

// ---------------------------------------------------------------- base64

/** base64 -> bytes (atob is global in browsers and Node >= 16). */
export function b64ToBytes(b64: string): Uint8Array {
	const bin = atob(b64);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}

/** bytes -> base64 (chunked, so large arrays do not overflow the argument list). */
export function bytesToB64(bytes: Uint8Array): string {
	let s = '';
	const CH = 0x8000;
	for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode(...bytes.subarray(i, i + CH));
	return btoa(s);
}

function readLE<T>(b64: string, size: number, make: (n: number) => T, get: (dv: DataView, off: number) => number, set: (out: T, i: number, v: number) => void): T {
	const bytes = b64ToBytes(b64);
	if (bytes.length % size !== 0) throw new Error(`base64 payload of ${bytes.length} bytes is not a multiple of ${size}`);
	const n = bytes.length / size;
	const out = make(n);
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	for (let i = 0; i < n; i++) set(out, i, get(dv, i * size));
	return out;
}

/** base64 of little-endian float32 -> Float32Array (-Infinity and NaN kept as stored). */
export function f32FromB64(b64: string): Float32Array {
	return readLE(b64, 4, (n) => new Float32Array(n), (dv, o) => dv.getFloat32(o, true), (a, i, v) => (a[i] = v));
}

/** base64 of little-endian float64 -> Float64Array. */
export function f64FromB64(b64: string): Float64Array {
	return readLE(b64, 8, (n) => new Float64Array(n), (dv, o) => dv.getFloat64(o, true), (a, i, v) => (a[i] = v));
}

/** base64 of little-endian int32 -> Int32Array. */
export function i32FromB64(b64: string): Int32Array {
	return readLE(b64, 4, (n) => new Int32Array(n), (dv, o) => dv.getInt32(o, true), (a, i, v) => (a[i] = v));
}

/** Float32Array -> base64 of little-endian float32. */
export function f32ToB64(a: ArrayLike<number>): string {
	const bytes = new Uint8Array(a.length * 4);
	const dv = new DataView(bytes.buffer);
	for (let i = 0; i < a.length; i++) dv.setFloat32(i * 4, a[i], true);
	return bytesToB64(bytes);
}

/** Float64Array -> base64 of little-endian float64. */
export function f64ToB64(a: ArrayLike<number>): string {
	const bytes = new Uint8Array(a.length * 8);
	const dv = new DataView(bytes.buffer);
	for (let i = 0; i < a.length; i++) dv.setFloat64(i * 8, a[i], true);
	return bytesToB64(bytes);
}

/** Int32 values -> base64 of little-endian int32. */
export function i32ToB64(a: ArrayLike<number>): string {
	const bytes = new Uint8Array(a.length * 4);
	const dv = new DataView(bytes.buffer);
	for (let i = 0; i < a.length; i++) dv.setInt32(i * 4, a[i], true);
	return bytesToB64(bytes);
}

/** Read float32 little-endian values from a buffer (weights.bin), independent of platform endianness. */
export function f32FromBuffer(buf: ArrayBuffer, offsetFloats = 0, count?: number): Float32Array {
	const n = count ?? buf.byteLength / 4 - offsetFloats;
	const dv = new DataView(buf);
	const out = new Float32Array(n);
	for (let i = 0; i < n; i++) out[i] = dv.getFloat32((offsetFloats + i) * 4, true);
	return out;
}
