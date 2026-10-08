// File helpers for the CLI: every text file is written as UTF-8 without BOM with LF line endings (Windows tools
// would otherwise add CRLF), JSON is read with a BOM tolerated.
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export function readJson<T = unknown>(path: string): T {
	return JSON.parse(readFileSync(path, 'utf8').replace(/^﻿/, '')) as T;
}

export function writeText(path: string, text: string): number {
	const data = Buffer.from(text.replace(/^﻿/, '').replace(/\r\n/g, '\n'), 'utf8');
	return writeBytes(path, data);
}

export function writeBytes(path: string, data: Uint8Array): number {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.tmp`;
	writeFileSync(tmp, data);
	renameSync(tmp, path);
	return data.byteLength;
}

export function writeJson(path: string, value: unknown, indent?: number): number {
	return writeText(path, JSON.stringify(value, null, indent) + '\n');
}

export const exists = existsSync;

/** A file-name-safe form of an id ('train:fit' -> 'train-fit'): ':' would make an NTFS alternate data stream on
 *  Windows, and '/', '\', '*', '?', '"', '<', '>', '|' are not allowed. Python uses the same rule (atlas_kit._io.file_safe). */
export const fileSafe = (id: string) => id.replace(/[^A-Za-z0-9._-]/g, '-');

/** POSIX-style path for messages. */
export const posix = (p: string) => p.replace(/\\/g, '/');
