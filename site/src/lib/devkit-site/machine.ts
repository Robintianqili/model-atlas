// What only one computer knows about itself: its own strings for the local-machine guard (local.ts) and the list of
// local-only blocks that mark-local.ts wraps in site/DEVKIT.md. Both are read from the git-ignored file
// src/lib/devkit-site/machine.local.json, so no published file quotes a maintainer's paths, names or machine notes.
// Node only (server code, tests and scripts).
//
// The file (all keys optional; a missing file is the same as {}: a fresh clone has none):
//   {
//     "strings": [{ "re": "<regular expression source>", "flags": "i", "what": "what it is, for the error message" }],
//     "blocks":  [{ "id": "...", "where": "...", "from": "...", "fromMode": "inline" | "line",
//                   "to": "...", "toMode": "inline" | "line" }]
//   }
// `strings` add to the generic patterns of local.ts (a /devkit/ page with one of them fails the build); `blocks` are
// mark-local.ts's block list (each block is found by the text it starts and ends with). The CLI copies the file into
// every lane (scripts/atlas/lib/lanes.ts LANE_LOCAL); the publication export leaves every *.local.json out.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** A string no public page may show: a pattern and what it is. */
export interface LocalString {
	re: RegExp;
	what: string;
}

/** A local-only block of DEVKIT.md (mark-local.ts). */
export interface Block {
	id: string;
	/** where it is, for the people who read the list */
	where: string;
	/** the block starts at this text (it must occur once in the file); `line`: the start marker goes on its own line
	 *  before the line that holds the text */
	from: string;
	fromMode: 'inline' | 'line';
	/** the block ends with the first occurrence of this text after `from`; `line`: the end marker goes on its own line
	 *  after the line that holds the end of the text */
	to: string;
	toMode: 'inline' | 'line';
}

export interface Machine {
	/** the file it was read from (null: there is none) */
	file: string | null;
	strings: LocalString[];
	blocks: Block[];
}

/** The file, relative to the site folder. */
export const MACHINE_FILE = 'src/lib/devkit-site/machine.local.json';

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const MODES = ['inline', 'line'];

/** Parse a machine file's text; throws an error that names the file and the entry. */
export function parseMachine(text: string, file: string): Machine {
	let j: unknown;
	try {
		j = JSON.parse(text.replace(/^﻿/, ''));
	} catch (e) {
		throw new Error(`${file}: not valid JSON (${(e as Error).message})`);
	}
	if (!isObj(j)) throw new Error(`${file}: expected an object with "strings" and "blocks"`);
	const strings: LocalString[] = [];
	const blocks: Block[] = [];
	const rawStrings = j.strings ?? [];
	const rawBlocks = j.blocks ?? [];
	if (!Array.isArray(rawStrings)) throw new Error(`${file}: "strings" must be an array`);
	if (!Array.isArray(rawBlocks)) throw new Error(`${file}: "blocks" must be an array`);
	rawStrings.forEach((s, i) => {
		if (!isObj(s) || typeof s.re !== 'string' || !s.re || typeof s.what !== 'string' || (s.flags !== undefined && typeof s.flags !== 'string'))
			throw new Error(`${file}: strings[${i}] needs "re" and "what" (strings) and an optional "flags"`);
		let re: RegExp;
		try {
			re = new RegExp(s.re, ((s.flags as string | undefined) ?? '').replace('g', ''));
		} catch (e) {
			throw new Error(`${file}: strings[${i}] is not a valid regular expression (${(e as Error).message})`);
		}
		strings.push({ re, what: s.what });
	});
	rawBlocks.forEach((b, i) => {
		const ok =
			isObj(b) &&
			['id', 'where', 'from', 'to'].every((k) => typeof b[k] === 'string' && (b[k] as string).length > 0) &&
			MODES.includes(b.fromMode as string) &&
			MODES.includes(b.toMode as string);
		if (!ok) throw new Error(`${file}: blocks[${i}] needs "id", "where", "from" and "to" (strings) and "fromMode" and "toMode" ("inline" or "line")`);
		const x = b as Record<string, string>;
		blocks.push({ id: x.id, where: x.where, from: x.from, fromMode: x.fromMode as Block['fromMode'], to: x.to, toMode: x.toMode as Block['toMode'] });
	});
	return { file, strings, blocks };
}

/** Where the file is: next to this module in the source tree, else under the current folder (the site folder of a build,
 *  which bundles this module elsewhere). Null when neither exists. (No `new URL('./…', import.meta.url)` here: Vite
 *  would take that for an asset and copy the file into the build.) */
export function machineFile(): string | null {
	const candidates: string[] = [];
	try {
		const dir = dirname(fileURLToPath(import.meta.url));
		if (/[\\/]devkit-site$/.test(dir)) candidates.push(join(dir, 'machine.local.json'));
	} catch {
		/* not a file: URL */
	}
	candidates.push(join(process.cwd(), ...MACHINE_FILE.split('/')));
	return candidates.find((p) => existsSync(p)) ?? null;
}

let memo: Machine | null = null;

/** This computer's strings and blocks (empty without a machine file); read once. */
export function machine(): Machine {
	if (memo) return memo;
	const file = machineFile();
	memo = file ? parseMachine(readFileSync(file, 'utf8'), file.replace(/\\/g, '/')) : { file: null, strings: [], blocks: [] };
	return memo;
}
