// Wrap the notes about the maintainers' own computer in site/DEVKIT.md in local-only markers (local.ts), so the /devkit/
// pages leave them out. Run by the maintainers after they add, move or reword a machine note; safe to run again: a file
// already marked this way is left as it is.
//
//   node src/lib/devkit-site/mark-local.ts [DEVKIT.md]            mark the file in place, print what was wrapped
//   node src/lib/devkit-site/mark-local.ts [DEVKIT.md] --check    exit 1 unless the file is marked (nothing written)
//
// The block list is this computer's: it quotes the notes, so it lives in the git-ignored machine.local.json (machine.ts,
// "blocks"), not in this file. Without it (a fresh clone) there is nothing to mark, and DEVKIT.md has no markers. Each
// block is found by the words it starts and ends with, so the script refuses (and writes nothing) when DEVKIT.md has
// changed there; then update the block in machine.local.json. A marker either sits on its own line (`line`) or inside a
// line (`inline`). The result is checked: removing the markers gives back the file byte for byte, and no local-machine
// string (local.ts) is left in the public text.
import { readFileSync, writeFileSync } from 'node:fs';
import { findLocalStrings, LOCAL_END, LOCAL_START, publicText } from './local.ts';
import { machine, MACHINE_FILE, type Block } from './machine.ts';

export type { Block };

/** The local-only blocks of site/DEVKIT.md on this computer (machine.local.json "blocks"; empty without the file). */
export const BLOCKS: Block[] = machine().blocks;

/** The file without any local-only marker (marker lines dropped, inline markers removed). */
export function unmark(text: string): string {
	return text
		.split('\n')
		.filter((l) => l !== LOCAL_START && l !== LOCAL_END)
		.join('\n')
		.split(LOCAL_START)
		.join('')
		.split(LOCAL_END)
		.join('');
}

export interface Placed {
	block: Block;
	/** 1-based lines of the unmarked file */
	fromLine: number;
	toLine: number;
	first: string;
	last: string;
}

/** Mark an unmarked text. Throws (naming the block) when a block's start is missing or not unique, or its end is
 *  missing. */
export function mark(text: string, blocks: Block[] = BLOCKS): { text: string; placed: Placed[] } {
	if (text.includes(LOCAL_START) || text.includes(LOCAL_END)) throw new Error('the text already has local-only markers: unmark() it first');
	const ins: [number, string][] = [];
	const placed: Placed[] = [];
	const lineOf = (at: number) => text.slice(0, at).split('\n').length;
	const lineStart = (at: number) => text.lastIndexOf('\n', at - 1) + 1;
	const lineEnd = (at: number) => {
		const e = text.indexOf('\n', at);
		return e < 0 ? text.length : e;
	};
	for (const b of blocks) {
		const at = text.indexOf(b.from);
		if (at < 0) throw new Error(`block '${b.id}' (${b.where}): its start "${b.from}" is not in the file`);
		if (text.indexOf(b.from, at + 1) >= 0) throw new Error(`block '${b.id}': its start "${b.from}" occurs more than once`);
		const toAt = text.indexOf(b.to, at);
		if (toAt < 0) throw new Error(`block '${b.id}' (${b.where}): no "${b.to}" after its start`);
		const end = toAt + b.to.length;
		const s = b.fromMode === 'line' ? lineStart(at) : at;
		const e = b.toMode === 'line' ? lineEnd(end - 1) : end;
		ins.push([s, b.fromMode === 'line' ? `${LOCAL_START}\n` : LOCAL_START]);
		ins.push([e, b.toMode === 'line' ? `\n${LOCAL_END}` : LOCAL_END]);
		const body = text.slice(s, e).trim();
		const words = body.split(/\s+/);
		placed.push({ block: b, fromLine: lineOf(s), toLine: lineOf(e), first: words.slice(0, 8).join(' '), last: words.slice(-6).join(' ') });
	}
	// insert from the end; at one offset an end marker goes before a start marker
	ins.sort((a, b) => b[0] - a[0] || (a[1].includes('start') ? -1 : 1));
	let out = text;
	for (const [at, s] of ins) out = out.slice(0, at) + s + out.slice(at);
	return { text: out, placed };
}

/** Is the text marked exactly as BLOCKS says? */
export function isMarked(text: string, blocks: Block[] = BLOCKS): boolean {
	try {
		return mark(unmark(text), blocks).text === text;
	} catch {
		return false;
	}
}

if (process.argv[1] && /mark-local\.ts$/.test(process.argv[1])) {
	const args = process.argv.slice(2);
	const file = args.find((a) => !a.startsWith('--')) ?? 'DEVKIT.md';
	const raw = readFileSync(file, 'utf8');
	const eol = raw.includes('\r\n') ? '\r\n' : '\n';
	const text = raw.replace(/\r\n/g, '\n');
	const report = (t: string) => {
		const left = findLocalStrings(publicText(t));
		if (left.length) {
			console.log(`  local-machine strings still in the public text (wrap them too):\n${left.map((l) => `    ${l}`).join('\n')}`);
			return false;
		}
		console.log('  public text: no local-machine string');
		return true;
	};
	if (!BLOCKS.length) {
		console.log(`${file}: no block list on this computer (${MACHINE_FILE} has no "blocks"; it is git-ignored): nothing to mark`);
		process.exitCode = text.includes(LOCAL_START) ? 1 : 0;
	} else if (isMarked(text)) {
		console.log(`${file}: already marked (${BLOCKS.length} local-only blocks)`);
		process.exitCode = report(text) ? 0 : 1;
	} else if (args.includes('--check')) {
		console.log(`${file}: NOT marked as the block list in ${MACHINE_FILE} says`);
		process.exitCode = 1;
	} else {
		const r = mark(text);
		if (unmark(r.text) !== text) throw new Error('internal: removing the markers does not give the file back');
		for (const p of r.placed) console.log(`  ${p.block.id.padEnd(16)} lines ${p.fromLine}-${p.toLine}  "${p.first} …" to "… ${p.last}"`);
		writeFileSync(file, r.text.split('\n').join(eol));
		console.log(`${file}: wrapped ${r.placed.length} local-only blocks`);
		process.exitCode = report(r.text) ? 0 : 1;
	}
}
