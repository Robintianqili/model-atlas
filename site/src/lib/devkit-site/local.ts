// Notes about the maintainers' own computer stay off the public website.
//
// site/DEVKIT.md is also the maintainers' working manual: a few of its lines describe the computer the project is built
// on (its drive and local paths, the Python it uses, its git setting, where its build agents keep scratch files). They
// are wrapped in a marker pair, around whole lines or inside a line:
//
//   <!-- local-only:start -->
//   ...
//   <!-- local-only:end -->
//
// What is between the markers is never rendered on a /devkit/ page, and a guide section's hash (the staleness check of
// its Chinese summary, guide.ts sectionHash) leaves it out, so editing a machine note never stales a summary. Markers
// inside a fenced code block are text, not markers. A start without an end (or the reverse) throws, so a typo fails the
// build instead of hiding the rest of the file or publishing the notes.
//
// The guard: a /devkit/ page whose data or HTML contains a local-machine string fails the build (data.server.ts),
// check-build.ts and devkit-site.test.ts. The strings are the generic LOCAL_STRINGS below plus this computer's own, read
// from the git-ignored machine.local.json (machine.ts), so no published file names a maintainer's paths or folders.
// Node only (machine.ts reads a file); the rest of this module is pure.
import { machine, type LocalString } from './machine.ts';

export const LOCAL_START = '<!-- local-only:start -->';
export const LOCAL_END = '<!-- local-only:end -->';

/** What a heading inside a local-only block leaves in the public text: its text only, as an HTML comment that renders
 *  nothing. The renderer still counts it, so the ids of the headings after it stay the file's own (GitHub numbers
 *  repeated titles across the whole file). */
export const HIDDEN_HEADING = /^<!-- local-only heading: (.*) -->$/;
const hiddenHeading = (text: string) => `<!-- local-only heading: ${text.replace(/-->/g, '--&gt;')} -->`;

const FENCE = /^\s*(```+|~~~+)/;
const HEADING = /^#{1,6}\s+(.*?)\s*#*\s*$/;

export interface Stripped {
	/** the public text */
	text: string;
	/** one entry per local-only block: the line of the public text where it was cut out (0-based) */
	cuts: number[];
}

export class LocalOnlyError extends Error {}

/** The public text of a Markdown file: every local-only block removed. A line that held only markers (or only text
 *  inside a block) is dropped; a line cut in the middle keeps its public part. */
export function stripLocalOnly(md: string): Stripped {
	const lines = md.replace(/\r\n/g, '\n').split('\n');
	const out: string[] = [];
	const cuts: number[] = [];
	let fence: string | null = null;
	let local = false;
	let openedAt = -1;
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const f = FENCE.exec(line);
		if (f || fence !== null) {
			// a fence line, or a line inside a fenced block: markers here are text
			if (f) fence = fence === null ? f[1] : line.trim().startsWith(fence) ? null : fence;
			if (!local) out.push(line);
			continue;
		}
		if (!line.includes('<!-- local-only:')) {
			if (!local) out.push(line);
			else {
				const h = HEADING.exec(line);
				if (h) out.push(hiddenHeading(h[1]));
			}
			continue;
		}
		let keep = '';
		let pos = 0;
		for (;;) {
			const s = line.indexOf(LOCAL_START, pos);
			const e = line.indexOf(LOCAL_END, pos);
			if (!local) {
				if (e >= 0 && (s < 0 || e < s)) throw new LocalOnlyError(`line ${i + 1}: ${LOCAL_END} without a ${LOCAL_START} before it`);
				if (s < 0) {
					keep += line.slice(pos);
					break;
				}
				keep += line.slice(pos, s);
				local = true;
				openedAt = i;
				cuts.push(out.length);
				pos = s + LOCAL_START.length;
			} else {
				if (s >= 0 && (e < 0 || s < e)) throw new LocalOnlyError(`line ${i + 1}: ${LOCAL_START} inside the local-only block opened at line ${openedAt + 1}`);
				if (e < 0) break;
				local = false;
				pos = e + LOCAL_END.length;
			}
		}
		const unknown = /<!-- local-only:(?!start -->|end -->)/.exec(keep);
		if (unknown) throw new LocalOnlyError(`line ${i + 1}: unknown marker "${keep.slice(unknown.index, unknown.index + 30)}…" (use ${LOCAL_START} and ${LOCAL_END})`);
		if (keep.trim() !== '') out.push(keep.replace(/[ \t]+$/, ''));
	}
	if (local) throw new LocalOnlyError(`line ${openedAt + 1}: ${LOCAL_START} has no ${LOCAL_END}`);
	return { text: out.join('\n'), cuts };
}

/** The public text, for hashing: no local-only blocks, no hidden-heading placeholders, no runs of blank lines (so a
 *  block removed between two paragraphs leaves the same text as one never written). */
export function publicText(md: string): string {
	return stripLocalOnly(md)
		.text.split('\n')
		.filter((l) => !HIDDEN_HEADING.test(l.trim()))
		.join('\n')
		.replace(/\n[ \t]*\n(?:[ \t]*\n)+/g, '\n\n');
}

// ------------------------------------------------------------------ the guard
export type { LocalString };

/** Strings that no computer's notes may show on a public page: the kinds of local detail every machine note has. The
 *  strings of one computer (its drive and folder names, its user name) are not here: they come from the git-ignored
 *  machine.local.json (machine.ts). AK004's meaning is "git refuses the repo", so any mention of the git setting that
 *  some drives need is caught. */
export const LOCAL_STRINGS: LocalString[] = [
	{ re: /safe\.directory/i, what: "git's safe.directory setting (a fix for one computer's drive)" },
	{ re: /\bthis machine\b/i, what: '"this machine"' },
	{ re: /\b[A-Z]:\\/, what: 'a Windows path with a drive letter (X:\\...)' },
	{ re: /\b[A-Z]:[\\/]+Users[\\/]/i, what: 'a Windows user folder (C:/Users/...)' },
	{ re: /\bAppData\b/, what: 'a Windows AppData path' },
	{ re: /(?<![\w.:-])\/(?:home|Users)\/[A-Za-z][\w.-]*\//, what: 'a home folder (/home/<name>/ or /Users/<name>/)' },
	{ re: /scratchpad/i, what: "an agent's scratchpad" }
];

/** The strings the guard refuses on this computer: the generic ones and this computer's own (machine.local.json). */
export function localStrings(): LocalString[] {
	return [...LOCAL_STRINGS, ...machine().strings];
}

/** The local-machine strings in a text: `what` and a short excerpt for each hit (empty when clean). */
export function findLocalStrings(text: string, list: LocalString[] = localStrings()): string[] {
	const out: string[] = [];
	for (const { re, what } of list) {
		const all = new RegExp(re.source, re.flags.replace('g', '') + 'g');
		for (const hit of text.matchAll(all)) {
			const at = hit.index ?? 0;
			out.push(`${what}: "…${text.slice(Math.max(0, at - 30), at + hit[0].length + 30).replace(/\s+/g, ' ')}…"`);
			if (out.length > 20) return out;
		}
	}
	return out;
}
