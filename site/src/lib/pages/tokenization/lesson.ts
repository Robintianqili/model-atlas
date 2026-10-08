/**
 * Tokenization guided lesson: small pure helpers (no Svelte, no DOM), so Node can check them
 * (tests-tokenization-r3/lesson.check.ts). Everything is computed from the learner's own text and
 * the trace that trainTrace() made from it.
 */
import { isCJK } from './realtok.ts';
import { visibleToken } from '../../kit/format.ts';
import type { Segment, TrainTrace } from './train.ts';
import { utf8Bytes } from '../../models/bpe.ts';

const enc = new TextEncoder();

export type CharSummary = {
	chars: number;
	bytes: number;
	/** byLen[n] = how many characters take n UTF-8 bytes (n = 1 … 4) */
	byLen: number[];
	/** Chinese (CJK) characters */
	cjk: number;
};

/** Characters, UTF-8 bytes, and how many characters take 1, 2, 3 or 4 bytes. */
export function charSummary(text: string): CharSummary {
	const byLen = [0, 0, 0, 0, 0];
	let chars = 0;
	let bytes = 0;
	let cjk = 0;
	for (const ch of text) {
		const n = enc.encode(ch).length;
		byLen[n]++;
		bytes += n;
		chars++;
		if (isCJK(ch.codePointAt(0) ?? 0)) cjk++;
	}
	return { chars, bytes, byLen, cjk };
}

/** Byte range of each of the first `max` characters (for the byte view: one tint per character). */
export function charRanges(text: string, max = Infinity): { start: number; end: number }[] {
	const out: { start: number; end: number }[] = [];
	let off = 0;
	for (const ch of text) {
		if (out.length >= max) break;
		const n = enc.encode(ch).length;
		out.push({ start: off, end: off + n });
		off += n;
	}
	return out;
}

/** UTF-8 bytes of one character, as upper-case hex ("E6 88 91"). */
export function hexOf(ch: string): string {
	return Array.from(enc.encode(ch), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
}

export type PretokenType = { text: string; n: number; first: number; bytes: number };

/** Pre-tokens in order, and the different ones by frequency (ties: the one seen first comes first). */
export function pretokenStats(text: string, pattern: string): { list: string[]; types: PretokenType[] } {
	const list = text.match(new RegExp(pattern, 'gu')) ?? [];
	const m = new Map<string, PretokenType>();
	list.forEach((p, i) => {
		const e = m.get(p);
		if (e) e.n++;
		else m.set(p, { text: p, n: 1, first: i, bytes: enc.encode(p).length });
	});
	const types = [...m.values()].sort((a, b) => b.n - a.n || a.first - b.first);
	return { list, types };
}

export type PairSource = { text: string; ids: number[]; f: number; per: number; first: number };

/**
 * Where the count of the pair (a, b) comes from: every different pre-token that contains it, with
 * its frequency f and the number of positions per occurrence (overlapping positions count, exactly
 * as trainTrace counts them). Sum of per × f over the list = the pair's count.
 */
export function pairSources(segs: Segment[], a: number, b: number): PairSource[] {
	const m = new Map<string, PairSource>();
	segs.forEach((s, i) => {
		const e = m.get(s.text);
		if (e) {
			e.f++;
			return;
		}
		let per = 0;
		for (let j = 0; j < s.ids.length - 1; j++) if (s.ids[j] === a && s.ids[j + 1] === b) per++;
		m.set(s.text, { text: s.text, ids: s.ids, f: 1, per, first: i });
	});
	return [...m.values()].filter((s) => s.per > 0).sort((x, y) => y.per * y.f - x.per * x.f || x.first - y.first);
}

/** Start indices of the non-overlapping occurrences of (a, b), left to right (as a merge replaces them). */
export function occurrences(ids: number[], a: number, b: number): number[] {
	const out: number[] = [];
	let i = 0;
	while (i < ids.length - 1) {
		if (ids[i] === a && ids[i + 1] === b) {
			out.push(i);
			i += 2;
		} else i++;
	}
	return out;
}

/** One drawn unit of the merge animation: a token, or an occurrence of the merged pair. */
export type MergeUnit = { kind: 'tok'; id: number; i: number } | { kind: 'occ'; a: number; b: number; i: number };

/**
 * The text before merge `m` (1-based), cut into units for the animation: a group per pre-token;
 * occurrences of the pair become one 'occ' unit. Draws at most `maxTokens` tokens of the text
 * (but counts the occurrences in all of it).
 */
export function mergeUnits(
	segsBefore: Segment[],
	pair: [number, number],
	maxTokens = 800
): { groups: MergeUnit[][]; shown: number; total: number; occ: number } {
	const [a, b] = pair;
	const groups: MergeUnit[][] = [];
	let shown = 0;
	let total = 0;
	let occ = 0;
	let k = 0;
	for (const s of segsBefore) {
		total += s.ids.length;
		const at = new Set(occurrences(s.ids, a, b));
		occ += at.size;
		if (shown >= maxTokens) continue;
		const g: MergeUnit[] = [];
		for (let j = 0; j < s.ids.length && shown < maxTokens; j++) {
			if (at.has(j)) {
				g.push({ kind: 'occ', a, b, i: k++ });
				j++;
				shown += 2;
			} else {
				g.push({ kind: 'tok', id: s.ids[j], i: k++ });
				shown++;
			}
		}
		groups.push(g);
	}
	return { groups, shown, total, occ };
}

/** The first merge (1-based) whose pair occurred only once in the text, or 0 when there is none. */
export function firstSingleton(trace: TrainTrace): number {
	const i = trace.steps.findIndex((s) => s.count === 1);
	return i < 0 ? 0 : i + 1;
}

/** Tokens in the whole text after k merges. */
export function tokensAt(trace: TrainTrace, k: number): number {
	const kk = Math.max(0, Math.min(k, trace.steps.length));
	return kk === 0 ? trace.initialTokenCount : trace.steps[kk - 1].tokenCount;
}

/**
 * The first pre-tokens of the text, up to about `bytes` bytes (at least one) and at most `maxTokens`
 * tokens: a short preview. A pre-token longer than that (2,000 Chinese characters without spaces are
 * one pre-token) is cut, and `cut` says so.
 */
export function previewSegs(segs: Segment[], bytes = 90, maxTokens = 48): Segment[] & { cut?: boolean } {
	const out: Segment[] & { cut?: boolean } = [];
	let n = 0;
	let toks = 0;
	for (const s of segs) {
		const b = enc.encode(s.text).length;
		if (out.length && (n + b > bytes || toks >= maxTokens)) break;
		const room = maxTokens - toks;
		if (s.ids.length > room) {
			out.push({ text: s.text, ids: s.ids.slice(0, room) });
			out.cut = true;
			break;
		}
		out.push(s);
		n += b;
		toks += s.ids.length;
	}
	return out;
}

/** A token's display text for plain-text labels: raw bytes "<0xEF><0xBC>" become "EF BC", a space "␣". */
export function plainToken(s: string): string {
	if (!s.includes('<0x')) return visibleToken(s);
	const parts: string[] = [];
	let last = 0;
	for (const m of s.matchAll(/((?:<0x[0-9A-F]{2}>)+)/g)) {
		const at = m.index ?? 0;
		if (at > last) parts.push(visibleToken(s.slice(last, at)));
		parts.push(m[1].replace(/<0x([0-9A-F]{2})>/g, '$1 ').trim());
		last = at + m[1].length;
	}
	if (last < s.length) parts.push(visibleToken(s.slice(last)));
	return parts.join(' ');
}

function mergeAll(ids: number[], a: number, b: number, newId: number): number[] {
	const out: number[] = [];
	let i = 0;
	while (i < ids.length) {
		if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
			out.push(newId);
			i += 2;
		} else {
			out.push(ids[i]);
			i += 1;
		}
	}
	return out;
}

/**
 * The text cut into pre-tokens with the first k merges applied, like segment() in train.ts, but
 * incremental: merges are applied in training order (which gives the same tokens as encoding by
 * rank), so moving k forward by one costs one pass over the different pre-tokens. Going back starts
 * again from the bytes. Keeps long single pre-tokens (2,000 Chinese characters, no spaces) fast
 * while a player steps through hundreds of merges.
 */
export class SegCursor {
	private texts: string[];
	/** index into `words` for every pre-token of the text */
	private order: number[];
	private words0: number[][];
	private words: number[][];
	private k = 0;
	private merges: [number, number][];
	constructor(text: string, pattern: string, merges: [number, number][]) {
		this.merges = merges;
		const list = text.match(new RegExp(pattern, 'gu')) ?? [];
		const idx = new Map<string, number>();
		this.texts = [];
		this.words0 = [];
		this.order = list.map((pt) => {
			let i = idx.get(pt);
			if (i === undefined) {
				i = this.texts.length;
				idx.set(pt, i);
				this.texts.push(pt);
				this.words0.push(utf8Bytes(pt));
			}
			return i;
		});
		this.words = this.words0;
	}
	/** segments after the first k merges (a fresh array; do not mutate the ids) */
	at(k: number): Segment[] {
		const kk = Math.max(0, Math.min(k, this.merges.length));
		if (kk < this.k) {
			this.words = this.words0;
			this.k = 0;
		}
		while (this.k < kk) {
			const [a, b] = this.merges[this.k];
			const id = 256 + this.k;
			this.words = this.words.map((w) => {
				for (let i = 0; i < w.length - 1; i++) if (w[i] === a && w[i + 1] === b) return mergeAll(w, a, b, id);
				return w;
			});
			this.k++;
		}
		return this.order.map((i) => ({ text: this.texts[i], ids: this.words[i] }));
	}
}
