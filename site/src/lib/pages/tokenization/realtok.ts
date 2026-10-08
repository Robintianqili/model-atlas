/**
 * Real tokenizers for the tokenization page, re-implemented in plain TypeScript (no DOM, no Svelte),
 * so the same code runs in the page's Web Worker and in Node (parity tests against Python).
 *
 *  - ByteBPE: byte-level BPE as in tiktoken (GPT-2, cl100k_base) and Hugging Face (Qwen2.5).
 *    Split the text into pre-tokens with a regex, UTF-8 encode each one, then repeatedly merge the
 *    adjacent pair whose concatenation has the lowest token id (= rank). For GPT-2 and Qwen2.5 this
 *    equals HF's merges-based BPE (merge i always makes id 256 + i; checked on 22,927 words).
 *  - WordPiece: BERT's normalizer (clean text, space around CJK, strip accents, lowercase),
 *    split on whitespace and punctuation, then greedy longest-match-first with "##" pieces;
 *    a word that cannot be covered becomes [UNK]. Adds [CLS] ... [SEP].
 *
 * Every token keeps the byte range it covers in the UTF-8 of the ORIGINAL text, so the page can
 * link the same characters across tokenizers. Parity: pipeline/tokenizers/parity.ts.
 */
import { bytesToDisplay } from '../../models/bpe.ts';
import { NOT_PUNCT, EXTRA_PUNCT, NOT_MN, NOT_CONTROL, EXTRA_REMOVED } from './bert-unicode.ts';

export type TokKind = 'special' | 'unk' | 'partial' | 'cont';

export type RealToken = {
	id: number;
	/** readable text: " the", "##ing", "<0xE7><0x88>", "[UNK]" */
	text: string;
	/** byte range in the UTF-8 of the original text (special tokens: empty range) */
	start: number;
	end: number;
	kind?: TokKind;
};

export type RealResult = {
	tokens: RealToken[];
	/** tokens that stand for text (special tokens such as [CLS] / [SEP] are not counted) */
	count: number;
	pretokens: number;
	unk: number;
	/** tokens holding only part of a character's bytes */
	partial: number;
	/** the normalizer changed the text (offsets then follow the normalized text) */
	normalized: boolean;
};

const enc = new TextEncoder();

// ------------------------------------------------------------------------------------------------
// Regex pieces. Built from code points so the source never contains escape sequences that tools
// could rewrite, and so "whitespace" means Unicode White_Space exactly like Rust's \s
// (JavaScript's \s also matches U+FEFF and misses U+0085).
// ------------------------------------------------------------------------------------------------

const WS_RANGES: [number, number][] = [
	[0x09, 0x0d],
	[0x20, 0x20],
	[0x85, 0x85],
	[0xa0, 0xa0],
	[0x1680, 0x1680],
	[0x2000, 0x200a],
	[0x2028, 0x2029],
	[0x202f, 0x202f],
	[0x205f, 0x205f],
	[0x3000, 0x3000]
];
const cp = (n: number) => String.fromCodePoint(n);
/** The inside of a character class for Unicode White_Space. */
export const WS = WS_RANGES.map(([a, b]) => (a === b ? cp(a) : `${cp(a)}-${cp(b)}`)).join('');

export function isWhiteSpace(c: number): boolean {
	for (const [a, b] of WS_RANGES) if (c >= a && c <= b) return true;
	return false;
}

/** GPT-2 (tiktoken "gpt2" / r50k, and HF gpt2). */
export function gpt2Pattern(): RegExp {
	return new RegExp(
		`'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`,
		'gu'
	);
}

/** Case-insensitive contractions as Rust's (?i:...) sees them (simple case folding: s ~ S ~ U+017F). */
const CONTRACTIONS_I = `'(?:[sS${cp(0x17f)}]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])`;

/** GPT-4 / GPT-3.5 cl100k_base, as in tiktoken >= 0.8 (including its `\s++$` branch). */
export function cl100kPattern(): RegExp {
	return new RegExp(
		`${CONTRACTIONS_I}|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^${WS}\\p{L}\\p{N}]+[\\r\\n]*|[${WS}]+$|[${WS}]*[\\r\\n]|[${WS}]+(?![^${WS}])|[${WS}]`,
		'gu'
	);
}

/** Qwen2 / Qwen2.5 (HF tokenizer.json Split regex): single digits, otherwise like cl100k. */
export function qwenPattern(): RegExp {
	return new RegExp(
		`${CONTRACTIONS_I}|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^${WS}\\p{L}\\p{N}]+[\\r\\n]*|[${WS}]*[\\r\\n]+|[${WS}]+(?![^${WS}])|[${WS}]+`,
		'gu'
	);
}

// ------------------------------------------------------------------------------------------------
// Byte-level BPE
// ------------------------------------------------------------------------------------------------

/** Parse a .tokb file: 'TOKB' | u32 LE count | count x (u8 length, bytes). Record i = token id i. */
export function parseTokb(buf: ArrayBuffer): Uint8Array[] {
	const u8 = new Uint8Array(buf);
	if (u8[0] !== 0x54 || u8[1] !== 0x4f || u8[2] !== 0x4b || u8[3] !== 0x42) throw new Error('not a TOKB file');
	const n = new DataView(buf).getUint32(4, true);
	const out: Uint8Array[] = new Array(n);
	let o = 8;
	for (let i = 0; i < n; i++) {
		const len = u8[o];
		out[i] = u8.subarray(o + 1, o + 1 + len);
		o += 1 + len;
	}
	if (o !== u8.length) throw new Error(`TOKB size mismatch: read ${o} of ${u8.length} bytes`);
	return out;
}

function key(b: Uint8Array, s: number, e: number): string {
	let k = '';
	for (let i = s; i < e; i++) k += String.fromCharCode(b[i]);
	return k;
}

function hasPartialBytes(display: string): boolean {
	return display.includes('<0x');
}

export class ByteBPE {
	readonly kind = 'bpe' as const;
	vocab: Uint8Array[];
	ranks: Map<string, number>;
	maxLen: number;
	re: RegExp;
	nfc: boolean;
	vocabSize: number;
	_display: (string | undefined)[];

	constructor(vocab: Uint8Array[], re: RegExp, opts: { nfc?: boolean; extra?: number } = {}) {
		this.vocab = vocab;
		this.re = re;
		this.nfc = !!opts.nfc;
		this.ranks = new Map();
		let maxLen = 1;
		for (let i = 0; i < vocab.length; i++) {
			this.ranks.set(key(vocab[i], 0, vocab[i].length), i);
			if (vocab[i].length > maxLen) maxLen = vocab[i].length;
		}
		this.maxLen = maxLen;
		this.vocabSize = vocab.length + (opts.extra ?? 0);
		this._display = new Array(vocab.length);
	}

	display(id: number): string {
		let d = this._display[id];
		if (d === undefined) d = this._display[id] = bytesToDisplay(this.vocab[id]);
		return d;
	}

	_rank(piece: Uint8Array, s: number, e: number): number {
		if (e - s > this.maxLen) return Infinity;
		return this.ranks.get(key(piece, s, e)) ?? Infinity;
	}

	/** tiktoken's _byte_pair_merge: returns token ids for one pre-token. */
	bpe(piece: Uint8Array): number[] {
		const n = piece.length;
		if (n === 0) return [];
		const whole = this._rank(piece, 0, n);
		if (whole !== Infinity) return [whole];
		// starts[i] = first byte of part i; rk[i] = rank of merging part i with part i + 1
		const starts: number[] = [];
		const rk: number[] = [];
		for (let i = 0; i < n - 1; i++) {
			starts.push(i);
			rk.push(this._rank(piece, i, i + 2));
		}
		starts.push(n - 1, n);
		rk.push(Infinity, Infinity);
		const getRank = (i: number) => (i + 3 < starts.length ? this._rank(piece, starts[i], starts[i + 3]) : Infinity);
		for (;;) {
			let minR = Infinity;
			let minI = -1;
			for (let i = 0; i < rk.length - 1; i++) {
				if (rk[i] < minR) {
					minR = rk[i];
					minI = i;
				}
			}
			if (minI < 0) break;
			const i = minI;
			if (i > 0) rk[i - 1] = getRank(i - 1);
			rk[i] = getRank(i);
			starts.splice(i + 1, 1);
			rk.splice(i + 1, 1);
		}
		const ids: number[] = [];
		for (let i = 0; i < starts.length - 1; i++) {
			const r = this._rank(piece, starts[i], starts[i + 1]);
			if (r === Infinity) throw new Error('byte missing from vocabulary');
			ids.push(r);
		}
		return ids;
	}

	encode(text: string): RealResult {
		const norm = this.nfc ? text.normalize('NFC') : text;
		const tokens: RealToken[] = [];
		let off = 0;
		let lastChar = 0;
		let pretokens = 0;
		let partial = 0;
		this.re.lastIndex = 0;
		for (const m of norm.matchAll(this.re)) {
			const idx = m.index ?? 0;
			if (idx > lastChar) off += enc.encode(norm.slice(lastChar, idx)).length; // never expected
			const bytes = enc.encode(m[0]);
			lastChar = idx + m[0].length;
			pretokens++;
			for (const id of this.bpe(bytes)) {
				const len = this.vocab[id].length;
				const d = this.display(id);
				const isPartial = hasPartialBytes(d);
				if (isPartial) partial++;
				tokens.push({ id, text: d, start: off, end: off + len, kind: isPartial ? 'partial' : undefined });
				off += len;
			}
		}
		return { tokens, count: tokens.length, pretokens, unk: 0, partial, normalized: norm !== text };
	}

	encodeIds(text: string): number[] {
		return this.encode(text).tokens.map((t) => t.id);
	}
}

// ------------------------------------------------------------------------------------------------
// WordPiece (BERT)
// ------------------------------------------------------------------------------------------------

const RE_P = /^\p{P}$/u;
// HF's BERT clean_text drops Cc, Cf, Co and Cs, but keeps unassigned code points (Cn):
// checked with tokenizers 0.23.2 (U+0378, U+2FFFF -> [UNK]; U+E000, U+200B -> removed).
const RE_C = /^[\p{Cc}\p{Cf}\p{Co}\p{Cs}]$/u;
const RE_MN = /^\p{Mn}$/u;

function inRanges(c: number, r: [number, number][]): boolean {
	let lo = 0;
	let hi = r.length - 1;
	while (lo <= hi) {
		const mid = (lo + hi) >> 1;
		if (c < r[mid][0]) hi = mid - 1;
		else if (c > r[mid][1]) lo = mid + 1;
		else return true;
	}
	return false;
}
function isAsciiPunct(c: number): boolean {
	return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}
function isBertPunct(ch: string): boolean {
	const c = ch.codePointAt(0) ?? 0;
	if (isAsciiPunct(c) || inRanges(c, EXTRA_PUNCT)) return true;
	return RE_P.test(ch) && !inRanges(c, NOT_PUNCT);
}
function isBertWhitespace(ch: string): boolean {
	const c = ch.codePointAt(0) ?? 0;
	return c === 9 || c === 10 || c === 13 || isWhiteSpace(c);
}
function isBertControl(ch: string): boolean {
	const c = ch.codePointAt(0) ?? 0;
	if (c === 9 || c === 10 || c === 13) return false;
	return RE_C.test(ch) && !inRanges(c, NOT_CONTROL);
}
/** Accent stripping removes these after NFD (nonspacing marks in HF's tables). */
function isStrippedMark(ch: string): boolean {
	const c = ch.codePointAt(0) ?? 0;
	if (inRanges(c, EXTRA_REMOVED)) return true;
	return RE_MN.test(ch) && !inRanges(c, NOT_MN);
}
function stripAccents(s: string): string {
	let out = '';
	for (const q of s.normalize('NFD')) if (!isStrippedMark(q)) out += q;
	return out;
}
export function isCJK(c: number): boolean {
	return (
		(c >= 0x4e00 && c <= 0x9fff) ||
		(c >= 0x3400 && c <= 0x4dbf) ||
		(c >= 0x20000 && c <= 0x2a6df) ||
		(c >= 0x2a700 && c <= 0x2b73f) ||
		(c >= 0x2b740 && c <= 0x2b81f) ||
		(c >= 0x2b920 && c <= 0x2ceaf) ||
		(c >= 0xf900 && c <= 0xfaff) ||
		(c >= 0x2f800 && c <= 0x2fa1f)
	);
}

function utf8Len(ch: string): number {
	const c = ch.codePointAt(0) ?? 0;
	if (c >= 0xd800 && c <= 0xdfff) return 3; // lone surrogate -> U+FFFD
	return c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
}

export class WordPiece {
	readonly kind = 'wordpiece' as const;
	vocab: Map<string, number>;
	inv: string[];
	lowercase: boolean;
	maxChars = 100;
	unkId: number;
	clsId: number;
	sepId: number;
	vocabSize: number;

	constructor(lines: string[], opts: { lowercase: boolean }) {
		this.inv = lines;
		this.vocab = new Map();
		lines.forEach((w, i) => {
			if (!this.vocab.has(w)) this.vocab.set(w, i);
		});
		this.lowercase = opts.lowercase;
		this.unkId = this.vocab.get('[UNK]') ?? 100;
		this.clsId = this.vocab.get('[CLS]') ?? 101;
		this.sepId = this.vocab.get('[SEP]') ?? 102;
		this.vocabSize = lines.length;
	}

	static fromText(txt: string, opts: { lowercase: boolean }): WordPiece {
		const lines = txt.split('\n').map((l) => l.replace(/\r$/, ''));
		while (lines.length && lines[lines.length - 1] === '') lines.pop();
		return new WordPiece(lines, opts);
	}

	/** BertNormalizer + BertPreTokenizer. Returns words as code-point arrays, with the index of the
	 *  original code point each one came from. */
	words(text: string): { chars: string[]; origin: number[] }[] {
		const src = Array.from(text);
		const words: { chars: string[]; origin: number[] }[] = [];
		let cur: { chars: string[]; origin: number[] } | null = null;
		const flush = () => {
			if (cur && cur.chars.length) words.push(cur);
			cur = null;
		};
		const push = (ch: string, o: number) => {
			if (isBertWhitespace(ch)) {
				flush();
				return;
			}
			if (isBertPunct(ch)) {
				flush();
				words.push({ chars: [ch], origin: [o] });
				return;
			}
			if (!cur) cur = { chars: [], origin: [] };
			cur.chars.push(ch);
			cur.origin.push(o);
		};
		src.forEach((ch0, i) => {
			const c = ch0.codePointAt(0) ?? 0;
			if (c === 0 || c === 0xfffd || isBertControl(ch0)) return; // clean_text: drop
			const ch = isBertWhitespace(ch0) ? ' ' : ch0;
			const parts = isCJK(c) ? [' ', ch, ' '] : [ch];
			for (const p of parts) {
				let s = p;
				if (this.lowercase) s = stripAccents(s).toLowerCase(); // uncased: strip accents, then lowercase
				for (const q of s) push(q, i);
			}
		});
		flush();
		return words;
	}

	/** Greedy longest-match-first. Returns [s, e) code-point ranges and ids, or null if the word
	 *  cannot be covered (then the whole word is [UNK]). */
	pieces(chars: string[]): { s: number; e: number; id: number }[] | null {
		if (chars.length > this.maxChars) return null;
		const out: { s: number; e: number; id: number }[] = [];
		let start = 0;
		while (start < chars.length) {
			let end = chars.length;
			let found = -1;
			while (start < end) {
				let sub = chars.slice(start, end).join('');
				if (start > 0) sub = '##' + sub;
				const id = this.vocab.get(sub);
				if (id !== undefined) {
					found = id;
					break;
				}
				end--;
			}
			if (found < 0) return null;
			out.push({ s: start, e: end, id: found });
			start = end;
		}
		return out;
	}

	encode(text: string): RealResult {
		const src = Array.from(text);
		const byteAt: number[] = [0];
		for (const ch of src) byteAt.push(byteAt[byteAt.length - 1] + utf8Len(ch));
		const total = byteAt[byteAt.length - 1];
		const tokens: RealToken[] = [{ id: this.clsId, text: '[CLS]', start: 0, end: 0, kind: 'special' }];
		let unk = 0;
		const words = this.words(text);
		for (const w of words) {
			const ps = this.pieces(w.chars);
			const a = byteAt[w.origin[0]];
			const b = byteAt[w.origin[w.origin.length - 1] + 1];
			if (!ps) {
				unk++;
				tokens.push({ id: this.unkId, text: '[UNK]', start: a, end: b, kind: 'unk' });
				continue;
			}
			for (const p of ps) {
				tokens.push({
					id: p.id,
					text: this.inv[p.id],
					start: byteAt[w.origin[p.s]],
					end: byteAt[w.origin[p.e - 1] + 1],
					kind: p.s > 0 ? 'cont' : undefined
				});
			}
		}
		tokens.push({ id: this.sepId, text: '[SEP]', start: total, end: total, kind: 'special' });
		return { tokens, count: tokens.length - 2, pretokens: words.length, unk, partial: 0, normalized: false };
	}

	encodeIds(text: string): number[] {
		return this.encode(text).tokens.map((t) => t.id);
	}
}

export type AnyTokenizer = ByteBPE | WordPiece;
