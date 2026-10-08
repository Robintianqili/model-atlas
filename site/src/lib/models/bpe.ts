/**
 * Byte-level BPE tokenizer (GPT-2 style) used by TinyGPT. Pure TypeScript: no DOM, no Svelte,
 * so it runs in the page, in a Worker and in Node (parity tests).
 *
 * It mirrors pipeline/tinygpt/bpe.py exactly; pipeline/parity/bpe.parity.ts checks that both give
 * identical ids on 300 test strings (English, Chinese, emoji, numbers, punctuation, whitespace runs).
 *
 * How text becomes ids:
 *   1. split on special tokens ("<|endoftext|>" becomes one id),
 *   2. pre-tokenize with a regex (words keep their leading space: " cat"),
 *   3. UTF-8 encode each pre-token: ids 0..255 are the raw bytes,
 *   4. repeatedly merge the adjacent pair with the lowest merge index (merge i makes id 256 + i).
 */

export type TokenizerJSON = {
  type: string;
  vocabSize: number;
  pretokenize: { pattern: string; flags: string };
  byteToToken: number[];
  merges: [number, number][];
  tokenBytes: string[];
  specialTokens: Record<string, number>;
  mergeCounts?: number[];
  trainedOn?: string;
  notes?: string;
};

/** One merge applied inside one pre-token. `tokens` is the token list right after this merge. */
export type MergeStep = { pair: [number, number]; rank: number; newId: number; tokens: number[] };

export type PretokenTrace = {
  text: string;
  bytes: number[];
  steps: MergeStep[];
  ids: number[];
  special: boolean;
};

export type EncodeSteps = { pretokens: PretokenTrace[]; ids: number[] };

export type FetchLike = (url: string) => Promise<{
  ok: boolean;
  status?: number;
  json(): Promise<unknown>;
  arrayBuffer(): Promise<ArrayBuffer>;
}>;

const encoder = new TextEncoder();

export function utf8Bytes(text: string): number[] {
  return Array.from(encoder.encode(text));
}

export function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return out;
}

function byteLabel(b: number): string {
  return '<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>';
}

/**
 * Decode bytes as UTF-8, but show every byte that is not part of a complete, valid character
 * as <0xE7> (e.g. a token holding only the first 2 of the 3 bytes of a Chinese character).
 */
export function bytesToDisplay(bytes: ArrayLike<number>): string {
  let out = '';
  const n = bytes.length;
  let i = 0;
  while (i < n) {
    const c = bytes[i];
    if (c < 0x80) {
      out += String.fromCharCode(c);
      i += 1;
      continue;
    }
    const need = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
    let cp = -1;
    if (need > 0 && i + need <= n - 1) {
      cp = c & (need === 1 ? 0x1f : need === 2 ? 0x0f : 0x07);
      for (let k = 1; k <= need; k++) {
        const b = bytes[i + k];
        if (b < 0x80 || b > 0xbf) {
          cp = -1;
          break;
        }
        cp = (cp << 6) | (b & 0x3f);
      }
      // reject overlong forms, UTF-16 surrogates and code points above U+10FFFF
      if (need === 2 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) cp = -1;
      if (need === 3 && (cp < 0x10000 || cp > 0x10ffff)) cp = -1;
    }
    if (cp >= 0) {
      out += String.fromCodePoint(cp);
      i += need + 1;
    } else {
      out += byteLabel(c);
      i += 1;
    }
  }
  return out;
}

/** Make spaces, tabs and newlines visible: space -> "·", tab -> "→", newline -> "↵". */
export function visibleWhitespace(s: string): string {
  return s.replace(/ /g, '·').replace(/\t/g, '→').replace(/\r?\n/g, '↵');
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

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export class BPETokenizer {
  vocabSize: number;
  merges: [number, number][];
  specialTokens: Record<string, number>;
  pattern: string;
  _ranks: Map<number, number>;
  _bytes: Uint8Array[];
  _specialById: Map<number, string>;
  _pretokenRe: RegExp;
  _specialRe: RegExp | null;
  _cache: Map<string, number[]>;

  constructor(json: TokenizerJSON) {
    this.merges = json.merges.map((m) => [m[0], m[1]] as [number, number]);
    this.pattern = json.pretokenize.pattern;
    this._pretokenRe = new RegExp(this.pattern, json.pretokenize.flags || 'gu');
    this._ranks = new Map();
    this._bytes = [];
    for (let b = 0; b < 256; b++) this._bytes.push(new Uint8Array([b]));
    this.merges.forEach(([a, b], i) => {
      this._ranks.set(a * 65536 + b, i);
      const x = this._bytes[a];
      const y = this._bytes[b];
      const z = new Uint8Array(x.length + y.length);
      z.set(x, 0);
      z.set(y, x.length);
      this._bytes.push(z);
    });
    this.specialTokens = {};
    this._specialById = new Map();
    const specials = Object.entries(json.specialTokens || {}).sort((p, q) => p[1] - q[1]);
    for (const [text, id] of specials) {
      if (id !== this._bytes.length) throw new Error(`special token ${text} has id ${id}, expected ${this._bytes.length}`);
      this.specialTokens[text] = id;
      this._specialById.set(id, text);
      this._bytes.push(encoder.encode(text));
    }
    this.vocabSize = this._bytes.length;
    if (json.vocabSize !== undefined && json.vocabSize !== this.vocabSize) {
      throw new Error(`vocabSize ${json.vocabSize} does not match merges (${this.vocabSize})`);
    }
    const names = Object.keys(this.specialTokens);
    this._specialRe = names.length ? new RegExp('(' + names.map(escapeRegExp).join('|') + ')') : null;
    this._cache = new Map();
  }

  /** Build a tokenizer from merges alone (e.g. merges the learner just trained with trainBPE). */
  static fromMerges(merges: [number, number][], pattern: string, specialTokens: string[] = []): BPETokenizer {
    const st: Record<string, number> = {};
    specialTokens.forEach((s, i) => (st[s] = 256 + merges.length + i));
    return new BPETokenizer({
      type: 'byte-level-bpe',
      vocabSize: 256 + merges.length + specialTokens.length,
      pretokenize: { pattern, flags: 'gu' },
      byteToToken: [],
      merges,
      tokenBytes: [],
      specialTokens: st,
    });
  }

  /** Split text (with no special tokens in it) into pre-tokens. */
  pretokenize(text: string): string[] {
    this._pretokenRe.lastIndex = 0;
    return text.match(this._pretokenRe) ?? [];
  }

  /** Split text into pieces; special tokens become their own pieces. */
  splitSpecial(text: string): { text: string; special: boolean }[] {
    if (!this._specialRe) return text ? [{ text, special: false }] : [];
    return text
      .split(this._specialRe)
      .filter((p) => p !== '')
      .map((p) => ({ text: p, special: p in this.specialTokens }));
  }

  /** Merge rank of the pair (a, b), or undefined if it was never learned. */
  rank(a: number, b: number): number | undefined {
    return this._ranks.get(a * 65536 + b);
  }

  /** Apply the merges to one pre-token's bytes, recording every step. */
  bpeSteps(bytes: number[]): { ids: number[]; steps: MergeStep[] } {
    let ids = bytes.slice();
    const steps: MergeStep[] = [];
    while (ids.length >= 2) {
      let bestRank = -1;
      let bestI = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this._ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && (bestRank < 0 || r < bestRank)) {
          bestRank = r;
          bestI = i;
        }
      }
      if (bestRank < 0) break;
      const a = ids[bestI];
      const b = ids[bestI + 1];
      ids = mergeAll(ids, a, b, 256 + bestRank);
      steps.push({ pair: [a, b], rank: bestRank, newId: 256 + bestRank, tokens: ids.slice() });
    }
    return { ids, steps };
  }

  _encodePretoken(piece: string): number[] {
    let ids = this._cache.get(piece);
    if (!ids) {
      ids = this.bpeSteps(utf8Bytes(piece)).ids;
      if (this._cache.size < 50000) this._cache.set(piece, ids);
    }
    return ids;
  }

  /** Text -> token ids. Special tokens written in the text (e.g. "<|endoftext|>") become one id
   *  unless allowSpecial is false. */
  encode(text: string, opts: { allowSpecial?: boolean } = {}): number[] {
    const allowSpecial = opts.allowSpecial ?? true;
    const out: number[] = [];
    const pieces = allowSpecial ? this.splitSpecial(text) : [{ text, special: false }];
    for (const p of pieces) {
      if (p.special) {
        out.push(this.specialTokens[p.text]);
        continue;
      }
      for (const pt of this.pretokenize(p.text)) {
        const ids = this._encodePretoken(pt);
        for (const id of ids) out.push(id);
      }
    }
    return out;
  }

  /** Like encode, but returns every pre-token with its bytes and its merge sequence,
   *  for the tokenization page. */
  encodeWithSteps(text: string): EncodeSteps {
    const pretokens: PretokenTrace[] = [];
    const ids: number[] = [];
    for (const p of this.splitSpecial(text)) {
      if (p.special) {
        const sid = this.specialTokens[p.text];
        pretokens.push({ text: p.text, bytes: [], steps: [], ids: [sid], special: true });
        ids.push(sid);
        continue;
      }
      for (const pt of this.pretokenize(p.text)) {
        const bytes = utf8Bytes(pt);
        const r = this.bpeSteps(bytes);
        pretokens.push({ text: pt, bytes, steps: r.steps, ids: r.ids, special: false });
        for (const id of r.ids) ids.push(id);
      }
    }
    return { pretokens, ids };
  }

  /** Token ids -> text. Invalid UTF-8 (e.g. half a character) becomes U+FFFD. */
  decode(ids: number[]): string {
    let len = 0;
    for (const id of ids) len += this._bytes[id].length;
    const buf = new Uint8Array(len);
    let o = 0;
    for (const id of ids) {
      buf.set(this._bytes[id], o);
      o += this._bytes[id].length;
    }
    // ignoreBOM: keep a leading U+FEFF like Python's bytes.decode does (the default strips it)
    return new TextDecoder('utf-8', { ignoreBOM: true }).decode(buf);
  }

  /** The raw UTF-8 bytes a token stands for. */
  tokenBytes(id: number): Uint8Array {
    return this._bytes[id];
  }

  isSpecial(id: number): boolean {
    return this._specialById.has(id);
  }

  /** Readable form of one token: " the", "我", "<0xE7><0x88>" for partial characters,
   *  "<|endoftext|>" for the special token. */
  tokenToDisplay(id: number, opts: { visibleWhitespace?: boolean } = {}): string {
    const special = this._specialById.get(id);
    if (special !== undefined) return special;
    const s = bytesToDisplay(this._bytes[id]);
    return opts.visibleWhitespace ? visibleWhitespace(s) : s;
  }
}

export async function loadTokenizer(fetchFn: FetchLike, baseUrl: string): Promise<BPETokenizer> {
  const url = baseUrl.replace(/\/+$/, '') + '/tokenizer.json';
  const res = await fetchFn(url);
  if (!res.ok) throw new Error(`could not load ${url} (${res.status ?? 'error'})`);
  return new BPETokenizer((await res.json()) as TokenizerJSON);
}

// ------------------------------------------------------------------------------------------------
// Training (for the merge-by-merge stepper on the tokenization page)
// ------------------------------------------------------------------------------------------------

export type TrainStep = {
  rank: number;
  pair: [number, number];
  newId: number;
  count: number; // how often the pair occurred when it was merged
  bytes: number[]; // bytes of the new token
  display: string; // readable form of the new token
  tokenCount: number; // total tokens in the training text after this merge
};

export type TrainResult = {
  merges: [number, number][];
  steps: TrainStep[];
  initialTokenCount: number; // = number of UTF-8 bytes
};

/**
 * Learn up to `numMerges` merges from `text`, exactly like pipeline/tinygpt/bpe.py train_bpe:
 * count adjacent pairs over all pre-tokens, merge the most frequent pair (ties: smaller (a, b)),
 * repeat. Recounts every step, which is fine for the few kilobytes a learner types.
 */
export function trainBPE(text: string, numMerges: number, pattern: string): TrainResult {
  const re = new RegExp(pattern, 'gu');
  const counts = new Map<string, number>();
  for (const pt of text.match(re) ?? []) counts.set(pt, (counts.get(pt) ?? 0) + 1);
  let words: number[][] = [];
  const freqs: number[] = [];
  for (const [pt, n] of counts) {
    words.push(utf8Bytes(pt));
    freqs.push(n);
  }
  const bytesOf: number[][] = [];
  for (let b = 0; b < 256; b++) bytesOf.push([b]);
  let tokenCount = words.reduce((s, w, i) => s + w.length * freqs[i], 0);
  const initialTokenCount = tokenCount;
  const merges: [number, number][] = [];
  const steps: TrainStep[] = [];
  for (let m = 0; m < numMerges; m++) {
    const pc = new Map<number, number>();
    words.forEach((w, wi) => {
      for (let i = 0; i < w.length - 1; i++) {
        const key = w[i] * 65536 + w[i + 1];
        pc.set(key, (pc.get(key) ?? 0) + freqs[wi]);
      }
    });
    let bestKey = -1;
    let bestN = 0;
    for (const [key, n] of pc) {
      if (n > bestN || (n === bestN && key < bestKey)) {
        bestKey = key;
        bestN = n;
      }
    }
    if (bestN <= 0) break;
    const a = Math.floor(bestKey / 65536);
    const b = bestKey % 65536;
    const newId = 256 + m;
    merges.push([a, b]);
    bytesOf.push(bytesOf[a].concat(bytesOf[b]));
    words = words.map((w) => mergeAll(w, a, b, newId));
    // recount (a pair like "aa" in "aaa" occurs twice but merges once)
    tokenCount = words.reduce((s, w, i) => s + w.length * freqs[i], 0);
    steps.push({
      rank: m,
      pair: [a, b],
      newId,
      count: bestN,
      bytes: bytesOf[newId],
      display: bytesToDisplay(bytesOf[newId]),
      tokenCount,
    });
  }
  return { merges, steps, initialTokenCount };
}
