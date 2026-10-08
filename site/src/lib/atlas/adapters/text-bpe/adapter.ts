// text-bpe (design §4.9; catalog InputKind 'text'): text -> the token ids of a byte-level BPE model, as an i32 tensor [T].
// The tokenizer is TinyGPT's own (src/lib/models/bpe.ts, which mirrors pipeline/tinygpt/bpe.py and is parity-checked);
// like drawing28's import of models/tinycnn.ts, this is the one import outside atlas/ this adapter needs (kit issue:
// the AK503 dependency scan must allow it).
//
// Raw inputs (TextBpeRaw):
//   { text, ids }   the normal form: Capture and presets() tokenize with the model's tokenizer.json, so the ids that run are
//                   exactly the ids shown. `ids` wins when both are given.
//   { ids } | number[]  token ids only (lesson fixtures, the edge fixtures below).
//   string | { text }   needs a tokenizer: options.tokenizer (a tokenizer.json object or a BPETokenizer). preprocess() is
//                   synchronous and cannot fetch, so without one it throws a clear error.
// Like tinygpt.ts, an input longer than the context window (spec.dims.T.max, else spec.context) keeps its LAST tokens.
import type { I18n } from '../../../i18n/types.ts';
import type { IntTensor, Tensor } from '../../types/core.ts';
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';
import { BPETokenizer, type TokenizerJSON } from '../../../models/bpe.ts';

export type TextBpeRaw = null | undefined | string | number[] | { text?: string; ids?: number[] };

const cache = new WeakMap<object, BPETokenizer>();

/** A BPETokenizer from options.tokenizer (tokenizer.json content or a tokenizer), memoized per object. */
export function tokenizerOf(t: unknown): BPETokenizer | null {
	if (!t || typeof t !== 'object') return null;
	if (t instanceof BPETokenizer) return t;
	const hit = cache.get(t);
	if (hit) return hit;
	const j = t as TokenizerJSON;
	if (!Array.isArray(j.merges) || !j.pretokenize) return null;
	const tok = new BPETokenizer(j);
	cache.set(t, tok);
	return tok;
}

/** The normal raw form for a text: { text, ids } with the model's own tokenizer. */
export function textInput(tokenizer: unknown, text: string): { text: string; ids: number[] } {
	const tok = tokenizerOf(tokenizer);
	if (!tok) throw new Error('text-bpe: textInput needs a tokenizer (the model’s tokenizer.json)');
	return { text, ids: tok.encode(text) };
}

function contextOf(spec: unknown): number {
	const s = spec as { dims?: Record<string, { max?: number }>; context?: number } | null;
	const m = s?.dims?.T?.max ?? s?.context;
	return typeof m === 'number' && m > 0 ? m : Infinity;
}

function vocabOf(spec: unknown): number {
	const v = (spec as { vocabSize?: number } | null)?.vocabSize;
	return typeof v === 'number' && v > 0 ? v : Infinity;
}

/** The ids a raw input stands for, before truncation. */
export function rawIds(raw: TextBpeRaw, opts: Record<string, unknown> = {}): { ids: number[]; text?: string } {
	if (raw === null || raw === undefined) return { ids: [] };
	if (Array.isArray(raw)) return { ids: raw.map(Number) };
	const text = typeof raw === 'string' ? raw : raw.text;
	if (typeof raw === 'object' && Array.isArray(raw.ids)) return { ids: raw.ids.map(Number), text };
	if (text === undefined || text === '') return { ids: [], text };
	const tok = tokenizerOf(opts.tokenizer);
	if (!tok) throw new Error('text-bpe: raw text needs token ids ({ text, ids }) or options.tokenizer (the model’s tokenizer.json)');
	return { ids: tok.encode(text), text };
}

export const textBpe: InputAdapter<TextBpeRaw> = {
	id: 'text-bpe',
	kind: 'text',
	preprocess(raw, spec, opts) {
		const { ids: all } = rawIds(raw, opts ?? {});
		const V = vocabOf(spec);
		const bad = all.find((id) => !Number.isInteger(id) || id < 0 || id >= V);
		if (bad !== undefined) throw new Error(`text-bpe: token id ${bad} is outside the vocabulary (0..${V - 1})`);
		const C = contextOf(spec);
		const ids = all.length > C ? all.slice(all.length - C) : all;
		const input: IntTensor = { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' };
		const shown: Tensor = { shape: [ids.length], data: Float32Array.from(ids) };
		const notes: I18n[] = [];
		const dropped = all.length - ids.length;
		if (dropped > 0)
			notes.push({
				en: `Only the last ${ids.length} tokens fit into the context window, so the first ${dropped} were dropped, as the model itself does.`,
				zh: `上下文窗口只装得下最后 ${ids.length} 个词元，所以前面 ${dropped} 个被丢掉了，模型本身也是这样做的。`
			});
		if (ids.length === 0) notes.push({ en: 'There is no text yet. Type something and every step fills with your tokens.', zh: '还没有文字。输入一些内容，每一步都会填上你的词元。' });
		const out: Preprocessed = {
			tensors: { input },
			steps: [{ id: 'ids', label: { en: 'Token ids', zh: '词元 id' }, tensor: shown, note: { en: 'One number per token: its row in the embedding table.', zh: '每个词元一个数：它在嵌入表里的行号。' } }],
			empty: ids.length === 0
		};
		if (notes.length) out.notes = notes;
		// token text for the views (types revision 1b: Preprocessed.tokens / vocab), when the model's tokenizer is given
		const tok = tokenizerOf(opts?.tokenizer);
		if (tok) {
			const text = (id: number) => tok.tokenToDisplay(id);
			out.tokens = ids.map(text);
			out.vocab = text;
		}
		return out;
	},
	edgeFixtures() {
		// ids only: they hold for any byte-level BPE whose ids 0-255 are the raw bytes (TinyGPT's tokenizer.json says so);
		// 97 is the byte 'a', a token by itself in every such tokenizer. A package with another tokenizer adds its own fixtures.
		return [
			{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', ids: [] } },
			{ id: 'one', label: { en: 'one token', zh: '一个词元' }, raw: { text: 'a', ids: [97] } },
			{ id: 'long', label: { en: '70 token ids (more than a 64-token context)', zh: '70 个词元 id（超过 64 的上下文）' }, raw: { ids: Array.from({ length: 70 }, (_, i) => 97 + (i % 26)) } }
		];
	},
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: {
				en: 'The model never sees letters. Your text is cut into {n|int} tokens, and each token becomes one id number.',
				zh: '模型看不到字母。你的文字被切成 {n|int} 个词元，每个词元变成一个 id 数字。'
			},
			how: {
				en: 'Frequent pieces such as “ the” are one token each; rare words are split into smaller pieces.',
				zh: '常见的片段（比如 “ the”）各算一个词元；少见的词会被切成更小的片段。'
			},
			look: { en: 'how many tokens your text became, and which words were split.', zh: '你的文字变成了多少个词元，哪些词被切开了。' }
		}
	}
};

export default textBpe;
