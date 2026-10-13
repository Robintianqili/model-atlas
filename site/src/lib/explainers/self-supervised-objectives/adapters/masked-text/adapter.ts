// self-supervised-objectives: local input adapter 'masked-text', copied from tinybert (credit: the tinybert package). The
// learner's text, as TinyGPT's tokenizer cuts it, with one token hidden. Raw input: { text, ids, mask, picked } where ids
// are the text's token ids (no special tokens), mask the index (counting from 0) of the hidden one, and picked marks a
// token the learner chose. The tensors:
//   ids      [CLS] sentence A [SEP] (B [SEP]) with the hidden token as [MASK]   (TinyBERT's input, i32 [T])
//   segment  0 for [CLS], A and its [SEP]; 1 for B and the last [SEP]            (TinyBERT's input, i32 [T])
//   input    the text's ids, nothing hidden                                    (TinyGPT's input)
// Change from tinybert: the vocabulary and token flags come from source_data.ts (a generated copy of TinyBERT's
// spec.json), because this concept package has no spec.json of its own (notes: Kit issues).
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { FLAGS, VOCAB } from '../../source_data.ts';

export type MaskedRaw = { text?: string; ids?: number[]; mask?: number; picked?: boolean } | null | undefined;
export const MASK = 1024;
export const CLS = 1025;
export const SEP = 1026;
const MAXC = 61; // 64 positions minus [CLS] and two [SEP]

const flags = (id: number) => Number(FLAGS[id] ?? 0);
const ends = (id: number) => (flags(id) & 1) !== 0;
const startsSpace = (id: number) => (flags(id) & 2) !== 0;
const blank = (id: number) => (flags(id) & 4) !== 0;
export const word = (id: number) => /\p{L}/u.test(VOCAB[id] ?? '');
export const vocab = (id: number) => VOCAB[id] ?? `#${id}`;

/** The text's token ids (at most 61: TinyBERT keeps the first ones). */
export function contentOf(raw: MaskedRaw): number[] {
	const ids = raw && Array.isArray(raw.ids) ? raw.ids.map(Number) : [];
	return ids.slice(0, MAXC);
}

/** The last token of sentence A, or -1 for one segment (pipeline/tinybert/data.py split_at). */
export function splitAt(c: number[]): number {
	for (let j = 0; j < c.length - 1; j++) if (ends(c[j]) && startsSpace(c[j + 1]) && c.slice(j + 1).some((t) => !blank(t))) return j;
	return -1;
}

/** The word token nearest the middle (the token hidden when the learner has not picked one). */
export function autoMask(c: number[]): number {
	let best = -1;
	for (let i = 0; i < c.length; i++) if (word(c[i]) && (best < 0 || Math.abs(i - (c.length - 1) / 2) < Math.abs(best - (c.length - 1) / 2))) best = i;
	return best >= 0 ? best : c.length ? 0 : -1;
}

export function maskOf(raw: MaskedRaw): number {
	const c = contentOf(raw);
	const m = raw?.mask;
	return typeof m === 'number' && Number.isInteger(m) && m >= 0 && m < c.length ? m : autoMask(c);
}

/** BERT's layout: ids, segments, the position of every text token, and the split. */
export function layout(c: number[], mask: number) {
	const j = splitAt(c);
	const ids = [CLS];
	const seg = [0];
	const where: number[] = [];
	c.forEach((t, i) => {
		if (j >= 0 && i === j + 1) (ids.push(SEP), seg.push(0));
		where.push(ids.length);
		ids.push(i === mask ? MASK : t);
		seg.push(j >= 0 && i > j ? 1 : 0);
	});
	ids.push(SEP);
	seg.push(j >= 0 ? 1 : 0);
	return { ids, seg, where, j };
}

const i32 = (a: number[]): IntTensor => ({ shape: [a.length], data: Int32Array.from(a), dtype: 'i32' });

/** Example sentences (tinybert's presets, written for that page in the style of TinyStories; ids from TinyGPT's tokenizer). */
export const PRESETS = [
	{ id: 'park', label: { en: 'At the park', zh: '在公园' }, raw: { text: 'Lily went to the park to play with her friends. She had a lot of fun.', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335, 410, 46, 338, 363, 258, 645, 369, 443, 46], mask: 4 } },
	{ id: 'ball', label: { en: 'The lost ball', zh: '丢了的球' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], mask: 2 } },
	{ id: 'dog', label: { en: 'A dog named Max', zh: '一只叫 Max 的狗' }, raw: { text: 'Once upon a time, there was a little dog named Max.', ids: [428, 437, 258, 397, 44, 400, 282, 258, 390, 463, 401, 512, 46], mask: 9 } },
	{ id: 'box', label: { en: 'A box', zh: '一个盒子' }, raw: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], mask: 4 } }
];

const adapter: InputAdapter<MaskedRaw> = {
	id: 'masked-text',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const all = raw && Array.isArray(raw.ids) ? raw.ids.length : 0;
		const c = contentOf(raw);
		if (!c.length)
			return { tensors: { ids: i32([]), segment: i32([]), input: i32([]) }, steps: [], empty: true, notes: [{ en: 'There is no text yet. Type a sentence and every step fills with your tokens.', zh: '还没有文字。输入一句话，每一步都会填上你的词元。' }] };
		const L = layout(c, maskOf(raw));
		const notes: I18n[] = [];
		if (all > MAXC) notes.push({ en: `TinyBERT reads at most ${MAXC} tokens of text (64 positions with [CLS] and [SEP]), so the last ${all - MAXC} were dropped.`, zh: `TinyBERT 最多读 ${MAXC} 个文字词元（加上 [CLS] 和 [SEP] 共 64 个位置），所以最后 ${all - MAXC} 个被丢掉了。` });
		return {
			tensors: { ids: i32(L.ids), segment: i32(L.seg), input: i32(c) },
			steps: [{ id: 'ids', label: { en: 'Token ids', zh: '词元 id' }, tensor: { shape: [c.length], data: Float32Array.from(c) }, note: { en: 'One number per token: its row in the token table.', zh: '每个词元一个数：它在词元表里的行号。' } }],
			empty: false,
			notes,
			tokens: c.map(vocab),
			vocab
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', ids: [] } },
		{ id: 'one', label: { en: 'one token', zh: '一个词元' }, raw: { text: 'a', ids: [97] } },
		{ id: 'long', label: { en: '70 token ids (more than fit)', zh: '70 个词元 id（超出容量）' }, raw: { ids: Array.from({ length: 70 }, (_, i) => 97 + (i % 26)) } }
	],
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: { en: 'Your text becomes {n|int} tokens, each one id number.', zh: '你的文字变成 {n|int} 个词元，每个词元是一个 id 数字。' },
			look: { en: 'one chip per token.', zh: '每个词元一个小块。' }
		}
	}
};

export default adapter;
