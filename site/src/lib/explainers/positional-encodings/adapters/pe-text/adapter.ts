// positional-encodings: local input adapter 'pe-text' (the text part copied from self-supervised-objectives' masked-text,
// itself from tinybert; credit: those packages). The learner's text as TinyGPT's tokenizer cuts it, which shuffled order to
// try, and two positions to probe the encodings with. Raw input: { text, ids, shuffle, p, k }:
//   ids      the text's token ids (TinyGPT keeps the last 64, as its own page does)
//   shuffle  the seed of the shuffled order (1, 2, 3, ...; pe.ts shuffleOf)
//   p        a key's position, 0 to 127 (TinyGPT's table has rows 0 to 63 only)
//   k        the distance from that key to the query, 1 to 63: the query sits at position p + k
// The tensor 'input' is TinyGPT's input (the kept ids).
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { tokText } from '../../vocab.ts';

export type PeRaw = { text?: string; ids?: number[]; shuffle?: number; p?: number; k?: number } | null | undefined;
export const CTX = 64;
export const P_MAX = 127;
export const K_MAX = 63;
const int = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : d);

/** The learner's settings, clamped: ids (the kept last 64), cut (how many were dropped), shuffle, p and k. */
export function peOf(input: unknown) {
	const raw = input as PeRaw;
	const all = raw && Array.isArray(raw.ids) ? raw.ids.map(Number).filter((v) => Number.isInteger(v) && v >= 0 && v < 1024) : [];
	return { ids: all.slice(-CTX), cut: Math.max(0, all.length - CTX), shuffle: int(raw?.shuffle, 1, 1e6, 1), p: int(raw?.p, 0, P_MAX, 5), k: int(raw?.k, 1, K_MAX, 3) };
}

const i32 = (a: number[]): IntTensor => ({ shape: [a.length], data: Int32Array.from(a), dtype: 'i32' });
const pk = { shuffle: 1, p: 5, k: 3 };

/** Example sentences (tinybert's presets, in the style of TinyStories; ids from TinyGPT's tokenizer). */
export const PRESETS = [
	{ id: 'dog', label: { en: 'A dog named Max', zh: '一只叫 Max 的狗' }, raw: { text: 'Once upon a time, there was a little dog named Max.', ids: [428, 437, 258, 397, 44, 400, 282, 258, 390, 463, 401, 512, 46], ...pk } },
	{ id: 'ball', label: { en: 'The lost ball', zh: '丢了的球' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], ...pk } },
	{ id: 'park', label: { en: 'At the park', zh: '在公园' }, raw: { text: 'Lily went to the park to play with her friends. She had a lot of fun.', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335, 410, 46, 338, 363, 258, 645, 369, 443, 46], ...pk } },
	{ id: 'box', label: { en: 'A box', zh: '一个盒子' }, raw: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], ...pk } }
];
export const START = PRESETS[0].raw;

const adapter: InputAdapter<PeRaw> = {
	id: 'pe-text',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const s = peOf(raw);
		if (!s.ids.length)
			return { tensors: { input: i32([]) }, steps: [], empty: true, notes: [{ en: 'There is no text yet. Type a sentence and every step fills with your tokens.', zh: '还没有文字。输入一句话，每一步都会填上你的词元。' }] };
		const notes: I18n[] = [];
		if (s.cut) notes.push({ en: `TinyGPT reads at most ${CTX} tokens, so it keeps the last ${CTX} and the first ${s.cut} were dropped.`, zh: `TinyGPT 最多读 ${CTX} 个词元，所以只保留最后 ${CTX} 个，前 ${s.cut} 个被丢掉了。` });
		return {
			tensors: { input: i32(s.ids) },
			steps: [{ id: 'ids', label: { en: 'Token ids', zh: '词元 id' }, tensor: { shape: [s.ids.length], data: Float32Array.from(s.ids) }, note: { en: 'One number per token: its row in the token table.', zh: '每个词元一个数：它在词元表里的行号。' } }],
			empty: false,
			notes,
			tokens: s.ids.map(tokText),
			vocab: tokText
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', ids: [] } },
		{ id: 'one', label: { en: 'one token', zh: '一个词元' }, raw: { text: 'a', ids: [97] } },
		{ id: 'two', label: { en: 'two tokens (one can move)', zh: '两个词元（只有一个能动）' }, raw: { text: 'a dog', ids: [97, 512] } },
		{ id: 'long', label: { en: '70 token ids (more than fit), far positions', zh: '70 个词元 id（超出容量），很远的位置' }, raw: { ids: Array.from({ length: 70 }, (_, i) => 97 + (i % 26)), shuffle: 7, p: P_MAX, k: K_MAX } }
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
