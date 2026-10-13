// masked-lm: local input adapter 'masked-text' (adapted from tinybert's masked-text, credit: the tinybert package; the
// vocabulary, token flags and example sentences come from source_data.ts, a generated copy of TinyBERT's files). The
// learner's text, as TinyGPT's tokenizer cuts it, with any set of its tokens chosen for the masked-LM puzzle. Raw input:
//   { text, ids, fate, swap, picked, focus, fromDraw }
//   fate[k]   what text token k (counting from 0) became: 0 not chosen, 1 [MASK], 2 a random token (swap[k]), 3 chosen
//             but kept as it is (training's 80/10/10 rule makes all three; a click makes 1 or 0)
//   picked    the learner clicked a token; focus: the token clicked last; fromDraw: the fates came from the 80/10/10 draw
// Without a valid fate list, the word nearest the middle is hidden. The tensors:
//   ids      [CLS] sentence A [SEP] (B [SEP]) with every chosen token as the model reads it   (TinyBERT's input, i32 [T])
//   segment  0 for [CLS], A and its [SEP]; 1 for B and the last [SEP]                         (TinyBERT's input, i32 [T])
//   input    the text's ids, nothing hidden                                                  (TinyGPT's input, i32 [n])
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { FLAGS, STORIES, VOCAB } from '../../source_data.ts';

export type MaskedRaw = { text?: string; ids?: number[]; fate?: number[]; swap?: number[]; picked?: boolean; focus?: number; fromDraw?: boolean } | null | undefined;
export const MASK = 1024;
export const CLS = 1025;
export const SEP = 1026;
export const MAXC = 61; // 64 positions minus [CLS] and two [SEP] (pipeline/tinybert/data.py N_MAX)
export const TEXT_VOCAB = 1023; // text tokens 0..1022 (1023 is <|endoftext|>): the random-token pool

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

/** The word token nearest the middle (the token hidden when the learner has not chosen any). */
export function autoMask(c: number[]): number {
	let best = -1;
	for (let i = 0; i < c.length; i++) if (word(c[i]) && (best < 0 || Math.abs(i - (c.length - 1) / 2) < Math.abs(best - (c.length - 1) / 2))) best = i;
	return best >= 0 ? best : c.length ? 0 : -1;
}

const okSwap = (s: unknown) => typeof s === 'number' && Number.isInteger(s) && s >= 0 && s < TEXT_VOCAB;

/** Per text token: its fate (0-3) and the id the model reads there. */
export function fatesOf(raw: MaskedRaw): { fate: number[]; shown: number[] } {
	const c = contentOf(raw);
	const f0 = raw?.fate;
	const valid = Array.isArray(f0) && f0.length >= c.length && c.every((_, k) => [0, 1, 2, 3].includes(Number(f0[k])));
	const k0 = autoMask(c);
	const fate = c.map((_, k) => (valid ? Number(f0[k]) : k === k0 ? 1 : 0)).map((f, k) => (f === 2 && !okSwap(raw?.swap?.[k]) ? 1 : f));
	const shown = c.map((t, k) => (fate[k] === 1 ? MASK : fate[k] === 2 ? Number(raw?.swap?.[k]) : t));
	return { fate, shown };
}

/** BERT's layout: ids, segments, the position of every text token, and the split (decided on the original text). */
export function layout(c: number[], shown: number[]) {
	const j = splitAt(c);
	const ids = [CLS];
	const seg = [0];
	const where: number[] = [];
	c.forEach((_, i) => {
		if (j >= 0 && i === j + 1) (ids.push(SEP), seg.push(0));
		where.push(ids.length);
		ids.push(shown[i]);
		seg.push(j >= 0 && i > j ? 1 : 0);
	});
	ids.push(SEP);
	seg.push(j >= 0 ? 1 : 0);
	return { ids, seg, where, j };
}

/** The raw input after a click on text token k: a chosen token goes back to itself, any other one becomes [MASK]. */
export function toggled(raw: MaskedRaw, k: number): MaskedRaw {
	const { fate } = fatesOf(raw);
	const next = fate.map((f, i) => (i === k ? (f > 0 ? 0 : 1) : f));
	return { ...(raw ?? {}), fate: next, swap: (raw?.swap ?? []).slice(0, next.length), picked: true, focus: k, fromDraw: false };
}

const i32 = (a: number[]): IntTensor => ({ shape: [a.length], data: Int32Array.from(a), dtype: 'i32' });
const oneHidden = (ids: number[], mask: number) => ids.map((_, i) => (i === mask ? 1 : 0));

/** Example sentences: tinybert's presets (written for that page in the style of TinyStories), the same token hidden. */
export const PRESETS = STORIES.map((s) => ({ id: s.id, label: { en: s.en, zh: s.zh }, raw: { text: s.text, ids: s.ids, fate: oneHidden(s.ids, s.mask) } as MaskedRaw }));

const adapter: InputAdapter<MaskedRaw> = {
	id: 'masked-text',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const all = raw && Array.isArray(raw.ids) ? raw.ids.length : 0;
		const c = contentOf(raw);
		if (!c.length)
			return { tensors: { ids: i32([]), segment: i32([]), input: i32([]) }, steps: [], empty: true, notes: [{ en: 'There is no text yet. Type a sentence and every step fills with your tokens.', zh: '还没有文字。输入一句话，每一步都会填上你的词元。' }] };
		const L = layout(c, fatesOf(raw).shown);
		const notes: I18n[] = [];
		if (all > MAXC) notes.push({ en: `TinyBERT reads at most ${MAXC} tokens of text (64 positions with [CLS] and [SEP]), so the last ${all - MAXC} were dropped.`, zh: `TinyBERT 最多读 ${MAXC} 个文字词元（加上 [CLS] 和 [SEP] 共 64 个位置），所以最后 ${all - MAXC} 个被丢掉了。` });
		return {
			tensors: { ids: i32(L.ids), segment: i32(L.seg), input: i32(c) },
			steps: [{ id: 'ids', label: { en: 'Token ids', zh: '词元 id' }, tensor: { shape: [L.ids.length], data: Float32Array.from(L.ids) }, note: { en: 'One number per token, as TinyBERT reads them: [CLS], your text with the chosen tokens replaced, [SEP].', zh: '每个词元一个数，按 TinyBERT 读到的样子：[CLS]、替换了被选中词元的你的文字、[SEP]。' } }],
			empty: false,
			notes,
			tokens: L.ids.map(vocab),
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
