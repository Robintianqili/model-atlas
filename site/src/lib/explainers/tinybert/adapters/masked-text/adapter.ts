// tinybert: local input adapter 'masked-text' (InputKind 'text'). The learner's text, as TinyGPT's tokenizer cuts it,
// with one token hidden. Raw input: { text, ids, mask, picked } where ids are the text's token ids (no special tokens),
// mask the index (counting from 0) of the hidden one, and picked marks a token the learner chose. The tensors:
//   ids      [CLS] sentence A [SEP] (B [SEP]) with the hidden token as [MASK]   (TinyBERT's input, i32 [T])
//   segment  0 for [CLS], A and its [SEP]; 1 for B and the last [SEP]            (TinyBERT's input, i32 [T])
//   input    the text's ids, nothing hidden                                    (TinyGPT's input, the 'gpt' source)
// The split rule and the token flags are pipeline/tinybert/data.py's (export.py writes spec.tokenFlags and spec.vocab).
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';

export type MaskedRaw = { text?: string; ids?: number[]; mask?: number; picked?: boolean } | null | undefined;
export type Spec = { vocab?: string[]; tokenFlags?: string; context?: number } | null | undefined;
export const MASK = 1024;
export const CLS = 1025;
export const SEP = 1026;
const MAXC = 61; // 64 positions minus [CLS] and two [SEP]

const flags = (spec: Spec, id: number) => Number(spec?.tokenFlags?.[id] ?? 0);
const ends = (spec: Spec, id: number) => (flags(spec, id) & 1) !== 0;
const startsSpace = (spec: Spec, id: number) => (flags(spec, id) & 2) !== 0;
const blank = (spec: Spec, id: number) => (flags(spec, id) & 4) !== 0;
export const word = (spec: Spec, id: number) => /\p{L}/u.test(spec?.vocab?.[id] ?? '');
export const vocabOf = (spec: Spec) => (id: number) => spec?.vocab?.[id] ?? `#${id}`;

/** The text's token ids (at most 61: TinyBERT keeps the first ones). */
export function contentOf(raw: MaskedRaw): number[] {
	const ids = raw && Array.isArray(raw.ids) ? raw.ids.map(Number) : [];
	return ids.slice(0, MAXC);
}

/** The last token of sentence A, or -1 for one segment (data.py split_at). */
export function splitAt(c: number[], spec: Spec): number {
	for (let j = 0; j < c.length - 1; j++) if (ends(spec, c[j]) && startsSpace(spec, c[j + 1]) && c.slice(j + 1).some((t) => !blank(spec, t))) return j;
	return -1;
}

/** The word token nearest the middle (the token hidden when the learner has not picked one). */
export function autoMask(c: number[], spec: Spec): number {
	let best = -1;
	for (let i = 0; i < c.length; i++) if (word(spec, c[i]) && (best < 0 || Math.abs(i - (c.length - 1) / 2) < Math.abs(best - (c.length - 1) / 2))) best = i;
	return best >= 0 ? best : c.length ? 0 : -1;
}

export function maskOf(raw: MaskedRaw, spec: Spec): number {
	const c = contentOf(raw);
	const m = raw?.mask;
	return typeof m === 'number' && Number.isInteger(m) && m >= 0 && m < c.length ? m : autoMask(c, spec);
}

/** BERT's layout: ids, segments, the position of every text token, and the split. */
export function layout(c: number[], mask: number, spec: Spec) {
	const j = splitAt(c, spec);
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

const adapter: InputAdapter<MaskedRaw> = {
	id: 'masked-text',
	kind: 'text',
	preprocess(raw, spec0): Preprocessed {
		const spec = spec0 as Spec;
		const all = raw && Array.isArray(raw.ids) ? raw.ids.length : 0;
		const c = contentOf(raw);
		const vocab = vocabOf(spec);
		if (!c.length)
			return { tensors: { ids: i32([]), segment: i32([]), input: i32([]) }, steps: [], empty: true, notes: [{ en: 'There is no text yet. Type a sentence and every step fills with your tokens.', zh: '还没有文字。输入一句话，每一步都会填上你的词元。' }] };
		const L = layout(c, maskOf(raw, spec), spec);
		const notes: I18n[] = [];
		if (all > MAXC) notes.push({ en: `TinyBERT reads at most ${MAXC} tokens of text (64 positions with [CLS] and [SEP]), so the last ${all - MAXC} were dropped.`, zh: `TinyBERT 最多读 ${MAXC} 个文字词元（加上 [CLS] 和 [SEP] 共 64 个位置），所以最后 ${all - MAXC} 个被丢掉了。` });
		return {
			tensors: { ids: i32(L.ids), segment: i32(L.seg), input: i32(c) },
			steps: [
				{ id: 'ids', label: { en: 'Token ids', zh: '词元 id' }, tensor: { shape: [L.ids.length], data: Float32Array.from(L.ids) }, note: { en: 'One number per token: its row in the token table.', zh: '每个词元一个数：它在词元表里的行号。' } },
				{ id: 'segment', label: { en: 'Segment ids', zh: '段 id' }, tensor: { shape: [L.seg.length], data: Float32Array.from(L.seg) }, note: { en: '0 for the first sentence, 1 for the rest.', zh: '第一句是 0，其余是 1。' } }
			],
			empty: false,
			notes,
			tokens: L.ids.map(vocab),
			vocab
		};
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { items?: { id: string; en: string; zh: string; text: string; ids: number[]; mask: number }[] } | undefined;
		return (f?.items ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: { text: p.text, ids: p.ids, mask: p.mask } }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', ids: [] } },
		{ id: 'one', label: { en: 'one token', zh: '一个词元' }, raw: { text: 'a', ids: [97] } },
		{ id: 'long', label: { en: '70 token ids (more than fit)', zh: '70 个词元 id（超出容量）' }, raw: { ids: Array.from({ length: 70 }, (_, i) => 97 + (i % 26)) } }
	],
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: { en: 'Your text becomes {n|int} tokens, each one id number.', zh: '你的文字变成 {n|int} 个词元，每个词元是一个 id 数字。' },
			look: { en: 'the [MASK] token, and the [CLS] and [SEP] tokens around your text.', zh: '[MASK] 词元，以及文字两边的 [CLS] 和 [SEP]。' }
		}
	}
};

export default adapter;
