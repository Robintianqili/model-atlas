// llava-toy: local input adapter 'drawing-question' (InputKinds drawing + text). The learner's raw input is a drawing
// (drawing28's raw forms: pad strokes, or a preset's b64) plus a question. Two tensors come out:
//   image  [1, 28, 28]  drawing28's drawingInput(): TinyCNN's preprocessing, so TinyViT sees the 784 numbers it was trained on
//   text   [T] i32      16 <image> placeholders (id 0), then the question's known words (at most 8, in order); unknown
//                       words are skipped; a question with no known word reads as "?" alone, so the language model always
//                       has a last row to answer from. T is 17 to 24. The graph replaces the placeholders' rows by the image tokens.
// tokens names the language model's rows: the 16 image tokens (▦0 to ▦15), then the words.
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { drawingInput, type DrawingRaw } from '../../../../atlas/adapters/drawing28/adapter.ts';

export const CLASSES = ['apple', 'bicycle', 'car', 'clock', 'fish', 'house', 'star', 'sun', 'tree', 'umbrella'];
/** pipeline/llava-toy/model.py VOCAB: id 0 is the <image> placeholder */
export const VOCAB = ['<image>', 'what', 'is', 'this', 'it', 'a', 'an', '?', 'yes', 'no', ...CLASSES];
export const N_IMG = 16;
export const MAX_WORDS = 8;
export const START_Q = 'what is this ?';

export type QRaw = { strokes?: unknown[]; b64?: string; question?: string; preset?: string };

/** The question's words: known ones get their id, the rest -1; only the first MAX_WORDS known ones are read. */
export function tokenize(q: string): { words: { w: string; id: number }[]; ids: number[]; cut: boolean } {
	const words = (q.toLowerCase().match(/[a-z]+|\?/g) ?? []).map((w) => ({ w, id: VOCAB.indexOf(w) > 0 ? VOCAB.indexOf(w) : -1 }));
	const known = words.filter((x) => x.id > 0).map((x) => x.id);
	return { words, ids: known.slice(0, MAX_WORDS), cut: known.length > MAX_WORDS };
}
/** The ids the model reads: the known words, or "?" alone when there is none. */
export const idsOf = (raw: unknown) => {
	const ids = tokenize(questionOf(raw)).ids;
	return ids.length ? ids : [VOCAB.indexOf('?')];
};
export const questionOf = (raw: unknown) => String((raw as QRaw | null)?.question ?? '');
export const textTensor = (ids: number[]): IntTensor => ({ shape: [N_IMG + ids.length], data: Int32Array.from([...new Array(N_IMG).fill(0), ...ids]), dtype: 'i32' });
/** The names of the language model's rows: ▦0 ... ▦15, then the words read. */
export const rowNames = (ids: number[]) => [...Array.from({ length: N_IMG }, (_, k) => `▦${k}`), ...ids.map((i) => VOCAB[i])];
const drawingOf = (raw: unknown): DrawingRaw => {
	const r = (raw ?? {}) as QRaw;
	return typeof r.b64 === 'string' ? { b64: r.b64 } : { strokes: (r.strokes ?? []) as never[] };
};

const NOTE = {
	blank: { en: 'Nothing drawn yet: TinyViT reads an empty grid, and the model still answers.', zh: '还没有画：TinyViT 读的是一张空网格，模型照样会回答。' },
	none: { en: 'No word of the question is one the model knows, so it reads “?” alone.', zh: '问题里没有一个词是模型认识的，所以它只读一个“?”。' }
};

const adapter: InputAdapter<QRaw> = {
	id: 'drawing-question',
	kind: 'drawing',
	preprocess(raw): Preprocessed {
		const d = drawingInput(drawingOf(raw));
		const image = { shape: [1, 28, 28], data: d.input };
		const tok = tokenize(questionOf(raw));
		const ids = idsOf(raw);
		const notes: I18n[] = [];
		if (d.empty) notes.push(NOTE.blank);
		if (!tok.ids.length) notes.push(NOTE.none);
		const skipped = [...new Set(tok.words.filter((w) => w.id < 0).map((w) => w.w))];
		if (skipped.length) notes.push({ en: `Skipped (not among the model’s ${VOCAB.length - 1} words): ${skipped.join(', ')}.`, zh: `跳过了（不在模型认识的 ${VOCAB.length - 1} 个词里）：${skipped.join('、')}。` });
		if (tok.cut) notes.push({ en: `Only the first ${MAX_WORDS} known words are read.`, zh: `只读前 ${MAX_WORDS} 个认识的词。` });
		return {
			tensors: { image, text: textTensor(ids) },
			steps: [
				{ id: 'image', label: { en: 'What TinyViT sees: 28×28 numbers', zh: 'TinyViT 看到的：28×28 个数' }, tensor: image },
				{ id: 'text', label: { en: 'What the language model reads: 16 <image> placeholders (id 0), then the word ids', zh: '语言模型读到的：16 个 <image> 占位符（id 0），然后是词的 id' }, note: { en: ids.map((i) => VOCAB[i] + ' = ' + i).join(' · '), zh: ids.map((i) => VOCAB[i] + ' = ' + i).join(' · ') } }
			],
			empty: false,
			notes,
			tokens: rowNames(ids),
			vocab: VOCAB
		};
	},
	/** TinyCNN's preset drawings (one per class and held-out class), each with the start question */
	presets(assets) {
		const file = (assets['presets.json'] ?? assets.presets) as { classes?: { id: string; en: string; zh?: string; items: { b64: string }[] }[]; heldOut?: { id: string; en: string; zh?: string; items: { b64: string }[] }[] } | undefined;
		const out: { id: string; label: I18n; raw: QRaw }[] = [];
		for (const c of file?.classes ?? []) if (c.items?.[0]) out.push({ id: c.id, label: { en: c.en, zh: c.zh ?? c.en }, raw: { b64: c.items[0].b64, question: START_Q, preset: c.id } });
		for (const c of file?.heldOut ?? [])
			if (c.items?.[0]) out.push({ id: `heldout-${c.id}`, label: { en: `${c.en} (not one of its words)`, zh: `${c.zh ?? c.en}（不在它认识的词里）` }, raw: { b64: c.items[0].b64, question: START_Q, preset: `heldout-${c.id}` } });
		return out;
	},
	edgeFixtures: () => [
		{ id: 'blank', label: { en: 'blank pad', zh: '空白画板' }, raw: { strokes: [], question: START_Q } },
		{ id: 'dot', label: { en: 'a single dot, one word', zh: '一个点，一个词' }, raw: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }], question: '?' } },
		{ id: 'no-words', label: { en: 'a question with no known word', zh: '一个没有认识的词的问题' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.8 }] }], question: 'hello there' } },
		{ id: 'long', label: { en: 'more than 8 known words', zh: '超过 8 个认识的词' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.5 }, { x: 0.8, y: 0.5 }] }], question: 'is this a car ? what is it ? is it a sun ?' } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: { en: 'The model never sees your strokes. TinyViT gets {n|int} numbers in a {shape} grid: 0 is empty paper and 1 is full ink.', zh: '模型看不到你的笔画。TinyViT 拿到的是 {shape} 网格里的 {n|int} 个数：0 是空白的纸，1 是满满的墨。' },
			look: { en: 'the bright cells: your ink.', zh: '亮的格子：你的墨迹。' }
		}
	}
};

export default adapter;

const P = (pts: number[][]) => ({ points: pts.map(([x, y]) => ({ x, y })), size: 0.055 });
/** The page's start drawing: a house in three strokes (walls, roof, door), on the pad so the learner can change it. */
export const HOUSE = [
	P([[0.27, 0.48], [0.27, 0.84], [0.73, 0.84], [0.73, 0.48], [0.27, 0.48]]),
	P([[0.2, 0.5], [0.5, 0.18], [0.8, 0.5]]),
	P([[0.45, 0.84], [0.45, 0.66], [0.56, 0.66], [0.56, 0.84]])
];
export const START: QRaw = { strokes: HOUSE, question: START_Q };
