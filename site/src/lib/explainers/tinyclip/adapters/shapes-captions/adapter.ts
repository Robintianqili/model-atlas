// tinyclip: local input adapter 'shapes-captions' (InputKinds drawing + text). The learner's raw input is a drawing in the
// framed pad (DrawPad strokes, 0..1) and up to three captions. Two tensors come out:
//   image  [1, 28, 28]  render(): the frame plus the strokes through drawing28 (TinyCNN's preprocessStrokes)
//   text   [K, L] i32   the K captions that have a known word (K ≥ 1), L = the longest (≤ 8 words), 0 pads; no caption
//                       at all gives one row [0], "a caption of no words"
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { EDGES, MAX_WORDS, PRESETS, VOCAB, render, strokesFor, tokenize, type Fixed, type Pt } from '../../scenes.ts';

export type ClipRaw = { strokes?: ({ points: Pt[]; size?: number } | Pt[])[]; captions?: string[]; preset?: string };
const pts = (s: { points: Pt[] } | Pt[]) => (Array.isArray(s) ? s : (s?.points ?? []));

/** A fixed drawing as the learner's input: DrawPad strokes (so the pad shows it) and its captions in three slots. */
export function rawOf(f: Fixed): ClipRaw {
	return { strokes: strokesFor(f).map((points) => ({ points, size: 0.045 })), captions: [...f.captions, '', ''].slice(0, 3), preset: f.id };
}
/** The captions the model reads (those with at least one known word), and their ids. */
export function captionsOf(raw: unknown): { text: string; ids: number[] }[] {
	const caps = ((raw as ClipRaw | null)?.captions ?? []).slice(0, 3);
	return caps.map((c) => ({ text: (c ?? '').trim(), ids: tokenize(c ?? '').ids })).filter((c) => c.ids.length > 0);
}
export function textTensor(rows: number[][]): IntTensor {
	const K = Math.max(rows.length, 1), L = Math.max(1, ...rows.map((r) => r.length));
	const data = new Int32Array(K * L);
	rows.forEach((r, k) => data.set(r, k * L));
	return { shape: [K, L], data, dtype: 'i32' };
}
export const imageOf = (raw: unknown) => ({ shape: [1, 28, 28], data: render(((raw as ClipRaw | null)?.strokes ?? []).map(pts)) });

const NOTE = {
	blank: { en: 'Nothing drawn yet: the model sees only the frame.', zh: '还没有画：模型只看到画框。' },
	none: { en: 'No caption has a word the model knows, so the text tower reads a caption of no words.', zh: '没有一句描述含有模型认识的词，所以文本塔读到的是一句“没有词”的描述。' }
};

const adapter: InputAdapter<ClipRaw> = {
	id: 'shapes-captions',
	kind: 'drawing',
	preprocess(raw): Preprocessed {
		const image = imageOf(raw);
		const caps = captionsOf(raw);
		const text = textTensor(caps.map((c) => c.ids));
		const skipped = [...new Set((raw?.captions ?? []).flatMap((c) => tokenize(c ?? '').words.filter((w) => w.id < 0).map((w) => w.w)))];
		const notes: I18n[] = [];
		if (!(raw?.strokes ?? []).some((s) => pts(s).length)) notes.push(NOTE.blank);
		if (!caps.length) notes.push(NOTE.none);
		if (skipped.length) notes.push({ en: `Skipped (not among the model’s ${VOCAB.length - 1} words): ${skipped.join(', ')}.`, zh: `跳过了（不在模型认识的 ${VOCAB.length - 1} 个词里）：${skipped.join('、')}。` });
		if ((raw?.captions ?? []).some((c) => tokenize(c ?? '').cut)) notes.push({ en: `Only the first ${MAX_WORDS} known words of a caption are read.`, zh: `一句描述只读前 ${MAX_WORDS} 个认识的词。` });
		return {
			tensors: { image, text },
			steps: [{ id: 'image', label: { en: 'What the image tower sees: 28×28 numbers', zh: '图像塔看到的：28×28 个数' }, tensor: image }],
			empty: false,
			notes,
			tokens: tokenize(caps[0]?.text ?? '').words.filter((w) => w.id > 0).map((w) => w.w),
			vocab: VOCAB
		};
	},
	edgeFixtures: () => [
		...EDGES.map((f) => ({ id: f.id, label: f.label, raw: rawOf(f) })),
		{ id: 'no-words', label: { en: 'captions with no known word', zh: '描述里没有认识的词' }, raw: { ...rawOf(PRESETS[0]), captions: ['blue', 'hello there', ''], preset: undefined } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: { en: 'The image tower never sees your strokes: it sees {n|int} numbers in a {shape} grid, 0 for paper and 1 for ink, with the frame around them.', zh: '图像塔看不到你的笔画，它看到的是 {shape} 网格里的 {n|int} 个数：0 是纸，1 是墨，四周还有画框。' },
			look: { en: 'The bright ring is the frame; it keeps the size and place of your shape.', zh: '亮的一圈是画框；它保留了你画的形状的大小和位置。' }
		}
	}
};

export default adapter;
