// crnn-ocr: local input adapter 'word-strokes'. The learner writes a number on the kit's square drawing pad (or picks a
// preset, written by glyphs.ts word() with a fixed seed and shown on the pad); glyphs.ts render() turns the strokes into
// the model's [1, 16, 64] strip. The raw input also carries what the learner says they wrote (`text`), which the
// lesson compares with what the model reads.
//   raw: { strokes: { points: { x, y }[] }[], text?: string, preset?: string } | null
import type { I18n, InputAdapter, Tensor } from '../../../../atlas/types/index.ts';
import { createRng, fnv1a32 } from '../../../../atlas/core/rng.ts';
import { H, LW, W, render, toPad, word, type Pt } from '../../glyphs.ts';

export type WordRaw = { strokes: { points: Pt[]; size?: number }[]; text?: string; preset?: string } | null | undefined;
export const PRESETS: { id: string; text: string; label: I18n }[] = [
	{ id: '2048', text: '2048', label: { en: '2048', zh: '2048' } },
	{ id: '1100', text: '1100', label: { en: '1100 (doubled digits)', zh: '1100（重复的数字）' } },
	{ id: '307', text: '307', label: { en: '307', zh: '307' } },
	{ id: '58', text: '58', label: { en: '58', zh: '58' } },
	{ id: '196473', text: '196473', label: { en: '196473 (six digits)', zh: '196473（六位数）' } },
	{ id: 'hex', text: 'HEX', label: { en: 'HEX (letters)', zh: 'HEX（字母）' } }
];
/** the pad strokes of a preset (pen width 0.03 of the pad, for the pad's own drawing) */
export function presetRaw(id: string): WordRaw {
	const p = PRESETS.find((q) => q.id === id);
	if (!p) throw new Error(`crnn-ocr: no preset '${id}'`);
	const strokes = toPad(word(p.text, createRng(fnv1a32(`crnn-ocr preset ${id}`)))).map((s) => ({ ...s, size: 0.03 }));
	return { strokes, text: p.text, preset: id };
}
export const strokesOf = (raw: unknown): Pt[][] =>
	raw && typeof raw === 'object' && Array.isArray((raw as { strokes?: unknown }).strokes) ? (raw as { strokes: { points: Pt[] }[] }).strokes.map((s) => s.points ?? []) : [];
/** the text the learner says they wrote: digits only ('' when none) */
export const textOf = (raw: unknown): string => String((raw as { text?: unknown } | null)?.text ?? '').replace(/[^0-9]/g, '');

const NOTE_EMPTY: I18n = {
	en: 'Nothing written yet. Write a number on the pad (or pick a preset) and every step fills with the numbers of your writing.',
	zh: '还没有写字。在画板上写一个数（或选一个预设），每一步都会填上你这次书写的数。'
};
const NOTE: I18n = {
	en: `Your strokes are drawn again with a pen ${LW} pixels wide, cropped to the ink, scaled so the ink is ${H - 3} pixels tall (or the whole number fits ${W - 3} pixels across) and centred, like every training word.`,
	zh: `你的笔画用 ${LW} 像素宽的笔重新画一遍，裁到有墨的范围，缩放到墨迹高 ${H - 3} 像素（或整个数能放进 ${W - 3} 像素宽），再居中，和每个训练用的词一样。`
};

const adapter: InputAdapter<WordRaw> = {
	id: 'word-strokes',
	kind: 'drawing',
	preprocess(raw) {
		const r = render(strokesOf(raw));
		const tensor: Tensor = { shape: [1, H, W], data: r.ink };
		return {
			tensors: { input: tensor },
			steps: [{ id: 'input', label: { en: 'What the model sees: 16 × 64 numbers', zh: '模型看到的：16 × 64 个数' }, tensor, note: NOTE }],
			empty: r.empty,
			...(r.empty ? { notes: [NOTE_EMPTY] } : {})
		};
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: p.label, raw: presetRaw(p.id) })),
	edgeFixtures: () => [
		{ id: 'blank', label: { en: 'blank pad', zh: '空白画板' }, raw: { strokes: [] } },
		{ id: 'dot', label: { en: 'a single dot', zh: '一个点' }, raw: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] } },
		{ id: 'dash', label: { en: 'one flat line', zh: '一条横线' }, raw: { strokes: [{ points: [{ x: 0.1, y: 0.5 }, { x: 0.9, y: 0.5 }] }] } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: { en: 'The model never sees your strokes. It sees {n|int} numbers in a {shape} strip: 0 is empty paper and 1 is full ink.', zh: '模型看不到你的笔画，它只看到 {shape} 长条里的 {n|int} 个数：0 是空白的纸，1 是满满的墨。' },
			how: { en: 'Your writing is cropped to the ink, scaled to the strip’s height and centred, like every training word.', zh: '你写的字先裁到有墨的地方，缩放到长条的高度并居中，和每个训练用的词一样。' },
			look: { en: 'the strongly colored cells: they are your ink.', zh: '颜色浓的格子：那就是你的墨迹。' }
		}
	}
};

export default adapter;
