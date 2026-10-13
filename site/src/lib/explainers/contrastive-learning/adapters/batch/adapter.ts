// contrastive-learning: local input adapter 'batch'. The learner's raw input is a batch of 2 to 8 pairs, each a drawing
// (strokes in the framed unit square, as TinyCLIP's pad) and its caption, plus the temperature τ of the loss:
//   { pairs: [{ strokes, caption, from? }], tau }     from: the id of the fixed pair it came from (absent: the learner's own)
// One tensor comes out: images [N, 28, 28], every drawing through render() (the frame plus the strokes, TinyCNN's
// preprocessStrokes), as TinyCLIP's image tower reads them. The captions are read as words by batch.ts (tokenize()).
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { EDGES, PARTNERS, PRESETS, VOCAB, render, strokesFor, tokenize, type Fixed, type Pt } from '../../scenes.ts';

export type PairRaw = { strokes: Pt[][]; caption: string; from?: string };
export type BatchRaw = { pairs: PairRaw[]; tau: number; preset?: string };

export const MIN_PAIRS = 2;
export const MAX_PAIRS = 8;
/** the temperature slider: τ from 0.01 to 2, on a log scale; the page starts at 0.2 (CLIP starts training at 0.07) */
export const TAU_RANGE: [number, number] = [0.01, 2];
export const TAU_START = 0.2;
export const clampTau = (t: unknown) => (typeof t === 'number' && Number.isFinite(t) ? Math.min(TAU_RANGE[1], Math.max(TAU_RANGE[0], t)) : TAU_START);

/** A fixed drawing (scenes.ts) as one pair of the batch, with its first caption. */
export const pairOf = (f: Fixed, caption = f.captions[0]): PairRaw => ({ strokes: strokesFor(f), caption, from: f.id });
const byId = (id: string) => [...PRESETS, ...PARTNERS, ...EXTRA].find((f) => f.id === id)!;

/** Two more fixed drawings: a second circle captioned just "a circle", a caption that fits the big circle of the
 *  starting batch too (a real false negative for the limits step), and a small star at the bottom. */
export const EXTRA: Fixed[] = [
	{ id: 'x-circle', label: { en: 'small circle, left', zh: '小圆，左边' }, scene: { kind: 'one', shape: 0, size: 0, place: 0 }, captions: ['a circle'] },
	{ id: 'x-star', label: { en: 'small star, bottom', zh: '小星星，下面' }, scene: { kind: 'one', shape: 4, size: 0, place: 3 }, captions: ['a small star at the bottom'] }
];
/** The pairs the widget offers to add (each once). */
export const POOL: Fixed[] = [PRESETS[0], PARTNERS[0], PARTNERS[1], PARTNERS[3], PARTNERS[2], PRESETS[3], PRESETS[2], PARTNERS[4], EXTRA[0], PRESETS[5], EXTRA[1]];

const batch = (ids: string[], tau = TAU_START, preset?: string): BatchRaw => ({ pairs: ids.map((id) => pairOf(byId(id))), tau, ...(preset ? { preset } : {}) });
/** The batch the page starts with: five pairs (two of them circles), τ = 0.2. */
export const START: BatchRaw = batch(['circle', 'x-circle', 'p-triangle', 'p-square', 'p-cross'], TAU_START, 'start');

/** Starting batches (the lesson's fixtures and the widget's buttons). */
export const BATCHES: { id: string; label: I18n; short: I18n; raw: BatchRaw }[] = [
	{ id: 'start', label: { en: 'Five pairs, two of them circles (τ = 0.2)', zh: '五对，其中两个是圆（τ = 0.2）' }, short: { en: 'Five pairs', zh: '五对' }, raw: START },
	{ id: 'four', label: { en: 'Four different pairs (τ = 0.07)', zh: '四对互不相同（τ = 0.07）' }, short: { en: 'Four pairs', zh: '四对' }, raw: batch(['circle', 'p-triangle', 'p-square', 'p-star'], 0.07, 'four') },
	{ id: 'eight', label: { en: 'Eight pairs (τ = 0.5)', zh: '八对（τ = 0.5）' }, short: { en: 'Eight pairs', zh: '八对' }, raw: batch(['circle', 'p-triangle', 'p-square', 'p-cross', 'star', 'triangle', 'p-pair', 'x-star'], 0.5, 'eight') }
];

/** The learner's pairs, cleaned: at most MAX_PAIRS, strokes as arrays of points, captions as strings. */
export function pairsOf(raw: unknown): PairRaw[] {
	const ps = ((raw as BatchRaw | null)?.pairs ?? []).slice(0, MAX_PAIRS);
	return ps.map((p) => ({ strokes: (p?.strokes ?? []).map((s) => (Array.isArray(s) ? s : ((s as { points?: Pt[] })?.points ?? []))), caption: String(p?.caption ?? ''), ...(p?.from ? { from: p.from } : {}) }));
}
export const tauOf = (raw: unknown) => clampTau((raw as BatchRaw | null)?.tau);
/** true when the batch holds a pair the learner drew or captioned (not one of the fixed pairs as given) */
export const hasOwnPair = (raw: unknown) => pairsOf(raw).some((p) => !p.from || p.caption !== byId(p.from)?.captions[0]);

/** A stable text key of one drawing (batch.ts memoizes vectors by it). */
export const drawingKey = (strokes: Pt[][]) => strokes.map((s) => s.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(' ')).join('|');

const NOTE = {
	few: { en: 'A batch needs at least two pairs: with one pair there is nothing to compare.', zh: '一批至少要有两对：只有一对时没有可以比较的对象。' },
	blank: { en: 'A drawing is empty: the model sees only the frame.', zh: '有一幅画是空的：模型只看到画框。' }
};

const adapter: InputAdapter<BatchRaw> = {
	id: 'batch',
	kind: 'drawing',
	preprocess(raw): Preprocessed {
		const pairs = pairsOf(raw);
		const N = pairs.length;
		const images = new Float32Array(N * 784);
		pairs.forEach((p, i) => images.set(render(p.strokes), i * 784));
		const t = { shape: [N, 28, 28], data: images };
		const notes: I18n[] = [];
		if (N < MIN_PAIRS) notes.push(NOTE.few);
		if (pairs.some((p) => !p.strokes.some((s) => s.length))) notes.push(NOTE.blank);
		const skipped = [...new Set(pairs.flatMap((p) => tokenize(p.caption).words.filter((w) => w.id < 0).map((w) => w.w)))];
		if (skipped.length) notes.push({ en: `Skipped (not among the model’s ${VOCAB.length - 1} words): ${skipped.join(', ')}.`, zh: `跳过了（不在模型认识的 ${VOCAB.length - 1} 个词里）：${skipped.join('、')}。` });
		return {
			tensors: { images: t },
			steps: [{ id: 'images', label: { en: 'The batch’s drawings, 28×28 numbers each', zh: '这一批的画，每幅 28×28 个数' }, tensor: t }],
			empty: N < MIN_PAIRS,
			notes
		};
	},
	edgeFixtures: () => [
		{ id: 'two', label: { en: 'the smallest batch: two pairs', zh: '最小的一批：两对' }, raw: batch(['circle', 'p-square']) },
		{ id: 'hot', label: { en: 'the starting batch at τ = 2 (the highest)', zh: '起始的一批，τ = 2（最高）' }, raw: { ...START, tau: TAU_RANGE[1], preset: undefined } },
		{ id: 'cold', label: { en: 'the starting batch at τ = 0.01 (the lowest)', zh: '起始的一批，τ = 0.01（最低）' }, raw: { ...START, tau: TAU_RANGE[0], preset: undefined } },
		// the first pair twice: rows and columns 0 and 1 tie, so the loss has a floor and is lowest at the slider's end
		{ id: 'twin', label: { en: 'the starting batch with its first pair twice', zh: '起始的一批，第一对出现两次' }, raw: { ...START, preset: undefined, pairs: START.pairs.map((p, i) => (i === 1 ? START.pairs[0] : p)) } },
		{
			id: 'odd',
			label: { en: 'a blank drawing, a dot, and captions without known words', zh: '一幅空白的画、一个点，以及没有认识的词的描述' },
			raw: { pairs: [pairOf(EDGES[0], 'hello'), pairOf(EDGES[1], ''), pairOf(PRESETS[1]), { strokes: strokesFor(PRESETS[4]), caption: 'a big cross at the top' }], tau: 0.3 }
		}
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: { en: 'The image tower sees each drawing as {n|int} numbers in all: a 28×28 grid per drawing, 0 for paper and 1 for ink, with the frame around it.', zh: '图像塔看到的每幅画都是一个 28×28 的网格，0 是纸，1 是墨，四周还有画框；整批一共 {n|int} 个数。' }
		}
	}
};

export default adapter;
