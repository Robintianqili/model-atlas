// drawing28 (design §4.9; catalog InputKind 'drawing'): a drawing -> the [1, 28, 28] input of a Quick, Draw!-style
// classifier, values 0 (paper) .. 1 (ink). The preprocessing is TinyCNN's own (src/lib/models/tinycnn.ts, the one import
// outside atlas/ the dependency rule allows for this adapter): nothing is re-implemented, so TinyCNN, TinyViT and TinyCLIP
// see exactly the same 784 numbers for the same drawing (design §9, "Preprocessing mismatch").
//
// Raw inputs (DrawingRaw):
//   { strokes }   pad strokes, points in any coordinate system (y down); kit DrawPad strokes ({ points: {x, y}[] }) or
//                 point lists. Rendered straight into 28×28 with the dataset's stroke width (preprocessStrokes).
//   { b64 }       a preset: base64 of 784 uint8, white on black (static/models/tinycnn/presets.json); input = value / 255.
//   { ink }       784 numbers in [0, 1], row-major (a raw grid, e.g. a lesson fixture); clamped, not rescaled.
//   { image }     RGBA pixels of a canvas or photo (preprocessDrawing: crop to the ink, centre, scale like the dataset).
//   a 784-number array is read as { ink }; an array of strokes as { strokes }; null / undefined is a blank grid.
import type { I18n } from '../../../i18n/types.ts';
import type { Tensor } from '../../types/core.ts';
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';
import { IMG, decodePreset, preprocessDrawing, preprocessStrokes, type ImageLike, type Point } from '../../../models/tinycnn.ts';

export interface StrokeLike {
	points: { x: number; y: number }[];
}
export type DrawingRaw =
	| null
	| undefined
	| { strokes: (StrokeLike | Point[])[] }
	| { b64: string }
	| { ink: ArrayLike<number> }
	| { image: ImageLike }
	| ArrayLike<number>
	| (StrokeLike | Point[])[];

const N = IMG * IMG;

/** Drop points closer than eps to the previous kept point (a copy of pages/tinycnn/data.ts simplifyStroke, which the
 *  TinyCNN page applies before preprocessStrokes; copied because adapters may not import page code). */
export function simplifyStroke<P extends { x: number; y: number }>(pts: P[], eps = 0.003): P[] {
	if (pts.length <= 2) return pts;
	const out: P[] = [pts[0]];
	let last = pts[0];
	const e2 = eps * eps;
	for (let i = 1; i < pts.length - 1; i++) {
		const p = pts[i];
		const dx = p.x - last.x;
		const dy = p.y - last.y;
		if (dx * dx + dy * dy >= e2) {
			out.push(p);
			last = p;
		}
	}
	out.push(pts[pts.length - 1]);
	return out;
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';
const isNumArray = (v: unknown): v is ArrayLike<number> =>
	(ArrayBuffer.isView(v) && !(v instanceof DataView)) || (Array.isArray(v) && (v.length === 0 || typeof v[0] === 'number'));

function toPoints(s: StrokeLike | Point[]): { x: number; y: number }[] {
	const pts = Array.isArray(s) ? s : (s?.points ?? []);
	return pts.map((p) => (Array.isArray(p) ? { x: p[0], y: p[1] } : { x: (p as { x: number }).x, y: (p as { y: number }).y })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
}

export type DrawingSource = 'blank' | 'strokes' | 'preset' | 'ink' | 'image';

/** The 784 model inputs for any raw drawing, plus where they came from. */
export function drawingInput(raw: unknown, opts: Record<string, unknown> = {}): { input: Float32Array; empty: boolean; source: DrawingSource } {
	if (raw === null || raw === undefined) return { input: new Float32Array(N), empty: true, source: 'blank' };
	if (isNumArray(raw)) {
		if (raw.length === 0) return { input: new Float32Array(N), empty: true, source: 'blank' };
		return ink(raw);
	}
	if (Array.isArray(raw)) return strokes(raw as (StrokeLike | Point[])[]);
	if (isObj(raw)) {
		if (Array.isArray(raw.strokes)) return strokes(raw.strokes as (StrokeLike | Point[])[]);
		if (typeof raw.b64 === 'string') {
			const input = decodePreset(raw.b64);
			if (input.length !== N) throw new Error(`drawing28: a preset must hold ${N} values, this one has ${input.length}`);
			return { input, empty: !input.some((v) => v > 0), source: 'preset' };
		}
		if (isNumArray(raw.ink)) return ink(raw.ink);
		if (isObj(raw.image)) {
			const threshold = typeof opts.threshold === 'number' ? opts.threshold : undefined;
			const p = preprocessDrawing(raw.image as unknown as ImageLike, threshold === undefined ? {} : { threshold });
			return { input: p.input, empty: p.empty, source: 'image' };
		}
	}
	throw new Error('drawing28: expected { strokes }, { b64 }, { ink }, { image }, a 784-number array or null');

	function ink(a: ArrayLike<number>) {
		if (a.length !== N) throw new Error(`drawing28: an ink grid must hold ${N} values (28×28), got ${a.length}`);
		const input = new Float32Array(N);
		let any = false;
		for (let i = 0; i < N; i++) {
			const v = Number(a[i]);
			input[i] = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
			if (input[i] > 0) any = true;
		}
		return { input, empty: !any, source: 'ink' as const };
	}
	function strokes(list: (StrokeLike | Point[])[]) {
		const lines = list.map((s) => simplifyStroke(toPoints(s))).filter((p) => p.length > 0);
		if (!lines.length) return { input: new Float32Array(N), empty: true, source: 'blank' as const };
		// K1: options.frame [x0, y0, x1, y1] (the canvas) keeps a drawing's size and place; by default the ink is cropped
		// and centred, as TinyCNN's training data was
		const f = (isObj(raw) && Array.isArray((raw as { frame?: unknown }).frame) ? (raw as { frame: unknown[] }).frame : opts.frame) as unknown;
		const frame = Array.isArray(f) && f.length === 4 && f.every((v) => typeof v === 'number' && Number.isFinite(v)) ? (f as [number, number, number, number]) : undefined;
		const p = preprocessStrokes(lines, frame ? { frame } : {});
		return { input: p.input, empty: p.empty, source: 'strokes' as const };
	}
}

const NOTE_EMPTY: I18n = {
	en: 'The grid is empty. Draw something (or pick a preset) and every step fills with the numbers of your drawing.',
	zh: '网格是空的。画点什么（或选一幅预设画），每一步都会填上你这幅画的数。'
};
const STEP_NOTE: Record<DrawingSource, I18n> = {
	blank: { en: 'Nothing drawn yet.', zh: '还没有画。' },
	strokes: {
		en: 'Your strokes are drawn again at the dataset’s line width, cropped to the ink, scaled into the middle of the grid and centred, like the training drawings.',
		zh: '你的笔画按数据集的线宽重新画一遍，裁到有墨的范围，缩放到网格中间并居中，和训练用的画一样。'
	},
	preset: { en: 'A drawing from the Quick, Draw! test set, already 28×28.', zh: '一幅来自 Quick, Draw! 测试集的画，本来就是 28×28。' },
	ink: { en: 'A grid given directly as numbers.', zh: '直接以数字给出的网格。' },
	image: {
		en: 'Your picture is turned into ink, cropped to it, scaled into the middle of the grid and centred.',
		zh: '你的图片先变成墨迹，再裁剪、缩放到网格中间并居中。'
	}
};

function shapeFrom(spec: unknown): number[] {
	const s = (spec as { inputs?: { shape?: unknown[] }[] } | null)?.inputs?.[0]?.shape;
	if (Array.isArray(s) && s.every((d) => typeof d === 'number') && s.reduce((a: number, b) => a * (b as number), 1) === N) return s as number[];
	return [1, IMG, IMG];
}

interface PresetFile {
	classes?: { id: string; en: string; zh?: string; items: { b64: string }[] }[];
	heldOut?: { id: string; en: string; zh?: string; items: { b64: string }[] }[];
}

export const drawing28: InputAdapter<DrawingRaw> = {
	id: 'drawing28',
	kind: 'drawing',
	preprocess(raw, spec, opts) {
		const { input, empty, source } = drawingInput(raw, opts ?? {});
		const tensor: Tensor = { shape: shapeFrom(spec), data: input };
		const out: Preprocessed = {
			tensors: { input: tensor },
			steps: [{ id: 'input', label: { en: 'What the model sees: 28×28 numbers', zh: '模型看到的：28×28 个数' }, tensor, note: STEP_NOTE[source] }],
			empty
		};
		if (empty) out.notes = [NOTE_EMPTY];
		return out;
	},
	/** One preset per class (the first item of each class in presets.json), then one per held-out class. */
	presets(assets) {
		const file = (assets['presets.json'] ?? assets.presets) as PresetFile | undefined;
		if (!file || !Array.isArray(file.classes)) return [];
		const out: { id: string; label: I18n; raw: DrawingRaw }[] = [];
		for (const c of file.classes) if (c.items?.[0]) out.push({ id: c.id, label: { en: c.en, zh: c.zh ?? c.en }, raw: { b64: c.items[0].b64 } });
		for (const c of file.heldOut ?? [])
			if (c.items?.[0]) out.push({ id: `heldout-${c.id}`, label: { en: `${c.en} (not one of its classes)`, zh: `${c.zh ?? c.en}（不在它认识的类别里）` }, raw: { b64: c.items[0].b64 } });
		return out;
	},
	edgeFixtures() {
		return [
			{ id: 'blank', label: { en: 'blank pad', zh: '空白画板' }, raw: { strokes: [] } },
			{ id: 'dot', label: { en: 'a single dot', zh: '一个点' }, raw: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] } },
			{ id: 'small', label: { en: 'a tiny stroke', zh: '很小的一笔' }, raw: { strokes: [{ points: [{ x: 0.47, y: 0.47 }, { x: 0.5, y: 0.52 }, { x: 0.53, y: 0.5 }] }] } }
		];
	},
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The model never sees your strokes. It sees {n|int} numbers in a {shape} grid: 0 is empty paper and 1 is full ink.',
				zh: '模型看不到你的笔画，它只看到 {shape} 网格里的 {n|int} 个数：0 是空白的纸，1 是满满的墨。'
			},
			how: {
				en: 'Your drawing is first cropped to the ink, scaled into the middle and centred, like every training drawing.',
				zh: '你的画先裁到有墨的地方，缩放到中间并居中，和每一幅训练用的画一样。'
			},
			look: {
				en: 'the bright cells: they are your ink, and these numbers are all the model works with.',
				zh: '亮的格子：那就是你的墨迹，模型能用的只有这些数。'
			}
		}
	}
};

export default drawing28;
