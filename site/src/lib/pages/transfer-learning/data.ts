/**
 * Types and loaders for the transfer-learning page's TinyCNN assets
 * (static/models/tinycnn/finetune.json, pca.json, heldout.json). Pure TS, no DOM.
 */
import type { I18n } from '$lib/i18n/types';

export type Method = 'head' | 'full' | 'scratch';
export const METHODS: Method[] = ['head', 'full', 'scratch'];

export interface FinetuneResult {
	n: number;
	acc: number[];
	mean: number;
	std: number;
}

export interface FinetuneJson {
	note: string;
	task: { classes: { id: string; en: string; zh: string }[]; testN: number };
	budget: { optimizer: string; steps: number; batch: string; augmentation: string; lr: Record<Method, number>; lrChosenOn: string };
	methods: Record<Method, { label: I18n; trainableParams: number; desc: I18n }>;
	ns: number[];
	seeds: number[];
	results: Record<Method, FinetuneResult[]>;
	baseModelGuesses: {
		note: string;
		classes: string[];
		byHeldOutClass: Record<string, { top1Share: Record<string, number>; meanProbs: number[] }>;
	};
	jsHeadCheck?: { n: number; jsMean: number; pythonMean: number };
}

export interface PcaPoint {
	c: number;
	x: number;
	y: number;
}

export interface PcaBasisJson {
	mean: number[];
	components: number[][];
	explainedVarianceRatio: number[];
	points: PcaPoint[];
}

export interface PcaJson {
	note: string;
	classes: { id: string; en: string; zh: string; heldOut: boolean }[];
	bases: { all: PcaBasisJson; heldOut: PcaBasisJson };
}

export interface HeldoutJson {
	note: string;
	classes: { id: string; en: string; zh: string; train: string[]; test: string[] }[];
}

/** The 10 known classes in model order (fallback before spec.json arrives; spec.classes wins). */
export const BASE_CLASSES: { id: string; en: string; zh: string }[] = [
	{ id: 'apple', en: 'apple', zh: '苹果' },
	{ id: 'bicycle', en: 'bicycle', zh: '自行车' },
	{ id: 'car', en: 'car', zh: '汽车' },
	{ id: 'clock', en: 'clock', zh: '时钟' },
	{ id: 'fish', en: 'fish', zh: '鱼' },
	{ id: 'house', en: 'house', zh: '房子' },
	{ id: 'star', en: 'star', zh: '星星' },
	{ id: 'sun', en: 'sun', zh: '太阳' },
	{ id: 'tree', en: 'tree', zh: '树' },
	{ id: 'umbrella', en: 'umbrella', zh: '雨伞' }
];

/** Parameter counts, from spec.json tensors (conv weights + biases) and finetune.json. */
export const PARAMS = {
	frozenBody: 4152, // conv1_1 80 + conv1_2 584 + conv2_1 1168 + conv2_2 2320
	newHead: 514, // 256 × 2 + 2
	oldHead: 2570, // 256 × 10 + 10
	full: 4666 // body + new head
};

/** Per-layer parameter counts (weights + biases), for the zoom panel. */
export const LAYER_PARAMS: Record<string, number> = {
	conv1_1: 8 * 1 * 9 + 8,
	conv1_2: 8 * 8 * 9 + 8,
	conv2_1: 16 * 8 * 9 + 16,
	conv2_2: 16 * 16 * 9 + 16
};

export async function fetchJson<T>(url: string): Promise<T> {
	const r = await fetch(url);
	if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
	return (await r.json()) as T;
}

/** Simplify a stroke for preprocessing: drop points closer than `eps` (normalized units) to the last kept one. */
export function simplifyStrokes(strokes: { points: { x: number; y: number }[] }[], eps = 0.004): [number, number][][] {
	return strokes.map((s) => {
		const out: [number, number][] = [];
		let lx = Infinity;
		let ly = Infinity;
		const pts = s.points;
		for (let i = 0; i < pts.length; i++) {
			const p = pts[i];
			const last = i === pts.length - 1;
			if (last || Math.hypot(p.x - lx, p.y - ly) >= eps) {
				out.push([p.x, p.y]);
				lx = p.x;
				ly = p.y;
			}
		}
		return out;
	});
}

/** Decode a base64 28×28 uint8 drawing (white on black) into the model input (value / 255). */
export function decodeB64(b64: string): Float32Array {
	const bin = atob(b64);
	const out = new Float32Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255;
	return out;
}

/** One training example as shown in the class columns. */
export interface SetItem {
	id: string;
	cls: 0 | 1;
	input: Float32Array;
	drawn: boolean;
	/** a preset drawing deliberately put in the wrong class (Try this) */
	flipped: boolean;
	/** the head's probability for class A at the current step, or null without a head */
	pA: number | null;
}
