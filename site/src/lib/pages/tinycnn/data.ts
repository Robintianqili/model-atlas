/**
 * TinyCNN page: types for the shipped asset files and small pure helpers.
 * No Svelte, no DOM. Model code lives in $lib/models/tinycnn.ts (loaded lazily by the page).
 */
import type { Box, LayerSpec, TinyCNNSpec, Trace } from '$lib/models/tinycnn';
import type { Domain } from '$lib/kit/color';

export type { Box, LayerSpec, TinyCNNSpec, Trace };

// ---------------------------------------------------------------- asset files

export interface PresetItem {
	b64: string;
	fileIndex: number;
}
export interface PresetGroup {
	id: string;
	en: string;
	zh: string;
	items: PresetItem[];
}
export interface PresetsFile {
	note: string;
	classes: PresetGroup[];
	heldOut: PresetGroup[];
}

export interface ChannelTop {
	class: number;
	fileIndex: number;
	act: number;
	y: number;
	x: number;
	/** receptive field [y0, x0, y1, x1] in the 28x28 image */
	rf: [number, number, number, number];
	/** base64 uint8, patchSize^2 */
	patch: string;
}
export interface ChannelInfo {
	channel: number;
	top: ChannelTop[];
	classMeanMax: number[];
	icon?: string;
	iconAct?: number;
}
export interface ChannelLayer {
	rfSize: number;
	jump: number;
	patchSize: number;
	channels: ChannelInfo[];
	iconCell: [number, number];
	iconRf: [number, number, number, number];
}
export interface ChannelsFile {
	note: string;
	pad: number;
	classes: string[];
	layers: Record<string, ChannelLayer>;
	iconNote?: string;
}

export interface HeldOutClass {
	id: string;
	en: string;
	zh: string;
	train: string[];
	test: string[];
}
export interface HeldOutFile {
	note: string;
	classes: HeldOutClass[];
}

/** Training facts as they appear in spec.json (precomputed). */
export interface TrainingFacts {
	dataset: string;
	datasetUrl?: string;
	license?: string;
	attribution?: string;
	split: { perClass: { train: number; val: number; test: number }; seed: number };
	epochs: number;
	bestEpoch: number;
	steps: number;
	batchSize: number;
	optimizer: string;
	augmentation: string;
	hardware: string;
	metrics: {
		test_acc: number;
		val_acc: number;
		test_acc_augmented: number;
		test_n: number;
		per_class_test_acc: Record<string, number>;
		confusion: number[][];
		confusion_note?: string;
		drawingPath?: { note: string; accuracy: Record<string, number>; checked?: string };
	};
	history: { epoch: number; train_loss: number; val_acc: number }[];
}

// ---------------------------------------------------------------- layers

/** Layers whose output is a stack of feature maps [C, H, W]. */
export const SPATIAL_OPS = new Set(['input', 'conv2d', 'relu', 'maxpool2d']);

export function isSpatial(l: LayerSpec | undefined): boolean {
	return !!l && SPATIAL_OPS.has(l.op);
}

/** The conv layer whose channel a relu / pool layer carries (for "what this channel responds to"). */
export function sourceConv(spec: TinyCNNSpec, id: string): string | null {
	let l = spec.layers.find((x) => x.id === id);
	while (l && l.op !== 'conv2d') {
		if (l.op !== 'relu' && l.op !== 'maxpool2d') return null;
		const prev = l.in[0];
		l = spec.layers.find((x) => x.id === prev);
	}
	return l ? l.id : null;
}

/** Number of selectable units in a layer for the overview (channels, classes, or 1). */
export function unitsOf(l: LayerSpec): number {
	if (SPATIAL_OPS.has(l.op)) return l.outShape[0];
	if (l.op === 'dense' || l.op === 'softmax') return l.outShape[0];
	return 1;
}

/** Parameter count of a layer (weights + biases) from the spec's tensors. */
export function layerParams(spec: TinyCNNSpec, id: string): { weight: number; bias: number; wShape: number[] } | null {
	const w = spec.tensors.find((t) => t.name === `${id}.weight`);
	const b = spec.tensors.find((t) => t.name === `${id}.bias`);
	if (!w || !b) return null;
	return { weight: w.length, bias: b.length, wShape: w.shape };
}

/** Value domains per layer (from this trace) plus one shared by every feature-map layer. */
export function traceDomains(spec: TinyCNNSpec, trace: Trace): { layer: Record<string, Domain>; global: Domain } {
	const layer: Record<string, Domain> = {};
	let gmax = 0;
	for (const l of spec.layers) {
		const d = trace.t[l.id]?.data;
		if (!d) continue;
		let m = 0;
		for (let i = 0; i < d.length; i++) {
			const a = Math.abs(d[i]);
			if (a > m) m = a;
		}
		if (!(m > 0)) m = 1;
		layer[l.id] = l.op === 'input' ? [0, 1] : l.op === 'softmax' ? [0, 1] : [-m, m];
		if (l.op === 'conv2d' || l.op === 'relu' || l.op === 'maxpool2d') gmax = Math.max(gmax, m);
	}
	return { layer, global: [-(gmax || 1), gmax || 1] };
}

/** Domain of one channel of a [C, H, W] tensor (diverging, symmetric). */
export function channelDomain(data: ArrayLike<number>, c: number, hw: number): Domain {
	let m = 0;
	const o = c * hw;
	for (let i = 0; i < hw; i++) {
		const a = Math.abs(data[o + i]);
		if (a > m) m = a;
	}
	return [-(m || 1), m || 1];
}

/** Position of the largest value in channel c of a [C, H, W] tensor. */
export function argmaxCell(data: ArrayLike<number>, c: number, h: number, w: number): { y: number; x: number } {
	let best = -Infinity;
	let bi = 0;
	const o = c * h * w;
	for (let i = 0; i < h * w; i++) {
		if (data[o + i] > best) {
			best = data[o + i];
			bi = i;
		}
	}
	return { y: Math.floor(bi / w), x: bi % w };
}

/** Cosine similarity of two equal-length arrays (0 when either is all zeros). */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let s = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		s += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	return na > 0 && nb > 0 ? s / Math.sqrt(na * nb) : 0;
}

/** Remove points closer than eps (normalized pad units) to the previous kept point; keeps first and last. */
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

/** Blank half of a 28x28 input (the "erase half" experiment). Returns a new array. */
export type EraseMode = 'none' | 'left' | 'right' | 'top' | 'bottom';
export function eraseHalf(input: Float32Array, mode: EraseMode, n = 28): Float32Array {
	if (mode === 'none') return input;
	const out = Float32Array.from(input);
	const h = n / 2;
	for (let y = 0; y < n; y++) {
		for (let x = 0; x < n; x++) {
			const hit = mode === 'left' ? x < h : mode === 'right' ? x >= h : mode === 'top' ? y < h : y >= h;
			if (hit) out[y * n + x] = 0;
		}
	}
	return out;
}

/** Decode base64 uint8 to floats in [0, 1] (presets, patches). */
export function b64ToFloats(b64: string): Float32Array {
	const bin = atob(b64);
	const out = new Float32Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255;
	return out;
}

/** Python source of the model (pipeline/tinycnn/common.py, lines 37-74), shown in the Code tab. */
export const MODEL_SOURCE_START = 37;
export const MODEL_SOURCE = `class TinyVGG(nn.Module):
    """conv3x3(no pad) -> ReLU -> conv3x3 -> ReLU -> maxpool2 -> conv3x3 -> ReLU -> conv3x3 -> ReLU -> maxpool2 -> flatten -> dense."""

    def __init__(self, ch=(8, 8, 16, 16), n_classes=10):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.ch = ch
        self.conv1_1 = nn.Conv2d(1, c1, 3)
        self.conv1_2 = nn.Conv2d(c1, c2, 3)
        self.conv2_1 = nn.Conv2d(c2, c3, 3)
        self.conv2_2 = nn.Conv2d(c3, c4, 3)
        self.dense = nn.Linear(c4 * 4 * 4, n_classes)

    def features(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        return x.flatten(1)

    def forward(self, x):
        return self.dense(self.features(x))

    def trace(self, x):
        """Every layer output, keyed by contract layer id (x: [B,1,28,28])."""
        t = {"input": x}
        t["conv1_1"] = self.conv1_1(x); t["relu1_1"] = F.relu(t["conv1_1"])
        t["conv1_2"] = self.conv1_2(t["relu1_1"]); t["relu1_2"] = F.relu(t["conv1_2"])
        t["pool1"] = F.max_pool2d(t["relu1_2"], 2)
        t["conv2_1"] = self.conv2_1(t["pool1"]); t["relu2_1"] = F.relu(t["conv2_1"])
        t["conv2_2"] = self.conv2_2(t["relu2_1"]); t["relu2_2"] = F.relu(t["conv2_2"])
        t["pool2"] = F.max_pool2d(t["relu2_2"], 2)
        t["flatten"] = t["pool2"].flatten(1)
        t["dense"] = self.dense(t["flatten"])
        t["softmax"] = F.softmax(t["dense"], dim=1)
        return t
`;

/** Lines of common.py (absolute line numbers) that define / compute each layer. */
export const CODE_LINES: Record<string, number[]> = {
	input: [50, 64],
	conv1_1: [44, 51, 65],
	relu1_1: [51, 65],
	conv1_2: [45, 52, 66],
	relu1_2: [52, 66],
	pool1: [53, 67],
	conv2_1: [46, 54, 68],
	relu2_1: [54, 68],
	conv2_2: [47, 55, 69],
	relu2_2: [55, 69],
	pool2: [56, 70],
	flatten: [57, 71],
	dense: [48, 60, 72],
	softmax: [73]
};
