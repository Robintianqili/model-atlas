// TinyCNN: a TinyVGG-style classifier for 28x28 drawings (Quick, Draw! classes).
// Pure TypeScript: no DOM, no Svelte. Runs the same in the browser, in a Worker and in Node.
//
//   load(fetchFn, baseUrl)            -> TinyCNN model object
//   model.forward(input)              -> Trace { order, t: { [layerId]: { shape, data } } }
//   model.convDetail / poolDetail / denseDetail / softmaxDetail   -> arithmetic for one output cell
//   preprocessDrawing / preprocessStrokes / preprocessPhoto      -> Float32Array [1,28,28] like training
//
// The layer graph is read from spec.json, so the same code runs any plain conv/relu/maxpool/flatten/dense
// stack with valid (no-padding) convolutions.

// ---------------------------------------------------------------- types

export interface Bilingual {
	en: string;
	zh: string;
}

export interface TensorSpec {
	name: string;
	shape: number[];
	offset: number; // in floats
	length: number; // in floats
}

export interface LayerSpec {
	id: string;
	op: string; // 'input' | 'conv2d' | 'relu' | 'maxpool2d' | 'flatten' | 'dense' | 'softmax'
	label: Bilingual;
	in: string[];
	outShape: number[];
	params?: Record<string, number>;
	rf?: { size: number; jump: number };
	desc?: Bilingual;
}

export interface ClassInfo {
	id: string;
	en: string;
	zh: string;
}

export interface TinyCNNSpec {
	slug: string;
	name: Bilingual;
	version: string;
	params: number;
	precision: string;
	input: { kind: string; shape: number[]; notes?: string; [k: string]: unknown };
	tensors: TensorSpec[];
	layers: LayerSpec[];
	classes: ClassInfo[];
	heldOutClasses?: ClassInfo[];
	training?: Record<string, unknown>;
	parity?: Record<string, unknown>;
	[k: string]: unknown;
}

export interface Tensor {
	shape: number[];
	data: Float32Array;
}

export interface Trace {
	order: string[];
	t: Record<string, Tensor>;
}

/** A fetch-like function: the browser's fetch, or a Node shim that reads files. */
export type FetchLike = (url: string) => Promise<{
	ok?: boolean;
	status?: number;
	json(): Promise<unknown>;
	arrayBuffer(): Promise<ArrayBuffer>;
}>;

/** Inclusive pixel box in some tensor's [H, W] grid. */
export interface Box {
	y0: number;
	x0: number;
	y1: number;
	x1: number;
}

export interface ConvDetail {
	layerId: string;
	inputId: string;
	outChannel: number;
	y: number;
	x: number;
	kernel: number;
	inChannels: number;
	/** Box of the window in the input tensor (inclusive). */
	window: Box;
	/** Receptive field of this cell in the 28x28 input image (inclusive). */
	rf: Box;
	/** Per input channel: the k*k input values, row-major. */
	windows: Float32Array[];
	/** Per input channel: the k*k kernel weights, row-major. */
	kernels: Float32Array[];
	/** Per input channel: windows[c][i] * kernels[c][i]. */
	products: Float32Array[];
	/** Per input channel: sum of its products. */
	channelSums: Float32Array;
	bias: number;
	/** sum(channelSums) + bias, computed here (equals the traced conv output up to fp32 rounding). */
	total: number;
	/** The value stored in the trace for this cell. */
	traced: number;
	/** max(0, total): what the following ReLU outputs. */
	relu: number;
}

export interface PoolDetail {
	layerId: string;
	inputId: string;
	channel: number;
	y: number;
	x: number;
	kernel: number;
	window: Box;
	rf: Box;
	/** The k*k input values, row-major. */
	values: Float32Array;
	/** Index into values of the maximum (first one on ties). */
	argmax: number;
	/** [y, x] of the maximum in the input tensor. */
	argmaxAt: [number, number];
	value: number;
	traced: number;
}

export interface DenseContribution {
	/** Index into the flattened input. */
	index: number;
	/** Where that input came from in the layer before flatten: channel, y, x. */
	c: number;
	y: number;
	x: number;
	input: number;
	weight: number;
	product: number;
}

export interface DenseDetail {
	layerId: string;
	classIndex: number;
	bias: number;
	/** bias + sum(products), computed here. */
	logit: number;
	traced: number;
	/** All products weight[i] * input[i], in flatten order (useful as a heatmap per channel). */
	products: Float32Array;
	/** Sum of products per source channel (e.g. 16 numbers for pool2). */
	channelSums: Float32Array;
	/** Largest positive products, descending. */
	topPositive: DenseContribution[];
	/** Most negative products, most negative first. */
	topNegative: DenseContribution[];
	/** Number of inputs that are exactly zero (ReLU switched off). */
	zeroInputs: number;
}

export interface SoftmaxDetail {
	logits: Float32Array;
	max: number;
	/** exp(logit - max) */
	exps: Float32Array;
	sumExp: number;
	probs: Float32Array;
	top: number;
}

export interface TinyCNN {
	spec: TinyCNNSpec;
	/** Weights by tensor name, e.g. 'conv1_1.weight' (shape [8,1,3,3]). */
	weights: Record<string, Tensor>;
	/** Full forward pass. input: 784 numbers in [0,1] ([1,28,28] row-major). */
	forward(input: ArrayLike<number>): Trace;
	/** Flatten features (the input of the dense layer). */
	features(input: ArrayLike<number>): Float32Array;
	/** Class probabilities and the index of the top class. */
	predict(input: ArrayLike<number>): { probs: Float32Array; top: number };
	convDetail(trace: Trace, layerId: string, outChannel: number, y: number, x: number): ConvDetail;
	poolDetail(trace: Trace, layerId: string, channel: number, y: number, x: number): PoolDetail;
	denseDetail(trace: Trace, classIndex: number, topK?: number): DenseDetail;
	softmaxDetail(trace: Trace): SoftmaxDetail;
	/** Input-image pixels that can influence cell (y, x) of a spatial layer (inclusive box). */
	receptiveField(layerId: string, y: number, x: number): Box;
	/** Map a flatten index to (channel, y, x) in the layer feeding flatten. */
	flattenIndexToCHW(index: number): { c: number; y: number; x: number };
	layer(id: string): LayerSpec;
}

// ---------------------------------------------------------------- loading

function joinUrl(base: string, file: string): string {
	if (!base) return file;
	return base.endsWith('/') ? base + file : base + '/' + file;
}

/** Fetch spec.json and weights.bin from baseUrl (e.g. `${base}/models/tinycnn`) and build the model. */
export async function load(fetchFn: FetchLike, baseUrl: string): Promise<TinyCNN> {
	const specRes = await fetchFn(joinUrl(baseUrl, 'spec.json'));
	if (specRes.ok === false) throw new Error(`tinycnn: spec.json HTTP ${specRes.status}`);
	const spec = (await specRes.json()) as TinyCNNSpec;
	const wRes = await fetchFn(joinUrl(baseUrl, 'weights.bin'));
	if (wRes.ok === false) throw new Error(`tinycnn: weights.bin HTTP ${wRes.status}`);
	const buf = await wRes.arrayBuffer();
	return fromParts(spec, buf);
}

/** Build the model synchronously from an already-loaded spec and weights buffer (float32 little-endian). */
export function fromParts(spec: TinyCNNSpec, weightsBuffer: ArrayBuffer): TinyCNN {
	const total = spec.tensors.reduce((m, t) => Math.max(m, t.offset + t.length), 0);
	if (weightsBuffer.byteLength < total * 4) {
		throw new Error(`tinycnn: weights.bin has ${weightsBuffer.byteLength} bytes, spec needs ${total * 4}`);
	}
	// Read little-endian explicitly (independent of platform endianness).
	const dv = new DataView(weightsBuffer);
	const all = new Float32Array(total);
	for (let i = 0; i < total; i++) all[i] = dv.getFloat32(i * 4, true);
	const weights: Record<string, Tensor> = {};
	for (const t of spec.tensors) {
		const n = t.shape.reduce((a, b) => a * b, 1);
		if (n !== t.length) throw new Error(`tinycnn: tensor ${t.name} shape/length mismatch`);
		weights[t.name] = { shape: t.shape.slice(), data: all.subarray(t.offset, t.offset + t.length) };
	}
	return new Model(spec, weights);
}

// ---------------------------------------------------------------- kernels

function conv2dValid(
	x: Float32Array, C: number, H: number, W: number,
	w: Float32Array, b: Float32Array, O: number, k: number, stride: number
): Tensor {
	const OH = Math.floor((H - k) / stride) + 1;
	const OW = Math.floor((W - k) / stride) + 1;
	const out = new Float32Array(O * OH * OW);
	for (let o = 0; o < O; o++) {
		for (let oy = 0; oy < OH; oy++) {
			for (let ox = 0; ox < OW; ox++) {
				let s = 0;
				for (let c = 0; c < C; c++) {
					const wBase = (o * C + c) * k * k;
					const xBase = c * H * W;
					for (let ky = 0; ky < k; ky++) {
						const row = xBase + (oy * stride + ky) * W + ox * stride;
						const wRow = wBase + ky * k;
						for (let kx = 0; kx < k; kx++) s += x[row + kx] * w[wRow + kx];
					}
				}
				out[(o * OH + oy) * OW + ox] = s + b[o];
			}
		}
	}
	return { shape: [O, OH, OW], data: out };
}

function maxpool(x: Float32Array, C: number, H: number, W: number, k: number, stride: number): Tensor {
	const OH = Math.floor((H - k) / stride) + 1;
	const OW = Math.floor((W - k) / stride) + 1;
	const out = new Float32Array(C * OH * OW);
	for (let c = 0; c < C; c++) {
		for (let oy = 0; oy < OH; oy++) {
			for (let ox = 0; ox < OW; ox++) {
				let m = -Infinity;
				for (let ky = 0; ky < k; ky++) {
					for (let kx = 0; kx < k; kx++) {
						const v = x[(c * H + oy * stride + ky) * W + ox * stride + kx];
						if (v > m) m = v;
					}
				}
				out[(c * OH + oy) * OW + ox] = m;
			}
		}
	}
	return { shape: [C, OH, OW], data: out };
}

function dense(x: Float32Array, w: Float32Array, b: Float32Array, O: number): Tensor {
	const I = x.length;
	const out = new Float32Array(O);
	for (let o = 0; o < O; o++) {
		let s = 0;
		const base = o * I;
		for (let i = 0; i < I; i++) s += w[base + i] * x[i];
		out[o] = s + b[o];
	}
	return { shape: [O], data: out };
}

function softmax(x: Float32Array): Float32Array {
	let m = -Infinity;
	for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
	let s = 0;
	const e = new Float64Array(x.length);
	for (let i = 0; i < x.length; i++) {
		e[i] = Math.exp(x[i] - m);
		s += e[i];
	}
	const out = new Float32Array(x.length);
	for (let i = 0; i < x.length; i++) out[i] = e[i] / s;
	return out;
}

// ---------------------------------------------------------------- model

class Model implements TinyCNN {
	spec: TinyCNNSpec;
	weights: Record<string, Tensor>;
	private byId: Map<string, LayerSpec>;

	constructor(spec: TinyCNNSpec, weights: Record<string, Tensor>) {
		this.spec = spec;
		this.weights = weights;
		this.byId = new Map(spec.layers.map((l) => [l.id, l]));
	}

	layer(id: string): LayerSpec {
		const l = this.byId.get(id);
		if (!l) throw new Error(`tinycnn: unknown layer ${id}`);
		return l;
	}

	private w(name: string): Tensor {
		const t = this.weights[name];
		if (!t) throw new Error(`tinycnn: missing tensor ${name}`);
		return t;
	}

	forward(input: ArrayLike<number>): Trace {
		const inShape = this.spec.input.shape;
		const n = inShape.reduce((a, b) => a * b, 1);
		if (input.length !== n) throw new Error(`tinycnn: input has ${input.length} values, expected ${n}`);
		const t: Record<string, Tensor> = {};
		const order: string[] = [];
		for (const L of this.spec.layers) {
			let out: Tensor;
			if (L.op === 'input') {
				out = { shape: inShape.slice(), data: Float32Array.from(input) };
			} else {
				const src = t[L.in[0]];
				if (!src) throw new Error(`tinycnn: layer ${L.id} input ${L.in[0]} not computed yet`);
				const p = L.params ?? {};
				if (L.op === 'conv2d') {
					const W = this.w(`${L.id}.weight`);
					const B = this.w(`${L.id}.bias`);
					const [C, H, Wd] = src.shape;
					out = conv2dValid(src.data, C, H, Wd, W.data, B.data, W.shape[0], W.shape[2], p.stride ?? 1);
				} else if (L.op === 'relu') {
					const d = new Float32Array(src.data.length);
					for (let i = 0; i < d.length; i++) d[i] = src.data[i] > 0 ? src.data[i] : 0;
					out = { shape: src.shape.slice(), data: d };
				} else if (L.op === 'maxpool2d') {
					const [C, H, Wd] = src.shape;
					const k = p.kernel ?? 2;
					out = maxpool(src.data, C, H, Wd, k, p.stride ?? k);
				} else if (L.op === 'flatten') {
					out = { shape: [src.data.length], data: Float32Array.from(src.data) };
				} else if (L.op === 'dense') {
					const W = this.w(`${L.id}.weight`);
					const B = this.w(`${L.id}.bias`);
					out = dense(src.data, W.data, B.data, W.shape[0]);
				} else if (L.op === 'softmax') {
					out = { shape: src.shape.slice(), data: softmax(src.data) };
				} else {
					throw new Error(`tinycnn: unsupported op ${L.op}`);
				}
			}
			t[L.id] = out;
			order.push(L.id);
		}
		return { order, t };
	}

	features(input: ArrayLike<number>): Float32Array {
		const tr = this.forward(input);
		const flat = this.spec.layers.find((l) => l.op === 'flatten');
		if (!flat) throw new Error('tinycnn: no flatten layer');
		return tr.t[flat.id].data;
	}

	predict(input: ArrayLike<number>): { probs: Float32Array; top: number } {
		const tr = this.forward(input);
		const last = this.spec.layers[this.spec.layers.length - 1];
		const probs = tr.t[last.id].data;
		let top = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
		return { probs, top };
	}

	receptiveField(layerId: string, y: number, x: number): Box {
		let box: Box = { y0: y, x0: x, y1: y, x1: x };
		let L = this.layer(layerId);
		while (L.op !== 'input') {
			const p = L.params ?? {};
			if (L.op === 'conv2d' || L.op === 'maxpool2d') {
				const k = p.kernel ?? 2;
				const s = p.stride ?? (L.op === 'conv2d' ? 1 : k);
				box = { y0: box.y0 * s, x0: box.x0 * s, y1: box.y1 * s + k - 1, x1: box.x1 * s + k - 1 };
			} else if (L.op !== 'relu') {
				// flatten / dense / softmax: the whole image
				const [, H, W] = this.spec.input.shape;
				return { y0: 0, x0: 0, y1: H - 1, x1: W - 1 };
			}
			L = this.layer(L.in[0]);
		}
		return box;
	}

	flattenIndexToCHW(index: number): { c: number; y: number; x: number } {
		const flat = this.spec.layers.find((l) => l.op === 'flatten');
		if (!flat) throw new Error('tinycnn: no flatten layer');
		const [, H, W] = this.layer(flat.in[0]).outShape;
		const c = Math.floor(index / (H * W));
		const r = index - c * H * W;
		return { c, y: Math.floor(r / W), x: r % W };
	}

	convDetail(trace: Trace, layerId: string, outChannel: number, y: number, x: number): ConvDetail {
		const L = this.layer(layerId);
		if (L.op !== 'conv2d') throw new Error(`tinycnn: ${layerId} is not a conv layer`);
		const inputId = L.in[0];
		const src = trace.t[inputId];
		const outT = trace.t[layerId];
		if (!src || !outT) throw new Error(`tinycnn: trace lacks ${inputId} or ${layerId}`);
		const W = this.w(`${layerId}.weight`);
		const B = this.w(`${layerId}.bias`);
		const C = W.shape[1];
		const k = W.shape[2];
		const H = src.shape[1];
		const Wd = src.shape[2];
		const [O, OH, OW] = outT.shape;
		if (outChannel < 0 || outChannel >= O || y < 0 || y >= OH || x < 0 || x >= OW) {
			throw new Error(`tinycnn: cell (${outChannel},${y},${x}) outside ${layerId} [${outT.shape}]`);
		}
		const s = L.params?.stride ?? 1;
		const windows: Float32Array[] = [];
		const kernels: Float32Array[] = [];
		const products: Float32Array[] = [];
		const channelSums = new Float32Array(C);
		let total = 0;
		for (let c = 0; c < C; c++) {
			const win = new Float32Array(k * k);
			const ker = new Float32Array(k * k);
			const prod = new Float32Array(k * k);
			let cs = 0;
			for (let ky = 0; ky < k; ky++) {
				for (let kx = 0; kx < k; kx++) {
					const i = ky * k + kx;
					win[i] = src.data[(c * H + y * s + ky) * Wd + x * s + kx];
					ker[i] = W.data[((outChannel * C + c) * k + ky) * k + kx];
					const pr = win[i] * ker[i];
					prod[i] = pr;
					cs += pr;
				}
			}
			windows.push(win);
			kernels.push(ker);
			products.push(prod);
			channelSums[c] = cs;
			total += cs;
		}
		const bias = B.data[outChannel];
		total = Math.fround(total + bias);
		return {
			layerId, inputId, outChannel, y, x, kernel: k, inChannels: C,
			window: { y0: y * s, x0: x * s, y1: y * s + k - 1, x1: x * s + k - 1 },
			rf: this.receptiveField(layerId, y, x),
			windows, kernels, products, channelSums, bias, total,
			traced: outT.data[(outChannel * OH + y) * OW + x],
			relu: total > 0 ? total : 0
		};
	}

	poolDetail(trace: Trace, layerId: string, channel: number, y: number, x: number): PoolDetail {
		const L = this.layer(layerId);
		if (L.op !== 'maxpool2d') throw new Error(`tinycnn: ${layerId} is not a pool layer`);
		const inputId = L.in[0];
		const src = trace.t[inputId];
		const outT = trace.t[layerId];
		const k = L.params?.kernel ?? 2;
		const s = L.params?.stride ?? k;
		const H = src.shape[1];
		const W = src.shape[2];
		const OH = outT.shape[1];
		const OW = outT.shape[2];
		const values = new Float32Array(k * k);
		let argmax = 0;
		for (let ky = 0; ky < k; ky++) {
			for (let kx = 0; kx < k; kx++) {
				const i = ky * k + kx;
				values[i] = src.data[(channel * H + y * s + ky) * W + x * s + kx];
				if (values[i] > values[argmax]) argmax = i;
			}
		}
		return {
			layerId, inputId, channel, y, x, kernel: k,
			window: { y0: y * s, x0: x * s, y1: y * s + k - 1, x1: x * s + k - 1 },
			rf: this.receptiveField(layerId, y, x),
			values, argmax,
			argmaxAt: [y * s + Math.floor(argmax / k), x * s + (argmax % k)],
			value: values[argmax],
			traced: outT.data[(channel * OH + y) * OW + x]
		};
	}

	denseDetail(trace: Trace, classIndex: number, topK = 8): DenseDetail {
		const L = this.spec.layers.find((l) => l.op === 'dense');
		if (!L) throw new Error('tinycnn: no dense layer');
		const src = trace.t[L.in[0]];
		const W = this.w(`${L.id}.weight`);
		const B = this.w(`${L.id}.bias`);
		const I = src.data.length;
		const flat = this.spec.layers.find((l) => l.op === 'flatten');
		const nC = flat ? this.layer(flat.in[0]).outShape[0] : 1;
		const per = I / nC;
		const products = new Float32Array(I);
		const channelSums = new Float32Array(nC);
		let s = 0;
		let zeros = 0;
		for (let i = 0; i < I; i++) {
			const pr = W.data[classIndex * I + i] * src.data[i];
			products[i] = pr;
			s += pr;
			channelSums[Math.floor(i / per)] += pr;
			if (src.data[i] === 0) zeros++;
		}
		const bias = B.data[classIndex];
		const mk = (i: number): DenseContribution => {
			const p = this.flattenIndexToCHW(i);
			return { index: i, c: p.c, y: p.y, x: p.x, input: src.data[i], weight: W.data[classIndex * I + i], product: products[i] };
		};
		const idx = Array.from({ length: I }, (_, i) => i);
		const pos = idx.filter((i) => products[i] > 0).sort((a, b) => products[b] - products[a]).slice(0, topK).map(mk);
		const neg = idx.filter((i) => products[i] < 0).sort((a, b) => products[a] - products[b]).slice(0, topK).map(mk);
		return {
			layerId: L.id, classIndex, bias,
			logit: Math.fround(s + bias),
			traced: trace.t[L.id].data[classIndex],
			products, channelSums, topPositive: pos, topNegative: neg, zeroInputs: zeros
		};
	}

	softmaxDetail(trace: Trace): SoftmaxDetail {
		const L = this.spec.layers.find((l) => l.op === 'softmax');
		if (!L) throw new Error('tinycnn: no softmax layer');
		const logits = trace.t[L.in[0]].data;
		let m = -Infinity;
		for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
		const exps = new Float32Array(logits.length);
		let s = 0;
		for (let i = 0; i < logits.length; i++) {
			const e = Math.exp(logits[i] - m);
			exps[i] = e;
			s += e;
		}
		const probs = trace.t[L.id].data;
		let top = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
		return { logits, max: m, exps, sumExp: s, probs, top };
	}
}

// ---------------------------------------------------------------- preprocessing
//
// Quick, Draw! bitmaps are white strokes on black. Each drawing is scaled so the long side of its
// ink box spans 26 of the 28 pixels (1-pixel margin) and is centred in both directions; the brightest
// pixel is 255. Training feeds pixel/255. The functions below reproduce that framing.

export const IMG = 28;
/** Pixels the long side of the ink box spans after framing (dataset: 26 of 28). */
export const FIT = 26;
/** Stroke width, in 28x28 pixels, used by preprocessStrokes (dataset median effective width ~1.5). */
export const DATASET_STROKE = 1.6;
/**
 * Suggested brush width for a drawing pad, as a fraction of the canvas side, assuming the learner fills
 * most of the pad (dataset strokes are ~1.5/26 of the drawing's long side). E.g. 280 px canvas -> 15 px.
 */
export const SUGGESTED_BRUSH_FRACTION = 0.055;

/** RGBA pixels, like ImageData (data.length = 4 * width * height). */
export interface ImageLike {
	data: ArrayLike<number>;
	width: number;
	height: number;
}

/** A grayscale ink map: larger = more ink. Values in any non-negative range. */
export interface InkLike {
	ink: ArrayLike<number>;
	width: number;
	height: number;
}

export interface Preprocessed {
	/** 784 floats in [0,1], [1,28,28] row-major, ready for model.forward. */
	input: Float32Array;
	/** True when no ink was found (input is all zeros). */
	empty: boolean;
	/** Ink bounding box in source pixels (inclusive), or null when empty. */
	bbox: Box | null;
	/** The square of the source that maps onto the central 26x26 (source units; may extend past the edges). */
	crop: { x: number; y: number; size: number } | null;
}

export interface PhotoPreprocessed extends Preprocessed {
	/** True when the photo had a bright background and was inverted. */
	inverted: boolean;
	/** The cleaned ink map (0..1) at working resolution (long side <= 256 px), so a page can show the step. */
	ink: Float32Array;
}

function isImageLike(x: unknown): x is ImageLike {
	const o = x as ImageLike;
	return !!o && typeof o.width === 'number' && typeof o.height === 'number' && !!o.data && o.data.length === 4 * o.width * o.height;
}

/**
 * Ink map (0..1) from RGBA pixels.
 *  - If any pixel is not fully opaque, the canvas is treated as strokes on a transparent background: ink = alpha.
 *  - Otherwise the background brightness is read from the border: dark strokes on a light background give
 *    ink = 1 - luminance, light strokes on a dark background give ink = luminance. The background level
 *    (median of the border) is then subtracted and the rest rescaled, so any solid pad colour reads as 0 ink.
 */
export function inkFromImage(img: ImageLike): Float32Array {
	const { data, width: w, height: h } = img;
	const n = w * h;
	const ink = new Float32Array(n);
	let transparent = false;
	for (let i = 0; i < n; i++) {
		if (data[4 * i + 3] < 255) {
			transparent = true;
			break;
		}
	}
	if (transparent) {
		for (let i = 0; i < n; i++) ink[i] = data[4 * i + 3] / 255;
		return ink;
	}
	const lum = luminance(img);
	const bright = borderMean(lum, w, h) > 0.5;
	for (let i = 0; i < n; i++) ink[i] = bright ? 1 - lum[i] : lum[i];
	// Subtract the background level (median of the border) so an off-white, cream or dark-grey pad gives
	// zero ink. Without this the whole canvas counted as ink and the drawing was framed as a small patch
	// (verification 2026-10-05: dark-mode pad #1e1e1e scored 70.7%; see notes/tinycnn.md "Verification").
	const bg = borderMedian(ink, w, h);
	const span = Math.max(1 - bg, 1e-3);
	for (let i = 0; i < n; i++) {
		const t = (ink[i] - bg) / span;
		ink[i] = t > 0 ? t : 0;
	}
	return ink;
}

function borderMedian(v: ArrayLike<number>, w: number, h: number): number {
	const b: number[] = [];
	for (let x = 0; x < w; x++) b.push(v[x], v[(h - 1) * w + x]);
	for (let y = 1; y < h - 1; y++) b.push(v[y * w], v[y * w + w - 1]);
	if (!b.length) return 0;
	b.sort((p, q) => p - q);
	return b[Math.floor((b.length - 1) / 2)];
}

function luminance(img: ImageLike): Float32Array {
	const { data, width: w, height: h } = img;
	const n = w * h;
	const lum = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		const a = data[4 * i + 3] / 255;
		// composite over white, Rec. 601 luma
		const l = (0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]) / 255;
		lum[i] = a * l + (1 - a);
	}
	return lum;
}

function borderMean(v: ArrayLike<number>, w: number, h: number): number {
	let s = 0;
	let c = 0;
	for (let x = 0; x < w; x++) {
		s += v[x] + v[(h - 1) * w + x];
		c += 2;
	}
	for (let y = 1; y < h - 1; y++) {
		s += v[y * w] + v[y * w + w - 1];
		c += 2;
	}
	return c ? s / c : 0;
}

/** Ink bounding box (inclusive) of values > threshold * max, or null if there is no ink. */
export function inkBBox(ink: ArrayLike<number>, w: number, h: number, threshold = 0.05): Box | null {
	let mx = 0;
	for (let i = 0; i < w * h; i++) if (ink[i] > mx) mx = ink[i];
	if (!(mx > 0)) return null;
	const thr = threshold * mx;
	let y0 = h, x0 = w, y1 = -1, x1 = -1;
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			if (ink[y * w + x] > thr) {
				if (y < y0) y0 = y;
				if (y > y1) y1 = y;
				if (x < x0) x0 = x;
				if (x > x1) x1 = x;
			}
		}
	}
	return y1 < 0 ? null : { y0, x0, y1, x1 };
}

/** Area-averaging weights mapping source cells [0,n) onto `out` output cells covering [start, start+len). */
function areaWeights(n: number, start: number, len: number, out: number): { lo: Int32Array; w: Float64Array[] } {
	const lo = new Int32Array(out);
	const ws: Float64Array[] = [];
	const step = len / out;
	for (let i = 0; i < out; i++) {
		const a = start + i * step;
		const b = a + step;
		const i0 = Math.max(0, Math.floor(a));
		const i1 = Math.min(n, Math.ceil(b));
		lo[i] = i0;
		const w = new Float64Array(Math.max(0, i1 - i0));
		for (let j = i0; j < i1; j++) {
			const ov = Math.min(b, j + 1) - Math.max(a, j);
			w[j - i0] = ov > 0 ? ov / step : 0;
		}
		ws.push(w);
	}
	return { lo, w: ws };
}

/**
 * Frame an ink map like the dataset: square crop centred on the ink box (side = long side of the box),
 * area-averaged onto the central 26x26 of a 28x28 image, then scaled so the brightest pixel is 1.
 */
export function frameInk(ink: ArrayLike<number>, w: number, h: number, threshold = 0.05): Preprocessed {
	const bbox = inkBBox(ink, w, h, threshold);
	const input = new Float32Array(IMG * IMG);
	if (!bbox) return { input, empty: true, bbox: null, crop: null };
	const bw = bbox.x1 - bbox.x0 + 1;
	const bh = bbox.y1 - bbox.y0 + 1;
	const side = Math.max(bw, bh);
	const cx = (bbox.x0 + bbox.x1 + 1) / 2;
	const cy = (bbox.y0 + bbox.y1 + 1) / 2;
	const sx = cx - side / 2;
	const sy = cy - side / 2;
	const margin = (IMG - FIT) / 2; // 1 pixel
	// The 28 output cells cover [sx - margin*per, sx + side + margin*per) in source units.
	const per = side / FIT;
	const X = areaWeights(w, sx - margin * per, IMG * per, IMG);
	const Y = areaWeights(h, sy - margin * per, IMG * per, IMG);
	// separable: rows first (IMG x w), then columns
	const tmp = new Float64Array(IMG * w);
	for (let oy = 0; oy < IMG; oy++) {
		const wy = Y.w[oy];
		const y0 = Y.lo[oy];
		for (let j = 0; j < wy.length; j++) {
			const wt = wy[j];
			if (wt === 0) continue;
			const row = (y0 + j) * w;
			for (let x = 0; x < w; x++) tmp[oy * w + x] += wt * ink[row + x];
		}
	}
	let mx = 0;
	for (let oy = 0; oy < IMG; oy++) {
		for (let ox = 0; ox < IMG; ox++) {
			const wx = X.w[ox];
			const x0 = X.lo[ox];
			let s = 0;
			for (let j = 0; j < wx.length; j++) s += wx[j] * tmp[oy * w + x0 + j];
			input[oy * IMG + ox] = s;
			if (s > mx) mx = s;
		}
	}
	if (mx > 0) for (let i = 0; i < input.length; i++) input[i] = input[i] / mx;
	return { input, empty: mx === 0, bbox, crop: { x: sx, y: sy, size: side } };
}

/**
 * Preprocess a drawing pad canvas.
 * Accepts ImageData-like RGBA pixels, an ink map {ink, width, height}, or a plain array of n*n alpha/ink
 * values from a square canvas. Strokes may be any colour on a transparent canvas, dark on light, or light on dark.
 */
export function preprocessDrawing(src: ImageLike | InkLike | ArrayLike<number>, opts: { threshold?: number } = {}): Preprocessed {
	let ink: ArrayLike<number>;
	let w: number;
	let h: number;
	if (isImageLike(src)) {
		ink = inkFromImage(src);
		w = src.width;
		h = src.height;
	} else if ((src as InkLike).ink) {
		const s = src as InkLike;
		ink = s.ink;
		w = s.width;
		h = s.height;
	} else {
		const a = src as ArrayLike<number>;
		const n = Math.round(Math.sqrt(a.length));
		if (n * n !== a.length) throw new Error('tinycnn: plain alpha array must come from a square canvas');
		ink = a;
		w = n;
		h = n;
	}
	return frameInk(ink, w, h, opts.threshold ?? 0.05);
}

/** Area-average resize of a single-channel map. */
function resizeArea(src: ArrayLike<number>, w: number, h: number, ow: number, oh: number): Float32Array {
	const X = areaWeights(w, 0, w, ow);
	const Y = areaWeights(h, 0, h, oh);
	const tmp = new Float64Array(oh * w);
	for (let oy = 0; oy < oh; oy++) {
		const wy = Y.w[oy];
		for (let j = 0; j < wy.length; j++) {
			const row = (Y.lo[oy] + j) * w;
			for (let x = 0; x < w; x++) tmp[oy * w + x] += wy[j] * src[row + x];
		}
	}
	const out = new Float32Array(ow * oh);
	for (let oy = 0; oy < oh; oy++) {
		for (let ox = 0; ox < ow; ox++) {
			const wx = X.w[ox];
			let s = 0;
			for (let j = 0; j < wx.length; j++) s += wx[j] * tmp[oy * w + X.lo[ox] + j];
			out[oy * ow + ox] = s;
		}
	}
	return out;
}

/** Separable min (mode 'min') or box-mean (mode 'mean') filter with radius r, edges clamped. */
function filter2(a: Float32Array, w: number, h: number, r: number, mode: 'min' | 'mean'): Float32Array {
	const pass = (src: Float32Array, horizontal: boolean): Float32Array => {
		const out = new Float32Array(w * h);
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				let acc = mode === 'min' ? Infinity : 0;
				let cnt = 0;
				for (let i = -r; i <= r; i++) {
					const xx = horizontal ? x + i : x;
					const yy = horizontal ? y : y + i;
					if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
					const v = src[yy * w + xx];
					if (mode === 'min') {
						if (v < acc) acc = v;
					} else {
						acc += v;
						cnt++;
					}
				}
				out[y * w + x] = mode === 'min' ? acc : acc / cnt;
			}
		}
		return out;
	};
	return pass(pass(a, true), false);
}

/** Working resolution (long side, pixels) for photos. */
export const PHOTO_WORK_SIZE = 256;

/**
 * Preprocess an uploaded photo (e.g. a sketch on paper).
 *  1. grayscale, area-downscaled so the long side is at most 256 px;
 *  2. invert if the background (image border) is bright, so strokes are bright;
 *  3. remove uneven lighting: subtract a background estimate (min filter wider than a stroke, then a box blur);
 *  4. contrast-stretch with a robust noise floor: values below median + 6 * MAD (noise) become 0,
 *     the strongest strokes become 1;
 *  5. the same framing as drawings.
 * bbox and crop are returned in source-photo pixels; `ink` is the cleaned map at working resolution (inkWidth x inkHeight).
 */
export function preprocessPhoto(img: ImageLike, opts: { threshold?: number; noiseK?: number } = {}): PhotoPreprocessed & { inkWidth: number; inkHeight: number } {
	const { width: w, height: h } = img;
	const lumFull = luminance(img);
	const f = Math.max(1, Math.max(w, h) / PHOTO_WORK_SIZE);
	const ww = Math.max(1, Math.round(w / f));
	const wh = Math.max(1, Math.round(h / f));
	const lum = f > 1 ? resizeArea(lumFull, w, h, ww, wh) : lumFull;
	const n = ww * wh;
	const inverted = borderMean(lum, ww, wh) > 0.5;
	const raw = new Float32Array(n);
	for (let i = 0; i < n; i++) raw[i] = inverted ? 1 - lum[i] : lum[i];
	// background: strokes are thin and bright, so a min filter wider than a stroke removes them
	const rMin = Math.max(2, Math.round(Math.max(ww, wh) / 50));
	const bg = filter2(filter2(raw, ww, wh, rMin, 'min'), ww, wh, 2 * rMin, 'mean');
	const flat = new Float32Array(n);
	for (let i = 0; i < n; i++) flat[i] = raw[i] - bg[i];
	// robust noise floor from the (mostly background) pixels
	const sorted = Float32Array.from(flat).sort();
	const med = sorted[Math.floor(0.5 * (n - 1))];
	const dev = new Float32Array(n);
	for (let i = 0; i < n; i++) dev[i] = Math.abs(flat[i] - med);
	dev.sort();
	const mad = dev[Math.floor(0.5 * (n - 1))] * 1.4826;
	const floor = med + (opts.noiseK ?? 6) * Math.max(mad, 1e-3);
	// stroke level: maximum of a 3x3 box-blurred map (robust to single hot pixels)
	const sm = filter2(flat, ww, wh, 1, 'mean');
	let hi = 0;
	for (let i = 0; i < n; i++) if (sm[i] > hi) hi = sm[i];
	const span = Math.max(hi - floor, 1e-3);
	const ink = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		const v = (flat[i] - floor) / span;
		ink[i] = v <= 0 ? 0 : v > 1 ? 1 : v;
	}
	const r = frameInk(ink, ww, wh, opts.threshold ?? 0.1);
	const sx = w / ww;
	const sy = h / wh;
	return {
		input: r.input,
		empty: r.empty,
		bbox: r.bbox && {
			y0: Math.floor(r.bbox.y0 * sy), x0: Math.floor(r.bbox.x0 * sx),
			y1: Math.min(h - 1, Math.ceil((r.bbox.y1 + 1) * sy) - 1), x1: Math.min(w - 1, Math.ceil((r.bbox.x1 + 1) * sx) - 1)
		},
		crop: r.crop && { x: r.crop.x * sx, y: r.crop.y * sy, size: r.crop.size * Math.max(sx, sy) },
		inverted,
		ink,
		inkWidth: ww,
		inkHeight: wh
	};
}

export type Point = [number, number] | { x: number; y: number };

/**
 * Preprocess a drawing given as strokes (lists of points in any coordinate system, y down).
 * The strokes are rendered straight into 28x28 with the dataset's stroke width, so the result does not
 * depend on the pad's brush size. Antialiased by supersampling (default 4x4 per pixel).
 */
export function preprocessStrokes(strokes: Point[][], opts: { lineWidth?: number; samples?: number; frame?: [number, number, number, number] } = {}): Preprocessed {
	const lw = opts.lineWidth ?? DATASET_STROKE;
	const S = opts.samples ?? 4;
	const pts: [number, number][][] = strokes
		.map((s) => s.map((p): [number, number] => (Array.isArray(p) ? [p[0], p[1]] : [(p as { x: number }).x, (p as { y: number }).y])))
		.filter((s) => s.length > 0);
	const input = new Float32Array(IMG * IMG);
	if (pts.length === 0) return { input, empty: true, bbox: null, crop: null };
	let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
	if (opts.frame) {
		// DevKit K1: a fixed frame (the canvas, [x0, y0, x1, y1]) instead of the ink's box, so size and place survive
		[x0, y0, x1, y1] = opts.frame;
	} else
	for (const s of pts) {
		for (const [x, y] of s) {
			if (x < x0) x0 = x;
			if (x > x1) x1 = x;
			if (y < y0) y0 = y;
			if (y > y1) y1 = y;
		}
	}
	const side = Math.max(x1 - x0, y1 - y0, 1e-6);
	// The centre-line box maps onto a (FIT - lw)-pixel box centred in the image, so the ink
	// (centre line +- lw/2) fills the 26-pixel box like the dataset.
	const r = lw / 2;
	const sc = (FIT - lw) / side;
	const ox = IMG / 2 - ((x0 + x1) / 2) * sc;
	const oy = IMG / 2 - ((y0 + y1) / 2) * sc;
	const segs: number[] = [];
	for (const s of pts) {
		if (s.length === 1) segs.push(s[0][0] * sc + ox, s[0][1] * sc + oy, s[0][0] * sc + ox, s[0][1] * sc + oy);
		for (let i = 1; i < s.length; i++) {
			segs.push(s[i - 1][0] * sc + ox, s[i - 1][1] * sc + oy, s[i][0] * sc + ox, s[i][1] * sc + oy);
		}
	}
	const r2 = r * r;
	let mx = 0;
	for (let py = 0; py < IMG; py++) {
		for (let px = 0; px < IMG; px++) {
			// skip pixels far from every segment's box
			let near = false;
			for (let k = 0; k < segs.length; k += 4) {
				if (px + 1 + r >= Math.min(segs[k], segs[k + 2]) && px - r <= Math.max(segs[k], segs[k + 2]) &&
					py + 1 + r >= Math.min(segs[k + 1], segs[k + 3]) && py - r <= Math.max(segs[k + 1], segs[k + 3])) {
					near = true;
					break;
				}
			}
			if (!near) continue;
			let hit = 0;
			for (let sy = 0; sy < S; sy++) {
				for (let sx = 0; sx < S; sx++) {
					const qx = px + (sx + 0.5) / S;
					const qy = py + (sy + 0.5) / S;
					for (let k = 0; k < segs.length; k += 4) {
						if (distSeg2(qx, qy, segs[k], segs[k + 1], segs[k + 2], segs[k + 3]) <= r2) {
							hit++;
							break;
						}
					}
				}
			}
			const v = hit / (S * S);
			input[py * IMG + px] = v;
			if (v > mx) mx = v;
		}
	}
	if (mx > 0) for (let i = 0; i < input.length; i++) input[i] /= mx;
	return {
		input,
		empty: mx === 0,
		bbox: { y0, x0, y1, x1 },
		crop: { x: (x0 + x1) / 2 - side / 2, y: (y0 + y1) / 2 - side / 2, size: side }
	};
}

function distSeg2(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
	const dx = bx - ax;
	const dy = by - ay;
	const L = dx * dx + dy * dy;
	let t = L > 0 ? ((px - ax) * dx + (py - ay) * dy) / L : 0;
	t = t < 0 ? 0 : t > 1 ? 1 : t;
	const ex = ax + t * dx - px;
	const ey = ay + t * dy - py;
	return ex * ex + ey * ey;
}

// ---------------------------------------------------------------- presets

/** Decode a preset (base64 of 784 uint8, white-on-black) into the model input (values / 255). */
export function decodePreset(b64: string): Float32Array {
	const bin = atob(b64);
	const out = new Float32Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255;
	return out;
}

/** Decode base64 of uint8 values into a Uint8Array (for channels.json patches). */
export function decodeBytes(b64: string): Uint8Array {
	const bin = atob(b64);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}
