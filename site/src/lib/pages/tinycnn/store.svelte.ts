/**
 * Page state for /m/tinycnn/: the input (pad strokes, preset or photo), the live forward pass and
 * the selected layer / unit. One instance per page, passed to the page components as a prop.
 *
 * The model module ($lib/models/tinycnn.ts) is imported lazily in init(), so the prerendered page
 * shell does not carry it. Every number shown on the page comes from `trace`, which is recomputed
 * (one forward pass, ~1-2 ms) whenever the input changes.
 */
import type { Stroke } from '$lib/kit/draw';
import type { ColorScope, Domain } from '$lib/kit/color';
import type { Preprocessed, TinyCNN } from '$lib/models/tinycnn';
import {
	argmaxCell,
	eraseHalf,
	isSpatial,
	simplifyStroke,
	traceDomains,
	type Box,
	type ChannelsFile,
	type EraseMode,
	type HeldOutFile,
	type PresetsFile,
	type TinyCNNSpec,
	type Trace
} from './data';

type Mod = typeof import('$lib/models/tinycnn');

export type Source =
	| { kind: 'pad' }
	| { kind: 'preset'; cls: string; idx: number; heldOut: boolean }
	| {
			kind: 'photo';
			name: string;
			url: string;
			w: number;
			h: number;
			inverted: boolean;
			ink: Float32Array;
			inkW: number;
			inkH: number;
			crop: { x: number; y: number; size: number } | null;
	  };

export interface Selection {
	layer: string;
	/** channel for feature maps, class for dense / softmax */
	ch: number;
	y: number;
	x: number;
}

/**
 * Something that holds a selection: the page store itself (overview / zoom / inspector) or a
 * LessonView (the guided lesson keeps its own, so the two never move each other). The zoom
 * components take an optional `view` prop and default to the store.
 */
export interface SelView {
	readonly sel: Selection;
	setCell(y: number, x: number): void;
	select(layer: string, ch?: number, cell?: { y: number; x: number }): void;
}

const EMPTY: Preprocessed = { input: new Float32Array(784), empty: true, bbox: null, crop: null };

export class CnnStore {
	spec: TinyCNNSpec;
	mod: Mod | null = null;
	model = $state.raw<TinyCNN | null>(null);
	presets = $state.raw<PresetsFile | null>(null);
	channels = $state.raw<ChannelsFile | null>(null);
	heldout = $state.raw<HeldOutFile | null>(null);
	loadError = $state<string | null>(null);
	/** bytes actually fetched so far (measured) */
	fetched = $state(0);

	/** pad strokes (normalized 0..1), bound to DrawPad */
	strokes = $state.raw<Stroke[]>([]);
	source = $state.raw<Source>({ kind: 'pad' });
	/** preprocessing result of the current source (before the erase experiment) */
	pre = $state.raw<Preprocessed>(EMPTY);
	erase = $state<EraseMode>('none');

	/** the exact model input: 784 numbers in [0, 1] */
	input = $derived(eraseHalf(this.pre.input, this.erase));
	/** the live forward pass and how long it took */
	run = $derived.by(() => {
		if (!this.model) return null;
		const t0 = performance.now();
		const trace = this.model.forward(this.input);
		return { trace, ms: performance.now() - t0 };
	});
	trace = $derived<Trace | null>(this.run?.trace ?? null);
	domains = $derived.by(() => (this.trace ? traceDomains(this.spec, this.trace) : null));
	probs = $derived(this.trace?.t.softmax.data ?? null);
	top = $derived.by(() => {
		const p = this.probs;
		if (!p) return -1;
		let k = 0;
		for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
		return k;
	});

	sel = $state<Selection>({ layer: 'conv1_1', ch: 0, y: 0, x: 0 });
	/** color scope of the feature maps: this map / whole layer / all layers */
	scope = $state<ColorScope>('layer');
	/** receptive field (in the 28x28 input) to outline on the input views while hovering */
	hoverRF = $state.raw<Box | null>(null);

	private pending: Stroke[] | null = null;
	private raf = 0;

	constructor(spec: TinyCNNSpec) {
		this.spec = spec;
	}

	layer(id: string) {
		return this.spec.layers.find((l) => l.id === id)!;
	}

	/** Load the model code and the files this page needs. */
	async init(baseUrl: string, firstStrokes: Stroke[]): Promise<void> {
		try {
			const get = async (f: string) => {
				const r = await fetch(`${baseUrl}/${f}`);
				if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
				const buf = await r.arrayBuffer();
				this.fetched += buf.byteLength;
				return buf;
			};
			const json = async <T>(f: string): Promise<T> => JSON.parse(new TextDecoder().decode(await get(f))) as T;
			const [mod, weights, presets, channels] = await Promise.all([
				import('$lib/models/tinycnn'),
				get('weights.bin'),
				json<PresetsFile>('presets.json'),
				json<ChannelsFile>('channels.json')
			]);
			this.mod = mod;
			this.presets = presets;
			this.channels = channels;
			this.model = mod.fromParts(this.spec, weights);
			this.setStrokes(firstStrokes);
			this.select('conv1_1', 0);
		} catch (e) {
			this.loadError = e instanceof Error ? e.message : String(e);
		}
	}

	async loadHeldout(baseUrl: string): Promise<HeldOutFile | null> {
		if (this.heldout) return this.heldout;
		const r = await fetch(`${baseUrl}/heldout.json`);
		if (!r.ok) return null;
		const buf = await r.arrayBuffer();
		this.fetched += buf.byteLength;
		this.heldout = JSON.parse(new TextDecoder().decode(buf)) as HeldOutFile;
		return this.heldout;
	}

	// ------------------------------------------------------------ inputs

	/** Preprocess strokes exactly like the dataset (stroke path) and make the pad the source. */
	setStrokes(strokes: Stroke[]) {
		if (strokes !== this.strokes) this.strokes = strokes;
		if (this.source.kind !== 'pad') this.source = { kind: 'pad' };
		if (!this.mod) return;
		const lines = strokes.map((s) => simplifyStroke(s.points)).filter((p) => p.length > 0);
		this.pre = lines.length ? this.mod.preprocessStrokes(lines) : EMPTY;
	}

	/** Called by DrawPad on every animation frame while drawing: coalesce to one update per frame. */
	drawing(strokes: Stroke[]) {
		this.pending = strokes;
		if (this.source.kind !== 'pad') this.source = { kind: 'pad' };
		if (this.raf) return;
		this.raf = requestAnimationFrame(() => {
			this.raf = 0;
			const s = this.pending;
			this.pending = null;
			if (!s || !this.mod) return;
			const lines = s.map((st) => simplifyStroke(st.points)).filter((p) => p.length > 0);
			this.pre = lines.length ? this.mod.preprocessStrokes(lines) : EMPTY;
		});
	}

	loadPreset(cls: string, idx: number) {
		if (!this.presets || !this.mod) return;
		const held = this.presets.heldOut.find((g) => g.id === cls);
		const g = held ?? this.presets.classes.find((c) => c.id === cls);
		if (!g) return;
		const i = Math.max(0, Math.min(g.items.length - 1, idx));
		this.strokes = [];
		this.source = { kind: 'preset', cls, idx: i, heldOut: !!held };
		this.pre = { input: this.mod.decodePreset(g.items[i].b64), empty: false, bbox: null, crop: null };
	}

	presetInput(cls: string, idx: number): Float32Array | null {
		if (!this.presets || !this.mod) return null;
		const g = this.presets.heldOut.find((c) => c.id === cls) ?? this.presets.classes.find((c) => c.id === cls);
		const it = g?.items[idx];
		return it ? this.mod.decodePreset(it.b64) : null;
	}

	/** Decode a photo, scale it down (long side <= 1024 px: keeps preprocessing well under 50 ms) and preprocess it. */
	async loadPhoto(file: File): Promise<string | null> {
		if (!this.mod) return 'model not loaded';
		if (!file.type.startsWith('image/')) return 'not an image';
		const url = URL.createObjectURL(file);
		try {
			const img = new Image();
			img.decoding = 'async';
			img.src = url;
			await img.decode();
			const f = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
			const w = Math.max(1, Math.round(img.naturalWidth * f));
			const h = Math.max(1, Math.round(img.naturalHeight * f));
			const c = document.createElement('canvas');
			c.width = w;
			c.height = h;
			const ctx = c.getContext('2d', { willReadFrequently: true });
			if (!ctx) return 'no canvas';
			ctx.fillStyle = '#fff';
			ctx.fillRect(0, 0, w, h);
			ctx.drawImage(img, 0, 0, w, h);
			const r = this.mod.preprocessPhoto(ctx.getImageData(0, 0, w, h));
			if (this.source.kind === 'photo') URL.revokeObjectURL(this.source.url);
			this.strokes = [];
			this.source = {
				kind: 'photo',
				name: file.name,
				url,
				w,
				h,
				inverted: r.inverted,
				ink: r.ink,
				inkW: r.inkWidth,
				inkH: r.inkHeight,
				crop: r.crop
			};
			this.pre = r;
			return r.empty ? 'empty' : null;
		} catch {
			URL.revokeObjectURL(url);
			return 'decode';
		}
	}

	// ------------------------------------------------------------ selection

	/** Select a layer and unit. Feature maps default to the unit's strongest cell. */
	select(layer: string, ch = this.sel.ch, cell?: { y: number; x: number }) {
		const s = this.resolve(layer, ch, cell);
		if (s) this.sel = s;
	}

	/** The selection for a layer and unit (clamped); feature maps default to the unit's strongest cell. */
	resolve(layer: string, ch: number, cell?: { y: number; x: number }): Selection | null {
		const l = this.layer(layer);
		if (!l) return null;
		const units = isSpatial(l) || l.op === 'dense' || l.op === 'softmax' ? l.outShape[0] : 1;
		const c = Math.max(0, Math.min(units - 1, ch));
		let y = 0;
		let x = 0;
		if (isSpatial(l)) {
			const [, H, W] = l.outShape;
			if (cell) {
				y = Math.max(0, Math.min(H - 1, cell.y));
				x = Math.max(0, Math.min(W - 1, cell.x));
			} else if (this.trace && l.op !== 'input') {
				({ y, x } = argmaxCell(this.trace.t[layer].data, c, H, W));
			} else {
				y = Math.floor(H / 2);
				x = Math.floor(W / 2);
			}
		}
		return { layer, ch: c, y, x };
	}

	/** Channel of a [C, H, W] layer with the largest single value in the current trace (0 before the first run). */
	strongestChannel(layerId: string): number {
		const l = this.layer(layerId);
		const d = this.trace?.t[layerId]?.data;
		if (!l || !d || !isSpatial(l)) return 0;
		const [C, H, W] = l.outShape;
		let best = 0;
		let bv = -Infinity;
		for (let c = 0; c < C; c++)
			for (let i = 0; i < H * W; i++)
				if (d[c * H * W + i] > bv) {
					bv = d[c * H * W + i];
					best = c;
				}
		return best;
	}

	setCell(y: number, x: number) {
		if (this.sel.y !== y || this.sel.x !== x) this.sel = { ...this.sel, y, x };
	}

	/** Domain for a feature map given the current color scope. */
	domainFor(layerId: string, own: () => Domain): Domain {
		const d = this.domains;
		const l = this.layer(layerId);
		if (l.op === 'input') return [0, 1];
		if (!d) return own();
		if (this.scope === 'global' && isSpatial(l)) return d.global;
		if (this.scope === 'unit') return own();
		return d.layer[layerId] ?? own();
	}

	receptiveField(layerId: string, y: number, x: number): Box | null {
		return this.model ? this.model.receptiveField(layerId, y, x) : null;
	}
}

/** The guided lesson's own selection (see SelView). */
export class LessonView implements SelView {
	S: CnnStore;
	sel = $state<Selection>({ layer: 'input', ch: 0, y: 14, x: 14 });
	/** true once the learner picked a unit or class inside the lesson (the lesson stops following the answer) */
	picked = $state(false);

	constructor(S: CnnStore) {
		this.S = S;
	}
	setCell(y: number, x: number) {
		if (this.sel.y !== y || this.sel.x !== x) this.sel = { ...this.sel, y, x };
	}
	/** a pick made by the learner (class chips, kernel buttons, …) */
	select(layer: string, ch = this.sel.ch, cell?: { y: number; x: number }) {
		const s = this.S.resolve(layer, ch, cell);
		if (s) {
			this.picked = true;
			this.sel = s;
		}
	}
	/** a selection made by the lesson itself (step hooks) */
	set(layer: string, ch: number, cell?: { y: number; x: number }) {
		const s = this.S.resolve(layer, ch, cell);
		if (s) {
			this.picked = false;
			this.sel = s;
		}
	}
}
