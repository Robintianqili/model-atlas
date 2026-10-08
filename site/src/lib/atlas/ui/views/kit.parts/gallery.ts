// Everything the /dev/atlas/ gallery needs, in ONE lazily imported module (Vite only: import.meta.glob).
//
// Why one module: every lazy chunk that imports kit components changes how Rollup splits the chunks the flagship pages
// share. Fifteen separate view chunks (plus the tokenizer) split /m/tinycnn/'s shell into 8 more files and pushed it
// from 298.9 KB to 302.5 KB gzip, over its 300 KB budget. Bundling the views (eager globs) and the real-model loader
// into one dynamic entry keeps the gallery's footprint on the other pages small; the TinyGPT inputs are pre-tokenized
// ids, so the gallery does not import models/bpe.ts (a module /m/tinygpt/ preloads).
import type { Component } from 'svelte';
import type { AtlasLayer, GraphModel, I18n, StageProps, Trace, ViewMeta } from '../../../types/index.ts';
import { loadGraph } from '../../../graph/executor.ts';
import { CORE_OPS } from '../../../ops/index.ts';
import { b64ToBytes } from '../../../core/io.ts';
import { fakeRuntime } from '../../../runtime/fake.ts';
import { traceToTape } from '../../../process/adapters.ts';
import { atFor } from '../../../runtime/sources.ts';
import { demoStage } from './demo.ts';
export { DEMO_NOTE } from './demo.ts';
export { default as LessonDemo } from './LessonDemo.svelte';
export { default as ViewStage } from './ViewStage.svelte';

export type Stage = Omit<StageProps, 'ctx'>;
export type MetaMod = { default: ViewMeta; variants?: () => Promise<{ label: I18n; stage: Stage }[]> };

const kitMetas = import.meta.glob<MetaMod>('../*/meta.ts', { eager: true });
const kitViews = import.meta.glob<Component<StageProps>>('../*/View.svelte', { eager: true, import: 'default' });
const pkgMetas = import.meta.glob<MetaMod>('../../../../explainers/*/views/*/meta.ts', { eager: true });
const pkgViews = import.meta.glob<Component<StageProps>>('../../../../explainers/*/views/*/View.svelte', { eager: true, import: 'default' });

export interface ViewEntry {
	key: string;
	mod: MetaMod;
	comp: Component<StageProps> | null;
}

/** Every kit view ('<key>') and package view ('<slug>/<key>'), sorted by key. */
export function allViews(): ViewEntry[] {
	const out: ViewEntry[] = [];
	for (const [path, mod] of Object.entries(kitMetas)) {
		const key = /\/([^/]+)\/meta\.ts$/.exec(path)?.[1] ?? path;
		out.push({ key, mod, comp: kitViews[path.replace(/meta\.ts$/, 'View.svelte')] ?? null });
	}
	for (const [path, mod] of Object.entries(pkgMetas)) {
		const m = /explainers\/([^/]+)\/views\/([^/]+)\/meta\.ts$/.exec(path);
		out.push({ key: m ? `${m[1]}/${m[2]}` : path, mod, comp: pkgViews[path.replace(/meta\.ts$/, 'View.svelte')] ?? null });
	}
	return out.sort((a, b) => a.key.localeCompare(b.key));
}

export function viewFor(key: string): Component<StageProps> | null {
	return allViews().find((v) => v.key === key)?.comp ?? null;
}

/** The zoom view a layer's op asks for (layer.ui.zoom wins); 'none' → layer.tensor. */
export function zoomKey(model: GraphModel, l: AtlasLayer): string {
	try {
		const z = l.ui?.zoom ?? model.ops.get(l.op).def.zoom;
		return z === 'none' ? 'layer.tensor' : z.startsWith('zoom.') || z.includes('/') ? z : `zoom.${z}`;
	} catch {
		return 'layer.tensor';
	}
}

/** Four demo stages in a kit Walkthrough (LessonDemo): the conv slide plays itself and calls ctx.done(). */
export function lessonItems() {
	const vs = allViews();
	const get = (k: string) => {
		const v = vs.find((x) => x.key === k);
		if (!v?.comp) throw new Error(`gallery: view ${k} is missing`);
		return { meta: v.mod.default, view: v.comp };
	};
	return [
		{ id: 'slide', title: { en: 'A kernel slides over the drawing', zh: '卷积核在画上滑动' }, ...get('zoom.weighted-sum'), stage: demoStage('cnn', 'conv1', undefined, { interval: 90 }), waitForDone: true },
		{ id: 'bend', title: { en: 'ReLU cuts the negative numbers', zh: 'ReLU 去掉负数' }, ...get('zoom.elementwise'), stage: demoStage('cnn', 'relu1', [1, 2, 3]) },
		{ id: 'pool', title: { en: 'Pooling keeps the largest', zh: '池化保留最大的数' }, ...get('zoom.reduce'), stage: demoStage('cnn', 'pool', undefined, { slide: false }) },
		{ id: 'answer', title: { en: 'Scores become shares', zh: '分数变成份额' }, ...get('kit.probbars'), stage: demoStage('cnn', 'probs') }
	];
}

// ------------------------------------------------ real models
export type RealName = 'tinycnn' | 'tinygpt';
export interface RealRun {
	model: GraphModel;
	trace: Trace;
	tokens?: string[];
	vocab?: (id: number) => string;
}
export interface RealModel {
	model: GraphModel;
	inputs: { id: string; label: I18n }[];
	run(i: number): RealRun;
}

/** TinyGPT inputs, tokenized offline with models/bpe.ts (BPETokenizer.encode on static/models/tinygpt/tokenizer.json). */
const TEXTS: { text: string; ids: number[] }[] = [
	{ text: 'Once upon a time, there was a little girl named Lily. She loved to play outside', ids: [428, 437, 258, 397, 44, 400, 282, 258, 390, 474, 401, 404, 46, 338, 501, 265, 324, 696] },
	{ text: 'The cat sat on the mat.', ids: [409, 458, 291, 116, 353, 262, 280, 296, 46] },
	{ text: 'Tom and Sue went to the park.', ids: [735, 266, 509, 427, 265, 262, 522, 46] }
];

type FetchFn = (url: string) => Promise<Response>;

export async function loadReal(fetch: FetchFn, base: string, name: RealName): Promise<RealModel> {
	const url = `${base}/models/${name}`;
	const model = await loadGraph(fetch, url, CORE_OPS);
	if (name === 'tinycnn') {
		const presets = (await (await fetch(`${url}/presets.json`)).json()) as { classes: { id: string; en: string; zh: string; items: { b64: string }[] }[] };
		return {
			model,
			inputs: presets.classes.map((c) => ({ id: c.id, label: { en: `a drawn ${c.en} (Quick, Draw! test set)`, zh: `画的${c.zh}（Quick, Draw! 测试集）` } })),
			run(i) {
				const c = presets.classes[Math.max(0, Math.min(presets.classes.length - 1, i))];
				const x = Float32Array.from(b64ToBytes(c.items[0].b64), (v) => v / 255);
				return { model, trace: model.forward({ shape: [1, 28, 28], data: x }) };
			}
		};
	}
	const tok = (await (await fetch(`${url}/tokenizer.json`)).json()) as { tokenBytes: string[]; specialTokens?: Record<string, number> };
	const special = new Map(Object.entries(tok.specialTokens ?? {}).map(([k, v]) => [v, k]));
	const dec = new TextDecoder('utf-8');
	const text = (id: number) => {
		const sp = special.get(id);
		if (sp) return sp;
		const hex = tok.tokenBytes[id] ?? '';
		const bytes = new Uint8Array(hex.length / 2);
		for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
		return dec.decode(bytes);
	};
	return {
		model,
		inputs: TEXTS.map((x, i) => ({ id: String(i), label: { en: `“${x.text}”`, zh: `“${x.text}”` } })),
		run(i) {
			const ids = TEXTS[Math.max(0, Math.min(TEXTS.length - 1, i))].ids;
			const trace = model.forward({ input: { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' } });
			return { model, trace, tokens: ids.map(text), vocab: text };
		}
	};
}

/** StageProps for a layer of a real run: the At a lesson step on that layer gets (provenance 'live'). */
export function realStage(r: RealRun, layer: string): Stage {
	const spec = r.model.spec;
	const rt = fakeRuntime({ slug: spec.slug, spec, model: r.model });
	rt.sources = { fwd: { kind: 'trace', status: 'ready', provenance: 'live', trace: r.trace, tape: traceToTape(spec, r.trace), model: r.model } };
	if (r.tokens) rt.pre = { tensors: {}, steps: [], empty: false, tokens: r.tokens } as StageProps['rt']['pre'];
	rt.sel = { layer };
	const at = atFor(rt, 'fwd', rt.sources.fwd.tape!.find({ layer }));
	at.layer = layer;
	const props: Record<string, unknown> = {};
	if (r.tokens) props.tokens = r.tokens;
	if (r.vocab) props.vocab = r.vocab;
	return { rt, source: 'fwd', at, focus: [], sel: rt.sel, provenance: 'live', autoplay: false, onend: () => {}, props };
}
