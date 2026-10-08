// One demo per core op for /devkit/reference/ops/: the layer of a demo model that uses the op, drawn by the op's own zoom
// view. Pure TS, Node-safe (devkit-site.test.ts checks that every op in schema/ops.json has a demo that runs).
//
// The demo models are the gallery's (atlas/ui/views/kit.parts/demo.ts: a tiny CNN, a tiny transformer, an LSTM) with the
// same layers and units the gallery's view variants use. Four ops appear in none of them (tanh, sigmoid, silu, concat):
// they get `acts`, one more tiny graph built here the same way (made-up input, the real kit ops through fromParts).
// Every demo is badged 'illustrative'.
import type { AtlasLayer, AtlasSpec, I18n, StageProps } from '../atlas/types/index.ts';
import type { At } from '../atlas/types/lesson.ts';
import type { Focus } from '../atlas/types/process.ts';
import { demoModel, demoStage, type DemoModel, type DemoName } from '../atlas/ui/views/kit.parts/demo.ts';
import { fromParts } from '../atlas/graph/executor.ts';
import { CORE_OPS } from '../atlas/ops/index.ts';
import { fakeRuntime } from '../atlas/runtime/fake.ts';
import { atFor } from '../atlas/runtime/sources.ts';

export type DemoRef = { model: DemoName | 'acts'; layer: string; unit?: number[] };

/** op -> the demo layer (and unit) that shows it. */
export const OP_DEMOS: Record<string, DemoRef> = {
	input: { model: 'cnn', layer: 'input' },
	conv2d: { model: 'cnn', layer: 'conv1' },
	linear: { model: 'cnn', layer: 'dense', unit: [1] },
	relu: { model: 'cnn', layer: 'relu1', unit: [1, 2, 3] },
	tanh: { model: 'acts', layer: 'tanh', unit: [4] },
	sigmoid: { model: 'acts', layer: 'sigmoid', unit: [1] },
	silu: { model: 'acts', layer: 'silu', unit: [2] },
	gelu: { model: 'gpt', layer: 'gelu', unit: [4, 9] },
	maxpool2d: { model: 'cnn', layer: 'pool' },
	flatten: { model: 'cnn', layer: 'flatten' },
	concat: { model: 'acts', layer: 'concat', unit: [8] },
	add: { model: 'gpt', layer: 'resid', unit: [4, 2] },
	embedding: { model: 'gpt', layer: 'tok_emb', unit: [2, 3] },
	position_embedding: { model: 'gpt', layer: 'pos_emb', unit: [4, 1] },
	layernorm: { model: 'gpt', layer: 'ln' },
	attention_scores: { model: 'gpt', layer: 'scores', unit: [1, 5, 2] },
	causal_mask: { model: 'gpt', layer: 'masked', unit: [0, 2, 4] },
	softmax: { model: 'cnn', layer: 'probs', unit: [1] },
	attention_mix: { model: 'gpt', layer: 'heads', unit: [5, 6] },
	lstm: { model: 'lstm', layer: 'lstm', unit: [3, 1] }
};

const L = (en: string, zh: string): I18n => ({ en, zh });

/** The made-up input of the `acts` graph: six numbers from clearly negative to clearly positive. */
export const ACTS_INPUT = [-2.5, -1, -0.3, 0.4, 1.2, 3];

let actsMemo: DemoModel | null = null;
/** input [6] -> tanh, sigmoid, silu (each on the input) and concat(input, tanh) [12]. No weights. */
export function actsModel(): DemoModel {
	if (actsMemo) return actsMemo;
	const n = ACTS_INPUT.length;
	const f = ['f'] as const;
	const layers: AtlasLayer[] = [
		{ id: 'input', op: 'input', label: L('Input', '输入'), in: ['x'], outShape: [n], axes: [...f] },
		{ id: 'tanh', op: 'tanh', label: L('Tanh', 'Tanh'), in: ['input'], outShape: [n], axes: [...f] },
		{ id: 'sigmoid', op: 'sigmoid', label: L('Sigmoid', 'Sigmoid'), in: ['input'], outShape: [n], axes: [...f] },
		{ id: 'silu', op: 'silu', label: L('SiLU', 'SiLU'), in: ['input'], outShape: [n], axes: [...f] },
		{ id: 'concat', op: 'concat', label: L('Input and tanh, joined', '输入与 tanh 拼接'), in: ['input', 'tanh'], outShape: [2 * n], params: { axis: 0 }, axes: [...f] }
	];
	const spec: AtlasSpec = {
		slug: 'demo-acts',
		name: L('Demo activations', '演示：激活函数'),
		version: 'demo',
		params: 0,
		precision: 'fp32',
		tensors: [],
		layers,
		inputs: [{ name: 'x', adapter: 'demo', kind: 'series', shape: [n], dtype: 'f32', axes: [...f] }],
		outputs: ['concat'],
		parity: { against: 'none (demo)', max_abs_diff: null, checked: null },
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'demo' }
	};
	const model = fromParts(spec, {}, CORE_OPS);
	const trace = model.forward({ shape: [n], data: Float32Array.from(ACTS_INPUT) });
	actsMemo = { model, trace, spec: model.spec };
	return actsMemo;
}

/** The view a layer of this op is drawn with: its zoom view ('none' -> layer.tensor), as the gallery picks it. */
export function zoomKeyOf(zoom: string): string {
	return zoom === 'none' ? 'layer.tensor' : zoom.startsWith('zoom.') ? zoom : `zoom.${zoom}`;
}

/** StageProps for an op's demo (provenance 'illustrative'), or null when the op has none. */
export function opStage(op: string): Omit<StageProps, 'ctx'> | null {
	const d = OP_DEMOS[op];
	if (!d) return null;
	if (d.model !== 'acts') return demoStage(d.model, d.layer, d.unit);
	// demoStage() for the acts graph (demo.ts takes only its own models)
	const m = actsModel();
	const rt = fakeRuntime({ slug: m.spec.slug, spec: m.spec, model: m.model, trace: m.trace });
	const sel = d.unit ? { layer: d.layer, unit: d.unit } : { layer: d.layer };
	rt.sel = sel;
	const index = rt.sources.fwd.tape!.find({ layer: d.layer });
	const at: At = index >= 0 ? atFor(rt, 'fwd', index) : { index: -1 };
	if (d.unit && !at.unit) at.unit = m.model.explain(m.trace, d.layer, d.unit);
	at.layer = d.layer;
	const focus: Focus[] = d.unit ? [{ field: d.layer, index: d.unit, role: 'out' }] : [];
	return { rt, source: 'fwd', at, focus, sel, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
}

/** The demo layer of an op (for its PyTorch line), or null. */
export function opLayer(op: string): { layer: AtlasLayer; spec: AtlasSpec } | null {
	const d = OP_DEMOS[op];
	if (!d) return null;
	const spec = d.model === 'acts' ? actsModel().spec : demoModel(d.model).spec;
	const layer = spec.layers.find((l) => l.id === d.layer);
	return layer ? { layer, spec } : null;
}
