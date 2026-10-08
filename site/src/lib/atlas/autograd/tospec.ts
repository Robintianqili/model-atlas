// Linear/MLP stacks -> spec 1.1 + float32 weights (design §4.5, W1-B1): the kind-A explainer on a live model
// ('live-model' sources). A custom Module gets a live-model view by writing toSpec() with stackToSpec (a list of
// { kind: 'linear', id, weight, bias } and { kind: 'act', id, act } items, in forward() order); the trainer's
// liveSpec() calls it on the rebuilt model. Layer ids are the PyTorch module names ('net.0', 'net.1'), so weight tensors resolve by the
// spec default `${id}.${slot}` and match the trainer's parameter names. Weights are stored as f32 (CONTRACTS weights.bin).
import type { AtlasLayer, AtlasSpec, Axis, I18n, InputDecl, TensorEntry } from '../types/index.ts';
import type { Var } from './var.ts';

export type SpecAct = 'tanh' | 'relu' | 'sigmoid' | 'silu' | 'gelu' | 'gelu_tanh' | 'softmax' | 'identity';

export type SpecStackItem = { kind: 'linear'; id: string; weight: Var; bias: Var | null } | { kind: 'act'; id: string; act: SpecAct };

export interface ToSpecOpts {
	slug?: string;
	/** axes of the (single) input, default ['f'] */
	inputAxes?: string[];
	name?: I18n;
	version?: string;
	/** the input adapter and catalog InputKind, default 'vector' */
	adapter?: string;
	inputKind?: string;
	/** several inputs joined by a concat layer first (flow-2d: [{ name: 'xy', size: 2 }, { name: 't', size: 1 }]) */
	inputs?: { name: string; size: number; axes?: string[]; adapter?: string; kind?: string }[];
	/** axes of the last layer's output, default ['f'] (e.g. ['class']) */
	outputAxes?: string[];
}

const AXES: readonly string[] = ['c', 'y', 'x', 't', 'd', 'h', 'tq', 'tk', 'n', 'f', 'class', 'vocab', 'p', 'xy', 'k'];

function axes(list: string[] | undefined, fallback: Axis[]): Axis[] {
	if (!list) return fallback;
	for (const a of list) if (!AXES.includes(a)) throw new Error(`toSpec: unknown axis '${a}' (allowed: ${AXES.join(', ')})`);
	return list as Axis[];
}

const ACT_LABEL: Record<Exclude<SpecAct, 'identity'>, I18n> = {
	tanh: { en: 'tanh', zh: 'tanh' },
	relu: { en: 'ReLU', zh: 'ReLU' },
	sigmoid: { en: 'Sigmoid', zh: 'Sigmoid' },
	silu: { en: 'SiLU', zh: 'SiLU' },
	gelu: { en: 'GELU', zh: 'GELU' },
	gelu_tanh: { en: 'GELU (tanh)', zh: 'GELU（tanh 近似）' },
	softmax: { en: 'Softmax', zh: 'Softmax' }
};

/** Build the spec and the f32 weight blob for a stack of linear layers and activations. */
export function stackToSpec(items: SpecStackItem[], o: ToSpecOpts = {}): { spec: AtlasSpec; weights: Float32Array } {
	const first = items.find((it) => it.kind === 'linear');
	if (!first || first.kind !== 'linear') throw new Error('toSpec: the stack has no linear layer');
	const inF = first.weight.shape[1];
	const slug = o.slug ?? 'live-model';
	const inputs: InputDecl[] = o.inputs
		? o.inputs.map((d) => ({ name: d.name, adapter: d.adapter ?? o.adapter ?? 'vector', kind: d.kind ?? o.inputKind ?? 'vector', shape: [d.size], dtype: 'f32', axes: axes(d.axes, ['f']) }))
		: [{ name: 'input', adapter: o.adapter ?? 'vector', kind: o.inputKind ?? 'vector', shape: [inF], dtype: 'f32', axes: axes(o.inputAxes, ['f']) }];
	const totalIn = inputs.reduce((p, d) => p + (d.shape[0] as number), 0);
	if (totalIn !== inF) throw new Error(`toSpec: the inputs add up to ${totalIn} features, the first layer takes ${inF}`);

	const layers: AtlasLayer[] = [];
	let prev: string;
	if (inputs.length > 1) {
		layers.push({
			id: 'concat',
			op: 'concat',
			label: { en: 'Join the inputs', zh: '拼接输入' },
			in: inputs.map((d) => d.name),
			outShape: [inF],
			params: { axis: 0 },
			axes: ['f']
		});
		prev = 'concat';
	} else prev = inputs[0].name;

	const tensors: TensorEntry[] = [];
	const chunks: Float64Array[] = [];
	let offset = 0;
	let params = 0;
	const pushTensor = (v: Var) => {
		if (!v.name) throw new Error('toSpec: every parameter needs a name');
		tensors.push({ name: v.name, shape: v.shape.slice(), offset, length: v.size });
		chunks.push(v.data);
		offset += v.size;
		params += v.size;
	};

	const lastLinear = items.map((it) => it.kind).lastIndexOf('linear');
	items.forEach((it, idx) => {
		if (it.id.includes('#')) throw new Error(`toSpec: layer id '${it.id}' contains '#'`);
		const isLast = idx === items.length - 1 || (idx === lastLinear && items.slice(idx + 1).every((x) => x.kind === 'act' && x.act === 'identity'));
		const outAxes = isLast ? axes(o.outputAxes, ['f']) : (['f'] as Axis[]);
		if (it.kind === 'linear') {
			const [outF, inner] = it.weight.shape;
			const expectedW = `${it.id}.weight`;
			const w: Record<string, string> = {};
			if (it.weight.name !== expectedW) w.weight = it.weight.name!;
			if (it.bias && it.bias.name !== `${it.id}.bias`) w.bias = it.bias.name!;
			pushTensor(it.weight);
			if (it.bias) pushTensor(it.bias);
			const layer: AtlasLayer = {
				id: it.id,
				op: 'linear',
				label: { en: `Linear ${inner} → ${outF}`, zh: `全连接 ${inner} → ${outF}` },
				in: [prev],
				outShape: [outF],
				axes: outAxes
			};
			if (Object.keys(w).length) layer.w = w;
			layers.push(layer);
			prev = it.id;
		} else {
			if (it.act === 'identity') return;
			const prevShape = layers.length ? layers[layers.length - 1].outShape : [inF];
			const layer: AtlasLayer = { id: it.id, op: it.act, label: ACT_LABEL[it.act], in: [prev], outShape: prevShape.slice(), axes: outAxes };
			if (it.act === 'softmax') layer.params = { axis: -1 };
			layers.push(layer);
			prev = it.id;
		}
	});

	const weights = new Float32Array(offset);
	let at = 0;
	for (const c of chunks) {
		weights.set(c, at);
		at += c.length;
	}
	const name: I18n = o.name ?? { en: `${slug} (live weights)`, zh: `${slug}（实时权重）` };
	const spec: AtlasSpec = {
		slug,
		name,
		version: o.version ?? 'live',
		params,
		precision: 'fp32',
		input: inputs.length === 1 ? { kind: inputs[0].kind, shape: inputs[0].shape.slice() } : undefined,
		tensors,
		layers,
		parity: { against: 'atlas-autograd-f64', max_abs_diff: null, checked: null, scope: 'forward' },
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'atlas/autograd/tospec.ts' },
		inputs,
		outputs: [prev]
	};
	if (spec.input === undefined) delete spec.input;
	return { spec, weights };
}
