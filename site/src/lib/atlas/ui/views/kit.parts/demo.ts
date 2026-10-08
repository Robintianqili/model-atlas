// Demo data for ViewMeta.demo() (W1-D2): three tiny models and one tiny process, run by the REAL kit code
// (CORE_OPS through the executor, explain(), the recorder) on made-up, seeded weights. So every number a demo shows
// was computed, but the weights mean nothing: the provenance is always 'illustrative'. Pure TS, Node-safe.
//
//   demoCnn()      input [1,8,8] (a drawn corner) → conv 1→2 (3×3) → ReLU → conv 2→3 → ReLU → max pool → flatten
//                  → dense 12→4 → softmax over 4 made-up classes
//   demoGpt()      6 tokens → token + position embedding → add → LayerNorm → q, k, v → scores → causal mask →
//                  softmax → mix → residual add → MLP 8→16 (GELU) → logits over a 10-token vocabulary → softmax
//   demoLstm()     a 6-step series of 2 features → LSTM with 3 units (gate sub-steps i f g o c h)
//   demoProcess()  a 3-arm bandit with running averages (choose → pull → update), PhaseDef formulas on every phase
//   demoStage()    StageProps for one of them: fakeRuntime + the At a lesson step would get
import type { AtlasLayer, AtlasSpec, GraphModel, I18n, IntTensor, ProcessDef, Tape, Tensor, Trace, UnitExplanation } from '../../../types/index.ts';
import type { At } from '../../../types/lesson.ts';
import type { ExplainerRuntime, SelectionState } from '../../../types/runtime.ts';
import type { Control, Focus, Frame } from '../../../types/process.ts';
import { createRng } from '../../../core/rng.ts';
import { sym } from '../../../core/tex.ts';
import { CORE_OPS } from '../../../ops/index.ts';
import { fromParts } from '../../../graph/executor.ts';
import { fakeRuntime } from '../../../runtime/fake.ts';
import { record } from '../../../process/record.ts';
import { atFor } from '../../../runtime/sources.ts';
import type { Props } from './stage.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const NOTE = L('Demo data: a tiny model with made-up weights, computed by the real kit ops.', '演示数据：一个权重随意设定的小模型，由真实的工具包算子计算。');

/** The note every demo carries (the gallery shows it next to the Illustrative badge). */
export const DEMO_NOTE = NOTE;

function normal(seed: number, n: number, scale: number): Float32Array {
	const r = createRng(seed);
	const a = new Float32Array(n);
	for (let i = 0; i < n; i++) a[i] = r.normal() * scale;
	return a;
}
const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float32Array.from(data) });

function specOf(slug: string, name: I18n, layers: AtlasLayer[], weights: Record<string, Tensor>, extra: Partial<AtlasSpec>): AtlasSpec {
	let offset = 0;
	const tensors = Object.entries(weights).map(([n, t]) => {
		const e = { name: n, shape: t.shape.slice(), offset, length: t.data.length };
		offset += t.data.length;
		return e;
	});
	return {
		slug,
		name,
		version: 'demo',
		params: offset,
		precision: 'fp32',
		tensors,
		layers,
		parity: { against: 'none (demo)', max_abs_diff: null, checked: null },
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'demo' },
		...extra
	};
}

export interface DemoModel {
	model: GraphModel;
	trace: Trace;
	spec: AtlasSpec;
	tokens?: string[];
	vocab?: string[];
}

// ------------------------------------------------------------------ CNN
let cnnMemo: DemoModel | null = null;
let cnnMemoB: DemoModel | null = null;

/** The drawn corner: a vertical stroke and a horizontal stroke, softened at the edges. */
export function demoDrawing(): Float32Array {
	const x = new Float32Array(64);
	const put = (r: number, c: number, v: number) => (x[r * 8 + c] = Math.max(x[r * 8 + c], v));
	for (let r = 1; r <= 6; r++) {
		put(r, 2, 1);
		put(r, 1, 0.35);
		put(r, 3, 0.25);
	}
	for (let c = 2; c <= 6; c++) {
		put(6, c, 1);
		put(5, c, 0.3);
		put(7, c, 0.4);
	}
	return x;
}

export const DEMO_CLASSES = [L('line', '直线'), L('corner', '拐角'), L('circle', '圆'), L('dot', '点')];

function buildCnn(seed: number): DemoModel {
	const w: Record<string, Tensor> = {
		// two hand-made 3×3 kernels: a vertical edge and a diagonal
		'conv1.weight': T([2, 1, 3, 3], [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6, 0.8, -0.2, -0.6, -0.2, 0.8, -0.2, -0.6, -0.2, 0.8]),
		'conv1.bias': T([2], [-0.1, -0.2]),
		'conv2.weight': T([3, 2, 3, 3], normal(seed + 2, 54, 0.35)),
		'conv2.bias': T([3], normal(seed + 3, 3, 0.1)),
		'dense.weight': T([4, 12], normal(seed + 4, 48, 0.6)),
		'dense.bias': T([4], [0, 0.9, -0.1, -0.4])
	};
	const layers: AtlasLayer[] = [
		{ id: 'input', op: 'input', label: L('Input', '输入'), in: ['x'], outShape: [1, 8, 8], axes: ['c', 'y', 'x'] },
		{ id: 'conv1', op: 'conv2d', label: L('Conv 1', '卷积 1'), in: ['input'], outShape: [2, 6, 6], params: { kernel: 3, stride: 1, padding: 0, inChannels: 1, outChannels: 2 }, axes: ['c', 'y', 'x'] },
		{ id: 'relu1', op: 'relu', label: L('ReLU 1', 'ReLU 1'), in: ['conv1'], outShape: [2, 6, 6], axes: ['c', 'y', 'x'] },
		{ id: 'conv2', op: 'conv2d', label: L('Conv 2', '卷积 2'), in: ['relu1'], outShape: [3, 4, 4], params: { kernel: 3, stride: 1, padding: 0, inChannels: 2, outChannels: 3 }, axes: ['c', 'y', 'x'] },
		{ id: 'relu2', op: 'relu', label: L('ReLU 2', 'ReLU 2'), in: ['conv2'], outShape: [3, 4, 4], axes: ['c', 'y', 'x'] },
		{ id: 'pool', op: 'maxpool2d', label: L('Max pool', '最大池化'), in: ['relu2'], outShape: [3, 2, 2], params: { kernel: 2, stride: 2 }, axes: ['c', 'y', 'x'] },
		{ id: 'flatten', op: 'flatten', label: L('Flatten', '展平'), in: ['pool'], outShape: [12], axes: ['f'] },
		{ id: 'dense', op: 'linear', label: L('Dense', '全连接'), in: ['flatten'], outShape: [4], axes: ['class'] },
		{ id: 'probs', op: 'softmax', label: L('Softmax', 'Softmax'), in: ['dense'], outShape: [4], axes: ['class'] }
	];
	const spec = specOf('demo-cnn', L('Demo CNN', '演示 CNN'), layers, w, {
		inputs: [{ name: 'x', adapter: 'demo', kind: 'drawing', shape: [1, 8, 8], dtype: 'f32', axes: ['c', 'y', 'x'] }],
		outputs: ['probs'],
		classes: DEMO_CLASSES
	});
	const model = fromParts(spec, w, CORE_OPS);
	const trace = model.forward({ shape: [1, 8, 8], data: demoDrawing() });
	return { model, trace, spec: model.spec };
}

export function demoCnn(): DemoModel {
	return (cnnMemo ??= buildCnn(11));
}
/** The same CNN with other seeded weights in conv2 and dense (the compare demo). */
export function demoCnnB(): DemoModel {
	return (cnnMemoB ??= buildCnn(23));
}

// ------------------------------------------------------------------ GPT-like
let gptMemo: DemoModel | null = null;
export const DEMO_VOCAB = ['The', ' cat', ' sat', ' on', ' the', ' mat', '.', ' dog', ' ran', ' a'];

export function demoGpt(): DemoModel {
	if (gptMemo) return gptMemo;
	const D = 8;
	const V = DEMO_VOCAB.length;
	const w: Record<string, Tensor> = {
		'tok_emb.weight': T([V, D], normal(31, V * D, 0.8)),
		'pos_emb.weight': T([8, D], normal(32, 8 * D, 0.4)),
		'ln.weight': T([D], Array.from(normal(33, D, 0.15), (v) => 1 + v)),
		'ln.bias': T([D], normal(34, D, 0.1)),
		'q.weight': T([D, D], normal(35, D * D, 0.5)),
		'q.bias': T([D], normal(36, D, 0.05)),
		'k.weight': T([D, D], normal(37, D * D, 0.5)),
		'k.bias': T([D], normal(38, D, 0.05)),
		'v.weight': T([D, D], normal(39, D * D, 0.5)),
		'v.bias': T([D], normal(40, D, 0.05)),
		'fc.weight': T([16, D], normal(41, 16 * D, 0.45)),
		'fc.bias': T([16], normal(42, 16, 0.1)),
		'logits.weight': T([V, 16], normal(43, V * 16, 0.4)),
		'logits.bias': T([V], normal(44, V, 0.05))
	};
	const td = ['t', 'd'] as const;
	const lin = (id: string, input: string, out: number, label: I18n): AtlasLayer => ({ id, op: 'linear', label, in: [input], outShape: ['T', out], axes: [...td] });
	const layers: AtlasLayer[] = [
		{ id: 'tok_emb', op: 'embedding', label: L('Token embedding', '词元嵌入'), in: ['ids'], outShape: ['T', D], axes: [...td] },
		{ id: 'pos_emb', op: 'position_embedding', label: L('Position embedding', '位置嵌入'), in: ['ids'], outShape: ['T', D], axes: [...td] },
		{ id: 'embed', op: 'add', label: L('Token + position', '词元 + 位置'), in: ['tok_emb', 'pos_emb'], outShape: ['T', D], axes: [...td] },
		{ id: 'ln', op: 'layernorm', label: L('LayerNorm', '层归一化'), in: ['embed'], outShape: ['T', D], params: { eps: 1e-5 }, axes: [...td] },
		lin('q', 'ln', D, L('Query', '查询')),
		lin('k', 'ln', D, L('Key', '键')),
		lin('v', 'ln', D, L('Value', '值')),
		{ id: 'scores', op: 'attention_scores', label: L('Attention scores', '注意力分数'), in: ['q', 'k'], outShape: [2, 'T', 'T'], params: { nHead: 2, dHead: 4, scale: 0.5 }, axes: ['h', 'tq', 'tk'] },
		{ id: 'masked', op: 'causal_mask', label: L('Causal mask', '因果掩码'), in: ['scores'], outShape: [2, 'T', 'T'], axes: ['h', 'tq', 'tk'] },
		{ id: 'weights', op: 'softmax', label: L('Attention weights', '注意力权重'), in: ['masked'], outShape: [2, 'T', 'T'], params: { axis: -1 }, axes: ['h', 'tq', 'tk'] },
		{ id: 'heads', op: 'attention_mix', label: L('Mix of values', '值的加权和'), in: ['weights', 'v'], outShape: ['T', D], params: { nHead: 2, dHead: 4 }, axes: [...td] },
		{ id: 'resid', op: 'add', label: L('Residual add', '残差相加'), in: ['embed', 'heads'], outShape: ['T', D], axes: [...td] },
		lin('fc', 'resid', 16, L('MLP up', 'MLP 升维')),
		{ id: 'gelu', op: 'gelu', label: L('GELU', 'GELU'), in: ['fc'], outShape: ['T', 16], params: { approx: 'tanh' }, axes: [...td] },
		{ id: 'logits', op: 'linear', label: L('Logits', 'Logits'), in: ['gelu'], outShape: ['T', V], axes: ['t', 'vocab'] },
		{ id: 'probs', op: 'softmax', label: L('Next-token probabilities', '下一个词元的概率'), in: ['logits'], outShape: ['T', V], params: { axis: -1 }, axes: ['t', 'vocab'] }
	];
	const spec = specOf('demo-gpt', L('Demo transformer', '演示 Transformer'), layers, w, {
		inputs: [{ name: 'ids', adapter: 'demo', kind: 'text', shape: ['T'], dtype: 'i32', axes: ['t'] }],
		dims: { T: { min: 1, max: 8, label: L('tokens', '词元数') } },
		outputs: ['probs'],
		vocabSize: V
	});
	const model = fromParts(spec, w, CORE_OPS);
	const ids: IntTensor = { shape: [6], data: Int32Array.from([0, 1, 2, 3, 4, 5]), dtype: 'i32' };
	const trace = model.forward({ ids });
	gptMemo = { model, trace, spec: model.spec, tokens: Array.from(ids.data, (i) => DEMO_VOCAB[i]), vocab: DEMO_VOCAB };
	return gptMemo;
}

// ------------------------------------------------------------------ LSTM
let lstmMemo: DemoModel | null = null;
export function demoLstm(): DemoModel {
	if (lstmMemo) return lstmMemo;
	const H = 3;
	const In = 2;
	const w: Record<string, Tensor> = {
		'lstm.weight_ih': T([4 * H, In], normal(51, 4 * H * In, 0.8)),
		'lstm.weight_hh': T([4 * H, H], normal(52, 4 * H * H, 0.5)),
		'lstm.bias_ih': T([4 * H], normal(53, 4 * H, 0.2)),
		'lstm.bias_hh': T([4 * H], normal(54, 4 * H, 0.2))
	};
	const layers: AtlasLayer[] = [{ id: 'lstm', op: 'lstm', label: L('LSTM', 'LSTM'), in: ['x'], outShape: ['T', H], params: { hidden: H, inputSize: In }, axes: ['t', 'd'] }];
	const spec = specOf('demo-lstm', L('Demo LSTM', '演示 LSTM'), layers, w, {
		inputs: [{ name: 'x', adapter: 'demo', kind: 'series', shape: ['T', In], dtype: 'f32', axes: ['t', 'f'] }],
		dims: { T: { min: 1, max: 16, label: L('steps', '步数') } },
		outputs: ['lstm']
	});
	const model = fromParts(spec, w, CORE_OPS);
	const x = new Float32Array(6 * In);
	for (let t = 0; t < 6; t++) {
		x[t * In] = Math.sin(t * 0.9);
		x[t * In + 1] = t < 3 ? 1 : -0.5;
	}
	const trace = model.forward({ x: { shape: [6, In], data: x } });
	lstmMemo = { model, trace, spec: model.spec };
	return lstmMemo;
}

// ------------------------------------------------------------------ a tiny process (kind C)
type BanditOpt = { steps: number; eps: number };
type BanditState = { Q: Tensor; n: Tensor; arm: number; r: number };
const TRUE_MEANS = [0.2, 0.5, 0.8];

export const banditProcess: ProcessDef<null, BanditOpt, BanditState> = {
	id: 'demo-bandit',
	kind: 'algorithm',
	title: L('A 3-arm bandit with running averages', '用滑动平均估计的三臂老虎机'),
	phases: {
		choose: {
			label: L('Choose an arm', '选一个臂'),
			kind: 'act',
			formula: { tex: `a = \\arg\\max_j ${sym('act', 'Q_j')} = {{arm}}`, values: (f) => ({ arm: (f.metrics?.arm ?? 0) + 1 }) },
			narration: { what: L('Pick the arm with the best estimate, or a random one with probability ε.', '选估计值最高的臂；以 ε 的概率随机选一个。') }
		},
		pull: {
			label: L('Pull it', '拉动它'),
			kind: 'reward',
			formula: { tex: `${sym('out', 'r')} = {{r}}`, values: (f) => ({ r: f.metrics?.r ?? 0 }) },
			narration: { what: L('The arm pays a reward: 1 or 0.', '这个臂给出奖励：1 或 0。') }
		},
		update: {
			label: L('Update the estimate', '更新估计值'),
			kind: 'update',
			formula: {
				tex: `${sym('act', "Q'")} = ${sym('act', 'Q')} + \\frac{1}{n}(${sym('out', 'r')} - ${sym('act', 'Q')}) = {{q}} + \\frac{1}{ {{n}} }({{r}} - {{q}}) = {{qNew}}`,
				values: (f) => ({ q: f.metrics?.q ?? 0, n: f.metrics?.n ?? 1, r: f.metrics?.r ?? 0, qNew: f.metrics?.qNew ?? 0 })
			},
			narration: { what: L('Move the estimate a step 1/n toward the reward.', '把估计值朝奖励方向移动 1/n。') }
		}
	},
	schema: {
		Q: { kind: 'tensor', label: L('estimates Q', '估计值 Q'), shape: [3], scale: 'sequential', role: 'optimizer' },
		n: { kind: 'tensor', label: L('pulls n', '拉动次数 n'), shape: [3], role: 'data' },
		arm: { kind: 'int', label: L('arm', '臂'), tol: 'exact' },
		r: { kind: 'scalar', label: L('reward', '奖励'), tol: 'exact' }
	},
	defaults: { steps: 12, eps: 0.2 },
	*run(_input, ctx): Generator<Frame<BanditState>, void, Control<BanditOpt> | undefined> {
		const Q = new Float64Array(3);
		const n = new Float64Array(3);
		for (let s = 0; s < ctx.opt.steps && !ctx.signal.cancelled; s++) {
			let arm = 0;
			for (let j = 1; j < 3; j++) if (Q[j] > Q[arm]) arm = j;
			const explore = ctx.rng.next() < ctx.opt.eps;
			if (explore) arm = ctx.rng.int(3);
			yield { phase: 'choose', step: s, state: { arm }, metrics: { arm, explore: explore ? 1 : 0 } };
			const r = ctx.rng.next() < TRUE_MEANS[arm] ? 1 : 0;
			yield { phase: 'pull', step: s, state: { r }, metrics: { arm, r } };
			const q = Q[arm];
			n[arm] += 1;
			Q[arm] = q + (r - q) / n[arm];
			yield {
				phase: 'update',
				step: s,
				state: { Q: { shape: [3], data: Float64Array.from(Q) }, n: { shape: [3], data: Float64Array.from(n) } },
				metrics: { arm, r, q, n: n[arm], qNew: Q[arm] }
			};
		}
	}
};

const banditMemo = new Map<string, Tape>();
/** The demo bandit tape (seed 3, ε 0.2, 12 pulls); other seeds and ε for the compare demo. */
export function demoProcess(seed = 3, eps = 0.2, steps = 12): Tape {
	const key = `${seed}:${eps}:${steps}`;
	let tp = banditMemo.get(key);
	if (!tp) banditMemo.set(key, (tp = record(banditProcess, null, { eps, steps }, { seed })));
	return tp;
}

// ------------------------------------------------------------------ StageProps
/** An extra demo for the gallery (meta.ts files may export `variants(): Promise<DemoVariant[]>`). */
export interface DemoVariant {
	label: I18n;
	stage: Omit<Props, 'ctx'>;
}

export type DemoName = 'cnn' | 'cnnB' | 'gpt' | 'lstm';

export function demoModel(name: DemoName): DemoModel {
	return name === 'cnn' ? demoCnn() : name === 'cnnB' ? demoCnnB() : name === 'gpt' ? demoGpt() : demoLstm();
}

/** A fakeRuntime around a demo model (trace source 'fwd'; provenance 'illustrative'). */
export function demoRuntime(name: DemoName, extra: { tapes?: Record<string, Tape>; traces?: Record<string, DemoModel> } = {}): ExplainerRuntime {
	const d = demoModel(name);
	const rt = fakeRuntime({ slug: d.spec.slug, spec: d.spec, model: d.model, trace: d.trace, tapes: extra.tapes });
	if (d.tokens) rt.pre = { tensors: {}, steps: [], empty: false, tokens: d.tokens } as ExplainerRuntime['pre'];
	for (const [src, m] of Object.entries(extra.traces ?? {})) {
		const other = fakeRuntime({ spec: m.spec, model: m.model, trace: m.trace, traceSource: src });
		rt.sources = { ...rt.sources, [src]: other.sources[src] };
	}
	return rt;
}

/** StageProps for a zoom/layer view on a demo model: the At a lesson step on `layer` (and `unit`) would get. */
export function demoStage(name: DemoName, layer: string, unit?: number[], props: Record<string, unknown> = {}): Omit<Props, 'ctx'> {
	const d = demoModel(name);
	const rt = demoRuntime(name);
	const sel: SelectionState = { layer };
	if (unit) sel.unit = unit;
	rt.sel = sel;
	const tape = rt.sources.fwd.tape!;
	const index = tape.find({ layer: layer.split('#')[0] });
	const at: At = index >= 0 ? atFor(rt, 'fwd', index) : { index: -1 };
	if (unit && !at.unit) {
		try {
			at.unit = d.model.explain(d.trace, layer, unit);
		} catch {
			/* the view explains it itself */
		}
	}
	at.layer = layer;
	const focus: Focus[] = unit ? [{ field: layer, index: unit, role: 'out' }] : [];
	const p: Record<string, unknown> = { ...props };
	if (d.tokens && !p.tokens) p.tokens = d.tokens;
	if (d.vocab && !p.vocab) p.vocab = d.vocab;
	return { rt, source: 'fwd', at, focus, sel, provenance: 'illustrative', autoplay: false, onend: () => {}, props: p };
}

/** StageProps on the demo process tape at frame `index` (source 'bandit'). */
export function processStage(index: number, props: Record<string, unknown> = {}): Omit<Props, 'ctx'> {
	const tape = demoProcess();
	const rt = fakeRuntime({ slug: 'demo-bandit', tapes: { bandit: tape } });
	const i = Math.max(0, Math.min(tape.length - 1, index));
	return { rt, source: 'bandit', at: atFor(rt, 'bandit', i), focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}

/** The unit explanation a demo stage resolves to (for describe() tests). */
export function demoUnit(name: DemoName, layer: string, unit: number[]): UnitExplanation {
	const d = demoModel(name);
	return d.model.explain(d.trace, layer, unit);
}

/** Two sources side by side for the compare view: the demo CNN with two sets of weights on the same drawing
 *  (traces), or two bandit runs with a different ε (tapes, metric 'qNew'). */
export function compareStage(kind: 'traces' | 'tapes'): Omit<Props, 'ctx'> {
	if (kind === 'traces') {
		const rt = demoRuntime('cnn', { traces: { other: demoCnnB() } });
		const compare = { a: 'fwd', b: 'other', label: [L('weights A', '权重 A'), L('weights B', '权重 B')] };
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', rt.sources.fwd.tape!.length - 1), focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { compare } };
	}
	const a = demoProcess(7, 0, 40);
	const b = demoProcess(7, 0.3, 40);
	const rt = fakeRuntime({ slug: 'demo-bandit', tapes: { greedy: a, explore: b } });
	const compare = { a: 'greedy', b: 'explore', label: [L('ε = 0 (always greedy)', 'ε = 0（总是贪心）'), L('ε = 0.3', 'ε = 0.3')] };
	return { rt, source: 'greedy', at: atFor(rt, 'greedy', a.length - 1), focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { compare, metric: 'qNew' } };
}
