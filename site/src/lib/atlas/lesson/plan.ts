// The auto-lesson planner (design §5.2). It walks spec.layers in order:
//  1. input step ('input'): the adapter's layer0 narration;
//  2. the first occurrence of an op type: a full step ('op:<id>': what + how, look). If op.unit.scan is set, the step plays
//     the implicit scan source 'scan:<id>' and waits for done;
//  3. a distinct repeat (op.distinct(p, prev)): a short step with the op's `repeat` beat;
//  4. repeated groups (inferred: a maximal run of layers whose ids share a prefix /^[A-Za-z]+\d+$/ before the first '.';
//     groups with the same op sequence are repeats): the first group's layers follow rules 2-3; every later run of
//     repeats gives one summary step ('group:<id>') and its layers get no steps;
//  5. shape-only ops (category 'shape') get a step only when the axes kind changes (grid -> vector);
//  6. output step ('output'): the answer plus its limits (a language model's vocab output speaks of the next token and
//     its vocabulary, not of classes; {top.label} is the token's text);
//  7. spec hints win: teach.lesson ('step' forces a step, 'fold' and 'skip' give none), teach.title, teach.narration.
// Then kind-B motifs per train source ('train:<motif>@<s>') and kind-C motifs per process source
// ('proc:event.<phase>@<s>', 'proc:run@<s>'). No pattern collapsing, no budget folding: lint warns outside 5-12 steps.
import type { I18n, Tensor } from '../types/core.ts';
import type { AtlasLayer, AtlasSpec } from '../types/spec.ts';
import type { NarrationSet, OpDef, OpRegistry } from '../types/op.ts';
import type { ProcessDef } from '../types/process.ts';
import type { At, StepDef, Tmpl, TmplVal } from '../types/lesson.ts';
import type { ExplainerManifest, InputAdapter } from '../types/explainer.ts';
import type { ExplainerRuntime, SourceDef } from '../types/runtime.ts';
import { SCAN_PREFIX, sourcesOf } from './define.ts';
import { outputId, outputRow, primaryTrace, shapeText, unitValue } from './facts.ts';
import { pickUnit } from '../runtime/selection.ts';
import { termsOf } from '../ops/common.ts';

export interface PlanOpts {
	adapters?: Record<string, InputAdapter>;
	processes?: Record<string, ProcessDef<any, any, any>>;
}

export interface ResolvedLayerOp {
	def: OpDef;
	params: Record<string, unknown>;
}

/** Op lookup with aliases, defaults from attrs, alias params, then layer params. */
export function opResolver(ops: OpDef[] | OpRegistry): (layer: AtlasLayer) => ResolvedLayerOp | null {
	const map = new Map<string, { def: OpDef; alias: Record<string, unknown> }>();
	if (Array.isArray(ops)) {
		for (const def of ops) {
			map.set(def.op, { def, alias: {} });
			for (const [a, p] of Object.entries(def.aliases ?? {})) map.set(a, { def, alias: { ...(p as object) } });
		}
	}
	return (layer) => {
		let hit = map.get(layer.op);
		if (!hit && !Array.isArray(ops) && ops.has(layer.op)) {
			const r = ops.get(layer.op);
			hit = { def: r.def, alias: r.params };
		}
		if (!hit) return null;
		const params: Record<string, unknown> = {};
		for (const [k, d] of Object.entries(hit.def.attrs ?? {})) if (d.default !== undefined) params[k] = d.default;
		return { def: hit.def, params: { ...params, ...hit.alias, ...(layer.params ?? {}) } };
	};
}

const GROUP_RE = /^[A-Za-z]+\d+$/;

export interface InferredGroup {
	id: string;
	from: number;
	to: number;
	sig: string;
}

/** Groups per §4.2: maximal runs of layers sharing an id prefix like 'L0' or 'B3' before the first '.'. */
export function inferGroups(spec: AtlasSpec): InferredGroup[] {
	const out: InferredGroup[] = [];
	spec.layers.forEach((l, i) => {
		const dot = l.id.indexOf('.');
		const prefix = dot > 0 ? l.id.slice(0, dot) : '';
		if (!prefix || !GROUP_RE.test(prefix)) return;
		const last = out[out.length - 1];
		if (last && last.id === prefix && last.to === i - 1) {
			last.to = i;
			last.sig += `|${l.op}`;
		} else out.push({ id: prefix, from: i, to: i, sig: l.op });
	});
	return out;
}

/** A group's name as the page labels its layers: the part of every layer label before ' · ' (TinyGPT: "Block 2 ·
 *  LayerNorm 1" -> "Block 2", "第 2 块 · 层归一化 1" -> "第 2 块"), when all layers of the group share it; else the id
 *  ("Block L1"). R0: the group step said "Blocks L1 to L3" while the page said Block 2-4. */
export function blockName(spec: AtlasSpec, g: { id: string; from: number; to: number }): I18n {
	const layers = spec.layers.slice(g.from, g.to + 1);
	const head = (s: string | undefined) => {
		if (!s) return null;
		const m = /^(.+?)\s*[·:—–]\s+/.exec(s);
		return m ? m[1].trim() : null;
	};
	const shared = (pick: (l: AtlasLayer) => string | undefined) => {
		const hs = layers.map((l) => head(pick(l)));
		return hs.length && hs[0] && hs.every((h) => h === hs[0]) ? hs[0] : null;
	};
	const en = shared((l) => l.label?.en);
	if (!en) return { en: `Block ${g.id}`, zh: `${g.id} 块` };
	return { en, zh: shared((l) => l.label?.zh) ?? en };
}

/** The last axis of a layer is the vocabulary: a language model's next-token output. */
export function isVocabOutput(layer: AtlasLayer | undefined): boolean {
	return !!layer?.axes?.length && layer.axes[layer.axes.length - 1] === 'vocab';
}

function joinI18n(parts: (I18n | undefined)[]): I18n {
	const ps = parts.filter((p): p is I18n => !!p);
	const en = ps.map((p) => p.en).join(' ');
	const zhParts = ps.map((p) => p.zh);
	return zhParts.every((z) => z !== undefined) && ps.length ? { en, zh: zhParts.join('') } : { en };
}

function axesKind(layer: AtlasLayer | undefined): string {
	if (!layer) return '';
	if (layer.axes?.length) {
		const a = layer.axes;
		if (a.includes('y') && a.includes('x')) return 'grid';
		if (a.includes('t') || a.includes('tq')) return 'sequence';
		return a.length === 1 ? 'vector' : a.join(',');
	}
	const r = layer.outShape.length;
	return r >= 3 ? 'grid' : r === 2 ? 'matrix' : 'vector';
}

const asVal = (v: unknown) => v as TmplVal;

/** Template variables every auto op step offers: the op's structural vars, the layer's shapes and label, and the
 *  explain() values of the step's unit. */
function opVars(def: OpDef, layer: AtlasLayer, spec: AtlasSpec) {
	return (_rt: ExplainerRuntime, at: At): Record<string, TmplVal> => {
		const out: Record<string, TmplVal> = {};
		for (const [k, v] of Object.entries(def.vars?.(layer, spec) ?? {})) out[k] = v;
		const t = (at.state as { out?: Tensor } | undefined)?.out;
		out.outShape = shapeText(t && (at.state as { layer?: string }).layer === layer.id ? t.shape : layer.outShape);
		out.id = layer.id;
		out.label = asVal(layer.label);
		if (at.unit) {
			for (const [k, v] of Object.entries(at.unit.values)) out[k] = v;
			out.result = at.unit.result;
			if (at.unit.bias !== undefined) out.bias = at.unit.bias;
			const n = unitValue(at.unit, 'terms');
			if (n !== undefined) out.terms = n;
		}
		return out;
	};
}

function tmpl(text: I18n, vars?: (rt: ExplainerRuntime, at: At) => Record<string, TmplVal>): Tmpl<ExplainerRuntime> {
	return vars ? { ...text, vars } : { ...text };
}

function opStep(layer: AtlasLayer, r: ResolvedLayerOp, spec: AtlasSpec, fwd: string, kind: 'first' | 'repeat'): StepDef {
	const { def } = r;
	const n: NarrationSet = def.narration;
	const what = layer.teach?.narration?.what ?? (kind === 'repeat' ? (n.repeat ?? n.what) : n.what);
	const how = layer.teach?.narration?.how ?? (kind === 'first' ? n.how : undefined);
	const vars = opVars(def, layer, spec);
	const scan = kind === 'first' && def.unit?.scan && def.unit.scan !== 'none';
	const step: StepDef = {
		id: `op:${layer.id}`,
		title: layer.teach?.title ?? (kind === 'first' ? def.label : layer.label),
		narration: tmpl(joinI18n([what, how]), vars),
		source: scan ? `${SCAN_PREFIX}${layer.id}` : fwd,
		at: scan ? 'first' : { layer: layer.id },
		select: (rt) => {
			const t = primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel })?.t[layer.id];
			return { layer: layer.id, unit: pickUnit(t, def.unit?.pick, { axes: layer.axes }) };
		},
		view: layer.ui?.zoom ?? (def.zoom !== 'none' ? `zoom.${def.zoom}` : 'layer.tensor'),
		provenance: 'live',
		terms: termsOf(def, layer, spec)
	};
	if (n.look) step.look = tmpl(n.look, vars);
	if (scan) {
		step.play = { from: 'first', to: 'last', interval: 14 };
		step.waitForDone = true;
	}
	if (!step.terms?.length) delete step.terms;
	return step;
}

const T = {
	inputTitle: { en: 'Your input becomes numbers', zh: '你的输入变成数字' },
	inputText: {
		en: 'The model never sees your input itself. It first turns it into {n|int} numbers, arranged as {shape}.',
		zh: '模型看不到你的输入本身。它先把输入变成 {n|int} 个数，排成 {shape} 的形状。'
	},
	outTitle: { en: 'The answer, and what it does not mean', zh: '答案，以及它不代表什么' },
	outClass: {
		en: 'The model answers {top.label} with {top.p|pct}; the runner-up is {runnerUp.label} with {runnerUp.p|pct}. These are shares among the {n|int} classes it knows, not certainty: it must give all its confidence to these classes, even for an input that is none of them.',
		zh: '模型的回答是 {top.label}，概率 {top.p|pct}；第二名是 {runnerUp.label}，{runnerUp.p|pct}。这只是在它认识的 {n|int} 个类别之间的分配，并不代表确定：即使输入不属于其中任何一类，它也必须把全部信心分给这些类别。'
	},
	outToken: {
		en: 'The model’s guess for the next token is {top.label}, with {top.p|pct}; the runner-up is {runnerUp.label} with {runnerUp.p|pct}. These are shares among all {n|int} tokens of its vocabulary, read at the last position of your text. They are not certainty: the model always spreads its belief over the tokens it knows, even when no continuation makes sense.',
		zh: '模型对下一个词元的猜测是 {top.label}，概率 {top.p|pct}；第二名是 {runnerUp.label}，{runnerUp.p|pct}。这些是它词表里全部 {n|int} 个词元之间的分配，读的是你文字最后一个位置的预测。它们并不代表确定：即使没有合理的接续，模型也总会把信心分给它认识的词元。'
	},
	// K2: a vocabulary output read at another row than the last (manifest output.row: a masked model's [MASK] row)
	outTokenAt: {
		en: 'The model’s guess for position {row} (counting from 0) is {top.label}, with {top.p|pct}; the runner-up is {runnerUp.label} with {runnerUp.p|pct}. These are shares among all {n|int} tokens of its vocabulary. They are not certainty: the model always spreads its belief over the tokens it knows, even when no word fits.',
		zh: '模型对第 {row} 个位置（从 0 开始数）的猜测是 {top.label}，概率 {top.p|pct}；第二名是 {runnerUp.label}，{runnerUp.p|pct}。这些是它词表里全部 {n|int} 个词元之间的分配。它们并不代表确定：即使没有合适的词，模型也总会把信心分给它认识的词元。'
	},
	outPlain: {
		en: 'The last layer gives {n|int} numbers, arranged as {shape}. They are only as good as the training data behind them.',
		zh: '最后一层给出 {n|int} 个数，排成 {shape} 的形状。它们的好坏取决于背后的训练数据。'
	},
	// {first}, {last}, {model}: the blocks' names as the page labels them ("Block 2", "第 2 块"; blockName below)
	groupText: {
		en: '{first} to {last} repeat the same steps as {model}, each with its own learned weights. Nothing new happens inside them, so the lesson does not walk through them one by one.',
		zh: '{first}到{last}重复{model}的同样步骤，只是各自有自己学到的权重。它们里面没有新的计算，所以课程不再逐个讲解。'
	},
	groupOne: {
		en: '{first} repeats the same steps as {model}, with its own learned weights. Nothing new happens inside it, so the lesson does not walk through it again.',
		zh: '{first}重复{model}的同样步骤，只是有自己学到的权重。它里面没有新的计算，所以课程不再重复讲解。'
	}
};

export function planLesson(spec: AtlasSpec | null, ops: OpDef[] | OpRegistry, manifest: ExplainerManifest, o: PlanOpts = {}): StepDef[] {
	const steps: StepDef[] = [];
	const sources = sourcesOf(manifest);
	const fwd = Object.entries(sources).find(([, d]) => d.kind === 'trace')?.[0];
	if (spec && fwd) planLayers(spec, ops, manifest, fwd, steps, o);
	for (const [name, def] of Object.entries(sources)) {
		if (def.kind === 'train') steps.push(...trainMotifs(name, def));
		if (def.kind === 'process') steps.push(...procMotifs(name, def, o.processes?.[def.process]));
	}
	return steps;
}

function planLayers(spec: AtlasSpec, ops: OpDef[] | OpRegistry, manifest: ExplainerManifest, fwd: string, steps: StepDef[], o: PlanOpts): void {
	const resolve = opResolver(ops);
	const groups = inferGroups(spec);
	const seenSig = new Map<string, string>();
	const skipLayer = new Set<number>();
	const summaryAt = new Map<number, { ids: string[]; model: string; lastLayer: string; names: I18n[]; modelName: I18n }>();
	const nameOf = new Map<string, I18n>();
	for (let g = 0; g < groups.length; g++) {
		const grp = groups[g];
		const model = seenSig.get(grp.sig);
		nameOf.set(grp.id, blockName(spec, grp));
		if (model === undefined) {
			seenSig.set(grp.sig, grp.id);
			continue;
		}
		for (let i = grp.from; i <= grp.to; i++) skipLayer.add(i);
		// extend the current summary when this repeat directly follows the previous one
		const prev = [...summaryAt.entries()].find(([, s]) => s.model === model && spec.layers.findIndex((l) => l.id === s.lastLayer) === grp.from - 1);
		if (prev) {
			prev[1].ids.push(grp.id);
			prev[1].names.push(nameOf.get(grp.id)!);
			prev[1].lastLayer = spec.layers[grp.to].id;
		} else summaryAt.set(grp.from, { ids: [grp.id], model, lastLayer: spec.layers[grp.to].id, names: [nameOf.get(grp.id)!], modelName: nameOf.get(model) ?? { en: `Block ${model}`, zh: `${model} 块` } });
	}
	const adapter = manifest.input?.adapter ? o.adapters?.[manifest.input.adapter] : undefined;
	const outId = spec.outputs?.[0] ?? spec.layers[spec.layers.length - 1]?.id;
	const seen = new Map<string, Record<string, unknown>>();
	let prevKind = '';
	let inputDone = false;
	spec.layers.forEach((layer, i) => {
		const summary = summaryAt.get(i);
		if (summary) {
			const ids = summary.ids;
			steps.push({
				id: `group:${ids[0]}`,
				title: { en: `${ids.length} more block${ids.length > 1 ? 's' : ''} like this one`, zh: `还有 ${ids.length} 个同样的块` },
				narration: tmpl(ids.length > 1 ? T.groupText : T.groupOne, () => ({ first: asVal(summary.names[0]), last: asVal(summary.names[summary.names.length - 1]), model: asVal(summary.modelName) })),
				source: fwd,
				at: { layer: summary.lastLayer },
				select: () => ({ layer: summary.lastLayer }),
				view: 'layer.tensor',
				provenance: 'live'
			});
		}
		if (skipLayer.has(i)) return;
		const r = resolve(layer);
		const hint = layer.teach?.lesson;
		const kind = axesKind(layer);
		if (!r) return;
		if (r.def.op === 'input' || r.def.category === 'io') {
			if (!inputDone && hint !== 'skip' && hint !== 'fold') {
				inputDone = true;
				steps.push(inputStep(layer, adapter, fwd));
			}
			prevKind = kind;
			return;
		}
		if (hint === 'skip' || hint === 'fold') {
			seen.set(r.def.op, r.params);
			prevKind = kind;
			return;
		}
		const prev = seen.get(r.def.op);
		const first = prev === undefined;
		let distinct = false;
		if (!first && r.def.distinct) {
			try {
				distinct = r.def.distinct(r.params, prev);
			} catch {
				distinct = false;
			}
		}
		const shapeOnly = r.def.category === 'shape';
		const axesChange = prevKind !== '' && kind !== prevKind;
		let want = first || distinct || hint === 'step';
		if (shapeOnly && hint !== 'step') want = axesChange;
		if (layer.id === outId && r.def.category === 'output' && hint !== 'step') want = false; // the output step covers it
		if (want) steps.push(opStep(layer, r, spec, fwd, first || hint === 'step' || shapeOnly ? 'first' : 'repeat'));
		seen.set(r.def.op, r.params);
		prevKind = kind;
	});
	if (!inputDone && spec.inputs?.length) steps.unshift(inputStep(null, adapter, fwd));
	steps.push(outputStep(spec, manifest, fwd, outId));
}

function inputStep(layer: AtlasLayer | null, adapter: InputAdapter | undefined, fwd: string): StepDef {
	const n = adapter?.layer0.narration;
	const text = n ? joinI18n([n.what, n.how]) : T.inputText;
	const vars = (rt: ExplainerRuntime): Record<string, TmplVal> => {
		const t = Object.values(rt.pre?.tensors ?? {})[0];
		return t ? { n: t.data.length, shape: shapeText(t.shape) } : {};
	};
	const step: StepDef = {
		id: 'input',
		title: T.inputTitle,
		narration: tmpl(text, vars),
		source: fwd,
		at: layer ? { layer: layer.id } : 'first',
		view: adapter?.layer0.view ?? 'layer.tensor',
		provenance: 'live'
	};
	if (n?.look) step.look = tmpl(n.look, vars);
	return step;
}

function outputStep(spec: AtlasSpec, manifest: ExplainerManifest, fwd: string, outLayer: string | undefined): StepDef {
	const layer = spec.layers.find((l) => l.id === outLayer);
	const classes = (spec.classes ?? spec.labels) as unknown[] | undefined;
	const vocab = isVocabOutput(layer) && !Array.isArray(classes);
	const classifier = Array.isArray(classes) || layer?.op === 'softmax' || vocab;
	const vars = (rt: ExplainerRuntime): Record<string, TmplVal> => {
		const tr = primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel });
		const id = outputId(rt.spec, tr);
		const t = id ? tr?.t[id] : undefined;
		// classes = the last axis (a [T, V] sequence output is read at its answer row, like top.*: K2, manifest output.row)
		const n = t ? (classifier ? t.shape[t.shape.length - 1] : t.data.length) : Array.isArray(classes) ? classes.length : NaN;
		return { n, shape: t ? shapeText(t.shape) : layer ? shapeText(layer.outShape) : '', row: t ? outputRow(rt, t) : NaN };
	};
	const atRow = manifest.output?.row !== undefined && manifest.output.row !== 'last';
	return {
		id: 'output',
		title: T.outTitle,
		narration: tmpl(vocab ? (atRow ? T.outTokenAt : T.outToken) : classifier ? T.outClass : T.outPlain, vars),
		source: fwd,
		at: outLayer ? { layer: outLayer } : 'last',
		select: () => (outLayer ? { layer: outLayer } : {}),
		view: manifest.output?.view ?? (classifier ? 'kit.probbars' : 'layer.tensor'),
		provenance: 'live'
	};
}

function trainMotifs(s: string, def: Extract<SourceDef, { kind: 'train' }>): StepDef[] {
	const fine = def.detail === 'fine';
	const steps: StepDef[] = [
		{
			id: `train:data@${s}`,
			title: { en: 'The examples it learns from', zh: '它学习用的样例' },
			narration: { en: 'These are the examples the network learns from. Each training step looks at a batch of them and nudges the weights.', zh: '这些是网络学习用的样例。每一步训练看其中一批，并把权重推动一点。' },
			source: s,
			at: 'first',
			view: 'plane2d',
			provenance: 'live'
		}
	];
	if (fine) {
		steps.push(
			{
				id: `train:loss@${s}`,
				title: { en: 'How wrong is it?', zh: '它错了多少？' },
				narration: tmpl({ en: 'The loss measures how wrong the current guesses are: here it is {frame.loss}. Training tries to make this number smaller.', zh: '损失衡量当前的猜测错了多少：这里是 {frame.loss}。训练就是要让这个数变小。' }),
				source: s,
				at: { phase: 'loss' },
				view: 'train-step',
				provenance: 'live'
			},
			{
				id: `train:grad@${s}`,
				title: { en: 'Which way to change each weight', zh: '每个权重该往哪边改' },
				narration: { en: 'Backpropagation gives every weight a gradient: how much the loss would grow if that weight grew a little.', zh: '反向传播给每个权重一个梯度：如果这个权重稍微变大，损失会增加多少。' },
				source: s,
				at: { phase: 'backward' },
				view: 'train-step',
				provenance: 'live'
			},
			{
				id: `train:update@${s}`,
				title: { en: 'Nudge every weight a little', zh: '把每个权重推动一点点' },
				narration: { en: 'Each weight moves a small step against its gradient. That is one training step.', zh: '每个权重朝着梯度的反方向移动一小步。这就是一步训练。' },
				source: s,
				at: { phase: 'update' },
				view: 'train-step',
				provenance: 'live'
			}
		);
	}
	steps.push({
		id: `train:watch@${s}`,
		title: { en: 'Watch it learn', zh: '看它学习' },
		narration: tmpl({ en: 'After {train.step|int} steps the loss is {train.loss}.', zh: '训练 {train.step|int} 步之后，损失是 {train.loss}。' }),
		source: s,
		at: 'last',
		view: 'loss-curve',
		provenance: 'live'
	});
	return steps;
}

function procMotifs(s: string, def: Extract<SourceDef, { kind: 'process' }>, p: ProcessDef<any, any, any> | undefined): StepDef[] {
	const steps: StepDef[] = [];
	for (const [phase, ph] of Object.entries(p?.phases ?? {})) {
		if (!ph.formula && !ph.narration) continue;
		steps.push({
			id: `proc:event.${phase}@${s}`,
			title: ph.label,
			narration: ph.narration?.what ?? { en: 'This is the first time this happens in the run. The formula shows the numbers it used.', zh: '这是运行中第一次发生这件事。公式里是它用到的数。' },
			source: s,
			at: { phase },
			view: ph.formula ? 'zoom.formula' : 'tape.table',
			provenance: def.precomputed ? 'precomputed' : 'live'
		});
	}
	steps.push({
		id: `proc:run@${s}`,
		title: p?.title ?? { en: 'The whole run', zh: '完整的运行' },
		narration: tmpl({ en: 'The whole run has {run.length|int} recorded steps. Scrub through them to see how it unfolds.', zh: '整个运行一共记录了 {run.length|int} 步。拖动进度条看它如何展开。' }),
		source: s,
		at: 'last',
		play: { from: 'first', to: 'last' },
		view: 'tape.table',
		provenance: def.precomputed ? 'precomputed' : 'live'
	});
	return steps;
}
