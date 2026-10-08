// The distribution kit.probbars shows (pure TS; meta.ts uses it for describe()).
import type { I18n } from '../../../types/index.ts';
import { answerRow, outputId, outputRow } from '../../../lesson/facts.ts';
import type { AnswerRow } from '../../../types/index.ts';
import { axesOfId, layerLabel, stageModel, stageTrace, type Props } from '../kit.parts/stage.ts';
import { isDistribution } from '../layer.tensor/pick.ts';

export interface Dist {
	id: string;
	p: Float64Array;
	/** which row of a [T, V] output (props.row, else the package's answer row on its output layer, else the last; K2) */
	row: number;
	rows: number;
	/** K2: the rows read as "the prediction after this token" (a causal model: no answer row other than the last) */
	next: boolean;
	top: number;
	axis: 'class' | 'vocab' | 'other';
	softmaxed: boolean;
	label: I18n;
}

export function distOf(p: Props, rowPick: number | null = null): Dist | null {
	const trace = stageTrace(p);
	const model = stageModel(p);
	if (!trace) return null;
	const spec = model?.spec ?? p.rt.spec;
	const id = (typeof p.props.layer === 'string' ? p.props.layer : undefined) ?? outputId(spec, trace);
	const t = id ? trace.t[id] : null;
	if (!id || !t) return null;
	const axes = axesOfId(model, id, t.shape);
	const last = axes[axes.length - 1];
	const V = t.shape[t.shape.length - 1];
	const rows = t.data.length / V;
	// K2: props.row ('last', 'first', an index, a function of the runtime), else manifest output.row on the output layer
	const isOut = id === outputId(spec, trace);
	const want = p.props.row !== undefined ? answerRow(p.props.row as AnswerRow, rows, p.rt) : isOut ? outputRow(p.rt, t) : rows - 1;
	const row = Math.max(0, Math.min(rows - 1, rowPick ?? want));
	const x = t.data.subarray(row * V, (row + 1) * V);
	const dist = isDistribution({ shape: [V], data: x });
	let pr: Float64Array;
	if (dist) pr = Float64Array.from(x);
	else {
		let m = -Infinity;
		for (const v of x) if (v > m) m = v;
		pr = Float64Array.from(x, (v) => Math.exp(v - m));
		const z = pr.reduce((a, b) => a + b, 0);
		for (let i = 0; i < V; i++) pr[i] /= z;
	}
	let top = 0;
	for (let i = 1; i < V; i++) if (pr[i] > pr[top]) top = i;
	const rowOpt = p.props.row !== undefined ? p.props.row : isOut ? (p.rt as { pkg?: { manifest?: { output?: { row?: unknown } } } }).pkg?.manifest?.output?.row : undefined;
	return { id, p: pr, row, rows, next: rowOpt === undefined || rowOpt === 'last', top, axis: last === 'class' || last === 'vocab' ? last : 'other', softmaxed: !dist, label: layerLabel(model, id) };
}
