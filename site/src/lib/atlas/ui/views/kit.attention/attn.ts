// The attention weights kit.attention shows (pure TS; meta.ts uses it for describe()).
import type { I18n } from '../../../types/index.ts';
import { layerLabel, stageLayerId, stageModel, stageTrace, type Props } from '../kit.parts/stage.ts';

export interface Attn {
	id: string;
	head: number;
	heads: number;
	tq: number;
	tk: number;
	/** the head's [Tq, Tk] weights */
	w: Float32Array | Float64Array;
	causal: boolean;
	query: number;
	label: I18n;
}

export function attnOf(p: Props, headPick: number | null = null): Attn | null {
	const trace = stageTrace(p);
	const model = stageModel(p);
	if (!trace) return null;
	let id = stageLayerId(p);
	if (!id || !trace.t[id] || trace.t[id].shape.length !== 3) {
		// the first [H, T, T] layer that holds probabilities (rows sum to 1)
		id = trace.order.find((k) => {
			const t = trace.t[k];
			if (t.shape.length !== 3) return false;
			let s = 0;
			for (let j = 0; j < t.shape[2]; j++) s += t.data[j];
			return Math.abs(s - 1) < 1e-3;
		});
	}
	if (!id) return null;
	const t = trace.t[id];
	const [H, Tq, Tk] = t.shape;
	const unit = p.at.unit?.layer === id ? p.at.unit.unit : p.sel.layer === id && Array.isArray(p.sel.unit) ? p.sel.unit : null;
	const head = Math.max(0, Math.min(H - 1, headPick ?? (typeof p.props.head === 'number' ? p.props.head : (unit?.[0] ?? 0))));
	const w = t.data.subarray(head * Tq * Tk, (head + 1) * Tq * Tk);
	let causal = Tq > 1;
	for (let i = 0; i < Tq && causal; i++) for (let j = i + 1; j < Tk; j++) if (w[i * Tk + j] !== 0) (causal = false), (j = Tk);
	const query = Math.max(0, Math.min(Tq - 1, typeof p.props.query === 'number' ? p.props.query : (unit?.[1] ?? Tq - 1)));
	return { id, head, heads: H, tq: Tq, tk: Tk, w, causal, query, label: layerLabel(model, id) };
}
