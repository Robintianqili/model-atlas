// Which tensor layer.tensor shows, and how (pure TS; meta.ts uses it for describe()).
//   props.field   a field of the frame's state (process and training tapes: 'Q', 'params' + props.key, …)
//   else          a layer or sub-step of the trace: props.layer, at.layer, sel.layer (trace frames carry `out`)
// Display by axes (design §5.1): c,y,x tiles · t,d token grid · h,tq,tk head grid · class/vocab bars or logit strip ·
// p,xy points · t series · anything else TensorView.
import type { Axis, I18n, Tensor } from '../../../types/index.ts';
import { axesOfId, layerLabel, stageModel, stageTrace, type Props } from '../kit.parts/stage.ts';

export type Display = 'tiles' | 'tokens' | 'heads' | 'dist' | 'logits' | 'points' | 'series' | 'strip' | 'grid' | 'tensor';

export interface Picked {
	id: string;
	t: Tensor;
	axes: Axis[];
	display: Display;
	label: I18n;
	kind: 'diverging' | 'sequential';
}

function isTensor(x: unknown): x is Tensor {
	const v = x as Tensor;
	return !!v && Array.isArray(v.shape) && !!v.data && typeof (v.data as ArrayLike<number>).length === 'number';
}

/** Rows (along the last axis) that are probability distributions: no negatives, each row sums to 1. */
export function isDistribution(t: Tensor): boolean {
	const n = t.shape[t.shape.length - 1] ?? t.data.length;
	if (!n) return false;
	for (let r = 0; r * n < t.data.length; r++) {
		let s = 0;
		for (let i = r * n; i < (r + 1) * n; i++) {
			const v = t.data[i];
			if (!(v >= 0)) return false;
			s += v;
		}
		if (Math.abs(s - 1) > 1e-3) return false;
	}
	return true;
}

export function displayOf(axes: readonly Axis[], t: Tensor): Display {
	const a = axes.join(',');
	const last = axes[axes.length - 1];
	if (a === 'c,y,x') return 'tiles';
	if (a === 'h,tq,tk') return 'heads';
	if (last === 'class' || last === 'vocab') return isDistribution(t) ? 'dist' : 'logits';
	if (a === 'p,xy') return 'points';
	if (a === 't,d' || a === 't,f') return 'tokens';
	if (a === 't') return 'series';
	if (t.shape.length === 1) return 'strip';
	if (t.shape.length === 2) return 'grid';
	return 'tensor';
}

export function pickTensor(p: Props): Picked | null {
	const field = typeof p.props.field === 'string' ? p.props.field : null;
	if (field) {
		const st = (p.at.state ?? {}) as Record<string, unknown>;
		let v = st[field];
		const key = typeof p.props.key === 'string' ? p.props.key : null;
		if (key && v && typeof v === 'object' && !isTensor(v)) v = (v as Record<string, unknown>)[key];
		if (!isTensor(v)) return null;
		const schema = (p.at.tape as { schema?: Record<string, { label?: I18n; scale?: string }> } | undefined)?.schema?.[field];
		const axes = (Array.isArray(p.props.axes) ? p.props.axes : v.shape.length === 3 ? ['c', 'y', 'x'] : v.shape.length === 2 ? ['y', 'x'] : ['f']) as Axis[];
		const label = schema?.label ?? { en: field, zh: field };
		return {
			id: key ? `${field}.${key}` : field,
			t: v,
			axes,
			display: displayOf(axes, v),
			label: key ? { en: `${label.en} · ${key}`, zh: `${label.zh ?? label.en} · ${key}` } : label,
			kind: schema?.scale === 'sequential' ? 'sequential' : 'diverging'
		};
	}
	const trace = stageTrace(p);
	const model = stageModel(p);
	const fromState = (p.at.state as { id?: string; out?: unknown } | undefined) ?? undefined;
	const id = (typeof p.props.layer === 'string' ? p.props.layer : undefined) ?? fromState?.id ?? p.at.layer ?? p.sel.layer ?? p.at.unit?.layer;
	if (!id) return null;
	const t = trace?.t[id] ?? (fromState?.id === id && isTensor(fromState.out) ? fromState.out : null);
	if (!t) return null;
	const axes = (Array.isArray(p.props.axes) ? p.props.axes : axesOfId(model, id, t.shape)) as Axis[];
	const display = displayOf(axes, t);
	let kind: 'diverging' | 'sequential' = display === 'dist' || display === 'heads' ? (isDistribution(t) ? 'sequential' : 'diverging') : 'diverging';
	try {
		const l = model?.layer(id.split('#')[0]);
		if (l && model?.ops.get(l.op).def.color === 'sequential' && !(display === 'heads' && !isDistribution(t))) kind = 'sequential';
	} catch {
		/* keep the guess */
	}
	return { id, t, axes, display, label: layerLabel(model, id), kind };
}
