// Factory for elementwise activations (relu, tanh, sigmoid, silu, gelu): one value in, one value out,
// same shape, the shared `zoom.elementwise` view with the point on the curve.
import type { AtlasLayer, AttrDecl, Axis, Box, I18n, NarrationSet, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { baseVars, outArray, store, unitBase } from './common.ts';

export interface PointwiseDef<P extends Record<string, unknown>> {
	op: string;
	label: I18n;
	attrs?: Record<string, AttrDecl>;
	aliases?: Record<string, Partial<P>>;
	/** f64 in, f64 out; the caller stores it (float32 rounding happens on store) */
	fn: (x: number, p: P) => number;
	/** curve id for the elementwise view ('relu', 'tanh', 'sigmoid', 'silu', 'gelu', 'gelu_tanh') */
	curve: (p: P) => string;
	/** extra named values for narration and the formula (e.g. GELU's Φ(x)) */
	extra?: (x: number, p: P) => Record<string, number>;
	narration: NarrationSet;
	/** extra template variables from the layer (baseVars are always included), e.g. GELU's exact-or-tanh phrase */
	vars?: (layer: AtlasLayer) => Record<string, string | number>;
	formula: string;
	color?: 'diverging' | 'sequential';
	terms?: string[];
	pytorch: (layer: AtlasLayer, p: P) => string;
	defaults: P;
}

export function pointwise<P extends Record<string, unknown>>(d: PointwiseDef<P>): OpDef<P> {
	const params = (layer: AtlasLayer): P => ({ ...d.defaults, ...(layer.params as Partial<P>) }) as P;
	return {
		op: d.op,
		version: 1,
		attrs: d.attrs ?? {},
		aliases: d.aliases,
		category: 'activation',
		label: d.label,
		inferShape: (ins) => ins[0].slice(),
		axes: (ins: Axis[][]) => ins[0].slice(),
		forward(xs, p) {
			const x = xs[0];
			const n = x.data.length;
			const y = outArray(n, x);
			const fn = d.fn;
			for (let i = 0; i < n; i++) y[i] = fn(x.data[i], p);
			return { shape: x.shape.slice(), data: y };
		},
		unit: { axes: (out: Axis[]) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
		explain(q): UnitExplanation {
			const f = flatIndex(q.out.shape, q.unit);
			const x = q.ins[0].data[f];
			const y = store(d.fn(x, q.p), q.out);
			const e = unitBase('elementwise', q.layer, q.unit, y, q.out.data[f]);
			e.reads = [{ tensor: q.layer.in[0], cells: [f] }];
			e.curve = { fn: d.curve(q.p), x, y };
			e.values = { x, y, ...(d.extra ? d.extra(x, q.p) : {}) };
			return e;
		},
		receptive: (box: Box) => ({ ...box }),
		formula: d.formula,
		narration: d.narration,
		vars: (layer, spec) => ({ ...baseVars(layer, spec), ...(d.vars ? d.vars(layer) : {}) }),
		zoom: 'elementwise',
		color: d.color ?? 'diverging',
		parity: { bar: 1e-4 },
		pytorch: (layer) => d.pytorch(layer, params(layer)),
		terms: d.terms ?? ['activation-function']
	};
}
