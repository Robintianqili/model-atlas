// Unit explanations over a whole trace (design §4.3, §6.2 item 8: `explain` equals the trace, all units or a sample).
//   checkExplain(model, trace, o)  -> { units, max, worst, failures }   |result − traced| per unit (−∞ == −∞)
//   pickUnit(model, trace, layer, pick) -> number[]                      the unit an auto step starts on (UnitPick)
//   unitCount(trace, id)                                                  number of units of a layer or sub-step
//
// What the fields of a UnitExplanation mean per kind (views and narration rely on this):
//   weighted-sum  terms[i] = { x: input, w: weight, p: x·w, xAt, wAt }; groups = per input channel / head; bias; values.s = Σp
//                 (conv2d, linear, attention_scores [x = q, w = k], attention_mix [x = v, w = attention weight])
//   reduce        terms[i] = { x: window value, w: 1 for the winner else 0, p: x·w }; values.argmax = winner (maxpool2d)
//   elementwise   values.x, values.y, curve { fn, x, y } (activations); add: terms [a, b] with w = 1, values.a / b
//   normalize     steps = mean, variance, std, x̂, ·γ, +β; values.mean / var / std / xhat / gamma / beta (layernorm)
//   softmax       terms[j] over the row = { x: score, w: exp(score − max), p: share }; steps; values.max / e / sum
//   lookup        values.id (embedding) / t, d; reads = the id and the table cell
//   index-map     values.index (and c/row/col for a [C,H,W] source) or values.part (concat); causal_mask: values.masked
//   gate          steps = f, i, g, c, o, h; values i f g o c h cPrev hPrev and the pre-activations zi zf zg zo (lstm)
//   none          values.y (input)
import type { GraphModel, Trace, UnitExplanation, UnitPick } from '../types/index.ts';
import { argmax, argmaxAbs, argmin, numel, unravel } from '../core/tensor.ts';
import { createRng, sampleIndices } from '../core/rng.ts';

export interface ExplainCheck {
	/** units explained */
	units: number;
	/** largest |result − traced| (Infinity when one side is NaN or an unmatched infinity) */
	max: number;
	worst: { layer: string; unit: number[]; result: number; traced: number } | null;
	/** units over the bar (at most 20 kept) */
	failures: { layer: string; unit: number[]; result: number; traced: number; error?: string }[];
	/** per layer id: units checked and the largest difference */
	perLayer: Record<string, { units: number; max: number }>;
}

function diff(a: number, b: number): number {
	if (a === b) return 0;
	const d = Math.abs(a - b);
	return Number.isNaN(d) ? Infinity : d;
}

/** Number of units (cells) of a traced tensor. */
export function unitCount(trace: Trace, id: string): number {
	const t = trace.t[id];
	return t ? numel(t.shape) : 0;
}

/**
 * Explain units and compare with the trace.
 *   layers: ids (and sub ids) to check; default every id of the trace that the model can explain
 *   sample: when set, check this many units drawn uniformly from all units of those layers (seeded); else every unit
 *   bar:    the gate (default 1e-5)
 */
export function checkExplain(model: GraphModel, trace: Trace, o: { layers?: string[]; sample?: number; seed?: number; bar?: number } = {}): ExplainCheck {
	const bar = o.bar ?? 1e-5;
	const ids = (o.layers ?? trace.order).filter((id) => {
		const parent = id.split('#')[0];
		try {
			const l = model.layer(parent);
			return !!model.ops.get(l.op).def.explain;
		} catch {
			return false;
		}
	});
	const res: ExplainCheck = { units: 0, max: 0, worst: null, failures: [], perLayer: {} };
	const visit = (id: string, flat: number) => {
		const t = trace.t[id];
		const unit = unravel(t.shape, flat);
		let e: UnitExplanation | null = null;
		let err: string | undefined;
		try {
			e = model.explain(trace, id, unit);
		} catch (x) {
			err = (x as Error).message;
		}
		const d = e ? diff(e.result, e.traced) : Infinity;
		const pl = (res.perLayer[id] ??= { units: 0, max: 0 });
		pl.units++;
		res.units++;
		if (d > pl.max) pl.max = d;
		if (res.worst === null || d > res.max) {
			res.max = d;
			res.worst = { layer: id, unit, result: e?.result ?? NaN, traced: e?.traced ?? t.data[flat] };
		}
		if (!(d <= bar) && res.failures.length < 20) res.failures.push({ layer: id, unit, result: e?.result ?? NaN, traced: t.data[flat], error: err });
	};
	if (o.sample === undefined) {
		for (const id of ids) {
			const n = unitCount(trace, id);
			for (let f = 0; f < n; f++) visit(id, f);
		}
	} else {
		const counts = ids.map((id) => unitCount(trace, id));
		const total = counts.reduce((a, b) => a + b, 0);
		const rng = createRng(o.seed ?? 1);
		for (const g of sampleIndices(rng, total, o.sample)) {
			let k = 0;
			let r = g;
			while (r >= counts[k]) r -= counts[k++];
			visit(ids[k], r);
		}
	}
	return res;
}

/** The unit an auto step starts on, for a layer (or sub id) of a trace. */
export function pickUnit(model: GraphModel, trace: Trace, id: string, pick: UnitPick): number[] {
	const t = trace.t[id];
	if (!t) throw new Error(`pickUnit: the trace has no '${id}'`);
	const n = t.data.length;
	if (typeof pick === 'object') return pick.unit.slice();
	switch (pick) {
		case 'first':
			return unravel(t.shape, 0);
		case 'last':
			return unravel(t.shape, n - 1);
		case 'center':
			return t.shape.map((d) => Math.floor((d - 1) / 2));
		case 'argmin':
			return unravel(t.shape, argmin(t.data));
		case 'argmaxAbs':
			return unravel(t.shape, argmaxAbs(t.data));
		case 'strongestChannel': {
			if (t.shape.length !== 3) return unravel(t.shape, argmax(t.data));
			// the channel with the largest total activation, then its strongest cell
			const [C, H, W] = t.shape;
			let bestC = 0;
			let bestS = -Infinity;
			for (let c = 0; c < C; c++) {
				let s = 0;
				for (let i = 0; i < H * W; i++) s += t.data[c * H * W + i];
				if (s > bestS) {
					bestS = s;
					bestC = c;
				}
			}
			return unravel(t.shape, argmax(t.data, bestC * H * W, (bestC + 1) * H * W));
		}
		case 'argmax':
		default: {
			void model;
			// rows (e.g. TinyGPT [T, V]): the argmax of the last row is the useful unit
			if (t.shape.length === 2) {
				const [T, V] = t.shape;
				return unravel(t.shape, argmax(t.data, (T - 1) * V, T * V));
			}
			return unravel(t.shape, argmax(t.data));
		}
	}
}
