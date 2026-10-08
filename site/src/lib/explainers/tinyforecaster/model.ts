// tinyforecaster: the network. It reads a window of L = 24 values as differences from the window's last value, an LSTM
// cell (1 input, 16 hidden numbers) runs over them, and a linear head turns the last hidden state h into the predicted
// change; the forecast is the last value plus that change:
//   r_t = x_t − x_23;   for t = 0..23: (h, c) = lstm(r_t, (h, c)), starting from h = c = 0
//   forecast = x_23 + (head.weight · h + head.bias)
// PyTorch twin: nn.LSTMCell(1, 16) named 'lstm' + nn.Linear(16, 1) named 'head' (pipeline/tinyforecaster/twin.py).
// toSpec(): the same network as a kind-A graph for the 'live-model' sources, on the kit's shared `lstm` op (gate
// sub-steps i f g o c h, view zoom.gate), `linear` and `add`, and the local ops `relative` and `last_step` (ops/steps.ts).
import type { AtlasSpec, Module, Rec, Rng, Tensor } from '../../atlas/types/index.ts';
import { LSTMCell, Linear, namedParams, type LSTMState } from '../../atlas/autograd/nn.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { input, noGrad, type Var } from '../../atlas/autograd/var.ts';
import { L } from './data.ts';

export const HIDDEN = 16;
const T2 = (en: string, zh: string) => ({ en, zh });

export class Forecaster implements Module {
	readonly lstm: LSTMCell;
	readonly head: Linear;
	constructor(rng: Rng, hidden = HIDDEN) {
		this.lstm = new LSTMCell('lstm', 1, hidden, rng);
		this.head = new Linear('head', hidden, 1, rng);
	}
	/** x [B, T]: B windows of T values -> [B, 1] forecasts. rec: the last step's gates ('lstm#i' … 'lstm#h') and 'head'. */
	forward(x: Var, rec?: Rec): Var {
		const T = x.shape[1];
		const last = op.slice(x, 1, T - 1, T);
		const r = op.sub(x, last);
		let s: LSTMState | null = null;
		for (let t = 0; t < T; t++) s = this.lstm.forward(op.slice(r, 1, t, t + 1), s, t === T - 1 ? rec : undefined);
		if (!s) throw new Error('Forecaster: empty window');
		return op.add(last, this.head.forward(s.h, rec));
	}
	named() {
		return namedParams(this.lstm, this.head);
	}
	params() {
		return Object.values(this.named());
	}
	/** window [24, 1] -> relative -> lstm [24, 16] -> last [16] -> head [1]; lastv [1]; out = lastv + head. The weights stay
	 *  float64, as trained (Module.toSpec is typed Float32Array: the cast keeps every lesson number equal to the trainer's;
	 *  notes, Kit issues), so the trace equals the autograd forward to rounding (≤ 1e-12, ops.test.ts). */
	toSpec(o: { slug: string }): { spec: AtlasSpec; weights: Float32Array } {
		const H = this.lstm.hidden;
		const named = this.named();
		let offset = 0;
		const tensors = Object.keys(named).map((name) => {
			const e = { name, shape: named[name].shape.slice(), offset, length: named[name].size };
			offset += e.length;
			return e;
		});
		const weights = new Float64Array(offset);
		for (const e of tensors) weights.set(named[e.name].data, e.offset);
		const spec: AtlasSpec = {
			slug: o.slug,
			name: T2('TinyForecaster (live weights)', 'TinyForecaster（实时权重）'),
			version: 'live',
			params: offset,
			precision: 'f64',
			tensors,
			layers: [
				{ id: 'window', op: 'input', label: T2('The window', '窗口'), in: ['window'], outShape: [L, 1], params: { name: 'window' }, axes: ['t', 'd'] },
				{ id: 'rel', op: 'relative', label: T2('Minus the last value', '减去最后一个值'), in: ['window'], outShape: [L, 1], axes: ['t', 'd'] },
				{ id: 'lstm', op: 'lstm', label: T2(`LSTM, ${H} numbers of memory`, `LSTM，${H} 个记忆数`), in: ['rel'], outShape: [L, H], params: { hidden: H, inputSize: 1 }, sub: ['i', 'f', 'g', 'o', 'c', 'h'], axes: ['t', 'd'] },
				{ id: 'last', op: 'last_step', label: T2('The last h', '最后一个 h'), in: ['lstm'], outShape: [H], axes: ['d'] },
				{ id: 'head', op: 'linear', label: T2('Predicted change', '预测的变化量'), in: ['last'], outShape: [1], axes: ['f'] },
				{ id: 'lastv', op: 'last_step', label: T2('The last value', '最后一个值'), in: ['window'], outShape: [1], axes: ['f'] },
				{ id: 'out', op: 'add', label: T2('The forecast', '预测值'), in: ['lastv', 'head'], outShape: [1], axes: ['f'] }
			],
			parity: { against: 'atlas-autograd-f64', max_abs_diff: null, checked: null, scope: 'forward' },
			atlas: { spec: '1.1', kinds: ['trace'], exporter: 'explainers/tinyforecaster/model.ts' },
			inputs: [{ name: 'window', adapter: 'series', kind: 'series', shape: [L, 1], dtype: 'f32', axes: ['t', 'd'] }],
			outputs: ['out']
		};
		return { spec, weights: weights as unknown as Float32Array };
	}
}

/** A fresh network with recorded parameters (a snapshot's state.params). */
export function netOf(params: Record<string, Tensor>, rng: Rng): Forecaster {
	const m = new Forecaster(rng);
	for (const [k, v] of Object.entries(m.named())) {
		const t = params[k];
		if (!t || t.data.length !== v.size) throw new Error(`tinyforecaster: parameter '${k}' is missing or has another size`);
		v.data.set(t.data);
	}
	return m;
}

/** One forecast per window (rows of `x`, L values each), without gradients. */
export function forecastAll(m: Forecaster, x: ArrayLike<number>, count: number): Float64Array {
	if (count === 0) return new Float64Array(0);
	return Float64Array.from(noGrad(() => m.forward(input({ shape: [count, L], data: Float64Array.from(x) }))).data);
}
