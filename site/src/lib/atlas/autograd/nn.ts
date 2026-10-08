// Modules (design §4.5, W1-B1). Parameter names follow PyTorch state_dict names and layouts ('net.0.weight', [out, in]),
// so Python twins load the same tensors and lesson Focus keys match. Init follows PyTorch's default distributions,
// drawn from the Rng you pass (TS is the source of truth for init; twins read the initial weights from TS).
import type { AtlasSpec, Module, Rec, Rng } from '../types/index.ts';
import { Var, param, sizeOf, zeros } from './var.ts';
import * as op from './ops.ts';
import { stackToSpec, type SpecStackItem, type ToSpecOpts } from './tospec.ts';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'silu' | 'gelu' | 'gelu_tanh' | 'identity';

export function activate(x: Var, act: Activation): Var {
	switch (act) {
		case 'tanh':
			return op.tanh(x);
		case 'relu':
			return op.relu(x);
		case 'sigmoid':
			return op.sigmoid(x);
		case 'silu':
			return op.silu(x);
		case 'gelu':
			return op.gelu(x, 'none');
		case 'gelu_tanh':
			return op.gelu(x, 'tanh');
		case 'identity':
			return x;
		default:
			throw new Error(`unknown activation '${String(act)}'`);
	}
}

/** U(−bound, bound), drawn in row-major order. */
function uniform(shape: number[], bound: number, rng: Rng): Float64Array {
	const out = new Float64Array(sizeOf(shape));
	for (let i = 0; i < out.length; i++) out[i] = (2 * rng.next() - 1) * bound;
	return out;
}

function normal(shape: number[], rng: Rng, std = 1): Float64Array {
	const out = new Float64Array(sizeOf(shape));
	for (let i = 0; i < out.length; i++) out[i] = std * rng.normal();
	return out;
}

const prefixed = (name: string, key: string) => (name ? `${name}.${key}` : key);

function namedOf(vars: (Var | null)[]): Record<string, Var> {
	const out: Record<string, Var> = {};
	for (const v of vars) if (v) out[v.name!] = v;
	return out;
}

/** Combine several modules' named parameters (later names must not repeat earlier ones). */
export function namedParams(...mods: { named(): Record<string, Var> }[]): Record<string, Var> {
	const out: Record<string, Var> = {};
	for (const m of mods) {
		for (const [k, v] of Object.entries(m.named())) {
			if (k in out) throw new Error(`duplicate parameter name '${k}'`);
			out[k] = v;
		}
	}
	return out;
}

/** y = x·Wᵀ + b. W [out, in], b [out]; PyTorch init U(±1/√in) for both. */
export class Linear implements Module {
	readonly name: string;
	readonly inFeatures: number;
	readonly outFeatures: number;
	readonly weight: Var;
	readonly bias: Var | null;
	constructor(name: string, inFeatures: number, outFeatures: number, rng: Rng, o: { bias?: boolean } = {}) {
		this.name = name;
		this.inFeatures = inFeatures;
		this.outFeatures = outFeatures;
		const bound = 1 / Math.sqrt(inFeatures);
		this.weight = param(prefixed(name, 'weight'), { shape: [outFeatures, inFeatures], data: uniform([outFeatures, inFeatures], bound, rng) });
		this.bias = o.bias === false ? null : param(prefixed(name, 'bias'), { shape: [outFeatures], data: uniform([outFeatures], bound, rng) });
	}
	forward(x: Var, rec?: Rec): Var {
		const y = op.linear(x, this.weight, this.bias);
		rec?.(this.name, y);
		return y;
	}
	named(): Record<string, Var> {
		return namedOf([this.weight, this.bias]);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
	toSpec(o: { slug: string; inputAxes?: string[] } & ToSpecOpts): { spec: AtlasSpec; weights: Float32Array } {
		return stackToSpec([{ kind: 'linear', id: this.name || 'linear', weight: this.weight, bias: this.bias }], o);
	}
}

/** What may follow the last linear layer of an MLP: an activation, or a softmax over the last axis. */
export type OutAct = Activation | 'softmax';

/**
 * A multilayer perceptron with PyTorch nn.Sequential naming: sizes [2, 8, 8, 2] with act 'tanh' is
 * Sequential(Linear(2,8), Tanh(), Linear(8,8), Tanh(), Linear(8,2)), parameters `${name}.0.weight`, `${name}.2.weight`, ….
 * By default there is no activation after the last layer (the output is the raw scores, "logits"). `outAct` adds one:
 * new MLP('net', [2, 2, 1], 'tanh', rng, { outAct: 'sigmoid' }) is Sequential(Linear, Tanh, Linear, Sigmoid), and the
 * sigmoid is recorded as `${name}.3` (no parameters, so the state_dict names do not change). forward() applies it;
 * logits() stops before it, for a loss that wants the raw scores (bceWithLogits, crossEntropy). toSpec() includes it,
 * so a 'live-model' source shows it as a layer. forward(x, rec) records every layer output under its module name.
 */
export class MLP implements Module {
	readonly name: string;
	readonly sizes: number[];
	readonly act: Activation;
	readonly outAct: OutAct;
	readonly layers: Linear[];
	constructor(name: string, sizes: number[], act: Activation, rng: Rng, o: { bias?: boolean; outAct?: OutAct } = {}) {
		if (sizes.length < 2) throw new Error('MLP needs at least an input and an output size');
		this.name = name;
		this.sizes = sizes.slice();
		this.act = act;
		this.outAct = o.outAct ?? 'identity';
		this.layers = [];
		for (let i = 0; i + 1 < sizes.length; i++) this.layers.push(new Linear(prefixed(name, String(2 * i)), sizes[i], sizes[i + 1], rng, { bias: o.bias }));
	}
	/** The raw scores of the last linear layer (no output activation). */
	logits(x: Var, rec?: Rec): Var {
		let h = x;
		const L = this.layers.length;
		for (let i = 0; i < L; i++) {
			h = this.layers[i].forward(h, rec);
			if (i + 1 < L && this.act !== 'identity') {
				h = activate(h, this.act);
				rec?.(prefixed(this.name, String(2 * i + 1)), h);
			}
		}
		return h;
	}
	forward(x: Var, rec?: Rec): Var {
		const z = this.logits(x, rec);
		if (this.outAct === 'identity') return z;
		const y = this.outAct === 'softmax' ? op.softmax(z, -1) : activate(z, this.outAct);
		rec?.(prefixed(this.name, String(2 * this.layers.length - 1)), y);
		return y;
	}
	named(): Record<string, Var> {
		return namedParams(...this.layers);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
	/** The kind-A spec of this MLP on its current weights (stored as f32), output activation included. */
	toSpec(o: { slug: string; inputAxes?: string[] } & ToSpecOpts): { spec: AtlasSpec; weights: Float32Array } {
		const items: SpecStackItem[] = [];
		this.layers.forEach((l, i) => {
			items.push({ kind: 'linear', id: l.name, weight: l.weight, bias: l.bias });
			if (i + 1 < this.layers.length && this.act !== 'identity') items.push({ kind: 'act', id: prefixed(this.name, String(2 * i + 1)), act: this.act });
		});
		if (this.outAct !== 'identity') items.push({ kind: 'act', id: prefixed(this.name, String(2 * this.layers.length - 1)), act: this.outAct });
		return stackToSpec(items, o);
	}
}

/** A lookup table: weight [num, dim], PyTorch init N(0, 1). forward(ids) = rows of the table. */
export class Embedding implements Module {
	readonly name: string;
	readonly weight: Var;
	constructor(name: string, num: number, dim: number, rng: Rng) {
		this.name = name;
		this.weight = param(prefixed(name, 'weight'), { shape: [num, dim], data: normal([num, dim], rng) });
	}
	forward(ids: ArrayLike<number>, rec?: Rec): Var {
		const y = op.gather(this.weight, 0, ids);
		rec?.(this.name, y);
		return y;
	}
	named(): Record<string, Var> {
		return namedOf([this.weight]);
	}
	params(): Var[] {
		return [this.weight];
	}
}

export interface LSTMState {
	h: Var;
	c: Var;
}

/**
 * One LSTM step with explicit weights (the math of PyTorch's lstm_cell): gates = x·W_ihᵀ + b_ih + h·W_hhᵀ + b_hh,
 * split into i, f, g, o; c' = σ(f)⊙c + σ(i)⊙tanh(g); h' = σ(o)⊙tanh(c'). rec ids `${name}#i` … `#c`, `#h`.
 */
export function lstmCell(
	x: Var,
	state: LSTMState | null,
	w: { weight_ih: Var; weight_hh: Var; bias_ih: Var | null; bias_hh: Var | null },
	rec?: Rec,
	name = 'lstm'
): LSTMState {
	const H = w.weight_hh.shape[1];
	const lead = x.shape.slice(0, -1);
	const h0 = state?.h ?? zeros([...lead, H]);
	const c0 = state?.c ?? zeros([...lead, H]);
	const gates = op.add(op.linear(x, w.weight_ih, w.bias_ih), op.linear(h0, w.weight_hh, w.bias_hh));
	const i = op.sigmoid(op.slice(gates, -1, 0, H));
	const f = op.sigmoid(op.slice(gates, -1, H, 2 * H));
	const g = op.tanh(op.slice(gates, -1, 2 * H, 3 * H));
	const o = op.sigmoid(op.slice(gates, -1, 3 * H, 4 * H));
	const c = op.add(op.mul(f, c0), op.mul(i, g));
	const h = op.mul(o, op.tanh(c));
	if (rec) {
		rec(`${name}#i`, i);
		rec(`${name}#f`, f);
		rec(`${name}#g`, g);
		rec(`${name}#o`, o);
		rec(`${name}#c`, c);
		rec(`${name}#h`, h);
	}
	return { h, c };
}

/**
 * PyTorch nn.LSTMCell: weight_ih [4H, In], weight_hh [4H, H], bias_ih [4H], bias_hh [4H] (both biases kept),
 * gate order i, f, g, o; c' = f⊙c + i⊙g; h' = o⊙tanh(c'); init U(±1/√H). rec ids: `${name}#i` … `#c`, `#h`.
 */
export class LSTMCell implements Module {
	readonly name: string;
	readonly inputSize: number;
	readonly hidden: number;
	readonly weight_ih: Var;
	readonly weight_hh: Var;
	readonly bias_ih: Var;
	readonly bias_hh: Var;
	constructor(name: string, inputSize: number, hidden: number, rng: Rng) {
		this.name = name;
		this.inputSize = inputSize;
		this.hidden = hidden;
		const b = 1 / Math.sqrt(hidden);
		const H4 = 4 * hidden;
		this.weight_ih = param(prefixed(name, 'weight_ih'), { shape: [H4, inputSize], data: uniform([H4, inputSize], b, rng) });
		this.weight_hh = param(prefixed(name, 'weight_hh'), { shape: [H4, hidden], data: uniform([H4, hidden], b, rng) });
		this.bias_ih = param(prefixed(name, 'bias_ih'), { shape: [H4], data: uniform([H4], b, rng) });
		this.bias_hh = param(prefixed(name, 'bias_hh'), { shape: [H4], data: uniform([H4], b, rng) });
	}
	/** One step. x [N, In] or [In]; state defaults to zeros. */
	forward(x: Var, state?: LSTMState | null, rec?: Rec): LSTMState {
		return lstmCell(x, state ?? null, this, rec, this.name);
	}
	/** Unroll over a sequence x [T, In] (or [N, T, In], batch first). Returns every hidden state stacked on the time axis. */
	run(x: Var, state?: LSTMState | null): { hs: Var; h: Var; c: Var } {
		const tAxis = x.shape.length - 2;
		const T = x.shape[tAxis];
		let s: LSTMState | null = state ?? null;
		const hs: Var[] = [];
		for (let t = 0; t < T; t++) {
			const xt = op.reshape(op.slice(x, tAxis, t, t + 1), x.shape.filter((_, d) => d !== tAxis));
			s = this.forward(xt, s);
			hs.push(s.h);
		}
		if (!s) throw new Error('LSTMCell.run: empty sequence');
		return { hs: op.stack(hs, tAxis), h: s.h, c: s.c };
	}
	named(): Record<string, Var> {
		return namedOf([this.weight_ih, this.weight_hh, this.bias_ih, this.bias_hh]);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}

/** PyTorch nn.Conv1d (+ causal padding for TCNs): weight [O, C, k], bias [O], init U(±1/√(C·k)). */
export class Conv1d implements Module {
	readonly name: string;
	readonly weight: Var;
	readonly bias: Var | null;
	readonly opts: op.Conv1dOpts;
	constructor(name: string, inChannels: number, outChannels: number, kernel: number, rng: Rng, o: op.Conv1dOpts & { bias?: boolean } = {}) {
		this.name = name;
		const bound = 1 / Math.sqrt(inChannels * kernel);
		this.weight = param(prefixed(name, 'weight'), { shape: [outChannels, inChannels, kernel], data: uniform([outChannels, inChannels, kernel], bound, rng) });
		this.bias = o.bias === false ? null : param(prefixed(name, 'bias'), { shape: [outChannels], data: uniform([outChannels], bound, rng) });
		this.opts = { stride: o.stride, dilation: o.dilation, padding: o.padding, causal: o.causal };
	}
	forward(x: Var, rec?: Rec): Var {
		const y = op.conv1d(x, this.weight, this.bias, this.opts);
		rec?.(this.name, y);
		return y;
	}
	named(): Record<string, Var> {
		return namedOf([this.weight, this.bias]);
	}
	params(): Var[] {
		return Object.values(this.named());
	}
}

/** PyTorch nn.LayerNorm over the last axis: weight ones, bias zeros. */
export class LayerNorm implements Module {
	readonly name: string;
	readonly weight: Var;
	readonly bias: Var;
	readonly eps: number;
	constructor(name: string, dim: number, o: { eps?: number } = {}) {
		this.name = name;
		this.eps = o.eps ?? 1e-5;
		this.weight = param(prefixed(name, 'weight'), { shape: [dim], data: new Float64Array(dim).fill(1) });
		this.bias = param(prefixed(name, 'bias'), { shape: [dim], data: new Float64Array(dim) });
	}
	forward(x: Var, rec?: Rec): Var {
		const y = op.layerNorm(x, this.weight, this.bias, this.eps);
		rec?.(this.name, y);
		return y;
	}
	named(): Record<string, Var> {
		return namedOf([this.weight, this.bias]);
	}
	params(): Var[] {
		return [this.weight, this.bias];
	}
}

/** Set parameter values by name (twins, fixtures, precomputed checkpoints). Every given name must exist with the same size. */
export function loadParams(m: { named(): Record<string, Var> }, values: Record<string, { shape?: number[]; data: ArrayLike<number> }>): void {
	const named = m.named();
	for (const [k, v] of Object.entries(values)) {
		const p = named[k];
		if (!p) throw new Error(`loadParams: no parameter named '${k}'`);
		if (v.data.length !== p.size) throw new Error(`loadParams: '${k}' has ${v.data.length} values, the parameter has ${p.size}`);
		p.data.set(v.data as ArrayLike<number>);
	}
}
