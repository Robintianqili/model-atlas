// Optimizers (design §4.5, W1-B1), written to follow PyTorch's single-tensor CPU formulas step for step
// (torch.optim.SGD, SGD with momentum, Adam, AdamW), so fp64 twins agree to rounding.
// They update parameter values in place and read p.grad. Parameters whose grad is null are skipped, as in PyTorch.
// state() exposes the moment buffers (m, v) for the optimizer glyphs; they are live: copy before storing them.
import type { Optimizer, Tensor } from '../types/index.ts';
import { Var } from './var.ts';

export type ParamsIn = Var[] | Record<string, Var> | { named(): Record<string, Var> };
export type OptimizerKind = 'sgd' | 'momentum' | 'adam' | 'adamw';

function entries(ps: ParamsIn): [string, Var][] {
	if (Array.isArray(ps)) return ps.map((p, i) => [p.name ?? `p${i}`, p]);
	if (typeof (ps as { named?: unknown }).named === 'function') return Object.entries((ps as { named(): Record<string, Var> }).named());
	return Object.entries(ps as Record<string, Var>);
}

type Settings = Partial<{ lr: number; momentum: number; betas: [number, number]; weightDecay: number }>;

/** Plain SGD: p ← p − lr·(g + wd·p). */
export class SGD implements Optimizer {
	readonly kind: OptimizerKind = 'sgd';
	lr: number;
	weightDecay: number;
	readonly params: [string, Var][];
	constructor(params: ParamsIn, o: { lr: number; weightDecay?: number }) {
		this.params = entries(params);
		this.lr = o.lr;
		this.weightDecay = o.weightDecay ?? 0;
	}
	step(): void {
		const lr = this.lr;
		const wd = this.weightDecay;
		for (const [, p] of this.params) {
			if (!p.grad) continue;
			const w = p.data;
			const g = p.grad.data;
			if (wd !== 0) for (let i = 0; i < w.length; i++) w[i] = w[i] + -lr * (g[i] + wd * w[i]);
			else for (let i = 0; i < w.length; i++) w[i] = w[i] + -lr * g[i];
		}
	}
	set(o: Settings): void {
		if (o.lr !== undefined) this.lr = o.lr;
		if (o.weightDecay !== undefined) this.weightDecay = o.weightDecay;
	}
	state(): Record<string, { m?: Tensor; v?: Tensor }> {
		return {};
	}
}

/** SGD with momentum (PyTorch): buf ← g on the first step, then buf ← μ·buf + (1 − dampening)·g; p ← p − lr·buf (or Nesterov). */
export class Momentum implements Optimizer {
	readonly kind: OptimizerKind = 'momentum';
	lr: number;
	momentum: number;
	dampening: number;
	nesterov: boolean;
	weightDecay: number;
	readonly params: [string, Var][];
	readonly buf = new Map<string, Float64Array>();
	constructor(params: ParamsIn, o: { lr: number; momentum?: number; dampening?: number; nesterov?: boolean; weightDecay?: number }) {
		this.params = entries(params);
		this.lr = o.lr;
		this.momentum = o.momentum ?? 0.9;
		this.dampening = o.dampening ?? 0;
		this.nesterov = o.nesterov ?? false;
		this.weightDecay = o.weightDecay ?? 0;
	}
	step(): void {
		const { lr, momentum: mu, dampening, nesterov, weightDecay: wd } = this;
		for (const [name, p] of this.params) {
			if (!p.grad) continue;
			const w = p.data;
			const g = p.grad.data;
			const d = new Float64Array(w.length);
			for (let i = 0; i < w.length; i++) d[i] = wd !== 0 ? g[i] + wd * w[i] : g[i];
			let b = this.buf.get(name);
			if (!b) {
				b = d.slice();
				this.buf.set(name, b);
			} else {
				const a = 1 - dampening;
				for (let i = 0; i < b.length; i++) b[i] = b[i] * mu + a * d[i];
			}
			if (nesterov) for (let i = 0; i < w.length; i++) w[i] = w[i] + -lr * (d[i] + mu * b[i]);
			else for (let i = 0; i < w.length; i++) w[i] = w[i] + -lr * b[i];
		}
	}
	set(o: Settings): void {
		if (o.lr !== undefined) this.lr = o.lr;
		if (o.momentum !== undefined) this.momentum = o.momentum;
		if (o.weightDecay !== undefined) this.weightDecay = o.weightDecay;
	}
	state(): Record<string, { m?: Tensor; v?: Tensor }> {
		const out: Record<string, { m?: Tensor; v?: Tensor }> = {};
		for (const [name, p] of this.params) {
			const b = this.buf.get(name);
			if (b) out[name] = { m: { shape: p.shape.slice(), data: b } };
		}
		return out;
	}
}

/**
 * Adam (PyTorch single-tensor formula). m ← lerp(m, g, 1 − β1); v ← β2·v + (1 − β2)·g·g;
 * p ← p − (lr / (1 − β1^t)) · m / (sqrt(v) / sqrt(1 − β2^t) + eps).
 * weightDecay: L2 added to the gradient (Adam) or decoupled p ← p·(1 − lr·wd) first (AdamW, `decoupled`).
 */
export class Adam implements Optimizer {
	readonly kind: OptimizerKind;
	lr: number;
	betas: [number, number];
	eps: number;
	weightDecay: number;
	readonly decoupled: boolean;
	readonly params: [string, Var][];
	readonly m = new Map<string, Float64Array>();
	readonly v = new Map<string, Float64Array>();
	readonly t = new Map<string, number>();
	constructor(params: ParamsIn, o: { lr?: number; betas?: [number, number]; eps?: number; weightDecay?: number; decoupled?: boolean } = {}) {
		this.params = entries(params);
		this.lr = o.lr ?? 1e-3;
		this.betas = o.betas ?? [0.9, 0.999];
		this.eps = o.eps ?? 1e-8;
		this.decoupled = o.decoupled ?? false;
		this.weightDecay = o.weightDecay ?? (this.decoupled ? 1e-2 : 0);
		this.kind = this.decoupled ? 'adamw' : 'adam';
	}
	step(): void {
		const [b1, b2] = this.betas;
		const { lr, eps, weightDecay: wd } = this;
		for (const [name, p] of this.params) {
			if (!p.grad) continue;
			const w = p.data;
			let g = p.grad.data;
			const n = w.length;
			let m = this.m.get(name);
			let v = this.v.get(name);
			if (!m || !v) {
				m = new Float64Array(n);
				v = new Float64Array(n);
				this.m.set(name, m);
				this.v.set(name, v);
			}
			const t = (this.t.get(name) ?? 0) + 1;
			this.t.set(name, t);
			if (wd !== 0) {
				if (this.decoupled) {
					const f = 1 - lr * wd;
					for (let i = 0; i < n; i++) w[i] = w[i] * f;
				} else {
					const gg = new Float64Array(n);
					for (let i = 0; i < n; i++) gg[i] = g[i] + wd * w[i];
					g = gg;
				}
			}
			const wl = 1 - b1;
			const c2 = 1 - b2;
			for (let i = 0; i < n; i++) {
				// torch lerp_: weight < 0.5 → self + weight·(end − self)
				m[i] = wl < 0.5 ? m[i] + wl * (g[i] - m[i]) : g[i] - (g[i] - m[i]) * (1 - wl);
				v[i] = v[i] * b2 + c2 * g[i] * g[i];
			}
			const bc1 = 1 - Math.pow(b1, t);
			const bc2 = 1 - Math.pow(b2, t);
			const stepSize = lr / bc1;
			const bc2sqrt = Math.sqrt(bc2);
			for (let i = 0; i < n; i++) {
				const denom = Math.sqrt(v[i]) / bc2sqrt + eps;
				w[i] = w[i] + (-stepSize * m[i]) / denom;
			}
		}
	}
	set(o: Settings): void {
		if (o.lr !== undefined) this.lr = o.lr;
		if (o.betas !== undefined) this.betas = o.betas;
		if (o.weightDecay !== undefined) this.weightDecay = o.weightDecay;
	}
	state(): Record<string, { m?: Tensor; v?: Tensor }> {
		const out: Record<string, { m?: Tensor; v?: Tensor }> = {};
		for (const [name, p] of this.params) {
			const m = this.m.get(name);
			const v = this.v.get(name);
			if (m && v) out[name] = { m: { shape: p.shape.slice(), data: m }, v: { shape: p.shape.slice(), data: v } };
		}
		return out;
	}
}

/** AdamW: Adam with decoupled weight decay (default 0.01, as PyTorch). */
export class AdamW extends Adam {
	constructor(params: ParamsIn, o: { lr?: number; betas?: [number, number]; eps?: number; weightDecay?: number } = {}) {
		super(params, { ...o, decoupled: true });
	}
}

export interface OptimizerOpts {
	lr: number;
	momentum?: number;
	betas?: [number, number];
	eps?: number;
	weightDecay?: number;
	nesterov?: boolean;
}

/** The optimizer a TrainOpt names ('sgd' | 'momentum' | 'adam' | 'adamw'). */
export function makeOptimizer(kind: OptimizerKind, params: ParamsIn, o: OptimizerOpts): SGD | Momentum | Adam {
	switch (kind) {
		case 'sgd':
			return new SGD(params, { lr: o.lr, weightDecay: o.weightDecay });
		case 'momentum':
			return new Momentum(params, { lr: o.lr, momentum: o.momentum, nesterov: o.nesterov, weightDecay: o.weightDecay });
		case 'adam':
			return new Adam(params, { lr: o.lr, betas: o.betas, eps: o.eps, weightDecay: o.weightDecay });
		case 'adamw':
			return new AdamW(params, { lr: o.lr, betas: o.betas, eps: o.eps, weightDecay: o.weightDecay });
		default:
			throw new Error(`unknown optimizer '${String(kind)}'`);
	}
}
