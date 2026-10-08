// Autograd core (design §4.5, W1-B1): Var, grad modes and the reverse-mode engine.
// Tensor-level reverse mode in float64. Every backward rule (ops.ts) is written with the same ops, so
// grad(..., { createGraph: true }) returns Vars that are themselves differentiable (PINN u'' and its parameter grads).
// When createGraph is false the engine runs the backward rules with grad mode off: no graph is built and the
// rules take their fast raw-array paths.
import type { Tensor, Var as VarContract } from '../types/index.ts';

/** A float64 tensor. All autograd values use it (CONTRACTS row-major layout). */
export interface T64 {
	shape: number[];
	data: Float64Array;
}

/** How a Var was made: its parents and the vector-Jacobian product. `need[i]` says whether parent i needs a gradient. */
export interface GradFn {
	op: string;
	parents: Var[];
	backward(g: Var, need: boolean[]): (Var | null)[];
}

let enabled = true;

/** True while operations record a graph (the default). */
export function isGradEnabled(): boolean {
	return enabled;
}

/** Run f with grad recording switched on or off, restoring the previous mode afterwards. */
export function withGradMode<T>(on: boolean, f: () => T): T {
	const prev = enabled;
	enabled = on;
	try {
		return f();
	} finally {
		enabled = prev;
	}
}

/** Run f without recording a graph (evaluation, metrics, derived fields, optimizer updates). */
export function noGrad<T>(f: () => T): T {
	return withGradMode(false, f);
}

export function sizeOf(shape: readonly number[]): number {
	let n = 1;
	for (const d of shape) n *= d;
	return n;
}

export class Var implements VarContract {
	readonly value: T64;
	grad: Var | null = null;
	readonly requiresGrad: boolean;
	readonly name?: string;
	/** null for leaves (parameters, inputs, constants) */
	readonly fn: GradFn | null;

	constructor(value: T64, requiresGrad: boolean, fn: GradFn | null, name?: string) {
		this.value = value;
		this.requiresGrad = requiresGrad;
		this.fn = fn;
		if (name !== undefined) this.name = name;
	}
	get shape(): number[] {
		return this.value.shape;
	}
	get data(): Float64Array {
		return this.value.data;
	}
	get size(): number {
		return this.value.data.length;
	}
	/** a leaf: no recorded parents */
	get isLeaf(): boolean {
		return this.fn === null;
	}
	/** The single value of a one-element Var. */
	item(): number {
		if (this.value.data.length !== 1) throw new Error(`item(): Var has ${this.value.data.length} elements, shape [${this.value.shape}]`);
		return this.value.data[0];
	}
	/** A copy of the value as a CONTRACTS Tensor (Float64Array). Safe to keep: later updates do not change it. */
	toTensor(): Tensor {
		return { shape: this.value.shape.slice(), data: this.value.data.slice() };
	}
	/** A new leaf with a copy of the value and no graph. */
	detach(): Var {
		return new Var({ shape: this.value.shape.slice(), data: this.value.data.slice() }, false, null);
	}
}

/** Build an op result. It records the graph only when grad mode is on and some parent requires a gradient. */
export function result(op: string, data: Float64Array, shape: number[], parents: Var[], backward: GradFn['backward']): Var {
	if (enabled) {
		for (const p of parents) if (p.requiresGrad) return new Var({ shape, data }, true, { op, parents, backward });
	}
	return new Var({ shape, data }, false, null);
}

/** A constant (no gradient) that takes ownership of `data`. */
export function raw(data: Float64Array, shape: number[]): Var {
	return new Var({ shape, data }, false, null);
}

type TensorLike = Tensor | { shape: number[]; data: ArrayLike<number> };

function toF64(t: TensorLike): T64 {
	const n = sizeOf(t.shape);
	if (t.data.length !== n) throw new Error(`tensor data has ${t.data.length} values but shape [${t.shape}] needs ${n}`);
	return { shape: t.shape.slice(), data: Float64Array.from(t.data as ArrayLike<number>) };
}

/** A trainable leaf. `name` is the PyTorch state_dict name ('net.0.weight'). The data is copied to float64. */
export function param(name: string, t: TensorLike): Var {
	return new Var(toF64(t), true, null, name);
}

/** A data leaf (copied to float64). With requiresGrad it can be differentiated against (PINN: u'(t)). */
export function input(t: TensorLike, requiresGrad = false): Var {
	return new Var(toF64(t), requiresGrad, null);
}

/** Shorthand: a constant from a shape and values. */
export function tensor(shape: number[], values: ArrayLike<number>): Var {
	return new Var(toF64({ shape, data: values }), false, null);
}

export function scalar(v: number): Var {
	return new Var({ shape: [], data: Float64Array.of(v) }, false, null);
}

export function full(shape: number[], v: number): Var {
	return new Var({ shape: shape.slice(), data: new Float64Array(sizeOf(shape)).fill(v) }, false, null);
}

export function zeros(shape: number[]): Var {
	return new Var({ shape: shape.slice(), data: new Float64Array(sizeOf(shape)) }, false, null);
}

export function ones(shape: number[]): Var {
	return full(shape, 1);
}

// ---------------------------------------------------------------- engine

export function sameShape(a: readonly number[], b: readonly number[]): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

/** Parents before children, over the part of the graph that requires gradients. Iterative (deep unrolls are fine). */
function topoOrder(roots: Var[]): Var[] {
	const order: Var[] = [];
	const seen = new Set<Var>();
	const stack: { v: Var; i: number }[] = [];
	for (const r of roots) {
		if (!r.requiresGrad || seen.has(r)) continue;
		seen.add(r);
		stack.push({ v: r, i: 0 });
		while (stack.length) {
			const top = stack[stack.length - 1];
			const ps = top.v.fn ? top.v.fn.parents : [];
			if (top.i < ps.length) {
				const p = ps[top.i++];
				if (p.requiresGrad && !seen.has(p)) {
					seen.add(p);
					stack.push({ v: p, i: 0 });
				}
			} else {
				stack.pop();
				order.push(top.v);
			}
		}
	}
	return order;
}

function addRaw(a: Var, b: Var): Var {
	if (!sameShape(a.shape, b.shape)) throw new Error(`gradient shape mismatch [${a.shape}] vs [${b.shape}]`);
	const x = a.data;
	const y = b.data;
	const out = new Float64Array(x.length);
	for (let i = 0; i < x.length; i++) out[i] = x[i] + y[i];
	return raw(out, a.shape.slice());
}

/** Set by ops.ts: the graph-recording add, used to accumulate gradients when createGraph is on. */
let addOp: ((a: Var, b: Var) => Var) | null = null;
export function _registerAdd(f: (a: Var, b: Var) => Var): void {
	addOp = f;
}

type Seed = Var | Tensor | number | null | undefined;

function seedFor(y: Var, s: Seed): Var {
	if (s === null || s === undefined) return full(y.shape, 1);
	if (typeof s === 'number') return full(y.shape, s);
	const v = s instanceof Var ? s : input(s);
	if (!sameShape(v.shape, y.shape)) throw new Error(`grad seed shape [${v.shape}] does not match output shape [${y.shape}]`);
	return v;
}

/**
 * Reverse-mode propagation from ys (with seeds) to `targets`, or to every reachable leaf when targets is null.
 * Runs with grad mode = createGraph, so with createGraph the returned gradients carry a graph.
 */
function propagate(ys: Var[], seeds: Seed[], targets: Var[] | null, createGraph: boolean): Map<Var, Var> {
	return withGradMode(createGraph, () => {
		const order = topoOrder(ys);
		const tset = targets ? new Set(targets) : null;
		// `leads`: Vars on a path to a target. Only their gradients are computed.
		const leads = new Set<Var>();
		for (const v of order) {
			if (tset ? tset.has(v) : v.fn === null) {
				leads.add(v);
				continue;
			}
			if (v.fn) {
				for (const p of v.fn.parents) {
					if (leads.has(p)) {
						leads.add(v);
						break;
					}
				}
			}
		}
		const grads = new Map<Var, Var>();
		const accumulate = (v: Var, g: Var) => {
			const prev = grads.get(v);
			if (!prev) grads.set(v, g);
			else grads.set(v, createGraph && addOp ? addOp(prev, g) : addRaw(prev, g));
		};
		ys.forEach((y, i) => {
			if (y.requiresGrad) accumulate(y, seedFor(y, seeds[i]));
		});
		for (let k = order.length - 1; k >= 0; k--) {
			const v = order[k];
			const fn = v.fn;
			const g = grads.get(v);
			if (!fn || !g) continue;
			const need = fn.parents.map((p) => leads.has(p));
			if (need.some(Boolean)) {
				const pg = fn.backward(g, need);
				for (let i = 0; i < fn.parents.length; i++) {
					const gi = pg[i];
					if (!need[i] || !gi) continue;
					if (!sameShape(gi.shape, fn.parents[i].shape)) {
						throw new Error(`${fn.op}: gradient for input ${i} has shape [${gi.shape}], expected [${fn.parents[i].shape}]`);
					}
					accumulate(fn.parents[i], gi);
				}
			}
			if (!tset || !tset.has(v)) grads.delete(v);
		}
		return grads;
	});
}

export interface GradOpts {
	/** Record the backward pass so the returned gradients can be differentiated again (default false). */
	createGraph?: boolean;
	/** Upstream gradients, one per y (default: ones, i.e. the gradient of sum(y)). */
	seeds?: Seed | Seed[];
	/** Return zeros for an x that no y depends on, instead of throwing (default false). */
	allowUnused?: boolean;
}

/**
 * Gradients of ys (seeded, default ones) with respect to xs. Returns a Var for a single x and an array for an array.
 * Like torch.autograd.grad: nothing is accumulated into .grad.
 */
export function grad(ys: Var | Var[], xs: Var, o?: GradOpts): Var;
export function grad(ys: Var | Var[], xs: Var[], o?: GradOpts): Var[];
export function grad(ys: Var | Var[], xs: Var | Var[], o: GradOpts = {}): Var | Var[] {
	const yl = Array.isArray(ys) ? ys : [ys];
	const xl = Array.isArray(xs) ? xs : [xs];
	const seeds: Seed[] = Array.isArray(o.seeds) ? o.seeds : yl.map(() => o.seeds as Seed);
	for (const x of xl) if (!x.requiresGrad) throw new Error('grad(): an x does not require grad (use param() or input(t, true))');
	if (!yl.some((y) => y.requiresGrad) && !o.allowUnused) throw new Error('grad(): no y depends on anything that requires grad');
	const g = propagate(yl, seeds, xl, !!o.createGraph);
	const out = xl.map((x, i) => {
		const gx = g.get(x);
		if (gx) return gx;
		if (!o.allowUnused) throw new Error(`grad(): x #${i}${x.name ? ` (${x.name})` : ''} is not used to compute y (pass allowUnused)`);
		return zeros(x.shape);
	});
	return Array.isArray(xs) ? out : out[0];
}

/** Accumulate d(loss)/d(leaf) into every reachable leaf's .grad (like loss.backward()). */
export function backward(loss: Var, o: { createGraph?: boolean; seed?: Var | Tensor | number } = {}): void {
	if (!loss.requiresGrad) throw new Error('backward(): the loss does not depend on anything that requires grad');
	const g = propagate([loss], [o.seed], null, !!o.createGraph);
	withGradMode(!!o.createGraph, () => {
		for (const [v, gv] of g) {
			if (v.fn !== null) continue;
			v.grad = v.grad ? (o.createGraph && addOp ? addOp(v.grad, gv) : addRaw(v.grad, gv)) : gv;
		}
	});
}

export type ParamSource = Var[] | Record<string, Var> | { params(): Var[] };

export function paramList(ps: ParamSource): Var[] {
	if (Array.isArray(ps)) return ps;
	if (typeof (ps as { params?: unknown }).params === 'function') return (ps as { params(): Var[] }).params();
	return Object.values(ps as Record<string, Var>);
}

/** Clear .grad on every parameter. */
export function zeroGrad(ps: ParamSource): void {
	for (const p of paramList(ps)) p.grad = null;
}
