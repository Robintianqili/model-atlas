// gradcheck / gradgradcheck (design §4.5, W1-B1): analytic gradients against central finite differences, in float64.
// The output is projected on a fixed pseudo-random vector r, s(x) = Σ f(x)·r, so one scalar covers the whole Jacobian.
// second: also checks the gradient of h(x) = Σ_i ⟨∂s/∂x_i, v_i⟩ (built with createGraph) against finite differences of h.
// Error measure: |analytic − numeric| / max(1, |numeric|) (the project's "relative").
import type { Tensor } from '../types/index.ts';
import { Var, grad, input, noGrad, raw } from './var.ts';
import { add, mul, sum } from './ops.ts';

export interface GradcheckOpts {
	eps?: number;
	tol?: number;
	second?: boolean;
	seed?: number;
	/** check at most this many entries per input (spread over the tensor); default all */
	maxPerInput?: number;
}

export interface GradcheckWorst {
	order: 1 | 2;
	input: number;
	index: number;
	analytic: number;
	numeric: number;
}

export interface GradcheckResult {
	maxRelErr: number;
	worst: GradcheckWorst | null;
	checked: number;
	pass: boolean;
}

/** A small deterministic stream for projection vectors (not for model randomness: that is core/rng.ts). */
function stream(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export function gradcheck(f: (...xs: Var[]) => Var, xs: (Var | Tensor)[], o: GradcheckOpts = {}): GradcheckResult {
	const eps = o.eps ?? 1e-6;
	const tol = o.tol ?? 1e-6;
	const next = stream(o.seed ?? 1234);
	const shapes = xs.map((x) => x.shape.slice());
	const base = xs.map((x) => Float64Array.from(x.data));
	const leaves = (vals: Float64Array[]) => vals.map((d, i) => input({ shape: shapes[i], data: d }, true));

	let r: Float64Array | null = null;
	const project = (y: Var): Var => {
		if (!r) {
			r = new Float64Array(y.size);
			for (let i = 0; i < r.length; i++) r[i] = 2 * next() - 1;
		}
		return sum(mul(y, raw(r.slice(), y.shape.slice())));
	};
	noGrad(() => project(f(...leaves(base)))); // fixes r

	const vs: Float64Array[] = base.map((b) => {
		const v = new Float64Array(b.length);
		for (let i = 0; i < v.length; i++) v[i] = 2 * next() - 1;
		return v;
	});

	/** s(x) */
	const s = (vals: Float64Array[]) => noGrad(() => project(f(...leaves(vals))).item());
	/** h(x) = Σ_i ⟨∂s/∂x_i, v_i⟩, with first-order gradients */
	const h = (vals: Float64Array[]) => {
		const ls = leaves(vals);
		const gs = grad(project(f(...ls)), ls, { allowUnused: true });
		let acc = 0;
		gs.forEach((g, i) => {
			for (let k = 0; k < g.size; k++) acc += g.data[k] * vs[i][k];
		});
		return acc;
	};

	let maxRelErr = 0;
	let worst: GradcheckWorst | null = null;
	let checked = 0;
	const compare = (order: 1 | 2, analytic: Float64Array[], fn: (vals: Float64Array[]) => number) => {
		for (let i = 0; i < base.length; i++) {
			const n = base[i].length;
			const count = o.maxPerInput ? Math.min(n, o.maxPerInput) : n;
			for (let c = 0; c < count; c++) {
				const k = count === n ? c : Math.floor((c * n) / count);
				const plus = base.map((b) => b.slice());
				const minus = base.map((b) => b.slice());
				plus[i][k] += eps;
				minus[i][k] -= eps;
				const numeric = (fn(plus) - fn(minus)) / (2 * eps);
				const a = analytic[i][k];
				let err = Math.abs(a - numeric) / Math.max(1, Math.abs(numeric));
				if (Number.isNaN(err)) err = Infinity;
				checked++;
				if (worst === null || err > maxRelErr) {
					maxRelErr = Math.max(maxRelErr, err);
					worst = { order, input: i, index: k, analytic: a, numeric };
				}
			}
		}
	};

	const l1 = leaves(base);
	const g1 = grad(project(f(...l1)), l1, { allowUnused: true }).map((g) => g.data);
	compare(1, g1, s);

	if (o.second) {
		const l2 = leaves(base);
		const gs = grad(project(f(...l2)), l2, { createGraph: true, allowUnused: true });
		let hv: Var | null = null;
		gs.forEach((g, i) => {
			const term = sum(mul(g, raw(vs[i].slice(), g.shape.slice())));
			hv = hv ? add(hv, term) : term;
		});
		const g2 = grad(hv!, l2, { allowUnused: true }).map((g) => g.data);
		compare(2, g2, h);
	}
	return { maxRelErr, worst, checked, pass: maxRelErr <= tol };
}
