// egnn: what the lesson, the 'mol' view and the 'spin' process read from a trace (pure TS, memoized per input tensor).
//   mol(rt)          the traced molecule: n, positions (float32 as the model read them), kinds, the network's energy
//                    and forces, and the toy formula's energy and forces on the same positions (mol.ts energy)
//   focus(rt)        the atom the lesson follows: the largest formula force on the learner's atoms before any turn or
//                    mirror (so turning cannot change which atom it is)
//   pushes(rt, i)    layer 1's push on atom i, one term per other atom j: (x_i − x_j) w_ij / (n − 1), and their sum
//   run(model, …)    the same network (the parity-checked executor) on other positions: energy and forces
//   variant(rt, how) the molecule turned by a number of degrees, mirrored or stretched (STRETCH times every distance),
//                    run again, with the largest differences
//                    from the turned (or mirrored) original forces
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { angleOf, centre, energy, mirror, molOf, turn, turnVec } from './mol.ts';
import { push } from './ops/egnn.ts';

type RT = ExplainerRuntime;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const kindsOf = (oneHot: ArrayLike<number>, n: number) => Array.from({ length: n }, (_, i) => (oneHot[3 * i + 1] ? 1 : oneHot[3 * i + 2] ? 2 : 0));
export const norm = (f: ArrayLike<number>, i: number) => Math.hypot(f[2 * i], f[2 * i + 1]);

export type Mol = { n: number; pos: Float32Array; kind: number[]; E: number; f: ArrayLike<number>; trueE: number; trueF: Float64Array };
const memoM = new WeakMap<Tensor, Mol>();
export function mol(rt: RT): Mol | null {
	const p = T(rt, 'pos'), k = T(rt, 'kind'), e = T(rt, 'energy'), f = T(rt, 'force') ?? T(rt, 'L1.push');
	if (!p || !k || !e || !f) return null;
	if (!memoM.has(p)) {
		const n = p.shape[0], kind = kindsOf(k.data, n), t = energy(p.data, kind);
		memoM.set(p, { n, pos: p.data as Float32Array, kind, E: e.data[0], f: f.data, trueE: t.E, trueF: t.F });
	}
	return memoM.get(p)!;
}

/** the atom with the largest formula force, on the learner's atoms before any turn or mirror */
export function focus(rt: RT): number {
	const m = molOf({ ...(rt.input as object), turn: 0, flip: false });
	if (!m.n) return 0;
	const F = energy(Float32Array.from(m.pos), m.kind).F;
	let best = 0;
	for (let i = 1; i < m.n; i++) if (norm(F, i) > norm(F, best) + 1e-12) best = i;
	return best;
}

/** layer 1's push on atom i: the term of every other atom j (x and y), its weight w_ij / (n − 1), and the sum */
export function pushes(rt: RT, i: number) {
	const p = T(rt, 'pos'), w = T(rt, 'L1.cw');
	if (!p || !w) return null;
	const n = p.shape[0];
	const ax = push(p.data, w.data, n, 2, i, 0), ay = push(p.data, w.data, n, 2, i, 1);
	return { n, terms: ax.terms.map((t, k) => ({ j: t.j, w: t.w, x: t.p, y: ay.terms[k].p })), x: ax.s, y: ay.s };
}

export type Out = { E: number; f: ArrayLike<number>; pos: Float32Array };
export function run(model: GraphModel, pos: ArrayLike<number>, kind: ArrayLike<number>): Out {
	const n = pos.length / 2, P = Float32Array.from(pos);
	const tr = model.forward({ pos: { shape: [n, 2], data: P }, kind: { shape: [n, 3], data: Float32Array.from(kind) } }, { record: 'layers' });
	return { E: tr.t.energy.data[0], f: (tr.t.force ?? tr.t['L1.push']).data, pos: P };
}

export type Variant = Out & { dE: number; df: number; at: number; ang0: number; ang: number };
const memoV = new WeakMap<Tensor, Map<string, Variant>>();
/** the molecule turned by `deg` degrees (or mirrored, deg = 'mirror'), run again; dE = E' − E and df = the largest
 *  |f' − (f turned or mirrored)| over every coordinate; ang0 / ang: the direction of the force on atom `at` before and after */
export function variant(rt: RT, deg: number | 'mirror' | 'stretch', at = focus(rt)): Variant | null {
	const m = mol(rt), p = T(rt, 'pos'), k = T(rt, 'kind');
	if (!m || !p || !k || !rt.model) return null;
	const key = `${deg}:${at}`;
	const cache = memoV.get(p) ?? new Map<string, Variant>();
	memoV.set(p, cache);
	const hit = cache.get(key);
	if (hit) return hit;
	const o = run(rt.model, deg === 'mirror' ? mirror(m.pos) : deg === 'stretch' ? stretch(m.pos) : turn(m.pos, deg), k.data);
	let df = 0;
	for (let i = 0; i < m.n; i++) {
		const [x, y] = deg === 'mirror' ? [-m.f[2 * i], m.f[2 * i + 1]] : deg === 'stretch' ? [m.f[2 * i], m.f[2 * i + 1]] : turnVec(m.f[2 * i], m.f[2 * i + 1], deg);
		df = Math.max(df, Math.abs(o.f[2 * i] - x), Math.abs(o.f[2 * i + 1] - y));
	}
	const v = { ...o, dE: o.E - m.E, df, at, ang0: angleOf(m.f[2 * at], m.f[2 * at + 1]), ang: angleOf(o.f[2 * at], o.f[2 * at + 1]) };
	cache.set(key, v);
	return v;
}

/** an angle difference folded into (−180, 180] */
export const wrap = (a: number) => 180 - ((((180 - a) % 360) + 360) % 360);

/** every distance times STRETCH: positions moved away from their centre */
export const STRETCH = 1.2;
export function stretch(pos: ArrayLike<number>): Float64Array {
	const { cx, cy } = centre(pos);
	return Float64Array.from(pos, (v, k) => (k % 2 ? cy + STRETCH * (v - cy) : cx + STRETCH * (v - cx)));
}
