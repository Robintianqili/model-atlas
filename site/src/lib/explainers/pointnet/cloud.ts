// pointnet: what the lesson and the 'cloud' view read from a trace (pure TS, memoized per trace or input tensor).
//   crit(rt)       the winners of the max: per feature the first point with the largest value (−1 when the feature is
//                  0 at every point), how many features each point wins, the critical points, the dead features
//   variants(rt)   the same network (rt.model, the parity-checked executor) on the shuffled table, on the critical rows
//                  only and with one stray point: probabilities, top class and the pooled row of each
//   cutRaw(raw, …)  the task button: the input with its critical points erased
import type { ExplainerRuntime, I18n, Tensor } from '../../atlas/types/index.ts';
import { winner } from './ops/max_points.ts';
import { CLASSES, LABELS, cloudOf, shuffled, withStray } from './shapes.ts';
import type { CloudRaw } from './adapters/points2d/adapter.ts';

type RT = ExplainerRuntime;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const argmax = (a: ArrayLike<number>) => {
	let b = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
	return b;
};
export const label = (i: number): I18n => LABELS[CLASSES[i]] ?? { en: '…', zh: '…' };

export function critical(h: Tensor) {
	const [n, F] = h.shape;
	const win: number[] = [], max: number[] = [];
	const wins = new Array<number>(n).fill(0);
	for (let f = 0; f < F; f++) {
		const w = winner(h.data, n, F, f);
		max.push(w.max);
		win.push(w.max > 0 ? w.at : -1);
		if (w.max > 0) wins[w.at]++;
	}
	const pts = [...new Set(win.filter((i) => i >= 0))].sort((a, b) => a - b);
	return { n, F, win, max, wins, pts, dead: win.filter((i) => i < 0).length, strongest: argmax(wins) };
}
const memoC = new WeakMap<Tensor, ReturnType<typeof critical>>();
export function crit(rt: RT) {
	const h = T(rt, 'relu2');
	if (!h) return null;
	if (!memoC.has(h)) memoC.set(h, critical(h));
	return memoC.get(h)!;
}

export type Run = { p: ArrayLike<number>; top: number; pool: ArrayLike<number>; xy: Float32Array };
function run(rt: RT, xy: Float32Array): Run | null {
	const n = xy.length / 2;
	if (!rt.model || !n) return null;
	const tr = rt.model.forward({ input: { shape: [n, 2], data: xy } }, { record: 'layers' });
	return { p: tr.t.softmax.data, top: argmax(tr.t.softmax.data), pool: tr.t.pool.data, xy };
}
const rows = (x: Tensor, idx: number[]) => Float32Array.from(idx.flatMap((i) => [x.data[2 * i], x.data[2 * i + 1]]));
const memoV = new WeakMap<Tensor, { order: number[]; shuf: Run | null; only: Run | null; stray: Run | null; scale: number }>();
/** the network on the shuffled table, on the critical rows only, and with one stray point (memoized per input) */
export function variants(rt: RT) {
	const x = T(rt, 'input');
	const c = crit(rt);
	if (!x || !c) return null;
	if (!memoV.has(x)) {
		const n = x.shape[0];
		const order = shuffled(n);
		const s = withStray(x.data);
		memoV.set(x, { order, shuf: run(rt, rows(x, order)), only: run(rt, rows(x, c.pts.length ? c.pts : [0])), stray: run(rt, s.xy), scale: s.scale });
	}
	return memoV.get(x)!;
}
/** the largest |a − b| over two equally long rows */
export const maxDiff = (a: ArrayLike<number> | undefined, b: ArrayLike<number> | undefined) => {
	if (!a || !b || a.length !== b.length) return NaN;
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
};

/** the task button: erase the critical points (indices into the cloud); cut counts the presses and keeps the answer
 *  before the first press (cls) and the points before it (back, for "Put them back") */
export function cutRaw(raw: CloudRaw, critical: number[], top: number): CloudRaw {
	const pts = cloudOf(raw), gone = new Set(critical);
	return { ...raw, points: pts.filter((_, i) => !gone.has(i)), cut: { n: (raw.cut?.n ?? 0) + 1, cls: raw.cut?.cls ?? top, back: raw.cut?.back ?? pts } };
}
