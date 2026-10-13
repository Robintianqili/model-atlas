// tiny-tcn: what the lesson and the tree view both read, on plain numbers (pure TS; no Svelte).
//   the followed output   time t* = split − 2: its output predicts c_{split−1}, so its forecast is of the last training
//                         value y_{split−1}, whose truth is known (steps 3 to 8 follow it)
//   the tree              the nodes the output at t* depends on: in layer l, the times t* − (sums of the dilations of
//                         the layers above l); the changes c_{t*−15..t*} are its 16 leaves
//   the long square       the limits step: times whose last 16 changes are all 0 give the same forecast
import type { At, TrainState } from '../../atlas/types/index.ts';
import { DILATIONS, RF, type Data, dataOf, longSquare } from './data.ts';
import { CH, forwardPlain, nodeTerms, type Params, type Plain } from './model.ts';

export const tStar = (d: Data) => d.split - 2;
export const paramsAt = (at: At) => (at.state as Partial<TrainState> | undefined)?.params;

const memo = new WeakMap<object, Plain>();
/** The network on the training part's changes (memoized per parameter set: a run belongs to one input). */
export function plainOf(P: Params, d: Data): Plain {
	let p = memo.get(P);
	if (!p) memo.set(P, (p = forwardPlain(P, d.c.subarray(0, d.split))));
	return p;
}

/** Offsets back from t of the tree's nodes in layer l (0..3; l = −1: the changes it reads). */
export function treeOffsets(l: number): number[] {
	let s = [0];
	for (let k = DILATIONS.length - 1; k > l; k--) s = s.flatMap((o) => [o, o + DILATIONS[k]]);
	return s.sort((a, b) => a - b);
}

/** The feature (channel) the narration follows at node (l, t): the one with the largest conv sum. */
export function featureOf(pl: Plain, l: number, t: number): number {
	let best = 0;
	for (let o = 1; o < CH; o++) if (pl.pre[l][o * pl.T + t] > pl.pre[l][best * pl.T + t]) best = o;
	return best;
}

/** Everything of node (l, t*, feature) on the step's weights, or null before the source ran. */
export function nodeAt(at: At, raw: unknown, l: number, feature?: number) {
	const P = paramsAt(at);
	if (!P) return null;
	const d = dataOf(raw);
	const t = tStar(d);
	const pl = plainOf(P, d);
	const ch = feature ?? featureOf(pl, l, t);
	return { d, t, P, pl, ch, ...nodeTerms(P, pl, d.c.subarray(0, d.split), l, t, ch) };
}

/** The head at t*: the 8 products, the bias, the predicted change and the forecast y_t + change. */
export function headAt(P: Params, d: Data) {
	const pl = plainOf(P, d);
	const t = tStar(d);
	const w = P['head.weight'].data;
	const terms = Array.from({ length: CH }, (_, i) => ({ ch: i, x: pl.h[3][i * pl.T + t], w: w[i] }));
	const change = pl.out[t];
	return { t, terms, sum: change - P['head.bias'].data[0], bias: P['head.bias'].data[0], change, last: d.y[t], forecast: d.y[t] + change, truth: d.y[t + 1] };
}

/** The forecast at t* after adding `by` to the change at time t* − back (back 15: the oldest leaf; 16: outside). */
export function nudged(P: Params, d: Data, back: number, by = 1): number {
	const c = d.c.slice(0, d.split);
	c[tStar(d) - back] += by;
	return forwardPlain(P, c).out[tStar(d)] + d.y[tStar(d)];
}

/** The limits step on the long square: the last stretch of times whose 16 changes are all 0, the forecasts there
 *  (all equal), and the truth after its end, where the wave jumps. */
export function longStretch(P: Params) {
	const d = dataOf(longSquare());
	const pl = plainOf(P, d);
	const zero = (t: number) => d.c.subarray(t - RF + 1, t + 1).every((v) => v === 0);
	let b = d.split - 2;
	while (b >= RF && !zero(b)) b--;
	let a = b;
	while (a - 1 >= RF && zero(a - 1)) a--;
	const f = (t: number) => d.y[t] + pl.out[t];
	// the jumps it trained on that follow 16 zero changes: training positions t (RF..split−2) with zero(t) whose target
	// change c_{t+1} is not 0 (the first one, at time 20, counts too: its 16 changes c_4..c_19 are real zeros)
	let jumps = 0;
	for (let t = RF; t <= d.split - 2; t++) if (zero(t) && d.c[t + 1] !== 0) jumps++;
	return { d, a, b, f: f(b), same: Array.from({ length: b - a + 1 }, (_, i) => f(a + i)).every((v) => v === f(b)), stay: d.y[b], jump: d.y[b + 1], miss: Math.abs(f(b) - d.y[b + 1]), jumps, forecasts: Float64Array.from({ length: d.split - 1 - RF }, (_, i) => f(RF + i)) };
}
