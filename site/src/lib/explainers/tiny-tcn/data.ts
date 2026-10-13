// tiny-tcn: the series every source learns from and the plain-number helpers they share (no autograd here). TS is the
// source of truth for the data: the PyTorch twin reads the numbers the TS side wrote.
//   c_t     the change at time t, y_t − y_{t−1} (c_0 = 0: nothing comes before the first value). The TCN reads changes.
//   hold    the last min(24, ⌊n/4⌋) values are kept aside; forecasts are checked there
//   split   n − hold, the first held-out time; the network and ARIMA train on times 0..split−1 only
//   RF      16: the output at time t reads the changes c_{t−15}..c_t (1 + 1 + 2 + 4 + 8 with kernel 2, dilations 1/2/4/8)
//   train positions   t = RF..split−2: every output whose 16 changes are real (t − 15 ≥ 1) and whose target c_{t+1} is in
//           the training part. The output at t predicts c_{t+1}, so the forecast of y_{t+1} is y_t + that change.
// A series too short for MIN_POS training positions (the edge fixtures: one value) is replaced by the sine preset, and
// `fallback` says so (the first lesson step tells the learner).
import { sine, square } from '../../atlas/data/series.ts';
import { seriesOf as adapterSeries } from '../../atlas/adapters/series/adapter.ts';

export const KERNEL = 2;
export const DILATIONS = [1, 2, 4, 8];
/** receptive field in changes: 1 + (k − 1)·Σd */
export const RF = 1 + (KERNEL - 1) * DILATIONS.reduce((s, d) => s + d, 0);
export const MAX_HOLD = 24;
export const MIN_POS = 16;

export interface Data {
	/** the series the sources use */
	y: Float64Array;
	/** its changes, c[0] = 0 */
	c: Float64Array;
	n: number;
	hold: number;
	split: number;
	/** training positions t = RF..split−2 */
	count: number;
	/** the learner's series was too short, so the sine preset is used */
	fallback: boolean;
	/** the learner's own length */
	given: number;
}

export const holdOf = (n: number) => Math.min(MAX_HOLD, Math.floor(n / 4));
export const changes = (y: ArrayLike<number>) => Float64Array.from(y, (v, t) => (t ? v - y[t - 1] : 0));

function build(y: Float64Array, given: number, fallback: boolean): Data {
	const n = y.length;
	const hold = holdOf(n);
	return { y, c: changes(y), n, hold, split: n - hold, count: n - hold - 1 - RF, fallback, given };
}

const cache = new WeakMap<object, Data>();
/** The data of a raw input: the series adapter's raw value ({ preset } or drawn { y }) or a plain list of numbers. */
export function dataOf(raw: unknown): Data {
	if (raw && typeof raw === 'object') {
		const hit = cache.get(raw);
		if (hit) return hit;
	}
	const y = Array.isArray(raw) ? Float64Array.from(raw as number[], (v) => (Number.isFinite(v) ? v : 0)) : adapterSeries(raw).y;
	let d = build(y, y.length, false);
	if (d.count < MIN_POS) d = build(Float64Array.from(sine(96).y), y.length, true);
	if (raw && typeof raw === 'object') cache.set(raw, d);
	return d;
}

/** The limits step's series: a square wave that stays 20 steps at each level (period 40), longer than the reach. */
export const LONG_PERIOD = 40;
export const longSquare = () => Array.from(square(120, { period: LONG_PERIOD }).y);
