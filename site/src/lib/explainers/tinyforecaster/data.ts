// tinyforecaster: the series every source learns from, and the plain-number helpers they share (no autograd here).
// TS is the source of truth for the data: the Python twin and the reference read the numbers the TS side wrote.
//   L       the window: the model reads the last L = 24 values
//   hold    the last min(24, ⌊n/4⌋) values are kept aside: no training window reaches them; forecasts are checked there
//   split   n − hold, the first held-out position; the training windows end before it
// A series too short for MIN_WINDOWS training windows (the edge fixtures: one value) is replaced by the sine preset, and
// `fallback` says so (the first lesson step tells the learner).
import { sine } from '../../atlas/data/series.ts';
import { seriesOf as adapterSeries } from '../../atlas/adapters/series/adapter.ts';

export const L = 24;
export const MAX_HOLD = 24;
export const MIN_WINDOWS = 4;
/** The random walks of the limits step: the series adapter's random-walk preset with these seeds (seed 1 is the preset). */
export const WALK_SEEDS = [1, 2, 3];

export interface Data {
	/** the series the sources use */
	y: Float64Array;
	n: number;
	hold: number;
	split: number;
	/** training windows: positions 0..split−L−1 start one, its target is the value right after it */
	windows: number;
	/** the learner's series was too short, so the sine preset is used */
	fallback: boolean;
	/** the learner's own length */
	given: number;
}

export const holdOf = (n: number) => Math.min(MAX_HOLD, Math.floor(n / 4));

function build(y: Float64Array, given: number, fallback: boolean): Data {
	const n = y.length;
	const hold = holdOf(n);
	return { y, n, hold, split: n - hold, windows: n - hold - L, fallback, given };
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
	if (d.windows < MIN_WINDOWS) d = build(Float64Array.from(sine(96).y), y.length, true);
	if (raw && typeof raw === 'object') cache.set(raw, d);
	return d;
}

/** Window k: the L values from position k (oldest first). */
export const windowAt = (d: Data, k: number) => d.y.slice(k, k + L);

/** The seasonal-naive baseline: "repeat the value from m steps ago". m is the lag from 1 to L whose one-step forecast
 *  was best on the training part (mean |y_t − y_{t−m}| over the same t = L..split−1 for every m; ties to the smaller m).
 *  Its forecast of the held-out part from the split: value split + k repeats position split + k − m·(⌊k/m⌋ + 1), always
 *  a value from before the split. */
export function seasonalNaive(d: Data): { m: number; fit: number; path: Float64Array; mae: number } {
	let m = 1;
	let fit = Infinity;
	for (let lag = 1; lag <= L; lag++) {
		let s = 0;
		for (let t = L; t < d.split; t++) s += Math.abs(d.y[t] - d.y[t - lag]);
		s /= d.split - L;
		if (s < fit - 1e-12) {
			fit = s;
			m = lag;
		}
	}
	const path = new Float64Array(d.hold);
	let e = 0;
	for (let k = 0; k < d.hold; k++) {
		path[k] = d.y[d.split + k - m * (Math.floor(k / m) + 1)];
		e += Math.abs(path[k] - d.y[d.split + k]);
	}
	return { m, fit, path, mae: e / d.hold };
}
