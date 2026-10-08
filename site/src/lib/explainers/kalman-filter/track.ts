// kalman-filter: the pieces every file shares (pure TS). The learner's input is a TrackRaw: the true track of a cart on a
// straight rail (a preset, or N positions the learner drew), the seed of the sensor noise and the filter's two settings
// (q and r, edited on the stage). dataOf() turns it into plain numbers, the input of every process (and, as JSON, of the
// Python reference): the true positions, the sensor readings (true + NOISE · a seeded normal draw), q, r and the
// window of the moving average. The filter itself (kfPredict, kfUpdate) is here too, so the process and the grid search
// of the best settings run the very same arithmetic; pipeline/kalman-filter/reference.py repeats it in Python.
import type { I18n } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export type TrackPreset = 'stopgo' | 'cruise' | 'wave' | 'turn';
export interface TrackRaw { preset?: TrackPreset; y?: number[]; seed?: number; q?: number; r?: number }
export interface TrackData { truth: number[]; z: number[]; q: number; r: number; w: number }

export const I = (en: string, zh: string): I18n => ({ en, zh });
/** seconds of track (one reading per second), the sensor's noise (standard deviation, metres), the plot's range */
export const N = 60;
export const NOISE = 1;
export const LO = -5;
export const HI = 35;
/** the slider positions of q and r (the grid the task searches), the starting settings, the moving average's window,
 *  the velocity doubt of the first guess, and the second the worked example looks at */
export const QS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
export const RS = [0.1, 0.25, 0.5, 1, 2, 4, 8];
export const Q0 = 0.01;
export const R0 = 1;
export const W = 5;
export const V0 = 1;
export const KDEMO = 12;
/** the limits step: the turn track (it turns back at second TURN_AT), filtered with the starting q and with a large one */
export const QBIG = 0.3;
export const TURN_AT = 30;

export const PRESETS: { id: TrackPreset; label: I18n }[] = [
	{ id: 'stopgo', label: I('Start, cruise, stop', '起步、匀速、停下') },
	{ id: 'cruise', label: I('Steady speed', '匀速') },
	{ id: 'wave', label: I('Back and forth', '来回摆动') },
	{ id: 'turn', label: I('Sudden turn', '突然掉头') }
];
export const START: TrackRaw = { preset: 'stopgo', seed: 1 };

const fin = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clamp = (v: number) => Math.min(HI, Math.max(LO, v));

/** the true positions of a preset: whole-second steps of p ← p + v + a/2, v ← v + a */
export function presetTruth(p: TrackPreset): number[] {
	const y: number[] = [];
	if (p === 'wave') for (let t = 0; t < N; t++) y.push(15 + 10 * Math.sin((2 * Math.PI * t) / 40));
	else if (p === 'turn') for (let t = 0; t < N; t++) y.push(t <= TURN_AT ? 2 + t : 2 + TURN_AT - (t - TURN_AT));
	else {
		let pos = 2, v = p === 'cruise' ? 0.4 : 0;
		for (let t = 0; t < N; t++) {
			y.push(pos);
			const a = p === 'cruise' ? 0 : t >= 5 && t < 15 ? 0.1 : t >= 30 && t < 40 ? -0.1 : 0;
			pos += v + a / 2;
			v += a;
		}
	}
	return y;
}

export function rawOf(raw: unknown): TrackRaw {
	return (raw && typeof raw === 'object' ? raw : {}) as TrackRaw;
}
/** the true track: the learner's drawing (cut or padded to N seconds, inside the plot's range), else the preset */
export function truthOf(raw: unknown): number[] {
	const r = rawOf(raw);
	if (Array.isArray(r.y)) {
		const y = r.y.slice(0, N).map((v) => clamp(fin(v) ? v : 0));
		while (y.length && y.length < N) y.push(y[y.length - 1]);
		return y;
	}
	return presetTruth(PRESETS.some((p) => p.id === r.preset) ? r.preset! : 'stopgo');
}
const pickOf = (v: unknown, xs: number[], d: number) => (fin(v) ? xs.reduce((b, x) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(b / v)) ? x : b), xs[0]) : d);
export const seedOf = (raw: unknown) => (fin(rawOf(raw).seed) ? Math.max(0, Math.floor(rawOf(raw).seed!)) : 1);
export const qOf = (raw: unknown) => pickOf(rawOf(raw).q, QS, Q0);
export const rOf = (raw: unknown) => pickOf(rawOf(raw).r, RS, R0);

/** plain numbers for a process: the readings are the true positions plus seeded noise; settings from the raw input
 *  unless `fixed` (the worked example always uses the starting settings) */
export function dataOf(raw: unknown, o: { fixed?: boolean } = {}): TrackData {
	const truth = truthOf(raw), rng = createRng(seedOf(raw));
	const z = truth.map((v) => v + NOISE * rng.normal());
	return { truth, z, q: o.fixed ? Q0 : qOf(raw), r: o.fixed ? R0 : rOf(raw), w: W };
}

/** the filter's belief: position p (m), velocity v (m/s), and its covariance [[P00, P01], [P01, P11]] */
export interface Belief { p: number; v: number; P00: number; P01: number; P11: number }
/** predict one second ahead with the constant-velocity model F = [[1, 1], [0, 1]]; an unknown push of variance q changes
 *  the velocity by a and the position by a/2, so Q = q · [[1/4, 1/2], [1/2, 1]] */
export function kfPredict(b: Belief, q: number): Belief {
	return { p: b.p + b.v, v: b.v, P00: b.P00 + 2 * b.P01 + b.P11 + q / 4, P01: b.P01 + b.P11 + q / 2, P11: b.P11 + q };
}
/** correct with a reading z of variance r (H = [1, 0]): gain K = P00/(P00 + r) for the position, Kv = P01/(P00 + r) */
export function kfUpdate(b: Belief, z: number, r: number): { b: Belief; y: number; S: number; K: number; Kv: number } {
	const y = z - b.p, S = b.P00 + r, K = b.P00 / S, Kv = b.P01 / S;
	return { b: { p: b.p + K * y, v: b.v + Kv * y, P00: (1 - K) * b.P00, P01: (1 - K) * b.P01, P11: b.P11 - Kv * b.P01 }, y, S, K, Kv };
}
export const kfStart = (z0: number, r: number): Belief => ({ p: z0, v: 0, P00: r, P01: 0, P11: V0 });
/** the whole run's estimates and standard deviations */
export function kfRun(z: number[], q: number, r: number): { est: number[]; sd: number[] } {
	if (!z.length) return { est: [], sd: [] };
	let b = kfStart(z[0], r);
	const est = [b.p], sd = [Math.sqrt(b.P00)];
	for (let k = 1; k < z.length; k++) {
		b = kfUpdate(kfPredict(b, q), z[k], r).b;
		est.push(b.p);
		sd.push(Math.sqrt(b.P00));
	}
	return { est, sd };
}
/** the trailing moving average of the last w readings (fewer at the start) */
export function movingAverage(z: number[], w: number): number[] {
	return z.map((_, k) => {
		let s = 0;
		const from = Math.max(0, k - w + 1);
		for (let j = from; j <= k; j++) s += z[j];
		return s / (k - from + 1);
	});
}
/** root mean square of a − b */
export function rmse(a: number[], b: number[]): number {
	let s = 0;
	for (let k = 0; k < a.length; k++) s += (a[k] - b[k]) * (a[k] - b[k]);
	return Math.sqrt(s / a.length);
}
