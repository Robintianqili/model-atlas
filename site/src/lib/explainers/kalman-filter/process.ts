// kalman-filter: process 'kf' (kind C). A Kalman filter with a constant-velocity motion model follows the sensor
// readings of a cart on a rail, one frame per event; pipeline/kalman-filter/reference.py repeats it in Python, frame by
// frame. The true positions are in the input only to score the run: the filter itself reads nothing but the readings.
//   start    the first reading is the first guess (velocity 0, unsure)
//   predict  second k: move one second with the estimated velocity; the doubt (covariance P) grows by Q
//   measure  the reading z arrives; the surprise y = z − p⁻
//   gain     K = P⁻ / (P⁻ + R): the share of the surprise the filter believes
//   update   p = p⁻ + K y; the doubt shrinks to (1 − K) P⁻
//   done     the run scored against the truth (raw readings, moving average, filter), and the best q, r on the sliders
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { I, kfPredict, kfRun, kfStart, kfUpdate, movingAverage, QS, rmse, RS, type TrackData } from './track.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });
export type KfState = { truth: Tensor; z: Tensor; ma: Tensor; x: Tensor; P: Tensor };
const lab = (en: string, zh: string) => ({ label: I(en, zh) });

export const kf: ProcessDef<TrackData, Record<string, never>, KfState> = {
	id: 'kf',
	kind: 'algorithm',
	title: I('A Kalman filter follows a noisy track', '卡尔曼滤波跟踪一条有噪声的轨迹'),
	phases: {
		start: { ...lab('The first reading is the first guess', '第一个读数就是第一个猜测'), kind: 'input' },
		predict: { ...lab('Predict: move on with the estimated speed', '预测：按估计的速度往前走'), kind: 'forward' },
		measure: { ...lab('A reading arrives', '读数到了'), kind: 'observe' },
		gain: { ...lab('The Kalman gain: how much to trust the reading', '卡尔曼增益：读数信几分'), kind: 'other' },
		update: { ...lab('Update: move part of the way to the reading', '更新：朝读数挪一部分'), kind: 'update' },
		done: { ...lab('The whole track, scored', '整条轨迹的成绩'), kind: 'output' }
	},
	schema: {
		truth: { kind: 'tensor', ...lab('True position each second (the filter never reads it)', '每秒的真实位置（滤波器从不读取）'), role: 'target', tol: 'exact' },
		z: { kind: 'tensor', ...lab('Sensor reading each second', '每秒的传感器读数'), role: 'data', tol: 'exact' },
		ma: { kind: 'tensor', ...lab('Moving average of the last readings', '最近几个读数的移动平均'), tol: 1e-12 },
		x: { kind: 'tensor', ...lab('The estimate: position (m) and velocity (m/s)', '估计：位置（米）和速度（米/秒）'), tol: 1e-12 },
		P: { kind: 'tensor', ...lab('Its covariance P (the doubt)', '它的协方差 P（不确定程度）'), tol: 1e-12 }
	},
	defaults: {},
	*run(d): ProcessRun<KfState, Record<string, never>> {
		const n = d.z.length, { q, r, w, z, truth } = d;
		if (!n) return;
		const ma = movingAverage(z, w);
		const x = (p: number, v: number) => T([2], [p, v]);
		const cov = (b: { P00: number; P01: number; P11: number }) => T([2, 2], [b.P00, b.P01, b.P01, b.P11]);
		let b = kfStart(z[0], r);
		yield { phase: 'start', step: 0, state: { truth: T([n], truth), z: T([n], z), ma: T([n], ma), x: x(b.p, b.v), P: cov(b) }, metrics: { n, k: 0, z0: z[0], p: b.p, v: b.v, P00: b.P00, P11: b.P11, q, r, w, rmseZ: rmse(z, truth) } };
		const est = [b.p], sd = [Math.sqrt(b.P00)];
		for (let k = 1; k < n; k++) {
			const bp = kfPredict(b, q);
			yield { phase: 'predict', step: k, state: { x: x(bp.p, bp.v), P: cov(bp) }, metrics: { k, p: b.p, v: b.v, P00: b.P00, P01: b.P01, P11: b.P11, pp: bp.p, vp: bp.v, Pp00: bp.P00, Pp01: bp.P01, Pp11: bp.P11, q, sd: Math.sqrt(b.P00), sdp: Math.sqrt(bp.P00) } };
			const u = kfUpdate(bp, z[k], r);
			yield { phase: 'measure', step: k, state: {}, metrics: { k, pp: bp.p, z: z[k], y: u.y, r, Pp00: bp.P00, sdp: Math.sqrt(bp.P00), sdz: Math.sqrt(r), truth: truth[k] } };
			yield { phase: 'gain', step: k, state: {}, metrics: { k, Pp00: bp.P00, r, S: u.S, K: u.K, Kv: u.Kv, pp: bp.p, z: z[k] } };
			b = u.b;
			est.push(b.p);
			sd.push(Math.sqrt(b.P00));
			const err = b.p - truth[k];
			yield { phase: 'update', step: k, state: { x: x(b.p, b.v), P: cov(b) }, metrics: { k, pp: bp.p, vp: bp.v, z: z[k], y: u.y, K: u.K, Kv: u.Kv, p: b.p, v: b.v, Pp00: bp.P00, P00: b.P00, P01: b.P01, P11: b.P11, sdp: Math.sqrt(bp.P00), sd: Math.sqrt(b.P00), sdz: Math.sqrt(r), truth: truth[k], err, errZ: z[k] - truth[k], errMa: ma[k] - truth[k], out: Math.abs(err) > 2 * Math.sqrt(b.P00) ? 1 : 0 } };
		}
		let worst = 0, worstK = 0, cover = 0;
		for (let k = 0; k < n; k++) {
			const e = Math.abs(est[k] - truth[k]);
			if (e > worst) (worst = e), (worstK = k);
			if (e <= 2 * sd[k]) cover++;
		}
		let best = { e: Infinity, q: QS[0], r: RS[0] };
		for (const qq of QS)
			for (const rr of RS) {
				const e = rmse(kfRun(z, qq, rr).est, truth);
				if (e < best.e) best = { e, q: qq, r: rr };
			}
		yield { phase: 'done', step: n - 1, state: {}, metrics: { n, q, r, w, rmseKf: rmse(est, truth), rmseZ: rmse(z, truth), rmseMa: rmse(ma, truth), worst, worstK, cover, bestQ: best.q, bestR: best.r, bestRmse: best.e } };
	}
};
