// tiny-tcn: the processes (kind C).
//   arima     the ARIMA(p, 1, q)-lite baseline (arima.ts) on the learner's series: the series and its changes, the fit
//             (the chosen order and coefficients), the forecast of the held-out part. Input: the series as plain numbers,
//             so the Python reference (pipeline/tiny-tcn/reference.py) reads the same list.
//   forecast  after the 'train' source: the trained TCN forecasts the held-out part (each forecast change is appended to
//             the changes and read back: a rollout), next to ARIMA's forecast and "repeat the last value". One frame.
//   npm run atlas -- run tiny-tcn arima --frames 0..3
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { dataOf, type Data } from './data.ts';
import { fitArima, forecastArima, P_MAX, Q_MAX } from './arima.ts';
import { nextChange, type Params } from './model.ts';

const T = (en: string, zh: string) => ({ en, zh });
const vec = (a: ArrayLike<number>): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
const mae = (path: ArrayLike<number>, d: Data) => Array.from(path).reduce((s, v, h) => s + Math.abs(v - d.y[d.split + h]), 0) / d.hold;

/** ARIMA on the training part of d: the fit, its forecast path and mean |miss| on the held-out part. */
export function arimaOf(d: Data) {
	const z = Array.from(d.c.subarray(1, d.split));
	const fit = fitArima(z);
	const fc = forecastArima(fit, z, d.y[d.split - 1], d.hold);
	return { fit, ...fc, mae: mae(fc.path, d) };
}

export type ArState = { series: Tensor; changes: Tensor; coef: Tensor; path: Tensor };

export const arima: ProcessDef<unknown, Record<string, never>, ArState> = {
	id: 'arima',
	kind: 'algorithm',
	title: T('ARIMA-lite: fit the changes, then forecast', 'ARIMA 简化版：拟合变化量，再预测'),
	phases: {
		series: { label: T('The series and its changes', '序列和它的变化量'), kind: 'input' },
		fit: { label: T('Fit the coefficients', '拟合系数'), kind: 'update' },
		forecast: { label: T('Forecast the held-out part', '预测留出的部分'), kind: 'output' }
	},
	schema: {
		series: { kind: 'tensor', label: T('The series', '序列'), role: 'data', tol: 1e-12 },
		changes: { kind: 'tensor', label: T('The changes', '变化量'), role: 'data', tol: 1e-12 },
		coef: { kind: 'tensor', label: T('μ, φ…, θ…', 'μ、φ…、θ…'), role: 'param', tol: 1e-9 },
		path: { kind: 'tensor', label: T('ARIMA forecast', 'ARIMA 的预测'), role: 'output', tol: 1e-9 }
	},
	metricTol: 1e-9,
	defaults: {},
	*run(input): ProcessRun<ArState, Record<string, never>> {
		const d = dataOf(input);
		yield { phase: 'series', state: { series: vec(d.y), changes: vec(d.c) }, metrics: { n: d.n, hold: d.hold, split: d.split, last: d.c[d.split - 1] } };
		const a = arimaOf(d);
		const f = a.fit;
		yield { phase: 'fit', state: { coef: vec([f.mu, ...f.phi, ...f.theta]) }, metrics: { p: f.p, q: f.q, mu: f.mu, phi1: f.phi[0] ?? 0, sigma: Math.sqrt(f.sigma2), N: f.N, pMax: P_MAX, qMax: Q_MAX, k: f.p + f.q + 1 } };
		yield { phase: 'forecast', state: { path: vec(a.path) }, metrics: { mae: a.mae, hold: d.hold, first: a.path[0] ?? NaN } };
	}
};

/** The trained TCN's forecast of the held-out part: each forecast change goes onto the changes and is read back. */
export function tcnPath(P: Params, d: Data): number[] {
	const c = Array.from(d.c.subarray(0, d.split));
	const path: number[] = [];
	let y = d.y[d.split - 1];
	for (let h = 0; h < d.hold; h++) {
		const v = nextChange(P, c.slice(-16));
		c.push(v);
		path.push((y += v));
	}
	return path;
}

export type FcState = { tcn: Tensor; arima: Tensor };

export const forecast: ProcessDef<unknown, Record<string, never>, FcState> = {
	id: 'forecast',
	kind: 'algorithm',
	title: T('The TCN and ARIMA forecast the held-out part', 'TCN 和 ARIMA 预测留出的部分'),
	phases: { compare: { label: T('Two forecasts of the held-out part', '对留出部分的两种预测'), kind: 'output' } },
	schema: {
		tcn: { kind: 'tensor', label: T('TCN forecast', 'TCN 的预测'), role: 'output', tol: 1e-9 },
		arima: { kind: 'tensor', label: T('ARIMA forecast', 'ARIMA 的预测'), role: 'output', tol: 1e-9 }
	},
	defaults: {},
	*run(input): ProcessRun<FcState, Record<string, never>> {
		const { y, params } = (input ?? {}) as { y?: number[]; params?: Params };
		if (!y || !params) return;
		const d = dataOf(y);
		const tp = tcnPath(params, d);
		const a = arimaOf(d);
		let naive = 0;
		for (let h = 0; h < d.hold; h++) naive += Math.abs(d.y[d.split - 1] - d.y[d.split + h]) / d.hold;
		yield { phase: 'compare', state: { tcn: vec(tp), arima: vec(a.path) }, metrics: { tcnMae: mae(tp, d), arimaMae: a.mae, naiveMae: naive, p: a.fit.p, q: a.fit.q, hold: d.hold } };
	}
};
