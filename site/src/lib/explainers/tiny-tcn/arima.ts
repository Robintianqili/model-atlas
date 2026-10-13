// tiny-tcn: the ARIMA(p, 1, q)-lite baseline, our own code on plain numbers (the npm "arima" package has no licence).
// ARIMA forecasts the changes z_t = y_t − y_{t−1} (d = 1, the same changes the TCN reads) as a drift μ, plus p earlier
// changes times coefficients φ (the AR part), plus q earlier forecast errors times θ (the MA part):
//   z_t = μ + φ_1 z_{t−1} + … + φ_p z_{t−p} + θ_1 e_{t−1} + … + θ_q e_{t−q} + e_t
// "Lite": fitted by least squares in two stages (Hannan & Rissanen 1982) instead of maximum likelihood:
//   1. a long AR(8) by least squares gives estimates of the errors e_t;
//   2. for every p ≤ 5, q ≤ 2 with at most N/3 coefficients, least squares on [1, z_{t−1..t−p}, e_{t−1..t−q}] over the same rows; the order with the
//      smallest AIC = N·ln(σ²) + 2(p + q + 1) wins (ties go to the smaller p, then q).
// Forecast: the errors of the training part are recomputed with the chosen coefficients; future errors are 0; each
// forecast change is added to the last value (rolled forward like the TCN's forecasts). Only the training part
// (times < split) is ever read. Python reference: pipeline/tiny-tcn/reference.py (process 'arima', the same loops).
export const P_MAX = 5;
export const Q_MAX = 2;
export const M_LONG = 8;
const RIDGE = 1e-8;

/** Least squares by the normal equations (+ a tiny ridge so a perfectly regular series still solves) and Gaussian
 *  elimination with partial pivoting. rows: the regressor rows, y: the targets. */
export function lstsq(rows: number[][], y: number[]): number[] {
	const k = rows[0].length;
	const A = Array.from({ length: k }, (_, i) => Array.from({ length: k + 1 }, (_, j) => (j === i ? RIDGE : 0)));
	for (let r = 0; r < rows.length; r++)
		for (let i = 0; i < k; i++) {
			for (let j = 0; j < k; j++) A[i][j] += rows[r][i] * rows[r][j];
			A[i][k] += rows[r][i] * y[r];
		}
	for (let i = 0; i < k; i++) {
		let p = i;
		for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
		[A[i], A[p]] = [A[p], A[i]];
		for (let r = i + 1; r < k; r++) {
			const f = A[r][i] / A[i][i];
			for (let j = i; j <= k; j++) A[r][j] -= f * A[i][j];
		}
	}
	const x = new Array<number>(k).fill(0);
	for (let i = k - 1; i >= 0; i--) {
		let s = A[i][k];
		for (let j = i + 1; j < k; j++) s -= A[i][j] * x[j];
		x[i] = s / A[i][i];
	}
	return x;
}

/** Are the roots of 1 − a_1 B − … − a_k B^k on or outside the unit circle (an AR part that does not explode; with
 *  a = −θ, an MA part whose errors can be recovered)? Step-down (Levinson–Durbin) recursion: every partial coefficient
 *  |κ| ≤ 1 (a pure sine sits exactly on the circle, κ = −1, and is kept). */
export function stable(a: number[]): boolean {
	let c = a.slice();
	for (let k = c.length; k >= 1; k--) {
		const kap = c[k - 1];
		if (!(Math.abs(kap) <= 1 + 1e-9)) return false;
		if (Math.abs(kap) >= 1 - 1e-12) return true;
		c = c.slice(0, k - 1).map((v, j) => (v + kap * c[k - 2 - j]) / (1 - kap * kap));
	}
	return true;
}

export interface ArimaFit {
	p: number;
	q: number;
	mu: number;
	phi: number[];
	theta: number[];
	sigma2: number;
	aic: number;
	/** rows of stage 2 */
	N: number;
}

const lagRow = (z: number[], e: number[], t: number, p: number, q: number) => {
	const row = [1];
	for (let i = 1; i <= p; i++) row.push(z[t - i]);
	for (let j = 1; j <= q; j++) row.push(e[t - j]);
	return row;
};

/** Fit on the changes z (the training part's z_1..z_{split−1}: index 0 is the change at time 1). */
export function fitArima(z: number[]): ArimaFit {
	const long = lstsq(z.slice(M_LONG).map((_, r) => lagRow(z, [], r + M_LONG, M_LONG, 0)), z.slice(M_LONG));
	const e = z.map((v, t) => (t < M_LONG ? 0 : v - lagRow(z, [], t, M_LONG, 0).reduce((s, x, i) => s + x * long[i], 0)));
	const t0 = M_LONG + Q_MAX;
	const ys = z.slice(t0);
	const N = ys.length;
	let best: ArimaFit | null = null;
	for (let p = 0; p <= P_MAX; p++)
		for (let q = 0; q <= Q_MAX; q++) {
			if (3 * (p + q + 1) > N) continue;
			const rows = ys.map((_, r) => lagRow(z, e, r + t0, p, q));
			const b = lstsq(rows, ys);
			let rss = 0;
			rows.forEach((row, r) => {
				const f = row.reduce((s, x, i) => s + x * b[i], 0);
				rss += (ys[r] - f) * (ys[r] - f);
			});
			const sigma2 = rss / N;
			const aic = N * Math.log(sigma2 + 1e-12) + 2 * (p + q + 1);
			if (!stable(b.slice(1, 1 + p)) || !stable(b.slice(1 + p).map((v) => -v))) continue;
			if (!best || aic < best.aic) best = { p, q, mu: b[0], phi: b.slice(1, 1 + p), theta: b.slice(1 + p), sigma2, aic, N };
		}
	return best!;
}

/** The fitted model's errors over z (recomputed with its own coefficients; earlier terms that do not exist count as 0),
 *  then `hold` forecast changes (future errors 0) added one by one to the last value `last`. */
export function forecastArima(f: ArimaFit, z: number[], last: number, hold: number): { changes: number[]; path: number[] } {
	const zz = z.slice();
	const e: number[] = [];
	const at = (a: number[], i: number) => (i >= 0 && i < a.length ? a[i] : 0);
	const predict = (t: number) => f.mu + f.phi.reduce((s, ph, i) => s + ph * at(zz, t - 1 - i), 0) + f.theta.reduce((s, th, j) => s + th * at(e, t - 1 - j), 0);
	for (let t = 0; t < z.length; t++) e.push(z[t] - predict(t));
	const changes: number[] = [];
	const path: number[] = [];
	let y = last;
	for (let h = 0; h < hold; h++) {
		const t = zz.length;
		const v = predict(t);
		zz.push(v);
		e.push(0);
		changes.push(v);
		path.push((y += v));
	}
	return { changes, path };
}
