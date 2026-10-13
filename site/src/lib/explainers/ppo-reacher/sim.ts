// ppo-reacher: the simulator, the Gaussian policy and GAE, plain float64 TS shared by the trainer, the process, the views,
// the lesson and the claims. pipeline/ppo-reacher/reference.py repeats this file line by line (same operation order).
// The arm: two links of 0.5 m in a horizontal plane (no gravity), the shoulder at the origin. Each joint is a motor:
// a torque a speeds it up, friction slows it down (ω' = ω + τ·(GAIN·a − DAMP·ω), then q' = q + τ·ω'); the two joints do not
// push on each other, as a real arm's links would. Every episode starts from the same pose (shoulder 0, elbow 90°, still)
// and lasts T steps of 0.05 s. Reward per step: −(distance from the fingertip to the target) − CTRL·(a1² + a2²).
// The policy sees 6 numbers: the target minus the fingertip (dx, dy), the two joint angles and the two joint speeds.
import type { Rng, Tensor } from '../../atlas/types/index.ts';

export const L1 = 0.5;
export const L2 = 0.5;
export const TAU = 0.05;
export const GAIN = 20;
export const DAMP = 10;
export const CTRL = 0.05;
export const Q0: [number, number] = [0, Math.PI / 2];
const LOG2PI = Math.log(2 * Math.PI);

/** [q1, q2, ω1, ω2]: shoulder and elbow angles (rad) and their speeds (rad/s) */
export type ArmState = [number, number, number, number];
export type Params = Record<string, Tensor>;
export type Pt = [number, number];

export const START: ArmState = [Q0[0], Q0[1], 0, 0];
export const elbow = (s: ArmState): Pt => [L1 * Math.cos(s[0]), L1 * Math.sin(s[0])];
export function tip(s: ArmState): Pt {
	const a = s[0] + s[1];
	return [L1 * Math.cos(s[0]) + L2 * Math.cos(a), L1 * Math.sin(s[0]) + L2 * Math.sin(a)];
}
export const dist = (p: Pt, g: Pt) => Math.hypot(p[0] - g[0], p[1] - g[1]);

/** What the networks see: [dx, dy, q1, q2, ω1, ω2] with (dx, dy) = target − fingertip. */
export function obs(s: ArmState, g: Pt): number[] {
	const p = tip(s);
	return [g[0] - p[0], g[1] - p[1], s[0], s[1], s[2], s[3]];
}

/** One step of 0.05 s with torques a: the new state and the reward. */
export function step(s: ArmState, a: Pt, g: Pt) {
	const w1 = s[2] + TAU * (GAIN * a[0] - DAMP * s[2]);
	const w2 = s[3] + TAU * (GAIN * a[1] - DAMP * s[3]);
	const s2: ArmState = [s[0] + TAU * w1, s[1] + TAU * w2, w1, w2];
	const d = dist(tip(s2), g);
	const cost = CTRL * (a[0] * a[0] + a[1] * a[1]);
	return { s2, d, cost, r: -d - cost };
}

/** A small MLP (tanh hidden layers, linear output) on one input, from a parameter dict: name.0, name.2, ... */
export function mlp(p: Params, name: string, x: number[]): number[] {
	let h = x;
	for (let l = 0; p[`${name}.${2 * l}.weight`]; l++) {
		const W = p[`${name}.${2 * l}.weight`], b = p[`${name}.${2 * l}.bias`], [O, I] = W.shape, y = new Array<number>(O);
		for (let o = 0, a = 0; o < O; o++, a = 0) {
			for (let i = 0; i < I; i++) a += W.data[o * I + i] * h[i];
			y[o] = a + b.data[o];
		}
		h = p[`${name}.${2 * l + 2}.weight`] ? y.map(Math.tanh) : y;
	}
	return h;
}

/** The Gaussian policy at one state: the means μ (network 'pi'), σ = e^logstd. */
export function policy(p: Params, o: number[]) {
	const mu = mlp(p, 'pi', o), ls = Array.from(p.logstd.data);
	return { mu: [mu[0], mu[1]] as Pt, logstd: [ls[0], ls[1]] as Pt, sigma: [Math.exp(ls[0]), Math.exp(ls[1])] as Pt };
}
export const value = (p: Params, o: number[]) => mlp(p, 'v', o)[0];

/** log π(a | s) of a diagonal Gaussian: Σ −½((a − μ)/σ)² − log σ − ½ log 2π. */
export function logProb(a: Pt, mu: Pt, logstd: Pt): number {
	let s = 0;
	for (let j = 0, z = 0; j < 2; j++) (z = (a[j] - mu[j]) / Math.exp(logstd[j])), (s += -0.5 * z * z - logstd[j] - 0.5 * LOG2PI);
	return s;
}

export interface Episode { obs: number[][]; act: Pt[]; logp: number[]; v: number[]; r: number[]; vLast: number }
/** One episode of T steps: at each step draw z1, z2 ~ N(0, 1) and act a = μ + σ·z. v holds V(s_t); vLast = V(s_T). */
export function rollout(p: Params, g: Pt, rng: Rng, T: number): Episode {
	const e: Episode = { obs: [], act: [], logp: [], v: [], r: [], vLast: 0 };
	let s = START;
	for (let t = 0; t < T; t++) {
		const o = obs(s, g), pol = policy(p, o), z1 = rng.normal(), z2 = rng.normal();
		const a: Pt = [pol.mu[0] + pol.sigma[0] * z1, pol.mu[1] + pol.sigma[1] * z2], st = step(s, a, g);
		e.obs.push(o), e.act.push(a), e.logp.push(logProb(a, pol.mu, pol.logstd)), e.v.push(value(p, o)), e.r.push(st.r);
		s = st.s2;
	}
	e.vLast = value(p, obs(s, g));
	return e;
}

/** GAE, backwards: δ_t = r_t + γ·V_(t+1) − V_t (V_T = vLast: the episode is cut off, not over), A_t = δ_t + γλ·A_(t+1);
 *  the value target R_t = A_t + V_t. */
export function gae(r: number[], v: number[], vLast: number, gamma: number, lambda: number) {
	const T = r.length, A = new Array<number>(T), delta = new Array<number>(T), R = new Array<number>(T);
	let next = 0;
	for (let t = T - 1; t >= 0; t--) {
		const v1 = t === T - 1 ? vLast : v[t + 1];
		delta[t] = r[t] + gamma * v1 - v[t];
		A[t] = next = delta[t] + gamma * lambda * next;
		R[t] = A[t] + v[t];
	}
	return { A, delta, R };
}

/** Mean and standard deviation (divided by n). */
export function meanStd(v: ArrayLike<number>): { mean: number; std: number } {
	let m = 0, q = 0;
	for (let i = 0; i < v.length; i++) m += v[i];
	m /= v.length;
	for (let i = 0; i < v.length; i++) q += (v[i] - m) * (v[i] - m);
	return { mean: m, std: Math.sqrt(q / v.length) };
}

/** A random target for training, in the upper half plane (y ≥ 0): radius uniform in [0.25, 0.95] m, angle uniform in
 *  [0, π]: two draws of rng.next(). */
export function randomTarget(rng: Rng): Pt {
	const rad = 0.25 + 0.7 * rng.next(), ang = Math.PI * rng.next();
	return [rad * Math.cos(ang), rad * Math.sin(ang)];
}

/** The learner's input: the target of the episode the lesson follows, and the clip ε of PPO (the task changes it). */
export type ReachRaw = { target: Pt; eps: number };
export const DEFAULT_TARGET: Pt = [-0.3, 0.6];
export const DEFAULT_EPS = 0.2;
export const EPS_CHOICES = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 100];
/** The learner's target, kept where training puts targets: y ≥ 0, 0.2 m ≤ distance from the shoulder ≤ 1 m (2 decimals). */
export function targetOf(raw: unknown): Pt {
	const g = (raw as Partial<ReachRaw> | null)?.target;
	let x = Number(Array.isArray(g) ? g[0] : NaN), y = Number(Array.isArray(g) ? g[1] : NaN);
	if (!Number.isFinite(x) || !Number.isFinite(y)) return DEFAULT_TARGET;
	y = Math.max(0, y);
	const r = Math.hypot(x, y), k = r > 1 ? 1 / r : r < 0.2 ? (r > 0 ? 0.2 / r : 1) : 1;
	if (r === 0) (x = 0), (y = 0.2);
	return [Math.round(x * k * 100) / 100, Math.round(y * k * 100) / 100];
}
export function epsOf(raw: unknown): number {
	const e = Number((raw as Partial<ReachRaw> | null)?.eps);
	return Number.isFinite(e) && e > 0 ? Math.min(100, e) : DEFAULT_EPS;
}

/** The policy without its randomness: T steps with a = μ (the trained arm's demonstration). */
export function meanEpisode(p: Params, g: Pt, T: number) {
	const states: ArmState[] = [START], d: number[] = [];
	let s = START;
	for (let t = 0; t < T; t++) {
		const st = step(s, policy(p, obs(s, g)).mu, g);
		states.push((s = st.s2)), d.push(st.d);
	}
	return { states, d, final: d[d.length - 1] };
}

/** The limits step's test targets: 8 at 0.6 m in the upper half plane (where training puts targets) and their mirror
 *  images below the shoulder (y < 0), where training never put one. */
export const EVAL_UP: Pt[] = Array.from({ length: 8 }, (_, k) => [0.6 * Math.cos((Math.PI * (k + 0.5)) / 8), 0.6 * Math.sin((Math.PI * (k + 0.5)) / 8)]);
export const EVAL_DOWN: Pt[] = EVAL_UP.map(([x, y]) => [x, -y]);
