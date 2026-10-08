// cart-pole: the simulator and the policy, plain float64 TS shared by the trainer, the process, the views and the lesson.
// Classic cart-pole physics (Barto, Sutton & Anderson 1983, in the form OpenAI Gym uses): a 1 kg cart on a track, a
// 0.1 kg pole of half-length 0.5 m, a push of 10 N left or right each step, Euler steps of 0.02 s. The episode ends when
// the pole leans more than 12° or the cart leaves the track (|x| > 2.4 m), or after maxLen steps. Reward +1 per step.
// The policy: MLP('net', [4, hidden, 1], 'tanh', { outAct: 'sigmoid' }) -> p(left). pipeline/cart-pole/reference.py
// repeats this file line by line (same operation order), so the simulator and the returns match the Python run.
import type { Rng, Tensor } from '../../atlas/types/index.ts';

export const GRAVITY = 9.8;
export const M_CART = 1.0;
export const M_POLE = 0.1;
export const M_TOTAL = M_CART + M_POLE;
export const HALF_LEN = 0.5;
export const PML = M_POLE * HALF_LEN;
export const FORCE = 10.0;
export const TAU = 0.02;
export const X_LIMIT = 2.4;
export const TH_LIMIT = (12 * 2 * Math.PI) / 360;
/** random starts: every number uniform in ±START */
export const START = 0.05;

/** [x, ẋ, θ, θ̇]: cart position (m), cart velocity (m/s), pole angle from upright (rad, + = leaning right), its rate (rad/s) */
export type State = [number, number, number, number];
export type Params = Record<string, Tensor>;
export interface Physics { force: number; sin: number; cos: number; temp: number; thAcc: number; xAcc: number; s2: State; done: number }

/** One Euler step of 0.02 s after a push (left = 1: −10 N, else +10 N). done: 1 pole fell, 2 cart left the track, 0 no. */
export function physics(s: State, left: number): Physics {
	const [x, xd, th, thd] = s;
	const force = left === 1 ? -FORCE : FORCE;
	const cos = Math.cos(th);
	const sin = Math.sin(th);
	const temp = (force + PML * thd * thd * sin) / M_TOTAL;
	const thAcc = (GRAVITY * sin - cos * temp) / (HALF_LEN * (4.0 / 3.0 - (M_POLE * cos * cos) / M_TOTAL));
	const xAcc = temp - (PML * thAcc * cos) / M_TOTAL;
	const s2: State = [x + TAU * xd, xd + TAU * xAcc, th + TAU * thd, thd + TAU * thAcc];
	const done = s2[2] < -TH_LIMIT || s2[2] > TH_LIMIT ? 1 : s2[0] < -X_LIMIT || s2[0] > X_LIMIT ? 2 : 0;
	return { force, sin, cos, temp, thAcc, xAcc, s2, done };
}

/** A random start near upright: four draws of (2u − 1)·0.05. */
export const randomStart = (rng: Rng): State => [0, 1, 2, 3].map(() => (2 * rng.next() - 1) * START) as State;

/** The policy on one state: hidden h = tanh(W1·s + b1), score z = W2·h + b2, p(left) = 1 / (1 + e^(−z)). */
export function policy(p: Params, s: ArrayLike<number>): { h: Float64Array; z: number; p: number } {
	const W1 = p['net.0.weight'], b1 = p['net.0.bias'], W2 = p['net.2.weight'], b2 = p['net.2.bias'];
	const H = W1.shape[0];
	const h = new Float64Array(H);
	for (let j = 0; j < H; j++) {
		let a = 0;
		for (let i = 0; i < 4; i++) a += W1.data[j * 4 + i] * s[i];
		h[j] = Math.tanh(a + b1.data[j]);
	}
	let z = 0;
	for (let j = 0; j < H; j++) z += W2.data[j] * h[j];
	z += b2.data[0];
	return { h, z, p: 1 / (1 + Math.exp(-z)) };
}

export interface Episode { states: State[]; p: number[]; u: number[]; left: number[]; len: number; end: number; last: State }
/** One episode: at each step draw u = rng.next(), push left when u < p(left). end: 1 fell, 2 off the track, 3 time limit. */
export function rollout(params: Params, start: State, rng: Rng, maxLen: number): Episode {
	const e: Episode = { states: [], p: [], u: [], left: [], len: 0, end: 3, last: start };
	let s = start;
	while (e.len < maxLen) {
		const p = policy(params, s).p, u = rng.next(), left = u < p ? 1 : 0;
		e.states.push(s), e.p.push(p), e.u.push(u), e.left.push(left);
		const f = physics(s, left);
		e.len++;
		s = e.last = f.s2;
		if (f.done) {
			e.end = f.done;
			break;
		}
	}
	return e;
}

/** Discounted returns of a reward of 1 per step: G[T−1] = 1, G[t] = 1 + γ·G[t+1]. */
export function returns(len: number, gamma: number): number[] {
	const G = new Array<number>(len);
	let g = 0;
	for (let t = len - 1; t >= 0; t--) G[t] = g = 1 + gamma * g;
	return G;
}

/** Mean and standard deviation (divided by n) of a list. */
export function meanStd(v: number[]): { mean: number; std: number } {
	let m = 0;
	for (const x of v) m += x;
	m /= v.length;
	let q = 0;
	for (const x of v) q += (x - m) * (x - m);
	return { mean: m, std: Math.sqrt(q / v.length) };
}

/** Normalized advantages A = (G − mean) / std over every step of a round (std 0: all 0). */
export const advantages = (G: number[]) => {
	const { mean, std } = meanStd(G);
	return G.map((g) => (std > 0 ? (g - mean) / std : 0));
};

/** Degrees, for the narration. */
export const deg = (rad: number) => (rad * 180) / Math.PI;
/** A round's average episode length (whole lengths over 10 episodes: one decimal) as it is: 199.9, never rounded to 200. */
export const lenText = (v: number) => (Number.isFinite(v) ? String(Math.round(v * 10) / 10) : NaN);

/** The learner's input: the start state of the episodes the lesson follows (the first episode of every round). */
export type CartRaw = { start: State };
export const DEFAULT_START: State = [0, 0, 0.05, 0];
/** Slider ranges of the four numbers (the Capture and startOf clamp to them). */
export const RANGES: [number, number][] = [[-2, 2], [-1, 1], [-0.2, 0.2], [-1, 1]];
export function startOf(raw: unknown): State {
	const s = (raw as Partial<CartRaw> | null)?.start;
	return [0, 1, 2, 3].map((i) => {
		const v = Number(Array.isArray(s) ? s[i] : DEFAULT_START[i]);
		return Number.isFinite(v) ? Math.min(RANGES[i][1], Math.max(RANGES[i][0], v)) : DEFAULT_START[i];
	}) as State;
}
