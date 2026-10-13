// optimizers: the page's fixed numbers and plain arithmetic, shared by the processes, the setup widget, the views and the
// claim tests. Nothing here is fitted or downloaded: three analytic loss surfaces of two weights (x, y), the four racers
// (the kit's own SGD, Momentum, Adam and AdamW from atlas/autograd/optim.ts, which follow torch.optim step for step),
// an exploding-gradient toy (a one-number recurrence run for eight steps), the learning-rate schedules and the setup of
// the learning-rate range test on a small network.
import type { I18n } from '../../atlas/types/index.ts';
import type { OptimizerKind } from '../../atlas/autograd/optim.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { num } from '../../atlas/lesson/say.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });

// ---------------------------------------------------------------- the racers

export type RacerKey = 'sgd' | 'momentum' | 'adam' | 'adamw';
export interface Racer {
	key: RacerKey;
	kind: OptimizerKind;
	name: I18n;
	lr: number;
}
/** Momentum's μ, Adam's β1 and β2 and ε (PyTorch's defaults), AdamW's weight decay λ. */
export const MU = 0.9;
export const BETAS: [number, number] = [0.9, 0.999];
export const EPS = 1e-8;
export const LAMBDA = 0.1;
/** One learning rate per optimizer, the same on every surface. SGD's is close to the largest the valley allows
 *  (2 / its steepest curvature, 4, is 0.5); momentum's η / (1 − μ) = 1 is its long-run step; Adam moves about η per
 *  coordinate per step whatever the gradient's size. */
export const RACERS: Racer[] = [
	{ key: 'sgd', kind: 'sgd', name: L('SGD', 'SGD'), lr: 0.4 },
	{ key: 'momentum', kind: 'momentum', name: L('momentum', '动量法'), lr: 0.1 },
	{ key: 'adam', kind: 'adam', name: L('Adam', 'Adam'), lr: 0.1 },
	{ key: 'adamw', kind: 'adamw', name: L('AdamW', 'AdamW'), lr: 0.1 }
];
export const racerIndex = (k: RacerKey) => RACERS.findIndex((r) => r.key === k);

// ---------------------------------------------------------------- the surfaces

export type SurfaceId = 'valley' | 'saddle' | 'bumpy' | 'cliff';
export interface Surface {
	id: SurfaceId;
	name: I18n;
	f(x: number, y: number): number;
	grad(x: number, y: number): [number, number];
	xr: [number, number];
	yr: [number, number];
	start: [number, number];
	/** the lowest points (every global minimum) and the lowest value */
	min: [number, number][];
	axes: [I18n, I18n];
}

const sq = (v: number) => v * v;
const XY: [I18n, I18n] = [L('weight x', '权重 x'), L('weight y', '权重 y')];

/** A long, narrow valley: 100 times steeper across (y) than along (x); the bottom is (1, 0). */
const valley: Surface = {
	id: 'valley',
	name: L('long valley', '狭长山谷'),
	f: (x, y) => 0.02 * sq(x - 1) + 2 * sq(y),
	grad: (x, y) => [0.04 * (x - 1), 4 * y],
	xr: [-4.5, 2.5],
	yr: [-1.5, 1.5],
	start: [-4, 1],
	min: [[1, 0]],
	axes: XY
};

/** A saddle: downhill along x into (0.5, 0), where the surface curves down gently in y towards two bottoms (0.5, ±1). */
const saddle: Surface = {
	id: 'saddle',
	name: L('saddle', '鞍点'),
	f: (x, y) => 0.5 * sq(x - 0.5) + 0.05 * sq(y * y - 1),
	grad: (x, y) => [x - 0.5, 0.2 * y * (y * y - 1)],
	xr: [-3, 3],
	yr: [-1.5, 1.5],
	start: [-2.5, 0.02],
	min: [
		[0.5, -1],
		[0.5, 1]
	],
	axes: XY
};

/** A bumpy bowl: a wide bowl with its bottom at (1, 0.5) and ripples that make small dips all over it. */
const bumpy: Surface = {
	id: 'bumpy',
	name: L('bumpy bowl', '坑洼的碗'),
	f: (x, y) => 0.1 * (sq(x - 1) + sq(y - 0.5)) + 0.15 * (2 - Math.cos(4 * (x - 1)) - Math.cos(4 * (y - 0.5))),
	grad: (x, y) => [0.2 * (x - 1) + 0.6 * Math.sin(4 * (x - 1)), 0.2 * (y - 0.5) + 0.6 * Math.sin(4 * (y - 0.5))],
	xr: [-3.5, 3.5],
	yr: [-2.5, 3.5],
	start: [-3, 3],
	min: [[1, 0.5]],
	axes: XY
};

// ---------------------------------------------------------------- the exploding-gradient toy

/** A one-number recurrence h_t = w·h_{t−1} + b, from h_0 = 0, run for T steps; the loss is (h_T − target)². Its
 *  gradient holds powers of w up to w^(T−1), so it explodes once w passes 1: a cliff in the landscape. */
export const CLIFF = { T: 8, target: 3, start: [0.5, 0.3] as [number, number], lr: 0.03, maxNorm: 1, steps: 30, blowUp: 1e6 };

/** The loss and its gradient (dL/dw, dL/db) by the chain rule through the T steps (backpropagation through time). */
export function cliffLoss(w: number, b: number): { loss: number; gw: number; gb: number; h: number } {
	let h = 0;
	let dw = 0;
	let db = 0;
	for (let t = 0; t < CLIFF.T; t++) {
		dw = h + w * dw;
		db = 1 + w * db;
		h = w * h + b;
	}
	const e = h - CLIFF.target;
	return { loss: e * e, gw: 2 * e * dw, gb: 2 * e * db, h };
}

const cliff: Surface = {
	id: 'cliff',
	name: L('cliff', '悬崖'),
	f: (w, b) => cliffLoss(w, b).loss,
	grad: (w, b) => {
		const r = cliffLoss(w, b);
		return [r.gw, r.gb];
	},
	xr: [-0.25, 1.35],
	yr: [-0.25, 1.25],
	start: CLIFF.start,
	min: [],
	axes: [L('weight w', '权重 w'), L('bias b', '偏置 b')]
};

/** Gradient clipping by norm, as torch.nn.utils.clip_grad_norm_: g · min(1, c / (‖g‖ + 10⁻⁶)). */
export function clipByNorm(g: [number, number], c: number): { g: [number, number]; norm: number; scale: number } {
	const norm = Math.hypot(g[0], g[1]);
	const scale = Math.min(1, c / (norm + 1e-6));
	return { g: [g[0] * scale, g[1] * scale], norm, scale };
}

export const SURFACES: Surface[] = [valley, saddle, bumpy, cliff];
export const RACE_SURFACES: SurfaceId[] = ['valley', 'saddle', 'bumpy'];
export const surfaceOf = (id: unknown): Surface => SURFACES.find((s) => s.id === id) ?? valley;
export const surfaceIndex = (id: SurfaceId) => SURFACES.findIndex((s) => s.id === id);
/** a league metric's name: valleySgd, saddleAdamw, ... */
export const leagueKey = (s: SurfaceId, r: RacerKey) => `${s}${r[0].toUpperCase()}${r.slice(1)}`;
/** The race's step count and the outside of the window where a racer counts as gone. */
export const RACE_STEPS = 80;

// ---------------------------------------------------------------- the learner's setup

export type ScheduleId = 'constant' | 'step' | 'cosine';
export interface Setup {
	surface: SurfaceId;
	x: number;
	y: number;
	schedule: ScheduleId;
	warmup: boolean;
	clip: boolean;
}
export const SCHEDULE_NAMES: Record<ScheduleId, I18n> = {
	constant: L('constant rate', '恒定学习率'),
	step: L('step decay', '阶梯衰减'),
	cosine: L('cosine decay', '余弦衰减')
};
export const START: Setup = { surface: 'valley', x: valley.start[0], y: valley.start[1], schedule: 'constant', warmup: false, clip: false };

const finite = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v));

/** Any raw input -> a valid setup: a race surface, a start inside its window, a schedule and two switches. */
export function setupOf(raw: unknown): Setup {
	const o = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof Setup, unknown>>;
	const id = RACE_SURFACES.includes(o.surface as SurfaceId) ? (o.surface as SurfaceId) : 'valley';
	const s = surfaceOf(id);
	return {
		surface: id,
		x: clamp(finite(o.x, s.start[0]), s.xr),
		y: clamp(finite(o.y, s.start[1]), s.yr),
		schedule: o.schedule === 'step' || o.schedule === 'cosine' ? o.schedule : 'constant',
		warmup: o.warmup === true,
		clip: o.clip === true
	};
}

// ---------------------------------------------------------------- schedules

/** The schedule demo: SGD on the small network at a peak rate past where the range test blows up. Step decay multiplies
 *  η by 0.1 at steps 100 and 150 (PyTorch MultiStepLR); cosine decay runs from the peak to 0 (CosineAnnealingLR); warm-up
 *  climbs linearly over the first 20 steps (η·(k + 1) / 20) and the decay starts after it. */
export const SCHED = { peak: 4, steps: 200, warm: 20, milestones: [100, 150], gamma: 0.1 };

export function lrAt(schedule: ScheduleId, warmup: boolean, k: number): number {
	const { peak, steps, warm, milestones, gamma } = SCHED;
	const ramp = warmup && k < warm ? (k + 1) / warm : 1;
	let f = 1;
	if (schedule === 'step') f = Math.pow(gamma, milestones.filter((m) => k >= m).length);
	if (schedule === 'cosine') {
		const k0 = warmup ? warm : 0;
		f = k < k0 ? 1 : 0.5 * (1 + Math.cos((Math.PI * (k - k0)) / (steps - k0)));
	}
	return peak * ramp * f;
}

// ---------------------------------------------------------------- the learning-rate range test

/** η grows by the same factor every step, from 10⁻³ to 10 in 100 steps (Smith's range test). */
export const RANGE = { lo: 1e-3, hi: 10, steps: 100 };
export const rangeLr = (k: number) => RANGE.lo * Math.pow(RANGE.hi / RANGE.lo, k / (RANGE.steps - 1));

/** What the range test reads off its curve: the steepest fall (the largest drop between the neighbours of a step, the
 *  steps being equal on a log scale), the lowest loss, and where the loss first climbs above its starting value after
 *  the lowest point. Indexes count from 0; NaN when there is none. */
export function readRange(lrs: ArrayLike<number>, losses: ArrayLike<number>) {
	const n = Math.min(lrs.length, losses.length);
	let low = 0;
	for (let k = 1; k < n; k++) if (Number.isFinite(losses[k]) && losses[k] < losses[low]) low = k;
	let steep = NaN;
	let drop = 0;
	for (let k = 1; k < low; k++) {
		const d = losses[k - 1] - losses[k + 1];
		if (d > drop) {
			drop = d;
			steep = k;
		}
	}
	let blow = NaN;
	for (let k = low + 1; k < n; k++)
		if (!(losses[k] <= losses[0])) {
			blow = k;
			break;
		}
	return { low, steep, blow };
}

// ---------------------------------------------------------------- narration helpers

/** A frame's metrics, and a setting as it was set ('0.4', not '0.400'; NaN, shown as "…", when missing). */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
export const setting = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? fmtSetting(v) : NaN);

/** numbers printed with the fewest significant digits (3 to 6) that keep different values apart (AK410) */
export const distinct = (vs: number[]): (I18n | number)[] => {
	for (const d of [3, 4, 5, 6]) {
		const out = vs.map((v) => num(v, 'n' + d));
		const seen = new Map<string, number>();
		if (out.every((o, i) => typeof o === 'number' || (seen.get(o.en) ?? vs[i]) === vs[i] ? (typeof o !== 'number' && seen.set(o.en, vs[i]), true) : false)) return out;
	}
	return vs.map((v) => num(v, 'n6'));
};
