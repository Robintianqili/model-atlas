// regularization: the knobs, the data and the plain numbers every part of the package shares (process, adapter, views,
// lesson, claim tests). The data: POOL points of the two-moons set (atlas/data/points2d.ts, seed DATA_SEED), every
// learner the same points; the noise knob scales the same random draws, so moving it moves the points smoothly. The
// train ratio puts the first share of the pool in the training set (the pool alternates classes, so both sets are
// balanced) and keeps the rest as test points. No third-party data.
import { moons, type Points2D } from '../../atlas/data/points2d.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const POOL = 200;
export const DATA_SEED = 7;
/** training: Adam at LR, all training points every step, STEPS updates, a check (both losses, early stopping) every EVERY */
export const STEPS = 600;
export const LR = 0.01;
export const EVERY = 10;
/** early stopping: stop after PATIENCE checks without a new lowest test loss */
export const PATIENCE = 10;

/** The knob values the widget offers (slider stops). */
export const NOISES = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
export const RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
export const WIDTHS = [2, 4, 8, 16, 32];
export const LAMBDAS = [0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1];
export const DROPOUTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

export interface Knobs {
	/** the spread of the points around the two moons */
	noise: number;
	/** the share of the pool used for training */
	ratio: number;
	/** units in each of the two hidden layers */
	width: number;
	/** λ, the weight of the L2 penalty λ·Σw² */
	l2: number;
	/** p, the chance that dropout switches a hidden unit off for one update */
	dropout: number;
}

/** The starting knobs: a big network, few noisy points, no regularization (it overfits). */
export const START: Knobs = { noise: 0.3, ratio: 0.2, width: 32, l2: 0, dropout: 0 };
/** The task's goal: a final test loss below this (the start ends far above it). */
export const TARGET = 0.3;

const nearest = (v: unknown, list: number[], d: number) => {
	if (typeof v !== 'number' || !Number.isFinite(v)) return d;
	let best = list[0];
	for (const x of list) if (Math.abs(x - v) < Math.abs(best - v)) best = x;
	return best;
};

/** Any raw input -> knobs on the widget's stops; anything else is START. */
export function knobsOf(raw: unknown): Knobs {
	const o = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof Knobs, unknown>>;
	return {
		noise: nearest(o.noise, NOISES, START.noise),
		ratio: nearest(o.ratio, RATIOS, START.ratio),
		width: nearest(o.width, WIDTHS, START.width),
		l2: nearest(o.l2, LAMBDAS, START.l2),
		dropout: nearest(o.dropout, DROPOUTS, START.dropout)
	};
}

export interface Split {
	train: Points2D;
	test: Points2D;
}

const part = (p: Points2D, from: number, to: number, name: string): Points2D => ({
	name,
	xy: p.xy.slice(2 * from, 2 * to),
	label: p.label.slice(from, to),
	n: to - from,
	classes: 2
});

const cache = new Map<string, Split>();
/** The training and test points for a noise level and a train ratio (cached). */
export function splitOf(noise: number, ratio: number): Split {
	const key = `${noise}|${ratio}`;
	let s = cache.get(key);
	if (!s) {
		const pool = moons(POOL, createRng(DATA_SEED), { noise });
		const nTrain = 2 * Math.round((ratio * POOL) / 2);
		s = { train: part(pool, 0, nTrain, 'train'), test: part(pool, nTrain, POOL, 'test') };
		cache.set(key, s);
	}
	return s;
}

/** Weights and biases of the 2 -> width -> width -> 2 network. */
export const paramCountOf = (width: number) => 2 * width + width + width * width + width + width * 2 + 2;

/** Narration helpers: a frame's metrics; a setting as it was set (0.003, not 0.00300; NaN until known). */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
export const setting = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);

/** A loss as the narration prints it: 3 significant digits as plain decimals down to 0.00001 ("0.000584", not
 *  "5.84e−4": easier in a first lesson; as mlp-playground's dec); a non-finite value stays NaN, so lint reports it. */
export const dec = (v: number | undefined): string | number => {
	if (typeof v !== 'number' || !Number.isFinite(v)) return NaN;
	if (v !== 0 && Math.abs(v) < 1e-3 && Math.abs(v) >= 1e-5) return v.toPrecision(3).replace('-', '−');
	return v;
};

/** The metric suffixes of the six runs of 'methods' (plain, then process.ts VARIANTS). */
export const SUFFIXES = ['', 'L2', 'Drop', 'Aug', 'Heavy', 'Small'];
/** The fewest test points any of the six runs gets wrong at a 'methods' frame. */
export const bestMiss = (m: Record<string, number>): number => {
	let best = Infinity;
	for (const s of SUFFIXES) if (Number.isFinite(m['testAcc' + s])) best = Math.min(best, Math.round((1 - m['testAcc' + s]) * m.nTest));
	return Number.isFinite(best) ? best : NaN;
};
