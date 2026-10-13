// residual-connections: the numbers every part of the page shares: the deep MLP's size, the two-moons points it learns,
// the training settings, the learner's setup ({ depth, skip }) and small narration helpers. No third-party data: the
// points come from the kit's seeded generator, so every learner gets the same 120 points.
import type { I18n } from '../../atlas/types/index.ts';
import { moons } from '../../atlas/data/points2d.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { num } from '../../atlas/lesson/say.ts';

/** Units in every hidden layer. */
export const WIDTH = 8;
/** The depth slider's range, in linear layers (the first takes the 2 coordinates, the last gives the 2 class scores). */
export const MIN_DEPTH = 3;
export const MAX_DEPTH = 20;
/** The depths the comparison trains, plain and with skips. */
export const DEPTHS = [3, 6, 10, 20];
/** Plain gradient descent on all 120 points at once: learning rate and number of updates. */
export const LR = 0.05;
export const STEPS = 300;
/** The two-moons points (seed 11, the same set as the loss-and-gradient-descent page). */
export const MOONS = moons(120, createRng(11), { noise: 0.15 });

/** The learner's setup: how many linear layers, and skip connections on (1) or off (0). */
export interface Setup {
	depth: number;
	skip: number;
}
export const START: Setup = { depth: MAX_DEPTH, skip: 0 };

/** Any raw input -> a setup: depth a whole number in [MIN_DEPTH, MAX_DEPTH] (default 20), skip 0 or 1 (default 0). */
export function setupOf(raw: unknown): Setup {
	const o = (raw && typeof raw === 'object' ? raw : {}) as { depth?: unknown; skip?: unknown };
	const d = typeof o.depth === 'number' && Number.isFinite(o.depth) ? Math.round(o.depth) : START.depth;
	return { depth: Math.min(MAX_DEPTH, Math.max(MIN_DEPTH, d)), skip: o.skip === 1 || o.skip === true ? 1 : 0 };
}

/** The hidden layer the identity-path steps zoom into: the middle one (always one of the layers that can have a skip,
 *  1 .. depth − 2). */
export const midLayer = (depth: number): number => Math.min(depth - 2, Math.max(1, Math.floor(depth / 2)));

/** Narration helpers: a frame's metrics; a rate as written ("0.1"); on/off words; a legend label. */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
export const rate = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);
export const PLAIN: I18n = { en: 'plain', zh: '无跳跃连接' };
export const SKIPS: I18n = { en: 'with skips', zh: '有跳跃连接' };
export const onOff = (skip: number): I18n => (skip ? { en: 'on', zh: '打开' } : { en: 'off', zh: '关闭' });
/** A ratio as a reader says it: whole numbers from 10 up ("62,930"), else two significant digits ("3.4"). */
export const times = (v: number) => num(v, v >= 10 ? 'int' : 'n2');
