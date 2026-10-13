// catastrophic-forgetting: the two tasks, the learner's settings and the narration helpers, shared by the processes, the
// settings widget, the lesson, the concept page and the claim tests. No third-party data: the points are drawn here
// from the kit's seeded rng, so every learner gets the same 60 + 60 points.
//   task A  the left group (x from −1 to −0.2): class 1 above the middle line y = 0, class 0 below it
//   task B  the right group (x from 0.2 to 1): the other way round, class 1 below y = 0, class 0 above it
// One network can get both right (it needs a boundary that bends at x = 0), but B's points alone say nothing about the
// left group, so training on B only is free to overwrite what A taught.
import type { I18n, Tensor } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export interface Task {
	xy: Float64Array;
	label: Int32Array;
	n: number;
}

/** No point closer than this to the middle line y = 0 (a clear gap between the classes). */
export const GAP = 0.1;
/** points per task */
export const N_TASK = 60;

/** Points with x in `xr` and y in [−0.9, 0.9], at least GAP from y = 0, classes alternating (point i has class i % 2). */
function makeTask(seed: number, xr: [number, number], above: 0 | 1): Task {
	const rng = createRng(seed);
	const xy = new Float64Array(2 * N_TASK);
	const label = new Int32Array(N_TASK);
	for (let i = 0; i < N_TASK; ) {
		const x = xr[0] + (xr[1] - xr[0]) * rng.next();
		const y = -0.9 + 1.8 * rng.next();
		if (Math.abs(y) < GAP) continue;
		const c = y > 0 ? above : 1 - above;
		if (c !== i % 2) continue;
		xy[2 * i] = x;
		xy[2 * i + 1] = y;
		label[i] = c;
		i++;
	}
	return { xy, label, n: N_TASK };
}

export const TASK_A = makeTask(1, [-1, -0.2], 1);
export const TASK_B = makeTask(2, [0.2, 1], 0);

/** Both tasks as the plane's points: A's 60 first, then B's 60 (point i < 60 is A's). */
export const ALL_XY: Tensor = { shape: [2 * N_TASK, 2], data: Float64Array.from([...TASK_A.xy, ...TASK_B.xy]) };
export const ALL_LABEL: Tensor = { shape: [2 * N_TASK], data: Float64Array.from([...TASK_A.label, ...TASK_B.label]) };
export const CLASSES: I18n[] = [
	{ en: 'class 0', zh: '第 0 类' },
	{ en: 'class 1', zh: '第 1 类' }
];

/** The learner's settings for the training on task B. */
export interface Settings {
	/** share of each task-B training batch that is stored task-A points, 0 to MAX_REPLAY */
	replay: number;
	/** λ of the pull toward the weights after task A: loss = B's loss + λ/2 · Σ (w − w_A)², 0 to MAX_PULL */
	pull: number;
	/** keep the first layer's weights and biases as task A left them */
	freeze: boolean;
	/** train on B with a learning rate ten times smaller */
	slow: boolean;
}

export const MAX_REPLAY = 0.5;
export const MAX_PULL = 1;
export const PLAIN: Settings = { replay: 0, pull: 0, freeze: false, slow: false };

const num = (v: unknown, d: number, hi: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(0, v)) : d);

/** Any raw input -> settings inside their ranges; anything else is plain fine-tuning (no protection). */
export function settingsOf(raw: unknown): Settings {
	const o = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof Settings, unknown>>;
	return { replay: num(o.replay, 0, MAX_REPLAY), pull: num(o.pull, 0, MAX_PULL), freeze: o.freeze === true, slow: o.slow === true };
}

/** How many stored task-A points join the N_TASK task-B points so that they make up `replay` of the batch. The memory
 *  is task A's first k points (balanced: classes alternate). */
export const replayCount = (replay: number): number => Math.min(N_TASK, Math.round((N_TASK * replay) / (1 - replay)));

/** Narration helpers: a frame's metrics; a setting as set ("0.6", not "0.600"); a share as a percentage. */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};
export const setting = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? String(Math.round(v * 1000) / 1000) : NaN);

/** A way of training on task B as a short name: "Plain fine-tuning", "Replay 10%", "Pull λ = 0.6", "Frozen first layer",
 *  "Learning rate ÷ 10", joined with " + " when several are on. */
export function remedyName(s: Settings): I18n {
	const parts: I18n[] = [];
	if (s.replay > 0) parts.push({ en: `Replay ${Math.round(s.replay * 100)}%`, zh: `回放 ${Math.round(s.replay * 100)}%` });
	if (s.pull > 0) parts.push({ en: `Pull λ = ${setting(s.pull)}`, zh: `拉力 λ = ${setting(s.pull)}` });
	if (s.freeze) parts.push({ en: 'Frozen first layer', zh: '冻结第一层' });
	if (s.slow) parts.push({ en: 'Learning rate ÷ 10', zh: '学习率 ÷ 10' });
	if (!parts.length) return { en: 'Plain fine-tuning', zh: '普通微调' };
	return { en: parts.map((p) => p.en).join(' + '), zh: parts.map((p) => p.zh).join(' + ') };
}
