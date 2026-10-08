// K2: process sources that wrap a trainer (the MLP playground's `play`, word2vec's `fit`). A ProcessDef made by
// trainingProcess(trainer), or a wrapper that spreads `...trainingProcess(trainer)` and passes its frames on, carries
// `trainer` (the trainer's id). When its frames are the trainer's frames with the source's options (checked here, as a
// package's claim test would), the trainer's PyTorch twin covers the source too: `atlas parity` runs the twin with the
// source's options and records it under the source's name, so the source's steps get the "= PyTorch fp64" badge. A
// wrapper that adds or changes frames or metrics (the MLP playground's lrPair runs two learning rates in lockstep and
// adds the second run's loss) is not covered and keeps the plain "Live" badge. Pure TS.
import type { ProcessDef } from '../types/process.ts';
import { record } from '../process/record.ts';

/** The trainer id a process wraps (ProcessDef.trainer), or null. */
export function wrappedTrainer(p: ProcessDef<any, any, any> | undefined | null): string | null {
	const t = (p as { trainer?: unknown } | undefined | null)?.trainer;
	return typeof t === 'string' && t.length > 0 ? t : null;
}

/** The training state fields that must agree (trainer.ts TRAIN_SCHEMA); a wrapper may add its own (MLP: 'act'). */
const TRAIN_FIELDS = ['step', 'loss', 'parts', 'lr', 'optimizer', 'params', 'prev', 'grads', 'opt', 'batch', 'acts'];

function same(a: unknown, b: unknown): boolean {
	if (a === b) return true;
	if (typeof a === 'number' && typeof b === 'number') return Number.isNaN(a) && Number.isNaN(b);
	if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
	if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) {
		const x = a as unknown as ArrayLike<number>;
		const y = b as unknown as ArrayLike<number>;
		if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || x.length !== y.length) return false;
		for (let i = 0; i < x.length; i++) if (!(x[i] === y[i] || (Number.isNaN(x[i]) && Number.isNaN(y[i])))) return false;
		return true;
	}
	if (Array.isArray(a) !== Array.isArray(b)) return false;
	const ka = Object.keys(a as object).sort();
	const kb = Object.keys(b as object).sort();
	if (ka.join('\u0001') !== kb.join('\u0001')) return false;
	return ka.every((k) => same((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

export interface SameFrames {
	same: boolean;
	/** frames compared */
	frames: number;
	/** the detail of the trainer run that matched */
	detail?: 'coarse' | 'fine';
	/** the first difference */
	why?: string;
}

/** Do `wrapper`'s frames equal `trainer`'s (the registered 'train:<id>' process) on the same input, options and seed,
 *  over at most `steps` training steps (default 50)? Compared: the frame count; phase, step and sub; every metric (none
 *  added or missing); the training state fields. The wrapper runs in `detail` (its source's); the trainer in both coarse
 *  and fine detail (a wrapper may ask for fine frames through its own options), and either may match. */
export function sameFramesAsTrainer(
	wrapper: ProcessDef<any, any, any>,
	trainer: ProcessDef<any, any, any>,
	input: unknown,
	opt: Record<string, unknown>,
	seed: number,
	o: { detail?: 'coarse' | 'fine'; steps?: number } = {}
): SameFrames {
	const cap = o.steps ?? 50;
	const want = Number(opt.steps ?? (wrapper.defaults as { steps?: number } | undefined)?.steps ?? (trainer.defaults as { steps?: number } | undefined)?.steps ?? cap);
	const run = { ...opt, steps: Math.max(1, Math.min(Number.isFinite(want) ? want : cap, cap)) };
	const maxFrames = 4 * Number(run.steps) + 16;
	let w;
	try {
		w = record(wrapper, input, run, { seed, detail: o.detail ?? 'coarse', maxFrames });
	} catch (e) {
		return { same: false, frames: 0, why: `the process threw: ${e instanceof Error ? e.message : String(e)}` };
	}
	let first: SameFrames | null = null;
	for (const detail of ['coarse', 'fine'] as const) {
		let t;
		try {
			t = record(trainer, input, run, { seed, detail, maxFrames });
		} catch (e) {
			return { same: false, frames: 0, why: `the trainer threw: ${e instanceof Error ? e.message : String(e)}` };
		}
		const r = compareTapes(w, t);
		if (r.same) return { ...r, detail };
		first ??= r;
	}
	return first!;
}

function compareTapes(w: { length: number; frame(i: number): any; state(i: number): any }, t: { length: number; frame(i: number): any; state(i: number): any }): SameFrames {
	if (w.length === 0) return { same: false, frames: 0, why: 'the process made no frames on this input' };
	if (w.length !== t.length) return { same: false, frames: 0, why: `${w.length} frames, the trainer ${t.length}` };
	for (let i = 0; i < w.length; i++) {
		const a = w.frame(i);
		const b = t.frame(i);
		if (a.phase !== b.phase || a.step !== b.step || a.sub !== b.sub) return { same: false, frames: i, why: `frame ${i}: ${a.phase} step ${a.step}, the trainer ${b.phase} step ${b.step}` };
		if (!same(a.metrics ?? {}, b.metrics ?? {})) return { same: false, frames: i, why: `frame ${i} (${a.phase}): other metrics (${Object.keys(a.metrics ?? {}).join(', ')} vs ${Object.keys(b.metrics ?? {}).join(', ')})` };
		const sa = w.state(i) as Record<string, unknown>;
		const sb = t.state(i) as Record<string, unknown>;
		for (const k of TRAIN_FIELDS) if (!same(sa?.[k], sb?.[k])) return { same: false, frames: i, why: `frame ${i} (${a.phase}): state field '${k}' differs` };
	}
	return { same: true, frames: w.length };
}
