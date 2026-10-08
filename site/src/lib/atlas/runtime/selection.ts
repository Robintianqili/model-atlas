// The lesson's own selection (design §4.7: steps never touch the explore view's selection) and the clamped cursor.
import type { Tensor } from '../types/core.ts';
import type { UnitPick } from '../types/op.ts';
import type { Cursor, Tape } from '../types/process.ts';
import type { ExplainerRuntime, SelectionState } from '../types/runtime.ts';

export function unflatten(shape: number[], flat: number): number[] {
	const idx = new Array<number>(shape.length);
	for (let d = shape.length - 1; d >= 0; d--) {
		idx[d] = flat % shape[d];
		flat = Math.floor(flat / shape[d]);
	}
	return idx;
}

/** A unit (output index) of `t` chosen by an op's UnitPick. `ch` restricts spatial picks to one channel of [C, H, W].
 *  `axes` (the layer's axes) makes the argmax picks follow the sequence: an attention table [h, tq, tk] is read at the
 *  LAST query's row of every head (or of head `ch`), and a sequence [t, ...] at its last position, where the next-token
 *  prediction is made (R0: the global argmax of a weights table is the trivial 1.0 of the first query). The views'
 *  kit.parts/stage.ts pickUnit follows the same rule. */
export function pickUnit(t: Tensor | undefined, pick: UnitPick | undefined, o?: { ch?: number; axes?: readonly string[] }): number[] | undefined {
	if (!t || t.data.length === 0) return undefined;
	const shape = t.shape;
	const d = t.data;
	if (pick && typeof pick === 'object') return pick.unit.map((v, k) => Math.max(0, Math.min((shape[k] ?? 1) - 1, v)));
	const ax = o?.axes;
	const argPick = pick === undefined || pick === 'argmax' || pick === 'argmaxAbs';
	if (argPick && ax && shape.length === 3 && ax[1] === 'tq') {
		const [H, Tq, Tk] = shape;
		const heads = o?.ch !== undefined ? [Math.max(0, Math.min(H - 1, o.ch))] : Array.from({ length: H }, (_, h) => h);
		const score = (v: number) => (!Number.isFinite(v) ? -Infinity : pick === 'argmaxAbs' ? Math.abs(v) : v);
		let b = -1;
		for (const h of heads)
			for (let k = 0; k < Tk; k++) {
				const i = (h * Tq + Tq - 1) * Tk + k;
				if (b < 0 || score(d[i]) > score(d[b])) b = i;
			}
		return unflatten(shape, b);
	}
	let from = 0;
	let to = d.length;
	if (argPick && ax && shape.length === 2 && ax[0] === 't') from = d.length - shape[1]; // the last position
	else if (o?.ch !== undefined && shape.length >= 3) {
		const per = d.length / shape[0];
		const ch = Math.max(0, Math.min(shape[0] - 1, o.ch));
		from = ch * per;
		to = from + per;
	}
	const best = (score: (v: number) => number) => {
		let b = from;
		for (let i = from + 1; i < to; i++) if (score(d[i]) > score(d[b])) b = i;
		return unflatten(shape, b);
	};
	switch (pick ?? 'argmax') {
		case 'argmin':
			return best((v) => -v);
		case 'argmaxAbs':
			return best((v) => Math.abs(v));
		case 'first':
			return unflatten(shape, from);
		case 'last':
			return unflatten(shape, to - 1);
		case 'center': {
			const u = shape.map((n) => Math.floor(n / 2));
			if (o?.ch !== undefined && shape.length >= 3) u[0] = Math.max(0, Math.min(shape[0] - 1, o.ch));
			return u;
		}
		case 'strongestChannel': {
			if (shape.length < 2) return best((v) => v);
			let b = 0;
			for (let i = 1; i < d.length; i++) if (d[i] > d[b]) b = i;
			return unflatten(shape, b);
		}
		default:
			return best((v) => v);
	}
}

/** Clamp a frame index into a tape (a new, shorter input never crashes a step). -1 when the tape is empty. */
export function clampIndex(tape: Tape | null | undefined, i: number): number {
	if (!tape || tape.length === 0) return -1;
	if (!Number.isFinite(i)) return tape.length - 1;
	return Math.max(0, Math.min(tape.length - 1, Math.floor(i)));
}

/** find() with clamping: a cursor that is not recorded falls back to the nearest sensible frame (numbers clamp; queries
 *  fall back to the last frame). Returns { index, found }. */
export function resolveCursor(tape: Tape | null | undefined, c: Cursor): { index: number; found: boolean } {
	if (!tape || tape.length === 0) return { index: -1, found: false };
	const i = tape.find(c);
	if (i >= 0) return { index: i, found: true };
	return { index: typeof c === 'number' ? clampIndex(tape, c) : tape.length - 1, found: false };
}

/** Replace the lesson selection (a new object, never a mutation: the browser wraps the runtime in $state.raw). */
export function setSelection(rt: ExplainerRuntime, sel: Partial<SelectionState>): SelectionState {
	const next: SelectionState = {};
	for (const [k, v] of Object.entries(sel)) if (v !== undefined) next[k] = v;
	rt.sel = next;
	return next;
}
