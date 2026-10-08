// The input of a live-model trace ("trace one point", design §5.2 motif train:predict; carry-over from integration 1a).
//
// DECISION (W1-D1): the runtime takes the traced example from the LESSON SELECTION, not from the adapter.
//   - The data set stays the learner's raw input: a points2d adapter keeps exposing the whole set ([P, 2] points, labels).
//     Choosing another point is a selection change (a lesson step's select(), a task, a click on a plane2d view); it
//     must not go through setInput(), which would re-preprocess the data and send { type: 'input' } to the train source
//     (a new training segment).
//   - liveModelInput(spec, pre, sel) builds the feed for the live spec's declared inputs:
//       1. a preprocessed tensor with the declared input's name and exactly its shape is used as is (a model whose
//          adapter already gives one example, e.g. a drawing);
//       2. otherwise the first preprocessed tensor whose trailing shape equals the declared shape is read as a data set
//          [P, ...shape] and row r = sel.p ?? sel.point ?? sel.unit[0] ?? 0 (clamped) is taken.
//     It returns the row it used, so views can highlight that point.
//   - The live-model source re-traces when the selected row (or the train tape) changes, like a scan re-keys on its unit.
// Pure TS (no Svelte), so Node tests and, after integration, runtime/runtime.ts can use it: runtime/ may not import ui/,
// so the integrator moves this file to runtime/liveinput.ts and calls it in the live-model branch of ready() (see
// notes/devkit-ui.md, "live-model input").
import type { AnyTensor, Tensor } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { Preprocessed, SelectionState } from '../types/runtime.ts';

/** The row a selection points at (sel.p, sel.point, else the first entry of sel.unit), or null. */
export function selectedRow(sel: SelectionState): number | null {
	for (const v of [sel.p, sel.point, sel.unit?.[0]]) if (typeof v === 'number' && Number.isFinite(v)) return Math.floor(v);
	return null;
}

function numericShape(shape: readonly (number | string)[]): number[] | null {
	return shape.every((d) => typeof d === 'number') ? (shape as number[]) : null;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
	return a.length === b.length && a.every((v, i) => v === b[i]);
}

export interface LiveInput {
	feed: Record<string, AnyTensor>;
	/** the data-set row traced, when the input came from a data set */
	row: number | null;
	/** declared inputs that could not be fed */
	missing: string[];
}

export function liveModelInput(spec: AtlasSpec, pre: Preprocessed | null, sel: SelectionState): LiveInput {
	const feed: Record<string, AnyTensor> = {};
	const missing: string[] = [];
	let row: number | null = null;
	const tensors = pre?.tensors ?? {};
	for (const d of spec.inputs ?? []) {
		const want = numericShape(d.shape);
		const own = tensors[d.name];
		if (own && (!want || sameShape(own.shape, want))) {
			feed[d.name] = own;
			continue;
		}
		if (!want) {
			missing.push(d.name);
			continue;
		}
		const set = Object.values(tensors).find((t) => t.shape.length === want.length + 1 && sameShape(t.shape.slice(1), want) && t.shape[0] > 0);
		if (!set) {
			missing.push(d.name);
			continue;
		}
		const n = set.shape[0];
		const per = set.data.length / n;
		const r = Math.max(0, Math.min(n - 1, selectedRow(sel) ?? 0));
		row = r;
		const data = set.data.slice(r * per, (r + 1) * per);
		feed[d.name] = 'dtype' in set ? { shape: want.slice(), data: data as Int32Array, dtype: 'i32' } : ({ shape: want.slice(), data } as Tensor);
	}
	return { feed, row, missing };
}
