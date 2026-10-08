// From a step to what its view needs: the lesson selection, the cursor, the frame and the unit (At).
// LessonPlayer (W1-D1), lint and the Node dry run all use stepAt(), so they agree on every number.
import type { Provenance, Trace } from '../types/core.ts';
import type { GraphModel, UnitExplanation } from '../types/op.ts';
import type { Cursor, CursorQuery, Frame } from '../types/process.ts';
import type { At, CompiledStep } from '../types/lesson.ts';
import type { ExplainerRuntime, SelectionState, SourceState } from '../types/runtime.ts';
import { explainAt, type FactCtx } from '../lesson/facts.ts';
import { resolveCursor, setSelection } from './selection.ts';

function ctxOf(rt: ExplainerRuntime): FactCtx {
	const memo = (rt as { explainMemo?: FactCtx['explainMemo'] }).explainMemo;
	return { spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel, explainMemo: memo };
}

/** The At for frame `index` of `source`. Trace sources: at.layer is the frame's layer and at.unit is explain() at the
 *  lesson selection when the selection is on that layer. Scan sources: the frame's own unit explanation. */
export function atFor(rt: ExplainerRuntime, source: string, index: number): At {
	const st: SourceState | undefined = rt.sources[source];
	const tape = st?.tape ?? undefined;
	if (!tape || index < 0 || index >= tape.length) return { index: -1, tape: tape ?? undefined };
	const frame = tape.frame(index);
	const state = tape.state(index) as Record<string, unknown>;
	const at: At = { index, frame, state, tape };
	if (st.kind === 'scan') {
		at.layer = state.layer as string;
		at.unit = state.explain as UnitExplanation | undefined;
	} else if (st.kind === 'trace' || st.kind === 'live-model') {
		const layer = state.layer as string | undefined;
		at.layer = layer;
		if (layer && rt.sel.layer === layer && rt.sel.unit) at.unit = explainAt(ctxOf(rt), st.trace as Trace, layer, rt.sel.unit, (st.model as GraphModel | null) ?? rt.model);
	}
	return at;
}

export interface StepPosition {
	at: At;
	cursor: Cursor | null;
	/** the cursor matched a recorded frame (false: the index was clamped, lint AK404) */
	found: boolean;
	index: number;
	/** the source did not run because the input is empty (the stage shows the adapter's empty note) */
	skipped: boolean;
	error?: string;
}

/** Enter a step: replace the lesson selection with step.select(rt), resolve the cursor, run the source as far as needed
 *  and build the At. A step's `compare` sources run to the same cursor too (K0: they used to show only what an earlier
 *  step had run). Never throws: problems come back in `error`. */
export async function stepAt(rt: ExplainerRuntime, step: Pick<CompiledStep, 'select' | 'at' | 'source'> & Partial<Pick<CompiledStep, 'compare'>>): Promise<StepPosition> {
	let cursor: Cursor | null = null;
	try {
		setSelection(rt, {});
		const sel: Partial<SelectionState> = step.select?.(rt) ?? {};
		setSelection(rt, sel);
		cursor = step.at(rt);
		await rt.ensure(step.source, cursor);
		for (const s of step.compare ? [step.compare.a, step.compare.b] : []) {
			if (s === step.source) continue;
			try {
				await rt.ensure(s, cursor);
			} catch {
				/* the compared side shows its own error; the step itself still works */
			}
		}
	} catch (e) {
		return { at: { index: -1 }, cursor, found: false, index: -1, skipped: false, error: e instanceof Error ? e.message : String(e) };
	}
	const st = rt.sources[step.source];
	if (!st?.tape || st.tape.length === 0) {
		if (rt.pre?.empty) return { at: { index: -1, skipped: true }, cursor, found: false, index: -1, skipped: true };
		return { at: { index: -1 }, cursor, found: false, index: -1, skipped: false, error: `source '${step.source}' has no frames (status ${st?.status ?? 'missing'}${st?.error ? `: ${st.error}` : ''})` };
	}
	const r = resolveCursor(st.tape, cursor);
	return { at: atFor(rt, step.source, r.index), cursor, found: r.found, index: r.index, skipped: false };
}

/** The frame of `source` that matches a step's moment on another source: the same cursor (a step at 'last' compares the
 *  ends of two runs of different length), or, while a range plays, the frame with the same phase, step and sub. -1 when
 *  the source has no frames. Used for a step's `compare` (K0: it used to align the two sources by frame index). */
export function alignedIndex(rt: ExplainerRuntime, source: string, cursor: Cursor | null, frame?: Frame): number {
	const tape = rt.sources[source]?.tape;
	if (!tape || tape.length === 0) return -1;
	if (frame) {
		const q: CursorQuery = { phase: frame.phase };
		if (frame.step !== undefined) q.step = frame.step;
		if (frame.sub !== undefined) q.sub = frame.sub;
		const i = tape.find(q);
		return i >= 0 ? i : tape.length - 1;
	}
	return resolveCursor(tape, cursor ?? 'last').index;
}

/** The badge a step should show: a 'live' step on a source that fell back to a precomputed tape says 'precomputed'. */
export function effectiveProvenance(rt: ExplainerRuntime, step: Pick<CompiledStep, 'provenance' | 'source'>): { mode: Provenance; note?: CompiledStep['provenance']['note'] } {
	const src = rt.sources[step.source]?.provenance;
	if (step.provenance.mode === 'live' && src && src !== 'live') return { mode: src, note: step.provenance.note };
	return step.provenance;
}
