// Traces that cross a thread (integration 1a carry-over). graph/executor.ts keeps a trace's declared inputs (token ids)
// in a WeakMap keyed by the trace object, because CONTRACTS traces hold layers and sub-steps only. A trace posted to or
// from a worker is a structured clone, so that WeakMap entry is lost and explain('tok_emb', …) could not find the ids.
// portableTrace() folds the declared inputs into trace.t (as float tensors, which explain() reads when the WeakMap has
// nothing), so the clone stays explainable. `order` is unchanged: the extra keys are inputs, not layers.
// On the main thread, keep the original trace object whenever one exists (the runtime does); use this only for a trace
// that must be posted.
import type { AnyTensor, Tensor, Trace } from '../types/core.ts';
import { traceInputs } from '../graph/executor.ts';

function asFloat(t: AnyTensor): Tensor {
	return { shape: t.shape.slice(), data: t.data instanceof Float64Array ? t.data.slice() : Float32Array.from(t.data) };
}

/** A copy-safe trace: the same order and tensors, plus each declared input (unless a layer already uses its name). */
export function portableTrace(trace: Trace, inputs?: Record<string, AnyTensor>): Trace {
	const ins = inputs ?? traceInputs(trace) ?? {};
	const t: Record<string, Tensor> = { ...trace.t };
	for (const [name, x] of Object.entries(ins)) if (!(name in t)) t[name] = asFloat(x);
	return { order: trace.order.slice(), t, ...(trace.meta ? { meta: { ...trace.meta } } : {}) };
}
