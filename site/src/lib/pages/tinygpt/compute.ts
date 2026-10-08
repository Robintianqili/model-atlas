/**
 * What one "run" computes, shared by the Web Worker (worker.ts) and the main-thread fallback
 * (engine.ts): the full traced forward pass of tinygpt.ts plus the logit lens.
 * Pure TS, no DOM.
 */
import type { ForwardResult, GenerateOptions, GenerateResult, TinyGPT, TinyGPTSpec } from '../../models/tinygpt';
import { lensBuffers, lensPoints, logitLens, type Lens } from './lens';

export type RunResult = {
	/** the traced forward pass (66 tensors) on the last ≤ 64 ids */
	forward: ForwardResult;
	lens: Lens;
	/** measured milliseconds for the forward pass and for the logit lens */
	ms: number;
	lensMs: number;
	/** where it ran */
	where: 'worker' | 'main';
};

export type WorkerRequest =
	| { type: 'init'; spec: TinyGPTSpec; weights: ArrayBuffer }
	| { type: 'run'; id: number; ids: number[] }
	| { type: 'generate'; id: number; ids: number[]; opts: GenerateOptions };

export type WorkerResponse =
	| { type: 'ready' }
	| { type: 'run'; id: number; result: Omit<RunResult, 'where'> }
	| { type: 'generate'; id: number; result: GenerateResult; ms: number }
	| { type: 'error'; id: number; message: string };

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export function computeLens(model: TinyGPT, f: ForwardResult): Lens {
	const T = f.ids.length;
	const ids = lensPoints(model.nLayer);
	return logitLens({
		T,
		D: model.dModel,
		V: model.vocabSize,
		eps: model.spec.lnEps ?? 1e-5,
		points: ids.map((id) => ({ id, data: f.t[id].data })),
		lnW: model.weight('ln_f.weight').data,
		lnB: model.weight('ln_f.bias').data,
		emb: model.weight('tok_emb.weight').data
	});
}

export function runAll(model: TinyGPT, ids: number[]): Omit<RunResult, 'where'> {
	const t0 = now();
	const forward = model.forward(ids);
	const t1 = now();
	const lens = computeLens(model, forward);
	const t2 = now();
	return { forward, lens, ms: t1 - t0, lensMs: t2 - t1 };
}

/** Every buffer of a run, for a zero-copy postMessage. */
export function runBuffers(r: Omit<RunResult, 'where'>): ArrayBuffer[] {
	const out: ArrayBuffer[] = [];
	const seen = new Set<ArrayBuffer>();
	for (const k of r.forward.order) {
		const b = r.forward.t[k].data.buffer as ArrayBuffer;
		if (!seen.has(b)) {
			seen.add(b);
			out.push(b);
		}
	}
	for (const b of lensBuffers(r.lens)) if (!seen.has(b)) (seen.add(b), out.push(b));
	return out;
}
