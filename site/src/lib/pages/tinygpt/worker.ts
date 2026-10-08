/**
 * TinyGPT Web Worker: keeps typing smooth by running the forward pass (≈ 20–40 ms at 64 tokens),
 * the logit lens and generation off the main thread. Results come back as transferred buffers.
 */
import { fromWeights, type TinyGPT } from '../../models/tinygpt';
import { runAll, runBuffers, type WorkerRequest, type WorkerResponse } from './compute';

let model: TinyGPT | null = null;

// typed loosely so this file does not need the "webworker" lib (it clashes with "dom" in svelte-check)
const ctx = self as unknown as {
	postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
	onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
};

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
	ctx.postMessage(msg, transfer);
}

ctx.onmessage = (e: MessageEvent<WorkerRequest>) => {
	const m = e.data;
	try {
		if (m.type === 'init') {
			model = fromWeights(m.spec, m.spec.tensors, new Float32Array(m.weights));
			post({ type: 'ready' });
			return;
		}
		if (!model) throw new Error('model not loaded');
		if (m.type === 'run') {
			const r = runAll(model, m.ids);
			post({ type: 'run', id: m.id, result: r }, runBuffers(r));
		} else if (m.type === 'generate') {
			const t0 = performance.now();
			const result = model.generate(m.ids, m.opts);
			post({ type: 'generate', id: m.id, result, ms: performance.now() - t0 });
		}
	} catch (err) {
		post({ type: 'error', id: 'id' in m ? m.id : -1, message: err instanceof Error ? err.message : String(err) });
	}
};
