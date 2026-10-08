/// <reference lib="webworker" />
// Computes TinyCNN flatten features (256 numbers per drawing) off the main thread.
// Messages in:  { type: 'init', spec, weights: ArrayBuffer }
//               { type: 'features', id, inputs: Float32Array[] }
// Messages out: { type: 'ready' } | { type: 'features', id, out: Float32Array[] } | { type: 'error', id?, message }
import { fromParts, type TinyCNN, type TinyCNNSpec } from '../../models/tinycnn';

let model: TinyCNN | null = null;

self.onmessage = (e: MessageEvent) => {
	const msg = e.data as
		| { type: 'init'; spec: TinyCNNSpec; weights: ArrayBuffer }
		| { type: 'features'; id: number; inputs: Float32Array[] };
	try {
		if (msg.type === 'init') {
			model = fromParts(msg.spec, msg.weights);
			(self as unknown as Worker).postMessage({ type: 'ready' });
		} else if (msg.type === 'features') {
			if (!model) throw new Error('worker: model not loaded');
			const out = msg.inputs.map((x) => model!.features(x));
			(self as unknown as Worker).postMessage(
				{ type: 'features', id: msg.id, out },
				out.map((f) => f.buffer as ArrayBuffer)
			);
		}
	} catch (err) {
		(self as unknown as Worker).postMessage({
			type: 'error',
			id: 'id' in msg ? msg.id : undefined,
			message: err instanceof Error ? err.message : String(err)
		});
	}
};
