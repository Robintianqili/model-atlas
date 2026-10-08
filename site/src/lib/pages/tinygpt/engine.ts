/**
 * Loads TinyGPT for the model page and runs it in a Web Worker (falls back to the main thread if
 * Workers are unavailable). Imported lazily from onMount, so none of this is in the page shell.
 *
 *   const eng = await loadEngine(absoluteModelUrl);
 *   const r = await eng.run(ids);            // RunResult: trace + logit lens + timings
 *   const g = await eng.generate(ids, opts); // tinygpt.ts generate() (same PRNG as Python)
 *
 * Files fetched: spec.json, weights.bin (fp32) and tokenizer.json — never the checkpoints.
 */
import { fromWeights, type GenerateOptions, type GenerateResult, type TinyGPT, type TinyGPTSpec } from '../../models/tinygpt';
import { BPETokenizer, type TokenizerJSON } from '../../models/bpe';
import { runAll, type RunResult, type WorkerRequest, type WorkerResponse } from './compute';

export type { RunResult };
export { samplingDistribution, mulberry32 } from '../../models/tinygpt';

type Waiter = { resolve: (v: unknown) => void; reject: (e: Error) => void };

export type Engine = {
	/** main-thread copy of the model (weights for the detail views; fallback runner) */
	model: TinyGPT;
	tok: BPETokenizer;
	spec: TinyGPTSpec;
	/** bytes actually downloaded */
	bytes: number;
	where: () => 'worker' | 'main';
	run(ids: number[]): Promise<RunResult>;
	generate(ids: number[], opts: GenerateOptions): Promise<{ result: GenerateResult; ms: number }>;
	dispose(): void;
};

async function get(url: string): Promise<Response> {
	const r = await fetch(url);
	if (!r.ok) throw new Error(`could not load ${url} (HTTP ${r.status})`);
	return r;
}

export async function loadEngine(modelUrl: string): Promise<Engine> {
	const u = modelUrl.replace(/\/+$/, '') + '/';
	const [specText, weights, tokText] = await Promise.all([
		get(u + 'spec.json').then((r) => r.text()),
		get(u + 'weights.bin').then((r) => r.arrayBuffer()),
		get(u + 'tokenizer.json').then((r) => r.text())
	]);
	const spec = JSON.parse(specText) as TinyGPTSpec;
	const model = fromWeights(spec, spec.tensors, new Float32Array(weights));
	const tok = new BPETokenizer(JSON.parse(tokText) as TokenizerJSON);
	const bytes = new TextEncoder().encode(specText).length + weights.byteLength + new TextEncoder().encode(tokText).length;

	let worker: Worker | null = null;
	let seq = 0;
	const waiting = new Map<number, Waiter>();

	function failAll(msg: string) {
		for (const w of waiting.values()) w.reject(new Error(msg));
		waiting.clear();
	}

	try {
		const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		const ready = new Promise<void>((resolve, reject) => {
			const timer = setTimeout(() => reject(new Error('worker timeout')), 10000);
			w.onmessage = (e: MessageEvent<WorkerResponse>) => {
				if (e.data.type === 'ready') {
					clearTimeout(timer);
					resolve();
				}
			};
			w.onerror = (e) => {
				clearTimeout(timer);
				reject(new Error(e.message || 'worker error'));
			};
		});
		const init: WorkerRequest = { type: 'init', spec, weights: weights.slice(0) };
		w.postMessage(init, [init.weights]);
		await ready;
		w.onmessage = (e: MessageEvent<WorkerResponse>) => {
			const m = e.data;
			if (m.type === 'ready') return;
			const wt = waiting.get(m.id);
			if (!wt) return;
			waiting.delete(m.id);
			if (m.type === 'error') wt.reject(new Error(m.message));
			else if (m.type === 'run') wt.resolve({ ...m.result, where: 'worker' } satisfies RunResult);
			else wt.resolve({ result: m.result, ms: m.ms });
		};
		w.onerror = (e) => {
			// the worker died: answer everything still waiting and use the main thread from now on
			failAll(e.message || 'worker error');
			worker?.terminate();
			worker = null;
		};
		worker = w;
	} catch {
		worker = null; // main-thread fallback
	}

	function ask<T>(req: WorkerRequest & { id: number }): Promise<T> {
		return new Promise<T>((resolve, reject) => {
			waiting.set(req.id, { resolve: resolve as (v: unknown) => void, reject });
			worker!.postMessage(req);
		});
	}

	return {
		model,
		tok,
		spec,
		bytes,
		where: () => (worker ? 'worker' : 'main'),
		async run(ids) {
			if (worker) {
				try {
					return await ask<RunResult>({ type: 'run', id: ++seq, ids });
				} catch {
					/* fall through to the main thread */
				}
			}
			return { ...runAll(model, ids), where: 'main' };
		},
		async generate(ids, opts) {
			if (worker) {
				try {
					return await ask<{ result: GenerateResult; ms: number }>({ type: 'generate', id: ++seq, ids, opts });
				} catch {
					/* fall through */
				}
			}
			const t0 = performance.now();
			const result = model.generate(ids, opts);
			return { result, ms: performance.now() - t0 };
		},
		dispose() {
			failAll('disposed');
			worker?.terminate();
			worker = null;
		}
	};
}
