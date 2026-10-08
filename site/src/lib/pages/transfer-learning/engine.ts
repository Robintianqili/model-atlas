/**
 * Loads TinyCNN once for the page and computes features for many drawings without blocking the UI:
 * in a Web Worker when available, otherwise on the main thread in small chunks that yield between
 * them. Single drawings (the pad, a newly added example) use the main-thread model directly.
 */
import type { TinyCNN, TinyCNNSpec } from '$lib/models/tinycnn';

export type TinyCNNLib = typeof import('$lib/models/tinycnn');

export interface Engine {
	model: TinyCNN;
	/** the model module (preprocessStrokes, …) */
	lib: TinyCNNLib;
	spec: TinyCNNSpec;
	/** features of many inputs (worker if possible) */
	featuresMany(inputs: Float32Array[]): Promise<Float32Array[]>;
	/** 'worker' or 'main' (fallback), for the notes/tests */
	where(): 'worker' | 'main';
	dispose(): void;
}

let shared: Promise<Engine> | null = null;

/** One engine per page load (the hook and Part 1 share it). */
export function getEngine(modelUrl: string): Promise<Engine> {
	if (!shared) shared = createEngine(modelUrl);
	return shared;
}

async function createEngine(modelUrl: string): Promise<Engine> {
	const mod = await import('$lib/models/tinycnn');
	const [specRes, wRes] = await Promise.all([fetch(`${modelUrl}/spec.json`), fetch(`${modelUrl}/weights.bin`)]);
	if (!specRes.ok || !wRes.ok) throw new Error('tinycnn: could not load spec.json / weights.bin');
	const spec = (await specRes.json()) as TinyCNNSpec;
	const buf = await wRes.arrayBuffer();
	const model = mod.fromParts(spec, buf);

	let worker: Worker | null = null;
	let ready: Promise<boolean> = Promise.resolve(false);
	const pending = new Map<number, { resolve: (v: Float32Array[]) => void; reject: (e: Error) => void }>();
	let nextId = 1;
	try {
		worker = new Worker(new URL('./features.worker.ts', import.meta.url), { type: 'module' });
		ready = new Promise<boolean>((resolve) => {
			const w = worker!;
			const timer = setTimeout(() => resolve(false), 8000);
			w.addEventListener('message', (e: MessageEvent) => {
				const m = e.data;
				if (m.type === 'ready') {
					clearTimeout(timer);
					resolve(true);
				} else if (m.type === 'features') {
					pending.get(m.id)?.resolve(m.out);
					pending.delete(m.id);
				} else if (m.type === 'error') {
					if (m.id !== undefined) {
						pending.get(m.id)?.reject(new Error(m.message));
						pending.delete(m.id);
					} else {
						clearTimeout(timer);
						resolve(false);
					}
				}
			});
			w.addEventListener('error', () => {
				clearTimeout(timer);
				resolve(false);
			});
			w.postMessage({ type: 'init', spec, weights: buf.slice(0) });
		});
	} catch {
		worker = null;
	}
	let useWorker = await ready;
	if (!useWorker && worker) {
		worker.terminate();
		worker = null;
	}

	async function mainThread(inputs: Float32Array[]): Promise<Float32Array[]> {
		const out: Float32Array[] = [];
		for (let i = 0; i < inputs.length; i++) {
			out.push(model.features(inputs[i]));
			// yield every few drawings so the page stays responsive (~1.5 ms each)
			if (i % 8 === 7) await new Promise((r) => setTimeout(r, 0));
		}
		return out;
	}

	return {
		model,
		lib: mod,
		spec,
		where: () => (useWorker ? 'worker' : 'main'),
		featuresMany(inputs) {
			if (!useWorker || !worker) return mainThread(inputs);
			const id = nextId++;
			return new Promise<Float32Array[]>((resolve, reject) => {
				pending.set(id, { resolve, reject });
				// copies (the caller keeps its inputs)
				worker!.postMessage({ type: 'features', id, inputs: inputs.map((x) => Float32Array.from(x)) });
			}).catch(() => {
				useWorker = false;
				return mainThread(inputs);
			});
		},
		dispose() {
			worker?.terminate();
			worker = null;
			shared = null;
		}
	};
}
