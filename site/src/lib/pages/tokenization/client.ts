/**
 * Main-thread side of tok.worker.ts: one worker per page, promise-based requests.
 * Create it in onMount (never during prerendering).
 */
import type { TrainTrace } from './train.ts';
import type { RealResult } from './realtok.ts';
import type { RealId } from './tokenizers.ts';

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

export class TokClient {
	worker: Worker;
	next = 1;
	pending = new Map<number, Pending>();

	constructor() {
		this.worker = new Worker(new URL('./tok.worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (e: MessageEvent) => {
			const p = this.pending.get(e.data.req);
			if (!p) return;
			this.pending.delete(e.data.req);
			if (e.data.ok) p.resolve(e.data);
			else p.reject(new Error(e.data.error));
		};
		this.worker.onerror = (e) => {
			for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker error'));
			this.pending.clear();
		};
	}

	_send<T>(msg: Record<string, unknown>): Promise<T> {
		const req = this.next++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(req, { resolve, reject });
			this.worker.postMessage({ ...msg, req });
		});
	}

	train(text: string, maxMerges: number, pattern: string, tinyMerges: [number, number][]) {
		return this._send<{ trace: TrainTrace; ms: number }>({ type: 'train', text, maxMerges, pattern, tinyMerges });
	}

	load(id: RealId, url: string) {
		return this._send<{ id: RealId; ms: number }>({ type: 'load', id, url });
	}

	encode(text: string, ids: RealId[]) {
		return this._send<{ results: Record<string, RealResult>[]; ms: Record<string, number> }>({
			type: 'encode',
			texts: [text],
			ids
		}).then((r) => ({ results: r.results[0] as Record<string, RealResult>, ms: r.ms }));
	}

	encodeMany(texts: string[], ids: RealId[]) {
		return this._send<{ results: Record<string, RealResult>[] }>({ type: 'encode', texts, ids }).then((r) => r.results);
	}

	/** [token count, [UNK] count] per text and tokenizer */
	counts(texts: string[], ids: RealId[]) {
		return this._send<{ results: Record<string, [number, number]>[] }>({ type: 'encode', texts, ids, counts: true }).then((r) => r.results);
	}

	destroy() {
		this.worker.terminate();
		for (const p of this.pending.values()) p.reject(new Error('closed'));
		this.pending.clear();
	}
}
