/**
 * Main-thread client for the page's compute engine. Starts the Web Worker (ntp.worker.ts); if a
 * worker cannot be created or fails to start, the same engine runs on the main thread instead
 * (slower, but everything still works).
 *
 *   const c = new NtpClient(absoluteModelUrl, (status) => …);
 *   const r = await c.score('final', ids, 'surprisal');   // null if a newer request on the channel replaced it
 */
import type {
	GenOptions,
	GenResult,
	GenStep,
	LoadStatus,
	NextResult,
	Reply,
	Request,
	ScoreResult,
	Which
} from './protocol.ts';

type Pending = { msg: Request; resolve: (v: unknown) => void; reject: (e: Error) => void; onStep?: (s: GenStep) => void };
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
type Call = DistOmit<Extract<Request, { req: number }>, 'req'>;

export class NtpClient {
	private worker: Worker | null = null;
	private local: ((m: Request) => Promise<void>) | null = null;
	private pending = new Map<number, Pending>();
	private seq = 1;
	private initMsg: Request;
	private heard = false;
	mode: 'worker' | 'main-thread' = 'worker';

	constructor(baseUrl: string, onStatus: (s: LoadStatus) => void) {
		this.initMsg = { type: 'init', baseUrl };
		this.onStatus = onStatus;
		try {
			const w = new Worker(new URL('./ntp.worker.ts', import.meta.url), { type: 'module' });
			w.onmessage = (e: MessageEvent<Reply>) => {
				this.heard = true;
				this.receive(e.data);
			};
			// a worker that cannot start (blocked, unsupported) -> run on the main thread instead
			w.onerror = (e) => {
				if (!this.heard) {
					e.preventDefault?.();
					this.fallback();
				}
			};
			this.worker = w;
			w.postMessage(this.initMsg);
		} catch {
			this.fallback();
		}
	}

	private onStatus: (s: LoadStatus) => void;

	private receive(r: Reply) {
		if (r.type === 'status') {
			this.onStatus(r.status);
			return;
		}
		const p = this.pending.get(r.req);
		if (!p) return;
		if (r.type === 'step') {
			p.onStep?.(r.data);
			return;
		}
		this.pending.delete(r.req);
		if (r.type === 'result') p.resolve(r.data);
		else if (r.type === 'cancelled') p.resolve(null);
		else p.reject(new Error(r.message));
	}

	private fallback() {
		if (this.mode === 'main-thread') return;
		this.worker?.terminate();
		this.worker = null;
		this.mode = 'main-thread';
		// the engine (and the model code) is only loaded on the main thread when the worker failed
		void import('./engine.ts').then(({ createHost }) => {
			this.local = createHost((r) => queueMicrotask(() => this.receive(r)), fetch.bind(globalThis));
			this.post(this.initMsg);
			for (const p of this.pending.values()) this.post(p.msg);
		});
	}

	private post(m: Request) {
		if (this.worker) this.worker.postMessage(m);
		else if (this.local) {
			const h = this.local;
			setTimeout(() => void h(m), 0);
		}
		// else: the fallback engine is still loading; pending requests are re-sent once it is ready
	}

	private call<T>(m: Call, onStep?: (s: GenStep) => void): Promise<T | null> {
		const req = this.seq++;
		const msg = { ...m, req } as Request;
		return new Promise<T | null>((resolve, reject) => {
			this.pending.set(req, { msg, resolve: resolve as (v: unknown) => void, reject, onStep });
			this.post(msg);
		});
	}

	load(which: Which): Promise<{ which: Which } | null> {
		return this.call({ type: 'load', channel: `load:${String(which)}`, which });
	}

	score(which: Which, ids: number[], channel: string, topN = 5): Promise<ScoreResult | null> {
		return this.call({ type: 'score', channel, which, ids, topN });
	}

	next(which: Which, ids: number[], channel: string, topN = 10): Promise<NextResult | null> {
		return this.call({ type: 'next', channel, which, ids, topN });
	}

	generate(
		which: Which,
		ids: number[],
		opts: GenOptions,
		channel: string,
		onStep?: (s: GenStep) => void
	): Promise<GenResult | null> {
		return this.call({ type: 'generate', channel, which, ids, opts }, onStep);
	}

	cancel(channel: string) {
		this.post({ type: 'cancel', channel });
	}

	dispose() {
		this.worker?.terminate();
		this.worker = null;
		this.local = null;
		for (const p of this.pending.values()) p.resolve(null);
		this.pending.clear();
	}
}
