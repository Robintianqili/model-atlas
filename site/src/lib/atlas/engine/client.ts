// Engine client (design §4.10): the main-thread end of one package worker. It starts the worker, routes replies to
// streams and calls, and switches to 'main' (main-thread fallback) when the worker cannot start, reports an error status,
// does not answer within readyTimeoutMs, or crashes. Semantics taken from the round-2 NtpClient: a newer request on a
// channel supersedes the older one (an older call resolves null; an older stream is told it was superseded).
// engine/runner.ts builds LiveTapes on top of it; most code never uses the client directly.
import type { Control, Frame, SchemaJson } from '../types/process.ts';
import type { EngineReply, EngineRequest } from '../types/engine.ts';
import type { OpenArgs } from './protocol.ts';

/** What the client needs from a Worker (a MessagePort pair works too, which the Node tests use). */
export interface WorkerLike {
	postMessage(m: unknown, transfer?: Transferable[]): void;
	terminate(): void;
	onmessage: ((e: MessageEvent) => void) | null;
	onerror?: ((e: Event) => void) | null;
	onmessageerror?: ((e: MessageEvent) => void) | null;
}

export type EngineMode = 'starting' | 'worker' | 'main';

export interface StreamHandlers {
	onSchema(schema: SchemaJson): void;
	onFrames(frames: Frame[], done: boolean): void;
	onError(code: string, message: string): void;
	/** the worker is gone (or never started): continue on the main thread */
	onFallback(reason: string): void;
	/** a newer stream was opened on the same channel */
	onSuperseded(): void;
}

export interface StreamHandle {
	readonly id: number;
	readonly channel: string;
	/** posts the control only while this stream is still the current one on its channel */
	control(c: Control): void;
	cancel(): void;
}

export interface ClientOpts {
	/** fall back when the worker has not posted anything after this many ms (default 10000) */
	readyTimeoutMs?: number;
}

export class EngineClient {
	private worker: WorkerLike | null = null;
	private seq = 1;
	private heard = false;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private readonly streams = new Map<number, { channel: string; h: StreamHandlers }>();
	private readonly current = new Map<string, number>(); // channel -> current stream id
	private readonly calls = new Map<number, { channel: string; resolve: (v: unknown) => void; reject: (e: Error) => void }>();
	private readonly callByChannel = new Map<string, number>();
	private readonly modeFns = new Set<(m: EngineMode, reason: string | null) => void>();
	mode: EngineMode = 'starting';
	/** why the client runs on the main thread (null while the worker is fine) */
	reason: string | null = null;
	/** replies received (status, schema, frames, results, errors) */
	received = 0;

	constructor(factory: (() => WorkerLike) | null | undefined, o: ClientOpts = {}) {
		if (!factory) {
			this.toMain('this package has no worker');
			return;
		}
		let w: WorkerLike;
		try {
			w = factory();
		} catch (e) {
			this.toMain(`the worker could not be created: ${e instanceof Error ? e.message : String(e)}`);
			return;
		}
		this.worker = w;
		w.onmessage = (e: MessageEvent) => this.receive(e.data as EngineReply);
		w.onerror = (e: Event) => {
			(e as { preventDefault?: () => void }).preventDefault?.();
			const msg = (e as ErrorEvent).message || 'error event';
			this.toMain(this.heard ? `the worker crashed: ${msg}` : `the worker failed to start: ${msg}`);
		};
		w.onmessageerror = () => this.toMain('a worker message could not be read');
		const ms = o.readyTimeoutMs ?? 10000;
		if (ms > 0 && Number.isFinite(ms)) {
			this.timer = setTimeout(() => {
				if (!this.heard) this.toMain(`the worker did not answer within ${ms} ms`);
			}, ms);
		}
	}

	/** Called with the mode whenever it changes ('starting' -> 'worker' or -> 'main'). */
	onMode(fn: (m: EngineMode, reason: string | null) => void): () => void {
		this.modeFns.add(fn);
		return () => this.modeFns.delete(fn);
	}

	private setMode(m: EngineMode, reason: string | null) {
		if (this.mode === m) return;
		this.mode = m;
		this.reason = reason;
		for (const fn of this.modeFns) fn(m, reason);
	}

	private post(m: EngineRequest) {
		if (this.mode === 'main' || !this.worker) return;
		try {
			this.worker.postMessage(m);
		} catch (e) {
			this.toMain(`could not post to the worker: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	private receive(r: EngineReply) {
		if (this.mode === 'main') return;
		this.received++;
		if (!this.heard) {
			this.heard = true;
			if (this.timer) clearTimeout(this.timer);
		}
		if (r.type === 'status') {
			if (r.status === 'error') this.toMain(`the worker reported: ${r.message ?? 'error'}`);
			else if (r.status === 'ready') this.setMode('worker', null);
			return;
		}
		if (this.mode === 'starting') this.setMode('worker', null);
		const s = this.streams.get(r.id);
		if (s) {
			if (r.type === 'schema') s.h.onSchema(r.schema);
			else if (r.type === 'frames') s.h.onFrames(r.frames, r.done === true);
			// a finished stream stays registered: a 'reset' control starts a fresh run on it
			else if (r.type === 'error') {
				this.dropStream(r.id);
				s.h.onError(r.code, r.message);
			}
			return;
		}
		const c = this.calls.get(r.id);
		if (c) {
			this.calls.delete(r.id);
			if (this.callByChannel.get(c.channel) === r.id) this.callByChannel.delete(c.channel);
			if (r.type === 'result') c.resolve(r.data);
			else if (r.type === 'error') c.reject(new Error(`${r.code}: ${r.message}`));
		}
	}

	private dropStream(id: number) {
		const s = this.streams.get(id);
		if (!s) return;
		this.streams.delete(id);
		if (this.current.get(s.channel) === id) this.current.delete(s.channel);
	}

	/** Switch to the main thread for good: every open stream gets onFallback, every pending call resolves null. */
	toMain(reason: string) {
		if (this.mode === 'main') return;
		if (this.timer) clearTimeout(this.timer);
		try {
			this.worker?.terminate();
		} catch {
			/* already gone */
		}
		this.worker = null;
		this.setMode('main', reason);
		const streams = [...this.streams.values()];
		this.streams.clear();
		this.current.clear();
		for (const s of streams) s.h.onFallback(reason);
		for (const c of this.calls.values()) c.resolve(null);
		this.calls.clear();
		this.callByChannel.clear();
	}

	/** Open a stream on `channel` (a source name). A stream already open on the channel is superseded. */
	stream(args: OpenArgs, channel: string, h: StreamHandlers, maxFps?: number): StreamHandle {
		const id = this.seq++;
		const prev = this.current.get(channel);
		if (prev !== undefined) {
			const old = this.streams.get(prev);
			this.dropStream(prev);
			old?.h.onSuperseded();
		}
		const handle: StreamHandle = {
			id,
			channel,
			control: (c: Control) => {
				if (this.current.get(channel) === id) this.post({ channel, type: 'control', control: c });
			},
			cancel: () => {
				if (this.current.get(channel) === id) this.post({ channel, type: 'cancel' });
				this.dropStream(id);
			}
		};
		if (this.mode === 'main') {
			// no worker: report it on the next microtask, after the caller has the handle
			queueMicrotask(() => h.onFallback(this.reason ?? 'main thread'));
			return handle;
		}
		this.streams.set(id, { channel, h });
		this.current.set(channel, id);
		this.post({ id, channel, type: 'stream', method: 'open', args, ...(maxFps ? { maxFps } : {}) });
		return handle;
	}

	/** A request/response call. A newer call on the same channel resolves the older one with null; so does a fallback. */
	call<T = unknown>(method: string, args: unknown = null, channel = `call:${method}`): Promise<T | null> {
		if (this.mode === 'main') return Promise.resolve(null);
		const id = this.seq++;
		const older = this.callByChannel.get(channel);
		if (older !== undefined) {
			this.calls.get(older)?.resolve(null);
			this.calls.delete(older);
		}
		this.callByChannel.set(channel, id);
		return new Promise<T | null>((resolve, reject) => {
			this.calls.set(id, { channel, resolve: resolve as (v: unknown) => void, reject });
			this.post({ id, channel, type: 'call', method, args });
		});
	}

	dispose() {
		if (this.timer) clearTimeout(this.timer);
		for (const s of this.streams.values()) s.h.onSuperseded();
		this.streams.clear();
		this.current.clear();
		for (const c of this.calls.values()) c.resolve(null);
		this.calls.clear();
		try {
			this.worker?.terminate();
		} catch {
			/* gone */
		}
		this.worker = null;
	}
}
