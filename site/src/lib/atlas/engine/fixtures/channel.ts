// Test fixture: a "worker" made of a MessageChannel and createHost(), so Node tests drive the real protocol (structured
// clone and transfer lists included) without a browser. Also used to fake broken workers.
import type { EngineReply, EngineRequest } from '../../types/engine.ts';
import type { WorkerLike } from '../client.ts';
import { createHost, type Host, type HostOpts, type ServeModules } from '../host.ts';
import type { Frame } from '../../types/process.ts';
import { checkTransfer } from '../protocol.ts';

export interface ChannelWorker extends WorkerLike {
	host: Host;
	/** every reply the host posted, with its transfer list */
	sent: { msg: EngineReply; transfer: ArrayBuffer[] }[];
	/** every request the client posted */
	got: EngineRequest[];
	closed: boolean;
	/** transfer-rule problems found in posted batches (checked before posting, while the buffers are still attached) */
	problems: string[];
	/** typed arrays transferred, and their bytes */
	transferred: { buffers: number; bytes: number };
}

export interface ChannelOpts extends HostOpts {
	/** post { type: 'status', status: 'ready' } first (default true) */
	ready?: boolean;
	/** post status 'error' instead of serving */
	statusError?: string;
	/** never answer anything (a hung worker) */
	silent?: boolean;
	/** fire onerror after this many 'frames' replies (a crash mid-run). Deterministic (R0: the flaky "crash mid-run"
	 *  test): the reply that triggers it is cut to its first frame and never says done, so the crash always lands mid-run
	 *  (a busy machine could batch the whole run into that one reply), and onerror fires only after the client has
	 *  received that reply (not on a 20 ms timer that could overtake a slow message). */
	crashAfterFrames?: number;
	/** keep the transfer lists in `sent` (default true) */
	record?: boolean;
}

export function channelWorker(mods: ServeModules, o: ChannelOpts = {}): ChannelWorker {
	const mc = new MessageChannel();
	let framesSent = 0;
	let dead = false;
	const w: ChannelWorker = {
		host: null as unknown as Host,
		sent: [],
		got: [],
		closed: false,
		problems: [],
		transferred: { buffers: 0, bytes: 0 },
		onmessage: null,
		onerror: null,
		postMessage(m: unknown, t?: Transferable[]) {
			if (w.closed) return;
			mc.port1.postMessage(m, (t ?? []) as Transferable[]);
		},
		terminate() {
			if (w.closed) return;
			w.closed = true;
			w.host.close();
			mc.port1.close();
			mc.port2.close();
		}
	};
	w.host = createHost(
		mods,
		(msg, transfer) => {
			if (w.closed || o.silent || dead) return;
			if (msg.type === 'frames') {
				w.problems.push(...checkTransfer(msg.frames, transfer));
				w.transferred.buffers += transfer.length;
				for (const b of transfer) w.transferred.bytes += b.byteLength;
			}
			const crash = msg.type === 'frames' && o.crashAfterFrames !== undefined && ++framesSent >= o.crashAfterFrames;
			const out = crash && msg.type === 'frames' ? { ...msg, frames: msg.frames.slice(0, 1), done: false } : msg;
			if (o.record !== false) w.sent.push({ msg: out, transfer });
			mc.port2.postMessage(out, transfer);
			if (crash) dead = true; // nothing more leaves this worker; onerror follows once the client has this reply
		},
		{ where: 'worker', ...o }
	);
	mc.port2.onmessage = (e: MessageEvent) => {
		w.got.push(e.data as EngineRequest);
		w.host.handle(e.data as EngineRequest);
	};
	let delivered = 0;
	mc.port1.onmessage = (e: MessageEvent) => {
		w.onmessage?.(e);
		if (o.crashAfterFrames !== undefined && (e.data as EngineReply)?.type === 'frames' && ++delivered === o.crashAfterFrames)
			setTimeout(() => w.onerror?.({ message: 'simulated crash', preventDefault() {} } as unknown as Event), 0);
	};
	if (!o.silent) {
		if (o.statusError) mc.port2.postMessage({ type: 'status', status: 'error', message: o.statusError } satisfies EngineReply);
		else if (o.ready !== false) mc.port2.postMessage({ type: 'status', status: 'ready' } satisfies EngineReply);
	}
	return w;
}

/** All frames the host posted on one stream id, in order. */
export function framesSent(w: ChannelWorker, id?: number): Frame[] {
	const out: Frame[] = [];
	for (const s of w.sent) if (s.msg.type === 'frames' && (id === undefined || s.msg.id === id)) out.push(...s.msg.frames);
	return out;
}
