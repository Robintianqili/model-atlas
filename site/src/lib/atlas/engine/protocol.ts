// The one worker protocol (design §4.10; frozen message types in types/engine.ts) plus the pieces both ends share:
// open-request extras, error codes, schema without functions, control merging, and the
// copy-and-dedupe frame packer that implements the transfer rules. The module registry lives in host.ts (it needs the
// trainer wrapper; the main thread never imports it).
//
// Transfer rules (types/engine.ts):
//  - frames carry stored fields only: derived fields (FieldDef.derive) and functions are dropped before posting.
//  - every typed array in a batch is COPIED into its own fresh, exactly sized ArrayBuffer at the moment the frame is made
//    (a process may mutate its live objects right after yielding), deduplicated by identity within the batch (a repeat is
//    re-used only while its bytes are unchanged), and those buffers are transferred. Parameter, optimizer or arena memory is
//    never transferred, and no buffer is listed twice.
//  - cursors never cross threads: the main thread evaluates them on its TapeStore.
// Pure TS: no DOM, no Svelte (it is bundled into every package worker).
import type { Control, Frame, SchemaJson, Schema, FieldDef } from '../types/process.ts';
import type { EngineReply, EngineRequest } from '../types/engine.ts';

export type { EngineReply, EngineRequest };

/** Recorder options that ride along with an 'open' request (extras on top of the frozen args). The host passes them to
 *  process/record.ts stepper(), so worker frames equal localRunner frames (anchors, keep, maxFrames, draws). */
export interface OpenExtras {
	keyEvery?: number;
	maxFrames?: number;
	keep?: 'all' | { every: number };
	/** start producing frames right away (default false: on demand, as localRunner) */
	autorun?: boolean;
}
export type OpenArgs = Extract<EngineRequest, { type: 'stream' }>['args'] & OpenExtras;
export type StreamRequest = Extract<EngineRequest, { type: 'stream' }>;
export type CallRequest = Extract<EngineRequest, { type: 'call' }>;

/** Error codes in 'error' replies. */
export const ENGINE_ERRORS = {
	unknownProcess: 'unknown-process',
	processError: 'process-error',
	unknownMethod: 'unknown-method',
	callError: 'call-error',
	badRequest: 'bad-request'
} as const;

/** Schema without functions (what worker messages and tape files carry). */
export function schemaJson(schema: Schema): SchemaJson {
	const out: SchemaJson = {};
	for (const [k, d] of Object.entries(schema)) {
		const { derive, ...rest } = d as FieldDef;
		out[k] = { ...rest, ...(derive || (d as { derived?: boolean }).derived ? { derived: true } : {}) };
	}
	return out;
}

/** Field names that are never sent (derived in the schema). */
export function derivedFields(schema: Schema | SchemaJson): Set<string> {
	const out = new Set<string>();
	for (const [k, d] of Object.entries(schema)) if (typeof (d as FieldDef).derive === 'function' || (d as { derived?: boolean }).derived) out.add(k);
	return out;
}

/** 'set' controls that arrive before the next frame are merged (the generator takes one control per frame). */
export function mergeControls(q: Control[], c: Control): void {
	const last = q[q.length - 1];
	if (c.type === 'set' && last?.type === 'set') q[q.length - 1] = { type: 'set', opt: { ...last.opt, ...c.opt } };
	else q.push(c);
}

// ------------------------------------------------------------------ copy-and-dedupe packer
type View = Exclude<ArrayBufferView, DataView>;

function isView(v: unknown): v is View {
	return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

function sameBytes(a: View, b: View): boolean {
	if (a.byteLength !== b.byteLength || a.constructor !== b.constructor) return false;
	const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
	const y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
	for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
	return true;
}

export interface PackStats {
	frames: number;
	/** typed arrays copied (fresh buffers) */
	copies: number;
	/** repeats re-used by identity (bytes unchanged) */
	reused: number;
	/** bytes in the transfer list */
	bytes: number;
}

/**
 * Collects frames for one 'frames' message. add() copies a frame immediately (so later in-place mutations by the process
 * cannot leak into it); take() returns the batch and its transfer list (each fresh buffer exactly once) and starts a new
 * batch. Shared typed arrays within a batch stay shared in the message (structured clone keeps identity).
 */
export class FramePacker {
	private frames: Frame[] = [];
	private memo = new Map<View, View>();
	private buffers = new Set<ArrayBuffer>();
	readonly total: PackStats = { frames: 0, copies: 0, reused: 0, bytes: 0 };
	private readonly drop: Set<string>;

	constructor(drop: Iterable<string> = []) {
		this.drop = new Set(drop);
	}

	get size(): number {
		return this.frames.length;
	}

	add(f: Frame): void {
		const state: Record<string, unknown> = {};
		for (const [k, v] of Object.entries((f.state ?? {}) as Record<string, unknown>)) {
			if (v === undefined || typeof v === 'function' || this.drop.has(k)) continue;
			state[k] = this.copy(v);
		}
		const out: Frame = { phase: f.phase, state };
		if (f.step !== undefined) out.step = f.step;
		if (f.sub !== undefined) out.sub = f.sub;
		if (f.changes) out.changes = this.copy(f.changes) as Frame['changes'];
		if (f.focus) out.focus = this.copy(f.focus) as Frame['focus'];
		if (f.metrics) out.metrics = { ...f.metrics };
		if (f.caption) out.caption = this.copy(f.caption) as Frame['caption'];
		if (f.draws) out.draws = f.draws.slice();
		this.frames.push(out);
		this.total.frames++;
	}

	take(): { frames: Frame[]; transfer: ArrayBuffer[] } {
		const frames = this.frames;
		const transfer = [...this.buffers];
		this.frames = [];
		this.memo = new Map();
		this.buffers = new Set();
		return { frames, transfer };
	}

	private copy(v: unknown): unknown {
		if (v === null || typeof v !== 'object') return typeof v === 'function' ? undefined : v;
		if (isView(v)) {
			const seen = this.memo.get(v);
			if (seen && sameBytes(seen, v)) {
				this.total.reused++;
				return seen;
			}
			const c = (v as unknown as { slice(): View }).slice(); // own, exactly sized buffer
			this.memo.set(v, c);
			this.buffers.add(c.buffer as ArrayBuffer);
			this.total.copies++;
			this.total.bytes += c.byteLength;
			return c;
		}
		if (Array.isArray(v)) return v.map((x) => this.copy(x));
		if (v instanceof ArrayBuffer) return v.slice(0);
		const out: Record<string, unknown> = {};
		for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
			if (x === undefined || typeof x === 'function') continue;
			out[k] = this.copy(x);
		}
		return out;
	}
}

/** Every ArrayBuffer listed in a transfer list is distinct and exactly the size of the view it backs (a test helper). */
export function checkTransfer(frames: Frame[], transfer: ArrayBuffer[]): string[] {
	const problems: string[] = [];
	if (new Set(transfer).size !== transfer.length) problems.push('a buffer is listed twice');
	const listed = new Set(transfer);
	const walk = (v: unknown, path: string) => {
		if (v === null || typeof v !== 'object') return;
		if (isView(v)) {
			if (!listed.has(v.buffer as ArrayBuffer)) problems.push(`${path}: buffer not in the transfer list`);
			if (v.byteOffset !== 0 || v.byteLength !== v.buffer.byteLength) problems.push(`${path}: view is not its whole buffer`);
			return;
		}
		for (const [k, x] of Object.entries(v as Record<string, unknown>)) walk(x, `${path}.${k}`);
	};
	frames.forEach((f, i) => walk(f, `frames[${i}]`));
	return problems;
}
