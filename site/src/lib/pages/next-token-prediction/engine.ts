/**
 * Compute engine for the next-token-prediction page. Pure TypeScript (no DOM, no Svelte), so it
 * runs inside a Web Worker (ntp.worker.ts) or, as a fallback, on the main thread (client.ts).
 *
 * It wraps $lib/models/tinygpt.ts:
 *   - loads the final fp32 model (weights.bin) and the fp16 training checkpoints lazily, once each
 *     (cached; progress is reported while a checkpoint file downloads);
 *   - scores a token sequence: p, rank, top-5 and entropy for every position (the same windows and
 *     the same numbers as tinygpt.surprisal(), plus the extra fields the page shows);
 *   - next-token logits for a prompt;
 *   - token-by-token generation that yields between tokens so it can be cancelled. The loop is the
 *     same as tinygpt.generate() (same mulberry32 PRNG, same sampler), so replay.json's greedy and
 *     sampled continuations are reproduced exactly for the probe prompts.
 */
import {
	load,
	mulberry32,
	samplingDistribution,
	type FetchLike,
	type TinyGPT
} from '../../models/tinygpt.ts';
import {
	keyOf,
	type GenOptions,
	type GenResult,
	type GenStep,
	type LoadStatus,
	type NextResult,
	type Reply,
	type Request,
	type ScoreResult,
	type TopTok,
	type Which
} from './protocol.ts';

export { keyOf };
export type { GenOptions, GenResult, GenStep, LoadStatus, NextResult, Reply, Request, ScoreResult, TopTok, Which };

/** The n largest entries (ties: smaller id first, like tinygpt.ts). */
export function topOf(p: ArrayLike<number>, n: number): TopTok[] {
	const best: TopTok[] = [];
	for (let i = 0; i < p.length; i++) {
		const v = p[i];
		if (best.length < n) {
			best.push({ id: i, p: v });
		} else if (v > best[n - 1].p) {
			best[n - 1] = { id: i, p: v };
		} else continue;
		let j = best.length - 1;
		while (j > 0 && best[j - 1].p < best[j].p) {
			const t = best[j - 1];
			best[j - 1] = best[j];
			best[j] = t;
			j--;
		}
	}
	return best;
}

/** Yield to the event loop (lets queued messages, e.g. a cancel, run). MessageChannel has no 4 ms clamp. */
function yieldNow(): Promise<void> {
	if (typeof MessageChannel !== 'undefined') {
		return new Promise((resolve) => {
			const ch = new MessageChannel();
			ch.port1.onmessage = () => {
				ch.port1.close();
				resolve();
			};
			ch.port2.postMessage(0);
		});
	}
	return new Promise((resolve) => setTimeout(resolve, 0));
}

function now(): number {
	return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export class Engine {
	private models = new Map<string, Promise<TinyGPT>>();
	private json = new Map<string, Promise<string>>();
	private baseUrl: string;
	private fetchImpl: typeof fetch;
	private onStatus: (s: LoadStatus) => void;

	constructor(baseUrl: string, fetchImpl: typeof fetch, onStatus: (s: LoadStatus) => void = () => {}) {
		this.baseUrl = baseUrl;
		this.fetchImpl = fetchImpl;
		this.onStatus = onStatus;
	}

	/** fetch wrapper for tinygpt.load: JSON is cached in memory; .bin files report progress. */
	private fetchFor(which: Which): FetchLike {
		return async (url: string) => {
			if (url.endsWith('.json')) {
				let text = this.json.get(url);
				if (!text) {
					text = this.fetchImpl(url).then((r) => {
						if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
						return r.text();
					});
					this.json.set(url, text);
					text.catch(() => this.json.delete(url));
				}
				const s = await text;
				return {
					ok: true,
					json: async () => JSON.parse(s),
					arrayBuffer: async () => new TextEncoder().encode(s).buffer as ArrayBuffer
				};
			}
			const r = await this.fetchImpl(url);
			if (!r.ok) {
				return { ok: false, status: r.status, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0) };
			}
			const total = Number(r.headers.get('content-length') ?? 0) || 0;
			let buf: ArrayBuffer;
			if (r.body && typeof r.body.getReader === 'function') {
				const reader = r.body.getReader();
				const chunks: Uint8Array[] = [];
				let loaded = 0;
				let last = 0;
				for (;;) {
					const { done, value } = await reader.read();
					if (done) break;
					chunks.push(value);
					loaded += value.length;
					const t = Date.now();
					if (t - last > 60) {
						last = t;
						this.onStatus({ which, state: 'loading', loaded, total });
					}
				}
				const out = new Uint8Array(loaded);
				let o = 0;
				for (const c of chunks) {
					out.set(c, o);
					o += c.length;
				}
				buf = out.buffer;
			} else {
				buf = await r.arrayBuffer();
			}
			this.onStatus({ which, state: 'loading', loaded: buf.byteLength, total: total || buf.byteLength });
			return {
				ok: true,
				json: async () => JSON.parse(new TextDecoder().decode(buf)),
				arrayBuffer: async () => buf
			};
		};
	}

	/** The model for `which`, loaded once (concurrent callers share the same promise). */
	model(which: Which): Promise<TinyGPT> {
		const key = keyOf(which);
		let m = this.models.get(key);
		if (!m) {
			this.onStatus({ which, state: 'loading', loaded: 0, total: 0 });
			m = load(this.fetchFor(which), this.baseUrl, which === 'final' ? {} : { checkpoint: which });
			this.models.set(key, m);
			m.then(
				(model) => {
					// bytes of the weights file: fp32 = 4 bytes per float, fp16 checkpoints = 2
					const floats = model.spec.tensors.reduce((a, t) => a + t.length, 0);
					const bytes = floats * (which === 'final' ? 4 : 2);
					this.onStatus({ which, state: 'ready', loaded: bytes, total: bytes });
				},
				(e: unknown) => {
					this.models.delete(key);
					this.onStatus({ which, state: 'error', loaded: 0, total: 0, message: String((e as Error)?.message ?? e) });
				}
			);
		}
		return m;
	}

	/**
	 * Score every token after the first: the same windows as tinygpt.surprisal() (64-token windows,
	 * stride 32; each position is scored once, by the first window that contains it and its context).
	 */
	score(model: TinyGPT, ids: number[], which: Which, topN = 5): ScoreResult {
		const t0 = now();
		const C = model.context;
		const V = model.vocabSize;
		const stride = C / 2;
		const p: number[] = [NaN];
		const rank: number[] = [0];
		const top: TopTok[][] = [[]];
		const entropy: number[] = [NaN];
		let windows = 0;
		let scoredUpTo = 1;
		for (let start = 0; scoredUpTo < ids.length; start += stride) {
			const win = ids.slice(start, start + C);
			const probs = model.forward(win).t.probs.data;
			windows++;
			const end = start + win.length;
			for (let pos = scoredUpTo; pos < end; pos++) {
				const o = (pos - 1 - start) * V;
				const row = probs.subarray(o, o + V);
				const pi = row[ids[pos]];
				let r = 1;
				let h = 0;
				for (let i = 0; i < V; i++) {
					const q = row[i];
					if (q > pi) r++;
					if (q > 0) h -= q * Math.log2(q);
				}
				p.push(pi);
				rank.push(r);
				top.push(topOf(row, topN));
				entropy.push(h);
			}
			scoredUpTo = end;
			if (end >= ids.length) break;
		}
		const scored = p.length - 1;
		let sumBits = 0;
		for (let i = 1; i < p.length; i++) sumBits += -Math.log2(p[i]);
		const meanBits = scored ? sumBits / scored : NaN;
		const meanNats = meanBits * Math.LN2;
		return {
			which,
			ids: ids.slice(),
			p,
			rank,
			top,
			entropy,
			meanNats,
			meanBits,
			perplexity: Math.exp(meanNats),
			scored,
			windows,
			ms: now() - t0
		};
	}

	next(model: TinyGPT, ids: number[], which: Which, topN = 10): NextResult {
		const t0 = now();
		const r = model.nextTokenProbs(ids);
		return { which, logits: r.logits, top: topOf(r.probs, topN), contextStart: r.truncated.dropped, ms: now() - t0 };
	}

	/**
	 * Generate token by token (same algorithm as tinygpt.generate()). Calls onStep after every token
	 * and yields to the event loop; returns null as soon as isCancelled() is true.
	 */
	async generate(
		model: TinyGPT,
		ids: number[],
		which: Which,
		opts: GenOptions,
		onStep: (s: GenStep) => void,
		isCancelled: () => boolean
	): Promise<GenResult | null> {
		const t0 = now();
		const C = model.context;
		const V = model.vocabSize;
		const maxNew = opts.maxNew ?? 24;
		const temperature = opts.temperature ?? 1;
		const topK = opts.topK ?? 0;
		const topP = opts.topP ?? 1;
		const stopAtEot = opts.stopAtEot ?? true;
		const eot = V - 1;
		const topN = opts.topN ?? 5;
		const rand = mulberry32(opts.seed ?? 42);
		const all = ids.slice();
		const steps: GenStep[] = [];
		for (let s = 0; s < maxNew; s++) {
			if (isCancelled()) return null;
			const start = Math.max(0, all.length - C);
			const r = model.nextTokenProbs(all.slice(start));
			const logits = r.logits;
			const pModel = r.probs;
			let token: number;
			let dist: ArrayLike<number>;
			let u: number | null = null;
			if (temperature <= 0) {
				token = 0;
				for (let i = 1; i < V; i++) if (logits[i] > logits[token]) token = i;
				dist = pModel;
			} else {
				const d = samplingDistribution(logits, temperature, topK, topP);
				dist = d;
				u = rand();
				token = -1;
				let cum = 0;
				for (let i = 0; i < V; i++) {
					if (d[i] === 0) continue;
					cum += d[i];
					token = i;
					if (cum > u) break;
				}
			}
			const step: GenStep = {
				index: s,
				token,
				p: temperature <= 0 ? 1 : dist[token],
				pModel: pModel[token],
				top: topOf(dist, topN),
				u,
				contextStart: start
			};
			steps.push(step);
			all.push(token);
			onStep(step);
			if (stopAtEot && token === eot) break;
			await yieldNow();
		}
		if (isCancelled()) return null;
		return { which, newIds: all.slice(ids.length), steps, ms: now() - t0 };
	}
}

// ------------------------------------------------------------------------------------------------
// message host (shared by the worker and the main-thread fallback)
// ------------------------------------------------------------------------------------------------

/**
 * A message handler around one Engine. Every request names a channel; a newer request on the same
 * channel (or a 'cancel') makes older ones on it stop at their next await and reply 'cancelled'.
 */
export function createHost(post: (r: Reply) => void, fetchImpl: typeof fetch) {
	let engine: Engine | null = null;
	const latest = new Map<string, number>();
	const stale = (channel: string, req: number) => latest.get(channel) !== req;

	return async function handle(m: Request): Promise<void> {
		if (m.type === 'init') {
			engine = new Engine(m.baseUrl, fetchImpl, (status) => post({ type: 'status', status }));
			return;
		}
		if (m.type === 'cancel') {
			latest.set(m.channel, -1);
			return;
		}
		const { req, channel } = m;
		latest.set(channel, req);
		if (!engine) {
			post({ type: 'error', req, message: 'engine not initialized' });
			return;
		}
		try {
			const model = await engine.model(m.which);
			if (m.type === 'load') {
				post({ type: 'result', req, data: { which: m.which } });
				return;
			}
			if (stale(channel, req)) {
				post({ type: 'cancelled', req });
				return;
			}
			if (m.type === 'score') {
				post({ type: 'result', req, data: engine.score(model, m.ids, m.which, m.topN ?? 5) });
			} else if (m.type === 'next') {
				post({ type: 'result', req, data: engine.next(model, m.ids, m.which, m.topN ?? 10) });
			} else if (m.type === 'generate') {
				const r = await engine.generate(
					model,
					m.ids,
					m.which,
					m.opts,
					(s) => post({ type: 'step', req, data: s }),
					() => stale(channel, req)
				);
				if (r) post({ type: 'result', req, data: r });
				else post({ type: 'cancelled', req });
			}
		} catch (e) {
			post({ type: 'error', req, message: String((e as Error)?.message ?? e) });
		}
	};
}
