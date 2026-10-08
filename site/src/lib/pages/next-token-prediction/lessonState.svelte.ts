/**
 * State of the guided lesson on the next-token-prediction page. It keeps its own selection (the
 * window it follows), so the lesson never moves the "Explore freely" views, and caches every
 * distribution it asked the worker for (final fp32 weights and the fp16 checkpoints).
 */
import type { NtpState } from './state.svelte';
import type { Which } from './protocol';
import { ctxStart, distFrom, distKey, type Dist } from './lessonMath';

export type CkInfo = {
	step: number;
	file: string;
	tokens: number;
	trainLoss: number;
	valLoss: number;
	label?: { en: string; zh: string };
};

const MAX_CACHE = 80;

export class NtpLesson {
	S: NtpState;
	/** the window the lesson follows: the index of its label token (null = the last window) */
	picked = $state<number | null>(null);
	/** every distribution fetched so far, by distKey() */
	dists = $state.raw<Record<string, Dist>>({});
	/** loss of the whole sentence at a checkpoint, by `${step}|${ids}` */
	ckLoss = $state.raw<Record<string, number>>({});
	/** checkpoints/index.json (without the tensor table), fetched when step 6 is first shown */
	ckpts = $state.raw<CkInfo[] | null>(null);
	error = $state<string | null>(null);
	private inflight = new Set<string>();
	private indexReq: Promise<void> | null = null;

	constructor(S: NtpState) {
		this.S = S;
	}

	/** tokens used by the page, and the number of (context -> next token) windows */
	n = $derived.by(() => this.S.ids.length);
	windows = $derived.by(() => Math.max(0, this.S.ids.length - 1));
	/** label position of the followed window (1 … n-1; 0 when there is no window) */
	pos = $derived.by(() => {
		const n = this.S.ids.length;
		if (n < 2) return 0;
		return Math.min(n - 1, Math.max(1, this.picked ?? n - 1));
	});
	start = $derived.by(() => ctxStart(this.pos));
	ctx = $derived.by(() => this.S.ids.slice(ctxStart(this.pos), this.pos));
	target = $derived.by(() => (this.S.ids.length >= 2 ? this.S.ids[this.pos] : -1));
	idsKey = $derived.by(() => this.S.ids.join(','));

	/** follow another window (a label position) */
	pick(pos: number | null) {
		if (pos === null) {
			this.picked = null;
			return;
		}
		const n = this.S.ids.length;
		this.picked = Math.min(Math.max(1, pos), Math.max(1, n - 1));
	}

	/** the followed window's distribution under `which`, if it has been computed */
	dist(which: Which): Dist | null {
		if (this.pos < 1) return null;
		return this.dists[distKey(which, this.ctx, this.target)] ?? null;
	}

	/** ask the worker for the followed window's logits under `which` (cached; one request per key) */
	request(which: Which) {
		const c = this.S.client;
		if (!c || this.pos < 1) return;
		const ctx = this.ctx;
		const target = this.target;
		const key = distKey(which, ctx, target);
		if (this.dists[key] || this.inflight.has(key)) return;
		this.inflight.add(key);
		c.next(which, ctx, `lesson:${which}`, 1).then(
			(r) => {
				this.inflight.delete(key);
				if (r) this.put(key, distFrom(r.logits, which, ctx, target));
			},
			(e: Error) => {
				this.inflight.delete(key);
				this.error = e.message;
			}
		);
	}

	private put(key: string, d: Dist) {
		const keys = Object.keys(this.dists);
		const next: Record<string, Dist> = {};
		for (const k of keys.slice(Math.max(0, keys.length - MAX_CACHE + 1))) next[k] = this.dists[k];
		next[key] = d;
		this.dists = next;
	}

	/** loss of the whole sentence at a checkpoint (live, cached per text) */
	sentenceLoss(step: number): number | null {
		const v = this.ckLoss[`${step}|${this.idsKey}`];
		return v === undefined ? null : v;
	}
	requestSentence(step: number) {
		const c = this.S.client;
		const ids = this.S.ids;
		if (!c || ids.length < 2) return;
		const key = `${step}|${ids.join(',')}`;
		if (this.ckLoss[key] !== undefined || this.inflight.has(key)) return;
		this.inflight.add(key);
		c.score(step, ids, `lesson-s:${step}`, 1).then(
			(r) => {
				this.inflight.delete(key);
				if (r) this.ckLoss = { ...this.ckLoss, [key]: r.meanNats };
			},
			() => this.inflight.delete(key)
		);
	}

	/** checkpoints/index.json: steps, files and the validation loss logged during training */
	loadIndex(url: string): Promise<void> {
		if (!this.indexReq) {
			this.indexReq = fetch(url)
				.then((r) => {
					if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
					return r.json();
				})
				.then((j: { checkpoints: CkInfo[] }) => {
					this.ckpts = j.checkpoints;
				})
				.catch((e: Error) => {
					this.indexReq = null;
					this.error = e.message;
				});
		}
		return this.indexReq;
	}
}
