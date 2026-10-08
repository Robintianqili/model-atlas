/**
 * Shared state of the next-token-prediction page: the learner's text, its tokens, the final
 * model's score of it, the linked token position (hover / selection) and model-loading status.
 * One instance is created by the route and passed to every part as `S`.
 */
import type { BPETokenizer } from '$lib/models/bpe';
import type { NtpClient } from './client';
import { keyOf, type LoadStatus, type ScoreResult, type Which } from './protocol';
import { DEFAULT_TEXT, MAX_TOKENS } from './facts';

export type RankBucket = 0 | 1 | 2 | 3;

/** GLTR-style rank buckets: top 1, 2–10, 11–100, beyond 100. */
export function bucketOf(rank: number): RankBucket {
	return rank <= 1 ? 0 : rank <= 10 ? 1 : rank <= 100 ? 2 : 3;
}

export function sameIds(a: readonly number[] | undefined, b: readonly number[]): boolean {
	if (!a || a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

export class NtpState {
	text = $state(DEFAULT_TEXT);
	tok = $state.raw<BPETokenizer | null>(null);
	client = $state.raw<NtpClient | null>(null);
	/** set when the tokenizer or the worker could not start */
	error = $state<string | null>(null);

	/** every token of the text, and the ones the page uses (first MAX_TOKENS) */
	allIds = $derived(this.tok ? this.tok.encode(this.text) : []);
	ids = $derived(this.allIds.length > MAX_TOKENS ? this.allIds.slice(0, MAX_TOKENS) : this.allIds);
	cut = $derived(Math.max(0, this.allIds.length - MAX_TOKENS));
	/** Chinese / Japanese / Korean characters (3 UTF-8 bytes each, almost never seen in training); curly quotes do not count */
	hasCjk = $derived(/[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]/.test(this.text));

	/** the final model's score of `ids` (may briefly lag behind the text while it recomputes) */
	score = $state.raw<ScoreResult | null>(null);
	fresh = $derived(!!this.score && sameIds(this.score.ids, this.ids));

	/** linked token position (a target index ≥ 1): hover is transient, focus is the selection */
	hover = $state<number | null>(null);
	focus = $state<number | null>(null);
	active = $derived(this.hover ?? this.focus);

	/** loading status per model ('final', 'ckpt:1000', …) */
	status = $state<Record<string, LoadStatus>>({});

	statusOf(w: Which): LoadStatus | undefined {
		return this.status[keyOf(w)];
	}

	/** readable text of one token (" the", "<0xE7>" for a partial UTF-8 character) */
	disp(id: number): string {
		if (id === undefined || id === null) return '';
		return this.tok ? this.tok.tokenToDisplay(id) : String(id);
	}

	decode(ids: number[]): string {
		return this.tok ? this.tok.decode(ids) : '';
	}

	setText(v: string) {
		this.text = v;
	}
}
