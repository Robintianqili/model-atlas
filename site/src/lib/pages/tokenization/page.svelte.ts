/**
 * The tokenization page's shared state (redesign M2): one instance, made by the route and passed to the lesson, the
 * input chip and sheet, and the lazy Explore / Pitfalls panels (viewProps), so the texts, TinyGPT's tokenizer and the
 * trained BPE trace survive switching views. The page trains the learner's text here (not in Part 1's BpeToy), because
 * the lesson needs the trace while Explore is not rendered.
 */
import { base } from '$app/paths';
import { BPETokenizer, type TokenizerJSON } from '$lib/models/bpe';
import { conn } from './state.svelte';
import { DEFAULT_SENTENCE, PRESETS } from './presets';
import type { TrainTrace } from './train';

export const MAX_SENTENCE = 400;
/** the same number of merges TinyGPT's tokenizer has: BPE training stops there */
export const MAX_MERGES = 767;

export class TokPage {
	/** the learner's sentence (the cover, lesson steps 6–7, Explore Parts 2–3) */
	sentence = $state(DEFAULT_SENTENCE);
	/** the training text (lesson steps 1–5, Explore Part 1) */
	trainText = $state(PRESETS.train[0].text);
	/** Part 1's merge count (Explore); the lesson keeps its own */
	k = $state(0);
	/** the BPE trace of `trained`, from the page's Web Worker */
	trace = $state.raw<TrainTrace | null>(null);
	/** the text `trace` was trained on (trainText may already be newer while the worker trains) */
	trained = $state('');
	training = $state(false);
	/** worker time of the last training run */
	ms = $state(0);
	trainError = $state('');
	tiny = $state.raw<BPETokenizer | null>(null);
	/** TinyGPT's pair counts at training time (tokenizer.json mergeCounts) */
	mergeCounts = $state.raw<number[]>([]);
	loadError = $state('');
	private seq = 0;

	/** TinyGPT's tokenizer (27 KB), fetched once on mount */
	async loadTiny(): Promise<void> {
		try {
			const res = await fetch(`${base}/models/tinygpt/tokenizer.json`);
			if (!res.ok) throw new Error(`tokenizer.json: HTTP ${res.status}`);
			const json = (await res.json()) as TokenizerJSON;
			this.tiny = new BPETokenizer(json);
			this.mergeCounts = json.mergeCounts ?? [];
		} catch (e) {
			this.loadError = e instanceof Error ? e.message : String(e);
		}
	}

	/** Train BPE on `txt` in the worker; a newer call wins. */
	async train(txt: string): Promise<void> {
		const c = conn.client;
		const tiny = this.tiny;
		if (!c || !tiny) return;
		const my = ++this.seq;
		this.training = true;
		try {
			const r = await c.train(txt, MAX_MERGES, tiny.pattern, tiny.merges);
			if (my !== this.seq) return;
			this.trace = r.trace;
			this.trained = txt;
			this.ms = r.ms;
			this.trainError = '';
			if (this.k > r.trace.merges.length) this.k = r.trace.merges.length;
		} catch (e) {
			if (my === this.seq) this.trainError = e instanceof Error ? e.message : String(e);
		} finally {
			if (my === this.seq) this.training = false;
		}
	}
}
