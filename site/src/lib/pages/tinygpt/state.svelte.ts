/**
 * Shared view state of the TinyGPT page: the latest run (trace + logit lens) and what the learner
 * has selected (block, head, token position, key, dimension, inspector layer). One instance per page,
 * created in +page.svelte and passed to the page components as `S`.
 *
 * `new TGState(S)` makes a linked view (used by the guided lesson): it reads and writes the same
 * model, tokenizer, run and labels as `S`, but keeps its own selection, so the lesson and the
 * "Explore freely" views never move each other.
 */
import type { RunResult } from './compute';
import type { BPETokenizer } from '../../models/bpe';
import type { TinyGPT } from '../../models/tinygpt';
import { visibleToken } from '$lib/kit/format';

export const D = 76;
export const H = 4;
export const DH = 19;
export const NL = 4;
export const V = 1024;
export const CTX = 64;
export const DMLP = 304;

export class TGState {
	/** the state this view shares its model, tokenizer, run and labels with (null = owns them) */
	readonly #src: TGState | null;
	#model = $state.raw<TinyGPT | null>(null);
	#tok = $state.raw<BPETokenizer | null>(null);
	#run = $state.raw<RunResult | null>(null);
	#labels = $state.raw<string[]>([]);

	block = $state(0);
	head = $state(0);
	/** selected query position; null = follow the last token */
	posSel = $state<number | null>(null);
	/** selected key position; null = the most-attended key */
	keySel = $state<number | null>(null);
	/** selected vector dimension for formulas */
	dim = $state(0);
	/** layer shown in the inspector */
	layerId = $state('embed');

	constructor(src: TGState | null = null) {
		this.#src = src;
	}

	get model(): TinyGPT | null {
		return this.#src ? this.#src.model : this.#model;
	}
	set model(v: TinyGPT | null) {
		if (this.#src) this.#src.model = v;
		else this.#model = v;
	}
	get tok(): BPETokenizer | null {
		return this.#src ? this.#src.tok : this.#tok;
	}
	set tok(v: BPETokenizer | null) {
		if (this.#src) this.#src.tok = v;
		else this.#tok = v;
	}
	get run(): RunResult | null {
		return this.#src ? this.#src.run : this.#run;
	}
	set run(v: RunResult | null) {
		if (this.#src) this.#src.run = v;
		else this.#run = v;
	}
	/** readable text of each token in the current window (whitespace made visible) */
	get labels(): string[] {
		return this.#src ? this.#src.labels : this.#labels;
	}
	set labels(v: string[]) {
		if (this.#src) this.#src.labels = v;
		else this.#labels = v;
	}

	get T(): number {
		return this.run?.forward.ids.length ?? 0;
	}
	get pos(): number {
		const T = this.T;
		if (!T) return 0;
		return this.posSel === null ? T - 1 : Math.min(this.posSel, T - 1);
	}
	get key(): number {
		const T = this.T;
		const i = this.pos;
		if (this.keySel !== null && this.keySel <= i) return this.keySel;
		const w = this.run?.forward.t[`L${this.block}.attn.weights`]?.data;
		if (!w || !T) return 0;
		const o = (this.head * T + i) * T;
		let best = 0;
		for (let j = 1; j <= i; j++) if (w[o + j] > w[o + best]) best = j;
		return best;
	}
	tensor(id: string) {
		return this.run?.forward.t[id];
	}
	label(i: number): string {
		return this.labels[i] ?? '';
	}
	selectPos(i: number) {
		const T = this.T;
		this.posSel = i >= T - 1 ? null : Math.max(0, i);
		this.keySel = null;
	}
	selectKey(j: number) {
		this.keySel = j;
	}
	/**
	 * Keep following the selected token while it and every token before it stay the same (typing at
	 * the end); after a new text (preset, Try this, an edit earlier in the text) follow the last token again.
	 * `before` / `after` are the full id lists; only the last CTX ids are in the window.
	 */
	keepSelection(before: number[], after: number[]) {
		const k = this.posSel;
		if (k === null) return;
		const win = (a: number[]) => a.slice(Math.max(0, a.length - CTX));
		const b = win(before);
		const a = win(after);
		for (let i = 0; i <= k; i++) {
			if (i >= a.length || a[i] !== b[i]) {
				this.posSel = null;
				this.keySel = null;
				return;
			}
		}
	}
	/** visible label of any token id */
	idLabel(id: number): string {
		return this.tok ? visibleToken(this.tok.tokenToDisplay(id)) : String(id);
	}
}
