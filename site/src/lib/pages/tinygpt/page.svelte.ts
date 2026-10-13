/**
 * Page state of /m/tinygpt/ (redesign M1): the model (a Web Worker engine), the learner's text and its tokens, the
 * queue of forward passes, and the sampling and generation controls. One instance per page, made by the route and
 * shared by the lesson, the input sheet and the lazy views (Explore reads it through viewProps), so a text typed in the
 * sheet or tokens generated in step 12 update every view.
 *
 * S is the page's view state (the Explore selection); LS is the lesson's own selection, linked to S (state.svelte.ts),
 * so the lesson and Explore never move each other. Every number comes from the live run of the current text.
 */
import { motion } from '$lib/kit/theme.svelte';
import type { TinyGPTSpec } from '$lib/models/tinygpt';
import type { Engine } from './engine';
import { TGState, V } from './state.svelte';
import { DEFAULT_TEXT } from './presets';
import { argmax, pickToken, row, type HeadMetric, type Pick } from './analysis';

type Fns = {
	samplingDistribution: (l: ArrayLike<number>, temp: number, k?: number, p?: number) => Float64Array;
	mulberry32: (seed: number) => () => number;
};

const EOT = 1023;

export class TGPage {
	/** the page's view state (the Explore selection; owns the model, tokenizer and run) */
	readonly S = new TGState();
	/** the guided lesson's own selection (followed token, head, key, …); shares the run with S */
	readonly LS = new TGState(this.S);

	spec = $state.raw<TinyGPTSpec | null>(null);
	loadState = $state<'loading' | 'ready' | 'error'>('loading');
	loadError = $state('');
	fns = $state.raw<Fns | null>(null);

	// ---------------------------------------------------------------- input
	text = $state(DEFAULT_TEXT);
	promptIds = $state.raw<number[]>([]);
	genIds = $state.raw<number[]>([]);
	readonly allIds = $derived([...this.promptIds, ...this.genIds]);

	// ---------------------------------------------------------------- runs (Web Worker)
	busy = $state(false);
	#eng: Engine | null = null;
	#wanted: number[] = [];
	#running: Promise<void> | null = null;
	#debounce: ReturnType<typeof setTimeout> | undefined;

	// ---------------------------------------------------------------- sampling (the Explore view's controls)
	temperature = $state(1);
	topK = $state(0);
	topP = $state(1);
	seed = $state(42);
	draws = $state(0);
	pick = $state<Pick | null>(null);
	auto = $state(false);
	#rng: (() => number) | null = null;
	/** the Explore view's block player step and attention-head metric */
	blockStep = $state(0);
	metric = $state<HeadMetric>('prev');

	readonly lastLogits = $derived(this.S.run ? row(this.S.run.forward.t.logits.data, this.S.T - 1, V) : null);
	readonly modelProbs = $derived(this.S.run ? row(this.S.run.forward.t.probs.data, this.S.T - 1, V) : null);
	readonly dist = $derived.by(() => {
		const l = this.lastLogits;
		const f = this.fns;
		if (!l || !f) return null;
		if (this.temperature <= 0) {
			const d = new Float64Array(V);
			d[argmax(l)] = 1;
			return d;
		}
		return f.samplingDistribution(l, this.temperature, this.topK, this.topP);
	});

	#pump(): Promise<void> {
		if (this.#running) return this.#running;
		this.busy = true;
		const p = (async () => {
			try {
				for (;;) {
					const ids = this.#wanted;
					if (!this.#eng || ids.length === 0) {
						this.S.run = null;
						this.S.labels = [];
						break;
					}
					const r = await this.#eng.run(ids);
					if (ids !== this.#wanted) continue; // the input changed meanwhile: run the newest
					this.S.labels = r.forward.ids.map((id) => this.S.idLabel(id));
					this.S.run = r;
					break;
				}
			} catch (err) {
				this.loadError = err instanceof Error ? err.message : String(err);
			}
		})();
		// cleared after the assignment below, also when the body finished synchronously (empty input)
		this.#running = p;
		void p.finally(() => {
			if (this.#running === p) {
				this.#running = null;
				this.busy = false;
			}
		});
		return p;
	}
	runNow = (): Promise<void> => {
		clearTimeout(this.#debounce);
		const ids = this.allIds;
		this.#wanted = ids;
		return this.#pump().then(() => (this.#wanted !== this.allIds ? this.runNow() : undefined));
	};
	#schedule(ms = 60) {
		clearTimeout(this.#debounce);
		this.#debounce = setTimeout(() => void this.runNow(), ms);
	}

	/** a new prompt (typed, a preset, Try this): tokenize, keep the followed tokens where they still match, run */
	setPrompt = (v: string, immediate = false): Promise<void> => {
		this.auto = false;
		const before = this.allIds;
		this.text = v;
		this.promptIds = this.S.tok ? this.S.tok.encode(v) : [];
		this.genIds = [];
		this.S.keepSelection(before, this.promptIds);
		this.LS.keepSelection(before, this.promptIds);
		this.resetRng();
		if (immediate) return this.runNow();
		this.#schedule();
		return Promise.resolve();
	};

	resetRng = () => {
		this.#rng = this.fns ? this.fns.mulberry32(this.seed) : null;
		this.draws = 0;
		this.pick = null;
	};

	/** Explore's sampler: draw one token from the current distribution (or the argmax at temperature 0) and append it */
	genStep = async (): Promise<number> => {
		if (!this.S.run) return -1;
		// sample from the distribution of exactly the current text (finish a pending or debounced run first)
		if (this.#running || this.#wanted !== this.allIds) await this.runNow();
		const dist = this.dist;
		const lastLogits = this.lastLogits;
		const modelProbs = this.modelProbs;
		if (!this.S.run || !dist || !lastLogits || !modelProbs) return -1;
		let token: number;
		let u: number | null = null;
		let lo = 0;
		let hi = 1;
		if (this.temperature <= 0) token = argmax(lastLogits);
		else {
			if (!this.#rng) this.resetRng();
			u = this.#rng!();
			this.draws += 1;
			({ token, lo, hi } = pickToken(dist, u));
		}
		this.pick = { token, u, lo, hi, p: dist[token], pModel: modelProbs[token], draw: this.draws, seed: this.seed };
		this.genIds = [...this.genIds, token];
		await this.runNow();
		return token;
	};
	autoGenerate = async () => {
		this.auto = true;
		let n = 0;
		while (this.auto && n < 80) {
			const tok = await this.genStep();
			n++;
			if (tok < 0 || tok === EOT) break;
			if (!motion.reduced) await new Promise((r) => setTimeout(r, 140));
		}
		this.auto = false;
	};
	/** the lesson's generation: append one token it picked (with its own settings) and run the model */
	appendToken = (token: number): Promise<void> => {
		this.auto = false;
		this.pick = null;
		this.genIds = [...this.genIds, token];
		return this.runNow();
	};
	/** wait until the run matches the current text (a debounced or pending run finishes first) */
	settle = async (): Promise<void> => {
		if (this.#running || this.#wanted !== this.allIds) await this.runNow();
	};
	removeGenerated = () => {
		this.auto = false;
		this.genIds = [];
		this.resetRng();
		void this.runNow();
	};
	keepAsPrompt = () => {
		if (!this.S.tok) return;
		const v = this.text + this.S.tok.decode(this.genIds);
		void this.setPrompt([...v].slice(-400).join(''), true);
	};
	onSeed = (s: number) => {
		this.seed = s;
		this.resetRng();
	};
	/** Try this: greedy against temperature `hot` from the same text and seed */
	compare = async (n: number, hot: number, sd: number) => {
		const eng = this.#eng;
		if (!eng || !this.S.tok || !this.allIds.length) return null;
		const a = await eng.generate(this.allIds, { maxNew: n, temperature: 0, stopAtEot: true });
		const b = await eng.generate(this.allIds, { maxNew: n, temperature: hot, seed: sd, stopAtEot: true });
		return { greedy: this.S.tok.decode(a.result.newIds), hot: this.S.tok.decode(b.result.newIds), ms: a.ms + b.ms };
	};

	/** Load the engine (the model code, the weights and the tokenizer) and run the first text; returns the disposer. */
	load(modelsUrl: string, start: string): () => void {
		let disposed = false;
		(async () => {
			try {
				const mod = await import('./engine');
				const e = await mod.loadEngine(modelsUrl);
				if (disposed) {
					e.dispose();
					return;
				}
				this.#eng = e;
				this.fns = { samplingDistribution: mod.samplingDistribution, mulberry32: mod.mulberry32 };
				this.S.model = e.model;
				this.S.tok = e.tok;
				this.spec = e.spec;
				await this.setPrompt([...start].slice(0, 400).join(''), true);
				this.loadState = 'ready';
			} catch (err) {
				this.loadError = err instanceof Error ? err.message : String(err);
				this.loadState = 'error';
			}
		})();
		return () => {
			disposed = true;
			this.auto = false;
			clearTimeout(this.#debounce);
			this.#eng?.dispose();
		};
	}
}
