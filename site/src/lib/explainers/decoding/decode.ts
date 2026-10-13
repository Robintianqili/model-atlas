// decoding: the arithmetic of the page (pure TS: the lesson's vars, the views, the claim tests and the worker share it).
// Everything starts from ONE row of numbers: TinyGPT's 1,024 next-token scores (logits) for the learner's prompt, the last
// row of the trace source 'gpt' (static/models/tinygpt, read only). Temperature, top-k and top-p only reshape that row;
// shapeOf() does it with the kit's own sampling arithmetic (atlas/process/library/sampling.ts samplingDistribution: the
// same function the generating process uses), so the toy and the generated text agree to the last digit.
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { samplingDistribution } from '../../atlas/process/library/sampling.ts';
import { tokenizerOf } from '../../atlas/adapters/text-bpe/adapter.ts';
import { visibleToken } from '../../atlas/core/format.ts';

/** The learner's decoding settings (they travel with the prompt in the raw input, so a task can read them). */
export type Settings = { temperature: number; topK: number; topP: number; seed: number };
export const DEFAULTS: Settings = { temperature: 0.8, topK: 0, topP: 0.9, seed: 8 };
/** TinyGPT's vocabulary size and context window (static/models/tinygpt/spec.json; claims.test.ts checks them). */
export const V = 1024;
export const CONTEXT = 64;
/** The settings the lesson's explanation steps show (named, so no number is typed into a narration). */
export const LOW_T = 0.5;
export const HIGH_T = 2;
export const K_DEMO = 5;
export const P_DEMO = 0.9;
/** Rows the bar views list (the most likely tokens at temperature 1). */
export const ROWS = 10;
/** Tokens each run writes: the greedy and pure-sampling runs are long enough to show a loop. */
export const LONG = 96;
export const SHORT = 48;

const clamp = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);

/** The settings of a raw input (missing or bad values take the defaults). */
export function settingsOf(raw: unknown): Settings {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<Settings>;
	return {
		temperature: clamp(r.temperature, 0, HIGH_T, DEFAULTS.temperature), topK: Math.round(clamp(r.topK, 0, V, DEFAULTS.topK)),
		topP: clamp(r.topP, 0.05, 1, DEFAULTS.topP), seed: Math.round(clamp(r.seed, 0, 1e9, DEFAULTS.seed))
	};
}

/** The next-token logits for the prompt: the last row of the trace's [T, V] logits (null before there is a prompt). */
export function logitsOf(rt: ExplainerRuntime): Float32Array | null {
	const t: Tensor | undefined = rt.sources.gpt?.trace?.t?.logits;
	if (!t || t.shape.length !== 2 || t.shape[0] < 1) return null;
	const n = t.shape[1];
	return (t.data as Float32Array).subarray((t.shape[0] - 1) * n, t.shape[0] * n);
}

/** Token ids by chance at temperature 1, largest first; ties go to the smaller id (as the kit's sampling does). */
export function ranked(p: ArrayLike<number>): number[] {
	return Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
}

// p1: chances at temperature 1 (the model's own); pT: after the temperature only (greedy: all on the top token); dist:
// after top-k and top-p, renormalized; kept: how many tokens can still be picked; keptMass: the share of pT the kept
// tokens had (each kept chance is divided by it); nK, nP: how many tokens top-k keeps (V when off) and top-p keeps
// on its own after top-k.
export interface Shaped {
	p1: Float64Array; pT: Float64Array; dist: Float64Array; order: number[]; kept: number; keptMass: number; nK: number; nP: number;
	greedy: boolean;
}

/** Temperature, then top-k, then top-p, then renormalize: samplingDistribution of the kit (temperature 0 = greedy). */
export function shapeOf(logits: ArrayLike<number>, s: Pick<Settings, 'temperature' | 'topK' | 'topP'>): Shaped {
	const p1 = samplingDistribution(logits, 1);
	const order = ranked(p1);
	const greedy = s.temperature <= 0;
	let pT: Float64Array;
	let dist: Float64Array;
	if (greedy) {
		pT = new Float64Array(p1.length);
		pT[order[0]] = 1;
		dist = pT;
	} else {
		pT = samplingDistribution(logits, s.temperature);
		dist = samplingDistribution(logits, s.temperature, s.topK, s.topP);
	}
	const nK = s.topK > 0 && s.topK < p1.length ? s.topK : p1.length;
	let nP = p1.length;
	if (s.topP < 1) {
		let cum = 0;
		for (let n = 0; n < nK; n++) {
			cum += pT[order[n]];
			if (cum >= s.topP) {
				nP = n + 1;
				break;
			}
		}
	}
	let kept = 0;
	let keptMass = 0;
	for (let i = 0; i < dist.length; i++)
		if (dist[i] > 0) {
			kept++;
			keptMass += pT[i];
		}
	return { p1, pT, dist, order, kept, keptMass, nK, nP: Math.min(nP, nK), greedy };
}

/** The share of the chance that tokens outside the first `n` of the order hold. */
export function restMass(p: ArrayLike<number>, order: number[], n: number): number {
	let s = 0;
	for (let r = n; r < order.length; r++) s += p[order[r]];
	return s;
}

// ---------------------------------------------------------------- tokens as text

type Tok = NonNullable<ReturnType<typeof tokenizerOf>>;
/** TinyGPT's tokenizer, from the auto asset tokenizer.json (static/models/tinygpt), or null. */
export function tokOf(rt: ExplainerRuntime | null | undefined): Tok | null {
	const a = (rt as { assets?: Record<string, unknown> } | null)?.assets;
	return tokenizerOf(a?.['tokenizer.json']);
}
/** One token as a short label: without its leading space; white space alone made visible ("↵"). */
export function label(tok: Tok | null, id: number): string {
	if (!tok || id < 0) return id < 0 ? '' : `#${id}`;
	const s = tok.tokenToDisplay(id);
	return s.trim() || visibleToken(s);
}
/** A run of tokens as text (line breaks shown as "↵", the end-of-text token as ⟨end⟩). */
export function textOf(tok: Tok | null, ids: number[]): string {
	if (!tok) return ids.map((i) => `#${i}`).join(' ');
	return ids.map((i) => (tok.isSpecial(i) ? ' ⟨end⟩ ' : tok.decode([i]))).join('').replace(/\n/g, ' ↵ ').replace(/\s+/g, ' ').trim();
}

// ---------------------------------------------------------------- repetition

/** How many 4-token phrases of `ids` repeat an earlier one (a phrase that comes back three times counts twice). */
export function repeats4(ids: number[]): number {
	const seen = new Set<string>();
	let r = 0;
	for (let i = 0; i + 4 <= ids.length; i++) {
		const k = ids.slice(i, i + 4).join(',');
		if (seen.has(k)) r++;
		else seen.add(k);
	}
	return r;
}

// where the phrase first starts and how long it is (tokens), and where every copy starts (non-overlapping)
export interface Loop {
	start: number; len: number; at: number[];
}
/** The longest phrase of at least 4 tokens that comes back (non-overlapping), the most copies winning a tie; null if none.
 *  O(n³) on at most a few hundred tokens. */
export function longestLoop(ids: number[]): Loop | null {
	let best: Loop | null = null;
	const n = ids.length;
	for (let len = Math.floor(n / 2); len >= 4; len--) {
		for (let s = 0; s + len <= n; s++) {
			const at = [s];
			for (let j = s + len; j + len <= n; j++) {
				let same = true;
				for (let q = 0; q < len && same; q++) same = ids[s + q] === ids[j + q];
				if (same) {
					at.push(j);
					j += len - 1;
				}
			}
			if (at.length > 1 && (!best || at.length > best.at.length)) best = { start: s, len, at };
		}
		if (best) return best;
	}
	return null;
}
