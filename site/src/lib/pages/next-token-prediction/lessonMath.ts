/**
 * Guided lesson ("Step by step") on the next-token-prediction page: small pure helpers
 * (no Svelte, no DOM), shared by the lesson state and its stage components.
 */
import type { TopTok, Which } from './protocol.ts';

/** TinyGPT's context window (tokens). */
export const CONTEXT = 64;

/**
 * First token the model sees when it predicts position `pos` (a label index ≥ 1). These are the
 * windows the page's score() uses (64 tokens, stride 32: each position is scored by the first
 * window that holds it and its context), so a window's p equals the p of the same position there.
 */
export function ctxStart(pos: number): number {
	return pos < CONTEXT ? 0 : 32 * Math.floor((pos - 32) / 32);
}

/** One window's full output: a probability for every token in the vocabulary. */
export type Dist = {
	/** what was asked: `${which}|${context ids}>${target}` */
	key: string;
	which: Which;
	/** softmax of the logits (float64), one entry per token id */
	probs: Float64Array;
	/** token ids sorted by probability, highest first (ties: smaller id first) */
	order: Uint16Array;
	/** the token that really came next */
	target: number;
	/** probability of the target, and its rank (1 = the model's top guess) */
	p: number;
	rank: number;
	/** sum of all probabilities (1 up to rounding) */
	sum: number;
	/** spread of the distribution, in bits */
	entropy: number;
	top: TopTok[];
};

export const distKey = (which: Which, ctx: readonly number[], target: number) => `${which}|${ctx.join(',')}>${target}`;

/** softmax(logits / T) in float64 (the same arithmetic as tinygpt.samplingDistribution without cut-offs). */
export function softmax64(logits: ArrayLike<number>, temperature = 1): Float64Array {
	const V = logits.length;
	const T = Math.max(1e-6, temperature);
	const p = new Float64Array(V);
	let m = -Infinity;
	for (let i = 0; i < V; i++) if (logits[i] / T > m) m = logits[i] / T;
	let s = 0;
	for (let i = 0; i < V; i++) {
		p[i] = Math.exp(logits[i] / T - m);
		s += p[i];
	}
	for (let i = 0; i < V; i++) p[i] /= s;
	return p;
}

/** Ids sorted by probability, highest first; ties go to the smaller id (like tinygpt.ts). */
export function sortedIds(p: ArrayLike<number>): Uint16Array {
	const idx = Array.from({ length: p.length }, (_, i) => i);
	idx.sort((a, b) => p[b] - p[a] || a - b);
	return Uint16Array.from(idx);
}

export function distFrom(logits: ArrayLike<number>, which: Which, ctx: readonly number[], target: number, topN = 10): Dist {
	const probs = softmax64(logits);
	const order = sortedIds(probs);
	const p = probs[target];
	let rank = 1;
	let sum = 0;
	let h = 0;
	for (let i = 0; i < probs.length; i++) {
		const q = probs[i];
		if (q > p) rank++;
		sum += q;
		if (q > 0) h -= q * Math.log2(q);
	}
	const top: TopTok[] = [];
	for (let r = 0; r < Math.min(topN, order.length); r++) top.push({ id: order[r], p: probs[order[r]] });
	return { key: distKey(which, ctx, target), which, probs, order, target, p, rank, sum, entropy: h, top };
}

/**
 * Sample by walking the ids from the most to the least likely (the order the lesson draws its
 * strip in): the token whose segment of the 0..1 strip contains u. Every token keeps exactly its
 * probability of being picked, whatever the walking order.
 */
export function pickSorted(dist: ArrayLike<number>, order: ArrayLike<number>, u: number): { id: number; from: number; to: number; rank: number } {
	let cum = 0;
	let last = { id: order[0], from: 0, to: 0, rank: 1 };
	for (let r = 0; r < order.length; r++) {
		const q = dist[order[r]];
		if (q === 0) continue;
		last = { id: order[r], from: cum, to: cum + q, rank: r + 1 };
		cum += q;
		if (cum > u) break;
	}
	return last;
}

/** A probability as a percentage that stays readable when tiny: 0.269 -> "26.9%", 0.000912 -> "0.091%". */
export function pct(p: number): string {
	if (!Number.isFinite(p)) return '—';
	if (p >= 0.001) return `${(p * 100).toFixed(1)}%`;
	if (p <= 0) return '0%';
	return `${Number((p * 100).toPrecision(2))}%`;
}

/**
 * The end of a token sequence as text, at most `max` characters, cut at a space so a word is never
 * shown half ("…bed. He was very curious, so", not "…ed. He was …"; "bed" is the two tokens " b" +
 * "ed", so token boundaries are not enough). Text without spaces (Chinese) is cut at a character.
 */
export function tailText(ids: readonly number[], decode: (ids: number[]) => string, max: number, disp?: (id: number) => string): string {
	// a window can start inside a multi-byte character (Chinese: 3 byte tokens each): drop that piece
	let s = decode(ids.slice()).replace(/^�+/, '');
	// ... and it can end inside one: show those last byte tokens as they are (<0xE7><0x90>), not as "�"
	let bytes = '';
	if (disp && s.endsWith('�')) {
		let j = 1;
		while (j < Math.min(3, ids.length) && decode(ids.slice(0, ids.length - j)).endsWith('�')) j++;
		s = decode(ids.slice(0, ids.length - j)).replace(/^�+/, '');
		bytes = ids.slice(ids.length - j).map(disp).join('');
	}
	const cs = [...s];
	const room = Math.max(4, max - bytes.length);
	if (cs.length <= room) return s + bytes;
	let t = cs.slice(cs.length - room).join('');
	if (!/\s/.test(cs[cs.length - room - 1])) {
		const j = t.search(/\s/);
		if (j >= 0 && j < t.length - 1) t = t.slice(j);
	}
	return '…' + t.replace(/^\s+/, '') + bytes;
}

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/**
 * A number for prose, with 3 significant digits; very large or small ones in school notation:
 * 1.67e-7 -> "1.67 × 10⁻⁷", 2830000 -> "2.83 × 10⁶" (instead of calculator "1.67e−7").
 */
export function sci(v: number): string {
	if (!Number.isFinite(v)) return '—';
	const a = Math.abs(v);
	if (a === 0 || (a >= 1e-3 && a < 1e5)) {
		const s = v.toPrecision(3);
		return (s.includes('e') ? String(Number(s)) : s).replace('-', '−');
	}
	const [m, e] = v.toExponential(2).split('e');
	return `${m.replace('-', '−')} × 10${[...String(Number(e))].map((c) => SUP[c] ?? c).join('')}`;
}

/** Checkpoint labels: 30000 -> "30k". */
export const shortStep = (s: number) => (s >= 1000 ? `${s / 1000}k` : String(s));
