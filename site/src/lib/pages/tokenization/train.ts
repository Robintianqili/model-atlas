/**
 * Byte-level BPE training with a full trace, for the "build a tokenizer from your own text" toy.
 * Pure TS (runs in the page's Worker and in Node).
 *
 * trainTrace() learns exactly the same merges as trainBPE() in $lib/models/bpe.ts (same pre-tokens,
 * same counting, same tie-break: higher count first, then the smaller pair (a, b)); it also records,
 * before every merge, the most frequent pairs. tests-tokenization/train.check.ts checks the equality.
 */
import { utf8Bytes } from '../../models/bpe.ts';

export type PairCount = { a: number; b: number; n: number };

export type TraceStep = {
	rank: number;
	pair: [number, number];
	newId: number;
	/** how often the pair occurred when it was merged */
	count: number;
	/** tokens in the whole text after this merge */
	tokenCount: number;
	/** the most frequent pairs just BEFORE this merge (top[0] is the merged pair) */
	top: PairCount[];
	/** number of different adjacent pairs before this merge */
	pairs: number;
};

export type TrainTrace = {
	merges: [number, number][];
	steps: TraceStep[];
	/** UTF-8 bytes of every token id (0-255 are the bytes themselves) */
	bytes: number[][];
	initialTokenCount: number;
	/** pre-tokens in the text, and how many different ones */
	pretokens: number;
	distinct: number;
	/** pairs left after the last merge (empty when every pre-token is one token) */
	finalTop: PairCount[];
	finalPairs: number;
	/** tokens in the text when cut to TinyGPT's first k merges, k = 0 … tinyMerges.length */
	tiny: number[];
};

function better(n1: number, k1: number, n2: number, k2: number): boolean {
	return n1 > n2 || (n1 === n2 && k1 < k2);
}

function mergeAll(ids: number[], a: number, b: number, newId: number): number[] {
	const out: number[] = [];
	let i = 0;
	while (i < ids.length) {
		if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
			out.push(newId);
			i += 2;
		} else {
			out.push(ids[i]);
			i += 1;
		}
	}
	return out;
}

function hasPair(w: number[], a: number, b: number): boolean {
	for (let i = 0; i < w.length - 1; i++) if (w[i] === a && w[i + 1] === b) return true;
	return false;
}

/** Count adjacent pairs over all distinct pre-tokens (weighted by frequency); return the top N. */
function countPairs(words: number[][], freqs: number[], topN: number): { top: PairCount[]; pairs: number; best: number; bestN: number } {
	const pc = new Map<number, number>();
	for (let wi = 0; wi < words.length; wi++) {
		const w = words[wi];
		const f = freqs[wi];
		for (let i = 0; i < w.length - 1; i++) {
			const key = w[i] * 65536 + w[i + 1];
			pc.set(key, (pc.get(key) ?? 0) + f);
		}
	}
	const top: { key: number; n: number }[] = [];
	for (const [key, n] of pc) {
		if (top.length < topN || better(n, key, top[top.length - 1].n, top[top.length - 1].key)) {
			let j = top.length;
			while (j > 0 && better(n, key, top[j - 1].n, top[j - 1].key)) j--;
			top.splice(j, 0, { key, n });
			if (top.length > topN) top.pop();
		}
	}
	return {
		top: top.map((t) => ({ a: Math.floor(t.key / 65536), b: t.key % 65536, n: t.n })),
		pairs: pc.size,
		best: top.length ? top[0].key : -1,
		bestN: top.length ? top[0].n : 0
	};
}

export function trainTrace(
	text: string,
	numMerges: number,
	pattern: string,
	opts: { topN?: number; tinyMerges?: [number, number][] } = {}
): TrainTrace {
	const topN = opts.topN ?? 8;
	const re = new RegExp(pattern, 'gu');
	const counts = new Map<string, number>();
	let pretokens = 0;
	for (const pt of text.match(re) ?? []) {
		counts.set(pt, (counts.get(pt) ?? 0) + 1);
		pretokens++;
	}
	const words0: number[][] = [];
	const freqs: number[] = [];
	for (const [pt, n] of counts) {
		words0.push(utf8Bytes(pt));
		freqs.push(n);
	}
	const bytes: number[][] = [];
	for (let b = 0; b < 256; b++) bytes.push([b]);
	const total = (ws: number[][]) => ws.reduce((s, w, i) => s + w.length * freqs[i], 0);
	const initialTokenCount = total(words0);

	let words = words0.map((w) => w.slice());
	let tokenCount = initialTokenCount;
	const merges: [number, number][] = [];
	const steps: TraceStep[] = [];
	let last = countPairs(words, freqs, topN);
	for (let m = 0; m < numMerges; m++) {
		if (last.bestN <= 0) break;
		const a = Math.floor(last.best / 65536);
		const b = last.best % 65536;
		const newId = 256 + m;
		merges.push([a, b]);
		bytes.push(bytes[a].concat(bytes[b]));
		words = words.map((w) => (hasPair(w, a, b) ? mergeAll(w, a, b, newId) : w));
		tokenCount = total(words);
		steps.push({ rank: m, pair: [a, b], newId, count: last.bestN, tokenCount, top: last.top, pairs: last.pairs });
		last = countPairs(words, freqs, topN);
	}

	// the same text cut to TinyGPT's first k merges (applying merges in order = encoding with them)
	const tiny: number[] = [];
	if (opts.tinyMerges) {
		let tw = words0.map((w) => w.slice());
		let tc = initialTokenCount;
		tiny.push(tc);
		opts.tinyMerges.forEach(([a, b], m) => {
			tw = tw.map((w) => (hasPair(w, a, b) ? mergeAll(w, a, b, 256 + m) : w));
			tc = total(tw);
			tiny.push(tc);
		});
	}

	return {
		merges,
		steps,
		bytes,
		initialTokenCount,
		pretokens,
		distinct: words0.length,
		finalTop: last.top,
		finalPairs: last.pairs,
		tiny
	};
}

/** One pre-token of the text and its tokens after the first k merges. */
export type Segment = { text: string; ids: number[] };

/**
 * Split `text` into pre-tokens and apply the first k merges to each (lowest rank first, as encode()
 * does; this equals applying the merges in training order).
 */
export function segment(text: string, pattern: string, merges: [number, number][], k: number): Segment[] {
	const ranks = new Map<number, number>();
	for (let i = 0; i < Math.min(k, merges.length); i++) ranks.set(merges[i][0] * 65536 + merges[i][1], i);
	const cache = new Map<string, number[]>();
	const re = new RegExp(pattern, 'gu');
	const out: Segment[] = [];
	for (const pt of text.match(re) ?? []) {
		let ids = cache.get(pt);
		if (!ids) {
			ids = utf8Bytes(pt);
			while (ids.length >= 2) {
				let best = -1;
				for (let i = 0; i < ids.length - 1; i++) {
					const r = ranks.get(ids[i] * 65536 + ids[i + 1]);
					if (r !== undefined && (best < 0 || r < best)) best = r;
				}
				if (best < 0) break;
				ids = mergeAll(ids, merges[best][0], merges[best][1], 256 + best);
			}
			cache.set(pt, ids);
		}
		out.push({ text: pt, ids });
	}
	return out;
}

export type MergeTree = { id: number; kids?: [MergeTree, MergeTree] };

/** How token `id` was built: a binary tree of merges down to single bytes. */
export function mergeTree(id: number, merges: [number, number][]): MergeTree {
	if (id < 256) return { id };
	const [a, b] = merges[id - 256];
	return { id, kids: [mergeTree(a, merges), mergeTree(b, merges)] };
}
