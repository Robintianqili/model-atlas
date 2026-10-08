// word2vec: the learner's mini-corpus -> sentences of words -> a vocabulary (ids by frequency) -> (center, context)
// pairs from a sliding window. Pure and shared by the adapter, the trainer, the processes, the views and the lesson.
// The pairs process has no Python reference (notes, kit issue 4): claims.test.ts checks these rules. Rules: ’ and ‘ become ', the text is lowercased, a sentence ends at . ! ? ; 。！？； or a line break, a word
// is a run of a-z, 0-9 and inner apostrophes ("don't"), or one Chinese character. The VMAX most frequent words form the
// vocabulary (ties: the word seen first); other words are removed before the pairs are made.
export type CorpusRaw = string | null | undefined | { text?: string; window?: number };
export interface Pair { c: number; o: number; s: number; i: number; j: number }
export interface Corpus { text: string; window: number; words: string[]; counts: number[]; sentences: number[][]; dropped: string[]; pairs: Pair[] }

/** the largest vocabulary (the weight tables have this many rows; only the first V are used) */
export const VMAX = 40;
export const WINDOWS = [1, 2, 3];
export const DEFAULT_WINDOW = 2;
/** the seeds of the two training runs: `one` and `run` (the same run: step 0 of run is the slowed-down step), and `run2` */
export const SEED = 11;
export const SEED2 = 2;
/** the start input: the first preset of presets.json (pipeline/word2vec/presets.py; a claim test keeps the two equal) */
export const PETS = {
	text: 'The cat was very happy. The cat was sad. The cat was happy. The cat wanted to catch the bird. The cat wanted to climb the tree. The dog was sad. The dog was very happy. The dog was not nice. The dog wanted to play too. The dog wanted the ball. The bird was very happy. The bird was sad. The bird was happy. The bird wanted to help. The bird wanted to play too.',
	window: DEFAULT_WINDOW
};
const SPLIT = /[.!?;。！？；\n]+/;
const WORD = /[a-z0-9]+(?:'[a-z0-9]+)*|[\u4e00-\u9fff]/g;

export const textOf = (raw: CorpusRaw): string => (typeof raw === 'string' ? raw : (raw?.text ?? ''));
export function windowOf(raw: CorpusRaw): number {
	const w = typeof raw === 'object' && raw ? Number(raw.window) : NaN;
	return WINDOWS.includes(w) ? w : DEFAULT_WINDOW;
}

/** the words of each sentence, before the vocabulary cap */
export function sentencesOf(text: string): string[][] {
	return text.replace(/[’‘]/g, "'").toLowerCase().split(SPLIT).map((s) => s.match(WORD) ?? []).filter((s) => s.length > 0);
}

let last: Corpus | null = null;
/** The whole corpus of a raw input (the last one is cached: the trainer asks on every step). */
export function corpusOf(raw: CorpusRaw): Corpus {
	const text = textOf(raw), window = windowOf(raw);
	if (last && last.text === text && last.window === window) return last;
	const words0 = sentencesOf(text);
	const count = new Map<string, number>();
	for (const s of words0) for (const w of s) count.set(w, (count.get(w) ?? 0) + 1);
	const order = [...count.keys()]; // first appearance
	const ranked = order.map((w, k) => ({ w, k, n: count.get(w)! })).sort((a, b) => b.n - a.n || a.k - b.k);
	const words = ranked.slice(0, VMAX).map((r) => r.w);
	const id = new Map(words.map((w, i) => [w, i]));
	const sentences = words0.map((s) => s.filter((w) => id.has(w)).map((w) => id.get(w)!)).filter((s) => s.length > 0);
	const pairs: Pair[] = [];
	sentences.forEach((ids, s) => {
		for (let i = 0; i < ids.length; i++)
			for (let j = Math.max(0, i - window); j <= Math.min(ids.length - 1, i + window); j++) if (j !== i) pairs.push({ c: ids[i], o: ids[j], s, i, j });
	});
	return (last = { text, window, words, counts: words.map((w) => count.get(w)!), sentences, dropped: ranked.slice(VMAX).map((r) => r.w), pairs });
}

/** how often each word appears as a context of each center word: [V × V], row = center */
export function cooccurrence(c: Corpus): number[][] {
	const V = c.words.length;
	const m = Array.from({ length: V }, () => new Array<number>(V).fill(0));
	for (const p of c.pairs) m[p.c][p.o] += 1;
	return m;
}
