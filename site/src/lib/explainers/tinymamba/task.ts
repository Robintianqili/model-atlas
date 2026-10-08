// tinymamba: the selective-copy task (pure TS; the source of truth for the data). A row of dots with a few letters
// among them, then one '?' per letter: at the k-th '?' the model must give back the k-th letter, in reading order.
// pipeline/tinymamba/data.py draws the same examples with ak.Rng in the same order (claims.test.ts checks 200 of them).
import type { Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const ALPHABET = '.abcdefgh?';
export const DOT = 0;
export const ASK = 9;
export const LETTERS = 8;
export const TASK = { bodyMin: 6, bodyMax: 16, kMax: 4 };
export const isLetter = (id: number) => id >= 1 && id <= LETTERS;

/** One training example: ids, and the letter to give back at each '?' (-1 elsewhere). */
export function example(rng: Rng, o = TASK): { ids: number[]; targets: number[] } {
	const n = o.bodyMin + rng.int(o.bodyMax - o.bodyMin + 1);
	const k = 1 + rng.int(o.kMax);
	const pos = Array.from({ length: n }, (_, i) => i);
	for (let i = 0; i < k; i++) {
		const j = i + rng.int(n - i);
		[pos[i], pos[j]] = [pos[j], pos[i]];
	}
	const places = pos.slice(0, k).sort((a, b) => a - b);
	const letters = Array.from({ length: k }, () => 1 + rng.int(LETTERS));
	const body = new Array<number>(n).fill(DOT);
	places.forEach((p, i) => (body[p] = letters[i]));
	return { ids: [...body, ...new Array<number>(k).fill(ASK)], targets: [...new Array<number>(n).fill(-1), ...letters] };
}

export const textOf = (ids: number[]) => ids.map((i) => ALPHABET[i] ?? '').join('');
export const batch = (seed: number, count: number) => {
	const rng = createRng(seed);
	return Array.from({ length: count }, () => example(rng));
};

/** What the learner's row asks for: the k-th '?' wants the k-th letter, when that letter comes before it (else -1). */
export function answersOf(ids: ArrayLike<number>) {
	const letters: number[] = [];
	const asks: { at: number; want: number; from: number }[] = [];
	for (let t = 0; t < ids.length; t++) {
		if (isLetter(ids[t])) letters.push(t);
		else if (ids[t] === ASK) {
			const from = letters[asks.length] ?? -1;
			asks.push({ at: t, want: from >= 0 ? ids[from] : -1, from });
		}
	}
	return { letters, asks };
}
