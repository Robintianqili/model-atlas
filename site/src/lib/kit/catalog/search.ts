/**
 * Site search (redesign spec §4.5 SearchPalette): match a query against the search index (search-index.json) in both
 * languages. Pure functions; the palette loads this module and the index on first use.
 *   const groups = searchGroups(entries, 'softmax', 'en')   -> [{ kind: 'l', hits: [...] }, { kind: 't', hits: [...] }]
 */
import type { SearchEntry } from './types';

export type SearchKind = SearchEntry['k'];
export interface SearchGroup {
	kind: SearchKind;
	hits: SearchEntry[];
}

/** display order and how many of each kind the palette shows */
const ORDER: [SearchKind, number][] = [
	['l', 6],
	['a', 3],
	['t', 5],
	['p', 3]
];

const norm = (s: string) => s.toLowerCase().normalize('NFKC');
const SPECIAL = /[.*+?^${}()|[\]\\]/g;
const esc = (s: string) => s.replace(SPECIAL, '\\$&');
/** w starts a word of s (after a space, a bracket, a slash or a hyphen) */
const wordStart = (s: string, w: string) => new RegExp('(^|[\\s(（/-])' + esc(w)).test(s);

/** How well one entry matches every word of the query (0 = no match). Names count more than the second line; a match
 *  at the start of the name or of a word counts more than one inside a word. */
export function score(e: SearchEntry, words: string[], lang: 'en' | 'zh'): number {
	if (!words.length) return 0;
	const first = norm(lang === 'zh' ? e.zh : e.en);
	const names = norm(`${e.en} ${e.zh}`);
	const sub = e.sub ? norm(`${e.sub[0]} ${e.sub[1]}`) : '';
	let total = 0;
	for (const w of words) {
		let s = 0;
		if (first.startsWith(w)) s = 100;
		else if (wordStart(names, w)) s = 70;
		else if (names.includes(w)) s = 50;
		else if (sub.includes(w)) s = 15;
		if (!s) return 0;
		total += s;
	}
	// a short name that matches is likelier to be the thing asked for
	return total + Math.max(0, 20 - first.length / 3);
}

export function searchGroups(entries: SearchEntry[], q: string, lang: 'en' | 'zh'): SearchGroup[] {
	const words = norm(q).trim().split(/\s+/).filter(Boolean);
	if (!words.length) return [];
	const scored = entries.map((e) => ({ e, s: score(e, words, lang) })).filter((x) => x.s > 0);
	return ORDER.map(([kind, n]) => ({
		kind,
		hits: scored
			.filter((x) => x.e.k === kind)
			.sort((a, b) => b.s - a.s)
			.slice(0, n)
			.map((x) => x.e)
	})).filter((g) => g.hits.length);
}
