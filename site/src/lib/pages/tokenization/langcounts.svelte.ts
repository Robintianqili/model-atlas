/**
 * The ten-language table's counts (live): one sentence with the same meaning in 10 languages, counted by your BPE (at
 * Part 1's merge count), TinyGPT and every real tokenizer loaded in the page's worker, plus the per-tokenizer summary
 * (English, cheapest and most expensive language). Explore's LangTable and the Pitfalls view both make one, so each
 * view has its numbers without the other being open. Create it during a component's initialisation (it runs an effect).
 */
import type { BPETokenizer } from '$lib/models/bpe';
import type { LangSummary } from './types';
import { segment, type TrainTrace } from './train';
import { PRESETS } from './presets';
import { REAL, type RowId } from './tokenizers';
import { real, conn, readyIds } from './state.svelte';

export const LANG_COLS: RowId[] = ['yours', 'tinygpt', 'gpt2', 'bert', 'cl100k', 'qwen'];

export interface LangRow {
	l: (typeof PRESETS.languages)[number];
	row: Partial<Record<RowId, number>>;
	unk: Partial<Record<RowId, number>>;
	chars: number;
	bytes: number;
}

export class LangCounts {
	/** [tokens, [UNK]] per language and real tokenizer, from the worker */
	realCounts = $state.raw<Record<string, [number, number]>[]>([]);
	private src: () => { tiny: BPETokenizer; trace: TrainTrace | null; k: number };

	constructor(src: () => { tiny: BPETokenizer; trace: TrainTrace | null; k: number }) {
		this.src = src;
		$effect(() => {
			void real.version;
			const c = conn.client;
			const ids = readyIds();
			if (!c || !ids.length) return;
			c.counts(
				PRESETS.languages.map((l) => l.text),
				ids
			)
				.then((r) => (this.realCounts = r))
				.catch(() => {});
		});
	}

	table = $derived.by((): LangRow[] => {
		const { tiny, trace, k } = this.src();
		const enc = new TextEncoder();
		const yours = (text: string) =>
			trace ? segment(text, tiny.pattern, trace.merges, Math.min(k, trace.merges.length)).reduce((s, p) => s + p.ids.length, 0) : undefined;
		return PRESETS.languages.map((l, li) => {
			const row: Partial<Record<RowId, number>> = { yours: yours(l.text), tinygpt: tiny.encode(l.text, { allowSpecial: false }).length };
			const unk: Partial<Record<RowId, number>> = {};
			for (const r of REAL)
				if (real.status[r.id] === 'ready' && this.realCounts[li]?.[r.id] !== undefined) {
					row[r.id] = this.realCounts[li][r.id][0];
					unk[r.id] = this.realCounts[li][r.id][1];
				}
			return { l, row, unk, chars: Array.from(l.text).length, bytes: enc.encode(l.text).length };
		});
	});

	/** per tokenizer: English, the cheapest and the most expensive language (counts with [UNK] lost text: skipped) */
	summary = $derived.by((): LangSummary[] => {
		const out: LangSummary[] = [];
		for (const c of LANG_COLS) {
			const en = this.table[0].row[c];
			if (en === undefined) continue;
			let min = { li: 0, v: en };
			let max = { li: 0, v: en };
			this.table.forEach((r, li) => {
				const v = r.row[c];
				if (v === undefined || r.unk[c]) return;
				if (v < min.v) min = { li, v };
				if (v > max.v) max = { li, v };
			});
			out.push({ col: c, en, min, max });
		}
		return out;
	});
}
