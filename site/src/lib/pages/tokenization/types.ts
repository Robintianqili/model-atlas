/** Shared types for the tokenization page components. */
import type { Text } from '../../i18n/types.ts';
import type { RowId } from './tokenizers.ts';

export type ChipItem = {
	text: string;
	id?: number;
	kind?: 'special' | 'unk' | 'partial' | 'cont';
	/** first token of a new pre-token (draws a small gap before it) */
	gap?: boolean;
	/** bytes as hex for the tooltip */
	hex?: string;
	/** extra tooltip line */
	note?: string;
};

export type CurveSeries = { id: string; name: Text; color: string; points: number[]; dashed?: boolean };

/** A "Try this" experiment: a predict-first question and a button that sets the page state. */
export interface TryItem {
	q: Text;
	go: Text;
	run: () => void;
}

/** Per tokenizer: tokens for the English sentence, and the cheapest / most expensive language. */
export type LangSummary = { col: RowId; en: number; min: { li: number; v: number }; max: { li: number; v: number } };
