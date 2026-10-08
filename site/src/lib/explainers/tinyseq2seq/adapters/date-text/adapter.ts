// tinyseq2seq: local input adapter 'date-text' (InputKind 'text'): the kit's text-char adapter (one id per character,
// from spec.alphabet; unknown characters become '?'; at most spec.dims.T.max characters) plus the preset dates of
// presets.json ({ items: [{ id, en, zh, text }] }, written by pipeline/tinyseq2seq/export.py).
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import { textChar, type TextCharRaw } from '../../../../atlas/adapters/text-char/adapter.ts';

/** The learner's text from a raw input (a string or { text }). */
export const textOf = (raw: unknown): string => (typeof raw === 'string' ? raw : ((raw as { text?: string } | null)?.text ?? ''));

const adapter: InputAdapter<TextCharRaw> = {
	...textChar,
	id: 'date-text',
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { items?: { id: string; en: string; zh: string; text: string }[] } | undefined;
		return (f?.items ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: p.text }));
	}
};

export default adapter;
