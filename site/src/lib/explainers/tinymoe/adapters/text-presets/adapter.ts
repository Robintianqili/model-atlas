// tinymoe: local input adapter 'text-presets' = the kit's text-char adapter (one id per character, the last 128 kept)
// plus two things it lacks: preset texts (presets.json, written by pipeline/tinymoe/export.py) and the normalization the
// model was trained with (pipeline/tinymoe/model.py NORMALIZE: curly quotes and dashes become plain ones), so a phone
// keyboard's “smart” quotes reach the model as the quotes it learned.
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import { textChar, type TextCharRaw } from '../../../../atlas/adapters/text-char/adapter.ts';

type PresetFile = { texts?: { id: string; en: string; zh: string; text: string }[] };
export const NORMALIZE: [RegExp, string][] = [
	[/[“”]/g, '"'],
	[/[‘’]/g, "'"],
	[/[–—]/g, '-'],
	[/…/g, '...'],
	[/ /g, ' ']
];
export const normalize = (s: string) => NORMALIZE.reduce((a, [re, to]) => a.replace(re, to), s);
const textOf = (raw: TextCharRaw) => (raw === null || raw === undefined ? '' : typeof raw === 'string' ? raw : (raw.text ?? ''));

const adapter: InputAdapter<TextCharRaw> = {
	...textChar,
	id: 'text-presets',
	preprocess: (raw, spec, opts) => textChar.preprocess(normalize(textOf(raw)), spec, opts),
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as PresetFile | undefined;
		return (f?.texts ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: p.text }));
	}
};

export default adapter;
