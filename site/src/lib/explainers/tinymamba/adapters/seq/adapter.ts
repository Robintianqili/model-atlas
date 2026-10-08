// tinymamba: local input adapter 'seq' = the kit's text-char adapter on the model's alphabet (spec.alphabet: '.', the
// letters a-h, '?'), plus the preset rows of presets.json (export.py) and a gentler reading of what the learner types:
// capital letters count as small ones and a space counts as a dot. Other characters are left out (the kit's note says
// which). At most 40 tokens: a longer row keeps its last 40.
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import { textChar, type TextCharRaw } from '../../../../atlas/adapters/text-char/adapter.ts';
import { ALPHABET } from '../../task.ts';

type PresetFile = { items?: { id: string; en: string; zh: string; text: string }[] };
const textOf = (raw: TextCharRaw) => (raw === null || raw === undefined ? '' : typeof raw === 'string' ? raw : (raw.text ?? ''));
export const tidy = (raw: TextCharRaw) => textOf(raw).toLowerCase().replace(/ /g, '.');

const adapter: InputAdapter<TextCharRaw> = {
	...textChar,
	id: 'seq',
	// the alphabet is task.ts's (claims.test.ts: spec.alphabet is the same), so preprocess also works without a spec (atlas run)
	preprocess: (raw, spec, opts) => textChar.preprocess(tidy(raw), spec, { vocab: ALPHABET, ...opts }),
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as PresetFile | undefined;
		return (f?.items ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: p.text }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'nothing typed', zh: '什么都没输入' }, raw: '' },
		{ id: 'ask', label: { en: 'a single ?', zh: '只有一个 ?' }, raw: '?' },
		{ id: 'long', label: { en: '60 tokens (more than fit)', zh: '60 个词元（超出容量）' }, raw: '..a..b..c..d..'.repeat(4) + '????' }
	],
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: { en: 'Your row is {n|int} tokens; each one becomes its id, its place in the alphabet . a b c d e f g h ?', zh: '你的这一行有 {n|int} 个词元；每个变成它的 id，也就是它在字母表 . a b c d e f g h ? 里的位置。' },
			look: { en: 'the dots, the letters and the question marks.', zh: '点、字母和问号。' }
		}
	}
};

export default adapter;
