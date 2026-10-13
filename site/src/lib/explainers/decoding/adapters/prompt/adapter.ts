// decoding: local input adapter 'prompt'. The kit's text-bpe adapter (TinyGPT's own tokenizer) plus the decoding
// settings, which travel with the prompt so that every step, the toy and a task can read them:
//   { text, ids, temperature, topK, topP, seed }     (decode.ts settingsOf fills missing settings with the defaults)
// preprocess() is text-bpe's with TinyGPT's context (64) and vocabulary (1,024), because this concept package has no
// spec.json of its own (notes: Kit issues). The presets' ids are TinyGPT's tokenizer output (claims.test.ts checks them).
import type { InputAdapter } from '../../../../atlas/types/index.ts';
import { textBpe } from '../../../../atlas/adapters/text-bpe/adapter.ts';
import { CONTEXT, DEFAULTS, V, type Settings } from '../../decode.ts';

export type PromptRaw = ({ text?: string; ids?: number[] } & Partial<Settings>) | null | undefined;

/** Prompts in the style of TinyStories (written for this page). */
export const PROMPTS = [
	{ id: 'park', label: { en: 'At the park', zh: '在公园' }, raw: { text: 'Lily went to the park to play with her', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335], ...DEFAULTS } },
	{ id: 'once', label: { en: 'Once upon a time', zh: '很久很久以前' }, raw: { text: 'Once upon a time, there was a little', ids: [428, 437, 258, 397, 44, 400, 282, 258, 390], ...DEFAULTS } },
	{ id: 'dog', label: { en: 'A big dog', zh: '一只大狗' }, raw: { text: 'One day, a big dog', ids: [382, 326, 44, 258, 345, 463], ...DEFAULTS } },
	{ id: 'cat', label: { en: 'The cat sat', zh: '猫坐在' }, raw: { text: 'The cat sat on the', ids: [409, 458, 291, 116, 353, 262], ...DEFAULTS } }
];

const SPEC = { context: CONTEXT, vocabSize: V };

const adapter: InputAdapter<PromptRaw> = {
	id: 'prompt',
	kind: 'text',
	preprocess: (raw, _spec, opts) => textBpe.preprocess(raw && typeof raw === 'object' ? { text: raw.text, ids: raw.ids } : raw, SPEC, opts),
	presets: () => PROMPTS,
	edgeFixtures: () => textBpe.edgeFixtures().map((f) => ({ ...f, raw: { ...(f.raw as object), ...DEFAULTS } })),
	layer0: textBpe.layer0
};

export default adapter;
