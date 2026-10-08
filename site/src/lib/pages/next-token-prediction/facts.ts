/**
 * Fixed facts about TinyGPT used in the page text. Single source: static/models/tinygpt/spec.json
 * ("training.metrics") and checkpoints/index.json, measured by the tinygpt stream (notes/tinygpt.md)
 * and re-checked by its verifier. If the model is retrained, update these numbers.
 */
export const FACTS = {
	vocab: 1024,
	context: 64,
	params: 364040,
	/** validation loss over all 6,976,704 tokens of TinyStories-valid, nats per token */
	valLoss: 1.9189,
	valPpl: 6.814,
	valTokens: 6976704,
	trainLoss: 1.9086,
	trainPpl: 6.743,
	bitsPerByte: 0.8752,
	/** ln(1024): the loss of a uniform guess over the vocabulary */
	uniformLoss: Math.log(1024),
	trainTokens: 32540880,
	tokensSeen: 983040000,
	epochs: 30.21,
	steps: 30000,
	finalBytes: 1456160,
	ckptBytes: 728080,
	/** parity bar for live fp32 runs (spec.parity.tolerance) */
	tolerance: 1e-4
} as const;

/** Sampling settings of replay.json's "sampled" continuations; the live panel uses the same. */
export const REPLAY_SAMPLING = { temperature: 0.8, seed: 1234, maxNew: 24 } as const;

/** Input limits: characters in the text box, tokens used by the page (the model's window is 64). */
export const MAX_CHARS = 200;
export const MAX_TOKENS = 96;

export const DEFAULT_TEXT = 'Tom found a shiny red key under his bed. He was very curious, so he';
/** a very predictable opening: every token is TinyGPT's top guess */
export const STORY_START = 'Once upon a time, there was a little girl named Lily. She loved to';

export const PRESETS: { text: string; label: { en: string; zh: string } }[] = [
	{ text: DEFAULT_TEXT, label: { en: 'Shiny key', zh: '闪亮的钥匙' } },
	{ text: STORY_START, label: { en: 'Story start', zh: '故事开头' } },
	{ text: 'She were very happy because the sun was shining.', label: { en: 'Grammar slip', zh: '语法错误' } },
	{ text: 'The purple elephant filed a tax return on the moon.', label: { en: 'Odd sentence', zh: '怪句子' } },
	{ text: 'The moon is made of cheese.', label: { en: 'False but story-like', zh: '错误但像童话' } },
	{ text: '我爱自然语言处理', label: { en: 'Chinese', zh: '中文' } }
];

/** Sentences for the "lower loss = truth?" check (measured: cheese 3.14 vs rock 4.74 nats/token). */
export const TRUTH_PAIR = {
	falseText: 'The moon is made of cheese.',
	trueText: 'The moon is made of rock.'
} as const;
