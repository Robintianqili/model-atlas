// kv-cache: local input adapter 'prompts': two prompts as TinyGPT's tokenizer cuts them (raw { text, ids, text2, ids2 },
// data.ts). The page reads prompt A's ids as the tensor 'prompt' (no trace runs on it: the processes take the ids from
// the raw input); token text comes from vocab.ts. Edge fixtures: no text, one token, 70 ids (more than PROMPT_MAX).
import type { I18n, InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { givenA, givenB, PRESETS, PROMPT_MAX, promptA, type PromptsRaw } from '../../data.ts';
import { VOCAB } from '../../vocab.ts';

const vocab = (id: number) => VOCAB[id] ?? `#${id}`;

const adapter: InputAdapter<PromptsRaw> = {
	id: 'prompts',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const a = promptA(raw);
		const t: IntTensor = { shape: [a.length], data: Int32Array.from(a), dtype: 'i32' };
		const notes: I18n[] = [];
		const cut = (n: number, which: I18n) =>
			n > PROMPT_MAX &&
			notes.push({
				en: `Prompt ${which.en} has ${n} tokens: this page keeps its first ${PROMPT_MAX}, so that the new tokens still fit TinyGPT’s 64-token window.`,
				zh: `提示词 ${which.zh} 有 ${n} 个词元：本页只保留前 ${PROMPT_MAX} 个，好让新生成的词元仍然装得进 TinyGPT 的 64 词元窗口。`
			});
		cut(givenA(raw), { en: 'A', zh: 'A' });
		cut(givenB(raw), { en: 'B', zh: 'B' });
		if (!givenA(raw)) notes.push({ en: 'Prompt A is empty, so TinyGPT starts from its end-of-text token, as at the start of a new story.', zh: '提示词 A 是空的，所以 TinyGPT 从它的文本结束词元开始，就像一个新故事的开头。' });
		return {
			tensors: { prompt: t },
			steps: [{ id: 'ids', label: { en: 'Prompt A as token ids', zh: '提示词 A 的词元 id' }, tensor: { shape: [a.length], data: Float32Array.from(a) } }],
			empty: false,
			notes,
			tokens: a.map(vocab),
			vocab
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', ids: [], text2: '', ids2: [] } },
		{ id: 'one', label: { en: 'one token each', zh: '各一个词元' }, raw: { text: 'a', ids: [97], text2: 'a', ids2: [97] } },
		{ id: 'long', label: { en: '70 token ids (more than the page keeps)', zh: '70 个词元 id（超过本页保留的数量）' }, raw: { ids: Array.from({ length: 70 }, (_, i) => 97 + (i % 26)), ids2: Array.from({ length: 70 }, (_, i) => 97 + (i % 13)) } }
	],
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: { en: 'Prompt A becomes {n|int} tokens, each one id number.', zh: '提示词 A 变成 {n|int} 个词元，每个词元是一个 id 数字。' },
			look: { en: 'one chip per token.', zh: '每个词元一个小块。' }
		}
	}
};

export default adapter;
