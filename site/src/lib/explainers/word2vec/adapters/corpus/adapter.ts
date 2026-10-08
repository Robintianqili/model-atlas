// word2vec: local input adapter 'corpus' (InputKind 'text'). The learner's raw input is { text, window }: a few sentences
// and the window size. The model's input is the list of word ids of the corpus (one id per kept word); a corpus without
// a single (center, context) pair is empty (nothing to train on), and the stage then shows the note below.
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { corpusOf, DEFAULT_WINDOW, VMAX, type CorpusRaw } from '../../corpus.ts';

export const NO_PAIRS: I18n = { en: 'There is nothing to learn from yet: write at least one sentence of two or more words.', zh: '还没有可学的内容：至少写一个包含两个或更多词的句子。' };

const adapter: InputAdapter<CorpusRaw> = {
	id: 'corpus',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const c = corpusOf(raw), ids = c.sentences.flat();
		const notes: I18n[] = [];
		if (c.dropped.length) {
			const list = c.dropped.slice(0, 12).join(', ') + (c.dropped.length > 12 ? ', …' : '');
			notes.push({ en: `Only the ${VMAX} most frequent words are used; these ${c.dropped.length} were left out: ${list}`, zh: `只用出现最多的 ${VMAX} 个词；这 ${c.dropped.length} 个词被去掉了：${list}` });
		}
		if (!c.pairs.length) notes.push(NO_PAIRS);
		return {
			tensors: { input: { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' } },
			steps: [{ id: 'ids', label: { en: 'Your words as ids', zh: '你的词，换成编号' }, tensor: { shape: [ids.length], data: Float32Array.from(ids) } }],
			empty: !c.pairs.length,
			notes,
			tokens: ids.map((i) => c.words[i]),
			vocab: c.words
		};
	},
	presets(assets) {
		const f = (assets['presets.json'] ?? assets.presets) as { presets?: { id: string; en: string; zh: string; text: string }[] } | undefined;
		return (f?.presets ?? []).map((p) => ({ id: p.id, label: { en: p.en, zh: p.zh }, raw: { text: p.text, window: DEFAULT_WINDOW } }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: { text: '', window: DEFAULT_WINDOW } },
		{ id: 'one-word', label: { en: 'a single word', zh: '只有一个词' }, raw: { text: 'Cat.', window: DEFAULT_WINDOW } },
		{ id: 'two-words', label: { en: 'one sentence of two words, window 1', zh: '一个两词的句子，窗口 1' }, raw: { text: 'Cats sleep.', window: 1 } },
		{ id: 'many-words', label: { en: 'more than 40 different words, window 3', zh: '超过 40 个不同的词，窗口 3' }, raw: { text: 'One two three four five six seven eight nine ten eleven twelve. Red blue green pink gray black white brown gold. Ant bee cow dog eel fox goat hen ibis jay kiwi lion mole newt owl pig quail rat seal toad yak zebra.', window: 3 } }
	],
	layer0: { view: 'word2vec/window', narration: { what: { en: 'Your sentences, cut into words; every different word gets a number, its id.', zh: '你的句子被切成词；每个不同的词都有一个编号（id）。' } } }
};

export default adapter;
