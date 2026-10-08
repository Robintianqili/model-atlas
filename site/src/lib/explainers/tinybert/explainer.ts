// tinybert: the package manifest (design §4.9). A small BERT-style encoder trained to fill in hidden tokens (masked-
// language modelling) on TinyStories, with TinyGPT's tokenizer and size, so the page can set the two side by side:
//   fwd   the exported TinyBERT (pipeline/tinybert/export.py) on the learner's text with one token hidden
//   gpt   TinyGPT (static/models/tinygpt, read through legacy.ts) on the same text, nothing hidden: its causal attention,
//         and its guess for the hidden token from the words before it only
//   hide  BERT's training-time masking (15% chosen; 80% [MASK], 10% random, 10% kept) replayed on the learner's text
// Measured numbers (accuracy on unseen stories, both sides vs left only vs TinyGPT) are in facts.json (stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'tinybert',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tinybert', auto: ['spec.json', 'weights.bin', 'facts.json', 'tokenizer.json'] },
	input: { adapter: 'masked-text', presets: 'presets.json' },
	output: { view: 'tinybert/guesses' },
	sources: {
		fwd: { kind: 'trace', inputs: ['ids', 'segment'] },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['input'] },
		hide: { kind: 'process', process: 'hide', run: 'main', seed: 90 }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			{ en: 'Say how your text becomes TinyBERT’s input: [CLS], your tokens with one replaced by [MASK], [SEP], and three learned rows added per token (token, position, segment).', zh: '说出你的文字如何变成 TinyBERT 的输入：[CLS]、其中一个换成 [MASK] 的词元、[SEP]，以及每个词元加上的三行学来的数（词元、位置、段）。' },
			{ en: 'Explain why the [MASK] row can read the words on both sides, and how that differs from TinyGPT’s causal mask.', zh: '解释为什么 [MASK] 行能读到两侧的词，以及这和 TinyGPT 的因果掩码有什么不同。' },
			{ en: 'Work out one attention score of the [MASK] row and read how much of its attention goes to each side of the gap.', zh: '算出 [MASK] 行的一个注意力分数，并读出它的注意力有多少落在空缺的每一侧。' },
			{ en: 'Describe how BERT makes its own training puzzles from plain text, and why it is a poor writer of stories word by word.', zh: '描述 BERT 如何从普通文本里自己出训练题，以及为什么它不擅长一个词一个词地写故事。' }
		],
		prequestion: {
			id: 'tinybert-pre',
			question: {
				en: 'TinyGPT predicts each word from the words before it. To guess a hidden word in the middle of a sentence, what can TinyBERT use?',
				zh: 'TinyGPT 用前面的词预测每一个词。要猜出句子中间一个被藏起来的词，TinyBERT 能用什么？'
			},
			options: [
				{ en: 'Only the words before the gap', zh: '只有空缺前面的词' },
				{ en: 'Only the words after the gap', zh: '只有空缺后面的词' },
				{ en: 'The words on both sides of the gap', zh: '空缺两侧的词' }
			],
			answer: 2,
			explain: {
				en: 'TinyBERT has no causal mask: every token can read every other token, before and after it. That is why it fills gaps well, and also why it is a poor writer: it was never trained to guess a word from the words before it alone.',
				zh: 'TinyBERT 没有因果掩码：每个词元都能读到其他所有词元，不管在前还是在后。所以它擅长填空，也正因如此，它不擅长写作：它从没被训练过只凭前面的词去猜一个词。'
			}
		},
		tryThis: [
			{
				id: 'end',
				title: { en: 'Hide the last word', zh: '藏起最后一个词' },
				setup: { input: { text: 'Lily went to the park to play with her friends. She had a lot of fun.', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335, 410, 46, 338, 363, 258, 645, 369, 443, 46], mask: 16, picked: true } },
				predict: { en: 'Only a full stop and [SEP] follow the gap now. Will TinyBERT still put “fun” first?', zh: '现在空缺后面只剩句号和 [SEP]。TinyBERT 还会把“fun”排第一吗？' },
				look: { en: 'Compare TinyBERT’s bars with TinyGPT’s, which never read the right side anyway.', zh: '把 TinyBERT 的条形图和 TinyGPT 的比一比，反正 TinyGPT 从来不读右边。' },
				goto: { step: 'limits' }
			},
			{
				id: 'first',
				title: { en: 'Hide the first word', zh: '藏起第一个词' },
				setup: { input: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], mask: 0, picked: true } },
				predict: { en: 'Nothing comes before “Tom”. Can either model guess it?', zh: '“Tom”前面什么都没有。两个模型能猜出它吗？' },
				look: { en: 'TinyGPT has nothing to read; TinyBERT reads everything after the gap. Its guesses start with ␣, a space: most of its training texts began in the middle of a story, where the first word has a space in front. “␣Tom” and “Tom” are different tokens.', zh: 'TinyGPT 没有东西可读；TinyBERT 读空缺后面的全部内容。它的猜测都以 ␣（空格）开头：它的训练文字大多从故事中间开始，那里的第一个词前面有空格。“␣Tom”和“Tom”是两个不同的词元。' },
				goto: { step: 'limits' }
			},
			{
				id: 'colour',
				title: { en: 'Hide a colour', zh: '藏起一个颜色' },
				setup: { input: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], mask: 8, picked: true } },
				predict: { en: 'Many words could stand before “ball”. Will TinyBERT’s top guess be “red”?', zh: '“ball”前面可以放很多词。TinyBERT 的第一猜测会是“red”吗？' },
				look: { en: 'Nothing in the text says which colour, or that it is a colour at all: TinyBERT ranks the words that fit, and another one can come first.', zh: '文字里没有任何线索说明是哪种颜色，甚至没说是颜色：TinyBERT 只能给合适的词排序，别的词可能排在第一。' },
				goto: { step: 'guess' }
			}
		],
		quiz: [
			{
				q: { en: 'Which row gives TinyBERT’s guess for the hidden token?', zh: 'TinyBERT 对被藏词元的猜测来自哪一行？' },
				options: [
					{ en: 'The [CLS] row', zh: '[CLS] 行' },
					{ en: 'The [MASK] row', zh: '[MASK] 行' },
					{ en: 'The last row', zh: '最后一行' }
				],
				answer: 1,
				explain: { en: 'Every position gets scores for every token; the guess for the gap is read from the row at the gap, the [MASK] row.', zh: '每个位置都会给每个词元打分；空缺处的猜测从空缺所在的那一行，也就是 [MASK] 行读出。' }
			},
			{
				q: { en: 'In training, one in ten chosen tokens is left unchanged. Why?', zh: '训练时，被选中的词元里每十个有一个保持不变。为什么？' },
				options: [
					{ en: 'To save time', zh: '为了省时间' },
					{ en: 'So the model cannot assume that every visible token is right, and learns useful rows for all tokens', zh: '让模型不能假定每个看得见的词元都是对的，从而为所有词元学到有用的行' },
					{ en: 'Because [MASK] is a rare token', zh: '因为 [MASK] 是少见的词元' }
				],
				answer: 1,
				explain: { en: 'Real text has no [MASK]. Random and unchanged tokens make the model check every token against its neighbours, not only the [MASK] ones.', zh: '真实的文字里没有 [MASK]。随机词元和不变的词元让模型对照邻居检查每一个词元，而不只是 [MASK]。' }
			}
		],
		misconception: {
			claim: { en: 'BERT is a GPT that writes better because it reads both ways.', zh: 'BERT 是一个因为双向阅读而写得更好的 GPT。' },
			truth: { en: 'BERT fills gaps from both sides; it was never trained to continue text, so it writes left to right poorly (step 10 measures how poorly).', zh: 'BERT 从两侧填补空缺；它从没被训练去续写文字，所以从左到右写得很差（第 10 步量出了差多少）。' },
			step: 'limits'
		},
		lessonNote: {
			en: 'Ten steps: hide one token, follow the [MASK] row as it reads both sides of the gap, see TinyBERT’s guesses, how training makes its own puzzles, and what it is poor at.',
			zh: '十步：藏起一个词元，跟着 [MASK] 行读空缺两侧的内容，看 TinyBERT 的猜测、训练如何自己出题，以及它不擅长什么。'
		}
	},
	credits: [
		{ name: 'TinyStories (TinyStoriesV2-GPT4)', url: 'https://huggingface.co/datasets/roneneldan/TinyStories', license: 'CDLA-Sharing-1.0', usedFor: 'TinyBERT training and evaluation text (TinyGPT’s tokenized copy: the first 100 MiB of the train split, and the validation split)', attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)' },
		{ name: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (ideas)', url: 'https://arxiv.org/abs/1810.04805', license: 'ideas only (paper; no code or text used)', usedFor: 'the input layout ([CLS], [SEP], segments) and the 15% / 80-10-10 masking rule; code written fresh', attribution: 'Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova (2019)' },
		{ name: 'nanoGPT / minGPT (ideas)', url: 'https://github.com/karpathy/nanoGPT', license: 'MIT', usedFor: 'TinyBERT reuses TinyGPT’s block design (pre-norm attention and MLP), whose ideas came from nanoGPT and minGPT; code written fresh', attribution: 'Andrej Karpathy' }
	],
	catalog: {
		params: { en: '364,420', zh: '364,420' },
		size: { en: '1.46 MB of weights', zh: '权重 1.46 MB' },
		runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' }
	},
	glossary: [
		{ id: 'masked-language-model', term: { en: 'masked language model (MLM)', zh: '掩码语言模型（MLM）' }, def: { en: 'A model trained to fill in hidden tokens from the tokens around them, on both sides. BERT is one.', zh: '一种训练任务是根据两侧的词元补出被藏起来的词元的模型。BERT 就是这样的模型。' }, related: ['self-supervised', 'pre-training'] },
		{ id: 'special-token', term: { en: 'special token', zh: '特殊词元' }, def: { en: 'A token that stands for no text but marks something for the model, such as [CLS] (the start), [SEP] (the end of a sentence) or [MASK] (a hidden token).', zh: '不代表任何文字、只为模型做标记的词元，例如 [CLS]（开头）、[SEP]（一句话的结尾）或 [MASK]（被藏起来的词元）。' }, related: ['token', 'vocabulary'] },
		{ id: 'segment-embedding', term: { en: 'segment embedding', zh: '段嵌入' }, def: { en: 'A learned row added to every token that says which sentence it belongs to (A or B), so a model that reads two sentences at once can tell them apart.', zh: '加到每个词元上的一行学来的数，表示它属于哪一句（A 或 B），让同时读两句话的模型能把它们分开。' }, related: ['embedding', 'positional-encoding'] }
	]
};

export default manifest;
