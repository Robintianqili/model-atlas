// masked-lm: the manifest of a CONCEPT package (/train/masked-lm/; concept.ts holds the page around the lesson). Masked
// language modelling, BERT's pre-training objective, on the learner's own text with two real models, read-only from
// their own folders (DEVKIT §18.1):
//   bert   TinyBERT (static/models/tinybert): the learner's text with the tokens they chose hidden as [MASK] (or, after
//          the 80/10/10 draw is sent to it, replaced by a random token or kept)
//   gpt    TinyGPT (static/models/tinygpt), trained on the same stories to predict the next token: the same text, nothing hidden
//   batch  BERT's 15% / 80-10-10 masking drawn on a small batch (the learner's text and TinyBERT's example sentences), with
//          the counts of the same rule on a full training batch (process 'batch', with its Python reference)
// Measured numbers (TinyBERT's and TinyGPT's validation numbers) are copies in facts.json (make_data.py).
import type { ExplainerManifest, I18n } from '../../atlas/types/index.ts';
import { PRESETS } from './adapters/masked-text/adapter.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it),
 *  and the concept catalog fields promote prints (interaction, engine, mode, links). */
type ConceptManifest = Omit<ExplainerManifest, 'type' | 'catalog'> & {
	type: 'concept';
	catalog?: ExplainerManifest['catalog'] & { interaction?: I18n; engine?: I18n; mode?: string; links?: string[] };
};

const manifest: ConceptManifest = {
	slug: 'masked-lm',
	type: 'concept',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/masked-lm', auto: ['facts.json'] },
	input: { adapter: 'masked-text', presetsFromAdapter: true, start: PRESETS[0].raw },
	sources: {
		bert: { kind: 'trace', assets: 'models/tinybert', inputs: ['ids', 'segment'] },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['input'] },
		batch: { kind: 'process', process: 'batch', run: 'main', seed: 90 }
	},
	page: {
		goals: [
			{ en: 'Hide tokens of your own sentence and read TinyBERT’s guesses for them, and say what the model can and cannot see while it guesses.', zh: '在你自己的句子里藏起词元，读出 TinyBERT 对它们的猜测，并说出模型猜的时候能看到什么、看不到什么。' },
			{ en: 'Compute the masked-language-model loss: −log p of the right token, averaged over the chosen positions only, and say why the other positions are left out.', zh: '算出掩码语言模型的损失：正确词元的 −log p，只在被选中的位置上求平均，并说明为什么其他位置不算。' },
			{ en: 'Apply BERT’s 15% and 80/10/10 rule to a batch, count what it does, and explain why some chosen tokens become random or stay unchanged.', zh: '在一个批次上运行 BERT 的 15% 和 80/10/10 规则，数一数它做了什么，并解释为什么有些被选中的词元变成随机词元或保持不变。' },
			{ en: 'Compare masked and causal language modelling on the same sentence: which context each model reads, how many positions each grades, and what each is poor at.', zh: '在同一句话上比较掩码语言建模和因果语言建模：每个模型读哪些上下文、批改多少个位置、各自不擅长什么。' }
		],
		prequestion: {
			id: 'masked-lm-pre',
			question: {
				en: 'In BERT’s training, a token chosen for the puzzle usually becomes [MASK]. Why not always?',
				zh: '在 BERT 的训练中，被选来出题的词元通常会变成 [MASK]。为什么不是每次都这样？'
			},
			options: [
				{ en: 'It would make the puzzle too easy', zh: '那样谜题就太容易了' },
				{ en: 'Real text has no [MASK], so the model must also learn to check the tokens it can see', zh: '真实的文字里没有 [MASK]，所以模型也得学会检查它看得见的词元' },
				{ en: 'To save computing time', zh: '为了节省计算时间' }
			],
			answer: 1,
			explain: {
				en: 'Later tasks feed the model plain text, without [MASK]. If every chosen token were [MASK], the model could trust every token it sees. So 10% of the chosen tokens become a random token and 10% stay as they are, and all of them are graded.',
				zh: '后续任务给模型的是普通文字，里面没有 [MASK]。如果每个被选中的词元都变成 [MASK]，模型就可以相信它看到的每个词元。所以被选中的词元中有 10% 变成随机词元、10% 保持不变，而且它们全都要被批改。'
			}
		},
		quiz: [
			{
				q: { en: 'On the same sentence, which positions does TinyGPT’s causal loss grade?', zh: '在同一句话上，TinyGPT 的因果损失批改哪些位置？' },
				options: [
					{ en: 'Only the hidden ones, like TinyBERT', zh: '只有被藏的位置，和 TinyBERT 一样' },
					{ en: 'Every token after the first', zh: '第一个之后的每个词元' },
					{ en: 'Only the last token', zh: '只有最后一个词元' }
				],
				answer: 1,
				explain: { en: 'Each position predicts the next token from the tokens before it, so every token after the first is an answer. A masked model is graded only at the chosen positions, about one token in seven.', zh: '每个位置都用前面的词元预测下一个词元，所以第一个之后的每个词元都是一个答案。掩码模型只在被选中的位置上被批改，大约每七个词元一个。' }
			},
			{
				q: { en: 'TinyBERT’s loss on unseen stories is lower than TinyGPT’s. Is TinyBERT the better language model?', zh: 'TinyBERT 在没见过的故事上的损失比 TinyGPT 低。TinyBERT 是更好的语言模型吗？' },
				options: [
					{ en: 'Yes: a lower loss is always better', zh: '是：损失越低总是越好' },
					{ en: 'No: the two losses grade different puzzles, so they cannot be compared', zh: '不是：两个损失批改的是不同的谜题，所以不能比较' }
				],
				answer: 1,
				explain: { en: 'TinyBERT is graded on gaps with text on both sides, TinyGPT on the next token with nothing after it. Given only the left side, as TinyGPT is, TinyBERT guesses worse.', zh: 'TinyBERT 是在两侧都有文字的空缺上被批改的，TinyGPT 是在后面什么都没有的下一个词元上被批改的。只给左边的内容时（TinyGPT 的处境），TinyBERT 反而猜得更差。' }
			}
		],
		misconception: {
			claim: { en: 'The masked-language-model loss grades every token of the sentence.', zh: '掩码语言模型的损失会批改句子里的每一个词元。' },
			truth: { en: 'Only the chosen tokens, about 15%, are graded. At every other position the right token is in the input, and TinyBERT already ranks it first almost everywhere, so grading those positions would mostly reward copying.', zh: '只批改被选中的词元，大约 15%。在其他位置上，正确词元就在输入里，TinyBERT 几乎处处都把它排在第一，所以批改这些位置多半只是在奖励照抄。' },
			step: 'copy'
		},
		lessonNote: {
			en: 'Ten steps on your own sentence: hide tokens and read TinyBERT’s guesses, compute the loss on the hidden positions only, draw the 80/10/10 rule on a batch and send it to TinyBERT, set TinyBERT beside TinyGPT, and see what masking cannot do.',
			zh: '在你自己的句子上走十步：藏起词元、读 TinyBERT 的猜测，只在被藏的位置上算损失，在一个批次上抽 80/10/10 规则并把它交给 TinyBERT，把 TinyBERT 和 TinyGPT 放在一起比较，再看掩码做不到什么。'
		}
	},
	credits: [
		{ name: 'TinyStories (TinyStoriesV2-GPT4)', url: 'https://huggingface.co/datasets/roneneldan/TinyStories', license: 'CDLA-Sharing-1.0', usedFor: 'the training and evaluation text of TinyGPT and TinyBERT, whose measured numbers this page quotes', attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)' },
		{ name: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (ideas)', url: 'https://arxiv.org/abs/1810.04805', license: 'ideas only (paper; no code or text used)', usedFor: 'masked language modelling, the [CLS] / [SEP] layout and the 15% / 80-10-10 rule, drawn and counted here', attribution: 'Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova (2019)' },
		{ name: 'ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ideas)', url: 'https://arxiv.org/abs/2003.10555', license: 'ideas only (paper; no code or text used)', usedFor: 'the point that a masked model learns from only the chosen ~15% of positions per example, and the further-reading link', attribution: 'Kevin Clark, Minh-Thang Luong, Quoc V. Le and Christopher D. Manning (2020)' }
	],
	catalog: {
		interaction: {
			en: 'Click any token of your sentence to hide it and read TinyBERT’s guesses live; compute the loss on the hidden positions only; watch the 80/10/10 rule draw a batch with real counts and send your row to TinyBERT; set TinyBERT beside TinyGPT on the same sentence to see which context each reads.',
			zh: '点你句子里的任意词元把它藏起来，实时读出 TinyBERT 的猜测；只在被藏的位置上算损失；看 80/10/10 规则在一个批次上抽签（真实计数），并把你那一行交给 TinyBERT；在同一句话上把 TinyBERT 和 TinyGPT 放在一起，看各自读到哪些上下文。'
		},
		engine: { en: 'TinyBERT + TinyGPT live', zh: 'TinyBERT + TinyGPT 实时' },
		mode: 'live',
		links: ['tinybert', 'tinygpt']
	},
	glossary: [
		{ id: 'masking-rate', term: { en: 'masking rate', zh: '掩码率' }, def: { en: 'The share of tokens chosen for the masked-LM puzzle in each example: 15% in BERT. A chosen token becomes [MASK] 80% of the time, a random token 10%, and stays as it is 10% (the 80/10/10 rule).', zh: '每个样本里被选中出掩码谜题的词元比例：BERT 是 15%。被选中的词元 80% 变成 [MASK]，10% 变成随机词元，10% 保持不变（80/10/10 规则）。' }, related: ['masked-language-model', 'special-token'] },
		{ id: 'loss-mask', term: { en: 'loss mask', zh: '损失掩码' }, def: { en: 'The set of positions a loss is computed on. In masked language modelling only the chosen tokens count; the others are skipped (in PyTorch with the label −100, the ignore_index of cross_entropy).', zh: '计算损失时使用的位置集合。在掩码语言建模里只有被选中的词元计入；其余位置被跳过（在 PyTorch 里用标签 −100，即 cross_entropy 的 ignore_index）。' }, related: ['loss', 'cross-entropy', 'masked-language-model'] },
		{ id: 'bidirectional-context', term: { en: 'bidirectional context', zh: '双向上下文' }, def: { en: 'The tokens on both sides of a position. A masked language model reads them all; a causal language model reads only the tokens before.', zh: '一个位置两侧的词元。掩码语言模型全部都读；因果语言模型只读前面的词元。' }, related: ['masked-language-model', 'causal-mask', 'attention'] }
	]
};

export default manifest;
