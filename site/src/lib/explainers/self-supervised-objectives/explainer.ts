// self-supervised-objectives: the manifest of a CONCEPT package (/train/self-supervised-objectives/; concept.ts holds the
// page around the lesson). One sentence and one drawing, many objectives: what each self-supervised objective hides and
// what it predicts. Two objectives run on real models, read-only from their own folders (DEVKIT §18.1):
//   bert  TinyBERT (static/models/tinybert), masked language modelling: the learner's text with one token as [MASK]
//   gpt   TinyGPT (static/models/tinygpt), causal language modelling: the same text, nothing hidden
//   hide  BERT's 15% / 80-10-10 masking replayed on the learner's text (process 'hide', copied from tinybert, with its
//         Python reference)
// The other objectives (next-sentence prediction, span corruption, text infilling, fill in the middle, masked
// autoencoding, JEPA, contrastive views, added noise) are drawings: patterns.ts and objectives.ts compute their mask
// patterns on the learner's text and on one Quick, Draw! drawing with each paper's settings; no model was trained with
// them, so their steps say "illustrative". Measured numbers come from the two packages (facts.json, make_data.py).
import type { ExplainerManifest, I18n } from '../../atlas/types/index.ts';
import { PRESETS } from './adapters/masked-text/adapter.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it),
 *  and the concept catalog fields promote prints (interaction, engine, mode, links). */
type ConceptManifest = Omit<ExplainerManifest, 'type' | 'catalog'> & {
	type: 'concept';
	catalog?: ExplainerManifest['catalog'] & { interaction?: I18n; engine?: I18n; mode?: string; links?: string[] };
};

const manifest: ConceptManifest = {
	slug: 'self-supervised-objectives',
	type: 'concept',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/self-supervised-objectives', auto: ['facts.json'] },
	input: { adapter: 'masked-text', presetsFromAdapter: true, start: PRESETS[0].raw },
	sources: {
		bert: { kind: 'trace', assets: 'models/tinybert', inputs: ['ids', 'segment'] },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['input'] },
		hide: { kind: 'process', process: 'hide', run: 'main', seed: 90 }
	},
	page: {
		goals: [
			{ en: 'For any self-supervised objective, say what the model sees, what is hidden from it and what it must predict, and draw it as a mask pattern on a sentence or an image.', zh: '对任何一种自监督目标，说出模型看到什么、什么被藏起来、它要预测什么，并把它画成一句话或一张图上的遮罩图案。' },
			{ en: 'Explain why one sentence grades a causal language model on every token after the first but a masked language model only on the few tokens it hid, and what the masked model gets in return.', zh: '解释为什么同一句话能在第一个之后的每个词元上给因果语言模型打分，却只在藏起的少数几个词元上给掩码语言模型打分，以及掩码模型换来了什么。' },
			{ en: 'Read live numbers from TinyGPT and TinyBERT on your own sentence, and say why each is poor at the other’s job.', zh: '在你自己的句子上读出 TinyGPT 和 TinyBERT 的实时数值，并说明为什么它们各自做不好对方的工作。' }
		],
		prequestion: {
			id: 'self-supervised-objectives-pre',
			question: {
				en: 'Nobody has labelled a single word of a pile of stories. Can a model still learn from it with right answers to check against?',
				zh: '一大堆故事里，没有人标注过任何一个词。模型还能从中学习、并且有正确答案可以对照吗？'
			},
			options: [
				{ en: 'No: without labels there is nothing to check against', zh: '不能：没有标签，就没有东西可以对照' },
				{ en: 'Yes: hide part of the text and use the hidden part as the answer', zh: '能：把文字藏起一部分，用藏起的部分当答案' },
				{ en: 'Only if a person checks its guesses', zh: '只有让人去检查它的猜测才行' }
			],
			answer: 1,
			explain: {
				en: 'The text is its own answer key: hide a token (or the rest of the sentence, or a patch of an image) and the hidden part is the right answer. Every objective on this page is one rule for what to hide and what to predict.',
				zh: '文字本身就是答案：藏起一个词元（或句子的后半段，或图片的一块），藏起来的部分就是正确答案。本页的每一种目标，都是一条“藏什么、预测什么”的规则。'
			}
		},
		quiz: [
			{
				q: { en: 'Which objective turns every token of a sentence after the first into an answer, in one pass?', zh: '哪种目标在一次计算里就把句子中第一个之后的每个词元都变成一个答案？' },
				options: [{ en: 'Causal language modelling (next token)', zh: '因果语言建模（预测下一个词元）' }, { en: 'Masked language modelling', zh: '掩码语言建模' }, { en: 'Next-sentence prediction', zh: '下一句预测' }],
				answer: 0,
				explain: { en: 'Each position predicts the token after it from the tokens before it, so all of them are answers at once. A masked model is graded only on the tokens that were chosen, and next-sentence prediction gives one label per pair.', zh: '每个位置都用它前面的词元去预测后面的那个，所以所有词元同时都是答案。掩码模型只在被选中的词元上打分，下一句预测每一对句子只有一个标签。' }
			},
			{
				q: { en: 'Masked autoencoding and JEPA both hide image patches and predict them from the visible ones. What differs most?', zh: '掩码自编码和 JEPA 都藏起图块，再根据看得见的图块去预测它们。最大的区别是什么？' },
				options: [{ en: 'What is predicted: pixels, or another network’s summary numbers', zh: '预测什么：像素，还是另一个网络算出的概括数值' }, { en: 'Whether the image is cut into patches at all', zh: '图片是否被切成图块' }, { en: 'Nothing', zh: '没有区别' }],
				answer: 0,
				explain: { en: 'Both see only the visible patches. MAE must paint the hidden pixels back; JEPA predicts numbers that summarize the hidden patches, so it need not get every pixel right.', zh: '两者都只看得见没被藏的图块。MAE 必须把被藏的像素画回来；JEPA 预测的是概括被藏图块的数值，所以不必把每个像素都算对。' }
			}
		],
		misconception: {
			claim: { en: 'A model trained to fill in hidden words is also good at writing the next word.', zh: '被训练来补出被藏词语的模型，也擅长写出下一个词。' },
			truth: { en: 'Each objective trains one skill. TinyBERT fills gaps with text on both sides; when the text stops right after the gap it guesses worse than TinyGPT, which was trained for exactly that.', zh: '每种目标只训练一种本领。TinyBERT 擅长填两侧都有文字的空；如果文字在空缺之后就结束，它猜得比 TinyGPT 还差，而 TinyGPT 正是为此而训练的。' },
			step: 'limits'
		},
		lessonNote: {
			en: 'Eleven steps on one sentence and one drawing: the causal and masked objectives on two live models, the 80/10/10 rule, then next-sentence prediction, spans, masked patches and contrastive views as labelled drawings, and what each objective cannot teach.',
			zh: '在一句话和一幅简笔画上走十一步：在两个实时模型上看因果目标和掩码目标、80/10/10 规则，再用标明“示意”的图看下一句预测、片段、被遮的图块和对比视图，最后看每种目标教不会什么。'
		}
	},
	credits: [
		{ name: 'TinyStories (TinyStoriesV2-GPT4)', url: 'https://huggingface.co/datasets/roneneldan/TinyStories', license: 'CDLA-Sharing-1.0', usedFor: 'the training and evaluation text of TinyGPT and TinyBERT, whose measured numbers this page quotes', attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)' },
		{ name: 'Quick, Draw! dataset', url: 'https://github.com/googlecreativelab/quickdraw-dataset', license: 'CC BY 4.0', usedFor: 'the one drawing (a house, from TinyViT’s presets) cut into patches for the image objectives', attribution: 'Quick, Draw! dataset by Google Creative Lab' },
		{ name: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (ideas)', url: 'https://arxiv.org/abs/1810.04805', license: 'ideas only (paper; no code or text used)', usedFor: 'masked language modelling, the 15% / 80-10-10 rule and next-sentence prediction, as drawn here', attribution: 'Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova (2019)' },
		{ name: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5; ideas)', url: 'https://arxiv.org/abs/1910.10683', license: 'ideas only (paper; no code or text used)', usedFor: 'span corruption: 15% of tokens in spans of mean length 3, sentinels (the random-span mask is written fresh)', attribution: 'Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li and Peter J. Liu (2020)' },
		{ name: 'BART: Denoising Sequence-to-Sequence Pre-training (ideas)', url: 'https://arxiv.org/abs/1910.13461', license: 'ideas only (paper; no code or text used)', usedFor: 'text infilling: 30% of tokens masked in spans with Poisson(3) lengths, one [MASK] per span, the whole text as the target', attribution: 'Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer (2020)' },
		{ name: 'Masked Autoencoders Are Scalable Vision Learners (MAE; ideas)', url: 'https://arxiv.org/abs/2111.06377', license: 'ideas only (paper; no code or text used)', usedFor: 'masked autoencoding: 75% of the patches hidden, the hidden pixels as the target', attribution: 'Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár and Ross Girshick (2022)' },
		{ name: 'A Simple Framework for Contrastive Learning of Visual Representations (SimCLR; ideas)', url: 'https://arxiv.org/abs/2002.05709', license: 'ideas only (paper; no code or text used)', usedFor: 'contrastive learning with two altered views of one image', attribution: 'Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton (2020)' },
		{ name: 'Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA; ideas)', url: 'https://arxiv.org/abs/2301.08243', license: 'ideas only (paper; no code or text used)', usedFor: 'predicting hidden patches as another network’s numbers instead of pixels', attribution: 'Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun and Nicolas Ballas (2023)' },
		{ name: 'RoBERTa: A Robustly Optimized BERT Pretraining Approach (ideas)', url: 'https://arxiv.org/abs/1907.11692', license: 'ideas only (paper; no code or text used)', usedFor: 'the pitfall on next-sentence prediction: removing it matched or slightly improved results on later tasks', attribution: 'Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer and Veselin Stoyanov (2019)' },
		{ name: 'Efficient Training of Language Models to Fill in the Middle (ideas)', url: 'https://arxiv.org/abs/2207.14255', license: 'ideas only (paper; no code or text used)', usedFor: 'fill in the middle: prefix and suffix moved in front of the middle', attribution: 'Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek and Mark Chen (2022)' }
	],
	catalog: {
		interaction: {
			en: 'One sentence and one drawing, many objectives: TinyGPT’s next-token probabilities and TinyBERT’s guesses for a token you click, live; the 80/10/10 rule on your text; next-sentence prediction, span corruption, text infilling, fill in the middle, masked autoencoding, JEPA, contrastive views and added noise drawn as labelled mask patterns.',
			zh: '一句话、一幅简笔画，多种目标：实时看 TinyGPT 预测下一个词元的概率，以及 TinyBERT 对你点中的词元的猜测；在你的文字上演示 80/10/10 规则；下一句预测、片段破坏、文本填充、中间填充、掩码自编码、JEPA、对比视图和加噪声都画成标明“示意”的遮罩图案。'
		},
		engine: { en: 'TinyGPT + TinyBERT live + illustrative mask diagrams (labeled)', zh: 'TinyGPT + TinyBERT 实时运行 + 标明“示意”的遮罩图' },
		mode: 'both',
		links: ['tinygpt', 'tinybert', 'tinyvit', 'tinyclip']
	},
	glossary: [
		{ id: 'next-sentence-prediction', term: { en: 'next-sentence prediction (NSP)', zh: '下一句预测（NSP）' }, def: { en: 'A pre-training task of the original BERT: given two sentences, say whether the second really came right after the first (IsNext) or was taken from another text (NotNext).', zh: '最初的 BERT 的一种预训练任务：给出两句话，判断第二句是真的紧跟在第一句后面（IsNext），还是取自另一段文字（NotNext）。' }, related: ['masked-language-model', 'self-supervised'] },
		{ id: 'span-corruption', term: { en: 'span corruption', zh: '片段破坏' }, def: { en: 'A pre-training task (T5): runs of consecutive tokens are hidden, each run replaced by one sentinel token, and the model writes out only the hidden runs.', zh: '一种预训练任务（T5）：把连续的几个词元藏起来，每一段换成一个哨兵词元，模型只需写出被藏的那几段。' }, related: ['masked-language-model', 'self-supervised'] },
		{ id: 'masked-autoencoder', term: { en: 'masked autoencoder (MAE)', zh: '掩码自编码器（MAE）' }, def: { en: 'An image model pre-trained by hiding most of an image’s patches and painting their pixels back from the visible ones.', zh: '一种图像模型的预训练方法：藏起一张图的大部分图块，再根据看得见的图块把它们的像素画回来。' }, related: ['self-supervised', 'image-patch'] },
		{ id: 'image-patch', term: { en: 'image patch', zh: '图块' }, def: { en: 'A small square of an image; a vision transformer reads an image as a row of patches, the way a language model reads tokens.', zh: '图片中的一小块方形区域；视觉 Transformer 把一张图读成一串图块，就像语言模型读一串词元。' }, related: ['token', 'transformer'] }
	]
};

export default manifest;
