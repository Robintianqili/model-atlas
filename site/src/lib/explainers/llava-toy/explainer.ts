// llava-toy: the package manifest (design §4.9, §7). A LLaVA-style connector toy on Quick, Draw! drawings:
//   fwd   the exported graph (pipeline/llava-toy/export.py) on the learner's drawing and question: the frozen TinyViT's
//         blocks 1-3, the 16 patch rows (image features), the MLP projector (image tokens), the question's word rows,
//         the joined sequence through the tiny decoder LM, and the next-word shares of every row (the answer: last row)
// The limits step quotes the ladder preset's answer from facts.json (export.py; claims.test.ts checks it in TS).
// Measured numbers (test accuracies, where the answer row looks) are in facts.json (stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { HOUSE, START } from './adapters/drawing-question/adapter.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'llava-toy',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/llava-toy', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'drawing-question', presets: 'presets.json', start: START },
	output: { view: 'kit.probbars', layer: 'probs' },
	sources: { fwd: { kind: 'trace' } },
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			I('Say how a frozen image encoder, a small projector and a language model are glued together, and which parts were trained.', '说出一个冻结的图像编码器、一个小投影层和一个语言模型是怎样接在一起的，以及哪些部分被训练过。'),
			I('Follow one drawing from 28×28 numbers to 16 image tokens as wide as the model’s word rows.', '跟着一幅画，从 28×28 个数走到 16 个和模型词行一样宽的图像词元。'),
			I('Read where the answer row looks: how much of its attention goes to the image tokens, and to which patches.', '读出写答案的那一行看向哪里：它的注意力有多少落在图像词元上，又落在哪些图块上。'),
			I('Change the question and watch the answer change; then see what a tiny vocabulary cannot do.', '换一个问题，看答案怎样跟着变；再看看一个很小的词表做不到什么。')
		],
		prequestion: {
			id: 'llava-toy-pre',
			question: I('The language model was trained only on word rows and answers. How can it answer a question about a drawing?', '这个语言模型只学过词行和答案。它怎么能回答关于一幅画的问题？'),
			options: [I('The drawing is turned into a caption first', '先把画变成一句文字描述'), I('The drawing becomes extra rows of numbers placed in front of the question’s words', '画变成额外的几行数，放在问题的词前面'), I('A separate classifier answers instead', '由另一个分类器代替它回答')],
			answer: 1,
			explain: I('The projector turns each patch of the drawing into a row as wide as a word row, and the language model reads those 16 image tokens in its sequence, next to the words. No caption is ever written.', '投影层把画的每个图块变成一行，和词行一样宽；语言模型在它的序列里，把这 16 个图像词元和词放在一起读。中间从来没有写出过一句描述。')
		},
		tryThis: [
			{ id: 'ladder', title: I('A ladder', '一架梯子'), setup: { preset: 'heldout-ladder' }, predict: I('It has no word for “ladder”. What will it answer to “what is this ?”', '它没有“ladder（梯子）”这个词。对“what is this ?”它会怎么回答？'), look: I('It still names one of its 10 things.', '它仍然会说出它那 10 样东西之一。'), goto: { step: 'answer' } },
			{ id: 'ask', title: I('Ask yes or no', '问是不是'), setup: { input: { strokes: HOUSE, question: 'is it a car ?' } }, predict: I('The same house, asked “is it a car ?”. Where does the answer row look now?', '同一幅房子，问“is it a car ?”。写答案的那一行现在看向哪里？'), look: I('Compare its weight on the image tokens with “what is this ?”.', '把它在图像词元上的权重和“what is this ?”时比较。'), goto: { step: 'look' } }
		],
		quiz: [
			{ q: I('Which part of llava-toy never changed during its training?', 'llava-toy 的哪一部分在它的训练中从没改变过？'), options: [I('TinyViT, the image encoder', '图像编码器 TinyViT'), I('The projector', '投影层'), I('The language model', '语言模型')], answer: 0, explain: I('TinyViT was trained earlier on its own and kept frozen; only the projector and the language model learned from the questions.', 'TinyViT 早先单独训练好，之后一直冻结；只有投影层和语言模型从问题中学习。') },
			{ q: I('Why does the projector give 64 numbers per row?', '为什么投影层给出的每一行是 64 个数？'), options: [I('64 is the width of the language model’s word rows', '64 是语言模型词行的宽度'), I('There are 64 patches', '一共有 64 个图块'), I('TinyViT gives 64 numbers per patch', 'TinyViT 每个图块给出 64 个数')], answer: 0, explain: I('Image tokens must be as wide as the word rows to sit in the same sequence; TinyViT gives 48 numbers per patch.', '图像词元必须和词行一样宽，才能放进同一个序列；TinyViT 每个图块给出 48 个数。') }
		],
		misconception: {
			claim: I('A vision-language model first describes the picture in words, then reasons about the words.', '视觉语言模型先用文字描述图片，再根据这些文字推理。'),
			truth: I('The picture enters as rows of numbers (image tokens) inside the language model’s sequence; attention reads them directly.', '图片以几行数（图像词元）的形式进入语言模型的序列；注意力直接读取它们。'),
			step: 'seq'
		},
		sections: { intros: { training: I('Only the projector and the language model were trained, on questions about the training drawings TinyViT itself learned from; TinyViT stayed frozen. The numbers here were measured on the test drawings.', '只有投影层和语言模型被训练，用的是关于 TinyViT 当初所学训练画的问题；TinyViT 保持冻结。这里的数是在测试画上测得的。') } },
		lessonNote: I('Eleven steps: your drawing through the frozen TinyViT, the projector that turns patches into image tokens, the joined sequence, where the answer row looks, the answer, your own question, and what a tiny vocabulary cannot do.', '十一步：你的画穿过冻结的 TinyViT，投影层把图块变成图像词元，拼接起来的序列，写答案的那一行看向哪里，答案，你自己的问题，以及一个很小的词表做不到什么。')
	},
	glossary: [
		{ id: 'vision-language-model', term: I('vision-language model', '视觉语言模型'), def: I('A language model that also reads images, usually as rows of numbers (image tokens) placed in its sequence of word rows.', '也能读图像的语言模型，通常把图像变成几行数（图像词元），放进它的词行序列里。'), related: ['token', 'embedding'] },
		{ id: 'image-token', term: I('image token', '图像词元'), def: I('One row of numbers made from a patch of an image, as wide as a word row, so a language model can read it next to the words.', '由图像的一个图块得到的一行数，和词行一样宽，所以语言模型能把它和词放在一起读。'), related: ['token', 'image-patch'] },
		{ id: 'projector', term: I('projector (connector)', '投影层（连接器）'), def: I('A small trained network that maps an image encoder’s features to the width of a language model’s rows.', '一个训练出来的小网络，把图像编码器的特征映射到语言模型每一行的宽度。'), related: ['dense-layer'] },
	],
	credits: [
		{ name: 'Quick, Draw! dataset', url: 'https://github.com/googlecreativelab/quickdraw-dataset', license: 'CC BY 4.0', usedFor: 'the training and test drawings (TinyCNN’s 10 classes and split) and the preset drawings', attribution: 'Quick, Draw! dataset by Google Creative Lab' },
		{ name: 'LLaVA: Liu et al., Visual Instruction Tuning (2023) and Improved Baselines with Visual Instruction Tuning (2023)', url: 'https://arxiv.org/abs/2304.08485', license: 'Papers (arXiv); ideas only (no code, weights or data used)', usedFor: 'a frozen vision encoder read at its second-to-last layer without the [CLS] row, the two-layer GELU projector, image tokens placed in the language model’s sequence and the loss on the answer only, at toy size', attribution: 'Haotian Liu, Chunyuan Li, Qingyang Wu, Yuheng Li and Yong Jae Lee' }
	],
	catalog: {
		hook: I('Glue a frozen TinyViT to a tiny GPT-style language model with a small projector, then ask about your drawing.', '用一个小投影层把冻结的 TinyViT 接到一个很小的 GPT 式语言模型上，然后就你的画提问。'),
		shows: I(
			'A frozen TinyViT reading your drawing, the two-layer projector worked out for one number, the question’s word rows from the embedding table, image tokens and words in one sequence under a causal mask, where the answer row’s attention falls on your drawing, the one-word answer to your own question, and what a tiny vocabulary cannot say.',
			'冻结的 TinyViT 读你的画、两层投影层中一个数的完整计算、从嵌入表查出的问题词行、在因果掩码下由图像词元和词组成的同一个序列、写答案的那一行的注意力落在你的画的哪里、对你自己的问题给出的一个词的答案，以及很小的词表说不出的东西。'
		),
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		params: I('78,484 trained + 88,080 frozen', '78,484 个可训练参数 + 88,080 个冻结参数'),
		size: I('651 KB of weights', '权重 651 KB'),
		prereqs: ['tinyvit', 'tinygpt']
	}
};

export default manifest;
