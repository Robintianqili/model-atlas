// crnn-ocr: the package manifest. A tiny CRNN (two convolutions, a bidirectional LSTM over 32 columns, 11 chances per
// column) trained with the CTC loss on synthetic handwriting-like numbers (glyphs.ts, make_data.ts), read on the
// learner's own writing:
//   fwd      the exported model (pipeline/crnn-ocr/export.py) traced on the learner's strip of 16 × 64 ink values
//   decode   CTC greedy decoding, one frame per column (process.ts; Python reference with PyTorch's CTC loss)
// Measured numbers (test accuracy, messier words, the blank's share of columns) are in facts.json (export.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { decodeInput } from './process.ts';
import { presetRaw } from './adapters/word-strokes/adapter.ts';

const manifest: ExplainerManifest = {
	slug: 'crnn-ocr',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/crnn-ocr', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'word-strokes', presetsFromAdapter: true, start: presetRaw('2048') },
	sources: {
		fwd: { kind: 'trace' },
		decode: { kind: 'process', process: 'decode', input: decodeInput }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{ en: 'Say how a strip of handwriting becomes a sequence of 32 columns, and what the convolutions and the two LSTMs add to each column.', zh: '说出一条手写的长条怎样变成 32 列的序列，以及卷积和两个 LSTM 给每一列加了什么。' },
			{ en: 'Decode a row of per-column symbols by hand with CTC’s rule: merge repeats, then drop blanks.', zh: '用 CTC 的规则亲手解码一行逐列的符号：先合并重复，再去掉空白。' },
			{ en: 'Explain why CTC needs the blank symbol, and why the model can be trained without saying which column holds which digit.', zh: '解释 CTC 为什么需要空白符号，以及为什么训练时不必告诉模型哪一列是哪个数字。' },
			{ en: 'Find writing the model misreads, and say what it cannot read at all.', zh: '找到模型会读错的写法，并说出它根本读不了什么。' }
		],
		prequestion: {
			id: 'crnn-ocr-pre',
			question: { en: 'Nobody tells this model where one digit ends and the next begins. How can it still read “2048”?', zh: '没有人告诉这个模型一个数字在哪里结束、下一个从哪里开始。它怎么还能读出“2048”？' },
			options: [
				{ en: 'It first cuts the writing into one box per digit, then reads each box', zh: '它先把字切成每个数字一个框，再逐个框去读' },
				{ en: 'It guesses a symbol for every thin column, and a fixed rule turns that row of guesses into text', zh: '它给每一窄列猜一个符号，再用一条固定的规则把这一排猜测变成文字' },
				{ en: 'It compares the whole picture with stored pictures of numbers', zh: '它把整幅图和存好的数字图片比较' }
			],
			answer: 1,
			explain: {
				en: 'The model never cuts the writing. It scores 11 symbols in each of 32 columns; CTC’s rule (merge repeats, drop blanks) turns the row of top symbols into text, and training only ever sees the whole word.',
				zh: '模型从不切分字迹。它在 32 列的每一列里给 11 个符号打分；CTC 的规则（合并重复、去掉空白）把这一排排第一的符号变成文字，训练时也只看到整个词。'
			}
		},
		lessonNote: {
			en: 'Eleven steps: your writing becomes columns, two LSTMs read them both ways, each column scores 11 symbols, and CTC turns the columns into text.',
			zh: '十一步：你写的字变成一列列，两个 LSTM 从两头读它们，每一列给 11 个符号打分，CTC 再把这些列变成文字。'
		}
	},
	credits: [
		{
			name: 'CRNN (Shi, Bai and Yao, 2015)',
			url: 'https://arxiv.org/abs/1507.05717',
			license: 'Paper (arXiv); ideas only (no code, weights or data used)',
			usedFor: 'the architecture: convolutional features cut into columns, a bidirectional LSTM over the columns and a CTC output, reimplemented at toy size',
			attribution: 'Baoguang Shi, Xiang Bai and Cong Yao, “An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition”'
		},
		{
			name: 'Connectionist temporal classification (Graves et al., ICML 2006)',
			url: 'https://www.cs.toronto.edu/~graves/icml_2006.pdf',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'the blank symbol, the collapse rule, the forward algorithm over the label with blanks and the CTC loss (training uses PyTorch’s ctc_loss; the page’s own forward algorithm is checked against it)',
			attribution: 'Alex Graves, Santiago Fernández, Faustino Gomez and Jürgen Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks”'
		}
	],
	glossary: [
		{ id: 'ocr', term: { en: 'OCR (optical character recognition)', zh: 'OCR（光学字符识别）' }, def: { en: 'Turning a picture of writing or print into text a computer can use.', zh: '把手写或印刷文字的图片变成计算机能用的文字。' }, related: ['ctc'] },
		{ id: 'ctc', term: { en: 'CTC (connectionist temporal classification)', zh: 'CTC（连接时序分类）' }, def: { en: 'A way to read a sequence of frames (here: columns) as a shorter text without saying which frame holds which character: every frame scores the characters and a blank, repeats are merged, blanks dropped, and training adds up the chances of every path that spells the right text.', zh: '一种把一串帧（这里是列）读成更短文字的方法，不需要说明哪一帧对应哪个字符：每一帧给各个字符和空白打分，合并重复、去掉空白，训练时把拼出正确文字的所有路径的概率加起来。' }, related: ['ctc-blank', 'ctc-path'] },
		{ id: 'ctc-blank', term: { en: 'blank (CTC)', zh: '空白（CTC）' }, def: { en: 'CTC’s extra symbol for “no new character in this frame”. It is dropped when the frames become text, and it separates a character from a repeat of itself (1 – 1 reads 11).', zh: 'CTC 额外加的符号，表示“这一帧没有新字符”。帧变成文字时它会被丢掉；它还能把一个字符和紧跟着的同一个字符隔开（1 – 1 读作 11）。' }, related: ['ctc'] },
		{ id: 'ctc-path', term: { en: 'path (CTC alignment)', zh: '路径（CTC 对齐）' }, def: { en: 'One symbol for every frame. Many paths collapse to the same text; the chance of a text is the sum of the chances of all its paths.', zh: '给每一帧选一个符号。许多条路径会合并成同一段文字；一段文字的概率是它所有路径概率的总和。' }, related: ['ctc', 'ctc-blank'] },
		{ id: 'bidirectional-rnn', term: { en: 'bidirectional RNN', zh: '双向 RNN' }, def: { en: 'Two recurrent networks over the same sequence, one reading it forwards and one backwards; their states are joined, so every position knows what comes before and after it.', zh: '在同一个序列上的两个循环网络，一个从前往后读，一个从后往前读；把它们的状态拼起来，每个位置就同时知道前面和后面有什么。' }, related: ['recurrent-network', 'lstm'] }
	],
	catalog: {
		hook: { en: 'Write a number and watch digits emerge from a row of column features.', zh: '写一个数，看数字怎样从一列列特征里浮现出来。' },
		shows: { en: 'Your writing as a strip of pixels, convolution features cut into 32 columns, two LSTMs reading them both ways, 11 symbol chances per column, greedy CTC decoding (merge repeats, drop blanks), the many paths that spell one text and the CTC loss summed over them, and letters it can only read as digits.', zh: '你写的字变成的像素长条、切成 32 列的卷积特征、从两头读这些列的两个 LSTM、每一列 11 个符号的概率、CTC 贪心解码（合并重复、去掉空白）、拼出同一段文字的许多条路径和把它们加起来的 CTC 损失，以及它只能读成数字的字母。' },
		runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' },
		params: { en: '43,435', zh: '43,435' },
		size: { en: '170 KB of weights', zh: '权重 170 KB' }
	}
};

export default manifest;
