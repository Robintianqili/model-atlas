// tiny-encdec: the package manifest (design §4.9). A tiny T5-style encoder-decoder Transformer (one encoder block, one
// decoder block, d = 32, two heads), trained offline (pipeline/tiny-encdec) on tinyseq2seq's date task, so the two can
// be compared: a date in one of eight formats becomes ISO 8601 (2021-03-05).
//   fwd     the exported model (export.py) traced on the learner's date: the encoder block in kit ops, the local op
//           'greedy' (greedy decoding; its output is the decoder input), then the decoder block in kit ops on it:
//           causal self-attention, cross-attention to the encoder's memory, feed-forward, head, softmax
//   decode  greedy decoding one event at a time (process.ts, float64; pipeline/tiny-encdec/reference.py is its twin)
// Measured numbers (test accuracy, formats it never saw, where cross-attention looks, the GRU's numbers) are in
// facts.json (train.py, export.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { decodeInput } from './process.ts';

const manifest: ExplainerManifest = {
	slug: 'tiny-encdec',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tiny-encdec', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'date-text', presets: 'presets.json', start: 'March 5, 2021' },
	output: { view: 'tiny-encdec/cross' },
	sources: {
		fwd: { kind: 'trace' },
		decode: { kind: 'process', process: 'decode', input: decodeInput }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{ en: 'Say how the encoder’s self-attention lets every character of your date read every other one, in both directions and all at once.', zh: '说出编码器的自注意力怎样让你日期里的每个字符同时读到所有其他字符，两个方向都可以。' },
			{ en: 'Explain the decoder’s causal mask: why a step may read only what was already written.', zh: '解释解码器的因果掩码：为什么一步只能读已经写下的东西。' },
			{ en: 'Work out one cross-attention step with real numbers: a query from the decoder, keys and values from the encoder, the score, the softmax and the mix.', zh: '用真实的数算出一步交叉注意力：查询来自解码器，键和值来自编码器，再算分数、softmax 和混合。' },
			{ en: 'Compare it with the GRU encoder–decoder on the same task, and find formats where both fail.', zh: '在同一个任务上把它和 GRU 编码器–解码器比一比，找出两者都会出错的格式。' }
		],
		prequestion: {
			id: 'tiny-encdec-pre', question: { en: 'To turn “March 5, 2021” into “2021-03-05”, the model must write the year first, but the year comes last in your text. How does a Transformer get to it?', zh: '要把“March 5, 2021”变成“2021-03-05”，模型得先写年份，可年份在你的文字最后。Transformer 怎样拿到它？' },
			options: [{ en: 'Each output step can read any input character directly, through cross-attention', zh: '每个输出步骤都能通过交叉注意力直接读任何一个输入字符' }, { en: 'The encoder reverses the text first', zh: '编码器先把文字倒过来' }, { en: 'It waits for a state that has read the whole text, as a recurrent network does', zh: '它像循环网络那样，等一个读完整段文字的状态' }],
			answer: 0, explain: { en: 'Cross-attention: at every step the decoder scores all the encoder’s character vectors and reads a weighted mix of them, wherever they are. Lesson steps 5 to 7 work one such step out.', zh: '靠交叉注意力：每一步，解码器都给编码器的所有字符向量打分，再按权重读它们的混合，不管它们在哪里。课程的第 5 到第 7 步会把这样的一步算出来。' }
		},
		tryThis: [
			{ id: 'may', title: { en: 'March or May?', zh: 'March 还是 May？' }, setup: { input: 'May 5, 2021' },
				predict: { en: 'Mar and May share “Ma”. Which characters will the row of the month’s last digit read now?', zh: 'Mar 和 May 都有“Ma”。现在月份最后一位数字那一行会读哪些字符？' },
				look: { en: 'Find the row of the month’s last digit (just before the second dash) and the column where its color is strongest.', zh: '找到月份最后一位数字那一行（第二个短横线前面那一行），看它在哪一列颜色最强。' }, goto: { step: 'decode' } },
			{ id: 'dots', title: { en: 'Numbers only', zh: '只有数字' }, setup: { input: '05.03.2021' },
				predict: { en: 'No month name at all. Which characters will the month’s rows read?', zh: '完全没有月份名字。月份那几行会读哪些字符？' },
				look: { en: 'Compare with “March 5, 2021”: the same answer, read from other places.', zh: '和“March 5, 2021”比一比：同样的答案，从别的地方读出来。' }, goto: { step: 'decode' } }
		],
		quiz: [
			{ q: { en: 'In the decoder’s self-attention, why is the upper-right triangle of every head empty?', zh: '在解码器的自注意力里，为什么每个头的右上三角都是空的？' }, options: [{ en: 'A position may read only itself and earlier positions: later symbols are not written yet', zh: '一个位置只能读自己和之前的位置：后面的符号还没写出来' }, { en: 'Those weights happened to be small', zh: '那些权重碰巧很小' }, { en: 'The decoder reads the input right to left', zh: '解码器从右往左读输入' }],
				answer: 0, explain: { en: 'The causal mask sets those scores to −∞, so their weights are exactly 0. When the model writes step t, the symbols after t do not exist yet, so it must never learn to use them.', zh: '因果掩码把这些分数设成 −∞，所以它们的权重正好是 0。模型写第 t 步时，t 之后的符号还不存在，所以绝不能学会用它们。' } },
			{ q: { en: 'In cross-attention, where do the queries, keys and values come from?', zh: '在交叉注意力里，查询、键和值分别来自哪里？' }, options: [{ en: 'Queries from the decoder; keys and values from the encoder’s output', zh: '查询来自解码器；键和值来自编码器的输出' }, { en: 'All three from the decoder', zh: '三者都来自解码器' }, { en: 'Queries from the encoder; keys and values from the decoder', zh: '查询来自编码器；键和值来自解码器' }],
				answer: 0, explain: { en: 'The decoder asks (the query), and every character of the input offers a key to be matched and a value to be read. That is how each output step reads the input.', zh: '解码器提问（查询），输入的每个字符提供一个用来匹配的键和一个被读取的值。每个输出步骤就是这样读输入的。' } }
		],
		misconception: {
			claim: { en: 'The Transformer reads the date left to right, like the GRU.', zh: 'Transformer 像 GRU 一样从左往右读日期。' },
			truth: { en: 'Its encoder reads every character at once, and each character can read the ones after it as well as the ones before it (lesson step 2). Order comes only from the position vectors added to the characters.', zh: '它的编码器一次读入所有字符，每个字符既能读它前面的，也能读它后面的（课程第 2 步）。顺序只来自加在字符上的位置向量。' },
			step: 'self'
		},
		lessonNote: { en: 'Eleven steps: the encoder reads your date with self-attention, the decoder writes the answer with causal self-attention and cross-attention, one cross-attention step is worked out with real numbers, and you compare it with the GRU on dates it never saw.', zh: '十一步：编码器用自注意力读你的日期，解码器用因果自注意力和交叉注意力写出答案；用真实的数算出一步交叉注意力，再在它没见过的日期上和 GRU 比一比。' }
	},
	credits: [
		{ name: 'Attention Is All You Need (ideas)', url: 'https://arxiv.org/abs/1706.03762', license: 'ideas only (paper; no code or text used)', usedFor: 'the encoder-decoder Transformer: scaled dot-product attention, several heads, the causal mask in the decoder and cross-attention to the encoder; code written fresh', attribution: 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin (2017)' },
		{ name: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5; ideas)', url: 'https://arxiv.org/abs/1910.10683', license: 'ideas only (paper; no code or text used)', usedFor: 'the T5-style choices: text to text, pre-norm blocks, no biases in the attention and feed-forward layers, a ReLU feed-forward (positions and normalization differ; see the notes)', attribution: 'Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li and Peter J. Liu (2020)' }
	],
	catalog: {
		runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' }, params: { en: '25,344', zh: '25,344' }, size: { en: '99 KB of weights', zh: '权重 99 KB' },
		hook: { en: 'The same date task as the GRU, with self-attention instead of a recurrent state.', zh: '和 GRU 同一个日期任务，用自注意力代替循环状态。' },
		shows: { en: 'Encoder self-attention in both directions, the decoder’s causal self-attention, one cross-attention score, softmax and mix worked out with real numbers, greedy decoding step by step, and a measured comparison with the GRU.', zh: '双向的编码器自注意力、解码器的因果自注意力、用真实数值算出的一个交叉注意力分数及其 softmax 和混合、一步一步的贪心解码，以及与 GRU 的实测对比。' }
	},
	glossary: [
		{ id: 'self-attention', term: { en: 'self-attention', zh: '自注意力' }, def: { en: 'Attention inside one sequence: every position makes a query and matches it against the keys of the same sequence’s positions, then reads a weighted mix of their values.', zh: '同一个序列内部的注意力：每个位置给出一个查询，和同一序列各位置的键去匹配，再按权重读取它们的值的混合。' }, related: ['attention', 'query-key-value', 'cross-attention'] },
		{ id: 'cross-attention', term: { en: 'cross-attention', zh: '交叉注意力' }, def: { en: 'Attention between two sequences: the queries come from one (the decoder’s positions), the keys and values from the other (the encoder’s output), so each output step reads the input.', zh: '两个序列之间的注意力：查询来自一个序列（解码器的各个位置），键和值来自另一个（编码器的输出），于是每个输出步骤都能读输入。' }, related: ['attention', 'encoder-decoder', 'self-attention'] }
	]
};

export default manifest;
