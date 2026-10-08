// tinyseq2seq: the package manifest (design §4.9). A GRU encoder-decoder with Bahdanau (additive) attention, trained
// offline on a synthetic task (pipeline/tinyseq2seq): a date in one of eight formats becomes ISO 8601 (2021-03-05).
//   fwd     the exported model (export.py) traced on the learner's date: encoder states, then 11 decoder steps with
//           their keys, queries, scores, weights, context vectors, states and chosen symbols (local ops gru, attn_decoder)
//   decode  greedy decoding one event at a time (process.ts, float64; pipeline/tinyseq2seq/reference.py is its twin)
// Measured numbers (test accuracy, the no-attention baseline, formats it never saw) are in facts.json (train.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { decodeInput } from './process.ts';

const manifest: ExplainerManifest = {
	slug: 'tinyseq2seq',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tinyseq2seq', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'date-text', presets: 'presets.json', start: 'March 5, 2021' },
	output: { view: 'tinyseq2seq/align' },
	sources: {
		fwd: { kind: 'trace' },
		decode: { kind: 'process', process: 'decode', input: decodeInput }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{ en: 'Say how the encoder turns your date into one state per character, and why its last state alone is a bottleneck.', zh: '说出编码器怎样把你的日期变成每个字符一个状态，以及为什么只用它的最后一个状态会成为瓶颈。' },
			{ en: 'Work out one attention step with real numbers: a score per character, the softmax weights and the context vector.', zh: '用真实的数算出一步注意力：每个字符一个分数、softmax 权重和上下文向量。' },
			{ en: 'Read the alignment matrix: which characters of your date each output character came from.', zh: '读懂对齐矩阵：输出的每个字符来自你日期里的哪些字符。' },
			{ en: 'Test the model on your own dates, and explain why it fails on formats it never saw.', zh: '用你自己的日期测试模型，并解释为什么它在没见过的格式上会出错。' }
		],
		prequestion: {
			id: 'tinyseq2seq-pre', question: { en: 'The model turns “March 5, 2021” into “2021-03-05”, one character at a time. To write the month, 03, what must it look at?', zh: '模型把“March 5, 2021”一个字符一个字符地变成“2021-03-05”。要写出月份 03，它必须看什么？' },
			options: [{ en: 'The month’s name, at least enough of it to tell March from May', zh: '月份的名字，至少要看到足以区分 March 和 May 的部分' }, { en: 'The year, 2021', zh: '年份 2021' }, { en: 'Nothing: it learned that the answer is usually 03', zh: '什么都不用看：它学到了答案通常是 03' }],
			answer: 0, explain: { en: 'March and May both start with “Ma”, so the month needs the third letter. Attention lets each output step look back at the characters it needs; step 9 shows where it looks for every output character.', zh: 'March 和 May 都以“Ma”开头，所以月份要看第三个字母。注意力让每一个输出步骤回头去看它需要的字符；第 9 步会显示它为每个输出字符看了哪里。' }
		},
		tryThis: [
			{ id: 'may', title: { en: 'March or May?', zh: 'March 还是 May？' }, setup: { input: 'May 5, 2021' },
				predict: { en: 'Mar and May share “Ma”. Where will the row of the month’s last digit look now?', zh: 'Mar 和 May 都有“Ma”。现在月份最后一位数字那一行会看哪里？' },
				look: { en: 'Find the row of the month’s last digit (the row just before the second dash) and the column it is darkest in.', zh: '在矩阵里找到月份最后一位数字那一行（第二个短横线上面那一行），看它在哪一列最深。' }, goto: { step: 'align' } },
			{ id: 'dots', title: { en: 'Numbers only', zh: '只有数字' }, setup: { input: '05.03.2021' },
				predict: { en: 'No month name at all. Which characters will the month’s row look at?', zh: '完全没有月份名字。月份那一行会看哪些字符？' },
				look: { en: 'Compare with “March 5, 2021”: the same answer, read from other places.', zh: '和“March 5, 2021”比一比：同样的答案，从别的地方读出来。' }, goto: { step: 'align' } }
		],
		quiz: [
			{ q: { en: 'At one decoder step, one character gets an attention weight of 0.9. What does that mean?', zh: '在解码器的某一步，一个字符得到 0.9 的注意力权重。这是什么意思？' }, options: [{ en: 'The context vector at that step is mostly that character’s encoder state', zh: '这一步的上下文向量主要就是这个字符的编码器状态' }, { en: 'That character will be copied into the output', zh: '这个字符会被原样抄到输出里' }, { en: 'The model is 90% sure of its answer', zh: '模型对答案有 90% 的把握' }],
				answer: 0, explain: { en: 'The weights mix the encoder states: with 0.9 on one character, the context vector is 90% that character’s state. What the step writes is decided afterwards, from the context and the decoder’s state.', zh: '权重用来混合编码器状态：一个字符占 0.9，上下文向量就有 90% 来自这个字符的状态。这一步写什么，是之后根据上下文和解码器状态决定的。' } },
			{ q: { en: 'Why does a decoder without attention have a bottleneck?', zh: '为什么没有注意力的解码器有瓶颈？' }, options: [{ en: 'It sees the whole input only through the encoder’s last state, a fixed number of numbers', zh: '它只能通过编码器的最后一个状态看到整个输入，而那只是固定数量的数' }, { en: 'It reads the input more slowly', zh: '它读输入读得更慢' }, { en: 'It has fewer output symbols', zh: '它的输出符号更少' }],
				answer: 0, explain: { en: 'However long the input, everything must squeeze through the same few numbers. Attention lets every decoder step look at all encoder states instead.', zh: '不管输入多长，所有信息都得挤过同样几个数。注意力则让解码器的每一步都能看到所有编码器状态。' } }
		],
		misconception: {
			claim: { en: 'Attention means the model understands dates.', zh: '有了注意力，模型就理解了日期。' },
			truth: { en: 'Attention is a learned way to look things up. On formats it never saw, the model still writes an answer, often a wrong one, and it never says it is unsure (step 11).', zh: '注意力是一种学来的查找方式。遇到没见过的格式，模型照样写出一个答案，常常是错的，而且从不说自己没把握（第 11 步）。' },
			step: 'limits'
		},
		lessonNote: { en: 'Eleven steps: the encoder reads your date, the decoder looks back at it with attention one output character at a time, and you watch the alignment matrix fill in; then test your own dates.', zh: '十一步：编码器读你的日期，解码器用注意力一个字符一个字符地回看它，你看着对齐矩阵一行行填满；然后用你自己的日期来测试。' },
		sections: { hide: ['unit'] }
	},
	credits: [
		{ name: 'Neural Machine Translation by Jointly Learning to Align and Translate (ideas)', url: 'https://arxiv.org/abs/1409.0473', license: 'ideas only (paper; no code or text used)', usedFor: 'additive (Bahdanau) attention: scores v·tanh(W_q s + W_k h), softmax weights, the context vector; code written fresh', attribution: 'Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio (2015)' },
		{ name: 'Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (ideas)', url: 'https://arxiv.org/abs/1406.1078', license: 'ideas only (paper; no code or text used)', usedFor: 'the GRU and the encoder-decoder without attention (the bottleneck baseline); PyTorch’s nn.GRU is used as is', attribution: 'Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk and Yoshua Bengio (2014)' }
	],
	catalog: { runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' }, params: { en: '16,941', zh: '16,941' }, size: { en: '66 KB of weights', zh: '权重 66 KB' }, shows: { en: 'Encoder states, attention scores, softmax weights and the context vector with real numbers, the alignment matrix filling in, and greedy decoding step by step.', zh: '编码器状态、注意力分数、softmax 权重和上下文向量（真实数值）、逐行填满的对齐矩阵，以及一步一步的贪心解码。' } },
	glossary: [
		{ id: 'encoder-decoder', term: { en: 'encoder-decoder (seq2seq)', zh: '编码器–解码器（seq2seq）' }, def: { en: 'A model in two parts: the encoder reads the whole input into numbers, the decoder writes the output one item at a time from those numbers. Seq2seq is short for sequence to sequence.', zh: '由两部分组成的模型：编码器把整个输入读成一些数，解码器再根据这些数一次写出一项输出。Seq2seq 是“序列到序列”的缩写。' }, related: ['attention', 'recurrent-network'] },
		{ id: 'context-vector', term: { en: 'context vector', zh: '上下文向量' }, def: { en: 'In attention, the weighted mix of the encoder states that one decoder step reads: each state is multiplied by its attention weight and the results are added up.', zh: '在注意力里，解码器某一步读到的编码器状态的加权混合：每个状态乘以它的注意力权重，再加起来。' }, related: ['attention', 'encoder-decoder'] },
		{ id: 'greedy-decoding', term: { en: 'greedy decoding', zh: '贪心解码' }, def: { en: 'Writing an output by always taking the item with the largest chance at each step, and feeding it back in as the next step’s input.', zh: '每一步都选概率最大的那一项来写输出，并把它作为下一步的输入送回去。' }, related: ['sampling', 'autoregressive'] },
		{ id: 'recurrent-network', term: { en: 'recurrent network (RNN)', zh: '循环神经网络（RNN）' }, def: { en: 'A network that reads a sequence one item at a time and passes a memory from each step to the next, using the same weights at every step.', zh: '一次读序列中的一项、并把一份记忆从每一步传到下一步的网络；每一步用的是同一组权重。' }, related: ['hidden-state'] },
		{ id: 'hidden-state', term: { en: 'hidden state (h)', zh: '隐藏状态（h）' }, def: { en: 'The numbers a recurrent network passes to the next step and to its output after each item it reads.', zh: '循环网络每读一项之后，传给下一步、也用来产生输出的那组数。' }, related: ['recurrent-network'] }
	]
};

export default manifest;
