// char-rnn: the package manifest (design §4.9, §7). A character-level LSTM (74 characters, 24-number embeddings, 64
// units, 29,626 parameters) trained on TinyStories, read on the learner's own text:
//   fwd     the exported model (pipeline/char-rnn/export.py) traced on the learner's text, gate sub-steps lstm#i..h
//   unroll  the same LSTM worked out one character at a time (process.ts; four frames per character)
//   sample  text written one character at a time, drawn by chance (temperature 1); greedy: always the top character
// Measured numbers (validation loss, what unit 51 does, forget-gate statistics) are in facts.json (units.py); the top
// contexts of every unit are in unitTop.json (precomputed on held-out stories).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { rnnInput as input } from './rnn.ts';

const manifest: ExplainerManifest = {
	slug: 'char-rnn',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/char-rnn', auto: ['spec.json', 'weights.bin', 'facts.json', 'unitTop.json'] },
	input: { adapter: 'text-presets', presets: 'presets.json', start: 'Once upon a time, there was a little girl named Lily.' },
	output: { view: 'kit.probbars' },
	sources: {
		fwd: { kind: 'trace' },
		unroll: { kind: 'process', process: 'unroll', input },
		sample: { kind: 'process', process: 'sample', seed: 1, opt: { temperature: 1 }, input },
		greedy: { kind: 'process', process: 'sample', seed: 1, opt: { temperature: 0 }, input }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{ en: 'Say what the hidden state h and the cell state c of an LSTM hold, and how they pass from one character to the next.', zh: '说出 LSTM 的隐藏状态 h 和细胞状态 c 存的是什么，以及它们怎样从一个字符传到下一个字符。' },
			{ en: 'Work out c = f·c + i·g for one unit with real numbers, and say what a forget gate near 0 or near 1 does.', zh: '用真实的数算出一个单元的 c = f·c + i·g，并说出遗忘门接近 0 或接近 1 时会发生什么。' },
			{ en: 'Explain how the model writes text by drawing one character at a time, and why always taking the top character differs from drawing.', zh: '解释模型怎样一次抽一个字符来写文字，以及总选第一名和按概率抽取有什么不同。' },
			{ en: 'Estimate how fast the memory fades from the forget gates.', zh: '根据遗忘门估计记忆消退得有多快。' }
		],
		prequestion: {
			id: 'char-rnn-pre',
			question: { en: 'The model reads “Once upon a tim”. How does it decide what comes next?', zh: '模型读到“Once upon a tim”。它怎样决定下一个字符？' },
			options: [
				{ en: 'It looks the whole phrase up in a list of stories', zh: '它在一份故事清单里查这整句话' },
				{ en: 'It keeps a small memory of numbers, updated once per character, and scores every character from it', zh: '它保留一份由数组成的小记忆，每读一个字符更新一次，再根据它给每个字符打分' },
				{ en: 'It only looks at the last character, “m”', zh: '它只看最后一个字符“m”' }
			],
			answer: 1,
			explain: {
				en: 'An LSTM carries two lists of 64 numbers from character to character. Everything it read so far has to fit into them, and the next-character scores are computed from h alone.',
				zh: 'LSTM 把两组各 64 个数从一个字符带到下一个字符。它读过的一切都必须装进这些数里，下一个字符的分数只由 h 算出。'
			}
		},
		misconception: {
			claim: { en: 'The network looks at the whole text at once, like a reader glancing at a page.', zh: '网络像读者扫一眼整页那样，一次看到全部文字。' },
			truth: { en: 'It reads one character at a time with the same weights, and everything earlier survives only in its two lists of 64 numbers.', zh: '它用同一组权重一次读一个字符，前面的一切只能靠那两组各 64 个数保存下来。' },
			step: 'unroll'
		},
		// the lesson walks through every layer; the inspector would draw a [T, 24..74] layer as a table up to 128 rows tall
		sections: { hide: ['inspector'] },
		lessonNote: {
			en: 'Eleven steps: your text becomes ids, the LSTM keeps a memory, one unit is worked out gate by gate, then the model guesses, writes text and shows how its memory fades.',
			zh: '十一步：你的文字变成 id，LSTM 保留一份记忆，一个单元逐个门地算出来；然后模型猜下一个字符、写出文字，并展示它的记忆怎样消退。'
		}
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'Training text (39.3 million characters from the start of the train split) and held-out validation text for the unit analysis of the char-rnn model',
			attribution: "Ronen Eldan and Yuanzhi Li, 'TinyStories: How Small Can Language Models Be and Still Speak Coherent English?' (2023)"
		}
	],
	glossary: [
		{
			id: 'recurrent-network',
			term: { en: 'recurrent network (RNN)', zh: '循环神经网络（RNN）' },
			def: { en: 'A network that reads a sequence one item at a time and passes a memory from each step to the next, using the same weights at every step.', zh: '一次读序列中的一项、并把一份记忆从每一步传到下一步的网络；每一步用的是同一组权重。' },
			related: ['hidden-state', 'cell-state']
		},
		{
			id: 'hidden-state',
			term: { en: 'hidden state (h)', zh: '隐藏状态（h）' },
			def: { en: 'The numbers a recurrent network passes to the next step and to its output after each item it reads.', zh: '循环网络每读一项之后，传给下一步、也用来产生输出的那组数。' },
			related: ['recurrent-network', 'cell-state']
		},
		{
			id: 'cell-state',
			term: { en: 'cell state (c)', zh: '细胞状态（c）' },
			def: { en: 'An LSTM’s inner memory: at each step it keeps a share of its old value (the forget gate) and adds a new one (the input gate times the candidate).', zh: 'LSTM 的内部记忆：每一步保留旧值的一部分（遗忘门），再加上一个新值（输入门乘以候选值）。' },
			related: ['gate', 'hidden-state']
		},
		{
			id: 'gate',
			term: { en: 'gate (LSTM)', zh: '门（LSTM）' },
			def: { en: 'A number between 0 and 1, made by a sigmoid, that scales how much passes: the forget gate scales the old memory, the input gate the new value, the output gate what is shown.', zh: '由 sigmoid 算出的 0 到 1 之间的数，决定通过多少：遗忘门缩放旧记忆，输入门缩放新值，输出门缩放露出来的部分。' },
			related: ['cell-state', 'activation-function']
		}
	],
	catalog: {
		runtime: { en: 'Own TS (atlas kit executor) in the browser', zh: '自研 TS（atlas 套件执行器），在浏览器中运行' },
		params: { en: '29,626', zh: '29,626' },
		size: { en: '116 KB of weights', zh: '权重 116 KB' }
	}
};

export default manifest;
