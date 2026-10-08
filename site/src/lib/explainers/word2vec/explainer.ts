// word2vec: the manifest. Skip-gram on the learner's own mini-corpus, trained live (kind B) after a sliding window
// turns the sentences into (center, context) pairs (kind C). Every training source has the same trainer ('sg'):
//   pairs  process 'pairs': the window, word by word (no Python reference: claim tests; notes, kit issue 4)
//   one    process 'fit', fine detail, 1 step: the very first training step, slowed down (forward, loss, backward, update)
//   run    process 'fit', 3,000 steps: the same run as `one` (same seed), on to the end, a snapshot every 25 steps
//   run2   the same training from another random start (SEED2): the limits step compares the two
//   train  trainer 'sg' as a train source: the twin's parity run (pipeline/word2vec/twin.py) and the Training section
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { PETS, SEED, SEED2 } from './corpus.ts';

const L = (en: string, zh: string) => ({ en, zh });
const fit = (seed: number, more: Record<string, unknown> = {}) => ({ kind: 'process' as const, process: 'fit', seed, ...more });

const manifest: ExplainerManifest = {
	slug: 'word2vec',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/word2vec', auto: [] },
	input: { adapter: 'corpus', presets: 'presets.json', start: PETS },
	sources: {
		pairs: { kind: 'process', process: 'pairs', run: 'main' },
		one: fit(SEED, { run: 'main', detail: 'fine', opt: { steps: 1 } }),
		run: fit(SEED, { run: 'worker' }),
		run2: fit(SEED2, { run: 'worker' }),
		train: { kind: 'train', trainer: 'sg', detail: 'coarse', opt: { seed: SEED } }
	},
	page: {
		goals: [
			L('Turn a few sentences into (center, context) training pairs with a sliding window, and count them.', '用滑动窗口把几个句子变成（中心词，上下文词）训练对，并数出有多少对。'),
			L('Follow one prediction with its numbers: one-hot input, a row of W_in, a score for every word, softmax, loss.', '带着具体数字走完一次预测：独热输入、W_in 的一行、每个词一个分数、softmax、损失。'),
			L('Explain why one SGD step moves the center word’s vector and the context word’s output vector toward each other.', '解释为什么一步 SGD 会让中心词的向量和上下文词的输出向量互相靠近。'),
			L('Predict which words end up close on the map, and say why a tiny corpus gives noisy neighbours.', '预测哪些词最后会在图上靠在一起，并说明为什么很小的语料给出的近邻很不稳定。')
		],
		prequestion: {
			id: 'word2vec-pre',
			question: L('word2vec learns a vector for every word from text alone. In “The cat was happy. The dog was happy.”, which two words should get similar vectors?', 'word2vec 只从文本中为每个词学一个向量。在 “The cat was happy. The dog was happy.” 里，哪两个词应该得到相似的向量？'),
			options: [L('“cat” and “was”: they stand side by side', '“cat” 和 “was”：它们紧挨着'), L('“cat” and “dog”: they stand in the same places', '“cat” 和 “dog”：它们出现在同样的位置'), L('No two: the model knows nothing about meaning', '没有：模型对意思一无所知')],
			answer: 1,
			explain: L(
				'Each word’s vector is trained to predict the words around it. “cat” and “dog” have the same neighbours, so they need the same prediction and their vectors become alike. The model knows nothing about meaning; sharing neighbours is all it sees.',
				'每个词的向量都被训练来预测它周围的词。“cat” 和 “dog” 的邻居相同，需要做出相同的预测，所以它们的向量会变得相像。模型对意思一无所知；它看到的只有“邻居相同”这一点。'
			)
		},
		tryThis: [
			{
				id: 'new-word',
				title: L('A new animal', '一种新动物'),
				setup: { input: { text: 'The cat was very happy. The cat was sad. The dog was sad. The dog was very happy. The fox was sad. The fox was very happy. The cat wanted to play. The dog wanted to play.', window: 2 } },
				predict: L('“fox” appears only where “cat” and “dog” appear, but never with “wanted”. Where will it land on the map?', '“fox” 只出现在 “cat” 和 “dog” 出现的位置，但从不和 “wanted” 一起出现。它会落在图上的哪里？'),
				look: L('After training, find fox in the list under the map: which word is nearest to it?', '训练之后，在图下面的列表里找到 fox：离它最近的是哪个词？'),
				goto: { step: 'window' }
			}
		],
		quiz: [
			{
				q: L('A vocabulary of 20 words and vectors of 5 numbers: how many numbers do W_in and W_out hold together?', '词表有 20 个词，向量有 5 个数：W_in 和 W_out 一共有多少个数？'),
				options: [L('100', '100'), L('200', '200'), L('25', '25')],
				answer: 1,
				explain: L('Each table has one row of 5 numbers per word: 20 × 5 = 100, and there are two tables.', '每张表为每个词存一行 5 个数：20 × 5 = 100，一共两张表。')
			}
		],
		misconception: {
			claim: L('Words that stand next to each other get similar vectors.', '紧挨着的词会得到相似的向量。'),
			truth: L('Words that have the same neighbours get similar vectors. “cat” and “was” stand side by side, but “cat” ends up near “dog” and “bird”, which stand in the same places.', '邻居相同的词才会得到相似的向量。“cat” 和 “was” 紧挨着，但 “cat” 最后靠近的是 “dog” 和 “bird”：它们出现在同样的位置。'),
			step: 'together'
		},
		lessonNote: L(
			'Eleven steps: your sentences become pairs, one training step slowed down (one-hot, a row of W_in, scores, softmax, loss, update), then 3,000 steps live, the window, and why a tiny corpus gives noisy neighbours.',
			'十一步：你的句子变成训练对；放慢看一步训练（独热、W_in 的一行、分数、softmax、损失、更新）；然后实时训练 3,000 步、改变窗口；最后看为什么很小的语料给出的近邻不稳定。'
		),
		sections: { titles: { training: L('Train it yourself', '自己训练') } }
	},
	glossary: [
		{ id: 'skip-gram', term: L('skip-gram', '跳字模型（skip-gram）'), def: L('word2vec’s training task: from one word (the center word), predict each word near it (its context words).', 'word2vec 的训练任务：由一个词（中心词）预测它附近的每个词（上下文词）。'), related: ['word-window'] },
		{ id: 'word-window', term: L('window, context words', '窗口、上下文词'), def: L('The words up to w places to the left and right of a center word, in the same sentence; each forms one (center, context) pair with it.', '中心词左右各 w 个位置以内、同一句子里的词；每个都和中心词组成一个（中心词，上下文词）对。'), related: ['skip-gram'] },
		{ id: 'one-hot', term: L('one-hot vector', '独热向量（one-hot）'), def: L('A list of 0s with a single 1; the place of the 1 says which item it names.', '一串 0 里只有一个 1；1 所在的位置表示它指的是哪一项。') },
		{ id: 'dot-product', term: L('dot product', '点积'), def: L('Multiply two vectors number by number and add the products. It is large when the two point the same way.', '把两个向量逐个数相乘，再把乘积加起来。两个向量方向越一致，点积越大。') },
		{ id: 'cosine-similarity', term: L('cosine similarity', '余弦相似度'), def: L('The dot product of two vectors divided by both their lengths: 1 for the same direction, 0 for unrelated, −1 for opposite.', '两个向量的点积除以它们的长度：方向相同为 1，无关为 0，相反为 −1。'), related: ['dot-product'] },
		{ id: 'pca', term: L('PCA (principal component analysis)', '主成分分析（PCA）'), def: L('Finds the directions in which a cloud of points spreads the most; the top two give the flat picture that keeps the most of the spread.', '找出一团点分布最开的方向；取前两个方向，就得到保留分布最多的平面图。') }
	],
	catalog: {
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		params: L('2 × V × 5 (V ≤ 40 words: at most 400 numbers)', '2 × V × 5（V ≤ 40 个词：最多 400 个数）'),
		size: L('about 2 KB of preset text; no weights shipped', '约 2 KB 预设文本；不附带权重')
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'The preset mini-corpora: short sentences chosen from the validation file by fixed rules (pipeline/word2vec/presets.py), and one Chinese translation of them',
			attribution: "Ronen Eldan and Yuanzhi Li, 'TinyStories: How Small Can Language Models Be and Still Speak Coherent English?' (2023), arXiv:2305.07759"
		},
		{
			name: 'wevi: word embedding visual inspector',
			url: 'https://github.com/ronxin/wevi',
			license: 'MIT',
			usedFor: 'The idea of training skip-gram live on a tiny corpus with both weight matrices, the softmax and a PCA map in view; reimplemented, no code copied',
			attribution: 'Xin Rong (wevi; and “word2vec Parameter Learning Explained”, arXiv:1411.2738)'
		},
		{
			name: 'word2vec (skip-gram)',
			url: 'https://arxiv.org/abs/1301.3781',
			license: 'Paper (ideas only; no text, figures or code used)',
			usedFor: 'The skip-gram model this page trains (with a full softmax instead of the paper’s faster approximations)',
			attribution: 'Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space” (2013)'
		}
	]
};

export default manifest;
