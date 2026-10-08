// tinyclip: the package manifest (design §4.9, §7). A tiny CLIP on synthetic shapes and captions:
//   fwd    the exported model (pipeline/tinyclip/export.py) on the learner's drawing and captions: both towers, the
//          score of the drawing against each caption, the softmax over the captions
//   synth  the training pairs, made by the same TS code that made the training set (12 here; train.py read 24,000);
//          phones use the same frames from tapes/synth (it has no Python reference: TS renders the drawings)
//   nce    the batch of 6 pairs (batch.ts: the learner's pair and 5 fixed ones through the exported model): the 6×6
//          scores, the row and column softmax, the InfoNCE loss and its gradient (kit autograd; reference.py twin)
//   init   the same graph with the starting weights of the training run (static/models/tinyclip/init), and nce0: the
//          same batch through them, i.e. the batch before training
// Measured numbers (test accuracies, the order ties) are in facts.json (pipeline/tinyclip/stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { PRESETS } from './scenes.ts';
import { rawOf } from './adapters/shapes-captions/adapter.ts';
import { batchOf } from './batch.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tinyclip',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tinyclip', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'shapes-captions', start: rawOf(PRESETS[0]) },
	output: { view: 'tinyclip/match', layer: 'probs', props: { mode: 'captions' } },
	sources: {
		fwd: { kind: 'trace' },
		synth: { kind: 'process', process: 'synth', run: 'main', precomputed: 'tapes/synth' },
		nce: { kind: 'process', process: 'nce', run: 'main', input: (rt) => batchOf(rt).batch },
		init: { kind: 'trace', assets: 'models/tinyclip/init' },
		nce0: { kind: 'process', process: 'nce', run: 'main', input: (rt) => batchOf(rt, 'init').batch }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			I('Say how a drawing and a caption each become 32 numbers of length 1, and how one score compares them.', '说出一幅画和一句描述如何各自变成长度为 1 的 32 个数，以及一个分数如何比较它们。'),
			I('Read a batch’s table of scores: which cells training makes large, and what the row and column softmax say.', '读懂一批图文对的分数表：训练让哪些格子变大，行和列的 softmax 各说明什么。'),
			I('Say what the InfoNCE gradient does to the true pairs and to the wrong ones.', '说出 InfoNCE 的梯度对正确的对和错误的对分别做了什么。'),
			I('Use captions as the classes of a classifier, and explain why word order is lost.', '把描述当作分类器的类别来用，并解释为什么词序丢失了。')
		],
		prequestion: {
			id: 'tinyclip-pre',
			question: I('Nobody told TinyCLIP “this drawing is a circle”. How can it still pick the caption “a circle” for a circle?', '没有人告诉过 TinyCLIP “这幅画是圆”。它怎么还能为一个圆选出“a circle”这句描述？'),
			options: [I('It has a hidden list of classes', '它有一份隐藏的类别表'), I('It learned to give matching drawings and captions similar numbers', '它学会了让相配的画和描述得到相近的数'), I('It guesses', '它是猜的')],
			answer: 1,
			explain: I('It was trained only on pairs: each drawing’s numbers were pulled toward its own caption’s numbers and away from the others. Any caption made of known words can then be compared with any drawing.', '它只用图文对训练：每幅画的数被拉向它自己那句描述的数，并远离其他描述的数。之后任何由认识的词组成的描述，都能和任何一幅画比较。')
		},
		tryThis: [
			{ id: 'stack', title: I('Circle above square', '圆在正方形上面'), setup: { input: rawOf(PRESETS[5]) }, predict: I('Swap the two shape words of the caption. Which score changes more: that one, or swapping “above” for “below”?', '把描述里的两个形状词对调。哪个分数变化更大：这个，还是把“above”换成“below”？'), look: I('Compare the three scores in the last step.', '比较最后一步里的三个分数。'), goto: { step: 'order' } }
		],
		misconception: { claim: I('TinyCLIP understands the sentence.', 'TinyCLIP 理解这句话。'), truth: I('Its text tower averages word rows, so it knows which words occur but not in which order.', '它的文本塔把词行求平均，所以它知道出现了哪些词，却不知道它们的顺序。'), step: 'order' },
		sections: { intros: { training: I('Trained in PyTorch with the InfoNCE loss: 60 epochs over the 20,000 training pairs in batches of 256, with AdamW. The page runs the exported weights; the numbers below were measured on the 2,000 test pairs.', '用 PyTorch 和 InfoNCE 损失训练：在 20,000 对训练图文对上训练 60 轮，每批 256 对，用 AdamW 优化。页面运行的是导出的权重；下面的数是在 2,000 对测试图文对上测出来的。') } },
		lessonNote: I('Ten steps: what it learned from, the two towers, the score of one pair, a batch’s table of scores, the loss and its gradient, your own captions, and where word order gets lost.', '十步：它学习用的材料、两座塔、一对的分数、一批的分数表、损失和它的梯度、你自己写的描述，以及词序在哪里丢失。')
	},
	glossary: [
		{ id: 'contrastive-learning', term: I('contrastive learning', '对比学习'), def: I('Learning by comparison: pull matching pairs together and push mismatched pairs apart, without class labels.', '通过比较来学习：把相配的对拉近，把不相配的对推远，不需要类别标签。'), related: ['self-supervised'] },
		{ id: 'cosine-similarity', term: I('cosine similarity', '余弦相似度'), def: I('Two vectors’ products added up, divided by both lengths: 1 when they point the same way, 0 at right angles, −1 when opposite.', '两个向量逐项相乘再相加，除以两者的长度：方向相同时为 1，垂直时为 0，相反时为 −1。'), related: ['embedding'] },
		{ id: 'infonce', term: I('InfoNCE loss', 'InfoNCE 损失'), def: I('CLIP’s loss: in each row and each column of a batch’s scores, the cross-entropy of the true pair after a softmax, averaged.', 'CLIP 的损失：在一批分数的每一行和每一列上做 softmax，取正确那一对的交叉熵，再求平均。'), related: ['cross-entropy', 'softmax'] },
		{ id: 'zero-shot', term: I('zero-shot classification', '零样本分类'), def: I('Classifying with classes the model was never trained on as classes, here by writing each class as a caption.', '用模型从没作为类别训练过的类别来分类；这里的做法是把每个类别写成一句描述。'), related: ['classifier'] },
		{ id: 'bag-of-words', term: I('bag of words', '词袋'), def: I('Text kept as the words it contains and how often, without their order.', '只保留文本里有哪些词、各出现几次，而不保留它们的顺序。'), related: ['token'] }
	],
	credits: [{ name: 'CLIP: Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)', url: 'https://arxiv.org/abs/2103.00020', license: 'Paper (arXiv); ideas only (no code, weights or data used)', usedFor: 'the two towers, the batch score table, the symmetric InfoNCE loss and the learned temperature, reimplemented at toy size', attribution: 'Alec Radford, Jong Wook Kim, Chris Hallacy and others (OpenAI)' }],
	catalog: { params: I('15,905', '15,905'), size: I('62 KB of weights', '权重 62 KB') }
};

export default manifest;
