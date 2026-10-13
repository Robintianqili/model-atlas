// contrastive-learning: the manifest of a CONCEPT package (/train/contrastive-learning/; the page around the lesson is
// concept.ts). The learner builds a batch of drawing-caption pairs (adapters/batch) and sets the temperature τ. Sources:
//   clip      the shipped TinyCLIP of /m/tinyclip/ (static/models/tinyclip, read only): only its model is used (inputs
//             ['none']: no trace of its own); batch.ts runs every pair through it
//   ck<u>     the same graph with the weights of a re-run of TinyCLIP's training after u updates
//             (static/models/contrastive-learning/ck/<u>/, pipeline/contrastive-learning/train.py + export.py)
//   nce       the batch's cosines, scores cos / τ, row and column softmax, symmetric InfoNCE loss and its gradient
//             (process 'nce'; Python twin reference.py)
//   progress  the batch through every checkpoint: the table of cosines from the start of training to its end
//   simclr    two distorted views of every drawing through TinyCLIP's image tower, and the loss over the 2N views
// Measured numbers of the re-run (test loss and accuracy at the first and last checkpoint) are in facts.json.
import type { ExplainerManifest, SourceDef } from '../../atlas/types/index.ts';
import { START } from './adapters/batch/adapter.ts';
import { CKPTS, batchIn, ckSource, progressIn, simclrIn } from './batch.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };
const I = (en: string, zh: string) => ({ en, zh });

const checkpoints: Record<string, SourceDef> = Object.fromEntries(CKPTS.map((u) => [ckSource(u), { kind: 'trace', assets: `models/contrastive-learning/ck/${u}`, inputs: ['none'] }]));

const manifest: ConceptManifest = {
	slug: 'contrastive-learning',
	type: 'concept',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/contrastive-learning', auto: ['facts.json'] },
	input: { adapter: 'batch', start: START },
	sources: {
		clip: { kind: 'trace', assets: 'models/tinyclip', inputs: ['none'] },
		...checkpoints,
		nce: { kind: 'process', process: 'nce', run: 'main', input: (rt) => batchIn(rt) },
		// the models travel by reference (run: 'main'), so these two have no Python reference; their badge says "Live"
		progress: { kind: 'process', process: 'progress', run: 'main', input: progressIn, reference: false },
		simclr: { kind: 'process', process: 'simclr', run: 'main', input: simclrIn, reference: false }
	},
	page: {
		goals: [
			I('Read a batch’s table of cosines: say which cells are the true pairs, and what training does to them and to all the others.', '读懂一批图文对的余弦表：说出哪些格子是正确的对，训练对它们和其余格子分别做了什么。'),
			I('Work out one row of the InfoNCE loss from its cosines and the temperature τ, and say what a very large and a very small τ do to it.', '根据一行的余弦和温度 τ 算出这一行的 InfoNCE 损失，并说出 τ 很大和很小时损失会怎样。'),
			I('Say how SimCLR makes true pairs without any captions, and why its distortions decide what the encoder learns to ignore.', '说出 SimCLR 如何在没有任何描述的情况下造出正确的对，以及为什么它用的变形决定了编码器学会忽略什么。'),
			I('Name two honest limits of the loss: false negatives inside a batch, and a guessing loss that grows with the batch size.', '说出这个损失的两个真实局限：同一批里的伪负样本，以及随批大小增长的瞎猜损失。')
		],
		prequestion: {
			id: 'contrastive-learning-pre',
			question: I('A model gets thousands of drawings, each with the caption someone wrote for it, and no class labels at all. What can it learn from that alone?', '一个模型拿到成千上万幅画，每幅都配着别人为它写的描述，却完全没有类别标签。只凭这些，它能学到什么？'),
			options: [I('Nothing useful: it needs labels such as “circle”', '学不到有用的东西：它需要“圆”这样的标签'), I('To give each drawing numbers close to its own caption’s, and far from the other captions’', '让每幅画的数靠近它自己那句描述的数，远离其他描述的数'), I('To repeat the captions word for word', '一字不差地复述描述')],
			answer: 1,
			explain: I('Which caption belongs to which drawing is a signal in itself. Contrastive learning pulls every matching pair together and pushes the other pairs of the batch apart; no class list is needed.', '“哪句描述属于哪幅画”本身就是信号。对比学习把每个相配的对拉近，把同一批里其他的对推远，不需要任何类别表。')
		},
		quiz: [
			{
				q: I('A batch has 8 pairs and the model cannot tell them apart at all: every score in a row is the same. What is its InfoNCE loss?', '一批有 8 对，模型完全分不清它们：一行里的每个分数都一样。它的 InfoNCE 损失是多少？'),
				options: [I('log 8, about 2.08', 'log 8，约 2.08'), I('0', '0'), I('8', '8')],
				answer: 0,
				explain: I('Equal scores give every caption the share 1/8, and the loss is −log(1/8) = log 8 ≈ 2.08 nats: the loss of a blind guess among 8.', '分数相同时每句描述的份额都是 1/8，损失是 −log(1/8) = log 8 ≈ 2.08 奈特：在 8 个里瞎猜的损失。')
			},
			{
				q: I('In SimCLR, which other item is the true partner of a distorted view?', '在 SimCLR 里，一个变形视图的正确搭档是哪一个？'),
				options: [I('The other distorted view of the same image', '同一张图的另一个变形视图'), I('The image’s caption', '这张图的描述'), I('Whichever image in the batch is most similar', '这一批里最相似的那张图')],
				answer: 0,
				explain: I('SimCLR uses no captions: two random distortions of one image are the positive pair, and every other view in the batch is a negative.', 'SimCLR 不用描述：同一张图的两个随机变形是正样本对，这一批里其他所有视图都是负样本。')
			}
		],
		misconception: {
			claim: I('Contrastive learning needs labels that say what each image shows.', '对比学习需要标出每张图是什么的标签。'),
			truth: I('It only needs to know which items belong together: a drawing and its own caption, or two distorted views of one drawing. Everything else in the batch serves as a negative.', '它只需要知道哪些东西属于一起：一幅画和它自己的描述，或者同一幅画的两个变形视图。同一批里的其他东西都当作负样本。'),
			step: 'pairs'
		},
		lessonNote: I('Eleven steps: a batch’s table of cosines, one row’s softmax and loss, the temperature, the gradient, a training run that turns the table into a diagonal, SimCLR’s two views, and the false negatives the loss cannot see.', '十一步：一批的余弦表、一行的 softmax 和损失、温度、梯度、把表变成对角线的一次训练、SimCLR 的两个视图，以及损失看不见的伪负样本。')
	},
	glossary: [
		{ id: 'positive-pair', term: I('positive and negative pairs', '正样本对与负样本对'), def: I('In contrastive learning, a positive pair belongs together (a drawing and its own caption, or two views of one image); every other pairing in the batch is a negative.', '在对比学习中，正样本对是本该在一起的一对（一幅画和它自己的描述，或同一张图的两个视图）；同一批里其他所有搭配都是负样本。'), related: ['contrastive-learning', 'infonce'] },
		{ id: 'hard-negative', term: I('hard negative', '难负样本'), def: I('A negative that the model scores almost as high as the true pair: the one the loss pushes down hardest.', '模型给出的分数几乎和正确的对一样高的负样本：损失把它往下推得最狠。'), related: ['positive-pair', 'infonce'] }
	],
	credits: [
		{ name: 'CLIP: Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)', url: 'https://arxiv.org/abs/2103.00020', license: 'Paper (arXiv); ideas only (no code, weights or data used)', usedFor: 'the batch table of image-caption scores, the symmetric InfoNCE loss and the learned temperature, at toy size (through TinyCLIP)', attribution: 'Alec Radford, Jong Wook Kim, Chris Hallacy and others (OpenAI)' },
		{ name: 'SimCLR: Chen et al., A Simple Framework for Contrastive Learning of Visual Representations (2020)', url: 'https://arxiv.org/abs/2002.05709', license: 'Paper (arXiv); ideas only (no code, weights or data used)', usedFor: 'two randomly distorted views of each image as the positive pair, and the loss over the 2N views (NT-Xent)', attribution: 'Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton (Google Research)' },
		{ name: 'CPC: van den Oord et al., Representation Learning with Contrastive Predictive Coding (2018)', url: 'https://arxiv.org/abs/1807.03748', license: 'Paper (arXiv); ideas only (no code, weights or data used)', usedFor: 'the InfoNCE loss: a softmax over one positive and the negatives, and its bound of log N', attribution: 'Aaron van den Oord, Yazhe Li, Oriol Vinyals (DeepMind)' }
	],
	catalog: {
		interaction: I('Add your own drawings and captions to a batch, see the N×N similarity matrix from the live TinyCLIP, drag the temperature and watch the InfoNCE loss; a training slider morphs the matrix from noise to a clean diagonal, and SimCLR’s two distorted views of each drawing make a second matrix.', '把你的画和描述加入一个批次，查看实时运行的 TinyCLIP 给出的 N×N 相似度矩阵，拖动温度看 InfoNCE 损失变化；训练进度滑块让矩阵从噪声变成清晰的对角线；每幅画的两个 SimCLR 变形视图构成第二个矩阵。'),
		engine: I('TinyCLIP live; checkpoints of a training re-run precomputed', 'TinyCLIP 实时运行；一次重新训练的检查点为预计算'),
		mode: 'live',
		links: ['tinyclip', 'clip']
	} as ExplainerManifest['catalog']
};

export default manifest;
