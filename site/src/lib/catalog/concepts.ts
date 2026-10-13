/**
 * Training-concepts track: every row of brief §6, grouped by section.
 * Live in v0: tokenization, next-token-prediction, transfer-learning (hand-built routes). Live since K2: loss-gradient-descent
 * (a concept package, explainers/loss-gradient-descent/, rendered by the /train/[slug] route). Concept packages that went
 * live later (batch 6 onward) carry status 'live' and a `lesson`; every other row is planned.
 * `links` are catalog model slugs (validated by scripts/check-catalog.ts). The brief's own link
 * names are mapped through LINK_ALIASES when they differ (tiny-cnn -> tinycnn, sd -> stable-diffusion, …).
 */
import type { CardData, ConceptEntry, ConceptSection, I18n, SectionSlug } from './types.ts';

const f = (en: string, zh: string): I18n => ({ en, zh });

export const SECTIONS: ConceptSection[] = [
	{
		slug: 'foundations',
		n: '6.1',
		name: f('Foundations of training', '训练基础'),
		blurb: f('Loss, gradients, optimizers and the tricks that make training work.', '损失、梯度、优化器，以及让训练奏效的各种技巧。')
	},
	{
		slug: 'pretraining',
		n: '6.2',
		name: f('Pre-training objectives and data', '预训练目标与数据'),
		blurb: f('What a model is asked to predict before it learns any particular task.', '在学习具体任务之前，模型被要求预测什么。')
	},
	{
		slug: 'finetuning',
		n: '6.3',
		name: f('Transfer learning and fine-tuning', '迁移学习与微调'),
		blurb: f('Reusing what a model already knows for a new task.', '把模型已经学到的东西用到新任务上。')
	},
	{
		slug: 'peft',
		n: '6.4',
		name: f('Parameter-efficient tuning', '参数高效微调'),
		blurb: f('Changing a few parameters instead of all of them.', '只改少量参数，而不是全部。')
	},
	{
		slug: 'alignment',
		n: '6.5',
		name: f('Alignment', '对齐'),
		blurb: f('Steering a model toward the answers people prefer.', '引导模型给出人们更想要的回答。')
	},
	{
		slug: 'compression',
		n: '6.6',
		name: f('Compression and efficiency', '压缩与效率'),
		blurb: f('Smaller numbers, fewer weights, smaller students.', '更少的位数、更少的权重、更小的学生模型。')
	},
	{
		slug: 'interpretability',
		n: '6.7',
		name: f('Interpretability and explainability', '可解释性'),
		blurb: f('Tools for asking why a model gave its answer.', '追问模型为什么给出这个答案的工具。')
	},
	{
		slug: 'evaluation',
		n: '6.8',
		name: f('Evaluation', '评估'),
		blurb: f('Measuring a model honestly.', '诚实地衡量一个模型。')
	},
	{
		slug: 'inference',
		n: '6.9',
		name: f('Inference and serving', '推理与部署'),
		blurb: f('Running a trained model fast and cheaply.', '让训练好的模型跑得又快又省。')
	},
	{
		slug: 'robustness',
		n: '6.10',
		name: f('Robustness, privacy and scale', '鲁棒性、隐私与规模'),
		blurb: f('Attacks, private training and training across many machines.', '攻击、隐私保护训练，以及多机训练。')
	}
];

export const SECTION_BY_SLUG: Record<SectionSlug, ConceptSection> = Object.fromEntries(
	SECTIONS.map((s) => [s.slug, s])
) as Record<SectionSlug, ConceptSection>;

/** Brief link names -> catalog model slugs. */
export const LINK_ALIASES: Record<string, string> = {
	'tiny-cnn': 'tinycnn',
	'tiny-gpt': 'tinygpt',
	'tiny-vit': 'tinyvit',
	'tiny-bert': 'tinybert',
	'tiny-clip': 'tinyclip',
	'tiny-mamba': 'tinymamba',
	'tiny-seq2seq': 'tinyseq2seq',
	whisper: 'whisper-tiny',
	sd: 'stable-diffusion',
	'flow-2d / sd': 'stable-diffusion'
};

export const CONCEPTS: ConceptEntry[] = [
	// ------------------------------------------------------------ 6.1 Foundations
	{
		slug: 'loss-gradient-descent',
		section: 'foundations',
		title: f('Loss and gradient descent', '损失与梯度下降'),
		status: 'live',
		v1: true,
		interaction: f(
			'"Beat the error score": turn weight and bias dials by hand, then press the gradient-descent step button and watch it do better.',
			'“打败误差分数”：手动转动权重和偏置旋钮，再按下梯度下降按钮，看它做得更好。'
		),
		engine: 'Single neuron in JS',
		mode: 'live',
		links: ['mlp-playground', 'tinycnn'],
		lesson: {
			steps: 9,
			summary: f(
				'Score a line through eight points, beat the score by hand, then watch gradient descent turn the dials with the real numbers, and break it with a step that is too big.',
				'给穿过八个点的直线打分，亲手打败这个分数，再看梯度下降用真实的数值转动旋钮，最后用过大的步长把它弄坏。'
			)
		}
	},
	{
		slug: 'backprop',
		section: 'foundations',
		title: f('Backpropagation and autograd', '反向传播与自动求导'),
		status: 'live',
		v1: true,
		interaction: f(
			'Edit any input value and the whole graph recomputes. Step the backward pass one chain-rule edge at a time, with gradients shown as "?" until they are reached, and check them numerically.',
			'修改任意输入值，整张计算图重新计算。一次走一条链式法则的边来执行反向传播，未算到的梯度显示为“?”，并可用数值梯度检验。'
		),
		engine: 'Own autograd core (checked against PyTorch)',
		mode: 'live',
		links: ['mlp-playground', 'tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Run a tiny computation graph forward, walk its backward pass one chain-rule edge at a time (a node used twice adds both shares), check it numerically, then watch the same rule fill in the nine gradients of a 2-2-1 network and fade through a flat tanh.',
				'把一张很小的计算图正向算一遍，再沿链式法则一条边一条边地走反向传播（被用两次的节点要把两份加起来），用数值方法检验，然后看同样的规则怎样算出 2-2-1 网络的九个梯度，又怎样在平坦的 tanh 处衰减。'
			)
		}
	},
	{
		slug: 'optimizers',
		section: 'foundations',
		title: f('Optimizers, learning-rate schedules and warm-up', '优化器、学习率调度与预热'),
		status: 'live',
		v1: true,
		interaction: f(
			'SGD, momentum, Adam and AdamW race on a long valley, a saddle and a bumpy bowl, with the optimizer state drawn as a momentum arrow and squared-gradient squares, and one update each of SGD, momentum and Adam worked out with the real numbers. Run a learning-rate range test on a small live network, pick a schedule with or without warm-up and watch it train, and toggle gradient clipping next to a cliff.',
			'SGD、动量法、Adam 和 AdamW 在狭长山谷、鞍点和坑洼的碗上赛跑，优化器状态画成动量箭头和梯度平方方块，并用真实的数值算出 SGD、动量法和 Adam 各一次更新。在一个实时训练的小网络上做学习率区间测试，选择带或不带预热的调度并观察训练，在悬崖边切换梯度裁剪。'
		),
		engine: 'Kit optimizers on analytic surfaces + live 2-8-8-2 MLP',
		mode: 'live',
		links: ['mlp-playground', 'char-rnn'],
		lesson: {
			steps: 11,
			summary: f(
				'Four optimizers race down a long valley; work out one update of SGD, momentum and Adam with the real numbers, race on your own landscape, see that no optimizer wins everywhere, then run a learning-rate range test, rescue a too-large rate with warm-up and stop an exploding gradient with clipping.',
				'四个优化器沿狭长山谷赛跑；用真实的数值算出 SGD、动量法和 Adam 的一次更新，在你自己的地形上比赛，看到没有哪个优化器处处都赢；然后做学习率区间测试，用预热救下过大的学习率，再用梯度裁剪拦住梯度爆炸。'
			)
		}
	},
	{
		slug: 'regularization',
		section: 'foundations',
		title: f('Overfitting and regularization', '过拟合与正则化'),
		status: 'live',
		v1: true,
		interaction: f(
			'Watch the training and test curves split apart as a big network overfits a few noisy points of two moons, predict whether a smaller network does better, then fight overfitting with early stopping, an L2 penalty (its real term in the loss), dropout (units blinking per update) and nudged training points. Turn five knobs until the test loss drops, and see how TinyCNN was trained (augmented drawings, weight decay, the kept epoch).',
			'看一个大网络在两个月牙的少量带噪声的点上过拟合，训练曲线和测试曲线逐渐分开；预测更小的网络会不会更好，再用早停、L2 惩罚（损失里真实的那一项）、dropout（单元每次更新随机熄灭）和挪动训练点来对付过拟合。调节五个旋钮让测试损失降下来，并看看 TinyCNN 是怎么训练的（增强过的手绘图、权重衰减、保留的那一轮）。'
		),
		engine: '2D MLP + TinyCNN',
		mode: 'live',
		links: ['mlp-playground', 'classic-ml', 'tinycnn'],
		lesson: {
			steps: 10,
			summary: f(
				'Watch a big network learn few noisy points by heart while its test loss climbs, then fight it with live runs: a smaller network, early stopping, an L2 penalty, dropout and augmented data; close the gap yourself and overdo it.',
				'看一个大网络把少量有噪声的点背下来、测试损失却不断上升，再用实时训练来对付它：更小的网络、早停、L2 惩罚、dropout 和数据增强；自己动手缩小差距，最后看看用力过猛会怎样。'
			)
		}
	},
	{
		slug: 'init-normalization',
		section: 'foundations',
		title: f('Initialization and normalization', '初始化与归一化'),
		status: 'live',
		v1: false,
		interaction: f(
			'Activation and gradient histograms per layer under zero, too-small, Xavier, He and too-large initialization; BatchNorm vs LayerNorm on the same tensor.',
			'在零初始化、过小、Xavier、He、过大初始化下逐层观察激活与梯度直方图；在同一张量上对比 BatchNorm 与 LayerNorm。'
		),
		engine: 'Tiny MLP / TinyGPT tensors',
		mode: 'live',
		links: ['tinygpt', 'resnet-18'],
		lesson: {
			steps: 10,
			summary: f(
				'Start one deep network five ways and watch each layer\'s numbers and gradients fade, hold or blow up; then normalize the same tensor with BatchNorm and LayerNorm, and see where BatchNorm breaks.',
				'用五种方式初始化同一个深层网络，看每一层的数值和梯度怎样衰减、保持或爆炸；再用 BatchNorm 和 LayerNorm 归一化同一个张量，看看 BatchNorm 在哪里失效。'
			)
		}
	},
	{
		slug: 'residual-connections',
		section: 'foundations',
		title: f('Residual connections and vanishing gradients', '残差连接与梯度消失'),
		status: 'live',
		v1: false,
		interaction: f(
			'Toggle skip connections in a 20-layer MLP and watch the gradient-norm bars per layer.',
			'在 20 层 MLP 中开关跳跃连接，观察每层的梯度范数条形图。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['resnet-18', 'tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Watch a 20-layer network fail to learn, find where its gradient vanishes, follow the gradient through one skip connection with the real numbers, then train both networks and change the depth.',
				'看一个 20 层的网络学不动，找到它的梯度在哪里消失；用真实的数值跟着梯度穿过一个跳跃连接；然后训练两个网络，再改变深度。'
			)
		}
	},
	{
		slug: 'loss-choice',
		section: 'foundations',
		title: f('Choosing a loss', '选择损失函数'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Cross-entropy vs mean squared error on the same classifier; label smoothing; focal loss on an imbalanced 2D set.',
			'同一个分类器上对比交叉熵与均方误差；标签平滑；在不平衡二维数据上的 focal loss。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['mlp-playground', 'tinycnn']
	},
	{
		slug: 'hyperparameter-search',
		section: 'foundations',
		title: f('Hyperparameter search and grokking', '超参数搜索与顿悟（grokking）'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Grid vs random search on a 2D objective, and a replay of a model that suddenly "groks" modular arithmetic long after it memorized the training set.',
			'在二维目标上对比网格搜索与随机搜索，并回放一个在记住训练集很久之后才突然“顿悟”模运算的模型。'
		),
		engine: 'JS + precomputed',
		mode: 'both',
		links: ['tinygpt']
	},

	// ------------------------------------------------------------ 6.2 Pre-training
	{
		slug: 'tokenization',
		section: 'pretraining',
		title: f('Tokenization', '分词'),
		status: 'live',
		v1: true,
		tier: 'P0',
		interaction: f(
			'Train a byte-pair encoder on your own text with a merge-by-merge stepper, then compare real tokenizers on the same text with a byte view (for example, how 我爱自然语言处理 is split).',
			'用逐步合并的步进器在你自己的文本上训练 BPE，再在同一段文本上对比真实分词器并查看字节（例如“我爱自然语言处理”如何被切分）。'
		),
		engine: 'Own BPE in JS + tokenizers',
		mode: 'live',
		links: ['tinygpt', 'tokenizers', 'tinybert', 'bert-base', 'whisper-tiny', 'clip'],
		lesson: {
			steps: 7,
			summary: f('Byte-pair encoding on your own text: bytes, pre-tokens, pair counts, merges, then real tokenizers.', '在你自己的文本上走一遍字节对编码：字节、预分词、相邻对计数、合并，再到真实分词器。')
		}
	},
	{
		slug: 'embeddings',
		section: 'pretraining',
		title: f('Embeddings', '嵌入'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Train word2vec on your own context/target pairs and watch the full matrices; build custom axes; see BERT give "bank" two different vectors in two sentences.',
			'在你自己的上下文/目标词对上训练 word2vec，观察完整矩阵；自定义坐标轴；看 BERT 如何在两个句子里给“bank”不同的向量。'
		),
		engine: 'Own JS word2vec + precomputed BERT',
		mode: 'both',
		links: ['word2vec', 'bert-base', 'tinyclip']
	},
	{
		slug: 'next-token-prediction',
		section: 'pretraining',
		title: f('Next-token prediction', '下一个词元预测'),
		status: 'live',
		v1: true,
		tier: 'P0',
		interaction: f(
			'One sentence becomes many (context → next) training examples; drag p(correct token) to watch −log p change; color your own sentence by surprise and read off the loss and perplexity; replay training from gibberish to text.',
			'一句话变成许多（上下文 → 下一个）训练样本；拖动正确词元的概率，看 −log p 如何变化；按“意外程度”给你的句子上色，读出损失与困惑度；回放从乱码到成句的训练过程。'
		),
		engine: 'TinyGPT live + precomputed checkpoints',
		mode: 'both',
		links: ['tinygpt', 'gpt-2', 'tinymamba'],
		lesson: {
			steps: 7,
			summary: f(
				'Your sentence from tokens to loss: windows, 1,024 probabilities, −log p, perplexity, training checkpoints, sampling.',
				'你的句子从词元到损失：窗口、1,024 个概率、−log p、困惑度、训练检查点、采样。'
			)
		}
	},
	{
		slug: 'decoding',
		section: 'pretraining',
		title: f('Decoding and sampling', '解码与采样'),
		status: 'live',
		v1: false,
		interaction: f(
			'Temperature, top-k and top-p applied to the same cached TinyGPT logits for your prompt, with the cutoffs drawn and the chances renormalized; greedy against sampling with a fixed seed, token by token; why greedy loops.',
			'在 TinyGPT 对你的提示算出的同一组缓存 logits 上调节温度、top-k、top-p，画出截断线并重新归一化概率；逐个词元比较贪心和固定种子的采样；看贪心为什么会循环。'
		),
		engine: 'TinyGPT live',
		mode: 'live',
		links: ['tinygpt', 'gpt-2', 'whisper-tiny', 'tinyseq2seq'],
		lesson: {
			steps: 10,
			summary: f(
				'One row of TinyGPT’s scores for your prompt becomes chances; temperature, top-k and top-p reshape them with the cutoffs drawn; then TinyGPT writes token by token, greedy against sampling with a fixed seed, and you see why greedy loops.',
				'TinyGPT 对你的提示算出的一行分数变成概率；温度、top-k、top-p 重新调整它们，并画出截断线；然后 TinyGPT 逐个词元写作，比较贪心和固定种子的采样，并看清贪心为什么会循环。'
			)
		}
	},
	{
		slug: 'self-supervised-objectives',
		section: 'pretraining',
		title: f('Self-supervised objectives: an overview', '自监督目标总览'),
		status: 'live',
		v1: true,
		interaction: f(
			'One sentence and one drawing, many objectives: TinyGPT’s next-token probabilities and TinyBERT’s guesses for a token you click, live; the 80/10/10 rule on your text; next-sentence prediction, span corruption, text infilling, fill in the middle, masked autoencoding, JEPA, contrastive views and added noise drawn as labelled mask patterns.',
			'一句话、一幅简笔画，多种目标：实时看 TinyGPT 预测下一个词元的概率，以及 TinyBERT 对你点中的词元的猜测；在你的文字上演示 80/10/10 规则；下一句预测、片段破坏、文本填充、中间填充、掩码自编码、JEPA、对比视图和加噪声都画成标明“示意”的遮罩图案。'
		),
		engine: 'TinyGPT + TinyBERT live + illustrative mask diagrams (labeled)',
		mode: 'both',
		links: ['tinygpt', 'tinybert', 'tinyvit', 'tinyclip'],
		lesson: {
			steps: 11,
			summary: f(
				'One sentence and one drawing, many objectives: what each hides and what it predicts. TinyGPT and TinyBERT run live on your text (next token, a token you click, the 80/10/10 rule); next-sentence prediction, spans, masked patches and contrastive views are labelled drawings.',
				'一句话、一幅简笔画，多种目标：每种目标藏起什么、预测什么。TinyGPT 和 TinyBERT 在你的文字上实时运行（下一个词元、你点中的词元、80/10/10 规则）；下一句预测、片段、被遮的图块和对比视图是标明“示意”的图。'
			)
		}
	},
	{
		slug: 'masked-lm',
		section: 'pretraining',
		title: f('Masked language modeling', '掩码语言模型'),
		status: 'live',
		v1: false,
		interaction: f(
			'Click any token of your sentence to hide it and read TinyBERT’s guesses live; compute the loss on the hidden positions only; watch the 80/10/10 rule draw a batch with real counts and send your row to TinyBERT; set TinyBERT beside TinyGPT on the same sentence to see which context each reads.',
			'点你句子里的任意词元把它藏起来，实时读出 TinyBERT 的猜测；只在被藏的位置上算损失；看 80/10/10 规则在一个批次上抽签（真实计数），并把你那一行交给 TinyBERT；在同一句话上把 TinyBERT 和 TinyGPT 放在一起，看各自读到哪些上下文。'
		),
		engine: 'TinyBERT + TinyGPT live',
		mode: 'live',
		links: ['tinybert', 'tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Hide tokens of your own sentence and read TinyBERT’s live guesses, compute the loss on the hidden positions only, watch the 80/10/10 rule draw a batch and send your row to TinyBERT, and set it beside TinyGPT on the same sentence.',
				'在你自己的句子里藏起词元，读 TinyBERT 的实时猜测；只在被藏的位置上算损失；看 80/10/10 规则在一个批次上抽签，并把你那一行交给 TinyBERT；再在同一句话上把它和 TinyGPT 放在一起比较。'
			)
		}
	},
	{
		slug: 'next-sentence-prediction',
		section: 'pretraining',
		title: f('Next-sentence prediction', '下一句预测'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Build sentence pairs (IsNext / NotNext) and see the [CLS] head output, and why later models dropped this objective.',
			'构造句子对（IsNext / NotNext），查看 [CLS] 头的输出，以及后来的模型为何去掉了这个目标。'
		),
		engine: 'BERT-base precomputed',
		mode: 'precomputed',
		links: ['bert-base']
	},
	{
		slug: 'contrastive-learning',
		section: 'pretraining',
		title: f('Contrastive learning (CLIP, InfoNCE, SimCLR)', '对比学习（CLIP、InfoNCE、SimCLR）'),
		status: 'live',
		v1: true,
		interaction: f(
			'Add your own drawings and captions to a batch, see the N×N similarity matrix from the live TinyCLIP, drag the temperature and watch the InfoNCE loss; a training slider morphs the matrix from noise to a clean diagonal, and SimCLR’s two distorted views of each drawing make a second matrix.',
			'把你的画和描述加入一个批次，查看实时运行的 TinyCLIP 给出的 N×N 相似度矩阵，拖动温度看 InfoNCE 损失变化；训练进度滑块让矩阵从噪声变成清晰的对角线；每幅画的两个 SimCLR 变形视图构成第二个矩阵。'
		),
		engine: 'TinyCLIP live; checkpoints of a training re-run precomputed',
		mode: 'live',
		links: ['tinyclip', 'clip'],
		lesson: {
			steps: 11,
			summary: f(
				'Build a batch of drawings and captions, read TinyCLIP’s table of cosines, work out the InfoNCE loss and drag its temperature, watch training turn the table into a diagonal, then make SimCLR’s pairs from two distorted views and meet the false negatives the loss cannot see.',
				'搭一批画和描述，读懂 TinyCLIP 的余弦表，算出 InfoNCE 损失并拖动它的温度，看训练把这张表变成一条对角线，再用两个变形视图造出 SimCLR 的正样本对，并认识损失看不见的伪负样本。'
			)
		}
	},
	{
		slug: 'masked-autoencoding',
		section: 'pretraining',
		title: f('Masked autoencoding', '掩码自编码'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Click patches to mask them in the TinyViT grid and move a masking-ratio slider to see reconstructions.',
			'在 TinyViT 网格中点击图块将其遮住，拖动遮蔽比例滑块查看重建结果。'
		),
		engine: 'Own tiny MAE on TinyViT patches',
		mode: 'both',
		links: ['tinyvit', 'dinov2']
	},
	{
		slug: 'self-distillation',
		section: 'pretraining',
		title: f('Self-distillation (DINO)', '自蒸馏（DINO）'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Student and teacher views of the same image; attention and patch-PCA maps emerging over training.',
			'同一张图的学生视角与教师视角；注意力图与图块 PCA 图在训练中逐渐出现。'
		),
		engine: 'DINOv2-small live attention + precomputed',
		mode: 'both',
		links: ['dinov2']
	},
	{
		slug: 'denoising-objectives',
		section: 'pretraining',
		title: f('Denoising objectives', '去噪目标'),
		status: 'planned',
		v1: false,
		interaction: f(
			'A noise-schedule slider; DDPM vs flow-matching targets on the same 2D data, shown as two equivalent views.',
			'噪声调度滑块；在同一份二维数据上对比 DDPM 与流匹配的训练目标，展示两者的等价视角。'
		),
		engine: '2D playground',
		mode: 'live',
		links: ['flow-2d', 'ddpm-unet']
	},
	{
		slug: 'scaling-laws',
		section: 'pretraining',
		title: f('Scaling laws', '缩放定律'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Drag a compute budget on a Chinchilla-style loss formula L(N, D), and compare with our own sweep of TinyGPT sizes.',
			'在 Chinchilla 式损失公式 L(N, D) 上拖动计算预算，并与我们自己的 TinyGPT 规模扫描对比。'
		),
		engine: 'Analytic + precomputed sweep',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'data-mixtures',
		section: 'pretraining',
		title: f('Data mixtures, curricula and curation', '数据配比、课程与清洗'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Choose a Chinese/English mix and see per-domain loss curves from precomputed TinyGPT runs; a curation toy with near-duplicate detection and a quality threshold.',
			'选择中英文配比，查看预计算 TinyGPT 训练中各领域的损失曲线；一个包含近似去重与质量阈值的数据清洗小玩具。'
		),
		engine: 'Precomputed runs + JS',
		mode: 'both',
		links: ['tinygpt']
	},

	// ------------------------------------------------------------ 6.3 Fine-tuning
	{
		slug: 'transfer-learning',
		section: 'finetuning',
		title: f('Transfer learning and fine-tuning', '迁移学习与微调'),
		status: 'live',
		v1: true,
		tier: 'P0',
		interaction: f(
			'Teach TinyCNN two new drawing classes of your own: keep the trained layers frozen (gray, locked) and train only a new head, live, then compare head-only, full fine-tuning and training from scratch (precomputed). Real-scale evidence from a BERT reproduction (frozen features vs full fine-tuning, the learning-rate search, small-data instability, and MNLI first, then RTE: +14.3 points).',
			'教 TinyCNN 认识你自己画的两个新类别：冻结已训练的层（灰色、上锁），实时只训练新的输出头，再对比只训练输出头、全量微调与从零训练（预计算）。附 BERT 复现中的真实规模证据（冻结特征 vs 全量微调、学习率搜索、小数据下的不稳定，以及先 MNLI 再 RTE：提高 14.3 个百分点）。'
		),
		engine: 'TinyCNN live head training + precomputed BERT results',
		mode: 'both',
		links: ['tinycnn', 'bert-base', 'resnet-18', 'kws'],
		lesson: {
			steps: 8,
			summary: f(
				'From a forced guess to a new head: frozen features, one training step by hand, few-example curves, BERT at real scale.',
				'从被迫的猜测到新的输出头：冻结特征、亲手走一步训练、少样本曲线、真实规模的 BERT。'
			)
		}
	},
	{
		slug: 'probe-vs-finetune',
		section: 'finetuning',
		title: f('Linear probe vs partial vs full fine-tuning', '线性探针、部分微调与全量微调'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Click blocks of a 12-block GPT to freeze or unfreeze them and see precomputed accuracy, trainable-parameter count and training time for that choice.',
			'点击 12 层 GPT 的各个块来冻结或解冻，查看该配置下的预计算准确率、可训练参数量与训练时间。'
		),
		engine: 'Precomputed sweeps + live head-only training',
		mode: 'both',
		links: ['gpt-2', 'bert-base', 'kws']
	},
	{
		slug: 'vision-finetuning',
		section: 'finetuning',
		title: f('Fine-tuning vision models', '视觉模型微调'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Fine-tune TinyCNN on two new classes with and without augmentation; ResNet-18 head-only vs full fine-tuning (precomputed).',
			'在两个新类别上微调 TinyCNN，对比有无数据增强；ResNet-18 只训练输出头与全量微调的对比（预计算）。'
		),
		engine: 'Own core live + precomputed',
		mode: 'both',
		links: ['tinycnn', 'resnet-18']
	},
	{
		slug: 'lp-ft',
		section: 'finetuning',
		title: f('LP-FT and feature distortion', 'LP-FT 与特征扭曲'),
		status: 'planned',
		v1: false,
		interaction: f(
			'An embedding scatter of in-distribution and out-of-distribution points drifting under full fine-tuning vs linear-probe-then-fine-tune.',
			'嵌入散点图：分布内与分布外的点在全量微调与“先线性探针再微调”下如何漂移。'
		),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['resnet-18', 'bert-base']
	},
	{
		slug: 'layerwise-lr',
		section: 'finetuning',
		title: f('Layer-wise learning rates', '分层学习率'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Gradient-norm and weight-change bars per layer, with a learning-rate slider and a decay factor.',
			'逐层的梯度范数与权重变化条形图，带学习率滑块和衰减系数。'
		),
		engine: 'Live small model + precomputed',
		mode: 'both',
		links: ['kws', 'bert-base']
	},
	{
		slug: 'catastrophic-forgetting',
		section: 'finetuning',
		title: f('Catastrophic forgetting', '灾难性遗忘'),
		status: 'live',
		v1: true,
		interaction: f(
			"Train a 2D MLP on task A, then task B, and watch A's decision boundary erode live. Try replay, a pull toward the old weights, freezing and a lower learning rate.",
			'先在任务 A 上训练二维 MLP，再训练任务 B，实时观察 A 的决策边界被侵蚀。尝试经验回放、向旧权重的拉力、冻结和更低的学习率。'
		),
		engine: 'Own autograd core in a worker',
		mode: 'live',
		links: ['mlp-playground'],
		lesson: {
			steps: 9,
			summary: f(
				'Train a small network on task A, then on task B, and watch A disappear; see why the two tasks fight over the same weights, then try replay, a pull toward the old weights, freezing and a smaller learning rate, and what each one costs.',
				'先在任务 A 上训练一个小网络，再训练任务 B，看着 A 被忘掉；弄清两个任务为什么在争同一组权重，再试试回放、拉向旧权重、冻结和更小的学习率，以及每种办法的代价。'
			)
		}
	},
	{
		slug: 'multitask-intermediate',
		section: 'finetuning',
		title: f('Multi-task and intermediate-task training', '多任务与中间任务训练'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Drag task blocks on a timeline (RTE alone, MNLI then RTE, a multi-task mix) and see the matching precomputed curves; a gradient-conflict toy with two loss bowls.',
			'在时间线上拖动任务块（只训 RTE、先 MNLI 再 RTE、多任务混合），查看对应的预计算曲线；以及一个两个损失碗相互冲突的梯度小玩具。'
		),
		engine: 'Precomputed BERT runs + JS toy',
		mode: 'both',
		links: ['bert-base']
	},
	{
		slug: 'continued-pretraining',
		section: 'finetuning',
		title: f('Continued pre-training and long-context extension', '继续预训练与长上下文扩展'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Continue TinyGPT on a new domain and watch in-domain vs general loss; extend the context with RoPE scaling vs naive extrapolation.',
			'在新领域上继续训练 TinyGPT，观察领域内与通用损失；用 RoPE 缩放与朴素外推对比上下文扩展。'
		),
		engine: 'Precomputed TinyGPT runs',
		mode: 'precomputed',
		links: ['tinygpt']
	},
	{
		slug: 'instruction-tuning',
		section: 'finetuning',
		title: f('Instruction tuning (SFT)', '指令微调（SFT）'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Write a conversation and see it rendered through a real chat template, tokenized, with the loss mask: assistant tokens trained, everything else ignored.',
			'写一段对话，看它经过真实的对话模板渲染与分词，并显示损失掩码：只训练助手的词元，其他都被忽略。'
		),
		engine: 'Chat template + tokenizer in JS',
		mode: 'both',
		links: ['tinygpt', 'gpt-2']
	},
	{
		slug: 'icl-vs-finetune-vs-rag',
		section: 'finetuning',
		title: f('In-context learning vs fine-tuning vs RAG', '上下文学习、微调与 RAG 对比'),
		status: 'planned',
		v1: false,
		interaction: f(
			'The same few examples given in the prompt, used for a few fine-tuning steps, or retrieved from a store; compare outputs, cost and freshness.',
			'同样几个示例：放进提示词、用于几步微调、或从库中检索；对比输出、成本与时效性。'
		),
		engine: 'Precomputed + retrieval page',
		mode: 'precomputed',
		links: ['tinygpt', 'gpt-2', 'sentence-embeddings']
	},
	{
		slug: 'few-shot-semi-supervised',
		section: 'finetuning',
		title: f('Few-shot, semi-supervised and domain adaptation', '少样本、半监督与领域自适应'),
		status: 'planned',
		v1: false,
		interaction: f(
			'2D toys: prototypes from 5 examples per class, pseudo-labels spreading past a confidence threshold, and a source-to-target domain shift with an alignment slider.',
			'二维小玩具：每类 5 个样本的原型、超过置信阈值后扩散的伪标签，以及带对齐滑块的源域到目标域迁移。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['mlp-playground', 'sentence-embeddings']
	},

	// ------------------------------------------------------------ 6.4 PEFT
	{
		slug: 'lora',
		section: 'peft',
		title: f('LoRA and QLoRA', 'LoRA 与 QLoRA'),
		status: 'live',
		v1: true,
		interaction: f(
			'A live parameter counter, r·(d_in + d_out) against d_in·d_out, from a 16 × 16 toy layer up to GPT-2 small; train W₀ + (α/r)·BA live at your rank with B starting at zero, so step 0 equals the base model; ranks 1 to 8 against full fine-tuning; and the training memory of full fine-tuning, LoRA and QLoRA.',
			'实时参数计数器：r·(d_in + d_out) 对比 d_in·d_out，从 16 × 16 的玩具层一直到 GPT-2 small；在你选的秩上实时训练 W₀ + (α/r)·BA，B 从零开始，所以第 0 步就是基础模型；秩 1 到 8 与全量微调对比；以及全量微调、LoRA 与 QLoRA 的训练内存。'
		),
		engine: 'Own autograd core in a worker',
		mode: 'live',
		links: ['gpt-2', 'tinygpt', 'flow-2d', 'stable-diffusion'],
		lesson: {
			steps: 10,
			summary: f(
				'Count what LoRA trains, watch B grow from zero on a real layer until it learns the change a new task needs, find the smallest rank that does the job, and see where LoRA saves less than it seems.',
				'数一数 LoRA 训练多少个数，看 B 在一个真实的层上从零长起来，直到学会新任务需要的改变，找出够用的最小秩，再看 LoRA 在哪里省得没有看上去那么多。'
			)
		}
	},
	{
		slug: 'adapters-prefix',
		section: 'peft',
		title: f('Adapters, BitFit, (IA)³ and prefix tuning', 'Adapter、BitFit、(IA)³ 与前缀微调'),
		status: 'planned',
		v1: false,
		interaction: f(
			'The same block diagram with each method’s trainable parameters lit up, with parameter counts and precomputed accuracy side by side.',
			'同一张结构图上点亮每种方法的可训练参数，并排显示参数量与预计算准确率。'
		),
		engine: 'Precomputed on BERT / TinyGPT',
		mode: 'precomputed',
		links: ['bert-base', 'tinygpt']
	},
	{
		slug: 'memory-budget',
		section: 'peft',
		title: f('Memory: what fits on your GPU', '显存：你的 GPU 装得下什么'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Stacked memory bars (weights, gradients, optimizer state, activations) for full fine-tuning vs LoRA vs QLoRA, with your GPU memory as an input.',
			'堆叠的显存条（权重、梯度、优化器状态、激活值），对比全量微调、LoRA 与 QLoRA，并以你的显存大小作为输入。'
		),
		engine: 'Analytic',
		mode: 'live',
		links: ['gpt-2', 'bert-base']
	},

	// ------------------------------------------------------------ 6.5 Alignment
	{
		slug: 'reward-modeling',
		section: 'alignment',
		title: f('Reward modeling', '奖励模型'),
		status: 'planned',
		v1: false,
		interaction: f(
			'You are the labeler: choose between response pairs while a Bradley–Terry reward model fits your choices.',
			'由你来标注：在成对回答中做选择，Bradley–Terry 奖励模型实时拟合你的偏好。'
		),
		engine: 'JS toy',
		mode: 'live',
		links: ['tinygpt']
	},
	{
		slug: 'best-of-n',
		section: 'alignment',
		title: f('Best-of-n and rejection sampling', 'Best-of-n 与拒绝采样'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Sample n completions, score them with the reward model, keep the best; reward vs n and KL vs n curves.',
			'采样 n 个续写，用奖励模型打分并保留最好的；绘制奖励随 n、KL 随 n 的变化曲线。'
		),
		engine: 'Precomputed TinyGPT samples + live scoring',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'rlhf',
		section: 'alignment',
		title: f('RLHF (PPO with a KL penalty)', 'RLHF（带 KL 惩罚的 PPO）'),
		status: 'planned',
		v1: false,
		interaction: f(
			'A KL-leash slider: the proxy reward keeps rising while true quality peaks and falls (Goodhart). A 1-D policy moving on one landscape under REINFORCE, PPO and GRPO.',
			'KL“牵绳”滑块：代理奖励持续上升，而真实质量先升后降（古德哈特定律）。一维策略在同一地形上分别按 REINFORCE、PPO、GRPO 移动。'
		),
		engine: 'JS toy',
		mode: 'live',
		links: ['cart-pole', 'tinygpt']
	},
	{
		slug: 'dpo',
		section: 'alignment',
		title: f('DPO and its variants', 'DPO 及其变体'),
		status: 'planned',
		v1: false,
		interaction: f(
			'The loss −log σ(β·Δ) with a β slider; it starts exactly at ln 2 ≈ 0.693 when the margin is zero. Switch between variants and see which tokens became more or less likely.',
			'带 β 滑块的损失 −log σ(β·Δ)；当间隔为零时它恰好从 ln 2 ≈ 0.693 开始。切换各种变体，查看哪些词元变得更可能或更不可能。'
		),
		engine: 'TinyGPT preference-tuned on our own pairs',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'grpo',
		section: 'alignment',
		title: f('GRPO and verifiable rewards', 'GRPO 与可验证奖励'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Group sampling, relative advantages and a verifiable reward on a toy arithmetic task, plus a real-scale replay.',
			'在玩具算术任务上演示组采样、相对优势与可验证奖励，并附真实规模的回放。'
		),
		engine: 'Toy + precomputed',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'constitutional-ai',
		section: 'alignment',
		title: f('Constitutional AI, RLAIF and reasoning distillation', '宪法式 AI、RLAIF 与推理蒸馏'),
		status: 'planned',
		v1: false,
		interaction: f(
			'A critique-and-revise loop on fixed prompts; a teacher’s reasoning traces distilled into TinyGPT, before and after.',
			'在固定提示上演示“批评—修改”循环；把教师模型的推理过程蒸馏进 TinyGPT，对比前后效果。'
		),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},
	{
		slug: 'knowledge-editing',
		section: 'alignment',
		title: f('Knowledge editing and machine unlearning', '知识编辑与机器遗忘'),
		status: 'planned',
		v1: false,
		tier: 'P2',
		interaction: f(
			'Edit one fact in TinyGPT with a rank-one update and watch related facts change; unlearn a subset and test for leakage.',
			'用秩一更新修改 TinyGPT 中的一条事实，观察相关事实如何变化；遗忘一部分数据并检测泄露。'
		),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},

	// ------------------------------------------------------------ 6.6 Compression
	{
		slug: 'number-formats',
		section: 'compression',
		title: f('Number formats and mixed precision', '数值格式与混合精度'),
		status: 'live',
		v1: true,
		interaction: f(
			'Click the bits of one number in FP32, FP16, BF16, FP8 (e4m3, e5m2), FP4, INT8, INT4 and NF4 and watch the stored value, the rounding error and the gap to the next number change; cast TinyGPT’s real weights to each format and measure its perplexity; see why mixed precision keeps fp32 master weights.',
			'点击一个数在 FP32、FP16、BF16、FP8（e4m3、e5m2）、FP4、INT8、INT4 和 NF4 中的各个比特位，看存下的值、舍入误差和到下一个数的间隔怎样变化；把 TinyGPT 的真实权重转换成每种格式，测出它的困惑度；再看混合精度为什么要保留 fp32 主权重。'
		),
		engine: 'JS',
		mode: 'live',
		links: ['tinygpt'],
		lesson: {
			steps: 11,
			summary: f(
				'Click the bits of one number in fp32, fp16, bf16, FP8, FP4, int8, int4 and NF4, see where each format runs out of range or precision, cast a real TinyGPT to each, and see why mixed precision keeps fp32 master weights.',
				'点击一个数在 fp32、fp16、bf16、FP8、FP4、int8、int4 和 NF4 中的各个比特位，看每种格式在哪里耗尽范围或精度，把真实的 TinyGPT 换成每种格式，再看混合精度为什么要保留 fp32 主权重。'
			)
		}
	},
	{
		slug: 'quantization',
		section: 'compression',
		title: f('Quantization', '量化'),
		status: 'live',
		v1: true,
		interaction: f(
			'A rounding ledger for 8 values (absmax vs affine, clipping, block size); one outlier wrecking the scale; quantize a TinyGPT layer and watch logits, top-1 and perplexity change.',
			'8 个数值的舍入账本（absmax 与仿射、截断、分块大小）；一个离群值毁掉缩放比例；量化 TinyGPT 的一层，观察 logits、top-1 与困惑度的变化。'
		),
		engine: 'JS + real models',
		mode: 'live',
		links: ['tinygpt', 'kokoro', 'slimsam', 'dinov2'],
		lesson: {
			steps: 11,
			summary: f(
				'Look at one number bit by bit, round eight numbers to int8 and int4 by hand, watch one outlier wreck the scale, then quantize a real TinyGPT in your browser and measure what it costs.',
				'逐位查看一个数，亲手把八个数舍入成 int8 和 int4，看一个离群值怎样毁掉缩放比例，再在浏览器里量化一个真实的 TinyGPT，测量它的代价。'
			)
		}
	},
	{
		slug: 'ptq-qat',
		section: 'compression',
		title: f('Post-training quantization vs quantization-aware training', '训练后量化与量化感知训练'),
		status: 'planned',
		v1: false,
		interaction: f(
			'The same TinyGPT layer under round-to-nearest, error-compensating (GPTQ-style) and activation-aware (AWQ-style) quantization, then QAT recovering accuracy.',
			'同一层 TinyGPT 分别用就近舍入、误差补偿（GPTQ 式）与激活感知（AWQ 式）量化，再看量化感知训练如何恢复精度。'
		),
		engine: 'JS + precomputed',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'pruning',
		section: 'compression',
		title: f('Pruning', '剪枝'),
		status: 'planned',
		v1: false,
		interaction: f(
			'A sparsity slider on TinyCNN (magnitude vs structured) with the mask drawn; prune, watch accuracy drop, then fine-tune to recover.',
			'在 TinyCNN 上拖动稀疏度滑块（按幅值与结构化），画出掩码；剪枝后准确率下降，再微调恢复。'
		),
		engine: 'TinyCNN live + precomputed',
		mode: 'both',
		links: ['tinycnn']
	},
	{
		slug: 'distillation',
		section: 'compression',
		title: f('Knowledge distillation', '知识蒸馏'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Teacher vs student soft labels with a temperature slider; a race between a student trained on hard labels and one trained on soft labels.',
			'带温度滑块的教师与学生软标签；用硬标签训练的学生与用软标签训练的学生赛跑。'
		),
		engine: 'JS toy + real models',
		mode: 'both',
		links: ['gpt-2', 'slimsam', 'tinycnn']
	},
	{
		slug: 'model-merging',
		section: 'compression',
		title: f('Model merging and weight averaging', '模型合并与权重平均'),
		status: 'planned',
		v1: false,
		tier: 'P2',
		interaction: f(
			'Interpolate between two TinyGPT fine-tunes and see the loss along the path.',
			'在两个 TinyGPT 微调版本之间插值，查看路径上的损失。'
		),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},

	// ------------------------------------------------------------ 6.7 Interpretability
	{
		slug: 'saliency',
		section: 'interpretability',
		title: f('Saliency and integrated gradients', '显著性图与积分梯度'),
		status: 'planned',
		v1: false,
		interaction: f(
			'On your TinyCNN drawing: plain gradients vs integrated gradients vs SmoothGrad, plus a sanity check that randomizes weights layer by layer.',
			'在你的 TinyCNN 画作上：对比普通梯度、积分梯度与 SmoothGrad，并做逐层随机化权重的合理性检验。'
		),
		engine: 'TinyCNN live',
		mode: 'live',
		links: ['tinycnn', 'resnet-18']
	},
	{
		slug: 'cam',
		section: 'interpretability',
		title: f('Class activation maps (CAM family)', '类激活图（CAM 家族）'),
		status: 'planned',
		v1: false,
		interaction: f(
			'A formula card CAM = ReLU(Σ w_k A_k) with live numbers, and a switcher over precomputed Grad-CAM-style maps.',
			'带实时数字的公式卡 CAM = ReLU(Σ w_k A_k)，以及在预计算的 Grad-CAM 类方法之间切换。'
		),
		engine: 'Live CAM + precomputed maps',
		mode: 'both',
		links: ['resnet-18']
	},
	{
		slug: 'shap-lime',
		section: 'interpretability',
		title: f('SHAP and LIME for tabular models', '表格模型的 SHAP 与 LIME'),
		status: 'planned',
		v1: false,
		interaction: f(
			"A SHAP waterfall for one row of the tree model; LIME's local linear fit drawn on the 2D canvas.",
			'树模型中一行数据的 SHAP 瀑布图；在二维画布上画出 LIME 的局部线性拟合。'
		),
		engine: 'JS',
		mode: 'live',
		links: ['gbt', 'classic-ml']
	},
	{
		slug: 'probing',
		section: 'interpretability',
		title: f('Probing classifiers', '探针分类器'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Train a linear probe on each TinyGPT or BERT layer for one property and read the accuracy per layer.',
			'在 TinyGPT 或 BERT 的每一层上为某个性质训练线性探针，读出每层的准确率。'
		),
		engine: 'Precomputed activations + live probe',
		mode: 'both',
		links: ['tinygpt', 'bert-base']
	},
	{
		slug: 'logit-lens',
		section: 'interpretability',
		title: f('Logit lens', 'Logit lens（逐层解码）'),
		status: 'planned',
		v1: false,
		interaction: f('Decode every layer of TinyGPT for your own prompt and see when the answer appears.', '对你自己的提示词逐层解码 TinyGPT，看答案在哪一层出现。'),
		engine: 'TinyGPT live; larger model precomputed',
		mode: 'both',
		links: ['tinygpt', 'logit-lens']
	},
	{
		slug: 'activation-patching',
		section: 'interpretability',
		title: f('Activation patching', '激活修补'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Swap one activation from a clean prompt into a corrupted one and watch the logit difference; a heatmap over layer × position.',
			'把干净提示中的一个激活值换进被破坏的提示中，观察 logit 差异；绘制层×位置的热力图。'
		),
		engine: 'TinyGPT live',
		mode: 'live',
		links: ['tinygpt']
	},
	{
		slug: 'sparse-autoencoders',
		section: 'interpretability',
		title: f('Sparse autoencoders and steering', '稀疏自编码器与激活引导'),
		status: 'planned',
		v1: false,
		interaction: f(
			"An SAE on TinyGPT's residual stream: top-activating tokens per feature, and a steering slider that adds a feature direction live.",
			'在 TinyGPT 残差流上训练的稀疏自编码器：每个特征的最高激活词元，以及实时添加特征方向的引导滑块。'
		),
		engine: 'Own SAE (precomputed) + live steering',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'attention-not-explanation',
		section: 'interpretability',
		title: f('Attention is not explanation', '注意力不等于解释'),
		status: 'planned',
		v1: false,
		interaction: f(
			'The same prediction with very different attention maps; attention vs gradient attribution side by side.',
			'同一个预测，注意力图却截然不同；注意力与梯度归因并排对比。'
		),
		engine: 'TinyGPT live',
		mode: 'live',
		links: ['tinygpt', 'bert-base']
	},

	// ------------------------------------------------------------ 6.8 Evaluation
	{
		slug: 'splits-leakage',
		section: 'evaluation',
		title: f('Data splits and leakage', '数据划分与泄露'),
		status: 'planned',
		v1: false,
		interaction: f('Drag points between train, validation and test; leak a test point and watch the score inflate.', '在训练、验证、测试集之间拖动数据点；泄露一个测试点，看分数虚高。'),
		engine: '2D MLP',
		mode: 'live',
		links: ['mlp-playground']
	},
	{
		slug: 'metrics-calibration',
		section: 'evaluation',
		title: f('Metrics and calibration', '指标与校准'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Confusion matrix, precision/recall and ROC; a reliability diagram for TinyCNN with a temperature-scaling slider.',
			'混淆矩阵、精确率/召回率与 ROC；TinyCNN 的可靠性图，以及温度缩放滑块。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['tinycnn', 'classic-ml']
	},
	{
		slug: 'perplexity-vs-accuracy',
		section: 'evaluation',
		title: f('Perplexity vs downstream accuracy', '困惑度与下游准确率'),
		status: 'planned',
		v1: false,
		interaction: f('TinyGPT checkpoints: the loss falls smoothly while a task metric jumps.', 'TinyGPT 检查点：损失平滑下降，而任务指标突然跃升。'),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},
	{
		slug: 'contamination',
		section: 'evaluation',
		title: f('Benchmark contamination', '评测集污染'),
		status: 'planned',
		v1: false,
		interaction: f('Fine-tune TinyGPT with test items leaked into training and compare clean vs contaminated scores.', '把测试题泄露进 TinyGPT 的训练数据，对比干净与被污染的分数。'),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},
	{
		slug: 'pass-at-k',
		section: 'evaluation',
		title: f('pass@k', 'pass@k'),
		status: 'planned',
		v1: false,
		interaction: f('Sample k answers from a toy arithmetic model and compute the unbiased pass@k estimate with live numbers.', '从玩具算术模型中采样 k 个答案，用实时数字计算无偏 pass@k 估计。'),
		engine: 'Analytic + precomputed samples',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'llm-as-judge',
		section: 'evaluation',
		title: f('LLM as a judge', '大模型当裁判'),
		status: 'planned',
		v1: false,
		tier: 'P2',
		interaction: f('Position-bias and verbosity-bias toggles on precomputed judgments.', '在预计算的评判结果上切换位置偏差与冗长偏差。'),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: []
	},

	// ------------------------------------------------------------ 6.9 Inference
	{
		slug: 'kv-cache',
		section: 'inference',
		title: f('KV cache and prompt caching', 'KV 缓存与提示缓存'),
		status: 'live',
		v1: false,
		interaction: f(
			'TinyGPT continues your prompt twice, without and with a KV cache: an attention grid shows the work done again in orange and the work the cache skips in gray, with live counts of multiply-adds and cache memory (GPT-2 small and Llama 2 7B by the same formula) and a check that both runs pick the same tokens. A second prompt reuses the first one’s cached prefix up to the first different token.',
			'TinyGPT 把你的提示词续写两遍，一遍不用 KV 缓存，一遍用：注意力网格里橙色是重复计算的部分，灰色是缓存省下的计算；实时统计乘加次数和缓存内存（用同一公式算出 GPT-2 small 和 Llama 2 7B 的），并核对两遍选出的词元相同。第二个提示词复用第一个提示词缓存的前缀，直到第一个不同的词元为止。'
		),
		engine: 'TinyGPT live',
		mode: 'live',
		links: ['tinygpt', 'gpt-2'],
		lesson: {
			steps: 10,
			summary: f(
				'Watch TinyGPT redo all its work for every new token, keep the keys and values in a cache instead, count the memory and the multiply-adds, check that the answers stay the same, then share a cached prefix between two prompts and see where sharing stops.',
				'看 TinyGPT 每生成一个新词元就把所有计算重做一遍，改用缓存保存键和值，算出内存和乘加次数，验证答案不变，再让两个提示词共享缓存的前缀，看共享在哪里停止。'
			)
		}
	},
	{
		slug: 'attention-variants',
		section: 'inference',
		title: f('Attention variants (MHA, MQA, GQA, MLA)', '注意力变体（MHA、MQA、GQA、MLA）'),
		status: 'live',
		v1: false,
		interaction: f(
			'Pick a published model or set your own layers, heads, KV heads, head width, tokens and bytes, and see the KV cache of MHA, GQA, MQA and MLA next to a head-sharing diagram. Follow one shared-key attention computation on TinyGPT, convert its trained heads to 2 and 1 KV heads in the browser and measure the damage, then see how much uptraining repairs (measured in PyTorch).',
			'选一个已发表的模型，或自己设定层数、头数、KV 头数、头宽、词元数和字节数，对照头共享示意图看 MHA、GQA、MQA 和 MLA 的 KV 缓存。在 TinyGPT 上跟着做一次共用键的注意力计算，在浏览器里把它训练好的头转换成 2 个和 1 个 KV 头并测出损失，再看继续训练能修复多少（用 PyTorch 测得）。'
		),
		engine: 'Analytic + TinyGPT live + precomputed',
		mode: 'both',
		links: ['tinygpt'],
		lesson: {
			steps: 11,
			summary: f(
				'Count what the KV cache stores, see which heads share keys and values in MHA, GQA, MQA and MLA, follow one shared-key attention computation on TinyGPT, size the caches of published models, and measure what sharing costs a trained model.',
				'数一数 KV 缓存存了什么，看清 MHA、GQA、MQA 和 MLA 中哪些头共用键和值，在 TinyGPT 上跟着做一次共用键的注意力计算，算出已发表模型的缓存大小，再测一测共享会让训练好的模型付出什么代价。'
			)
		}
	},
	{
		slug: 'positional-encodings',
		section: 'inference',
		title: f('Positional encodings and long context', '位置编码与长上下文'),
		status: 'live',
		v1: false,
		interaction: f(
			'Sinusoidal vs learned vs RoPE (rotate the 2D pair) vs ALiBi (a bias ramp), and extrapolation beyond the training context.',
			'正弦、可学习、RoPE（旋转二维对）与 ALiBi（偏置斜坡）的对比，以及超出训练长度的外推。'
		),
		engine: 'Live + precomputed',
		mode: 'both',
		links: ['tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Shuffle your sentence and watch attention not notice, read TinyGPT’s learned position table, compute a sinusoidal encoding, turn a query with RoPE, ramp scores with ALiBi, then see which encodings survive text four times longer than training.',
				'打乱你的句子，看注意力察觉不到；读 TinyGPT 学到的位置表，算一个正弦位置编码，用 RoPE 旋转查询，用 ALiBi 给分数加斜坡，再看哪些编码在比训练长四倍的文本上还能用。'
			)
		}
	},
	{
		slug: 'speculative-decoding',
		section: 'inference',
		title: f('Speculative decoding', '投机解码'),
		status: 'planned',
		v1: false,
		interaction: f('A tiny draft model proposes k tokens and TinyGPT verifies them; accept/reject animation and a speed-up counter.', '小型草稿模型提出 k 个词元，由 TinyGPT 验证；接受/拒绝动画与加速计数器。'),
		engine: 'Live',
		mode: 'live',
		links: ['tinygpt']
	},
	{
		slug: 'flash-attention',
		section: 'inference',
		title: f('Batching and FlashAttention tiling', '批处理与 FlashAttention 分块'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Tile-by-tile softmax with a running max and sum, a memory-read counter vs naive attention, and a static vs continuous batching timeline.',
			'带滑动最大值与累加和的分块 softmax、与朴素注意力对比的内存读取计数器，以及静态与连续批处理的时间线。'
		),
		engine: 'JS',
		mode: 'live',
		links: ['tinygpt']
	},
	{
		slug: 'test-time-compute',
		section: 'inference',
		title: f('Test-time compute and reasoning', '测试时计算与推理'),
		status: 'planned',
		v1: false,
		interaction: f('Majority vote and best-of-n on toy arithmetic; accuracy vs number of samples; chain-of-thought length vs accuracy.', '在玩具算术上做多数投票与 best-of-n；准确率随样本数变化；思维链长度与准确率。'),
		engine: 'Precomputed',
		mode: 'precomputed',
		links: ['tinygpt']
	},

	// ------------------------------------------------------------ 6.10 Robustness
	{
		slug: 'adversarial-robustness',
		section: 'robustness',
		title: f('Adversarial examples and robustness', '对抗样本与鲁棒性'),
		status: 'planned',
		v1: false,
		interaction: f(
			'FGSM and PGD on your drawing; an adversarial-training toggle on the 2D MLP that makes the boundary margin grow.',
			'在你的画上做 FGSM 与 PGD；在二维 MLP 上开启对抗训练，看决策边界的间隔变大。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['tinycnn', 'adversarial-tinycnn', 'mlp-playground']
	},
	{
		slug: 'federated-learning',
		section: 'robustness',
		title: f('Federated learning', '联邦学习'),
		status: 'planned',
		v1: false,
		interaction: f('4 simulated clients with different 2D data; FedAvg rounds and client drift.', '4 个持有不同二维数据的模拟客户端；FedAvg 轮次与客户端漂移。'),
		engine: 'Own core',
		mode: 'live',
		links: ['mlp-playground']
	},
	{
		slug: 'differential-privacy',
		section: 'robustness',
		title: f('Differential privacy (DP-SGD)', '差分隐私（DP-SGD）'),
		status: 'planned',
		v1: false,
		interaction: f('Per-example clipping plus Gaussian noise on the 2D MLP; accuracy vs ε; one "canary" point\'s influence.', '在二维 MLP 上做逐样本裁剪加高斯噪声；准确率随 ε 变化；一个“金丝雀”点的影响。'),
		engine: 'Own core',
		mode: 'live',
		links: ['mlp-playground']
	},
	{
		slug: 'training-at-scale',
		section: 'robustness',
		title: f('Training at scale', '大规模训练'),
		status: 'planned',
		v1: false,
		interaction: f('A memory breakdown, data/tensor/pipeline parallelism diagrams and an all-reduce animation.', '显存拆解、数据/张量/流水线并行示意图，以及 all-reduce 动画。'),
		engine: 'Analytic',
		mode: 'live',
		links: ['gpt-2']
	}
];

export const CONCEPT_BY_SLUG: Record<string, ConceptEntry> = Object.fromEntries(CONCEPTS.map((c) => [c.slug, c]));

/** Concepts grouped by section, in section order. */
export function conceptsBySection(): { section: ConceptSection; items: ConceptEntry[] }[] {
	return SECTIONS.map((section) => ({ section, items: CONCEPTS.filter((c) => c.section === section.slug) }));
}

/** Concepts that link to a model slug (for "Concepts that use this model"). */
export function conceptsForModel(slug: string): ConceptEntry[] {
	return CONCEPTS.filter((c) => c.links.includes(slug));
}

/** A concept's card data (CardData in types.ts); `withSection` adds the section's name (kit/ConceptCard.svelte showSection). */
export function conceptCard(c: ConceptEntry, withSection = false): CardData {
	const card: CardData = { kind: 'concept', slug: c.slug, title: c.title, status: c.status, hook: c.interaction };
	if (withSection) card.area = SECTION_BY_SLUG[c.section].name;
	if (c.v1) card.v1 = true;
	return card;
}
