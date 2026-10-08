/**
 * Training-concepts track: every row of brief §6, grouped by section.
 * Live in v0: tokenization, next-token-prediction, transfer-learning (hand-built routes). Live since K2: loss-gradient-descent
 * (a concept package, explainers/loss-gradient-descent/, rendered by the /train/[slug] route). Everything else is planned.
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
		status: 'planned',
		v1: true,
		interaction: f(
			'Edit any input value and the whole graph recomputes. Step the backward pass one chain-rule edge at a time, with gradients shown as "?" until they are reached, and check them numerically.',
			'修改任意输入值，整张计算图重新计算。一次走一条链式法则的边来执行反向传播，未算到的梯度显示为“?”，并可用数值梯度检验。'
		),
		engine: 'In-house Value engine (micrograd-style); the same core powers live training',
		mode: 'live',
		links: ['mlp-playground', 'tinygpt']
	},
	{
		slug: 'optimizers',
		section: 'foundations',
		title: f('Optimizers, learning-rate schedules and warm-up', '优化器、学习率调度与预热'),
		status: 'planned',
		v1: true,
		interaction: f(
			'SGD, momentum, Adam and AdamW race on one loss surface, with the optimizer state drawn as arrows and boxes. Pick a schedule, run an LR range test on TinyGPT and scrub through a grid of precomputed runs.',
			'SGD、动量、Adam、AdamW 在同一个损失曲面上赛跑，优化器状态画成箭头与方框。选择学习率调度，在 TinyGPT 上做学习率区间测试，并浏览预计算的训练网格。'
		),
		engine: 'JS on analytic surfaces + live 2D MLP + precomputed TinyGPT sweep',
		mode: 'both',
		links: ['tinygpt', 'mlp-playground']
	},
	{
		slug: 'regularization',
		section: 'foundations',
		title: f('Batch size, regularization and overfitting', '批大小、正则化与过拟合'),
		status: 'planned',
		v1: true,
		interaction: f(
			'Watch train and test curves split apart; add L1/L2, weight decay, dropout (units blinking per step) and data augmentation on TinyCNN drawings; mark early stopping.',
			'看训练曲线和测试曲线如何分开；加入 L1/L2、权重衰减、dropout（神经元逐步闪烁）以及 TinyCNN 手绘数据增强；标出早停点。'
		),
		engine: '2D MLP + TinyCNN',
		mode: 'live',
		links: ['mlp-playground', 'classic-ml', 'tinycnn']
	},
	{
		slug: 'init-normalization',
		section: 'foundations',
		title: f('Initialization and normalization', '初始化与归一化'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Activation and gradient histograms per layer under zero, too-small, Xavier, He and too-large initialization; BatchNorm vs LayerNorm on the same tensor.',
			'在零初始化、过小、Xavier、He、过大初始化下逐层观察激活与梯度直方图；在同一张量上对比 BatchNorm 与 LayerNorm。'
		),
		engine: 'Tiny MLP / TinyGPT tensors',
		mode: 'live',
		links: ['tinygpt', 'resnet-18']
	},
	{
		slug: 'residual-connections',
		section: 'foundations',
		title: f('Residual connections and vanishing gradients', '残差连接与梯度消失'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Toggle skip connections in a 20-layer MLP and watch the gradient-norm bars per layer.',
			'在 20 层 MLP 中开关跳跃连接，观察每层的梯度范数条形图。'
		),
		engine: 'Own core',
		mode: 'live',
		links: ['resnet-18', 'tinygpt']
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
		status: 'planned',
		v1: false,
		interaction: f(
			'Apply temperature, top-k and top-p to the same cached logits and generate token by token.',
			'在同一组缓存的 logits 上调节温度、top-k、top-p，并逐个词元生成。'
		),
		engine: 'Shared output-head component',
		mode: 'live',
		links: ['tinygpt', 'gpt-2', 'whisper-tiny', 'tinyseq2seq']
	},
	{
		slug: 'self-supervised-objectives',
		section: 'pretraining',
		title: f('Self-supervised objectives: an overview', '自监督目标总览'),
		status: 'planned',
		v1: true,
		interaction: f(
			'One sentence or image, many objectives: what is hidden and what is predicted under causal LM, masked LM, span corruption, denoising, fill-in-the-middle, contrastive learning, MAE, JEPA and diffusion, each drawn as a mask pattern.',
			'同一句话或同一张图，多种目标：因果语言模型、掩码语言模型、片段破坏、去噪、中间填充、对比学习、MAE、JEPA 与扩散各自遮住什么、预测什么，都画成遮罩图案。'
		),
		engine: 'TinyGPT live + illustrative mask diagrams (labeled)',
		mode: 'both',
		links: ['tinygpt', 'tinybert', 'tinyvit']
	},
	{
		slug: 'masked-lm',
		section: 'pretraining',
		title: f('Masked language modeling', '掩码语言模型'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Click any token to mask it, animate the 80/10/10 rule, and compare TinyBERT with TinyGPT trained on the same corpus; mask amino acids with ESM-2.',
			'点击任意词元将其遮住，动画演示 80/10/10 规则，对比在同一语料上训练的 TinyBERT 与 TinyGPT；用 ESM-2 遮蔽氨基酸。'
		),
		engine: 'TinyBERT, ESM-2',
		mode: 'live',
		links: ['tinybert', 'bert-base', 'esm-2']
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
		status: 'planned',
		v1: true,
		interaction: f(
			'Add your own drawings and captions to a batch, see the N×N similarity matrix, drag the temperature and watch the loss; a training slider morphs the matrix from noise to a clean diagonal.',
			'把你的画和描述加入一个批次，查看 N×N 相似度矩阵，拖动温度看损失变化；训练进度滑块让矩阵从噪声变成清晰的对角线。'
		),
		engine: 'TinyCLIP live; CLIP precomputed',
		mode: 'live',
		links: ['tinyclip', 'clip']
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
		status: 'planned',
		v1: true,
		interaction: f(
			"Train a 2D MLP on task A, then task B, and watch A's decision boundary erode live. Try replay, a pull toward the old weights, freezing and a lower learning rate.",
			'先在任务 A 上训练二维 MLP，再训练任务 B，实时观察 A 的决策边界被侵蚀。尝试经验回放、向旧权重的拉力、冻结和更低的学习率。'
		),
		engine: '2D playground (live) + TinyGPT (precomputed)',
		mode: 'both',
		links: ['mlp-playground', 'tinygpt']
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
		title: f('LoRA, QLoRA and DoRA', 'LoRA、QLoRA 与 DoRA'),
		status: 'planned',
		v1: true,
		interaction: f(
			'A live parameter counter (r·(d_in + d_out) vs d_in·d_out); animate W₀ + (α/r)·BA with B starting at zero, so step 0 equals the base model; live LoRA on TinyGPT and precomputed rank sweeps.',
			'实时参数计数器（r·(d_in + d_out) 对比 d_in·d_out）；动画演示 W₀ + (α/r)·BA，B 从零开始，因此第 0 步等于基础模型；在 TinyGPT 上实时 LoRA，以及预计算的秩扫描。'
		),
		engine: 'Analytic JS + own core + precomputed sweeps',
		mode: 'both',
		links: ['gpt-2', 'tinygpt', 'flow-2d', 'stable-diffusion']
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
		status: 'planned',
		v1: true,
		interaction: f(
			'Click bits for FP32, FP16, BF16, FP8 (e4m3, e5m2), FP4, INT4 and NF4 and see the value, range and rounding error change.',
			'点击 FP32、FP16、BF16、FP8（e4m3、e5m2）、FP4、INT4、NF4 的各个比特位，查看数值、范围与舍入误差的变化。'
		),
		engine: 'JS',
		mode: 'live',
		links: []
	},
	{
		slug: 'quantization',
		section: 'compression',
		title: f('Quantization', '量化'),
		status: 'planned',
		v1: true,
		interaction: f(
			'A rounding ledger for 8 values (absmax vs affine, clipping, block size); one outlier wrecking the scale; quantize a TinyGPT layer and watch logits, top-1 and perplexity change.',
			'8 个数值的舍入账本（absmax 与仿射、截断、分块大小）；一个离群值毁掉缩放比例；量化 TinyGPT 的一层，观察 logits、top-1 与困惑度的变化。'
		),
		engine: 'JS + real models',
		mode: 'live',
		links: ['tinygpt', 'kokoro', 'slimsam', 'dinov2']
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
		status: 'planned',
		v1: false,
		interaction: f(
			'TinyGPT generation with and without the cache: recomputed cells grayed out, a cache-memory counter and shared-prefix reuse across two prompts.',
			'TinyGPT 有无缓存的生成对比：重复计算的单元变灰、缓存内存计数器，以及两个提示词之间的共享前缀复用。'
		),
		engine: 'TinyGPT live',
		mode: 'live',
		links: ['tinygpt', 'gpt-2']
	},
	{
		slug: 'attention-variants',
		section: 'inference',
		title: f('Attention variants (MHA, MQA, GQA, MLA)', '注意力变体（MHA、MQA、GQA、MLA）'),
		status: 'planned',
		v1: false,
		interaction: f('A head-sharing diagram with a KV-cache size calculator, and TinyGPT trained with MHA vs GQA.', '头共享示意图与 KV 缓存大小计算器，以及分别用 MHA 与 GQA 训练的 TinyGPT。'),
		engine: 'Analytic + precomputed',
		mode: 'both',
		links: ['tinygpt']
	},
	{
		slug: 'positional-encodings',
		section: 'inference',
		title: f('Positional encodings and long context', '位置编码与长上下文'),
		status: 'planned',
		v1: false,
		interaction: f(
			'Sinusoidal vs learned vs RoPE (rotate the 2D pair) vs ALiBi (a bias ramp), and extrapolation beyond the training context.',
			'正弦、可学习、RoPE（旋转二维对）与 ALiBi（偏置斜坡）的对比，以及超出训练长度的外推。'
		),
		engine: 'Live + precomputed',
		mode: 'both',
		links: ['tinygpt']
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
