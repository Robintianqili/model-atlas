// lora: the manifest of a CONCEPT package (K2): routed at /train/lora/ by the concept route (AutoConceptPage: goals,
// hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). LoRA (low-rank adaptation) on one 16 × 16 layer,
// trained live with the kit's autograd and Adam at the learner's rank r, plus the analytic parameter counter and memory
// bars for real layer sizes (TinyGPT, GPT-2 small). Sources:
//   fit    LoRA at the learner's rank: frame 'start' (B = 0) then one frame per Adam update      (process 'lora')
//   ranks  the same training at r = 1, 2, 4, 8 and full fine-tuning, in lockstep; ignores the input (process 'ranks')
// Both are kind C processes run in the package worker, checked frame by frame against pipeline/lora/reference.py
// (PyTorch float64 autograd and torch.optim.Adam on the same world; `atlas parity lora`). The claim tests also check
// them against the closed forms of data.ts (loss = ‖ΔW − ΔW*‖²/16, the floors). The page around the lesson is concept.ts.
import type { ExplainerManifest } from '../../atlas/types/index.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const OPT = { steps: 100, lr: 0.02 };

const manifest: ConceptManifest = {
	slug: 'lora',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'rank', start: { r: 4 } },
	sources: {
		fit: { kind: 'process', process: 'lora', opt: OPT },
		ranks: { kind: 'process', process: 'ranks', input: () => null, opt: OPT }
	},
	page: {
		goals: [
			{
				en: 'Count what LoRA trains: r·(d_in + d_out) numbers instead of d_in·d_out, and say why the saving grows with the size of the layer.',
				zh: '算出 LoRA 训练多少个数：r·(d_in + d_out) 个，而不是 d_in·d_out 个，并说明为什么层越大省得越多。'
			},
			{
				en: 'Explain why W0 + (α/r)·B·A with B = 0 at the start gives exactly the base model before training.',
				zh: '解释为什么 W0 + (α/r)·B·A 在 B 从 0 开始时，训练前恰好就是基础模型。'
			},
			{
				en: 'Predict from the needed change how low a rank-r adapter can get, and check it on live runs.',
				zh: '根据所需的改变量，预测秩为 r 的适配器能把损失降到多低，并用实时运行来验证。'
			},
			{
				en: 'Tell trainable numbers from training memory: weights, gradients and optimizer state.',
				zh: '分清可训练的数和训练所需的内存：权重、梯度与优化器状态。'
			}
		],
		prequestion: {
			id: 'lora-pre',
			question: {
				en: 'A model is already trained. To teach it a new task, which numbers have to change?',
				zh: '一个模型已经训练好了。要教它一个新任务，需要改哪些数？'
			},
			options: [
				{ en: 'All of its weights, or it cannot learn anything new', zh: '它的全部权重，否则学不会新东西' },
				{ en: 'A small add-on can be enough: the change it needs is often simple', zh: '加一个小小的附加部分可能就够了：它需要的改变往往很简单' },
				{ en: 'None: a trained model cannot be changed', zh: '一个都不用改：训练好的模型无法改变' }
			],
			answer: 1,
			explain: {
				en: 'LoRA freezes every weight and learns only a small low-rank change on top. It works when the change a task needs has a few strong directions, which this page lets you measure.',
				zh: 'LoRA 冻结所有权重，只在上面学一个小的低秩改变量。当任务需要的改变只有少数几个强方向时，它就够用；本页让你亲手量一量。'
			}
		},
		quiz: [
			{
				q: { en: 'A 1,000 × 1,000 layer gets a LoRA adapter of rank 4. How many numbers does the adapter train?', zh: '一个 1,000 × 1,000 的层加上秩为 4 的 LoRA 适配器。适配器要训练多少个数？' },
				options: [{ en: '8,000', zh: '8,000' }, { en: '4,000', zh: '4,000' }, { en: '1,000,000', zh: '1,000,000' }],
				answer: 0,
				explain: { en: 'r·(d_in + d_out) = 4 × (1,000 + 1,000) = 8,000: B is 1,000 × 4 and A is 4 × 1,000. Full fine-tuning would train all 1,000,000.', zh: 'r·(d_in + d_out) = 4 × (1,000 + 1,000) = 8,000：B 是 1,000 × 4，A 是 4 × 1,000。全量微调要训练全部 1,000,000 个。' }
			},
			{
				q: { en: 'Why does B start at zero instead of with random numbers?', zh: '为什么 B 从零开始，而不是随机数？' },
				options: [
					{ en: 'So that B·A = 0 and training starts from exactly the base model', zh: '这样 B·A = 0，训练恰好从基础模型出发' },
					{ en: 'To use less memory', zh: '为了少占内存' },
					{ en: 'Because zero is the best answer for B', zh: '因为 0 就是 B 的最佳答案' }
				],
				answer: 0,
				explain: { en: 'Any matrix times zero is zero, so W0 + (α/r)·B·A = W0 before the first update: the adapted model starts as the base model and only drifts as far as the task asks.', zh: '任何矩阵乘以零都是零，所以第一次更新之前 W0 + (α/r)·B·A = W0：适配后的模型从基础模型出发，只按任务需要的程度偏离。' }
			}
		],
		misconception: {
			claim: { en: 'LoRA saves memory in proportion to the parameters it does not train.', zh: 'LoRA 省下的内存与它不训练的参数成正比。' },
			truth: {
				en: 'The frozen weights still sit in memory. LoRA removes their gradients and optimizer state, so on GPT-2 small at rank 8 it trains about one percent of the numbers but still needs about a quarter of the memory.',
				zh: '冻结的权重仍然占着内存。LoRA 省掉的是它们的梯度和优化器状态，所以在 GPT-2 small 上，秩为 8 时它只训练约百分之一的数，内存却仍要大约四分之一。'
			},
			step: 'memory'
		},
		lessonNote: {
			en: 'Ten steps: count what LoRA trains, watch B grow from zero on a real layer, find the smallest rank that does the job, then see what LoRA cannot do.',
			zh: '十步：数一数 LoRA 训练多少个数，看 B 在一个真实的层上从零长起来，找出够用的最小秩，再看 LoRA 做不到什么。'
		}
	},
	glossary: [
		{
			id: 'matrix-rank',
			term: { en: 'rank (of a matrix)', zh: '秩（矩阵的）' },
			def: {
				en: 'How many independent directions a matrix has. A d_out × d_in matrix of rank r can be written as B·A with B of size d_out × r and A of size r × d_in.',
				zh: '一个矩阵有多少个相互独立的方向。秩为 r 的 d_out × d_in 矩阵可以写成 B·A，其中 B 是 d_out × r，A 是 r × d_in。'
			},
			related: ['lora']
		},
		{
			id: 'optimizer-state',
			term: { en: 'optimizer state', zh: '优化器状态' },
			def: {
				en: 'Numbers an optimizer keeps for every trainable parameter between updates; Adam keeps two running averages, of the gradient and of its square.',
				zh: '优化器在两次更新之间为每个可训练参数保存的数；Adam 保存两个滑动平均：梯度的，以及梯度平方的。'
			},
			related: ['optimizer']
		}
	],
	credits: [
		{
			name: 'Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)',
			url: 'https://arxiv.org/abs/2106.09685',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'The update W0 + (α/r)·B·A with W0 frozen, B starting at zero and a fixed α, and the idea that the change a task needs has a low intrinsic rank',
			attribution: 'Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang and Weizhu Chen'
		},
		{
			name: 'microsoft/LoRA (loralib)',
			url: 'https://github.com/microsoft/LoRA',
			license: 'MIT; idea only (no code copied)',
			usedFor: 'The initialization convention: A drawn like a PyTorch linear layer\'s weights (uniform in ±1/√d_in), B all zeros',
			attribution: 'Microsoft Corporation'
		},
		{
			name: 'Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023)',
			url: 'https://arxiv.org/abs/2305.14314',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'The QLoRA bar of the memory step: the frozen weights stored with 4 bits each, the adapter trained as in LoRA',
			attribution: 'Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer'
		},
		{
			name: 'Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2, 2019)',
			url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
			license: 'Paper; published model configuration only',
			usedFor: 'The sizes of GPT-2 small (12 blocks, width 768, MLP width 3,072, 50,257 tokens, 1,024 positions) for the parameter counter and the memory bars',
			attribution: 'Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever'
		}
	]
};

export default manifest;
