// init-normalization: the manifest of a CONCEPT package (K2): routed at /train/init-normalization/ by the concept route,
// which renders AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links).
// A 10-layer MLP (64 units per layer, 256 random examples) run once forward and once backward by the kit's autograd,
// under the five classic initializations, with tanh or ReLU, with or without BatchNorm / LayerNorm; then BatchNorm vs
// LayerNorm on the same small tensor and on TinyGPT's real residual stream. Sources:
//   mine    the learner's setup (σ, activation, normalization)                               (process 'mlp')
//   xavier, zero, small, large   tanh, no normalization, σ = Xavier, 0, 0.01, 1                (process 'mlp')
//   relus   ReLU, no normalization, all five initializations                                 (process 'inits')
//   inits   the learner's activation and normalization, all five initializations             (process 'inits')
//   gpt     TinyGPT's spec and weights (static/models/tinygpt, read only); no forward pass here  (trace, inputs: ids)
//   tgpt    TinyGPT on its replay probe: block 1's output normalized two ways                    (process 'gpt')
// No source trains anything. The network runs use the kit's autograd (tested against PyTorch fp64 in the kit's own
// tests) and TinyGPT runs the kit's executor (parity-checked against PyTorch when TinyGPT was exported); no Python
// reference of their own, so every badge says "Live". The page around the lesson is concept.ts.
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { INIT_STD, setupOf, type Setup } from './data.ts';
import type { GptInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const tanh = (std: number) => (): Setup => ({ act: 'tanh', std, norm: 'none' });

/** TinyGPT's spec and weights for the 'gpt' process, copied once per loaded model (each tensor gets its own buffer, so
 *  the worker receives them, and the same object every time, so a new learner input keeps the run). Copied from the
 *  quantization package (explainer.ts gptInput). */
const copies = new WeakMap<object, GptInput>();
export function gptInput(rt: ExplainerRuntime): GptInput {
	const m = rt.sources.gpt?.model;
	if (!m) return null;
	let c = copies.get(m);
	if (!c) {
		const weights: Record<string, Tensor> = {};
		for (const [k, t] of Object.entries(m.weights)) weights[k] = { shape: t.shape.slice(), data: (t.data as Float32Array).slice() };
		c = { spec: JSON.parse(JSON.stringify(m.spec)), weights };
		copies.set(m, c);
	}
	return c;
}

const manifest: ConceptManifest = {
	slug: 'init-normalization',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'setup', start: { act: 'tanh', std: INIT_STD.xavier, norm: 'none' } },
	sources: {
		mine: { kind: 'process', process: 'mlp', reference: false },
		xavier: { kind: 'process', process: 'mlp', input: tanh(INIT_STD.xavier), reference: false },
		zero: { kind: 'process', process: 'mlp', input: tanh(0), reference: false },
		small: { kind: 'process', process: 'mlp', input: tanh(INIT_STD.small), reference: false },
		large: { kind: 'process', process: 'mlp', input: tanh(INIT_STD.large), reference: false },
		relus: { kind: 'process', process: 'inits', input: () => ({ act: 'relu', norm: 'none' }), reference: false },
		inits: {
			kind: 'process',
			process: 'inits',
			input: (rt) => {
				const s = setupOf(rt.input);
				return { act: s.act, norm: s.norm };
			},
			reference: false
		},
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['ids'] },
		tgpt: { kind: 'process', process: 'gpt', input: gptInput, reference: false }
	},
	page: {
		goals: [
			{
				en: 'Read a deep network\'s activation and gradient histograms layer by layer, and spot numbers that fade away or blow up.',
				zh: '逐层读懂一个深层网络的激活值直方图和梯度直方图，看出哪些数在逐渐消失，哪些在爆炸。'
			},
			{
				en: 'Explain why starting weights that are all zero, too small or too large fail, and why Xavier suits tanh and He suits ReLU.',
				zh: '解释为什么全零、过小或过大的初始权重会失败，以及为什么 Xavier 适合 tanh、He 适合 ReLU。'
			},
			{
				en: 'Say over which axis BatchNorm and LayerNorm normalize, work out one value of each by hand, and say when BatchNorm breaks.',
				zh: '说出 BatchNorm 和 LayerNorm 分别沿哪个轴归一化，亲手算出两者各一个值，并说出 BatchNorm 在什么情况下会失效。'
			}
		],
		prequestion: {
			id: 'init-normalization-pre',
			question: {
				en: 'Training changes every weight anyway. Does it matter which random numbers a deep network starts from?',
				zh: '训练反正会改变每一个权重。一个深层网络从哪些随机数出发，重要吗？'
			},
			options: [
				{ en: 'Yes: a bad start can make the signals and the gradients fade or blow up, so training never gets going', zh: '重要：糟糕的起点会让信号和梯度消失或爆炸，训练根本无法开始' },
				{ en: 'No: training fixes any start', zh: '不重要：训练能纠正任何起点' },
				{ en: 'Only the speed changes, never the result', zh: '只影响速度，不影响结果' }
			],
			answer: 0,
			explain: {
				en: 'Training moves each weight by its gradient. If the numbers shrink layer by layer, the gradients reaching the first layers shrink too and those layers barely move; if they grow, the first updates throw the weights far off.',
				zh: '训练按梯度移动每个权重。如果数值一层层缩小，传回前几层的梯度也会缩小，这些层几乎不动；如果数值一层层变大，最初的几次更新就会把权重甩得很远。'
			}
		},
		quiz: [
			{
				q: { en: 'A layer has 256 inputs and uses ReLU. What spread σ does He initialization give its weights?', zh: '某层有 256 个输入，使用 ReLU。He 初始化给它的权重取多大的标准差 σ？' },
				options: [{ en: '√(2/256) ≈ 0.088', zh: '√(2/256) ≈ 0.088' }, { en: '√(1/256) = 0.0625', zh: '√(1/256) = 0.0625' }, { en: '2/256 ≈ 0.0078', zh: '2/256 ≈ 0.0078' }],
				answer: 0,
				explain: {
					en: 'He initialization sets σ² = 2 / fan-in: ReLU zeroes about half of its inputs, and the factor 2 makes up for the lost half of the variance. √(1/256) is what Xavier gives a square layer.',
					zh: 'He 初始化取 σ² = 2 / 输入数：ReLU 会把大约一半的输入变成 0，系数 2 正好补回丢掉的那一半方差。√(1/256) 是 Xavier 给方形层的值。'
				}
			},
			{
				q: { en: 'A batch holds 32 sentences of 10 tokens, each token 76 numbers. Over how many numbers does LayerNorm take one mean?', zh: '一个批次有 32 个句子，每句 10 个词元，每个词元 76 个数。LayerNorm 每次对多少个数求一个均值？' },
				options: [{ en: '76', zh: '76' }, { en: '320 (32 × 10)', zh: '320（32 × 10）' }, { en: '24,320 (all of them)', zh: '24,320（全部）' }],
				answer: 0,
				explain: {
					en: 'LayerNorm averages the 76 numbers of one token. BatchNorm on the same tensor would average each of the 76 features over the 320 tokens of the batch.',
					zh: 'LayerNorm 对一个词元的 76 个数求平均。同一个张量上的 BatchNorm 则是对 76 个特征中的每一个，在批次的 320 个词元上求平均。'
				}
			}
		],
		misconception: {
			claim: { en: 'If a network starts badly, training will fix it.', zh: '网络起点不好，训练会把它纠正过来。' },
			truth: {
				en: 'Training follows the gradients, and a bad start breaks them: with every weight at 0 no weight gets a gradient at all, and with weights that are too small the gradient that reaches the first layer is almost 0.',
				zh: '训练靠的是梯度，而糟糕的起点会破坏梯度：所有权重都为 0 时，没有一个权重能得到梯度；权重太小时，传回第一层的梯度几乎为 0。'
			},
			step: 'zero'
		},
		lessonNote: {
			en: 'Ten steps: one deep network started five ways, layer by layer, forward and backward; then BatchNorm and LayerNorm on the same numbers, and the case where BatchNorm breaks.',
			zh: '十步：同一个深层网络用五种方式初始化，一层层看前向和反向；再在同一组数上比较 BatchNorm 和 LayerNorm，以及 BatchNorm 失效的情形。'
		}
	},
	glossary: [
		{
			id: 'weight-initialization', term: { en: 'weight initialization', zh: '权重初始化' }, related: ['xavier-he-init', 'vanishing-gradient'],
			def: {
				en: 'How a network\'s weights are set before training starts: usually random numbers with a chosen spread σ. Too small and the numbers fade layer by layer; too large and they blow up or saturate.',
				zh: '训练开始前怎样给网络的权重赋值：通常是取一个选定标准差 σ 的随机数。σ 太小，数值一层层衰减；σ 太大，数值爆炸或饱和。'
			}
		},
		{
			id: 'xavier-he-init', term: { en: 'Xavier and He initialization', zh: 'Xavier 初始化和 He 初始化' }, related: ['weight-initialization', 'relu'],
			def: {
				en: 'Rules that pick σ from a layer\'s size so the variance stays the same from layer to layer: Xavier σ = √(2 / (fan-in + fan-out)) for tanh, He σ = √(2 / fan-in) for ReLU, which zeroes about half of its inputs.',
				zh: '根据层的大小选择 σ、让方差逐层保持不变的规则：Xavier 取 σ = √(2 / (输入数 + 输出数))，适合 tanh；He 取 σ = √(2 / 输入数)，适合会把大约一半输入变成 0 的 ReLU。'
			}
		},
		{
			id: 'exploding-gradient', term: { en: 'exploding gradient', zh: '梯度爆炸' }, related: ['vanishing-gradient', 'backpropagation'],
			def: {
				en: 'A gradient that grows on its way back through the layers because it is multiplied by large weights again and again; the first update then throws the weights far off.',
				zh: '梯度在反向穿过各层时被很大的权重一再相乘而越变越大；第一次更新就会把权重甩得很远。'
			}
		},
		{
			id: 'saturation', term: { en: 'saturation', zh: '饱和' }, related: ['activation-function', 'vanishing-gradient'],
			def: {
				en: 'A unit is saturated when its input is so large that its activation function is flat there, as tanh is near ±1: its output barely changes, and it passes back almost no gradient.',
				zh: '当一个单元的输入大到激活函数在那里几乎是平的（比如 tanh 接近 ±1 时），这个单元就饱和了：它的输出几乎不变，也几乎传不回梯度。'
			}
		},
		{
			id: 'batch-norm', term: { en: 'batch normalization (BatchNorm)', zh: '批归一化（BatchNorm）' }, related: ['layer-norm', 'batch'],
			def: {
				en: 'Rescaling each unit over the examples of a batch: subtract the batch\'s mean of that unit and divide by its spread, then apply a learned scale and shift. It needs a batch; LayerNorm instead works on one example at a time.',
				zh: '对每个单元在一个批次的样例上重新缩放：减去这个单元在批次上的均值、除以它的标准差，再乘以可学习的缩放并加上偏移。它需要一个批次；LayerNorm 则一次只处理一个样例。'
			}
		}
	],
	credits: [
		{
			name: 'AI Notes: Initializing neural networks (ideas)',
			url: 'https://www.deeplearning.ai/ai-notes/initialization/',
			license: 'ideas only (no code or text used)',
			usedFor: 'the idea of showing per-layer histograms of a deep network under zero, too small, Xavier and too large initialization; network, data and code written fresh',
			attribution: 'Katanforoosh and Kunin, deeplearning.ai (2019)'
		}
	]
};

export default manifest;
