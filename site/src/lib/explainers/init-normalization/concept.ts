// init-normalization: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { ACT_NAME, DEPTH, INITS, INIT_NAME, NORM_NAME, PROBE_TOKENS, actOfCode, initOf, metricsOf, normOfCode, stdText } from './data.ts';

export const hook: I18n = {
	en: 'Before a network learns anything, every weight has to start somewhere. Start them all at 0, too small or too large, and a deep network\'s numbers fade to nothing or blow up layer by layer, and its gradients do the same. Here a 10-layer network draws every layer\'s numbers as a histogram, so you can watch it happen, and then BatchNorm and LayerNorm hold the numbers in range.',
	zh: '网络在学到任何东西之前，每个权重都得有个起点。如果全部从 0 出发，或者太小、太大，一个深层网络的数值就会一层层衰减到几乎为 0，或者一层层爆炸，它的梯度也一样。这里一个 10 层网络把每一层的数值画成直方图，让你亲眼看到这个过程，再看 BatchNorm 和 LayerNorm 怎样把数值稳在合适的范围里。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your network, layer by layer', zh: '你的网络，一层一层看' },
	source: 'mine',
	at: { phase: 'backward' },
	view: 'init-normalization/layers',
	provenance: 'live',
	narration: {
		en: 'Your network uses {act}, σ = {std} ({init}) and {norm}. The activations\' spread goes from {s1} at layer 1 to {s10} at layer {top}, and the gradient flowing back from {g10} at layer {top} to {g1} at layer 1. Try all five starts with both activations, then switch on a normalization.',
		zh: '你的网络用的是 {act}、σ = {std}（{init}）、{norm}。激活值的标准差从第 1 层的 {s1} 变到第 {top} 层的 {s10}，传回来的梯度从第 {top} 层的 {g10} 变到第 1 层的 {g1}。把五种起点和两种激活函数都试一遍，再打开一种归一化。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			const id = initOf(m.std);
			return {
				act: ACT_NAME[actOfCode(m.act)],
				std: stdText(m.std),
				init: id ? INIT_NAME[id] : { en: 'your own σ', zh: '你自己设的 σ' },
				norm: NORM_NAME[normOfCode(m.norm)],
				s1: m.s1,
				s10: m.s10,
				g10: m.g10,
				g1: m.g1,
				top: DEPTH
			};
		}
	},
	look: { en: 'Each row is one layer; the bars at the right are each layer\'s spread on a log scale.', zh: '每一行是一层；右边的横条是各层在对数刻度上的标准差。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT\'s real numbers, normalized two ways', zh: 'TinyGPT 的真实数值，用两种方式归一化' },
	source: 'tgpt',
	at: 'last',
	view: { key: 'init-normalization/axes', props: () => ({ mode: 'gpt' }) },
	provenance: 'live',
	terms: ['layer-norm', 'batch-norm'],
	narration: {
		en: 'TinyGPT\'s first block turns the {n} tokens of “{probe}” into {n} rows of {d} numbers. LayerNorm, as TinyGPT does it, scales each token\'s row: “{tok}” has mean {rm} and spread {rs}, and our LayerNorm matches TinyGPT\'s own within {diff}. BatchNorm would scale each dimension\'s column over the tokens instead, mixing in later tokens: dimension {c} (counting from 0) is near {cm} for every token, so BatchNorm would turn “{tok}”\'s {x} into {bn}, while LayerNorm keeps it large, {ln}.',
		zh: 'TinyGPT 的第 1 块把“{probe}”的 {n} 个词元变成 {n} 行、每行 {d} 个数。LayerNorm 像 TinyGPT 那样，对每个词元的一行做缩放：“{tok}”的均值为 {rm}，标准差为 {rs}；我们算的 LayerNorm 与 TinyGPT 自己的层相差不超过 {diff}。BatchNorm 则会对每个维度的一列跨词元缩放，把后面的词元也混进来：维度 {c}（从 0 开始数）在每个词元上都接近 {cm}，所以 BatchNorm 会把“{tok}”的 {x} 变成 {bn}，而 LayerNorm 让它保持很大：{ln}。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return {
				n: m.tokens,
				d: m.dims,
				probe: PROBE_TOKENS.join(''),
				tok: (PROBE_TOKENS[m.row] ?? '').trim(),
				rm: m.rowMean,
				rs: m.rowStd,
				c: m.col,
				cm: m.colMean,
				x: m.x,
				bn: m.bn,
				ln: m.ln,
				diff: m.diff
			};
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Five starts for your activation and normalization', zh: '你的激活函数和归一化下的五种起点' },
	source: 'inits',
	at: 'last',
	view: 'init-normalization/spreads',
	provenance: 'live',
	narration: {
		en: 'The same network under all five starts, with {act} and {norm}. At layer {top}, the four starts that are not zero end with spreads from {lo|n4} ({loName}) to {hi|n4} ({hiName}), a factor of {ratio} apart. Switch a normalization on or off in your input and watch that factor.',
		zh: '同一个网络的五种起点，用 {act}、{norm}。到第 {top} 层，四种非零起点的标准差从 {lo|n4}（{loName}）到 {hi|n4}（{hiName}），相差 {ratio} 倍。在你的输入里打开或关闭归一化，看这个倍数怎样变化。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			const ids = INITS.filter((id) => id !== 'zero').sort((p, q) => m[`${p}10`] - m[`${q}10`]);
			const lo = ids[0];
			const hi = ids[ids.length - 1];
			return {
				top: DEPTH,
				act: ACT_NAME[actOfCode(m.act)],
				norm: NORM_NAME[normOfCode(m.norm)],
				lo: m[`${lo}10`],
				hi: m[`${hi}10`],
				loName: INIT_NAME[lo],
				hiName: INIT_NAME[hi],
				ratio: m[`${hi}10`] / m[`${lo}10`]
			};
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'Large starting weights make tanh saturate, so the gradients vanish.', zh: '初始权重太大会让 tanh 饱和，所以梯度会消失。' },
		truth: {
			en: 'They do saturate, but in this network the gradient grows on its way back: the few units that are not saturated pass it on multiplied by the large weights, about twice per layer. Too small vanishes; too large explodes.',
			zh: '确实会饱和，但在这个网络里梯度在反向传回时反而变大：少数没有饱和的单元把梯度乘以很大的权重再传回去，每层大约翻一倍。太小会消失，太大会爆炸。'
		},
		step: 'large'
	},
	{
		claim: { en: 'Xavier initialization suits every activation function.', zh: 'Xavier 初始化适合所有激活函数。' },
		truth: {
			en: 'Xavier keeps the variance for tanh, which is almost a straight line near 0. ReLU zeroes about half of its inputs, so with Xavier each layer loses about half of the variance; He initialization doubles σ² to make up.',
			zh: 'Xavier 能让 tanh 的方差保持不变，因为 tanh 在 0 附近几乎是一条直线。ReLU 会把大约一半的输入变成 0，所以用 Xavier 时每层大约丢掉一半方差；He 初始化把 σ² 加倍来补偿。'
		},
		step: 'relu'
	},
	{
		claim: { en: 'BatchNorm and LayerNorm are the same thing.', zh: 'BatchNorm 和 LayerNorm 是一回事。' },
		truth: {
			en: 'Both subtract a mean and divide by a spread, but over different axes: BatchNorm over the examples of a batch, for each unit; LayerNorm over the units of one example. In training on a batch of one, BatchNorm turns every number into 0.',
			zh: '两者都是减去均值、除以标准差，但沿不同的轴：BatchNorm 对每个单元跨批次里的样例计算；LayerNorm 跨一个样例的所有单元计算。训练时如果批次只有一个样例，BatchNorm 会把每个数都变成 0。'
		},
		step: 'layernorm'
	}
];

export const further = [
	{ label: { en: 'TinyGPT on this site: the lesson step “LayerNorm steadies each vector”', zh: '本站的 TinyGPT：课程中的“层归一化让每个向量保持稳定”一步' }, href: '/m/tinygpt/#lesson-ln' },
	{ label: { en: 'Glorot and Bengio (2010): Understanding the difficulty of training deep feedforward neural networks', zh: 'Glorot 与 Bengio（2010）：理解训练深层前馈神经网络的困难' }, href: 'https://proceedings.mlr.press/v9/glorot10a.html' },
	{ label: { en: 'He et al. (2015): Delving deep into rectifiers (He initialization)', zh: 'He 等（2015）：深入研究整流器（He 初始化）' }, href: 'https://arxiv.org/abs/1502.01852' },
	{ label: { en: 'Ioffe and Szegedy (2015): Batch normalization', zh: 'Ioffe 与 Szegedy（2015）：批归一化' }, href: 'https://arxiv.org/abs/1502.03167' },
	{ label: { en: 'Ba, Kiros and Hinton (2016): Layer normalization', zh: 'Ba、Kiros 与 Hinton（2016）：层归一化' }, href: 'https://arxiv.org/abs/1607.06450' },
	{ label: { en: 'deeplearning.ai AI Notes: Initializing neural networks', zh: 'deeplearning.ai AI Notes：神经网络的初始化' }, href: 'https://www.deeplearning.ai/ai-notes/initialization/' }
];
