// quantization: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';

export const hook: I18n = {
	en: 'Every weight of a trained network is a floating-point number, usually fp32. Quantization stores each one as a small whole number, a code of 8 or even 4 bits, and keeps one shared scale that turns the codes back into numbers. The model gets several times smaller. This page asks what the rounding costs, and how to keep that cost small, first on eight numbers you choose, then on a real language model running in your browser.',
	zh: '训练好的网络里，每个权重都是一个浮点数，通常是 fp32。量化把每个权重存成一个小整数，也就是 8 位甚至 4 位的编码，再保留一个共用的缩放比例，用来把编码换回数值。模型因此小了好几倍。本页要回答的是：舍入的代价是什么，怎样让这个代价变小？先在你自己选的八个数上试，再在浏览器里运行的一个真实语言模型上试。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your eight values, any setting', zh: '你的八个值，任意设置' },
	source: 'toy',
	at: 'last',
	view: { key: 'quantization/ledger', props: () => ({ controls: true }) },
	provenance: 'live',
	narration: {
		en: 'Your eight values, from {lo} to {hi}. Pick the bits, the scheme, clipping and the block size in the stage, and watch the codes, the rulers and the errors change. Edit the values, or start from a set, in the input widget.',
		zh: '你的八个值，从 {lo} 到 {hi}。在舞台里选择位数、方案、是否截断和块大小，看编码、尺子和误差怎样变化。在输入控件里可以修改这些值，或者从一组预设开始。',
		vars: (_rt, at) => {
			const x = Array.from(((at.state as { x?: { data: ArrayLike<number> } } | undefined)?.x?.data ?? []) as ArrayLike<number>);
			return { lo: x.length ? Math.min(...x) : NaN, hi: x.length ? Math.max(...x) : NaN };
		}
	},
	look: { en: 'The buttons in the stage change the setting; the numbers stay yours.', zh: '舞台里的按钮切换设置；数值始终是你的。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT, quantized in your browser', zh: '在你的浏览器里量化 TinyGPT' },
	source: 'sweep',
	at: 'last',
	view: { key: 'quantization/gpt', props: () => ({ controls: true }) },
	provenance: 'live',
	terms: ['perplexity', 'kl-divergence'],
	narration: {
		en: 'TinyGPT with {params} parameters, scored on {tokens} tokens of four stories in fp32 and in ten quantized settings. Pick a setting in the stage: the four numbers and the next-token bars of the probe follow it.',
		zh: '有 {params} 个参数的 TinyGPT，用 fp32 和十种量化设置在四个故事的 {tokens} 个词元上评测。在舞台里选一种设置：四个指标和探针句的下一词元柱状图都会跟着变。',
		vars: (_rt, at) => ({ params: at.frame?.metrics?.params ?? NaN, tokens: at.frame?.metrics?.tokens ?? NaN })
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Every weight matrix: int8 against int4', zh: '所有权重矩阵：int8 对比 int4' },
	source: 'sweep',
	at: 'last',
	view: { key: 'quantization/gpt', props: () => ({ table: 'all', probe: false }) },
	provenance: 'live',
	narration: {
		en: 'The six whole-model settings side by side. int8 per tensor reaches a KL divergence of {kl8t}; int4 per tensor, {kl4t}. On {fact:bigTokens} tokens PyTorch measures perplexities of {fact:pplAll8t} and {fact:pplAll4t} for them, against {fact:bigPpl0} in fp32.',
		zh: '六种整模型设置并排对比。int8 按张量时 KL 散度为 {kl8t}；int4 按张量时为 {kl4t}。在 {fact:bigTokens} 个词元上，PyTorch 测得它们的困惑度分别是 {fact:pplAll8t} 和 {fact:pplAll4t}，而 fp32 是 {fact:bigPpl0}。',
		vars: (_rt, at) => {
			const get = (step: number) => {
				const i = at.tape?.find({ phase: 'all', step }) ?? -1;
				return i >= 0 ? (at.tape!.frame(i).metrics?.kl ?? NaN) : NaN;
			};
			return { kl8t: get(0), kl4t: get(3) };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'Rounding changes each weight so little that quantization is free.', zh: '舍入只让每个权重变化一点点，所以量化没有代价。' },
		truth: {
			en: 'The step between codes is set by the largest value of the group. One outlier makes the steps coarse for every other value, and on TinyGPT int4 with one scale per matrix breaks the model.',
			zh: '编码之间的步长由组里最大的值决定。一个离群值会让其他所有值的刻度都变粗；在 TinyGPT 上，int4 且每个矩阵只用一个缩放比例时，模型就坏掉了。'
		},
		step: 'outlier'
	},
	{
		claim: { en: 'int4 makes the model file eight times smaller than fp32.', zh: 'int4 能让模型文件比 fp32 小八倍。' },
		truth: {
			en: 'Only the quantized weights shrink eightfold. The scales cost extra bits, and whatever stays in floating point (here the token embedding) keeps its size.',
			zh: '只有被量化的权重缩小到八分之一。缩放比例要额外占位，仍保持浮点的部分（这里是词元嵌入）大小不变。'
		},
		step: 'whole'
	},
	{
		claim: { en: 'If perplexity barely moves, the quantized model gives the same answers.', zh: '只要困惑度几乎不变，量化后的模型给出的答案就一样。' },
		truth: {
			en: 'Perplexity averages over many tokens and can even improve by chance on a short text. The top-1 agreement and the KL divergence compare the two models position by position and show differences that perplexity hides.',
			zh: '困惑度是许多词元上的平均，在短文本上甚至可能碰巧变好。top-1 一致率和 KL 散度逐个位置比较两个模型，能看出困惑度掩盖的差别。'
		},
		step: 'layer'
	}
];

export const further = [
	{ label: { en: 'Quantization from the ground up (Sam Rose, ngrok blog)', zh: '从零开始理解量化（Sam Rose，ngrok 博客）' }, href: 'https://ngrok.com/blog/quantization' },
	{ label: { en: 'LLM.int8(): 8-bit matrix multiplication for transformers at scale (Dettmers et al., 2022)', zh: 'LLM.int8()：大规模 Transformer 的 8 位矩阵乘法（Dettmers 等，2022）' }, href: 'https://arxiv.org/abs/2208.07339' },
	{ label: { en: 'A survey of quantization methods for efficient neural network inference (Gholami et al., 2021)', zh: '高效神经网络推理的量化方法综述（Gholami 等，2021）' }, href: 'https://arxiv.org/abs/2103.13630' }
];
