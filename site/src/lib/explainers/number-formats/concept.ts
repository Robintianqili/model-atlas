// number-formats: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { show } from './views/bits/meta.ts';
import { castIndex } from './process.ts';

export const hook: I18n = {
	en: 'Every number inside a neural network is stored in a fixed number of bits. fp32 uses 32; training today often runs in 16, and inference in 8 or even 4. A format is a deal: its bits are split between how far it reaches and how finely it steps. This page lets you click the bits of one number in nine formats and watch what it becomes, then casts a real language model to each of them.',
	zh: '神经网络里的每个数都存放在固定数目的比特里。fp32 用 32 位；如今训练常用 16 位，推理用 8 位甚至 4 位。每种格式都是一种取舍：它的比特要在“能表示多远”和“刻度有多细”之间分配。在本页，你可以点击一个数在九种格式里的各个比特位，看它变成什么，再把一个真实的语言模型换成其中每一种格式。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your number in all nine formats', zh: '你的数在全部九种格式里' },
	source: 'formats',
	at: 'first',
	view: { key: 'number-formats/bits', props: () => ({ detail: 'fp32', flip: true }) },
	provenance: 'live',
	narration: {
		en: 'Your x = {x} in every format of the page. Press any bit of any row to flip it: x becomes the value of the new pattern, and every row follows. Type a number, or pick one, in the input widget.',
		zh: '你的 x = {x} 在本页每种格式里的样子。按任意一行的任意一位就能翻转它：x 变成新位模式的值，每一行都会跟着变。也可以在输入控件里输入或挑选一个数。',
		vars: (_rt, at) => ({ x: show(at.frame?.metrics?.x ?? NaN, 9) })
	},
	look: { en: 'The status at the end of each row says what happened: rounded, subnormal, overflow, clipped.', zh: '每行末尾的状态说明发生了什么：舍入、次正规数、上溢、截断。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT in every format', zh: '每种格式下的 TinyGPT' },
	source: 'sweep',
	at: 'last',
	view: { key: 'number-formats/gpt', props: () => ({ controls: true }) },
	provenance: 'live',
	terms: ['perplexity', 'kl-divergence'],
	narration: {
		en: 'TinyGPT with {params} parameters, its weight matrices cast to each format in your browser and scored on {tokens} tokens of four stories. Pick a setting in the stage: the four numbers and the next-token bars of the probe follow it. On so few tokens a gap of a few hundredths is noise: e4m3 with one scale per matrix scores {pt|n4} here against {pn|n4} without, while on {fact:bigTokens} tokens PyTorch measures {fact:pplE4m3T|n4} against {fact:pplE4m3|n4}.',
		zh: '有 {params} 个参数的 TinyGPT，在你的浏览器里把权重矩阵换成每种格式，在四个故事的 {tokens} 个词元上评测。在舞台里选一种设置：四个指标和探针句的下一词元柱状图都会跟着变。词元这么少，困惑度相差零点零几只是噪声：e4m3 每个矩阵配一个缩放比例时，这里的困惑度是 {pt|n4}，不加缩放是 {pn|n4}；而在 {fact:bigTokens} 个词元上，PyTorch 测得的是 {fact:pplE4m3T|n4} 和 {fact:pplE4m3|n4}。',
		vars: (_rt, at) => {
			const p = (id: string) => {
				const i = at.tape?.find({ phase: 'cast', step: castIndex(id) }) ?? -1;
				return i >= 0 ? (at.tape!.frame(i).metrics?.ppl ?? NaN) : NaN;
			};
			return { params: at.frame?.metrics?.params ?? NaN, tokens: at.frame?.metrics?.tokens ?? NaN, pt: p('e4m3-t'), pn: p('e4m3') };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Range against precision, side by side', zh: '范围与精度，并排对比' },
	source: 'formats',
	at: 'first',
	view: { key: 'number-formats/range', props: () => ({ formats: ['fp32', 'bf16', 'fp16', 'e5m2', 'e4m3', 'fp4', 'int8', 'int4', 'nf4'], table: true }) },
	provenance: 'live',
	narration: {
		en: 'Every format of the page on one axis of sizes, with your x = {x} marked, and the trade-off table under it: the more exponent bits, the longer the bar; the more fraction bits, the smaller the gap at 1. The integer formats and NF4 are drawn with a scale of 1.',
		zh: '本页所有格式画在同一条大小轴上，标出了你的 x = {x}，下面是取舍表：指数位越多，条越长；尾数位越多，1 处的间隔越小。整数格式和 NF4 按缩放比例 1 画出。',
		vars: (_rt, at) => ({ x: show(at.frame?.metrics?.x ?? NaN, 9) })
	}
};

export const pitfalls = [
	{
		claim: { en: 'fp16 and bf16 are interchangeable: both use 16 bits.', zh: 'fp16 和 bf16 可以互换：两者都是 16 位。' },
		truth: {
			en: 'bf16 reaches as far as fp32 with coarse steps; fp16 has finer steps but overflows above 65,504 and loses small gradients. Code written for one can overflow or underflow in the other.',
			zh: 'bf16 的范围和 fp32 一样远，但刻度粗；fp16 的刻度更细，却在超过 65,504 时上溢，还会丢掉很小的梯度。为其中一种写的代码，换成另一种就可能上溢或下溢。'
		},
		step: 'range16'
	},
	{
		claim: { en: 'A smaller format only rounds a little more.', zh: '更小的格式只是多舍入一点。' },
		truth: {
			en: 'Out of range, numbers do not round a little: they become 0, infinity or NaN (not a number). Without a scale, e4m3 stores 0.001 almost twice too large, and FP4 turns most of TinyGPT’s weights into 0 and breaks the model.',
			zh: '超出范围时，数不只是多舍入一点：它们会变成 0、无穷大或 NaN（非数）。不加缩放时，e4m3 把 0.001 存成了几乎两倍大，FP4 把 TinyGPT 的大部分权重变成 0，模型就坏了。'
		},
		step: 'tiny'
	},
	{
		claim: { en: 'Training in bf16 means keeping the weights only in bf16.', zh: '用 bf16 训练，就是只用 bf16 保存权重。' },
		truth: {
			en: 'Mixed-precision training computes in bf16 but keeps an fp32 master copy of the weights: an update smaller than half a bf16 gap would otherwise round away, step after step.',
			zh: '混合精度训练用 bf16 计算，却保留一份 fp32 主权重：否则小于 bf16 半个间隔的更新会一步又一步地被舍入掉。'
		},
		step: 'mixed'
	}
];

export const further = [
	{ label: { en: 'Quantization on this site: scales, outliers and int4 on TinyGPT', zh: '本站的量化页面：缩放比例、离群值与 TinyGPT 上的 int4' }, href: '/train/quantization/' },
	{ label: { en: 'LoRA and QLoRA on this site (QLoRA stores the base model in NF4)', zh: '本站的 LoRA 与 QLoRA 页面（QLoRA 用 NF4 存基础模型）' }, href: '/train/lora/' },
	{ label: { en: 'FP8 formats for deep learning (Micikevicius et al., 2022)', zh: '用于深度学习的 FP8 格式（Micikevicius 等，2022）' }, href: 'https://arxiv.org/abs/2209.05433' },
	{ label: { en: 'Mixed precision training (Micikevicius et al., 2018)', zh: '混合精度训练（Micikevicius 等，2018）' }, href: 'https://arxiv.org/abs/1710.03740' },
	{ label: { en: 'QLoRA: efficient finetuning of quantized LLMs, with NF4 (Dettmers et al., 2023)', zh: 'QLoRA：量化大模型的高效微调，提出 NF4（Dettmers 等，2023）' }, href: 'https://arxiv.org/abs/2305.14314' },
	{ label: { en: 'Microscaling data formats for deep learning, with FP4 (Rouhani et al., 2023)', zh: '用于深度学习的微缩放数据格式，含 FP4（Rouhani 等，2023）' }, href: 'https://arxiv.org/abs/2310.10537' }
];
