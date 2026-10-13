// init-normalization: the guided lesson (ten steps; the concept page's primary path). Every number is live: each source
// runs the toy network once forward and once backward with the kit's autograd (process 'mlp' or 'inits').
//   deep -> zero -> predict (small σ) -> small (vanishing) -> large (saturation, exploding) -> relu (Xavier vs He)
//   -> batchnorm -> layernorm -> task (rescue a bad start with a normalization) -> limits (BatchNorm on a batch of one)
import type { At } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmtSetting, paren } from '../../atlas/core/format.ts';
import { ACT_NAME, BATCH, DEPTH, INIT_STD, NB, NF, NORM_NAME, N_WEIGHTS, SAT, WIDTH, actOfCode, metricsOf, normOfCode, setupOf, stdText } from './data.ts';
import { STARTS } from './adapters/setup/adapter.ts';
import { axesOf, cellOf } from './views/axes/meta.ts';

/** The chosen cell of the 4 × 6 tensor at this frame, under the view's mode. */
function cell(at: At, mode: 'batch' | 'layer' | 'one') {
	const pic = axesOf({ at, props: { mode } });
	if (!pic) return { r: NaN, c: NaN, x: NaN, m: NaN, sd: NaN, y: NaN, other: NaN };
	const k = cellOf(pic);
	const r = metricsOf(at).cr;
	if (mode === 'layer') return { r, c: pic.c, x: k.x, m: k.row.mean, sd: k.row.std, y: k.ln, other: k.bn };
	if (mode === 'one') return { r, c: pic.c, x: k.x, m: k.col.mean, sd: k.col.std, y: k.bn, other: k.ln };
	return { r, c: pic.c, x: k.x, m: k.col.mean, sd: k.col.std, y: k.bn, other: k.ln };
}
const setupText = (at: At) => {
	const m = metricsOf(at);
	return { act: ACT_NAME[actOfCode(m.act)], std: stdText(m.std), norm: NORM_NAME[normOfCode(m.norm)] };
};

const lesson = defineLesson({
	title: { en: 'Initialization and normalization, step by step', zh: '初始化与归一化，分步课程' },
	summary: {
		en: 'Start one deep network five ways and watch each layer\'s numbers and gradients fade, hold or blow up; then normalize the same tensor with BatchNorm and LayerNorm, and see where BatchNorm breaks.',
		zh: '用五种方式初始化同一个深层网络，看每一层的数值和梯度怎样衰减、保持或爆炸；再用 BatchNorm 和 LayerNorm 归一化同一个张量，看看 BatchNorm 在哪里失效。'
	},
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'deep',
			title: { en: 'Ten layers deep', zh: '十层深的网络' },
			source: 'xavier',
			at: { phase: 'forward' },
			view: { key: 'init-normalization/layers', props: () => ({ grads: false }) },
			provenance: 'live',
			terms: ['weight-initialization', 'activation', 'variance'],
			narration: {
				en: 'The toy network sends {b} examples of {w} random numbers through {d} layers of {w} tanh units. Every weight starts as a random number with spread (standard deviation) σ = {std}, the Xavier value √(1/{w}) for these layers. The activations\' spread is {s1} at layer 1 and {s10} at layer {top}: it shrinks slowly, by a factor of about {f} per layer.',
				zh: '这个玩具网络把 {b} 个样例（每个是 {w} 个随机数）送过 {d} 层、每层 {w} 个 tanh 单元。每个权重一开始都是标准差 σ = {std} 的随机数，也就是这些层的 Xavier 取值 √(1/{w})。激活值的标准差在第 1 层是 {s1}，在第 {top} 层是 {s10}：缓慢缩小，每层大约乘以 {f}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { b: BATCH, w: WIDTH, d: DEPTH, top: DEPTH, std: stdText(m.std), s1: m.s1, s10: m.s10, f: m.f };
				}
			},
			look: {
				en: 'Each row is one layer: the histogram of all its activations, then its spread as a bar on a log scale.',
				zh: '每一行是一层：先是这一层所有激活值的直方图，再用对数刻度上的横条表示它的标准差。'
			}
		},
		{
			id: 'zero',
			title: { en: 'Every weight 0: nothing to learn from', zh: '所有权重为 0：无从学起' },
			source: 'zero',
			at: { phase: 'backward' },
			view: 'init-normalization/layers',
			provenance: 'live',
			terms: ['gradient', 'backpropagation'],
			narration: {
				en: 'Now every weight starts at 0. Every weighted sum is 0, so every unit of every layer outputs tanh(0) = 0. Backpropagation has nothing to carry back: {nz} of the {nw} weights get a gradient other than 0, and only the output\'s bias gets one ({gb}). The network cannot even start to learn.',
				zh: '现在每个权重一开始都是 0。每个加权和都是 0，所以每一层的每个单元都输出 tanh(0) = 0。反向传播没有东西可以传回去：{nw} 个权重中，梯度不为 0 的有 {nz} 个，只有输出的偏置得到了梯度（{gb}）。这个网络连学习的第一步都迈不出去。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { nz: m.nz, nw: N_WEIGHTS, gb: m.gb };
				}
			},
			look: { en: 'Every histogram is a single spike at 0 and every spread is 0: each activation and each gradient is exactly 0.', zh: '每个直方图都只是 0 处的一根尖峰，每个标准差都是 0：每个激活值和每个梯度都正好是 0。' }
		},
		{
			id: 'predict',
			title: { en: 'Weights that are too small', zh: '权重太小' },
			source: 'small',
			at: { phase: 'forward' },
			view: { key: 'init-normalization/layers', props: () => ({ grads: false }) },
			provenance: 'live',
			predict: {
				id: 'init-normalization-small',
				question: { en: 'What happens to the spread of the activations on the way from layer 1 to layer 10?', zh: '从第 1 层到第 10 层，激活值的标准差会怎样变化？' },
				options: [
					{ en: 'It shrinks at every layer, to almost nothing', zh: '每一层都在缩小，最后几乎为 0' },
					{ en: 'It stays about the same: tanh keeps it', zh: '基本不变：tanh 会保持它' },
					{ en: 'It grows: the small numbers add up', zh: '变大：小数会越加越多' }
				],
				answer: 0,
				explain: {
					en: 'Each weighted sum adds 64 inputs times weights of spread 0.01, which multiplies the spread by about 0.01 × √64 = 0.08; near 0, tanh is almost a straight line and keeps it. Ten layers multiply the spread by about 0.08¹⁰, roughly 10⁻¹¹.',
					zh: '每个加权和把 64 个输入乘以标准差为 0.01 的权重再相加，这会让标准差大约乘以 0.01 × √64 = 0.08；在 0 附近 tanh 几乎是一条直线，会保持这个值。十层下来，标准差大约乘以 0.08¹⁰，约为 10⁻¹¹。'
				}
			},
			narration: {
				en: 'Same network, but now every weight starts with spread σ = {std}, about {k} times smaller than Xavier\'s {xavier}. The input still has spread {s0}. Answer first, then compare layer 1 with layer {top} in the stage.',
				zh: '还是同一个网络，但现在每个权重一开始的标准差是 σ = {std}，大约是 Xavier 取值 {xavier} 的 {k} 分之一。输入的标准差仍然是 {s0}。先回答，再在舞台里比较第 1 层和第 {top} 层。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { top: DEPTH, std: stdText(m.std), xavier: stdText(INIT_STD.xavier), k: INIT_STD.xavier / m.std, s0: m.s0 };
				}
			}
		},
		{
			id: 'small',
			title: { en: 'The numbers fade, and so do the gradients', zh: '数值逐层衰减，梯度也一样' },
			source: 'small',
			at: { phase: 'backward' },
			view: 'init-normalization/layers',
			provenance: 'live',
			terms: ['vanishing-gradient'],
			narration: {
				en: 'Layer 1\'s activations have spread {s1}, layer {top}\'s only {s10}: each layer multiplied the spread by about {f}. The gradient fades the same way on its way back, from {g10} at layer {top} to {g1} at layer 1. That is a vanishing gradient: the first layers would barely move.',
				zh: '第 1 层激活值的标准差是 {s1}，第 {top} 层只有 {s10}：每经过一层，标准差大约乘以 {f}。梯度在反向传回时也同样衰减，从第 {top} 层的 {g10} 降到第 1 层的 {g1}。这就是梯度消失：前面几层几乎不会动。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { top: DEPTH, s1: m.s1, s10: m.s10, f: m.f, g10: m.g10, g1: m.g1 };
				}
			},
			look: {
				en: 'The spread bars shrink down the activation panel and up the gradient panel, about one notch per layer.',
				zh: '标准差横条在激活值面板里自上而下缩短，在梯度面板里自下而上缩短，每层大约短一截。'
			}
		},
		{
			id: 'large',
			title: { en: 'Weights that are too large', zh: '权重太大' },
			source: 'large',
			at: { phase: 'backward' },
			view: 'init-normalization/layers',
			provenance: 'live',
			terms: ['saturation', 'exploding-gradient'],
			narration: {
				en: 'With σ = {std}, layer 1\'s weighted sums have spread {zs1}, far out where tanh is flat: {sat|pct} of layer {top}\'s outputs lie beyond ±{satAt}, so the histograms pile up at the ends. Saturated units pass back almost nothing, yet the large weights multiply what does pass. The gradient grows from {g10} at layer {top} to {g1} at layer 1: an exploding gradient.',
				zh: 'σ = {std} 时，第 1 层加权和的标准差是 {zs1}，远远落在 tanh 平坦的区域：第 {top} 层有 {sat|pct} 的输出超出 ±{satAt}，所以直方图堆在两端。饱和的单元几乎传不回梯度，但很大的权重会把能传回的那部分放大。梯度从第 {top} 层的 {g10} 增长到第 1 层的 {g1}：这就是梯度爆炸。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { top: DEPTH, std: stdText(m.std), zs1: m.zs1, sat: m.sat, satAt: fmtSetting(SAT), g10: m.g10, g1: m.g1 };
				}
			},
			look: {
				en: 'The activation histograms are two spikes at −1 and 1; the gradient bars grow toward layer 1.',
				zh: '激活值直方图变成 −1 和 1 处的两根尖峰；梯度横条越靠近第 1 层越长。'
			}
		},
		{
			id: 'relu',
			title: { en: 'ReLU needs He initialization', zh: 'ReLU 需要 He 初始化' },
			source: 'relus',
			at: 'last',
			view: 'init-normalization/spreads',
			provenance: 'live',
			terms: ['relu', 'xavier-he-init'],
			narration: {
				en: 'Now ReLU, which turns every negative weighted sum into 0. With Xavier\'s σ = {sx}, the spread falls from {x1} at layer 1 to {x10} at layer {top}: ReLU drops about half of each layer\'s variance. He initialization doubles the weights\' variance to make up, σ = √(2/{w}) = {sh}, and the spread holds: {h1} at layer 1, {h10} at layer {top}.',
				zh: '现在换成 ReLU，它把每个负的加权和变成 0。用 Xavier 的 σ = {sx} 时，标准差从第 1 层的 {x1} 降到第 {top} 层的 {x10}：ReLU 让每层丢掉大约一半的方差。He 初始化把权重的方差加倍来补偿，σ = √(2/{w}) = {sh}，标准差就稳住了：第 1 层 {h1}，第 {top} 层 {h10}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { top: DEPTH, w: WIDTH, sx: stdText(m.xavierStd), x1: m.xavier1, x10: m.xavier10, sh: stdText(m.heStd), h1: m.he1, h10: m.he10 };
				}
			},
			look: {
				en: 'Each line is one initialization; the spreads are on a logarithmic axis. Zero is all 0 and has no line.',
				zh: '每条线是一种初始化，标准差用对数刻度表示。全零初始化的值都是 0，所以没有线。'
			}
		},
		{
			id: 'batchnorm',
			title: { en: 'BatchNorm: one unit over the batch', zh: 'BatchNorm：一个单元，跨整个批次' },
			source: 'large',
			at: { phase: 'forward' },
			view: { key: 'init-normalization/axes', props: () => ({ mode: 'batch' }) },
			provenance: 'live',
			terms: ['batch-norm', 'batch'],
			narration: {
				en: 'Normalization fixes the scale directly instead of relying on σ. In the stage are {nb} examples × {nf} units of layer 1\'s weighted sums at σ = {std}. BatchNorm works on one unit, a column, at a time, over the examples of the batch: unit {c} (counting from 0) has mean {m} and spread {sd}, so its {x} becomes ({x} − {mp}) / {sd} ≈ {y}. Every column comes out with mean 0 and spread 1.',
				zh: '归一化直接修正数值的尺度，而不是指望选对 σ。舞台里是 σ = {std} 时第 1 层加权和的 {nb} 个样例 × {nf} 个单元。BatchNorm 一次处理一个单元（一列），跨批次里的所有样例：单元 {c}（从 0 开始数）的均值为 {m}、标准差为 {sd}，所以它的 {x} 变成 ({x} − {mp}) / {sd} ≈ {y}。每一列出来都是均值 0、标准差 1。',
				vars: (_rt, at) => {
					const k = cell(at, 'batch');
					return { nb: NB, nf: NF, std: stdText(metricsOf(at).std), c: k.c, m: k.m, mp: paren(k.m), sd: k.sd, x: k.x, y: k.y };
				}
			},
			look: {
				en: 'The outlined column is the one worked out; its mean and spread are in the rows under the table.',
				zh: '加框的那一列就是正在计算的列；它的均值和标准差在表格下面两行。'
			}
		},
		{
			id: 'layernorm',
			title: { en: 'LayerNorm: one example over its units', zh: 'LayerNorm：一个样例，跨它的所有单元' },
			source: 'large',
			at: { phase: 'forward' },
			view: { key: 'init-normalization/axes', props: () => ({ mode: 'layer' }) },
			provenance: 'live',
			terms: ['layer-norm'],
			narration: {
				en: 'LayerNorm takes the same tensor the other way: one example, a row, at a time, over its {nf} units. Example {r} (counting from 0) has mean {m} and spread {sd}, so the same {x} becomes {y}, not {other} as under BatchNorm. It never looks at the other examples. TinyGPT puts a LayerNorm before every attention and every MLP.',
				zh: 'LayerNorm 按另一个方向处理同一个张量：一次处理一个样例（一行），跨它的 {nf} 个单元。样例 {r}（从 0 开始数）的均值为 {m}、标准差为 {sd}，所以同一个 {x} 变成 {y}，而不是 BatchNorm 下的 {other}。它从不看其他样例。TinyGPT 在每个注意力和每个 MLP 之前都放了一个 LayerNorm。',
				vars: (_rt, at) => {
					const k = cell(at, 'layer');
					return { nf: NF, r: k.r, m: k.m, sd: k.sd, x: k.x, y: k.y, other: k.other };
				}
			},
			look: {
				en: 'The outlined row is the example worked out; its mean and spread are in the two columns at the right of the table.',
				zh: '加框的那一行就是正在计算的样例；它的均值和标准差在表格右边的两列。'
			}
		},
		{
			id: 'task',
			title: { en: 'Rescue a bad start', zh: '挽救一个糟糕的起点' },
			source: 'mine',
			at: { phase: 'backward' },
			view: 'init-normalization/layers',
			provenance: 'live',
			task: {
				prompt: {
					en: 'In the side panel, choose ReLU and the too-large start (σ = 1), then switch on BatchNorm or LayerNorm.',
					zh: '在侧栏里选 ReLU 和过大的起点（σ = 1），然后打开 BatchNorm 或 LayerNorm。'
				},
				done: (rt) => {
					const s = setupOf(rt.input);
					return s.act === 'relu' && s.std >= INIT_STD.large && s.norm !== 'none';
				},
				hint: {
					en: 'First look at ReLU with σ = 1 and no normalization: the spread grows at every layer. Then try both normalizations.',
					zh: '先看 ReLU、σ = 1、不归一化的情形：标准差每层都在变大。然后两种归一化都试一试。'
				},
				waiting: { en: 'Not yet: ReLU, σ = 1 and a normalization are all needed.', zh: '还没有：需要同时选 ReLU、σ = 1 和一种归一化。' },
				doneText: { en: 'Done: the same bad start, kept in range by normalization.', zh: '完成：同样糟糕的起点，被归一化稳在了合适的范围里。' }
			},
			narration: {
				en: 'Normalizing every layer\'s weighted sums makes any start other than zero matter much less. Your network uses {act}, σ = {std} and {norm}. Its activations\' spread goes from {s1} at layer 1 to {s10} at layer {top}, and the gradient from {g10} at layer {top} to {g1} at layer 1.',
				zh: '把每一层的加权和都做归一化，只要不是全零，从哪个起点出发就不那么要紧了。你的网络用的是 {act}、σ = {std}、{norm}。它的激活值标准差从第 1 层的 {s1} 变到第 {top} 层的 {s10}，梯度从第 {top} 层的 {g10} 变到第 1 层的 {g1}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { ...setupText(at), top: DEPTH, s1: m.s1, s10: m.s10, g10: m.g10, g1: m.g1 };
				}
			},
			look: {
				en: 'Watch the last layer\'s spread bars in both panels as you switch a normalization on and off.',
				zh: '打开或关闭归一化时，留意两个面板里最后一层的标准差横条。'
			}
		},
		{
			id: 'limits',
			title: { en: 'Where BatchNorm breaks: a batch of one', zh: 'BatchNorm 在哪里失效：只有一个样例的批次' },
			source: 'large',
			at: { phase: 'forward' },
			view: { key: 'init-normalization/axes', props: () => ({ mode: 'one' }) },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['inference'],
			narration: {
				en: 'In training, BatchNorm takes each unit\'s mean and spread from the batch. Give it example {r} (counting from 0) alone and each unit\'s mean is the number itself, so its {x} becomes ({x} − {xp}) / √(0 + ε) = {y}: every number of the example is erased. At inference it uses averages saved during training instead. LayerNorm gives {other} with or without other examples, in training and at inference alike: one reason language models such as TinyGPT use it.',
				zh: '训练时，BatchNorm 从批次里算出每个单元的均值和标准差。只给它样例 {r}（从 0 开始数），每个单元的均值就是这个数本身，所以它的 {x} 变成 ({x} − {xp}) / √(0 + ε) = {y}：这个样例的每个数都被抹掉了。推理时它改用训练中保存下来的平均值。LayerNorm 不管有没有其他样例都给出 {other}，训练和推理时都一样：这是 TinyGPT 这类语言模型选用它的原因之一。',
				vars: (_rt, at) => {
					const k = cell(at, 'one');
					return { r: k.r, x: k.x, xp: paren(k.x), y: k.y, other: k.other };
				}
			},
			look: {
				en: 'In the table “BatchNorm on a batch of one”, the mean row repeats the example and every spread is 0.',
				zh: '在“批次只有一个样例时的 BatchNorm”表里，均值一行和样例本身完全相同，每个标准差都是 0。'
			}
		}
	]
});

export default lesson;
