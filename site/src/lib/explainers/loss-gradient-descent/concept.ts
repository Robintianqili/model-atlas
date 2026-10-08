// loss-gradient-descent: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { At, I18n, StepDef, Tensor, TrainState } from '../../atlas/types/index.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { eta, gradientOf, lineText, maxStableRate, metricsOf, rate, scoreOf } from './data.ts';
import { PARAMS } from './net.ts';


/** The first-layer weight with the largest gradient at an update frame of the real network: [out, in]. */
function biggestGrad(at: At): number[] {
	const g = (at.state as Partial<TrainState> | undefined)?.grads?.['net.0.weight'] as Tensor | undefined;
	if (!g) return [0, 0];
	let k = 0;
	for (let i = 1; i < g.data.length; i++) if (Math.abs(g.data[i]) > Math.abs(g.data[k])) k = i;
	return [Math.floor(k / g.shape[1]), k % g.shape[1]];
}

export const hook: I18n = {
	en: 'Most networks on this site learned the same way. Someone wrote down one number that says how wrong the network is, the loss, and then nudged its weights again and again to make that number smaller. Here the network is a single neuron with two dials, so you can do it yourself before the computer does.',
	zh: '这个网站上的大多数网络都是这样学会的：有人写下一个数来表示网络错得有多厉害，也就是损失，然后一次又一次地微调权重，让这个数变小。这里的网络只是一个有两个旋钮的神经元，所以你可以先自己动手，再看计算机怎么做。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Turn the dials, then let gradient descent turn them', zh: '先自己转旋钮，再让梯度下降来转' },
	source: 'gd',
	at: { phase: 'loss', step: 0 },
	view: { key: 'loss-gradient-descent/fit', props: () => ({ mark: 'worst', stepButton: true }) },
	provenance: 'live',
	narration: {
		en: 'Your line y = {line} scores {loss}. One gradient-descent step with η = {lr} would move the dials to w = {w1} and b = {b1}, for a score of {loss1}. Press the button to take it, again and again, or move a dial and start over.',
		zh: '你的直线 y = {line} 得分 {loss}。用 η = {lr} 走一步梯度下降，会把旋钮移到 w = {w1}、b = {b1}，分数变成 {loss1}。按下按钮走这一步，可以一直按，也可以动一下旋钮重新开始。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			const g = gradientOf(m.w, m.b);
			const w1 = m.w - m.lr * g.gw;
			const b1 = m.b - m.lr * g.gb;
			return { line: lineText(m.w, m.b), loss: m.loss, lr: rate(m.lr), w1, b1, loss1: scoreOf(w1, b1) };
		}
	},
	look: { en: 'The numbers under the dials and the bars change together: the score is the mean of the bars.', zh: '旋钮下的数字和柱子一起变化：分数就是柱子的平均值。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'The same update on a network with many weights', zh: '在有很多权重的网络上做同样的更新' },
	source: 'realStep',
	at: { phase: 'update', step: 0 },
	view: { key: 'train-step', props: (_rt, at) => ({ key: 'net.0.weight', index: biggestGrad(at) }) },
	focus: (_rt, at) => [{ field: 'params', key: 'net.0.weight', index: biggestGrad(at), role: 'param' }],
	provenance: 'live',
	terms: ['gradient-descent'],
	narration: {
		en: 'A network with {params} weights learns by the very same rule. In its first update, the weight with the steepest slope goes from {old} to {old} − {lr} × ({g}) = {new}. Every other weight takes its own step at the same moment, each against its own slope.',
		zh: '一个有 {params} 个权重的网络也按完全相同的规则学习。在第一次更新中，斜率最陡的那个权重从 {old} 变成 {old} − {lr} × ({g}) = {new}。同一时刻，其他每个权重也都逆着自己的斜率走一步。',
		vars: (_rt, at) => {
			const u = updateOf(at, 'net.0.weight', biggestGrad(at));
			return { params: PARAMS, old: u.old, g: u.g, new: u.new, lr: rate(u.lr) };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Three learning rates from your dials', zh: '从你的旋钮出发的三种学习率' },
	source: 'rates',
	at: 'last',
	view: {
		key: 'loss-curve',
		props: (_rt, at) => {
			const m = metricsOf(at);
			// the order picks the colors: η = 0.1 the accent, η = 0.01 orange, η = 0.7 green (going up)
			return { metrics: ['loss', 'lossSlow', 'lossBig'], log: true, labels: { lossSlow: eta(m.lrSlow), loss: eta(m.lr), lossBig: eta(m.lrBig) } };
		}
	},
	provenance: 'live',
	narration: {
		en: 'The same start, three step sizes. After {n} updates η = {lrSlow} has only reached {slow}, while η = {lr} is at {good}. η = {lrBig} overshoots and climbs to {big}: on this bowl every rate above {maxRate} does.',
		zh: '同样的起点，三种步长。更新 {n} 次之后，η = {lrSlow} 只降到 {slow}，而 η = {lr} 已经到 {good}。η = {lrBig} 一再越过谷底，分数爬升到 {big}：在这个碗形地形上，任何大于 {maxRate} 的学习率都会这样。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { n: m.step, lr: rate(m.lr), good: m.loss, lrSlow: rate(m.lrSlow), slow: m.lossSlow, lrBig: rate(m.lrBig), big: m.lossBig, maxRate: maxStableRate() };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'A bigger learning rate always learns faster.', zh: '学习率越大，学得越快。' },
		truth: {
			en: 'Only up to a point. Above it (two thirds, for this bowl) every step overshoots the bottom and the score grows; far below it the dials barely move.',
			zh: '只在一定范围内成立。超过某个值（对这个碗形地形是三分之二），每一步都会越过谷底，分数反而变大；远小于它时，旋钮几乎不动。'
		},
		step: 'too-big'
	},
	{
		claim: { en: 'Training is finished when the score reaches 0.', zh: '分数降到 0，训练才算完成。' },
		truth: {
			en: 'These eight points do not lie on any straight line, so even the best line scores above 0. Gradient descent finds the bottom of the bowl, not a perfect fit.',
			zh: '这八个点不在任何一条直线上，所以即使最好的直线分数也大于 0。梯度下降找到的是碗的最低点，而不是完美的拟合。'
		},
		step: 'descend'
	}
];

export const further = [
	{ label: { en: 'Dive into Deep Learning: gradient descent (free online book)', zh: '《动手学深度学习》：梯度下降（免费在线书）' }, href: 'https://d2l.ai/chapter_optimization/gd.html' },
	{ label: { en: 'micrograd: a tiny autograd engine in Python (Andrej Karpathy)', zh: 'micrograd：一个用 Python 写的很小的自动求导引擎（Andrej Karpathy）' }, href: 'https://github.com/karpathy/micrograd' }
];
