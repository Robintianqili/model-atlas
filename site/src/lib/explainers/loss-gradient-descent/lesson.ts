// loss-gradient-descent: the guided lesson (nine steps; the concept page's primary path). Every number is live: the
// gd sources run the kit's autograd from the learner's dials; the real sources train a 2-8-8-2 network.
//   points -> score -> beat (task) -> predict -> gradient -> update -> descend (play) -> too-big (breakIt) -> limits
import type { At } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { XS, YS, TARGET, bestLine, dialsOf, eta, lineText, maxStableRate, metricsOf, rate, scoreOf, worstPoint } from './data.ts';
import { MOONS, PARAMS } from './net.ts';
import { STARTS } from './adapters/dials/adapter.ts';

const dialsAt = (at: At) => {
	const s = (at.state ?? {}) as { w?: number; b?: number };
	return { w: s.w ?? NaN, b: s.b ?? NaN };
};
/** The worst-missed point of the dials at this frame: its x, answer, prediction, error and squared error. */
function worst(at: At) {
	const { w, b } = dialsAt(at);
	if (!Number.isFinite(w)) return { x: NaN, y: NaN, p: NaN, e: NaN, e2: NaN, i: NaN };
	const { i, e } = worstPoint(w, b);
	return { i, x: XS[i], y: YS[i], p: w * XS[i] + b, e, e2: e * e };
}
const best = bestLine();
const lesson = defineLesson({
	title: { en: 'Loss and gradient descent, step by step', zh: '损失与梯度下降，分步课程' },
	summary: {
		en: 'Score a line through eight points, beat the score by hand, then watch gradient descent turn the dials with the real numbers, and break it with a step that is too big.',
		zh: '给穿过八个点的直线打分，亲手打败这个分数，再看梯度下降用真实的数值转动旋钮，最后用过大的步长把它弄坏。'
	},
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'points',
			title: { en: 'Eight points and your line', zh: '八个点和你的直线' },
			source: 'gd',
			at: { phase: 'loss', step: 0 },
			view: { key: 'loss-gradient-descent/fit', props: () => ({ squares: false, mark: 'worst' }) },
			provenance: 'live',
			terms: ['neuron', 'weight', 'bias'],
			narration: {
				en: 'Each dot is one example: an input x and the answer y we want. A single neuron answers w·x + b, so your two dials draw the line y = {line}. At x = {x} it says {p}, but the answer is {y}: an error of {e}.',
				zh: '每个点是一个样例：输入 x 和我们想要的答案 y。一个神经元的回答是 w·x + b，所以你的两个旋钮画出直线 y = {line}。在 x = {x} 处它说 {p}，而答案是 {y}：误差是 {e}。',
				vars: (_rt, at) => {
					const d = dialsAt(at);
					return { line: lineText(d.w, d.b), ...worst(at) };
				}
			},
			look: { en: 'The ringed point is the one your line misses most; the dashed segments are the errors.', zh: '带圈的点是你的直线偏得最远的点；虚线段就是误差。' }
		},
		{
			id: 'score',
			title: { en: 'One number for how wrong: the error score', zh: '用一个数表示错得多厉害：误差分数' },
			source: 'gd',
			at: { phase: 'loss', step: 0 },
			view: { key: 'loss-gradient-descent/fit', props: () => ({ mark: 'worst' }) },
			provenance: 'live',
			terms: ['loss', 'mse'],
			narration: {
				en: 'Square each error, so misses above and below the line cannot cancel and big misses count most: the worst error, {e}, becomes {e2}. The error score is the mean of the eight squares: {loss}. A line through every point would score 0.',
				zh: '把每个误差平方，这样线上方和下方的偏差不会互相抵消，大的偏差也最重要：最大的误差 {e} 变成 {e2}。误差分数是这八个平方的平均值：{loss}。穿过每个点的直线得分是 0。',
				vars: (_rt, at) => {
					const k = worst(at);
					return { e: k.e, e2: k.e2, loss: metricsOf(at).loss };
				}
			},
			look: { en: 'Each bar is one point\'s squared error; the dashed line is their mean, the score.', zh: '每根柱子是一个点的误差平方；虚线是它们的平均值，也就是分数。' }
		},
		{
			id: 'beat',
			title: { en: 'Beat the score by hand', zh: '亲手打败这个分数' },
			source: 'gd',
			at: { phase: 'loss', step: 0 },
			view: { key: 'loss-gradient-descent/fit', props: () => ({ mark: 'worst' }) },
			provenance: 'live',
			task: {
				prompt: { en: 'Turn the two dials until the error score is below 0.1.', zh: '转动两个旋钮，直到误差分数低于 0.1。' },
				done: (rt) => {
					const d = dialsOf(rt.input);
					return scoreOf(d.w, d.b) < TARGET;
				},
				hint: {
					en: 'Tilt the line with w until it follows the dots, then slide it up or down with b. Watch the tallest bar.',
					zh: '先用 w 把直线转到顺着点的方向，再用 b 把它上下移动。盯住最高的那根柱子。'
				},
				waiting: { en: 'Not yet: the score is still at or above the goal.', zh: '还没有：分数仍然不低于目标。' },
				doneText: { en: 'Done: you beat the goal by hand.', zh: '完成：你亲手达到了目标。' }
			},
			narration: {
				en: 'Your line scores {loss} now. Turn the dials, in the side panel or in the toy below the lesson, and watch the bars: the score is lowest when the line runs through the middle of the dots. The best line scores {best}.',
				zh: '你的直线现在得 {loss} 分。在侧栏或课程下方的玩具实验里转动旋钮，看柱子的变化：直线从点的中间穿过时分数最低。最好的直线得 {best} 分。',
				vars: (_rt, at) => ({ loss: metricsOf(at).loss, best: best.score })
			}
		},
		{
			id: 'predict',
			title: { en: 'Let gradient descent turn the dials', zh: '让梯度下降来转动旋钮' },
			source: 'gd',
			at: { phase: 'update', step: 0 },
			view: { key: 'loss-gradient-descent/fit', props: () => ({ mark: 'worst' }) },
			provenance: 'live',
			predict: {
				id: 'loss-gradient-descent-one-step',
				question: { en: 'Gradient descent turns both dials once, a little, in the direction that lowers the score fastest. What happens to the score?', zh: '梯度下降把两个旋钮各转一次，转一点点，方向是让分数下降最快的方向。分数会怎样？' },
				options: [
					{ en: 'It drops, but not to the lowest possible score', zh: '下降了，但没有降到最低可能的分数' },
					{ en: 'It drops straight to the lowest possible score', zh: '直接降到最低可能的分数' },
					{ en: 'It could just as well go up', zh: '也可能上升' }
				],
				answer: 0,
				explain: {
					en: 'The slope only describes the ground right where the dials are. A small step against it goes downhill, so the score falls; the bottom is further away than one small step, so it takes many.',
					zh: '斜率只描述旋钮当前位置的地面。逆着它走一小步就是下坡，所以分数会下降；但最低点比一小步远，所以需要走很多步。'
				}
			},
			narration: {
				en: 'Your dials score {loss}. Gradient descent now changes both of them at once, by itself. Answer first, then look at the new line and its bars.',
				zh: '你的旋钮得分是 {loss}。梯度下降现在自己同时改变这两个旋钮。先回答，再看新的直线和它的柱子。',
				vars: (_rt, at) => ({ loss: metricsOf(at).loss })
			}
		},
		{
			id: 'gradient',
			title: { en: 'The slope under each dial: the gradient', zh: '每个旋钮方向上的斜率：梯度' },
			source: 'gd',
			at: { phase: 'grad', step: 0 },
			view: { key: 'loss-gradient-descent/bowl', props: () => ({ arrow: true }) },
			provenance: 'live',
			terms: ['gradient', 'backpropagation'],
			narration: {
				en: 'Picture the score for every pair of dials: a landscape whose lowest point is the best line. The gradient is its slope where your dials are: {gw} along w and {gb} along b. Automatic differentiation (autograd) in your browser got both by backpropagation: the chain rule, run backwards through the eight squared errors.',
				zh: '想象每一对旋钮取值的分数：一片地形，最低点就是最好的直线。梯度是你的旋钮所在位置的坡度：w 方向 {gw}，b 方向 {gb}。你的浏览器里的自动求导（autograd）用反向传播得到了这两个数：沿着八个误差平方，把链式法则倒着算一遍。',
				vars: (_rt, at) => ({ gw: metricsOf(at).gw, gb: metricsOf(at).gb })
			},
			look: { en: 'The ring is your dials, the cross the lowest score. The arrow points against the slope: downhill.', zh: '圆环是你的旋钮，叉号是最低分。箭头逆着斜率，指向下坡。' }
		},
		{
			id: 'update',
			title: { en: 'One step downhill', zh: '向下走一步' },
			source: 'gd',
			at: { phase: 'update', step: 0 },
			view: 'zoom.formula',
			provenance: 'live',
			terms: ['learning-rate', 'gradient-descent'],
			narration: {
				en: 'Each dial moves against its own slope, scaled by the learning rate η = {lr}: w goes from {w} to {w1}, and b from {b} to {b1}. The score falls from {loss|n4} to {loss1|n4}, by {drop}.',
				zh: '每个旋钮都逆着自己的斜率移动，移动量按学习率 η = {lr} 缩放：w 从 {w} 变成 {w1}，b 从 {b} 变成 {b1}。分数从 {loss|n4} 降到 {loss1|n4}，下降了 {drop}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { lr: rate(m.lr), w: m.w, w1: m.w1, b: m.b, b1: m.b1, loss: m.loss, loss1: m.loss1, drop: m.loss - m.loss1 };
				}
			}
		},
		{
			id: 'descend',
			title: { en: 'Repeat: many small steps', zh: '重复：很多小步' },
			source: 'run',
			at: 'last',
			view: 'loss-gradient-descent/bowl',
			play: { from: 'first', to: 'last', interval: 260, unit: { en: 'update', zh: '次更新' } },
			provenance: 'live',
			narration: {
				en: 'Gradient descent repeats the same three moves: score, slope, step. After {n} updates the dials are w = {w} and b = {b}, and the score is {loss}, only {gap} above the lowest possible.',
				zh: '梯度下降不断重复同样的三步：打分、求斜率、走一步。更新 {n} 次之后，旋钮是 w = {w}、b = {b}，分数是 {loss}，只比最低可能的分数高 {gap}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { n: m.step, w: m.w, b: m.b, loss: m.loss, gap: m.loss - best.score };
				}
			},
			look: { en: 'The steps shrink near the bottom: the ground flattens, so the slope, and each step, gets smaller.', zh: '靠近底部时步子变小：地面变平，斜率变小，每一步也随之变小。' }
		},
		{
			id: 'too-big',
			title: { en: 'A step that is too big', zh: '步子太大' },
			source: 'rates',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: (_rt, at) => ({ metrics: ['loss', 'lossBig'], log: true, labels: { loss: eta(metricsOf(at).lr), lossBig: eta(metricsOf(at).lrBig) } })
			},
			provenance: 'live',
			kindOf: 'breakIt',
			narration: {
				en: 'With η = {lrBig} each step overshoots the bottom and lands higher on the other side of the valley. After {n} updates the score is {lossBig}, up from {start}. On this bowl any learning rate above {maxRate} fails this way.',
				zh: '当 η = {lrBig} 时，每一步都越过谷底，落在山谷另一侧更高的地方。更新 {n} 次之后，分数是 {lossBig}，而开始时是 {start}。在这个碗形地形上，任何大于 {maxRate} 的学习率都会这样失败。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { lrBig: rate(m.lrBig), n: m.step, lossBig: m.lossBig, start: at.tape?.frame(0).metrics?.lossBig ?? NaN, maxRate: maxStableRate() };
				}
			},
			look: { en: 'The score axis is logarithmic: every grid line is ten times the one below.', zh: '分数轴是对数刻度：每条网格线是下面一条的十倍。' }
		},
		{
			id: 'limits',
			title: { en: 'A real network: the same rule, a rougher landscape', zh: '真实的网络：同样的规则，更崎岖的地形' },
			source: 'real',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metrics: ['loss', 'loss2'], labels: { loss: { en: 'start 1', zh: '起点 1' }, loss2: { en: 'start 2', zh: '起点 2' } } }) },
			provenance: 'live',
			kindOf: 'limits',
			narration: {
				en: 'The same three moves train a network with {params} weights to sort {pts} points: after {n} updates its score is {loss1}. Its landscape has {params} dials and many valleys, not one bowl. From other random weights it ends at a similar score, {loss2}, with different weights: its first weight is {wb}, not {wa}. Gradient descent finds a bottom, not the bottom.',
				zh: '同样的三步训练一个有 {params} 个权重的网络去给 {pts} 个点分类：更新 {n} 次之后，它的分数是 {loss1}。它的地形有 {params} 个旋钮，有很多山谷，而不是一个碗。从另一组随机权重出发，它停在相近的分数 {loss2}，但权重不同：它的第一个权重是 {wb}，而不是 {wa}。梯度下降找到的是某一个谷底，而不一定是最低的那个。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { params: PARAMS, pts: MOONS.n, n: m.step, loss1: m.loss, loss2: m.loss2, wa: m.w0, wb: m.w0b };
				}
			}
		}
	]
});

export default lesson;
