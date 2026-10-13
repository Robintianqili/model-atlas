// residual-connections: the guided lesson (ten steps; the concept page's primary path). Every number is live: the grads
// sources run one forward and backward pass of the kit's autograd at the starting weights, race and depths train the
// networks with plain gradient descent; TinyGPT's counts come from facts.json.
//   hook: deeper -> predict | toy: vanish -> factor -> skip -> flat -> task | real: race | compare: depths | limits
import type { At, I18n, TmplVal } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { num } from '../../atlas/lesson/say.ts';
import { DEPTHS, MAX_DEPTH, MOONS, PLAIN, SKIPS, WIDTH, metricsOf, onOff, rate, setupOf, times } from './data.ts';
import { STARTS } from './adapters/setup/adapter.ts';

const words = (v: I18n) => v as unknown as TmplVal;
/** The frame metrics of another frame of the same tape (0 = the first). */
const frameMetrics = (at: At, i: number): Record<string, number> => at.tape?.frame(i).metrics ?? {};

const bars = (props: Record<string, unknown>) => ({ key: 'residual-connections/bars', props: () => props });

const lesson = defineLesson({
	title: { en: 'Residual connections, step by step', zh: '残差连接，分步课程' },
	summary: {
		en: 'Watch a 20-layer network fail to learn, find where its gradient vanishes, follow the gradient through one skip connection with the real numbers, then train both networks and change the depth.',
		zh: '看一个 20 层的网络学不动，找到它的梯度在哪里消失；用真实的数值跟着梯度穿过一个跳跃连接；然后训练两个网络，再改变深度。'
	},
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'deeper',
			title: { en: 'More layers, worse network', zh: '层更多，网络反而更差' },
			source: 'depths',
			at: 'last',
			view: bars({ mode: 'depths', plainOnly: true }),
			provenance: 'live',
			terms: ['mlp', 'layer', 'loss'],
			narration: {
				en: 'This multilayer perceptron (MLP) has {width} units in each hidden layer and sorts {pts} points, laid out as two interlocking half-moons, into two classes. Trained the same way, {steps} updates of gradient descent, the {d0}-layer network reaches a loss of {p0}. The {dN}-layer one is stuck at {pN}: ln 2, the loss of a coin flip. It sorts only {accN} of the points.',
				zh: '这个多层感知机（MLP）每个隐藏层有 {width} 个单元，要把两个月牙形的 {pts} 个点分成两类。用同样的方法训练，即 {steps} 次梯度下降更新，{d0} 层的网络损失降到 {p0}。{dN} 层的网络却卡在 {pN}，也就是 ln 2，抛硬币的损失。它只分对了 {accN} 的点。',
				vars: (_rt, at) => {
					const a = frameMetrics(at, 0);
					const z = metricsOf(at);
					return { width: WIDTH, pts: MOONS.n, steps: z.steps, d0: a.depth, p0: a.lossPlain, dN: z.depth, pN: z.lossPlain, accN: num(z.accPlain, 'pct') };
				}
			},
			look: { en: 'Each row is one depth: its loss after training. The dashed line is the loss of a coin flip.', zh: '每一行是一种深度：训练后的损失。虚线是抛硬币的损失。' }
		},
		{
			id: 'predict',
			title: { en: 'Where does the training signal go?', zh: '训练信号去哪儿了？' },
			source: 'plain',
			at: { phase: 'forward' },
			view: bars({ mode: 'grads' }),
			provenance: 'live',
			terms: ['backpropagation', 'gradient'],
			predict: {
				id: 'residual-connections-vanish',
				question: {
					en: 'Backpropagation now gives each of the layers a gradient: how fast the loss changes as that layer\'s weights change. Next to the last layer\'s, how big is layer 0\'s?',
					zh: '反向传播现在给每一层一个梯度：这一层的权重变化时，损失变化得有多快。和最后一层的相比，第 0 层的梯度有多大？'
				},
				options: [
					{ en: 'About as big', zh: '差不多大' },
					{ en: 'Thousands of times smaller, or even smaller', zh: '只有它的几千分之一，甚至更小' },
					{ en: 'Much bigger, because layer 0 comes first', zh: '大得多，因为第 0 层排在最前面' }
				],
				answer: 1,
				explain: {
					en: 'Much smaller. To reach layer 0 the gradient goes back through every later layer, and each one multiplies it by its weights and by the slope of its tanh. Here those factors are mostly below 1, so the product shrinks with every layer it crosses.',
					zh: '小得多。梯度要倒着穿过后面的每一层才能到达第 0 层，而每一层都要把它乘上自己的权重和 tanh 的斜率。这里这些因子大多小于 1，所以每穿过一层，乘积就再缩小一些。'
				}
			},
			narration: {
				en: 'Before any training, the {depth}-layer network starts from random weights; its loss is {loss}. Backpropagation will now run from the loss back to layer 0 (we count layers from 0, as PyTorch does). Answer first, then go on.',
				zh: '训练之前，这个 {depth} 层的网络从随机权重出发，损失是 {loss}。反向传播马上要从损失一路倒推回第 0 层（层从 0 开始数，和 PyTorch 一样）。先回答，再继续。',
				vars: (_rt, at) => ({ depth: metricsOf(at).depth, loss: metricsOf(at).loss })
			}
		},
		{
			id: 'vanish',
			title: { en: 'The gradient vanishes on its way back', zh: '梯度在倒推的路上消失了' },
			source: 'plain',
			at: { phase: 'backward' },
			view: bars({ mode: 'grads', mark: 'ends' }),
			provenance: 'live',
			terms: ['gradient-norm', 'vanishing-gradient'],
			narration: {
				en: 'Each bar is the size of one layer\'s weight gradient, its gradient norm, on a log axis. The last layer, layer {last}, gets {gLast}; layer 0 gets {g0}, {ratio} times smaller. Gradient descent moves each weight in proportion to its gradient, so the first layers barely move: their gradient has vanished.',
				zh: '每根条是一层权重梯度的大小，也就是梯度范数，画在对数轴上。最后一层（第 {last} 层）得到 {gLast}；第 0 层只得到 {g0}，两者相差 {ratio} 倍。梯度下降让每个权重按它的梯度大小移动，所以最前面几层几乎不动：它们的梯度消失了。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { last: m.depth - 1, gLast: m.gLast, g0: m.g0, ratio: times(m.ratio) };
				}
			},
			look: { en: 'The outlined rows are layer 0 and the last layer. Each labelled tick is a hundred times the one to its left.', zh: '加了外框的两行是第 0 层和最后一层。每个标注的刻度是它左边那个的一百倍。' }
		},
		{
			id: 'factor',
			title: { en: 'One factor per layer, multiplied', zh: '每层一个因子，连乘起来' },
			source: 'plain',
			at: { phase: 'block' },
			view: bars({ mode: 'flow', mark: 'mid' }),
			provenance: 'live',
			terms: ['chain-rule'],
			narration: {
				en: 'To pass back through a layer, the gradient is multiplied by that layer\'s weights and by the slope of its tanh, which is never above 1. Through layer {l} its size goes from {upN} to {downN}, a factor of {f}. From layer {lastH}, the last hidden layer, back to layer 0 these factors multiply to {prod}.',
				zh: '梯度每倒着穿过一层，都要乘上这一层的权重，再乘上它的 tanh 的斜率，而这个斜率从不超过 1。穿过第 {l} 层时，梯度的大小从 {upN} 变成 {downN}，相当于乘了 {f}。从最后一个隐藏层（第 {lastH} 层）倒推回第 0 层，这些因子连乘起来是 {prod}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { l: m.layer, upN: m.upN, downN: m.totalN, f: m.factor, lastH: m.depth - 2, prod: m.h0 / m.hLast };
				}
			},
			look: {
				en: 'Each row is the size of the gradient at one layer\'s output. The outlined pair: layer {l}\'s output, and its input, which is layer {prev}\'s output.',
				zh: '每一行是某一层输出处梯度的大小。加了外框的一对：第 {l} 层的输出，以及它的输入，也就是第 {prev} 层的输出。',
				vars: (_rt, at) => ({ l: metricsOf(at).layer, prev: metricsOf(at).layer - 1 })
			}
		},
		{
			id: 'skip',
			title: { en: 'A skip connection: the gradient passes untouched', zh: '跳跃连接：梯度原样通过' },
			source: 'res',
			at: { phase: 'skip' },
			view: 'zoom.formula',
			provenance: 'live',
			terms: ['residual-connection', 'skip-connection'],
			narration: {
				en: 'A skip connection adds a layer\'s input to its output: h + tanh(W·h + b). Going back, the derivative of that addition is exactly 1, so the arriving gradient passes through the skip unchanged, beside the block\'s own share. Take the largest number arriving at layer {l}\'s output: {up|n4} arrives, the block sends back {block|n4}, the skip sends back {up|n4}, and {total|n4} leaves.',
				zh: '跳跃连接把一层的输入加到它的输出上：h + tanh(W·h + b)。倒推时，这个加法的导数正好是 1，所以到达的梯度原样穿过跳跃连接，和块自己传回的那一份并排走。在到达第 {l} 层输出的数里取最大的一个：到达的是 {up|n4}，块传回 {block|n4}，跳跃连接传回 {up|n4}，离开时是 {total|n4}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { l: m.layer, up: m.up, block: m.block, total: m.total };
				}
			},
			look: {
				en: 'g is the gradient of the loss at a layer\'s output, z = W·h + b the layer\'s weighted sums, and ⊙ multiplies entry by entry. In the first two lines the first term comes through the skip and the second through the block; the second line puts in the live numbers and the third adds them.',
				zh: 'g 是损失在某一层输出处的梯度，z = W·h + b 是这一层的加权和，⊙ 表示逐个元素相乘。前两行里，第一项都来自跳跃连接，第二项都来自块；第二行代入了实时的数值，第三行是它们的和。'
			}
		},
		{
			id: 'flat',
			title: { en: 'With skips, every layer hears the loss', zh: '有了跳跃连接，每一层都能“听见”损失' },
			source: 'res',
			at: { phase: 'backward' },
			view: bars({ mode: 'grads', both: true, mark: 'ends' }),
			provenance: 'live',
			narration: {
				en: 'With a skip on each of the {n} layers from layer 1 to layer {lastH} (layer 0 and the last layer change the size of the vector, so they have none), the same starting weights give every layer a gradient of a similar size: layer 0 gets {g0} and the last layer {gLast}, only {ratio} times as much. Without the skips the gap was {oRatio} times.',
				zh: '从第 1 层到第 {lastH} 层，这 {n} 层每层都加上跳跃连接（第 0 层和最后一层会改变向量的大小，所以没有）。同样的初始权重下，每一层的梯度大小都差不多：第 0 层得到 {g0}，最后一层 {gLast}，两者只相差 {ratio} 倍。没有跳跃连接时，相差 {oRatio} 倍。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { n: m.depth - 2, lastH: m.depth - 2, g0: m.g0, gLast: m.gLast, ratio: times(m.ratio), oRatio: times(m.oRatio) };
				}
			},
			look: { en: 'Each layer has two bars now: without skips and with skips, from the very same starting weights.', zh: '现在每一层有两根条：无跳跃连接和有跳跃连接，起点是完全相同的权重。' }
		},
		{
			id: 'task',
			title: { en: 'Switch the skips yourself', zh: '自己打开跳跃连接' },
			source: 'bars',
			at: { phase: 'backward' },
			view: bars({ mode: 'grads', both: true, mark: 'ends' }),
			provenance: 'live',
			task: {
				prompt: { en: 'Switch the skip connections on and set the network to 20 layers.', zh: '打开跳跃连接，并把网络设成 20 层。' },
				done: (rt) => {
					const s = setupOf(rt.input);
					return s.skip === 1 && s.depth === 20;
				},
				hint: { en: 'Use the switch and the slider in the side panel, or in the toy below the lesson.', zh: '用侧栏里的开关和滑块，或者课程下方玩具实验里的。' },
				waiting: { en: 'Not yet: the skips are off, or the network does not have 20 layers.', zh: '还没有：跳跃连接没打开，或者网络不是 20 层。' },
				doneText: { en: 'Done: twenty layers with skips. Compare the first bars with the plain ones.', zh: '完成：20 层，有跳跃连接。把最前面几根条和无跳跃连接的比一比。' }
			},
			narration: {
				en: 'Your network has {depth} layers, skips {onoff}: layer 0 gets {g0} and the last layer {gLast}, a ratio of {ratio}. With the switch flipped, the same weights would give a ratio of {oRatio}. Change the switch and the number of layers, and watch the first bars.',
				zh: '你的网络有 {depth} 层，跳跃连接{onoff}：第 0 层得到 {g0}，最后一层 {gLast}，相差 {ratio} 倍。把开关反过来，同样的权重会相差 {oRatio} 倍。改变开关和层数，看最前面几根条的变化。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { depth: m.depth, onoff: words(onOff(m.skip)), g0: m.g0, gLast: m.gLast, ratio: times(m.ratio), oRatio: times(m.oRatio) };
				}
			},
			look: { en: 'The outlined rows are layer 0 and the last layer; each row has a bar without skips and a bar with skips.', zh: '加了外框的两行是第 0 层和最后一层；每一行都有一根无跳跃连接的条和一根有跳跃连接的条。' }
		},
		{
			id: 'race',
			title: { en: 'Train them both', zh: '把两个都训练一遍' },
			source: 'race',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metrics: ['loss', 'lossPlain'], log: true, labels: { loss: SKIPS, lossPlain: PLAIN } }) },
			provenance: 'live',
			terms: ['gradient-descent', 'learning-rate'],
			narration: {
				en: 'Now train both {depth}-layer networks from the same starting weights: {n} updates of gradient descent with learning rate {lr}. Without skips the loss stays at {lossPlain} and {accPlain} of the points are right; with skips it falls to {loss} and {acc} are right. Over the run, layer 0\'s weights moved {move} with skips and only {movePlain} without.',
				zh: '现在从同样的初始权重出发训练这两个 {depth} 层网络：用学习率 {lr} 做 {n} 次梯度下降更新。没有跳跃连接时，损失停在 {lossPlain}，分对 {accPlain} 的点；有跳跃连接时，损失降到 {loss}，分对 {acc}。整个训练中，有跳跃连接时第 0 层的权重一共移动了 {move}，没有时只移动了 {movePlain}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { depth: MAX_DEPTH, n: m.step, lr: rate(m.lr), lossPlain: m.lossPlain, accPlain: num(m.accPlain, 'pct'), loss: m.loss, acc: num(m.accuracy, 'pct'), move: m.move, movePlain: m.movePlain };
				}
			},
			look: { en: 'The loss axis is logarithmic: every grid line is ten times the one below.', zh: '损失轴是对数刻度：每条网格线是下面一条的十倍。' }
		},
		{
			id: 'depths',
			title: { en: 'Deeper helps, once the gradient gets through', zh: '梯度传得过去，更深才有用' },
			source: 'depths',
			at: 'last',
			view: bars({ mode: 'depths' }),
			provenance: 'live',
			narration: {
				en: 'The same training at {d0}, {d1}, {d2} and {d3} layers. Without skips, more layers do worse: the final loss goes from {p0} to {p3}. With skips, more layers do better: from {r0} to {r3}. With only {d0} layers there is little to skip, and the two end close: {p0} against {r0}.',
				zh: '同样的训练，分别用 {d0}、{d1}、{d2} 和 {d3} 层。没有跳跃连接时，层越多反而越差：最终损失从 {p0} 变成 {p3}。有跳跃连接时，层越多越好：从 {r0} 降到 {r3}。只有 {d0} 层时没什么可跳过的，两者的结果很接近：{p0} 对 {r0}。',
				vars: (_rt, at) => {
					const f = DEPTHS.map((_, i) => frameMetrics(at, i));
					const z = f[DEPTHS.length - 1];
					return { d0: f[0].depth, d1: f[1].depth, d2: f[2].depth, d3: z.depth, p0: f[0].lossPlain, p3: z.lossPlain, r0: f[0].loss, r3: z.loss };
				}
			},
			look: { en: 'Each row is one depth, with the loss after training without skips and with skips.', zh: '每一行是一种深度，分别给出无跳跃连接和有跳跃连接训练后的损失。' }
		},
		{
			id: 'limits',
			title: { en: 'What skips do not fix', zh: '跳跃连接解决不了的问题' },
			source: 'res',
			at: { phase: 'backward' },
			view: bars({ mode: 'acts', both: true }),
			provenance: 'live',
			kindOf: 'limits',
			terms: ['layer-norm', 'residual-stream'],
			narration: {
				en: 'Skips are not free. Each block adds its output to what passes through, so the signal grows with depth: its typical size is {r0} after layer 0 and {rLast} after layer {lastH}, against {oLast} without skips. Bigger signals give bigger scores, and a starting loss of {loss} instead of {oLoss}. That is why networks with skips usually add normalization too: TinyGPT has {fact:gptAdds} residual additions and {fact:gptNorms} LayerNorms.',
				zh: '跳跃连接不是没有代价的。每个块都把自己的输出加到穿过的信号上，所以信号随深度变大：它的典型大小在第 0 层之后是 {r0}，在第 {lastH} 层之后是 {rLast}，而没有跳跃连接时只有 {oLast}。信号更大，打分也更大，起始损失是 {loss} 而不是 {oLoss}。所以带跳跃连接的网络通常还会加上归一化：TinyGPT 有 {fact:gptAdds} 次残差相加和 {fact:gptNorms} 个 LayerNorm。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { r0: m.rms0, rLast: m.rmsLast, lastH: m.depth - 2, oLast: m.oRmsLast, loss: m.loss, oLoss: m.oLoss };
				}
			},
			look: { en: 'Each row is the typical size (root mean square) of one layer\'s output, without and with skips.', zh: '每一行是某一层输出的典型大小（均方根），分别是无跳跃连接和有跳跃连接。' }
		}
	]
});

export default lesson;
