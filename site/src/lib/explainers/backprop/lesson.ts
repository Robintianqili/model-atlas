// backprop: the guided lesson (ten steps; the concept page's primary path). Every number is live: the graph sources run
// the kit's autograd on the learner's leaves (or on the saturated ones), the net source on the 2-2-1 network.
//   forward -> local -> predict -> twice -> chain (play) -> task -> check -> net -> cost -> limits
import type { At } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { TASK_G, TASK_L, leavesOf, nodeGrad, nodeVal, par } from './data.ts';
import { leafGrads, lossOf } from './process.ts';
import { STARTS } from './adapters/leaves/adapter.ts';

const m = (at: At): Record<string, number> => at.frame?.metrics ?? {};
/** The state of a source's frame matching a cursor (the last edge, when a step narrates the end of a run it plays). */
const stateAt = (at: At, c: { phase: string; step?: number }): unknown => (at.tape && at.tape.find(c) >= 0 ? at.tape.state(at.tape.find(c)) : undefined);
const LAST_EDGE = { phase: 'edge', step: 8 };
/** The values the edge steps narrate, from one state. */
const graphVars = (s: unknown) => ({
	...Object.fromEntries(['x', 'z', 'L'].map((id) => [id, nodeVal(s, id)])),
	...Object.fromEntries(['a', 'z', 'w'].map((id) => [`g${id}`, nodeGrad(s, id)])),
	da: 1 - nodeVal(s, 'a') ** 2,
	gp: par(nodeGrad(s, 'p'))
});

const lesson = defineLesson({
	title: { en: 'Backpropagation, one edge at a time', zh: '反向传播，一次一条边' },
	summary: {
		en: 'Run a tiny computation graph forward, walk its backward pass one chain-rule edge at a time (a node used twice adds both shares), check it numerically, then watch the same rule fill in the nine gradients of a 2-2-1 network and fade through a flat tanh.',
		zh: '把一张很小的计算图正向算一遍，再沿链式法则一条边一条边地走反向传播（被用两次的节点要把两份加起来），用数值方法检验，然后看同样的规则怎样算出 2-2-1 网络的九个梯度，又怎样在平坦的 tanh 处衰减。'
	},
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'forward',
			title: { en: 'A formula as a graph: the forward pass', zh: '把公式画成图：前向传播' },
			source: 'graph',
			at: { phase: 'forward' },
			view: 'backprop/graph',
			provenance: 'live',
			terms: ['computation-graph', 'forward-pass', 'neuron', 'loss', 'gradient'],
			narration: {
				en: 'One neuron and its squared error, L = (tanh(w·x + b) − y)², written as five small operations on four leaves, the numbers you set: x, w, b and y. Each node computes one number from the arrows coming in: p = w·x = {p}, z = p + b = {z}, a = tanh(z) = {a}, e = a − y = {e}, and the loss L = e·e = {L}. This is the forward pass.',
				zh: '一个神经元和它的平方误差 L = (tanh(w·x + b) − y)²，可以写成五个小运算，它们从四个叶子节点出发，也就是你设定的四个数 x、w、b 和 y。每个节点用箭头送进来的数算出一个新数：p = w·x = {p}，z = p + b = {z}，a = tanh(z) = {a}，e = a − y = {e}，损失 L = e·e = {L}。这就是前向传播。',
				vars: (_rt, at) => Object.fromEntries(['p', 'z', 'a', 'e', 'L'].map((id) => [id, nodeVal(at.state, id)]))
			},
			look: { en: 'Each node\'s "grad" line, its gradient dL/d(node), still shows "?": the backward pass has not started.', zh: '每个节点的“梯度”一行，也就是 dL/d(该节点)，都还显示“?”：反向传播还没有开始。' }
		},
		{
			id: 'local',
			title: { en: 'Each operation knows its own slope', zh: '每个运算都知道自己的斜率' },
			source: 'graph',
			at: { phase: 'forward' },
			view: { key: 'backprop/graph', props: () => ({ locals: true }) },
			provenance: 'live',
			narration: {
				en: 'Each operation only has to know how its output moves when one input moves a little: its local derivative. For p = w·x, nudging w moves p by x times as much, so its local derivative with respect to w is x = {x}. For a = tanh(z) it is 1 − a² = {da}, and for a sum it is 1. The edges in the stage carry these numbers.',
				zh: '每个运算只需要知道：某个输入稍微变一点时，它的输出跟着变多少，这叫局部导数。对 p = w·x，w 变一点，p 就变 x 倍那么多，所以 p 对 w 的局部导数就是 x = {x}。对 a = tanh(z)，局部导数是 1 − a² = {da}；对加法则是 1。舞台里的每条边上都标着这些数。',
				vars: (_rt, at) => ({ x: nodeVal(at.state, 'x'), da: graphVars(at.state).da })
			},
			look: { en: 'The label on each edge: the local derivative, computed from this forward pass\'s values.', zh: '每条边上的标签：局部导数，用这次前向传播的数值算出。' }
		},
		{
			id: 'predict',
			title: { en: 'e is used twice: predict its gradient', zh: 'e 被用了两次：预测它的梯度' },
			source: 'graph',
			at: { phase: 'seed' },
			view: 'backprop/graph',
			provenance: 'live',
			predict: {
				id: 'backprop-twice',
				question: {
					en: 'L = e·e uses e twice, so the backward pass sends a share back along each of the two edges into e. What will dL/de be?',
					zh: 'L = e·e 用了 e 两次，所以反向传播会沿着通向 e 的两条边各送回一份。dL/de 会是多少？'
				},
				options: [
					{ en: '2·e: one share of e from each edge, added', zh: '2·e：每条边送回一份 e，加起来' },
					{ en: 'e: the second share replaces the first', zh: 'e：第二份覆盖了第一份' },
					{ en: 'e·e, the same as L', zh: 'e·e，和 L 一样' }
				],
				answer: 0,
				explain: {
					en: 'Along each edge the share is dL/dL times the other factor: 1 × e. A node used twice changes L along both paths, so the shares add: e + e = 2e, the derivative of e² you know from school.',
					zh: '每条边送回的份额是 dL/dL 乘以另一个因子：1 × e。被用了两次的节点会沿两条路径影响 L，所以两份相加：e + e = 2e，正是学校里学过的 e² 的导数。'
				}
			},
			narration: {
				en: 'The backward pass starts at the end: dL/dL = 1, since L changes exactly as much as itself. Every other node still shows "?". Next comes e = {e}, which L = e·e uses twice. Answer before the stage shows it.',
				zh: '反向传播从末尾开始：dL/dL = 1，因为 L 的变化量就是它自己的变化量。其他节点都还显示“?”。下一个是 e = {e}，L = e·e 用了它两次。先回答，再看舞台。',
				vars: (_rt, at) => ({ e: nodeVal(at.state, 'e') })
			},
			look: { en: 'The two edges from e into L.', zh: '从 e 通向 L 的两条边。' }
		},
		{
			id: 'twice',
			title: { en: 'grad +=: the two shares add up', zh: 'grad +=：两份加起来' },
			source: 'graph',
			at: { phase: 'edge', step: 1 },
			view: 'backprop/graph',
			provenance: 'live',
			narration: {
				en: 'Along the first edge, e receives 1 × e = {s1}. Along the second it receives the same again, and the engine adds it: grad e += {s2}, so dL/de = {ge}, which is 2·e. Without the += it would keep only one share, half the true slope.',
				zh: '沿第一条边，e 收到 1 × e = {s1}。沿第二条边它又收到同样的一份，引擎把它加上：grad e += {s2}，所以 dL/de = {ge}，也就是 2·e。如果没有这个 +=，它只会留下一份，只有真实斜率的一半。',
				vars: (_rt, at) => ({ s1: m(at).before, s2: m(at).share, ge: m(at).after })
			},
			look: { en: 'The "grad +=" mark on e: its gradient was already known, and the second share is added to it.', zh: 'e 上的“grad +=”标记：它的梯度之前已经有了，第二份是加上去的。' }
		},
		{
			id: 'chain',
			title: { en: 'The chain rule, one edge at a time', zh: '链式法则，一次一条边' },
			source: 'graph',
			at: LAST_EDGE,
			view: 'backprop/graph',
			play: { from: { phase: 'edge', step: 2 }, to: LAST_EDGE, interval: 1100, unit: { en: 'edge {n} (counting from 0)', zh: '编号 {n} 的边（从 0 数起）' } },
			provenance: 'live',
			terms: ['chain-rule', 'backpropagation'],
			narration: {
				en: 'Each edge does one multiplication: the gradient already at the node the arrow points to, times the edge\'s local derivative, goes back to the node the arrow comes from. Through tanh the gradient goes from {ga} to {gz}, because 1 − a² = {da}. At the end, dL/dw = dL/dp × x = {gp} × {x} = {gw}: the chain rule multiplies the local derivatives along the path from L to w.',
				zh: '每条边只做一次乘法：箭头所指的节点已经拿到的梯度，乘以这条边的局部导数，再传回箭头出发的节点。经过 tanh 时，梯度从 {ga} 变成 {gz}，因为 1 − a² = {da}。最后，dL/dw = dL/dp × x = {gp} × {x} = {gw}：链式法则把从 L 到 w 一路上的局部导数乘在一起。',
				vars: (_rt, at) => graphVars(stateAt(at, LAST_EDGE))
			},
			look: { en: 'The highlighted edge and the graph\'s caption line: the gradient arriving, the local derivative and their product.', zh: '高亮的边和图的说明行：到达的梯度、局部导数和它们的乘积。' }
		},
		{
			id: 'task',
			title: { en: 'Starve w of its gradient', zh: '让 w 得不到梯度' },
			source: 'graph',
			at: LAST_EDGE,
			view: 'backprop/graph',
			provenance: 'live',
			task: {
				prompt: { en: 'Change the leaves until the loss L is above 0.5 while w\'s gradient is almost 0 (|dL/dw| below 0.01).', zh: '调整叶子节点，让损失 L 大于 0.5，同时 w 的梯度几乎为 0（|dL/dw| 小于 0.01）。' },
				done: (rt) => {
					const l = leavesOf(rt.input);
					return lossOf(l) > TASK_L && Math.abs(leafGrads(l).w) < TASK_G;
				},
				hint: {
					en: 'dL/dw is a product: dL/da × (1 − a²) × x. Keep e large and make one factor nearly 0: x near 0, or w·x + b far from 0 so that tanh goes flat.',
					zh: 'dL/dw 是一个乘积：dL/da × (1 − a²) × x。保持 e 很大，同时让其中一个因子接近 0：让 x 接近 0，或者让 w·x + b 离 0 很远，使 tanh 变平。'
				},
				waiting: { en: 'Not yet: L is still at most the goal, or w\'s gradient is not yet that small.', zh: '还没有：L 还没有超过目标，或者 w 的梯度还不够小。' },
				doneText: { en: 'Done: a large loss, and almost no gradient for w.', zh: '完成：损失很大，而 w 几乎没有梯度。' }
			},
			narration: {
				en: 'Now L = {L} and dL/dw = {gw}. That gradient is a product of the factors on w\'s path: dL/da = {ga}, 1 − a² = {da} and x = {x}. Make the loss large while one factor is nearly 0, and w stops learning although the answer is wrong.',
				zh: '现在 L = {L}，dL/dw = {gw}。这个梯度是 w 那条路径上各因子的乘积：dL/da = {ga}，1 − a² = {da}，x = {x}。让损失很大，同时让某个因子接近 0，那么尽管答案是错的，w 也不再学习。',
				vars: (_rt, at) => graphVars(at.state)
			},
			look: { en: 'w\'s gradient, and the three factors along its path.', zh: 'w 的梯度，以及它路径上的三个因子。' }
		},
		{
			id: 'check',
			title: { en: 'Check it numerically', zh: '用数值方法检验' },
			source: 'graph',
			at: { phase: 'check' },
			view: 'backprop/graph',
			provenance: 'live',
			terms: ['finite-difference'],
			narration: {
				en: 'Is the backward pass right? Nudge w up and down by h = {h}, run the graph forward twice, and divide the change in L by 2h: {nw}, against backpropagation\'s {gw}. Over all four leaves the largest difference is {diff}. This costs two forward passes per leaf, so it is a test, not the way training gets its gradients.',
				zh: '反向传播算得对吗？把 w 加上、减去 h = {h}，把图正向算两次，再用 L 的变化除以 2h：得到 {nw}，反向传播给的是 {gw}。四个叶子节点中，最大的差是 {diff}。每个叶子都要做两次前向传播，所以这只是检验，训练时并不这样求梯度。',
				vars: (_rt, at) => ({ h: fmtSetting(m(at).h), nw: m(at).nw, gw: nodeGrad(at.state, 'w'), diff: m(at).maxDiff })
			},
			look: { en: 'The check table in the stage: for each leaf, backpropagation, the numerical estimate and their difference.', zh: '舞台里的检验表：每个叶子节点的反向传播结果、数值估计和两者之差。' }
		},
		{
			id: 'net',
			title: { en: 'The same rule in a 2-2-1 network', zh: '2-2-1 网络里的同一条规则' },
			source: 'net',
			at: { phase: 'first' },
			view: 'backprop/net',
			provenance: 'live',
			terms: ['dense-layer', 'activation-function'],
			narration: {
				en: 'A 2-2-1 network (two inputs, two tanh units, one output ŷ) is a bigger graph of the same operations, with nine parameters. The backward pass goes layer by layer: dL/dŷ = {gy}; a hidden unit\'s weighted sum z gets dL/dŷ times the unit\'s output weight times its tanh slope, so dL/dz₁ = {gz1}. A first-layer weight\'s gradient is the gradient of the unit it feeds times the input it multiplies: dL/dW₁₂ = {gz1} × {x2} = {gW}.',
				zh: '2-2-1 网络（两个输入、两个 tanh 隐藏单元、一个输出 ŷ）是由同样的运算搭成的一张更大的图，有九个参数。反向传播逐层倒着走：dL/dŷ = {gy}；隐藏单元的加权和 z 得到的梯度，等于 dL/dŷ 乘以这个单元的输出权重，再乘以它的 tanh 斜率，所以 dL/dz₁ = {gz1}。第一层某个权重的梯度，等于它连向的单元的梯度乘以它所乘的输入：dL/dW₁₂ = {gz1} × {x2} = {gW}。',
				vars: (_rt, at) => ({ gy: m(at).gy, gz1: par(m(at).gz1), x2: par(nodeVal(at.state, 1)), gW: (at.state as { pgrads?: { data: ArrayLike<number> } } | undefined)?.pgrads?.data[1] ?? NaN })
			},
			look: { en: 'The parameter table: one backward pass filled in every gradient.', zh: '参数表：一次反向传播填好了所有梯度。' }
		},
		{
			id: 'cost',
			title: { en: 'Backpropagation computes, gradient descent learns', zh: '反向传播负责计算，梯度下降负责学习' },
			source: 'net',
			at: { phase: 'update' },
			view: { key: 'backprop/net', props: () => ({ check: true }) },
			provenance: 'live',
			terms: ['gradient-descent', 'learning-rate', 'optimizer'],
			narration: {
				en: 'The numerical check agrees on all {params} parameters, within {diff}, but it needed {passes} extra forward passes; backpropagation needed one backward pass. Gradient descent then uses the gradients: one step with learning rate η = {lr} lowers L from {L0} to {L1}. Backpropagation only computes the gradients; the optimizer does the learning.',
				zh: '数值检验在全部 {params} 个参数上都与反向传播一致，相差不超过 {diff}，但它多做了 {passes} 次前向传播；反向传播只做了一次反向的计算。然后梯度下降使用这些梯度：用学习率 η = {lr} 走一步，L 从 {L0} 降到 {L1}。反向传播只负责算梯度，学习是优化器做的事。',
				vars: (_rt, at) => ({ params: m(at).params, diff: m(at).maxDiff, passes: m(at).passes, lr: fmtSetting(m(at).lr), L0: m(at).L, L1: m(at).L1 })
			},
			look: { en: 'The two gradient columns of the table agree; the last column is each parameter after the update.', zh: '表格里两列梯度一致；最后一列是更新之后的每个参数。' }
		},
		{
			id: 'limits',
			title: { en: 'When tanh goes flat, the gradient vanishes', zh: 'tanh 变平时，梯度消失' },
			source: 'sat',
			at: LAST_EDGE,
			view: 'backprop/graph',
			provenance: 'live',
			kindOf: 'limits',
			terms: ['vanishing-gradient'],
			narration: {
				en: 'Here the leaves are fixed, not yours: w·x + b = {z}, where tanh is almost flat, and its local derivative 1 − a² is only {da}. The loss is large, L = {L}, yet every node before tanh, from z back to the leaves, receives almost nothing: dL/dw = {gw}. Backpropagation is exact, but exactly tiny; in a deeper network such factors multiply layer after layer, and the first layers barely learn.',
				zh: '这里用的是一组固定的叶子，不是你设定的：w·x + b = {z}，tanh 几乎是平的，它的局部导数 1 − a² 只有 {da}。损失很大，L = {L}，可是 tanh 之前的节点，从 z 一直到叶子，几乎什么都收不到：dL/dw = {gw}。反向传播是精确的，只是精确地等于一个极小的数；在更深的网络里，这样的因子一层层相乘，前面几层几乎学不动。',
				vars: (_rt, at) => graphVars(at.state)
			},
			look: { en: 'The gradient at a, and the far smaller one at z.', zh: 'a 上的梯度，以及 z 上小得多的梯度。' }
		}
	]
});

export default lesson;
