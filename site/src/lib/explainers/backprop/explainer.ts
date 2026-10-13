// backprop: the manifest of a CONCEPT package (K2): routed at /train/backprop/ by the concept route, which renders
// AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links).
// Backpropagation as the chain rule on a small computation graph: one neuron with a squared error, written as five
// operations on four leaves the learner sets; the backward pass crosses one edge at a time (process.ts). Sources:
//   graph  the toy graph on the learner's leaves: forward, seed, nine edges, numerical check      (process 'graph')
//   sat    the same graph on fixed saturated leaves (w·x + b = 6), for the limits step            (process 'graph')
//   net    a 2-2-1 network with written-out weights: forward, three backward layers, check, update (process 'net')
// The page around the lesson is concept.ts.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { SAT } from './data.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const manifest: ConceptManifest = {
	slug: 'backprop',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'leaves', start: { x: 1.5, w: 0.8, b: -0.5, y: 0.9 } },
	sources: {
		graph: { kind: 'process', process: 'graph' },
		sat: { kind: 'process', process: 'graph', input: () => ({ ...SAT }) },
		net: { kind: 'process', process: 'net', input: () => null }
	},
	page: {
		goals: [
			{
				en: 'Write a small formula as a computation graph, run it forward, and name each node\'s local derivative.',
				zh: '把一个小公式写成计算图，做一遍前向传播，并说出每个节点的局部导数。'
			},
			{
				en: 'Run the backward pass one edge at a time: each gradient is the child\'s gradient times a local derivative, and a node used twice adds both shares (grad +=).',
				zh: '一次走一条边地执行反向传播：每个梯度都是子节点的梯度乘以局部导数；一个被用了两次的节点要把两份加起来（grad +=）。'
			},
			{
				en: 'Check a gradient numerically, and explain why training uses backpropagation instead and where its gradients become tiny.',
				zh: '用数值方法检验梯度，并说明为什么训练用的是反向传播，以及它的梯度在什么情况下会变得极小。'
			}
		],
		prequestion: {
			id: 'backprop-pre',
			question: {
				en: 'A network has a million weights. To learn, it needs the slope of the loss for every one of them. What does backpropagation cost?',
				zh: '一个网络有一百万个权重。为了学习，它需要损失对每一个权重的斜率。反向传播要花多少代价？'
			},
			options: [{ en: 'About one extra pass backwards through the network, for all the slopes at once', zh: '大约多走一遍反向的计算，一次得到所有斜率' }, { en: 'One extra pass per weight, a million passes', zh: '每个权重多走一遍，一共一百万遍' }, { en: 'It guesses the slopes, so it costs nothing', zh: '它是猜出这些斜率的，所以没有代价' }],
			answer: 0,
			explain: {
				en: 'The backward pass visits every operation once, from the loss back to the leaves, and every weight\'s slope falls out on the way. Nudging the weights one at a time would need two forward passes per weight.',
				zh: '反向传播从损失出发，把每个运算倒着走一遍，沿途就得到每个权重的斜率。如果一个一个地微调权重，每个权重都要做两次前向传播。'
			}
		},
		quiz: [
			{
				q: { en: 'A node m feeds two later nodes. Along the first edge it receives a share of 0.3, along the second −0.1. What is dL/dm?', zh: '节点 m 连到后面两个节点。沿第一条边它收到 0.3，沿第二条边收到 −0.1。dL/dm 是多少？' },
				options: [{ en: '0.2', zh: '0.2' }, { en: '0.3 (the first share)', zh: '0.3（第一份）' }, { en: '−0.1 (the last share)', zh: '−0.1（最后一份）' }],
				answer: 0,
				explain: { en: 'A node used twice changes L along both paths, so the shares add up: 0.3 + (−0.1) = 0.2. That is the "grad +=" of every autograd engine.', zh: '被用了两次的节点会沿两条路径影响 L，所以两份要加起来：0.3 + (−0.1) = 0.2。这就是每个自动求导引擎里的“grad +=”。' }
			},
			{
				q: { en: 'At a tanh node the output is a = 0.99. Roughly what fraction of the arriving gradient passes back to z?', zh: '某个 tanh 节点的输出 a = 0.99。到达的梯度大约有多少能传回 z？' },
				options: [{ en: 'About 2% (1 − a² ≈ 0.02)', zh: '大约 2%（1 − a² ≈ 0.02）' }, { en: 'All of it', zh: '全部' }, { en: 'About 99%', zh: '大约 99%' }],
				answer: 0,
				explain: { en: 'The local derivative of tanh is 1 − a² = 1 − 0.9801 = 0.0199. A flat tanh passes back only a sliver of the gradient: the start of a vanishing gradient.', zh: 'tanh 的局部导数是 1 − a² = 1 − 0.9801 = 0.0199。平坦的 tanh 只能传回一点点梯度：这就是梯度消失的开端。' }
			}
		],
		misconception: {
			claim: { en: 'Backpropagation is how a network learns.', zh: '反向传播就是网络学习的方法。' },
			truth: {
				en: 'Backpropagation only computes the gradients. Learning is the next step: an optimizer such as gradient descent uses them to change the weights.',
				zh: '反向传播只负责算出梯度。学习是下一步：由梯度下降之类的优化器用这些梯度去改变权重。'
			},
			step: 'cost'
		},
		lessonNote: {
			en: 'Ten steps: run a tiny graph forward, walk its backward pass one edge at a time, check it numerically, then do the same on a 2-2-1 network and see where gradients vanish.',
			zh: '十步：先把一张小计算图正向算一遍，再一条边一条边地走反向传播，用数值方法检验，然后在 2-2-1 网络上做同样的事，并看看梯度在哪里消失。'
		}
	},
	glossary: [
		{
			id: 'computation-graph', term: { en: 'computation graph', zh: '计算图' }, related: ['backpropagation', 'chain-rule'],
			def: { en: 'A formula written as a network of simple operations: each node computes one number from the nodes that point into it. Autograd records it during the forward pass and walks it backwards.', zh: '把一个公式写成由简单运算组成的网络：每个节点用指向它的节点算出一个数。自动求导在前向传播时把它记下来，再倒着走一遍。' }
		},
		{
			id: 'vanishing-gradient', term: { en: 'vanishing gradient', zh: '梯度消失' }, related: ['backpropagation', 'gradient'],
			def: { en: 'A gradient that becomes tiny on its way back because it is multiplied by small local derivatives, such as those of a saturated tanh; the weights before them, in the earlier layers, barely learn.', zh: '梯度在反向传回的路上被一些很小的局部导数（例如饱和的 tanh）相乘而变得极小，排在它们前面、更靠近输入的权重几乎学不动。' }
		}
	],
	credits: [
		{
			name: 'micrograd (ideas)',
			url: 'https://github.com/karpathy/micrograd',
			license: 'MIT',
			usedFor: 'Ideas for the backpropagation page: a graph of scalar operations whose backward pass adds each edge\'s share into the parent\'s gradient (grad +=), and the numerical gradient check. Code written fresh for this project; every number comes from the site\'s own autograd',
			attribution: 'Andrej Karpathy, micrograd (MIT License)'
		}
	]
};

export default manifest;
