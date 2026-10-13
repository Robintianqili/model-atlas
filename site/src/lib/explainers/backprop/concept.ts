// backprop: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { nodeGrad, nodeVal } from './data.ts';

export const hook: I18n = {
	en: 'The networks on this site learned from gradients: for each weight, how the loss would change if that weight moved a little. Backpropagation gets all of them in one sweep backwards through the computation. Here the computation is small enough to watch every number on the way.',
	zh: '这个网站上的网络都是靠梯度学会的：对每个权重来说，梯度就是这个权重稍微动一下时损失会怎样变化。反向传播沿着计算倒着扫一遍，就得到所有梯度。这里的计算很小，你可以看清一路上的每一个数。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Step the backward pass yourself', zh: '自己一步步走反向传播' },
	source: 'graph',
	at: { phase: 'forward' },
	view: { key: 'backprop/graph', props: () => ({ stepper: true }) },
	provenance: 'live',
	narration: {
		en: 'Your leaves give L = {L}. Each press of the step button crosses one edge, from L back to the leaves, and fills in one gradient. Change a leaf and the whole graph recomputes; the check button compares every gradient with nudging its leaf.',
		zh: '你的叶子节点给出 L = {L}。每按一次“走一条边”，反向传播就从 L 往叶子方向跨过一条边，填好一个梯度。改动任意一个叶子，整张图都会重新计算；“数值检验”按钮会把每个梯度和微调对应叶子的结果作比较。',
		vars: (_rt, at) => ({ L: nodeVal(at.state, 'L') })
	},
	look: { en: 'The graph\'s caption line says which edge was crossed and what it multiplied.', zh: '图的说明行写着刚跨过哪条边、乘了什么。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'Nine gradients of a 2-2-1 network in one backward pass', zh: '一次反向传播得到 2-2-1 网络的九个梯度' },
	source: 'net',
	at: { phase: 'check' },
	view: { key: 'backprop/net', props: () => ({ check: true }) },
	provenance: 'live',
	terms: ['backpropagation'],
	narration: {
		en: 'The network answers {yh} for a target of {t}, a loss of {L}. One backward pass gives all {params} parameter gradients; nudging each parameter instead takes {passes} forward passes and agrees within {diff}. A bigger network of the same tanh units learns live in the MLP playground.',
		zh: '网络的回答是 {yh}，目标是 {t}，损失为 {L}。一次反向传播就得到全部 {params} 个参数的梯度；改用逐个微调参数的办法要做 {passes} 次前向传播，两者相差不超过 {diff}。由同样的 tanh 单元搭成的更大的网络，可以在 MLP 游乐场里实时训练。',
		vars: (_rt, at) => {
			const f = at.frame?.metrics ?? {};
			return { yh: f.yh, t: f.t, L: f.L, params: f.params, passes: f.passes, diff: f.maxDiff };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Your graph next to a saturated one', zh: '你的计算图和一个饱和的计算图' },
	source: 'graph',
	at: { phase: 'edge', step: 8 },
	view: 'backprop/graph',
	compare: { a: 'graph', b: 'sat', label: [{ en: 'Your leaves', zh: '你的叶子节点' }, { en: 'A saturated neuron', zh: '一个饱和的神经元' }] },
	provenance: 'live',
	terms: ['vanishing-gradient'],
	narration: {
		en: 'The same graph and the same backward pass, twice. On your leaves dL/dw = {gw}. On the saturated leaves tanh passes back only {daS} times the gradient that reaches it, and dL/dw = {gwS}.',
		zh: '同一张图、同样的反向传播，算两次。在你的叶子上，dL/dw = {gw}。在饱和的叶子上，tanh 只把到达它的梯度的 {daS} 倍传回去，dL/dw = {gwS}。',
		vars: (rt, at) => {
			const t = rt.sources.sat?.tape;
			const i = t ? t.find({ phase: 'edge', step: 8 }) : -1;
			const s = i >= 0 ? t!.state(i) : undefined;
			const a = nodeVal(s, 'a');
			return { gw: nodeGrad(at.state, 'w'), daS: 1 - a * a, gwS: nodeGrad(s, 'w') };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'A node used twice gets the gradient of one of its uses.', zh: '被用了两次的节点只得到其中一次使用的梯度。' },
		truth: {
			en: 'Its shares add up (grad +=), because it changes the loss along every path. This site\'s autograd, like PyTorch, also adds each new backward pass into a parameter\'s stored gradient, which is why training loops clear it before every step.',
			zh: '它的各份梯度要加起来（grad +=），因为它沿每条路径都会影响损失。本站的自动求导和 PyTorch 一样，也会把每次新的反向传播加到参数已存的梯度上，所以训练循环在每一步之前都要先把它清零。'
		},
		step: 'twice'
	},
	{
		claim: { en: 'The numerical gradient is just as good, and simpler.', zh: '数值梯度一样好，而且更简单。' },
		truth: {
			en: 'It is an estimate, and it needs two forward passes for every parameter. Backpropagation gives the exact gradients in one backward pass, so the numerical one is only used to test it.',
			zh: '它只是一个估计，而且每个参数都要做两次前向传播。反向传播用一次反向的计算就得到精确的梯度，所以数值梯度只用来检验它。'
		},
		step: 'check'
	},
	{
		claim: { en: 'An exact gradient is always a useful gradient.', zh: '精确的梯度总是有用的梯度。' },
		truth: {
			en: 'Exact can be almost zero. One small factor on the path, such as a flat tanh or an input near 0, makes the whole product small, and the weights behind it barely move.',
			zh: '精确的梯度也可能几乎为零。路径上只要有一个很小的因子，例如平坦的 tanh 或接近 0 的输入，整个乘积就会很小，它后面的权重几乎不动。'
		},
		step: 'limits'
	}
];

export const further = [
	{ label: { en: 'The MLP playground: a small network trained live with these gradients', zh: 'MLP 游乐场：用这些梯度实时训练的小网络' }, href: '/m/mlp-playground/' },
	{ label: { en: 'Loss and gradient descent: what the gradients are used for', zh: '损失与梯度下降：梯度拿来做什么' }, href: '/train/loss-gradient-descent/' },
	{ label: { en: 'CS231n notes: backpropagation, intuitions (Stanford)', zh: 'CS231n 课程笔记：反向传播的直观理解（斯坦福）' }, href: 'https://cs231n.github.io/optimization-2/' },
	{ label: { en: 'micrograd: a tiny autograd engine in Python (Andrej Karpathy)', zh: 'micrograd：一个用 Python 写的很小的自动求导引擎（Andrej Karpathy）' }, href: 'https://github.com/karpathy/micrograd' }
];
