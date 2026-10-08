// The five elementwise activations of CORE_OPS: relu, tanh, sigmoid, silu, gelu (+ alias gelu_tanh).
// Each is one definition on the shared elementwise zoom (design §4.3). The one-file-per-op entry points
// (relu.ts, tanh.ts, …) re-export these.
import { sym } from '../core/tex.ts';
import { erf, t, words } from './common.ts';
import { pointwise } from './pointwise.ts';

type NoP = Record<string, never>;

export const relu = pointwise<NoP>({
	op: 'relu',
	label: t('ReLU', 'ReLU'),
	defaults: {},
	// tinycnn.ts: d[i] = src[i] > 0 ? src[i] : 0  (NaN and −0 become 0)
	fn: (v) => (v > 0 ? v : 0),
	curve: () => 'relu',
	narration: {
		what: t('ReLU keeps positive numbers and replaces negative numbers with 0. Each cell is handled on its own.', 'ReLU 保留正数，把负数换成 0。每个格子单独处理。'),
		how: t('For every cell: output = max(0, input). The shape stays {outShape}.', '对每个格子：输出 = max(0, 输入)。形状不变，仍是 {outShape}。'),
		why: t(
			'Without a bend like this, stacked weighted sums would collapse into one weighted sum. ReLU is the simplest bend, and it keeps learning easy.',
			'没有这样的“弯折”，多层加权和叠起来仍然只是一个加权和。ReLU 是最简单的弯折，而且便于学习。'
		),
		look: t('Every negative cell of the previous layer is now exactly 0.', '上一层所有负的格子现在都正好是 0。'),
		unit: t('Input {x}, output {y}.', '输入 {x}，输出 {y}。')
	},
	formula: `${sym('out', 'y')} = \\max(0,\\ ${sym('in', 'x')}) = \\max(0,\\ {{x}}) = {{y}}`,
	terms: ['relu', 'activation-function'],
	pytorch: () => 'nn.ReLU()'
});

export const tanh = pointwise<NoP>({
	op: 'tanh',
	label: t('Tanh', 'Tanh'),
	defaults: {},
	fn: (v) => Math.tanh(v),
	curve: () => 'tanh',
	narration: {
		what: t(
			'Tanh squashes every number into the range from −1 to 1. Small numbers pass almost unchanged; large ones flatten out near −1 or 1.',
			'Tanh 把每个数压到 −1 到 1 之间。小的数几乎不变，大的数被压平在 −1 或 1 附近。'
		),
		how: t('For every cell: output = tanh(input). The shape stays {outShape}.', '对每个格子：输出 = tanh(输入)。形状不变，仍是 {outShape}。'),
		why: t('The bend lets stacked layers draw curved boundaries, and the output always stays in a fixed range.', '这个弯曲让多层网络能画出弯曲的边界，而且输出始终在固定范围内。'),
		look: t('Values near −1 or 1 are saturated: changing the input there barely changes the output.', '接近 −1 或 1 的值已经“饱和”：输入再变，输出也几乎不变。'),
		unit: t('tanh({x}) = {y}.', 'tanh({x}) = {y}。')
	},
	formula: `${sym('out', 'y')} = \\tanh(${sym('in', 'x')}) = \\tanh({{x}}) = {{y}}`,
	pytorch: () => 'nn.Tanh()'
});

export const sigmoid = pointwise<NoP>({
	op: 'sigmoid',
	label: t('Sigmoid', 'Sigmoid'),
	defaults: {},
	fn: (v) => 1 / (1 + Math.exp(-v)),
	curve: () => 'sigmoid',
	color: 'sequential',
	narration: {
		what: t(
			'The sigmoid squashes every number into the range from 0 to 1, so it can be read as a share or as how far a switch is open.',
			'sigmoid 把每个数压到 0 到 1 之间，所以可以把它看成一个比例，或者一个开关打开的程度。'
		),
		how: t(
			'For every cell: output = 1 / (1 + e^−x). An input of 0 gives one half; large positive inputs approach 1 and large negative inputs approach 0.',
			'对每个格子：输出 = 1 / (1 + e^−x)。输入 0 得到一半；很大的正数接近 1，很大的负数接近 0。'
		),
		why: t(
			'Gates in recurrent networks and yes-or-no outputs use it, because a number between 0 and 1 can scale another number down smoothly.',
			'循环网络里的“门”和“是或否”输出都用它，因为 0 到 1 之间的数可以平滑地把另一个数按比例缩小。'
		),
		look: t('Cells near 1 are fully on; cells near 0 are off.', '接近 1 的格子是“全开”，接近 0 的格子是“关”。'),
		unit: t('σ({x}) = {y}.', 'σ({x}) = {y}。')
	},
	formula: `${sym('out', 'y')} = \\frac{1}{1 + e^{-${sym('in', 'x')}}} = \\frac{1}{1 + e^{-({{x}})}} = {{y}}`,
	pytorch: () => 'nn.Sigmoid()'
});

export const silu = pointwise<NoP>({
	op: 'silu',
	label: t('SiLU', 'SiLU'),
	defaults: {},
	fn: (v) => v / (1 + Math.exp(-v)),
	curve: () => 'silu',
	extra: (x) => ({ sig: 1 / (1 + Math.exp(-x)) }),
	narration: {
		what: t(
			'SiLU multiplies each number by its own sigmoid, x · σ(x). Large positive numbers pass almost unchanged, large negative numbers become nearly 0, and small negative numbers dip slightly below 0.',
			'SiLU 把每个数乘以它自己的 sigmoid：x · σ(x)。大的正数几乎不变，大的负数接近 0，较小的负数会略低于 0。'
		),
		how: t('For every cell: output = x / (1 + e^−x). The shape stays {outShape}.', '对每个格子：输出 = x / (1 + e^−x)。形状不变，仍是 {outShape}。'),
		why: t('It is a smooth version of ReLU: a small change of the input always changes the output a little, which helps training.', '它是 ReLU 的平滑版本：输入的微小变化总会让输出稍微变化，这有助于训练。'),
		look: t('Compare with ReLU: negative cells are not cut to exactly 0.', '和 ReLU 比较：负的格子没有被直接截成 0。'),
		unit: t('{x} × σ({x}) = {x} × {sig} = {y}.', '{x} × σ({x}) = {x} × {sig} = {y}。')
	},
	formula: `${sym('out', 'y')} = ${sym('in', 'x')}\\,\\sigma(${sym('in', 'x')}) = {{x}} \\cdot {{sig}} = {{y}}`,
	pytorch: () => 'nn.SiLU()'
});

type GeluP = { approx: 'none' | 'tanh' };
const GELU_C = Math.sqrt(2 / Math.PI); // as tinygpt.ts
const geluPhi = (v: number, p: GeluP) =>
	p.approx === 'tanh' ? 0.5 * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v))) : 0.5 * (1 + erf(v / Math.SQRT2));

export const gelu = pointwise<GeluP>({
	op: 'gelu',
	label: t('GELU', 'GELU'),
	defaults: { approx: 'none' },
	attrs: { approx: { type: 'string', default: 'none', doc: "'none' (exact, with erf) or 'tanh' (the GPT-2 approximation)" } },
	aliases: { gelu_tanh: { approx: 'tanh' } },
	// tinygpt.ts gelu: y = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)))  (same expression, bit for bit)
	fn: (v, p) => {
		const a = p.approx as string | undefined;
		if (a === 'tanh') return 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
		if (a !== undefined && a !== 'none') throw new Error(`gelu: approx must be 'none' or 'tanh', got ${JSON.stringify(a)}`);
		return 0.5 * v * (1 + erf(v / Math.SQRT2));
	},
	curve: (p) => (p.approx === 'tanh' ? 'gelu_tanh' : 'gelu'),
	vars: (layer) => ({
		approxNote:
			layer.op === 'gelu_tanh' || (layer.params ?? {}).approx === 'tanh'
				? words('This layer computes Φ with a common tanh shortcut, which is very close to the exact curve.', '这一层用一个常见的 tanh 公式近似计算 Φ，和精确曲线非常接近。')
				: words('This layer computes Φ exactly.', '这一层精确地计算 Φ。')
	}),
	extra: (x, p) => ({ phi: geluPhi(x, p) }),
	narration: {
		what: t(
			'GELU multiplies each number by a share between 0 and 1 that grows with the number: big positive numbers pass almost unchanged, big negative numbers become almost 0, and numbers near 0 are partly let through.',
			'GELU 把每个数乘以一个介于 0 和 1 之间、随这个数变大而变大的比例：大的正数几乎原样通过，大的负数几乎变成 0，接近 0 的数只通过一部分。'
		),
		// {approxNote}: which Φ this layer computes (approx 'tanh', also the alias gelu_tanh, or the exact erf form)
		how: t(
			'Output = x · Φ(x), where Φ(x) is the share of a standard bell curve that lies below x. {approxNote}',
			'输出 = x · Φ(x)，其中 Φ(x) 是标准钟形曲线落在 x 左边的面积占比。{approxNote}'
		),
		why: t('Like ReLU it bends, but smoothly. Transformers such as GPT use it inside their MLP blocks.', '它和 ReLU 一样会“弯折”，但弯得很平滑。GPT 等 Transformer 在 MLP 部分使用它。'),
		look: t('Small negative values are not cut to exactly 0: they dip slightly below 0.', '较小的负数没有被截成 0，而是略低于 0。'),
		unit: t('{x} × Φ({x}) = {x} × {phi} = {y}.', '{x} × Φ({x}) = {x} × {phi} = {y}。')
	},
	formula: `${sym('out', 'y')} = ${sym('in', 'x')}\\,\\Phi(${sym('in', 'x')}) = {{x}} \\cdot {{phi}} = {{y}}`,
	pytorch: (_l, p) => (p.approx === 'tanh' ? "nn.GELU(approximate='tanh')" : 'nn.GELU()')
});
