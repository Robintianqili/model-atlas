// hello-xor-2: process 'sigmoidStep' (kind C). One plain gradient-descent step of the 2-2-1 network with a SIGMOID
// output on the four XOR points, slowed down into its phases: init -> forward (one frame per point) -> loss (binary
// cross-entropy) -> backward -> update -> check. Every number is computed here, in float64, and recorded;
// pipeline/hello-xor-2/reference.py is an independent Python run that must give the same frames
// (`npm run atlas -- parity hello-xor-2`, AK204 on the first difference).
// The starting weights come from ctx.rng (never Math.random): w = scale * (2u - 1) with u = rng.next(), in the order
// W1[0][0], W1[0][1], W1[1][0], W1[1][1], b1[0], b1[1], W2[0], W2[1], b2. next() is bit-identical in Python (ak.Rng).
// With a sigmoid output and binary cross-entropy, the gradient at the output score is simply (yhat - y) / 4.
//   npm run atlas -- run hello-xor-2 step1
import type { Control, Frame, ProcessDef, Tensor } from '../../atlas/types/index.ts';

export type StepOpt = { lr: number; scale: number };
export type StepState = {
	W1: Tensor;
	b1: Tensor;
	W2: Tensor;
	b2: Tensor;
	x: Tensor;
	z1: Tensor;
	h: Tensor;
	z2: number;
	yhat: number;
	target: number;
	loss: number;
	gW1: Tensor;
	gb1: Tensor;
	gW2: Tensor;
	gb2: Tensor;
	lossAfter: number;
};
/** Spelled out so every yielded frame is checked against Frame<StepState> on its own. */
type Run = Generator<Frame<StepState>, void, Control<StepOpt> | undefined>;

/** The XOR truth table (the same four points as pipeline/hello-xor-2/model.py and the trainer). */
export const XS: [number, number][] = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
];
export const YS = [0, 1, 1, 0];

const T = (shape: number[], data: number[]): Tensor => ({ shape, data: Float64Array.from(data) });

interface Weights {
	W1: number[][];
	b1: number[];
	W2: number[];
	b2: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function forward(w: Weights, x: [number, number]): { z1: number[]; h: number[]; z2: number; yhat: number } {
	const z1 = [0, 1].map((j) => w.W1[j][0] * x[0] + w.W1[j][1] * x[1] + w.b1[j]);
	const h = z1.map((z) => Math.tanh(z));
	const z2 = w.W2[0] * h[0] + w.W2[1] * h[1] + w.b2;
	return { z1, h, z2, yhat: sigmoid(z2) };
}

/** Binary cross-entropy of one prediction p for the target y (0 or 1). */
const bce = (p: number, y: number) => -(y * Math.log(p) + (1 - y) * Math.log(1 - p));

function lossOf(w: Weights): number {
	let s = 0;
	for (let i = 0; i < 4; i++) s += bce(forward(w, XS[i]).yhat, YS[i]);
	return s / 4;
}

const tensors = (w: Weights) => ({ W1: T([2, 2], w.W1.flat()), b1: T([2], w.b1), W2: T([1, 2], w.W2), b2: T([1], [w.b2]) });

export const sigmoidStep: ProcessDef<unknown, StepOpt, StepState> = {
	id: 'sigmoidStep',
	kind: 'algorithm',
	title: { en: 'One gradient-descent step with a sigmoid output, slowed down', zh: '放慢的一步梯度下降（sigmoid 输出）' },
	phases: {
		init: {
			label: { en: 'Starting weights', zh: '起始权重' },
			kind: 'other',
			narration: {
				what: {
					en: 'The network starts from {frame.count|int} random numbers, its weights and biases, between {frame.lo} and {frame.hi}. With these numbers it gets XOR wrong.',
					zh: '网络从 {frame.count|int} 个随机数出发，也就是它的权重和偏置，它们在 {frame.lo} 和 {frame.hi} 之间。用这些数，它还算不对 XOR。'
				}
			}
		},
		forward: {
			label: { en: 'Forward pass, one point', zh: '前向计算：一个点' },
			kind: 'forward',
			// one product per line, then the sum and the sigmoid, so every filled line stays short on a phone
			formula: {
				tex: 'z = w_1 h_1 + w_2 h_2 + b = {{p1}} + {{p2}} + ({{b}}) = {{z}};\\quad \\hat y = \\sigma(z) = \\frac{1}{1 + e^{-z}} = {{yhat}}',
				values: (f) => ({
					p1: f.metrics!.w1 * f.metrics!.h1,
					p2: f.metrics!.w2 * f.metrics!.h2,
					b: f.metrics!.b,
					z: f.metrics!.z,
					yhat: f.metrics!.yhat
				})
			},
			narration: {
				what: {
					en: 'The point ({frame.x1}, {frame.x2}) goes through both layers to the score {frame.z}. The sigmoid turns that score into the probability {frame.yhat}, and the right answer is {frame.target}.',
					zh: '点 ({frame.x1}, {frame.x2}) 依次通过两层，得到分数 {frame.z}。sigmoid 把这个分数变成概率 {frame.yhat}，正确答案是 {frame.target}。'
				}
			}
		},
		loss: {
			label: { en: 'Loss', zh: '损失' },
			kind: 'loss',
			formula: {
				tex: 'L = -\\tfrac{1}{4} \\sum_i \\big[ y_i \\log \\hat y_i + (1 - y_i) \\log (1 - \\hat y_i) \\big] = {{loss}}',
				values: (f) => ({ loss: f.metrics!.loss })
			},
			narration: {
				what: {
					en: 'The loss is the binary cross-entropy: for each point, minus the log of the probability the network gave to the right answer, averaged over the four points: {frame.loss}. A perfect network would score 0.',
					zh: '损失是二元交叉熵：对每个点，取网络给正确答案的概率的对数再取负，然后在四个点上平均：{frame.loss}。完美的网络得分是 0。'
				}
			}
		},
		backward: {
			label: { en: 'Backward pass', zh: '反向传播' },
			kind: 'backward',
			narration: {
				what: {
					en: 'With a sigmoid and cross-entropy, the gradient at each output score is just (probability − target) / 4. The chain rule carries it back: the first output weight gets {frame.g}.',
					zh: '用 sigmoid 加交叉熵时，每个输出分数上的梯度就是（概率 − 目标）/ 4。链式法则再把它传回去：第一个输出权重的梯度是 {frame.g}。'
				}
			}
		},
		update: {
			label: { en: 'Update', zh: '更新' },
			kind: 'update',
			formula: {
				tex: 'w_{\\text{new}} = w - \\eta\\, g = {{old}} - {{lr}} \\times ({{g}}) = {{new}}',
				values: (f) => ({ old: f.metrics!.old, lr: f.metrics!.lr, g: f.metrics!.g, new: f.metrics!.new })
			},
			narration: {
				what: {
					en: 'Every weight takes a small step against its gradient. The first output weight goes from {frame.old} to {frame.new}.',
					zh: '每个权重都朝梯度的反方向走一小步。第一个输出权重从 {frame.old} 变成 {frame.new}。'
				}
			}
		},
		check: {
			label: { en: 'Did it help?', zh: '有用吗？' },
			kind: 'observe',
			narration: {
				what: {
					en: 'With the new weights the loss is {frame.lossAfter}, down from {frame.loss}. Training repeats this step many times.',
					zh: '用新的权重，损失是 {frame.lossAfter}，之前是 {frame.loss}。训练就是把这一步重复很多次。'
				}
			}
		}
	},
	schema: {
		W1: { kind: 'tensor', label: { en: 'Hidden-layer weights', zh: '隐藏层权重' }, shape: [2, 2], role: 'param', scale: 'diverging', tol: 1e-12 },
		b1: { kind: 'tensor', label: { en: 'Hidden-layer biases', zh: '隐藏层偏置' }, shape: [2], role: 'param', scale: 'diverging', tol: 1e-12 },
		W2: { kind: 'tensor', label: { en: 'Output weights', zh: '输出层权重' }, shape: [1, 2], role: 'param', scale: 'diverging', tol: 1e-12 },
		b2: { kind: 'tensor', label: { en: 'Output bias', zh: '输出层偏置' }, shape: [1], role: 'param', scale: 'diverging', tol: 1e-12 },
		x: { kind: 'tensor', label: { en: 'The point', zh: '这个点' }, shape: [2], role: 'data', tol: 'exact' },
		z1: { kind: 'tensor', label: { en: 'Hidden sums', zh: '隐藏层的和' }, shape: [2], role: 'activation', scale: 'diverging', tol: 1e-12 },
		h: { kind: 'tensor', label: { en: 'Hidden values (after tanh)', zh: '隐藏值（tanh 之后）' }, shape: [2], role: 'activation', scale: 'diverging', tol: 1e-12 },
		z2: { kind: 'scalar', label: { en: 'Output score (before the sigmoid)', zh: '输出分数（sigmoid 之前）' }, role: 'activation', tol: 1e-12 },
		yhat: { kind: 'scalar', label: { en: 'Network output (a probability)', zh: '网络输出（一个概率）' }, role: 'output', tol: 1e-12 },
		target: { kind: 'scalar', label: { en: 'Right answer', zh: '正确答案' }, role: 'target', tol: 'exact' },
		loss: { kind: 'scalar', label: { en: 'Loss (binary cross-entropy)', zh: '损失（二元交叉熵）' }, role: 'output', tol: 1e-12 },
		gW1: { kind: 'tensor', label: { en: 'Gradient of the hidden weights', zh: '隐藏层权重的梯度' }, shape: [2, 2], role: 'grad', scale: 'diverging', tol: 1e-12 },
		gb1: { kind: 'tensor', label: { en: 'Gradient of the hidden biases', zh: '隐藏层偏置的梯度' }, shape: [2], role: 'grad', scale: 'diverging', tol: 1e-12 },
		gW2: { kind: 'tensor', label: { en: 'Gradient of the output weights', zh: '输出层权重的梯度' }, shape: [1, 2], role: 'grad', scale: 'diverging', tol: 1e-12 },
		gb2: { kind: 'tensor', label: { en: 'Gradient of the output bias', zh: '输出层偏置的梯度' }, shape: [1], role: 'grad', scale: 'diverging', tol: 1e-12 },
		lossAfter: { kind: 'scalar', label: { en: 'Loss after the step', zh: '这一步之后的损失' }, role: 'output', tol: 1e-12 }
	},
	defaults: { lr: 0.5, scale: 1 },
	*run(_input, ctx): Run {
		const { lr, scale } = ctx.opt;
		const draw = () => scale * (2 * ctx.rng.next() - 1);
		const w: Weights = { W1: [[0, 0], [0, 0]], b1: [0, 0], W2: [0, 0], b2: 0 };
		w.W1[0][0] = draw();
		w.W1[0][1] = draw();
		w.W1[1][0] = draw();
		w.W1[1][1] = draw();
		w.b1[0] = draw();
		w.b1[1] = draw();
		w.W2[0] = draw();
		w.W2[1] = draw();
		w.b2 = draw();
		yield { phase: 'init', state: tensors(w), metrics: { count: 9, lo: -scale, hi: scale } };

		// forward: one frame per point
		const fw = XS.map((x) => forward(w, x));
		for (let i = 0; i < 4; i++) {
			const { z1, h, z2, yhat } = fw[i];
			yield {
				phase: 'forward',
				step: 0,
				sub: i,
				state: { x: T([2], XS[i]), z1: T([2], z1), h: T([2], h), z2, yhat, target: YS[i] },
				metrics: { x1: XS[i][0], x2: XS[i][1], h1: h[0], h2: h[1], w1: w.W2[0], w2: w.W2[1], b: w.b2, z: z2, yhat, target: YS[i] },
				focus: [{ field: 'x', role: 'in' }, { field: 'yhat', role: 'out' }]
			};
		}

		// loss: binary cross-entropy averaged over the four points
		let s = 0;
		for (let i = 0; i < 4; i++) s += bce(fw[i].yhat, YS[i]);
		const loss = s / 4;
		yield { phase: 'loss', step: 0, state: { loss }, metrics: { loss } };

		// backward: dL/dz2_i = (yhat_i - y_i) / 4 (sigmoid + cross-entropy), then the chain rule through fc2, tanh and fc1
		const gW1 = [[0, 0], [0, 0]];
		const gb1 = [0, 0];
		const gW2 = [0, 0];
		let gb2 = 0;
		for (let i = 0; i < 4; i++) {
			const { h, yhat } = fw[i];
			const d = (yhat - YS[i]) / 4;
			gW2[0] += d * h[0];
			gW2[1] += d * h[1];
			gb2 += d;
			for (let j = 0; j < 2; j++) {
				const dz = d * w.W2[j] * (1 - h[j] * h[j]);
				gW1[j][0] += dz * XS[i][0];
				gW1[j][1] += dz * XS[i][1];
				gb1[j] += dz;
			}
		}
		yield {
			phase: 'backward',
			step: 0,
			state: { gW1: T([2, 2], gW1.flat()), gb1: T([2], gb1), gW2: T([1, 2], gW2), gb2: T([1], [gb2]) },
			metrics: { g: gW2[0], loss }
		};

		// update: w <- w - lr * g for every weight
		const old = w.W2[0];
		for (let j = 0; j < 2; j++) {
			w.W1[j][0] -= lr * gW1[j][0];
			w.W1[j][1] -= lr * gW1[j][1];
			w.b1[j] -= lr * gb1[j];
			w.W2[j] -= lr * gW2[j];
		}
		w.b2 -= lr * gb2;
		const lossAfter = lossOf(w);
		yield { phase: 'update', step: 0, state: tensors(w), metrics: { lr, old, g: gW2[0], new: w.W2[0], loss, lossAfter } };

		yield { phase: 'check', step: 0, state: { lossAfter }, metrics: { loss, lossAfter } };
	}
};
