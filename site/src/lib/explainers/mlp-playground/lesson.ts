// mlp-playground: the guided lesson (design §7 outline), one idea per step. Every number is a {placeholder} computed
// from frames that ran; every sentence that is not live is tested in claims.test.ts.
import type { At, Cursor, ExplainerRuntime, LessonDef, Tensor, TmplVal, TrainState } from '../../atlas/types/index.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { dataOf, dec, evalPoints, type Act, type Params } from './net.ts';

type St = Partial<TrainState> & { act?: Act };
const L = (en: string, zh: string) => ({ en, zh });
const metric = (at: At, k: string): number => at.frame?.metrics?.[k] ?? NaN;
/** a metric of another frame of the step's tape (NaN when that frame is not recorded) */
function metricAt(at: At, c: Cursor, k: string): number {
	const i = at.tape ? at.tape.find(c) : -1;
	return i >= 0 ? (at.tape!.frame(i).metrics?.[k] ?? NaN) : NaN;
}
/** a metric at every recorded step from 1 up to the step's frame (step 0 is the start) */
function afterStart(at: At, k: string): number[] {
	const vs: number[] = [];
	for (let i = 0; at.tape && i <= at.index; i++) {
		const f = at.tape.frame(i);
		if ((f.step ?? 0) >= 1 && f.phase !== 'init' && typeof f.metrics?.[k] === 'number') vs.push(f.metrics[k]);
	}
	return vs;
}
/** the share of steps on which a metric went up (the claims test checks the same on the runs) */
const upShare = (vs: number[]) => (vs.length > 1 ? vs.slice(1).filter((v, i) => v > vs[i]).length / (vs.length - 1) : NaN);
const paramsOf = (at: At) => ((at.state as St | undefined)?.params ?? {}) as Params;
/** the weights before the first update (the init frame of the step's tape) */
const startParams = (at: At) => ((at.tape && at.tape.length ? (at.tape.state(0) as St).params : undefined) ?? {}) as Params;
/** the nudge of step "gradient": how much the loss changes when one weight grows by EPS */
export const EPS = 0.01;
/** the points the step's source trains on (its opt.data, else the learner's) */
const pointsAt = (rt: ExplainerRuntime, at: At) => dataOf(rt.input, (at.tape?.opt ?? null) as { data?: unknown } | null);
const evalAt = (rt: ExplainerRuntime, at: At) => evalPoints(paramsOf(at), (at.state as St).act === 'relu' ? 'relu' : 'tanh', pointsAt(rt, at));
const W = (p: Params, key: string, i: number) => p[key]?.data[i] ?? NaN;
const width = (at: At) => paramsOf(at)['net.0.weight']?.shape[0] ?? NaN;
const nParams = (p: Params) => Object.values(p).reduce((s, t) => s + t.data.length, 0);
/** a computed number as text, in parentheses when negative ("a × (−b)") */
const par = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));
const sign = (v: number) => `${v < 0 ? '−' : '+'} ${fmt3(Math.abs(v))}`;
/** a setting as written in the source (0.03, not 0.0300) */
const setting = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);
/** the weight with the largest gradient: key, [out, in], value */
function largestGrad(at: At): { key: string; index: [number, number]; g: number } {
	let best = { key: 'net.0.weight', index: [0, 0] as [number, number], g: NaN };
	for (const [key, t] of Object.entries(((at.state as St).grads ?? {}) as Record<string, Tensor>))
		if (key.endsWith('.weight')) t.data.forEach((g, k) => !(Math.abs(g) <= Math.abs(best.g)) && (best = { key, index: [Math.floor(k / t.shape[1]), k % t.shape[1]], g }));
	return best;
}
const grid = (show: string, extra: Record<string, unknown> = {}) => ({ key: 'mlp-playground/neuron-grid', props: () => ({ show, ...extra }) });

const lesson: LessonDef = {
	title: L('MLP playground: how a small network learns to sort points', 'MLP 游乐场：一个小网络如何学会给点分类'),
	summary: L(
		'One neuron is a soft straight line, a layer is several, and the next layer bends them into curved borders. Then one training step slowed down, training to the end, and two ways it fails.',
		'一个神经元是一条软直线，一层是好几条，下一层再把它们弯成曲线边界。然后放慢看一步训练、训练到底，以及两种失败的方式。'
	),
	steps: [
		{
			id: 'points',
			source: 'run',
			at: 'first',
			view: { key: 'plane2d', props: () => ({ grid: null }) },
			provenance: 'live',
			title: L('Two kinds of points', '两类点'),
			narration: {
				en: 'Each point is two numbers, its position x and y, plus a class: which of two groups it belongs to, shown by its color. Points in your data: {n}, with {n0} in class 0 and {n1} in class 1. The network’s job is to color the whole plane so that every point lands in the region of its own class.',
				zh: '每个点是两个数，即它的位置 x 和 y，外加一个类别：它属于两组中的哪一组，用颜色表示。你的数据有 {n} 个点，第 0 类 {n0} 个，第 1 类 {n1} 个。网络的任务是给整个平面涂色，让每个点都落在自己那一类的区域里。',
				vars: (rt, at) => {
					const p = pointsAt(rt, at);
					const n1 = p.label.reduce((s, c) => s + c, 0);
					return { n: p.n, n0: p.n - n1, n1 };
				}
			},
			look: L('The two colors: where would you draw the border between them?', '两种颜色：你会把它们的边界画在哪里？'),
			terms: ['label', 'classifier']
		},
		{
			id: 'weighted-sum',
			source: 'pt',
			at: { layer: 'net.0' },
			select: () => ({ layer: 'net.0', unit: [0] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: L('One neuron, one point', '一个神经元，一个点'),
			narration: {
				en: 'A neuron multiplies each input by its own weight, adds the products and adds a bias. For your first point (x = {px}, y = {py}), neuron 0 (counting from 0) of the first layer computes z = {w0} × {x} + {w1} × {y} + {b} = {p0} + {p1} + {b} = {z}. Then tanh squashes z into the range from −1 to 1: tanh({z}) = {a}. Training set these weights, in {steps} steps in your browser; steps 6 to 9 show how.',
				zh: '一个神经元把每个输入乘上自己的权重，把乘积加起来，再加上一个偏置。对你的第一个点（x = {px}，y = {py}），第一层的 0 号神经元（从 0 开始数）算出 z = {w0} × {x} + {w1} × {y} + {b} = {p0} + {p1} + {b} = {z}。然后 tanh 把 z 压缩到 −1 到 1 之间：tanh({z}) = {a}。这些权重是训练出来的：在你的浏览器里训练了 {steps} 步；第 6 到第 9 步讲训练是怎么回事。',
				vars: (rt): Record<string, TmplVal> => {
					const tape = rt.sources.train?.tape;
					const s = (tape?.length ? tape.state(tape.length - 1) : {}) as St;
					const p = (s.params ?? {}) as Params;
					const t = rt.sources.pt?.trace?.t;
					const x = rt.pre?.tensors.input?.data;
					if (!x || !p['net.0.weight']) return { steps: NaN };
					const [w0, w1] = [W(p, 'net.0.weight', 0), W(p, 'net.0.weight', 1)];
					// the two products as the stage's table shows them, so the chain adds up on screen
					return { px: fmt3(x[0]), py: fmt3(x[1]), w0: fmt3(w0), x: par(x[0]), w1: par(w1), y: par(x[1]), b: par(W(p, 'net.0.bias', 0)), p0: fmt3(w0 * x[0]), p1: par(w1 * x[1]), z: t?.['net.0']?.data[0] ?? NaN, a: t?.['net.1']?.data[0] ?? NaN, steps: s.step ?? NaN };
				}
			},
			look: L('The table: each input times its weight (x·w), the products added up (Σ), plus the bias b. In the table, x₀ and x₁ are your point’s x and y, and y is the sum z.', '看表：每个输入乘以它的权重（x·w），乘积加起来（Σ），再加上偏置 b。表里把你的点的 x、y 记作 x₀、x₁，把这里的和 z 记作 y。'),
			terms: ['neuron', 'weight', 'bias']
		},
		{
			id: 'soft-line',
			source: 'run',
			at: 'last',
			view: grid('neuron', { layer: 0, unit: 0, point: 0 }),
			provenance: 'live',
			title: L('One neuron is a soft line', '一个神经元是一条软直线'),
			narration: {
				en: 'Now do the same sum at every place of the plane. The neuron outputs 0 exactly where {line} = 0, and that is a straight line (drawn). On one side its output rises towards +1 (orange), on the other it falls towards −1 (blue), smoothly: a soft line. The two weights turn the line, the bias shifts it.',
				zh: '现在把同样的加法放到平面上的每个位置去做。神经元恰好在 {line} = 0 的地方输出 0，这些地方连成一条直线（图中画出）。直线一侧的输出逐渐升到 +1（橙色），另一侧降到 −1（蓝色），过渡是平滑的：一条软直线。两个权重决定直线的方向，偏置让它平移。',
				vars: (_rt, at) => {
					const w = [0, 1].map((i) => W(paramsOf(at), 'net.0.weight', i)).concat(W(paramsOf(at), 'net.0.bias', 0));
					return { line: w.every(Number.isFinite) ? `${fmt3(w[0])} × x ${sign(w[1])} × y ${sign(w[2])}` : NaN };
				}
			},
			look: L('The line where the color turns from blue to orange. Here the colors are the neuron’s output, not the classes; the ringed point is your first point.', '颜色从蓝变橙的那条线。这里的颜色是神经元的输出，不是类别；圈出的点是你的第一个点。'),
			terms: ['activation-function']
		},
		{
			id: 'layer',
			source: 'run',
			at: 'last',
			view: grid('layer', { layer: 0 }),
			provenance: 'live',
			title: L('A layer is several lines', '一层就是好几条线'),
			narration: {
				en: 'The first layer has {k} neurons side by side. Each has its own two weights and bias, so each draws its own soft line in its own direction. Training placed these {k} lines; by themselves they are all straight.',
				zh: '第一层有 {k} 个并排的神经元。每个都有自己的两个权重和一个偏置，所以每个都朝自己的方向画一条软直线。这 {k} 条线的位置是训练出来的；单独看，它们都是直的。',
				vars: (_rt, at) => ({ k: width(at) })
			},
			look: L('Each square is one neuron: compare the directions of their lines.', '每个方块是一个神经元：比较它们的线的方向。'),
			terms: ['layer']
		},
		{
			id: 'combine',
			source: 'run',
			at: 'last',
			view: grid('net'),
			provenance: 'live',
			title: L('The next layer bends the lines', '下一层把线弯起来'),
			narration: {
				en: 'Each neuron of layer 2 takes a weighted sum of all {k} first-layer outputs and squashes it with tanh again, so its border is no longer straight. The output layer takes weighted sums of layer 2 as well: two scores, one per class, turned into the share for class 1 at every place. Its border, where both classes get one half (the line in the output square), puts {acc|pct} of your points on their own side.',
				zh: '第 2 层的每个神经元把第一层全部 {k} 个输出做加权和，再用 tanh 压缩一次，所以它的边界不再是直的。输出层也对第 2 层做加权和，得到两个分数（每类一个），再换算成平面上每个位置属于第 1 类的比例。它的边界是两类各占一半的地方（输出方块里的线），这条边界让你的点中 {acc|pct} 落在自己那一侧。',
				vars: (rt, at) => ({ k: width(at), acc: evalAt(rt, at).accuracy })
			},
			look: L('Follow the edges: an orange edge (a positive weight) passes its square’s pattern on, a blue edge (a negative weight) passes it on flipped; thicker means a bigger weight.', '顺着连线看：橙色连线（正权重）把它那个方块的图案原样传下去，蓝色连线（负权重）把图案反过来传下去；越粗表示权重越大。'),
			terms: ['mlp', 'dense-layer']
		},
		{
			id: 'loss',
			source: 'run',
			at: 'first',
			view: { key: 'plane2d', props: (rt, at) => ({ highlight: evalAt(rt, at).worst }) },
			provenance: 'live',
			title: L('How wrong is each point?', '每个点错得有多厉害？'),
			predict: {
				id: 'mlp-playground-loss',
				question: L(
					'Before training, the weights are random. Which points will have the largest loss (the network most wrong about them)?',
					'训练之前，权重是随机的。哪些点的损失最大（网络对它们错得最厉害）？'
				),
				options: [L('Points deep inside the other class’s color', '深深落在另一类颜色里的点'), L('Points right on the 50/50 border', '正好在五五开边界上的点'), L('Every point has the same loss', '每个点的损失都一样')],
				answer: 0,
				explain: L(
					'The loss of a point is −ln of the share the network gives to the point’s own class (ln: the natural logarithm). Deep in the other color that share is close to 0, and −ln of a number close to 0 is large. On the border the share is one half, and −ln of one half is about 0.69.',
					'一个点的损失是网络给这个点自己那一类的比例取 −ln（ln 是自然对数）。深深落在另一类颜色里时，这个比例接近 0，而接近 0 的数取 −ln 会很大。在边界上，这个比例是一半，−ln 一半约等于 0.69。'
				)
			},
			narration: {
				en: 'Back to the start, before training: the weights are random. Training needs a number that says how wrong the network is, the loss: for one point, −ln of the share the network gives to the point’s own class (ln is the natural logarithm; this loss is called cross-entropy). The ringed point gets only {p|pct} for its own class, so its loss is −ln {pp|n4} = {lw}, the largest of all (the smallest is {lmin}). The network’s loss is the average over all points: {mean|n4}.',
				zh: '回到起点，训练之前：权重是随机的。训练需要一个数来衡量网络错得有多厉害，这就是损失：对一个点，它是网络给这个点自己那一类的比例取 −ln（ln 是自然对数；这种损失叫交叉熵）。圈出的点只得到自己那一类 {p|pct} 的比例，所以它的损失是 −ln {pp|n4} = {lw}，在所有点里最大（最小的是 {lmin}）。网络的损失是所有点的平均：{mean|n4}。',
				vars: (rt, at) => {
					const e = evalAt(rt, at);
					const pts = pointsAt(rt, at);
					const p = pts.label[e.worst] === 1 ? e.p1[e.worst] : 1 - e.p1[e.worst];
					// pp: 4 digits, so −ln of the printed share gives the printed loss on a calculator
					return { p, pp: p, lw: e.loss[e.worst], lmin: Math.min(...e.loss), mean: e.mean };
				}
			},
			look: L('The background is the random network’s answer, pale because it is unsure everywhere; hover a point to read its share.', '背景是随机网络给出的回答，颜色很淡，因为它到处都拿不准；把鼠标移到点上可以读出它的比例。'),
			terms: ['loss', 'cross-entropy']
		},
		{
			id: 'gradient',
			source: 'sgd1',
			at: { phase: 'backward', step: 0 },
			view: { key: 'network', props: () => ({ color: 'grad' }) },
			focus: (_rt, at) => [{ field: 'grads', key: largestGrad(at).key, index: largestGrad(at).index, role: 'grad' }],
			provenance: 'live',
			title: L('A gradient on every edge', '每条连线都有梯度'),
			narration: {
				en: 'To lower the loss, training asks of each of the {count} weights and biases: if this number grows a little, how fast does the loss change? That rate is its gradient: the highlighted edge has the largest in size, {g}, and indeed nudging its weight up by {eps} changes the loss by {dl}, close to {g} × {eps}. Backpropagation finds all {count} gradients in one pass backwards through the network; blue edges have a negative gradient (growing them lowers the loss), orange ones a positive gradient.',
				zh: '为了降低损失，训练对全部 {count} 个权重和偏置逐个发问：这个数稍微变大一点，损失变化得有多快？这个变化率就是它的梯度：高亮的连线梯度最大（按大小算），是 {g}；确实，把它的权重调大 {eps}，损失就变化 {dl}，接近 {g} × {eps}。反向传播沿网络倒着走一遍就算出全部 {count} 个梯度；蓝色连线的梯度为负（把它调大会降低损失），橙色的为正。',
				vars: (rt, at): Record<string, TmplVal> => {
					const b = largestGrad(at);
					// the nudge, computed on the weights this gradient belongs to (before the update)
					const p0 = startParams(at), w = p0[b.key];
					if (!w) return { count: NaN };
					const bumped: Params = { ...p0, [b.key]: { shape: w.shape, data: Float64Array.from(w.data) } };
					bumped[b.key].data[b.index[0] * w.shape[1] + b.index[1]] += EPS;
					const pts = pointsAt(rt, at);
					const dl = evalPoints(bumped, 'tanh', pts).mean - evalPoints(p0, 'tanh', pts).mean;
					return { count: nParams(((at.state as St).grads ?? {}) as Params), g: b.g, eps: setting(EPS), dl: dec(dl) };
				}
			},
			look: L('The highlighted edge: the weight whose change matters most right now.', '高亮的连线：此刻变化影响最大的那个权重。'),
			terms: ['gradient', 'backpropagation']
		},
		{
			id: 'update',
			source: 'sgd1',
			at: { phase: 'update', step: 0 },
			view: 'train-step',
			focus: (_rt, at) => [{ field: 'params', key: largestGrad(at).key, index: largestGrad(at).index, role: 'param' }],
			provenance: 'live',
			title: L('One step downhill', '往下走一步'),
			narration: {
				en: 'Gradient descent moves every weight a small step against its gradient, scaled by the learning rate {lr}: the highlighted weight goes from {old} to {old} − {lr} × ({g}) = {new}. All {count} numbers move at once, and the loss over all your points falls by {drop} to {after|n4}.',
				zh: '梯度下降让每个权重沿梯度的反方向走一小步，步长由学习率 {lr} 缩放：高亮的权重从 {old} 变成 {old} − {lr} × ({g}) = {new}。全部 {count} 个数同时移动，在你所有的点上，损失下降了 {drop}，变成 {after|n4}。',
				vars: (rt, at) => {
					const b = largestGrad(at);
					const u = updateOf(at, b.key, b.index);
					const after = evalAt(rt, at).mean;
					return { old: u.old, g: u.g, new: u.new, lr: setting(u.lr), count: nParams(paramsOf(at)), drop: metric(at, 'loss') - after, after };
				}
			},
			look: L('Old weight, gradient, learning rate, new weight.', '旧权重、梯度、学习率、新权重。'),
			terms: ['gradient-descent', 'learning-rate']
		},
		{
			id: 'watch',
			source: 'run',
			at: 'last',
			view: grid('net'),
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 90 },
			provenance: 'live',
			title: L('Repeat, and watch it learn', '重复下去，看它学会'),
			narration: {
				en: 'Training repeats such steps. This run uses Adam, an optimizer that also adapts each weight’s step size, with learning rate {lr} for {steps} steps on all your points each time. The loss falls from {l0} to {l1}, and {acc|pct} of the points end up on their own side.',
				zh: '训练就是一遍遍重复这样的一步。这次运行用的是 Adam，一种还会调整每个权重步长的优化器，学习率 {lr}，训练 {steps} 步，每一步都用你全部的点。损失从 {l0} 降到 {l1}，{acc|pct} 的点最后落在自己那一侧。',
				vars: (rt, at) => ({ lr: setting(metric(at, 'lr')), steps: at.frame?.step ?? NaN, l0: dec(metricAt(at, { phase: 'snapshot', step: 0 }, 'loss')), l1: dec(metric(at, 'loss')), acc: metric(at, 'accuracy') })
			},
			look: L('Every square changes at once: the lines turn, the output border bends around the points, and the loss under the picture falls.', '所有方块同时在变：线在转动，输出的边界绕着点弯曲，图下方的损失在下降。'),
			terms: ['optimizer']
		},
		{
			id: 'spiral',
			source: 'spiral2',
			at: 'last',
			view: grid('net', { choices: [2, 4, 8].map((k) => ({ source: `spiral${k}`, label: L(`${k} neurons per layer`, `每层 ${k} 个神经元`) })) }),
			provenance: 'live',
			kindOf: 'limits',
			title: L('Too small for a spiral', '对螺旋来说太小了'),
			narration: {
				en: 'A harder shape, the same for everyone: two spiral arms, {n} points. With only {k} neurons per layer the network has {k} lines to bend, and after {steps} steps it gets just {acc|pct} of the points right; its loss is still {loss}. Training longer does not fix this: the network is too small for the shape.',
				zh: '一个更难的形状，每个人看到的都一样：两条螺旋臂，{n} 个点。每层只有 {k} 个神经元时，网络只有 {k} 条线可以弯，训练 {steps} 步后只答对 {acc|pct} 的点，损失还有 {loss}。训练更久也解决不了：网络对这个形状来说太小了。',
				vars: (rt, at) => ({ n: pointsAt(rt, at).n, k: width(at), steps: at.frame?.step ?? NaN, acc: metric(at, 'accuracy'), loss: metric(at, 'loss') })
			},
			task: {
				prompt: L('Give the network more neurons: click “8 neurons per layer” right above the picture (on this step, that button is the input to change) and watch it learn the spiral.', '给网络更多神经元：点图片正上方的“每层 8 个神经元”（这一步要改的输入就是这个按钮），看它学会螺旋。'),
				done: (rt) => {
					const t = rt.sources.spiral8?.tape;
					return !!t?.length && (t.frame(t.length - 1).metrics?.accuracy ?? 0) >= 0.9;
				},
				hint: L('The three buttons sit between this box and the picture.', '三个按钮就在这个框和图片之间。')
			},
			look: L('Compare the output square for 2, 4 and 8 neurons per layer.', '比较每层 2、4、8 个神经元时的输出方块。')
		},
		{
			id: 'high-lr',
			source: 'lrs',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metrics: ['loss', 'hiLoss'], labels: { loss: L('learning rate 0.03', '学习率 0.03'), hiLoss: L('learning rate 3', '学习率 3') } }) },
			provenance: 'live',
			kindOf: 'breakIt',
			title: L('Too big a step', '步子太大'),
			narration: {
				en: 'The learning rate sets the size of each step. Same spiral, same network with 8 neurons per layer, same start, but learning rate {hi} instead of {lo}: after one step the loss jumps from {l0} to {j1}. The steps keep overshooting: the loss goes up on {ups|pct} of them and never gets below {hiMin} again, and in the end only {hiAcc|pct} of the points are right. With {lo} the loss falls to {loLast}, with {loAcc|pct} right.',
				zh: '学习率决定每一步迈多大。同样的螺旋、同样每层 8 个神经元的网络、同样的起点，只把学习率从 {lo} 换成 {hi}：走一步之后损失就从 {l0} 跳到 {j1}。之后的步子还是一再迈过头：{ups|pct} 的步子让损失不降反升，损失再也没有低于 {hiMin}，最后只答对 {hiAcc|pct} 的点。学习率为 {lo} 时，损失降到 {loLast}，答对 {loAcc|pct}。',
				vars: (_rt, at) => {
					const opt = (at.tape?.opt ?? {}) as { lr?: number; lrHi?: number };
					// the lr-3 run is chaotic (its last digits differ between browsers and Node): narrate only what holds for every
					// such run (claims.test.ts perturbs it): it rises on many steps and stays far above the lr-0.03 run
					const hiLoss = afterStart(at, 'hiLoss');
					return { hi: setting(opt.lrHi), lo: setting(opt.lr), l0: metricAt(at, { phase: 'snapshot', step: 0 }, 'loss'), j1: metricAt(at, { step: 1 }, 'hiLoss'), ups: upShare(hiLoss), hiMin: hiLoss.length ? Math.min(...hiLoss) : NaN, hiAcc: metric(at, 'hiAccuracy'), loLast: metric(at, 'loss'), loAcc: metric(at, 'accuracy') };
				}
			},
			look: L('The two curves on a log scale: one goes down, the other jumps up and stays high.', '对数刻度上的两条曲线：一条往下走，另一条跳上去后一直很高。'),
			terms: ['learning-rate']
		}
	],
	// inputs every step must survive, besides the adapter's edge fixtures (no points, one point, one class)
	fixtures: [
		...(['moons', 'circles', 'xor', 'spiral', 'gaussians'] as const).map((id) => ({ id, label: L(`Preset: ${id}`, `预设：${id}`), raw: presetRaw(id) })),
		{ id: 'painted', label: L('A few painted points', '几个手画的点'), raw: { points: [-0.8, -0.5, -0.2, 0.1, 0.4, 0.7].flatMap((x, i) => [{ x, y: 0.4 + 0.1 * i, c: 0 }, { x, y: -0.3 - 0.1 * i, c: 1 }]) } }
	]
};

export default lesson;
