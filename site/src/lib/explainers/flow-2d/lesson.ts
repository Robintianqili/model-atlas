// flow-2d: the guided lesson (design §7 outline), one idea per step. Every number is a {placeholder} computed from the
// frames that ran (or the learner's points); every sentence that is not live is tested in claims.test.ts.
import type { At, ExplainerRuntime, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { dataOf, showcase, SIZES, velocity, type Params } from './flow.ts';
import { middleOf, pairsOf, predOf, runEnd, spreadAround } from './views/plane/meta.ts';
import { presetOf } from './adapters/points2d/adapter.ts';

const L = (en: string, zh: string) => ({ en, zh });
type St = { params?: Params; grads?: Params; batch?: Record<string, Tensor>; step?: number; x?: Tensor };
const st = (at: At) => (at.state ?? {}) as St;
const metric = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
/** a computed number as text, in parentheses when negative ("a × (−b)") */
const par = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));
/** a step size as written when short (0.02, 0.25), else 3 significant digits */
const short = (v: number) => (String(v).length <= 6 ? String(v) : fmt3(v));
const plane = (mode: string, extra: (rt: ExplainerRuntime, at: At) => Record<string, unknown> = () => ({})) => ({ key: 'flow-2d/plane', props: (rt: ExplainerRuntime, at: At) => ({ mode, pair: showcase(rt.input)?.i ?? 0, ...extra(rt, at) }) });
const nParams = (p: Params | undefined) => Object.values(p ?? {}).reduce((s, t) => s + t.data.length, 0);
/** the pair the lesson follows (flow.ts showcase) in the frame's batch: its index, ends, t, point x_t and target x1 − x0 */
function pairAt(rt: ExplainerRuntime, at: At) {
	const p = pairsOf({ at });
	const k = showcase(rt.input)?.i ?? 0;
	if (!p || k >= p.n) return null;
	const inp = st(at).batch!.inp.data;
	return { i: k, n: p.n, x0: p.x0[2 * k], y0: p.x0[2 * k + 1], x1: p.x1[2 * k], y1: p.x1[2 * k + 1], t: p.t[k], xt: inp[3 * k], yt: inp[3 * k + 1], dx: p.target[2 * k], dy: p.target[2 * k + 1] };
}
/** the mean squared error of these weights on the frame's batch */
function batchLoss(at: At, params: Params): number {
	const b = st(at).batch;
	if (!b) return NaN;
	const n = b.inp.shape[0];
	const xy = new Float64Array(2 * n);
	const ts = new Float64Array(n);
	for (let i = 0; i < n; i++) (xy[2 * i] = b.inp.data[3 * i]), (xy[2 * i + 1] = b.inp.data[3 * i + 1]), (ts[i] = b.inp.data[3 * i + 2]);
	const v = velocity(params, xy, ts, n);
	let s = 0;
	for (let i = 0; i < 2 * n; i++) s += (v[i] - b.target.data[i]) ** 2;
	return s / (2 * n);
}
/** the weight with the largest gradient: key, [out, in], value */
function largestGrad(at: At): { key: string; index: [number, number]; g: number } {
	let best = { key: 'net.0.weight', index: [0, 0] as [number, number], g: NaN };
	for (const [key, t] of Object.entries(st(at).grads ?? {}))
		if (key.endsWith('.weight')) t.data.forEach((g, k) => !(Math.abs(g) <= Math.abs(best.g)) && (best = { key, index: [Math.floor(k / t.shape[1]), k % t.shape[1]], g }));
	return best;
}
/** what a weight `net.<l>.weight[out, in]` connects, in words (MLP('net', [3, 64, 64, 64, 2]): net.0 reads x, y, t;
 *  net.6 writes the arrow), counting neurons from 0 */
export function wireOf(key: string, [o, i]: [number, number]): { en: string; zh: string } {
	const l = Number(key.split('.')[1]) / 2;
	const ord = (n: number) => ['first', 'second', 'third'][n - 1] ?? `${n}th`;
	const inp = l === 0 ? L(`the input ${'xyt'[i]}`, `输入 ${'xyt'[i]}`) : L(`neuron ${i} of the ${ord(l)} layer of ${SIZES[l]}`, `第 ${l} 层（${SIZES[l]} 个神经元）的 ${i} 号神经元`);
	const out = l + 2 >= SIZES.length ? L(`the arrow’s ${'xy'[o]}`, `箭头的 ${'xy'[o]}`) : L(`neuron ${o} of the ${ord(l + 1)} layer of ${SIZES[l + 1]}`, `第 ${l + 1} 层（${SIZES[l + 1]} 个神经元）的 ${o} 号神经元`);
	return L(`${inp.en} to ${out.en} (neurons counted from 0)`, `${inp.zh}和${out.zh}（神经元从 0 开始数）`);
}
/** the mean loss over `n` recorded training steps from the start (first) or before the step's frame (last; `skip`
 *  leaves out that many of the latest steps first) */
function meanLoss(at: At, n: number, from: 'first' | 'last', skip = 0): number {
	const vs: number[] = [];
	for (let i = 0; at.tape && i <= at.index; i++) {
		const f = at.tape.frame(i);
		if ((f.phase === 'step' || f.phase === 'snapshot') && typeof f.metrics?.loss === 'number' && (f.step ?? 0) < (st(at).step ?? 0)) vs.push(f.metrics.loss);
	}
	const w = from === 'first' ? vs.slice(0, n) : !skip ? vs.slice(-n) : vs.length >= n + skip ? vs.slice(vs.length - n - skip, vs.length - skip) : [];
	return w.length ? w.reduce((a, b) => a + b, 0) / w.length : NaN;
}
const xOf = (at: At, phase: string, step: number) => {
	const i = at.tape ? at.tape.find({ phase, step }) : -1;
	return i >= 0 ? ((at.tape!.state(i) as St).x?.data ?? null) : null;
};
/** the share of sampled points whose last Euler move (of K) points more than 15° away from their first: the paths bend */
function bentShare(at: At, K: number): number {
	const [s0, s1, e0, e1] = [xOf(at, 'noise', 0), xOf(at, 'move', 0), xOf(at, 'move', K - 2), xOf(at, 'move', K - 1)];
	if (!s0 || !s1 || !e0 || !e1) return NaN;
	const turn = (i: number, [ux, uy, wx, wy] = [s1[2 * i] - s0[2 * i], s1[2 * i + 1] - s0[2 * i + 1], e1[2 * i] - e0[2 * i], e1[2 * i + 1] - e0[2 * i + 1]]) => (ux * wx + uy * wy) / (Math.hypot(ux, uy) * Math.hypot(wx, wy));
	return Array.from({ length: s0.length / 2 }, (_, i) => turn(i)).filter((c) => c < Math.cos(Math.PI / 12)).length / (s0.length / 2);
}

const lesson: LessonDef = {
	title: L('2-D flow matching: from noise to your shape', '二维流匹配：从噪声到你的形状'),
	summary: L(
		'Pairs of noise and data joined by straight lines, a network that learns the arrow along them, then Euler steps that carry fresh noise onto your shape, and what fewer, bigger steps do.',
		'把噪声和数据用直线配成对，让网络学会沿线的箭头，再用欧拉法一步步把新的噪声带到你的形状上，最后看步数更少、步子更大时会怎样。'
	),
	steps: [
		{
			id: 'data',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('data'),
			provenance: 'live',
			title: L('Your shape', '你的形状'),
			narration: {
				en: 'Your shape: {pts}. A generative model learns to make new points that look as if they came from the same shape. This one starts from random noise and moves it onto your shape; the next steps build the training data it learns that from.',
				zh: '你的形状：{pts}。生成模型学的是造出新的点，让它们看起来像是来自同一个形状。这个模型从随机噪声出发，把噪声移到你的形状上；接下来几步搭出它学这件事用的训练数据。',
				vars: (rt) => ({ pts: plural(dataOf(rt.input).n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' }) })
			},
			look: L('The green points. Pick another preset or paint your own in the input box: every step follows.', '绿色的点。可以在输入框里换一个预设，或者自己画：后面每一步都会跟着变。'),
			terms: ['generative-model']
		},
		{
			id: 'noise',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('noise'),
			provenance: 'live',
			title: L('Noise', '噪声'),
			narration: {
				en: 'Every new point starts as noise: two random numbers, x and y, drawn from the standard normal distribution, a bell curve around 0 with spread 1. Training draws a batch of {b} noise points for each step (grey); {inside|pct} of their x and y values lie between −2 and 2, as the bell curve predicts. Noise knows nothing about your shape.',
				zh: '每个新点都从噪声开始：两个随机数 x 和 y，取自标准正态分布，也就是以 0 为中心、标准差（宽度）为 1 的钟形曲线。训练的每一步抽一批 {b} 个噪声点（灰色）；它们的 x 值和 y 值有 {inside|pct} 落在 −2 到 2 之间，正如钟形曲线所预言。噪声对你的形状一无所知。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const p = pairsOf({ at });
					return p ? { b: p.n, inside: Array.from(p.x0).filter((v) => Math.abs(v) < 2).length / p.x0.length } : { b: NaN };
				}
			},
			look: L('Grey: noise, spread over the whole plane. Green: your points.', '灰色是噪声，铺满整个平面；绿色是你的点。'),
			terms: ['gaussian-noise', 'batch']
		},
		{
			id: 'pairs',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('pairs'),
			provenance: 'live',
			title: L('Straight pairs', '直线配对'),
			narration: {
				en: 'Each noise point x₀ gets a partner x₁: a data point picked at random (one point may be picked more than once). A straight line joins the two. Pair {i} (counting from 0, the thick line) runs from x₀ = ({a0}, {b0}) to x₁ = ({a1}, {b1}). These {b} pairs are what the first training step learns from.',
				zh: '每个噪声点 x₀ 都配一个伙伴 x₁：随机挑的一个数据点（同一个点可能被挑中不止一次）。两者之间连一条直线。第 {i} 对（从 0 开始数，粗线那条）从 x₀ = ({a0}, {b0}) 连到 x₁ = ({a1}, {b1})。训练的第一步就从这 {b} 对里学。',
				vars: (rt, at): Record<string, TmplVal> => {
					const q = pairAt(rt, at);
					return q ? { i: q.i, a0: fmt3(q.x0), b0: fmt3(q.y0), a1: fmt3(q.x1), b1: fmt3(q.y1), b: q.n } : { b: NaN };
				}
			},
			look: L('The thick line: one grey noise point x₀ joined to one green data point x₁. The thin lines are the other pairs.', '粗线：一个灰色噪声点 x₀ 连到一个绿色数据点 x₁。细线是其他的配对。')
		},
		{
			id: 'slider',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('slider', (rt, at) => ({ t: pairAt(rt, at)?.t ?? 0.5 })),
			provenance: 'live',
			title: L('A point on the line at time t', '时刻 t 在线上的点'),
			narration: {
				en: 'A time t between 0 and 1 says how far along its line a point is: x_t = (1 − t)·x₀ + t·x₁. At t = 0 every point is noise; at t = 1 every point sits on your shape. Training gives each pair its own random t: pair {i} got t = {t}, so its point is x_t = ({xt}, {yt}).',
				zh: '0 到 1 之间的时间 t 表示点沿着自己的线走了多远：x_t = (1 − t)·x₀ + t·x₁。t = 0 时每个点都是噪声；t = 1 时每个点都落在你的形状上。训练时每一对都有自己的随机 t：第 {i} 对抽到 t = {t}，所以它的点是 x_t = ({xt}, {yt})。',
				vars: (rt, at): Record<string, TmplVal> => {
					const q = pairAt(rt, at);
					return q ? { i: q.i, t: q.t, xt: fmt3(q.xt), yt: fmt3(q.yt) } : { t: NaN };
				}
			},
			look: L('Drag the slider: all the pairs slide together from the noise (t = 0) to your shape (t = 1).', '拖动滑块：所有配对一起从噪声（t = 0）滑到你的形状（t = 1）。')
		},
		{
			id: 'target',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('pair'),
			provenance: 'live',
			title: L('The network’s job', '网络的任务'),
			narration: {
				en: 'Along a straight line a point moves at a constant velocity: it covers x₁ − x₀ in one unit of time. For pair {i} that velocity is ({dx}, {dy}), the green arrow, and it is the same at every t. This is the network’s job: given only where a point is (x and y) and the time t, output this arrow. It never sees which pair the point came from.',
				zh: '沿着直线，点以恒定的速度移动：一个单位时间走完 x₁ − x₀。第 {i} 对的速度是 ({dx}, {dy})，就是绿色箭头，而且在每个 t 都一样。这就是网络的任务：只知道点在哪里（x 和 y）以及时间 t，输出这个箭头。它从来不知道这个点来自哪一对。',
				vars: (rt, at): Record<string, TmplVal> => {
					const q = pairAt(rt, at);
					return q ? { i: q.i, dx: fmt3(q.dx), dy: fmt3(q.dy) } : { dx: NaN, dy: NaN };
				}
			},
			look: L('The green arrow starts at the ringed pair’s point x_t and points along its line.', '绿色箭头从圈出的那一对的点 x_t 出发，沿着它的线指出去。'),
			terms: ['velocity-field']
		},
		{
			id: 'network',
			source: 'job',
			at: { layer: 'net.0' },
			select: () => ({ layer: 'net.0', unit: [0] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: L('Inside the network', '网络内部'),
			narration: {
				en: 'The network v is an MLP, {arch}: three numbers in, two out (the arrow), {count} weights and biases. For pair {i}’s point and time, neuron 0 (counting from 0) of its first layer computes {w0} × {x} + {w1} × {y} + {w2} × {t} + {b} = {p0} + {p1} + {p2} + {b} = {z}; the activation function SiLU, which keeps a large positive number almost as it is and squashes a negative one toward 0, turns that into {a}. Right now every weight is random: training has not started.',
				zh: '网络 v 是一个 MLP，{arch}：输入三个数，输出两个数（箭头），一共 {count} 个权重和偏置。对第 {i} 对的点和时间，第一层的 0 号神经元（从 0 开始数）算出 {w0} × {x} + {w1} × {y} + {w2} × {t} + {b} = {p0} + {p1} + {p2} + {b} = {z}；激活函数 SiLU 让大的正数几乎原样通过、把负数压向 0，于是它变成 {a}。现在所有权重都是随机的：训练还没开始。',
				vars: (rt): Record<string, TmplVal> => {
					const tape = rt.sources.t0?.tape;
					const p = (tape?.length ? (tape.state(tape.length - 1) as St).params : undefined) ?? {};
					const x = rt.pre?.tensors.input?.data;
					const tr = rt.sources.job?.trace?.t;
					const w = p['net.0.weight']?.data;
					if (!x || !w || !tr) return { count: NaN };
					const sizes = [p['net.0.weight'].shape[1], ...[0, 2, 4, 6].map((l) => p[`net.${l}.weight`]?.shape[0] ?? NaN)];
					return {
						i: showcase(rt.input)?.i ?? NaN,
						arch: sizes.join(' → '),
						count: nParams(p),
						w0: fmt3(w[0]), w1: par(w[1]), w2: par(w[2]), x: par(x[0]), y: par(x[1]), t: fmt3(x[2]), b: par(p['net.0.bias'].data[0]),
						p0: fmt3(w[0] * x[0]), p1: par(w[1] * x[1]), p2: par(w[2] * x[2]),
						z: tr['net.0']?.data[0] ?? NaN, a: tr['net.1']?.data[0] ?? NaN
					};
				}
			},
			look: L('The table: rows 0, 1 and 2 are the x, y and t of the ringed point (here x_i only means input i, not the noise x₀ or the data x₁). Each is multiplied by its weight w, the products are added (Σ), then the bias b: the result y is the z of the narration.', '看表：第 0、1、2 行是圈出的那个点的 x、y 和 t（这里的 x_i 只表示第 i 个输入，不是噪声 x₀，也不是数据 x₁）。每个数乘以自己的权重 w，乘积相加（Σ），再加上偏置 b：结果 y 就是讲解里的 z。'),
			terms: ['mlp', 'neuron', 'weight', 'bias', 'activation-function']
		},
		{
			id: 'arrow',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('pair', () => ({ arrows: ['target', 'pred'] })),
			provenance: 'live',
			title: L('The untrained arrow', '未训练的箭头'),
			predict: {
				id: 'flow-2d-arrow',
				question: L('How will the untrained network’s arrow at the ringed pair compare with the target arrow?', '未训练的网络在圈出的那一对上给出的箭头，和目标箭头比会怎样？'),
				options: [
					L('About the same: the network already points toward the data', '差不多：网络已经指向数据了'),
					L('Much shorter, pointing wherever its random weights send it', '短得多，指向哪里由随机权重决定'),
					L('Exactly zero: an untrained network outputs nothing', '正好是零：未训练的网络什么都不输出')
				],
				answer: 1,
				explain: L(
					'Random starting weights are small, so the pink arrow is tiny: less than a tenth as long as the green one, but not zero. Its direction comes from the random weights, not from your shape. Making the pink arrow match the green one, for every pair and every t, is what training is for.',
					'随机的初始权重都很小，所以粉色箭头很短：不到绿色箭头的十分之一，但不是零。它的方向来自随机权重，而不是你的形状。让粉色箭头和绿色箭头对上，而且对每一对、每个 t 都对上，正是训练要做的事。'
				)
			},
			narration: {
				en: 'Now ask the untrained network for its arrow at pair {i}’s point x_t = ({xt}, {yt}) and time t = {t}. The target is ({dx}, {dy}). Make your guess before the arrows appear.',
				zh: '现在问未训练的网络：在第 {i} 对的点 x_t = ({xt}, {yt})、时间 t = {t}，箭头是什么？目标是 ({dx}, {dy})。先猜一猜，再看箭头。',
				vars: (rt, at): Record<string, TmplVal> => {
					const q = pairAt(rt, at);
					return q ? { i: q.i, xt: fmt3(q.xt), yt: fmt3(q.yt), t: q.t, dx: fmt3(q.dx), dy: fmt3(q.dy) } : { t: NaN };
				}
			},
			look: L('Green: the target x₁ − x₀. Pink: what the network says now. Both arrows start at the ringed point.', '绿色：目标 x₁ − x₀。粉色：网络现在给出的箭头。两个箭头都从圈出的点出发。')
		},
		{
			id: 'loss',
			source: 'sgd1',
			at: { phase: 'loss', step: 0 },
			view: plane('pair', () => ({ arrows: ['target', 'pred'] })),
			provenance: 'live',
			title: L('How far off: the loss', '差多远：损失'),
			narration: {
				en: 'The loss says how far off the arrows are: for each pair, square the two differences between the network’s arrow and the target, and average them. For pair {i}: (({px} − {dx})² + ({py} − {dy})²) / 2 = {e0}. The average over all {b} pairs is the loss of this batch: {loss|n4}. A loss of 0 would mean every arrow is exactly right.',
				zh: '损失衡量箭头差多远：对每一对，把网络箭头和目标在两个方向上的差各自平方，再取平均。第 {i} 对是 (({px} − {dx})² + ({py} − {dy})²) / 2 = {e0}。对全部 {b} 对取平均，就是这一批的损失：{loss|n4}。损失为 0 就表示每个箭头都完全对。',
				vars: (rt, at): Record<string, TmplVal> => {
					const q = pairAt(rt, at);
					const v = q ? predOf({ at }, q.i) : null;
					if (!q || !v) return { loss: NaN };
					return { i: q.i, px: fmt3(v[0]), py: fmt3(v[1]), dx: par(q.dx), dy: par(q.dy), e0: ((v[0] - q.dx) ** 2 + (v[1] - q.dy) ** 2) / 2, b: q.n, loss: metric(at, 'loss') };
				}
			},
			look: L('The gap between the pink and the green arrow: its squared size, averaged over the pairs, is the loss.', '粉色箭头和绿色箭头之间的差距：把它平方，再对所有配对取平均，就是损失。'),
			terms: ['loss', 'mse']
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
				en: 'Backpropagation finds the gradient of the loss for all {count} weights and biases, and one gradient-descent step moves each a little against its gradient. The weight with the largest gradient connects {where}: it goes from {old} to {old} − {lr} × {g} = {new}. On the same {b} pairs the loss falls by {drop}, to {after|n4}.',
				zh: '反向传播求出损失对全部 {count} 个权重和偏置的梯度，梯度下降的一步让每个参数沿梯度的反方向移动一点点。梯度最大的那个权重连接{where}：它从 {old} 变成 {old} − {lr} × {g} = {new}。在同样的 {b} 对上，损失下降了 {drop}，变成 {after|n4}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const best = largestGrad(at);
					const u = updateOf(at, best.key, best.index);
					const after = batchLoss(at, st(at).params ?? {});
					return { count: nParams(st(at).params), where: wireOf(best.key, best.index), old: u.old, g: par(u.g), new: u.new, lr: String(u.lr), b: st(at).batch?.inp.shape[0] ?? NaN, drop: metric(at, 'loss') - after, after };
				}
			},
			look: L('The rule that ran: old weight − learning rate × gradient = new weight. The stage names the weight layer.weight[row, column]: the row is the neuron it feeds, the column the input it reads; net.6 is the last layer, whose rows 0 and 1 are the arrow’s x and y.', '实际执行的规则：旧权重 − 学习率 × 梯度 = 新权重。图中权重的名字是“层.weight[行, 列]”：行是它送往的神经元，列是它读入的输入；net.6 是最后一层，它的第 0 行和第 1 行就是箭头的 x 和 y。'),
			terms: ['gradient', 'backpropagation', 'gradient-descent', 'learning-rate']
		},
		{
			id: 'watch',
			source: 'flow',
			at: { phase: 'snapshot', last: true },
			view: plane('field'),
			play: { from: { phase: 'snapshot', step: 0 }, to: { phase: 'snapshot', last: true }, phases: ['snapshot'], interval: 160 },
			provenance: 'live',
			title: L('Watch the arrows learn', '看箭头学起来'),
			narration: {
				en: 'Training repeats such steps {steps} times, each on {b} fresh random pairs, with Adam, an optimizer that also adapts each weight’s step size (learning rate {lr}). Averaged over {win} steps, the loss falls from {l0} to {l1}{floor}. The arrows are the network’s answer everywhere at the slider’s time: drag it and watch them turn.',
				zh: '训练把这样的一步重复 {steps} 次，每次用 {b} 对新的随机配对，优化器是 Adam，它还会调整每个权重的步长（学习率 {lr}）。按 {win} 步取平均，损失从 {l0} 降到 {l1}{floor}。箭头是网络在滑块所示时刻、在每个位置给出的回答：拖动滑块，看它们转向。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const win = 100;
					const l1 = meanLoss(at, win, 'last');
					return {
						steps: st(at).step ?? NaN,
						b: (at.tape?.opt as { batch?: number } | undefined)?.batch ?? NaN,
						lr: String(metric(at, 'lr')),
						win, l0: meanLoss(at, win, 'first'),
						l1,
						// "and no lower" only when the loss has levelled off: the last window's mean is within 0.05 of the mean two windows earlier
						floor: pick(l1 > 0.25 && Math.abs(meanLoss(at, win, 'last', 2 * win) - l1) < 0.05, L(', and no lower: where many lines cross, one place and time has many different targets, and the network can only answer with their average', '，再也降不下去：许多直线交叉的地方，同一位置、同一时刻有许多不同的目标，网络只能给出它们的平均'), L('', ''))
					};
				}
			},
			look: L('The arrows (scaled to fit) and the training step under the picture. Play replays training from copies of the network saved along the way.', '看箭头（按比例缩放）和图下方的训练步数。“播放”用训练途中保存的网络副本，把训练重放一遍。'),
			terms: ['optimizer', 'velocity-field']
		},
		{
			id: 'euler',
			source: 'flow',
			at: { phase: 'move', last: true },
			view: plane('euler'),
			play: { from: { phase: 'noise' }, to: { phase: 'move', last: true }, phases: ['noise', 'move'], interval: 90, unit: L('Euler step', '欧拉步') },
			provenance: 'live',
			title: L('From noise to new points', '从噪声到新的点'),
			narration: {
				en: 'To make new points, take {n} fresh noise points and follow the arrows in {k} small steps (Euler’s method): x ← x + h·v(x, t), with h = 1/{k} = {h}. Point 0’s first step goes from ({x0}, {y0}) along the arrow ({vx}, {vy}) to ({x1}, {y1}). At t = 1 the points lie on average {d1} from the nearest data point; as noise they were {d0} away.',
				zh: '要造新的点，就取 {n} 个新的噪声点，沿着箭头走 {k} 小步（欧拉法）：x ← x + h·v(x, t)，其中 h = 1/{k} = {h}。0 号点的第一步从 ({x0}, {y0}) 沿箭头 ({vx}, {vy}) 走到 ({x1}, {y1})。到 t = 1 时，这些点离最近的数据点平均 {d1}；还是噪声时，它们平均离 {d0}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const a = xOf(at, 'noise', 0);
					const b = xOf(at, 'move', 0);
					const p = st(at).params;
					const e = runEnd({ at }, 'move');
					if (!a || !b || !p || !e) return { n: NaN };
					const v = velocity(p, a, 0, 1);
					return { n: a.length / 2, k: e.K, h: short(1 / e.K), x0: fmt3(a[0]), y0: fmt3(a[1]), vx: fmt3(v[0]), vy: fmt3(v[1]), x1: fmt3(b[0]), y1: fmt3(b[1]), d1: e.dist, d0: e.dist0 };
				}
			},
			look: L('Pink: the sampled points and the paths they took. Press Play to watch them move.', '粉色：采样的点和它们走过的路径。按“播放”看它们移动。'),
			task: {
				prompt: L('Your turn: pick Ring (under the picture or in the input box) or paint a shape of your own, and watch the network learn it and these points follow.', '轮到你了：选“圆环”（图下方或输入框里都可以），或者自己画一个形状，看网络学会它，这些点也跟着过去。'),
				done: (rt) => JSON.stringify(rt.input) !== JSON.stringify(presetOf('smiley')) && rt.sources.flow?.status === 'ready',
				waiting: L('Not yet: pick another shape, then wait for the new training to finish.', '还没有：先换一个形状，再等新的训练结束。')
			},
			terms: ['euler-method']
		},
		{
			id: 'few',
			source: 'flow',
			at: { phase: 'moveFew', last: true },
			view: plane('few'),
			provenance: 'live',
			kindOf: 'breakIt',
			title: L('Four big steps instead of fifty', '四大步代替五十小步'),
			narration: {
				en: 'Each Euler step follows the arrow where the step starts. At t = 0 almost every arrow points to the middle of your shape (the cross), so {kf} big steps of h = {hf} (left) {pull}: they end {r4} from the middle on average, against {r50} after {k} small steps (right){turn}, and {rd} for your points. {verdict} Diffusion models (the planned Tiny DDPM page) also step from noise to data.',
				zh: '欧拉法的每一步都只沿着这一步起点处的箭头走。t = 0 时，几乎每个箭头都指向你的形状的中间（十字），所以 {kf} 大步（h = {hf}，左）{pull}：它们最后平均离中间 {r4}；{k} 小步（右）{turn}最后离中间 {r50}；你自己的点离中间 {rd}。{verdict}扩散模型（见规划中的 Tiny DDPM 页面）也是一步步从噪声走到数据。',
				vars: (rt, at): Record<string, TmplVal> => {
					const a = runEnd({ at }, 'move');
					const b = runEnd({ at }, 'moveFew');
					const data = dataOf(rt.input);
					if (!a || !b || !data.n) return { k: NaN };
					const m = middleOf(data);
					const [r4, r50, rd] = [spreadAround(b.x.data, b.x.shape[0], m), spreadAround(a.x.data, a.x.shape[0], m), spreadAround(data.xy, data.n, m)];
					const inward = r4 < 0.9 * r50 && rd > 0.05;
					const [d4, d50] = [fmt3(b.dist), fmt3(a.dist)];
					return {
						k: a.K, kf: b.K, hf: short(1 / b.K), r4, r50, rd,
						pull: pick(inward, L('pull the points inward', '把点往里拉'), L('land about where small steps do', '落点和小步差不多')),
						turn: pick(inward && bentShare(at, a.K) > 0.5, L(', which follow the arrows as they turn', '跟着箭头转弯，'), L('', '')),
						verdict: [
							L(`That blurs your shape: on average ${d4} from the data, against ${d50}.`, `这让形状变模糊了：平均离数据 ${d4}，而 ${a.K} 步时是 ${d50}。`),
							L(`They still land about as close to the data (${d4} against ${d50}), only nearer the middle.`, `它们离数据还是差不多近（${d4} 对 ${d50}），只是更靠近中间。`),
							L(`Both land about as close to the data (${d4} against ${d50}).`, `两者离数据差不多近（${d4} 对 ${d50}）。`)
						][b.dist > 1.1 * a.dist ? 0 : inward ? 1 : 2]
					};
				}
			},
			look: L('Left: a few big steps; right: many small ones. Compare each pink cloud with the faint green shape and with the cross in the middle.', '左边是几大步，右边是许多小步。把每团粉色的点和淡绿色的形状、和中间的十字比一比。')
		}
	],
	// inputs every step must survive, besides the adapter's edge fixtures (no points, one point, one class)
	fixtures: [
		{
			id: 'painted',
			label: L('A few painted points', '几个手画的点'),
			raw: { points: [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6].flatMap((x, i) => [{ x, y: 0.5 - 0.1 * i }, { x, y: -0.5 + 0.1 * i }]) }
		}
	]
};

export default lesson;
