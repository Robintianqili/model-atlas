// tiny-tcn: the guided lesson. Eleven steps on the learner's series, every number live:
// series -> changes -> causal conv (init) -> reach (predict) -> the tree (init) -> residual block (init) -> head and
// forecast (init) -> training on every position at once -> ARIMA-lite -> the comparison (task) -> limits: the long square.
// Steps 3 to 7 use the starting weights (source init) and follow one output, at time t* = split − 2 (look.ts).
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { SERIES_PRESETS, seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';
import { paren } from '../../atlas/core/format.ts';
import { dataOf, DILATIONS, LONG_PERIOD, RF } from './data.ts';
import { CH } from './model.ts';
import { tcn } from './trainer.ts';
import { headAt, longStretch, nodeAt, nudged, paramsAt, tStar } from './look.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
type Kind = 'data' | 'model' | 'truth' | 'error';
const vec = (a: ArrayLike<number>): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
/** A curve1d line from time `from` on. */
const line = (y: ArrayLike<number>, from: number, kind: Kind, label: I18n) => ({ y: vec(y), x: vec(Array.from({ length: y.length }, (_, i) => from + i)), kind, label });
const M = (at: At) => (at.frame?.metrics ?? {}) as Record<string, number>;
const D = (rt: ExplainerRuntime) => dataOf(rt.input);
/** A value that is 0 up to rounding prints as 0, not as −2.45e−16. */
const tiny = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);
const TRAIN = I('training part', '训练部分');
const HELD = I('held-out part (true values)', '留出部分（真实值）');
const HELD_BOX = { en: 'held out', zh: '留出' };
const NONE = I('', '');
/** vars() before the step's source has run: no values (the narration then leaves its numbers out) */
const NOVARS: Record<string, TmplVal> = {};
const tree = (mode: string) => ({ key: 'tiny-tcn/tree', props: () => ({ mode }) });
const LAYER = (l: number) => I(`layer ${l + 1}`, `第 ${l + 1} 层`);
/** The held-out part with the forecasts: the last 2·RF training values, the truth, and the given lines. */
const heldOut = (rt: ExplainerRuntime, lines: { y: ArrayLike<number>; kind: Kind; label: I18n }[]) => {
	const d = D(rt);
	const from = Math.max(0, d.split - 2 * RF);
	const last = d.y[d.split - 1];
	return { series: [line(d.y.subarray(from, d.split), from, 'data', TRAIN), line(d.y.subarray(d.split - 1), d.split - 1, 'truth', HELD), ...lines.map((l) => line([last, ...Array.from(l.y)], d.split - 1, l.kind, l.label))], windows: [{ from: d.split, to: d.n - 1, label: HELD_BOX }], points: null };
};

/** Step 9's third sentence: what the chosen order leaves (p = 0 has no AR part to call a convolution). */
const arPart = (p: number | undefined, q: number | undefined) =>
	p === undefined ? ''
	: p > 0 ? I('Its AR part is one causal convolution of width p, without dilation or ReLU.', '它的 AR 部分就是一个宽度为 p 的因果卷积，没有空洞，也没有 ReLU。')
	: q ? I('Here p = 0: only μ and the MA part are left.', '这里 p = 0：只剩下 μ 和 MA 部分。')
	: I('Here p = q = 0: every forecast change is just μ.', '这里 p = q = 0：每个预测的变化量都只是 μ。');

const lesson: LessonDef = {
	title: I('A tiny TCN, step by step', '一步步看小型 TCN'),
	summary: I(
		'A temporal convolutional network learns your series in the browser: causal convolutions, dilations and the receptive field, the tree behind one forecast, residual blocks, training on every position at once, an ARIMA baseline and a pattern longer than the network can see.',
		'一个时间卷积网络在浏览器里学习你的序列：因果卷积、空洞率与感受野、一个预测背后的节点树、残差块、在所有位置上同时训练、ARIMA 基准，以及一种比网络视野更长的规律。'
	),
	steps: [
		{
			id: 'series',
			source: 'arima',
			at: { phase: 'series' },
			view: { key: 'curve1d', props: (rt) => {
				const d = D(rt);
				return { series: [line(d.y.subarray(0, d.split), 0, 'data', TRAIN), line(d.y.subarray(d.split - 1), d.split - 1, 'truth', HELD)], windows: [{ from: d.split, to: d.n - 1, label: HELD_BOX }], points: null };
			} },
			provenance: 'live',
			terms: ['time-series', 'dataset-split'],
			title: I('Your series', '你的序列'),
			narration: {
				en: 'This is a time series: {n} numbers in time order. A forecaster learns from the past to guess what comes next. To check it fairly, the last {hold} values are held out (shaded): nothing trains on them, and the forecasts are compared with them at the end.{fallback}',
				zh: '这是一条时间序列：按时间排好的 {n} 个数。预测模型从过去学习，去猜接下来会是什么。为了公平地检验，最后 {hold} 个值被留出（涂色部分）：任何模型都不拿它们来训练，最后再拿预测和它们比较。{fallback}',
				vars: (rt) => {
					const d = D(rt);
					return { n: d.n, hold: d.hold, fallback: pick(d.fallback, I(` Your own series has only ${d.given} value${d.given === 1 ? '' : 's'}, too few to train and check, so every step uses the sine wave instead.`, `你自己的序列只有 ${d.given} 个值，不够训练和检验，所以各步改用正弦波。`), NONE) };
				}
			},
			look: I('The plain line is what the models may learn from; the dashed part is kept for the final check.', '实线是模型可以学习的部分；虚线部分留到最后检验用。')
		},
		{
			id: 'changes',
			source: 'arima',
			at: { phase: 'series' },
			view: { key: 'curve1d', props: (rt) => {
				const d = D(rt);
				return { series: [line(d.c.subarray(1, d.split), 1, 'data', I('change c_t = y_t − y_(t−1)', '变化量 c_t = y_t − y_(t−1)'))], points: null, yLabel: I('change', '变化量'), xLabel: I('time t', '时间 t') };
			} },
			provenance: 'live',
			terms: ['differencing'],
			title: I('The network reads changes', '网络读的是变化量'),
			narration: {
				en: 'The network does not read the values themselves but their changes: the change at time t (times count from 0) is c_t = y_t − y_(t−1). The last change of the training part is {y1} − {y0} = {c}. The network answers with a change too, which is added to the last value to make the forecast. ARIMA, the baseline of step 9, calls this differencing, its d = 1.',
				zh: '网络读的不是数值本身，而是它们的变化量：时刻 t（时刻从 0 数起）的变化量是 c_t = y_t − y_(t−1)。训练部分的最后一个变化量是 {y1} − {y0} = {c}。网络给出的答案也是一个变化量，把它加到最后一个值上，就得到预测。第 9 步的基准 ARIMA 把这叫作差分，即它的 d = 1。',
				vars: (rt) => {
					const d = D(rt);
					return { y1: tiny(d.y[d.split - 1]), y0: paren(tiny(d.y[d.split - 2])), c: tiny(d.c[d.split - 1]) };
				}
			},
			look: I('A rising stretch of the series is a stretch of positive changes; a straight trend would be a flat line here.', '序列上升的一段就是一段正的变化量；一条直线趋势在这里会变成一条平线。')
		},
		{
			id: 'causal',
			source: 'init',
			at: 'first',
			view: tree('causal'),
			provenance: 'live',
			terms: ['tcn', 'causal-convolution', 'kernel', 'relu', 'padding'],
			title: I('Layer 1: a causal convolution', '第 1 层：因果卷积'),
			narration: {
				en: 'Layer 1 is a convolution with a kernel of 2: for each of its {CH} features, the output at time t multiplies the change at t − 1 and the change at t by two weights and adds a bias. For feature {ch} (counting from 0) at time {t}: {w0} × {x0} + {w1} × {x1} + {b} = {pre}, and the ReLU keeps {h}. Zeros are padded only at the start, before time 0, so no output ever reads a later change: the convolution is causal.',
				zh: '第 1 层是卷积核为 2 的卷积：它有 {CH} 个特征，每个特征在时刻 t 的输出，都是把 t − 1 和 t 两个时刻的变化量各乘一个权重再相加，最后加上偏置。时刻 {t} 的特征 {ch}（从 0 数起）：{w0} × {x0} + {w1} × {x1} + {b} = {pre}，ReLU 之后是 {h}。只在序列开头（时刻 0 之前）补零，所以任何输出都读不到更晚的变化量：这个卷积是因果的。',
				vars: (rt, at) => {
					const n = nodeAt(at, rt.input, 0);
					if (!n) return NOVARS;
					return { CH, ch: n.ch, t: n.t, w0: n.terms[0].w, x0: paren(tiny(n.terms[0].x)), w1: paren(n.terms[1].w), x1: paren(tiny(n.terms[1].x)), b: paren(n.bias), pre: n.pre, h: n.h };
				}
			},
			look: I('These weights are still random (training comes in step 8). In the stage, the two changes this node reads are joined to it; every other node of the row is computed the same way at its own time.', '这些权重还是随机的（第 8 步才训练）。图上把这个节点读的两个变化量连到了它上面；这一行的其他节点在各自的时刻用同样的方法算出来。')
		},
		{
			id: 'reach',
			source: 'init',
			at: 'first',
			view: tree('reach'),
			provenance: 'live',
			terms: ['dilation', 'receptive-field'],
			title: I('Dilations: how far back?', '空洞率：能看多远？'),
			predict: {
				id: 'tiny-tcn-reach',
				question: I('The output in the head row of the stage sits on layer 4. How many changes does it read in all?', '图中“输出头”那一行的输出位于第 4 层之上。它一共读了多少个变化量？'),
				options: [I('4, one per layer', '4 个，每层一个'), I('8', '8 个'), I('16', '16 个'), I('Every change before it', '它之前的全部变化量')],
				answer: 2,
				explain: I('16. Each layer doubles the reach: layer 1 reads 2 changes, layer 2 reaches 2 more back, layer 3 another 4, layer 4 another 8: 1 + 1 + 2 + 4 + 8 = 16 changes, from t − 15 to t. Nothing older can reach this output.', '16 个。每一层都把视野翻一倍：第 1 层读 2 个变化量，第 2 层再往回 2 个，第 3 层再往回 4 个，第 4 层再往回 8 个：1 + 1 + 2 + 4 + 8 = 16 个变化量，从 t − 15 到 t。更早的都到不了这个输出。')
			},
			narration: {
				en: 'Layers 2 to 4 also read two inputs each, but further apart: a gap of {d2} in layer 2, {d3} in layer 3 and {d4} in layer 4. This gap is the dilation. Follow the lines down from the output at time {t}: every node joins the two nodes it reads in the layer below.',
				zh: '第 2 到第 4 层也是每次读两个输入，但两个输入隔得更远：第 2 层隔 {d2}，第 3 层隔 {d3}，第 4 层隔 {d4}。这个间隔叫作空洞率（也叫膨胀率）。从时刻 {t} 的输出沿着连线往下走：每个节点都连着它在下一层读的两个节点。',
				vars: (rt) => ({ d2: DILATIONS[1], d3: DILATIONS[2], d4: DILATIONS[3], t: tStar(D(rt)) })
			},
			look: I('Count the changes at the bottom that some path reaches; the faint nodes are computed too, but for other outputs.', '数一数底部有路径连到的变化量；浅色的节点也会被计算，只是属于别的输出。')
		},
		{
			id: 'tree',
			source: 'init',
			at: 'first',
			view: tree('tree'),
			provenance: 'live',
			terms: ['receptive-field'],
			title: I('The tree behind one forecast', '一个预测背后的节点树'),
			narration: {
				en: 'This is the receptive field of the output at time {t}: the {RF} changes from time {from} to {t}, then {n1} nodes of layer 1, {n2} of layer 2, {n3} of layer 3 and one of layer 4. Add 1 to the newest change the network reads, at time {t}, and the forecast moves by {moved}. Add 1 instead to the change just outside the tree, at time {outside}, one step before its oldest change, and the forecast stays exactly {f}: no path leads from there.',
				zh: '这就是时刻 {t} 那个输出的感受野：从时刻 {from} 到 {t} 的 {RF} 个变化量，往上是第 1 层的 {n1} 个节点、第 2 层的 {n2} 个、第 3 层的 {n3} 个和第 4 层的 1 个。给网络读的最新变化量（时刻 {t}）加 1，预测就变动 {moved}。改为给紧挨在树外的变化量加 1，即比树里最早的那个还早一步的时刻 {outside}，预测仍然正好是 {f}：那里没有路径通上来。',
				vars: (rt, at) => {
					const P = paramsAt(at);
					if (!P) return NOVARS;
					const d = D(rt);
					const t = tStar(d);
					const f = headAt(P, d).forecast;
					return { t, RF, from: t - RF + 1, n1: RF / 2, n2: RF / 4, n3: RF / 8, moved: nudged(P, d, 0) - f, outside: t - RF, f };
				}
			},
			look: I('Pick any node of the tree in the stage to see its numbers: the inputs it reads, its weights and its sum.', '在图上点树里的任意一个节点，就能看到它的数：它读的输入、它的权重和它的和。')
		},
		{
			id: 'residual',
			source: 'init',
			at: 'first',
			view: tree('residual'),
			provenance: 'live',
			terms: ['residual-connection'],
			title: I('Residual blocks', '残差块'),
			narration: {
				en: 'Layers 2 to 4 are residual blocks: each adds what its convolution and ReLU make to what came in, h = skip + ReLU(conv). In layer 4 at time {t}, feature {ch} (counting from 0) gets {skip} + {a} = {h}. Where the ReLU gives 0, here for {zeros} of the {CH} features, the block passes its input on unchanged.',
				zh: '第 2 到第 4 层是残差块：每块把卷积和 ReLU 算出的东西加到输入上，h = 跳连 + ReLU(卷积)。在第 4 层、时刻 {t}，特征 {ch}（从 0 数起）得到 {skip} + {a} = {h}。凡是 ReLU 给出 0 的地方（这里 {CH} 个特征里有 {zeros} 个），这一块就把输入原样传下去。',
				vars: (rt, at) => {
					const n = nodeAt(at, rt.input, 3);
					if (!n) return NOVARS;
					let zeros = 0;
					for (let o = 0; o < CH; o++) if (n.pl.a[3][o * n.pl.T + n.t] === 0) zeros++;
					return { t: n.t, ch: n.ch, skip: n.skip, a: n.a, h: n.h, zeros, CH };
				}
			},
			look: I('Pick other features with the buttons in the stage: wherever the ReLU gives 0, the block’s output equals its input exactly, so a block only has to learn a correction to what it is given.', '用图上的按钮换别的特征看看：凡是 ReLU 给出 0 的地方，这一块的输出就正好等于输入，所以每一块只需要学一个对输入的修正量。')
		},
		{
			id: 'head',
			source: 'init',
			at: 'first',
			view: tree('head'),
			provenance: 'live',
			terms: ['head', 'weight', 'bias'],
			title: I('The head: one forecast', '输出头：一个预测'),
			narration: {
				en: 'The head turns the {CH} features of layer 4 at time {t} into one number, the predicted next change: each feature times its weight, added up ({s}), plus the bias {b}, gives {change}. The forecast is the last value plus that change: {last} + {plus} = {forecast}. The true value is {truth}, so these random starting weights miss by {miss}.',
				zh: '输出头把第 4 层在时刻 {t} 的 {CH} 个特征变成一个数，即预测的下一个变化量：每个特征乘上自己的权重，加起来（{s}），再加上偏置 {b}，得到 {change}。预测值就是最后一个值加上这个变化量：{last} + {plus} = {forecast}。真实值是 {truth}，所以这些随机的初始权重差了 {miss}。',
				vars: (rt, at) => {
					const P = paramsAt(at);
					if (!P) return NOVARS;
					const h = headAt(P, D(rt));
					return { CH, t: h.t, s: h.sum, b: h.bias, change: h.change, last: tiny(h.last), plus: paren(h.change), forecast: h.forecast, truth: tiny(h.truth), miss: Math.abs(h.forecast - h.truth) };
				}
			},
			look: I('The head is a 1×1 convolution: the same weights turn every time step’s features into that step’s forecast.', '输出头是一个 1×1 卷积：同一组权重把每个时间步的特征变成那一步的预测。')
		},
		{
			id: 'train',
			source: 'train',
			at: 'last',
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'] },
			view: { key: 'loss-curve', props: () => ({ metrics: ['loss'], labels: { loss: I('mean squared error, all training positions', '均方误差（全部训练位置）') }, log: true }) },
			provenance: 'live',
			terms: ['loss', 'mse', 'lstm', 'adam', 'learning-rate'],
			title: I('Train every position at once', '所有位置同时训练'),
			narration: {
				en: 'One pass over the training part gives the outputs at all {count} positions whose {RF} changes and next change are known, and the loss is their mean squared error. An LSTM would read the series one step at a time; the TCN computes each layer for all times together. {steps} Adam steps (learning rate {lr}) changed all {params} weights and biases: the loss fell from {mse0} to {mse1}, and the forecast of step 7 is now {after}, off by {missAfter} from the true {truth}.',
				zh: '把训练部分整个过一遍，就得到全部 {count} 个位置的输出（这些位置的 {RF} 个变化量和下一个变化量都已知），损失就是它们的均方误差。LSTM 要一步一步地读序列；TCN 则对所有时刻一起算每一层。{steps} 步 Adam（学习率 {lr}）改动了全部 {params} 个权重和偏置：损失从 {mse0} 降到 {mse1}，第 7 步那个预测现在是 {after}，和真实值 {truth} 差 {missAfter}。',
				vars: (rt, at) => {
					const d = D(rt);
					const P = paramsAt(at);
					const tape = at.tape!;
					const h = P ? headAt(P, d) : null;
					return { RF, count: d.count, steps: at.frame?.step ?? NaN, lr: String(tcn.defaults.lr), params: Object.values(P ?? {}).reduce((s, t) => s + t.data.length, 0), mse0: tape.frame(0).metrics?.mse ?? NaN, mse1: M(at).mse, after: h?.forecast ?? NaN, truth: tiny(h?.truth ?? NaN), missAfter: h ? Math.abs(h.forecast - h.truth) : NaN };
				}
			},
			look: I('The curve is drawn on a log scale, so equal distances down mean equal factors less error.', '曲线画在对数刻度上，所以往下走同样的距离，误差就缩小同样的倍数。')
		},
		{
			id: 'arima',
			source: 'arima',
			at: { phase: 'forecast' },
			view: { key: 'curve1d', props: (rt, at) => heldOut(rt, [{ y: (at.state as { path?: Tensor } | undefined)?.path?.data ?? [], kind: 'error', label: I('ARIMA forecast', 'ARIMA 的预测') }]) },
			provenance: 'live',
			terms: ['arima', 'aic', 'baseline'],
			title: I('A classic baseline: ARIMA', '经典基准：ARIMA'),
			narration: {
				en: 'ARIMA reads the same changes and predicts each one as a drift μ plus fixed coefficients times the last p changes (the AR part) and times its last q forecast errors (the MA part): all linear. Least squares fits every order with p up to {pMax} and q up to {qMax}; AIC picks ARIMA({p}, 1, {q}) for your series, {k} in all. {ar} Over the {hold} held-out values its forecast misses by {mae} on average.',
				zh: 'ARIMA 读同样的变化量，把每个变化量预测为：漂移 μ，加上最近 p 个变化量各乘一个固定系数（AR 部分），再加上它最近 q 个预测误差各乘一个固定系数（MA 部分），全是线性的。最小二乘法拟合 p 不超过 {pMax}、q 不超过 {qMax} 的每一种阶数；AIC 为你的序列选出 ARIMA({p}, 1, {q})，一共 {k}。{ar}在 {hold} 个留出值上，它的预测平均差 {mae}。',
				vars: (rt, at) => {
					const tape = at.tape!;
					let fit: Record<string, number> = {};
					for (let i = 0; i <= at.index; i++) if (tape.frame(i).phase === 'fit') fit = tape.frame(i).metrics ?? {};
					return { pMax: fit.pMax, qMax: fit.qMax, p: fit.p, q: fit.q, k: plural(fit.k, { one: '{n} coefficient', many: '{n} coefficients', zh: '{n} 个系数' }), hold: M(at).hold, mae: tiny(M(at).mae), ar: arPart(fit.p, fit.q) };
				}
			},
			look: I('Its forecast starts where the training part ends and never sees the dashed truth.', '它的预测从训练部分结束的地方开始，从没见过虚线的真实值。')
		},
		{
			id: 'compare',
			source: 'forecast',
			at: { phase: 'compare' },
			view: { key: 'curve1d', props: (rt, at) => {
				const s = at.state as { tcn?: Tensor; arima?: Tensor } | undefined;
				return heldOut(rt, [{ y: s?.tcn?.data ?? [], kind: 'model', label: I('TCN forecast', 'TCN 的预测') }, { y: s?.arima?.data ?? [], kind: 'error', label: I('ARIMA forecast', 'ARIMA 的预测') }]);
			} },
			provenance: 'live',
			terms: ['rollout', 'baseline'],
			title: I('TCN or ARIMA?', 'TCN 还是 ARIMA？'),
			narration: {
				en: 'The trained TCN forecasts the {hold} held-out values one after another, appending each forecast change to the changes it reads (a rollout), as ARIMA does. Average miss: the TCN {tcn}, ARIMA {arima}, and repeating the last value {naive}. {verdict}',
				zh: '训练好的 TCN 一个接一个地预测 {hold} 个留出值，把每个预测的变化量接到它读的变化量后面（滚动预测），ARIMA 也是这样做的。平均误差：TCN {tcn}，ARIMA {arima}，重复最后一个值 {naive}。{verdict}',
				vars: (_rt, at) => {
					const m = M(at);
					const [a, b] = [tiny(m.tcnMae), tiny(m.arimaMae)];
					return { hold: m.hold, tcn: a, arima: b, naive: tiny(m.naiveMae), verdict: !(Number.isFinite(a) && Number.isFinite(b)) ? NONE : pick(a < b, I('On this series the TCN wins.', '在这条序列上，TCN 赢了。'), I('On this series ARIMA wins: a few linear coefficients capture it better than the network.', '在这条序列上，ARIMA 赢了：几个线性系数比网络抓得更准。')) };
				}
			},
			look: I('Both lines start where the training part ends; neither sees the dashed truth.', '两条线都从训练部分结束的地方开始；谁都没见过虚线的真实值。'),
			task: {
				prompt: I('Find a series on which ARIMA beats the TCN: pick another preset (try the sine wave) or redraw the curve.', '找一条让 ARIMA 胜过 TCN 的序列：换一个预设（试试正弦波），或者重画曲线。'),
				done: (_rt, at) => tiny(M(at).arimaMae) < tiny(M(at).tcnMae),
				waiting: I('Not yet: ARIMA has not beaten the TCN on this series.', '还没有：在这条序列上，ARIMA 还没胜过 TCN。'),
				doneText: I('Done: on this series ARIMA misses less on average. On the sine and the trend it finds the exact rule; the network only gets close.', '完成：在这条序列上，ARIMA 的平均误差更小。在正弦波和直线趋势上，它能找到精确的规则，网络只能接近。')
			},
			linger: 12000
		},
		{
			id: 'limits',
			source: 'long',
			at: 'last',
			view: { key: 'curve1d', props: (_rt, at) => {
				const P = paramsAt(at);
				if (!P) return { series: [], points: null };
				const s = longStretch(P);
				const from = Math.max(0, s.b - 2 * RF - 3);
				return {
					series: [line(s.d.y.subarray(from, s.d.split), from, 'data', I(`square wave, period ${LONG_PERIOD}`, `方波，周期 ${LONG_PERIOD}`)), line(s.forecasts.subarray(from - RF), from + 1, 'model', I('TCN forecast, one step ahead', 'TCN 的单步预测')), { y: vec([s.f, s.jump]), x: vec([s.b + 1, s.b + 1]), kind: 'error', label: I('the miss at the jump', '跳变处的误差') }],
					windows: [{ from: s.b - RF + 1, to: s.b, label: I(`${RF} changes, all 0`, `${RF} 个变化量，全为 0`) }],
					points: null
				};
			} },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['receptive-field'],
			title: I('Limits: longer than the reach', '局限：比视野更长'),
			narration: {
				en: 'A TCN cannot use what lies beyond its reach. This square wave, trained on like your series, stays {half} steps at each level, longer than the {RF} changes the network reads. From time {a} to {b} the last {RF} changes are all 0: the same numbers, so the same forecast every time, {f}. The wave stays at {stay} until {b}, then jumps to {jump}: a miss of {miss}, and the network misses each of the {jumps} jumps it trained on.',
				zh: 'TCN 用不上视野之外的东西。网络像对你的序列那样在这个方波上训练过；方波在每个高度上停留 {half} 步，比网络读的 {RF} 个变化量还长。从时刻 {a} 到 {b}，最近 {RF} 个变化量全是 0：数一样，预测也就每次都一样，都是 {f}。方波一直停在 {stay}，直到 {b}，然后跳到 {jump}：差了 {miss}；它训练时见过的 {jumps} 次跳变，网络每一次都错过了。',
				vars: (_rt, at) => {
					const P = paramsAt(at);
					if (!P) return NOVARS;
					const s = longStretch(P);
					return { half: LONG_PERIOD / 2, RF, a: s.a, b: s.b, f: s.f, stay: s.stay, jump: s.jump, miss: s.miss, jumps: s.jumps };
				}
			},
			look: {
				en: 'More data or training cannot fix it; a longer reach can. A fifth layer with dilation {d5} would read {rf5} changes, enough to see the last jump {half} steps back.',
				zh: '更多的数据或训练都解决不了；更长的视野可以。加第五层、空洞率为 {d5}，就能读 {rf5} 个变化量，足以看到 {half} 步之前的上一次跳变。',
				vars: () => ({ d5: 2 * DILATIONS[3], rf5: RF + 2 * DILATIONS[3], half: LONG_PERIOD / 2 })
			}
		}
	],
	fixtures: [
		{ id: 'drawn', label: I('A hand-drawn curve', '一条手画的曲线'), raw: { y: Array.from({ length: 80 }, (_, i) => Math.round(100 * (0.8 * Math.sin(i / 5) + (i % 20 < 10 ? 0.3 : -0.3))) / 100) } },
		// the presets the input widget offers (options.presets in the manifest), so lint and the claims run on each
		...SERIES_PRESETS.filter((p) => ['sine', 'square', 'trend', 'randomWalk'].includes(p.id)).map((p) => ({ id: p.id, label: p.label, raw: seriesPresetRaw(p.id) }))
	]
};

export default lesson;
