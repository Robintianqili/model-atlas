// tinyforecaster: the guided lesson (design §7 outline). Ten steps on the learner's series, every number live:
// series -> windows (predict) -> gates -> the head -> last value + change -> loss -> training -> rollout (predict) ->
// the baseline (task) -> limits: random walks. Steps 3-6 use the starting weights (sources init / net0), steps 7-9 the
// trained ones (train, forecast), step 10 three fixed random walks (walksSaved: the walks process, recorded ahead of time).
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor, Trace, TrainState } from '../../atlas/types/index.ts';
import { num, pick, plural } from '../../atlas/lesson/say.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { dataOf, L, type Data } from './data.ts';
import { forecastAll, HIDDEN, netOf } from './model.ts';
import { lstm, trainingSet } from './trainer.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
type Kind = 'data' | 'model' | 'truth' | 'error';
const vec = (a: ArrayLike<number>): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
/** A curve1d line from position `from` on. */
const line = (y: ArrayLike<number>, from: number, kind: Kind, label: I18n) => ({ y: vec(y), x: vec(Array.from({ length: y.length }, (_, i) => from + i)), kind, label });
const M = (at: At) => (at.frame?.metrics ?? {}) as Record<string, number>;
const D = (rt: ExplainerRuntime) => dataOf(rt.input);
/** A value that is 0 up to rounding (sin 2π, a sine copied one period back) prints as 0, not as −2.45e−16. */
const tiny = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);
/** "0.412 − 0.0830" / "0.412 + 0.0830" with the narration's number format. */
const sum = (a: number, b: number): I18n | number => {
	const fa = num(a);
	const fb = num(Math.abs(b));
	if (typeof fa === 'number' || typeof fb === 'number') return NaN;
	const op = b < 0 ? '−' : '+';
	return { en: `${fa.en} ${op} ${fb.en}`, zh: `${fa.zh} ${op} ${fb.zh}` };
};

/** Two numbers printed with as many significant digits (3 to 6) as it takes to tell them apart. */
function apart(a: number, b: number): [I18n | number, I18n | number] {
	for (const f of ['n3', 'n4', 'n5', 'n6']) {
		const x = num(a, f);
		const y = num(b, f);
		if (typeof x === 'number' || typeof y === 'number' || x.en !== y.en || f === 'n6') return [x, y];
	}
	return [num(a), num(b)];
}

const TRAIN = I('training part', '训练部分');
const HELD = I('held-out part (true values)', '留出部分（真实值）');
const HELD_BOX = { en: 'held out', zh: '留出' };
const NONE = I('', '');
/** "walk 1", "walks 2 and 3" / "第 1 条", "第 2、3 条" (walks count from 1, as the scrubber does). */
const walkList = (a: number[]): I18n =>
	a.length === 1 ? I(`walk ${a[0]}`, `第 ${a[0]} 条`) : I(`walks ${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`, `第 ${a.join('、')} 条`);

/** The last training window (the one net0 traces): its values and its target, the last value of the training part. */
const lastWindow = (d: Data) => ({ x: d.y.slice(d.windows - 1, d.windows - 1 + L), target: d.y[d.split - 1] });
/** The forecast of the last training window by recorded parameters (memoized: vars and props ask more than once). */
const memo = new WeakMap<object, number>();
function forecastOf(params: Record<string, Tensor> | undefined, d: Data): number {
	if (!params) return NaN;
	const hit = memo.get(params);
	if (hit !== undefined) return hit;
	const v = forecastAll(netOf(params, createRng(1)), lastWindow(d).x, 1)[0];
	memo.set(params, v);
	return v;
}
/** The forecasts of every training window by recorded parameters (memoized: step 6's chart and look share them). */
const all = new WeakMap<object, Float64Array>();
function forecasts(params: Record<string, Tensor>, d: Data): Float64Array {
	let f = all.get(params);
	if (!f) {
		const s = trainingSet(d);
		all.set(params, (f = forecastAll(netOf(params, createRng(1)), s.x, s.count)));
	}
	return f;
}
/** The largest |predicted change| (forecast − last value) over the training windows, for recorded parameters. */
export function startChange(d: Data, params: Record<string, Tensor> | undefined): number {
	if (!params) return NaN;
	const s = trainingSet(d);
	return forecasts(params, d).reduce((m, v, k) => Math.max(m, Math.abs(v - s.x[k * L + L - 1])), 0);
}
const paramsAt = (at: At) => (at.state as Partial<TrainState> | undefined)?.params;
const trace = (rt: ExplainerRuntime) => rt.sources.net0?.trace as Trace | undefined;
const val = (rt: ExplainerRuntime, id: string) => trace(rt)?.t[id]?.data[0] ?? NaN;
/** The true next value after the last training window (the dot of step 2's last window, the target of step 5). */
const target = (rt: ExplainerRuntime) => tiny(lastWindow(D(rt)).target);

const lesson: LessonDef = {
	title: I('TinyForecaster, step by step', '一步步看 TinyForecaster'),
	summary: I(
		'An LSTM learns your series in the browser: windows, gates, one forecast worked out, training, forecasting the held-out part, a baseline to beat and a series no model can forecast.',
		'一个 LSTM 在浏览器里学习你的序列：窗口、门、算出一个预测、训练、预测留出的部分、要打败的基准，以及一种任何模型都预测不了的序列。'
	),
	steps: [
		{
			id: 'series',
			source: 'windows',
			at: { phase: 'series' },
			view: {
				key: 'curve1d',
				props: (rt) => {
					const d = D(rt);
					return { series: [line(d.y.subarray(0, d.split), 0, 'data', TRAIN), line(d.y.subarray(d.split - 1), d.split - 1, 'truth', HELD)], windows: [{ from: d.split, to: d.n - 1, label: HELD_BOX }], points: null };
				}
			},
			provenance: 'live',
			terms: ['time-series', 'dataset-split'],
			title: I('Your series', '你的序列'),
			narration: {
				en: 'This is a time series: {n} numbers in time order, one per time step. A forecaster learns from the past to guess what comes next. To check it fairly, the last {hold} values are held out (shaded): it never trains on them, and its forecasts are compared with them at the end.{fallback}',
				zh: '这是一条时间序列：按时间排好的 {n} 个数，每个时间步一个。预测模型从过去学习，去猜接下来会是什么。为了公平地检验它，最后 {hold} 个值被留出（涂色部分）：它从不在这些值上训练，最后再拿它的预测和这些值比较。{fallback}',
				vars: (rt) => {
					const d = D(rt);
					return { n: d.n, hold: d.hold, fallback: pick(d.fallback, I(` Your own series has only ${d.given} value${d.given === 1 ? '' : 's'}, too few for windows and a check, so every step uses the sine wave instead.`, `你自己的序列只有 ${d.given} 个值，不够切窗口和检验，所以各步改用正弦波。`), NONE) };
				}
			},
			look: I('The plain line is what the forecaster may learn from; the dashed part is kept for the final check.', '实线是预测模型可以学习的部分；虚线部分留到最后检验用。')
		},
		{
			id: 'windows',
			source: 'windows',
			at: { phase: 'window', step: 0 },
			play: { from: { phase: 'window', step: 0 }, to: { phase: 'window', last: true }, interval: 400, unit: I('window', '窗口') },
			view: {
				key: 'curve1d',
				props: (rt, at) => {
					const d = D(rt);
					const s = M(at).start ?? 0;
					return { series: [line(d.y.subarray(0, d.split), 0, 'data', TRAIN)], windows: [{ from: s, to: s + L - 1, label: I('window', '窗口') }], highlight: s + L, points: null };
				}
			},
			provenance: 'live',
			terms: ['forecast-window'],
			title: I('Cut it into windows', '切成窗口'),
			predict: {
				id: 'tinyforecaster-overlap',
				question: I('Two neighbouring windows: how many of their values are the same?', '相邻的两个窗口：它们有多少个值是相同的？'),
				options: [I('None: each value is used once', '没有：每个值只用一次'), I('23 of their 24 values', '24 个值里有 23 个'), I('All 24', '全部 24 个')],
				answer: 1,
				explain: I('23. The window slides one step at a time, so every value turns up in many windows, each time at a different place. That is how one short series gives many training examples.', '23 个。窗口每次只滑动一步，所以每个值会出现在很多个窗口里，每次的位置都不同。一条短短的序列就是这样变出许多训练样例的。')
			},
			narration: {
				en: 'A training example is a window of {L} values in a row (shaded) and the value right after it, its target (the dot): {target} for the first window. The model reads the window and must guess the target. The training part gives {count} such examples.',
				zh: '一个训练样例就是连续 {L} 个值组成的窗口（涂色部分），加上紧跟其后的那个值，也就是它的目标（圆点）：第一个窗口的目标是 {target}。模型读这个窗口，要猜出目标。训练部分一共能切出 {count} 个这样的样例。',
				vars: (_rt, at) => ({ L: M(at).L, target: tiny(M(at).target), count: M(at).count })
			},
			look: {
				en: 'Play slides the window one step at a time. The last one, window {last} (counting from 0), is the window steps 3 to 5 follow: its target, {next}, is the value the network will try to forecast.',
				zh: '播放时窗口每次滑动一步。最后一个窗口，即窗口 {last}（从 0 数起），就是第 3 到第 5 步要跟踪的窗口：它的目标 {next} 就是网络要去预测的值。',
				vars: (rt) => ({ last: D(rt).windows - 1, next: target(rt) })
			}
		},
		{
			id: 'gates',
			source: 'net0',
			at: { layer: 'lstm' },
			select: () => ({ layer: 'lstm', unit: [L - 1, 0] }),
			view: 'zoom.gate',
			provenance: 'live',
			terms: ['lstm', 'gate'],
			title: I('Inside the LSTM (starting weights)', 'LSTM 内部（初始权重）'),
			narration: {
				en: 'The network reads the last training window one value at a time, oldest first, each minus the last value (so the last one reads 0). An LSTM carries a memory c of {H} numbers from step to step; gates (numbers between 0 and 1) decide what to keep. At the last step, for memory number {j} (unit {j} on the stage), the forget gate {f} keeps {fPct} of the old value {cPrev}, and the input gate {i} adds that share of the proposed value {g}: c becomes {c}.',
				zh: '网络逐个读取最后一个训练窗口里的值，从最早的开始，每个值都先减去最后一个值（所以最后一个读到 0）。LSTM 把一份由 {H} 个数组成的记忆 c 从一步带到下一步，由门（介于 0 和 1 之间的数）决定保留什么。在最后一步，第 {j} 个记忆数（图上的单元 {j}）的遗忘门 {f} 保留旧值 {cPrev} 的 {fPct}，输入门 {i} 按这个比例加入这一步提议的新值 {g}：c 变成 {c}。',
				vars: (_rt, at) => {
					const v = at.unit?.values ?? {};
					const [cPrev, g] = apart(v.cPrev, v.g);
					return { H: HIDDEN, j: v.j, f: v.f, fPct: { v: v.f, fmt: 'pct' }, cPrev, i: v.i, g, c: v.c };
				}
			},
			look: {
				en: 'Then the output gate o = {o} lets a share of tanh(c) out as the output h = {h}: row 0 of the head’s table in step 4. These weights are still random: training will change every gate.',
				zh: '接着，输出门 o = {o} 让 tanh(c) 的一部分作为输出 h = {h} 送出去：它就是第 4 步输出头表格的第 0 行。这些权重还是随机的：训练会改变每一个门。',
				vars: (_rt, at) => ({ o: at.unit?.values.o ?? NaN, h: at.unit?.values.h ?? NaN })
			}
		},
		{
			id: 'head',
			source: 'net0',
			at: { layer: 'head' },
			select: () => ({ layer: 'head', unit: [0] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			terms: ['head', 'weight', 'bias'],
			title: I('The head: one number from h', '输出头：从 h 得到一个数'),
			narration: {
				en: 'A last layer, the head, turns the LSTM’s last output h ({H} numbers, one per memory number) into one number, the predicted change: each number times its own weight, all added, plus a bias. Here the {H} products add up to {s}, and the bias {b} makes {change}.',
				zh: '最后一层叫输出头，它把 LSTM 最后的输出 h（{H} 个数，每个记忆数一个）变成一个数，即预测的变化量：每个数乘上自己的权重，全部加起来，再加上偏置。这里 {H} 个乘积加起来是 {s}，再加上偏置 {b}，得到 {change}。',
				vars: (rt, at) => ({ H: HIDDEN, s: at.unit?.values.s ?? NaN, b: at.unit?.values.b ?? NaN, change: val(rt, 'head') })
			},
			look: {
				en: 'Each row of the table multiplies one number of h by its weight; scroll the table to see all {H} rows. The bars then add the sum and the bias.',
				zh: '表格的每一行是 h 的一个数乘上它的权重；滚动表格可以看到全部 {H} 行。右边的条形再把总和与偏置加起来。',
				vars: () => ({ H: HIDDEN })
			}
		},
		{
			id: 'forecast',
			source: 'net0',
			at: { layer: 'out' },
			select: () => ({ layer: 'out', unit: [0] }),
			view: 'zoom.elementwise',
			provenance: 'live',
			title: I('Last value + change = forecast', '最后一个值 + 变化量 = 预测'),
			narration: {
				en: 'The forecast is the window’s last value plus the predicted change: {sum} = {forecast}. Predicting the change, not the value itself, lets the network continue a series above every value it trained on (try the straight trend). The true next value is {next}, so these random starting weights miss by {miss}.',
				zh: '预测值就是窗口最后一个值加上预测的变化量：{sum} = {forecast}。网络预测的是变化量而不是值本身，所以它能把序列延续到比所有训练值都高的地方（试试直线趋势）。真实的下一个值是 {next}，所以这些随机的初始权重差了 {miss}。',
				vars: (rt) => ({ sum: sum(val(rt, 'lastv'), val(rt, 'head')), forecast: val(rt, 'out'), next: target(rt), miss: Math.abs(val(rt, 'out') - target(rt)) })
			},
			look: {
				en: 'a is the window’s last value, b the predicted change of step 4. The true next value, {next}, is the dot after window {last} in step 2.',
				zh: 'a 是窗口的最后一个值，b 是第 4 步预测的变化量。真实的下一个值 {next}，就是第 2 步里窗口 {last} 后面的那个圆点。',
				vars: (rt) => ({ next: target(rt), last: D(rt).windows - 1 })
			}
		},
		{
			id: 'loss',
			source: 'train',
			at: 'first',
			view: {
				key: 'curve1d',
				props: (rt, at) => {
					const d = D(rt);
					const p = paramsAt(at);
					const f = p ? forecasts(p, d) : new Float64Array(0);
					return { series: [line(d.y.subarray(0, d.split), 0, 'data', TRAIN), line(f, L, 'model', I('forecasts, starting weights', '预测（初始权重）'))], points: null };
				}
			},
			provenance: 'live',
			terms: ['loss', 'mse'],
			title: I('How wrong? The loss', '错得有多厉害？损失'),
			narration: {
				en: 'To learn, the network needs one number that says how wrong it is: the loss. Each miss is squared, so a miss twice as big counts four times as much (step 5’s window missed by {miss}, squared {sq}), and the squares are averaged over all {count} training windows: the mean squared error is {mse}. The chart shows the starting weights’ forecast for every training window.',
				zh: '要学习，网络需要一个数来说明它错得有多厉害：损失。每次的误差先平方，所以大一倍的误差算四倍（第 5 步那个窗口差了 {miss}，平方后是 {sq}），再对全部 {count} 个训练窗口求平均：均方误差是 {mse}。图上画的是初始权重对每个训练窗口的预测。',
				vars: (rt, at) => {
					const d = D(rt);
					const miss = Math.abs(forecastOf(paramsAt(at), d) - lastWindow(d).target);
					return { miss, sq: miss * miss, count: d.windows, mse: M(at).mse };
				}
			},
			look: {
				en: 'The starting forecasts follow the series one step late: the predicted change is never more than {most}, so each forecast stays close to its window’s last value.',
				zh: '初始权重的预测比序列晚一步：预测的变化量从不超过 {most}，所以每个预测都离窗口的最后一个值很近。',
				vars: (rt, at) => ({ most: startChange(D(rt), paramsAt(at)) })
			}
		},
		{
			id: 'train',
			source: 'train',
			at: 'last',
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'] },
			view: { key: 'loss-curve', props: () => ({ metrics: ['loss', 'mse'], labels: { loss: I('loss of the random windows', '随机窗口的损失'), mse: I('mean squared error, all windows', '均方误差（全部窗口）') } }) },
			provenance: 'live',
			terms: ['optimizer', 'learning-rate', 'batch'],
			title: I('Train it, live', '实时训练'),
			narration: {
				en: 'Training repeats one move {steps} times: pick {batch} windows at random, measure their loss, and nudge all {params} weights a little so it drops (the Adam optimizer, learning rate {lr}). The mean squared error over all windows fell from {mse0} to {mse1}. Step 5’s window is now forecast as {after}: off by {missAfter} from the true {next}, where the starting weights were off by {missBefore}.{worse}',
				zh: '训练把同一个动作重复 {steps} 次：随机挑 {batch} 个窗口，算出它们的损失，再把全部 {params} 个权重各挪一点，让损失变小（Adam 优化器，学习率 {lr}）。所有窗口上的均方误差从 {mse0} 降到了 {mse1}。第 5 步那个窗口现在的预测是 {after}：和真实值 {next} 差 {missAfter}，而初始权重差了 {missBefore}。{worse}',
				vars: (rt, at) => {
					const d = D(rt);
					const tape = at.tape!;
					const opt = { ...lstm.defaults, ...((tape as { opt?: Record<string, unknown> }).opt ?? {}) };
					const after = forecastOf(paramsAt(at), d);
					const missAfter = Math.abs(after - lastWindow(d).target);
					const missBefore = Math.abs(forecastOf((tape.state(0) as Partial<TrainState>).params, d) - lastWindow(d).target);
					return { steps: at.frame?.step ?? NaN, batch: opt.batch, params: Object.values(paramsAt(at) ?? {}).reduce((s, t) => s + t.data.length, 0), lr: String(opt.lr), mse0: tape.frame(0).metrics?.mse ?? NaN, mse1: M(at).mse, after, next: target(rt), missAfter, missBefore, worse: pick(missAfter > missBefore, I(' This one window got worse: training lowers the mean squared error over all windows, not every miss.', '这一个窗口反而变差了：训练降低的是全部窗口上的均方误差，而不是每一个误差。'), NONE) };
				}
			},
			look: I('The jagged line is the loss of each step’s random windows; the smooth one, measured every few steps, is the mean squared error over all windows.', '锯齿状的线是每一步那几个随机窗口的损失；平滑的线每隔几步测一次，是全部窗口上的均方误差。')
		},
		{
			id: 'rollout',
			source: 'forecast',
			at: { phase: 'predict', last: true },
			play: { from: { phase: 'start' }, to: { phase: 'predict', last: true }, phases: ['start', 'predict'], interval: 500, unit: I('forecast', '预测') },
			view: {
				key: 'curve1d',
				props: (rt, at) => {
					const d = D(rt);
					const path = (at.state as { path?: Tensor } | undefined)?.path?.data ?? [];
					const done = Array.from(path).filter((v) => Number.isFinite(v));
					return { series: [line(d.y.subarray(0, d.split), 0, 'data', TRAIN), line(d.y.subarray(d.split - 1), d.split - 1, 'truth', HELD), line([d.y[d.split - 1], ...done], d.split - 1, 'model', I('network forecast', '网络的预测'))], windows: [{ from: d.split, to: d.n - 1, label: HELD_BOX }], points: null };
				}
			},
			provenance: 'live',
			terms: ['rollout', 'autoregressive'],
			title: I('Forecast the held-out part', '预测留出的部分'),
			predict: {
				id: 'tinyforecaster-feedback',
				question: I('The first forecast reads the last 24 known values. To make the second forecast, the model again needs the 24 values before it. Where does the newest of them come from?', '第一个预测读的是最后 24 个已知值。要做第二个预测，模型同样需要它前面的 24 个值。其中最新的那个值从哪里来？'),
				options: [I('From the true series', '来自真实的序列'), I('From the model’s own first forecast', '来自模型自己的第一个预测'), I('It is left out', '把它空着')],
				answer: 1,
				explain: I('From its own forecast. The true held-out values are unknown when you forecast, so each forecast is appended to the history and read back as input: a rollout. A miss early on becomes part of the input of every later step, so errors can build up.', '来自它自己的预测。做预测时并不知道留出部分的真实值，所以每个预测都被接到历史后面、再当作输入读回去：这叫滚动预测。早先的一个误差会成为后面每一步输入的一部分，所以误差可能越积越大。')
			},
			narration: {
				en: 'Now the real test: starting where the training part ends, forecast all {hold} held-out values one after another, without seeing any of them. The first {q} forecasts miss the truth by {early} on average, the last {q} by {late}: {trend}',
				zh: '现在是真正的检验：从训练部分结束的地方开始，一个接一个地预测全部 {hold} 个留出的值，一个都不看。前 {q} 个预测和真实值平均差 {early}，最后 {q} 个差 {late}：{trend}',
				vars: (rt, at) => {
					const d = D(rt);
					const tape = at.tape!;
					const errs: number[] = [];
					for (let i = 0; i <= at.index; i++) if (tape.frame(i).phase === 'predict') errs.push(Math.abs(tape.frame(i).metrics!.err));
					const q = Math.max(1, Math.floor(d.hold / 3));
					const avg = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
					const e = avg(errs.slice(0, q));
					const l = errs.length >= d.hold ? avg(errs.slice(d.hold - q)) : NaN;
					const [early, late] = apart(e, l);
					const trend = l > 1.2 * e ? I('the misses grew along the way.', '误差一路变大了。') : l < e / 1.2 ? I('here the misses did not grow.', '这次误差没有变大。') : I('about the same size.', '大小差不多。');
					return { hold: d.hold, q, early, late, trend };
				}
			},
			look: I('The dashed line is the truth the model never saw; the “network forecast” line grows by one forecast at a time as you play.', '虚线是模型从没见过的真实值；“网络的预测”这条线在播放时一次长出一个预测。')
		},
		{
			id: 'baseline',
			source: 'forecast',
			at: { phase: 'naive' },
			view: {
				key: 'curve1d',
				props: (rt, at) => {
					const d = D(rt);
					const s = at.state as { path?: Tensor; naive?: Tensor } | undefined;
					const from = Math.max(0, d.split - 2 * L);
					const m = M(at).m;
					return {
						series: [line(d.y.subarray(from, d.split), from, 'data', TRAIN), line(d.y.subarray(d.split - 1), d.split - 1, 'truth', HELD), line([d.y[d.split - 1], ...(s?.path?.data ?? [])], d.split - 1, 'model', I('network', '网络')), line([d.y[d.split - 1], ...(s?.naive?.data ?? [])], d.split - 1, 'error', m === 1 ? I('repeat the last value', '重复最后一个值') : I(`repeat the last ${m} values`, `重复最后 ${m} 个值`))],
						windows: [{ from: d.split, to: d.n - 1, label: HELD_BOX }],
						points: null
					};
				}
			},
			provenance: 'live',
			terms: ['baseline'],
			title: I('Beat the baseline?', '能打败基准吗？'),
			narration: {
				en: 'Is the network worth it? Compare it with a baseline that learns nothing: repeat {lastM} over and over, as if the series came back {every} (the season length that worked best on the training part). On the held-out part this rule misses by {naive} on average, the network by {net}: {verdict}',
				zh: '这个网络值得用吗？拿它和一个什么都不学的基准比一比：把{lastM}一遍遍重复下去，就像这条序列{every}重来一次（这是在训练部分上效果最好的季节长度）。在留出部分上，这条规则平均差 {naive}，网络差 {net}：{verdict}',
				vars: (_rt, at) => {
					const m = M(at).m;
					const naive = tiny(M(at).naiveMae);
					const net = tiny(M(at).lstmMae);
					return { lastM: plural(m, { one: 'the last known value', many: 'the last {n} known values', zh: m === 1 ? '最后一个已知值' : '最后 {n} 个已知值' }), every: plural(m, { one: 'every step', many: 'every {n} steps', zh: m === 1 ? '每一步' : '每 {n} 步' }), naive, net, verdict: pick(net < naive, I('the network wins on this series.', '在这条序列上，网络赢了。'), I('the rule wins, so here the network was not worth training.', '规则赢了，所以在这条序列上，网络白训练了。')) };
				}
			},
			look: {
				en: 'Both forecasts start where the training part ends and never see the dashed truth. One stretch of held-out values can favour either side by luck: step {limits} tests that on random walks.{exact}',
				zh: '两条预测都从训练部分结束的地方开始，都没见过虚线的真实值。只看一段留出值，哪一方占优可能只是运气：第 {limits} 步会在随机游走上检验这一点。{exact}',
				vars: (_rt, at) => ({ limits: (lesson.steps as { id?: string }[]).findIndex((s) => s.id === 'walk') + 1, exact: pick(tiny(M(at).naiveMae) === 0, I(' Here the rule’s line lies exactly on the dashed truth.', '这里规则的那条线正好盖住了虚线的真实值。'), NONE) })
			},
			task: {
				prompt: I('Find a series on which the rule wins: pick another preset (try the sine wave) or redraw part of the curve.', '找一条让规则获胜的序列：换一个预设（试试正弦波），或者重画曲线的一部分。'),
				done: (_rt, at) => tiny(M(at).naiveMae) < tiny(M(at).lstmMae),
				waiting: I('Not yet: on this series the network still wins.', '还没有：在这条序列上，网络仍然赢。')
			},
			// under Play, time to read the new verdict once the task is done (the kit's 4 s is too short for a changed narration)
			linger: 12000
		},
		{
			id: 'walk',
			source: 'walksSaved',
			at: { phase: 'walk', last: true },
			play: { from: { phase: 'walk', step: 0 }, to: { phase: 'walk', last: true }, interval: 1500, unit: I('walk', '游走'), by: 'sub' },
			view: {
				key: 'curve1d',
				props: (_rt, at) => {
					const s = at.state as { series?: Tensor; one?: Tensor } | undefined;
					const y = s?.series?.data ?? new Float64Array(0);
					const split = M(at).split ?? y.length;
					const from = Math.max(0, split - L);
					return {
						series: [line(y.subarray(from, split), from, 'data', TRAIN), line(y.subarray(split - 1), split - 1, 'truth', HELD), line(s?.one?.data ?? [], split, 'model', I('network, one step ahead', '网络（单步预测）')), line(y.subarray(split - 1, y.length - 1), split, 'error', I('repeat the last value', '重复最后一个值'))],
						windows: [{ from: split, to: y.length - 1, label: HELD_BOX }],
						points: null
					};
				}
			},
			provenance: 'precomputed',
			kindOf: 'limits',
			terms: ['random-walk', 'overfitting'],
			title: I('Limits: a random walk', '局限：随机游走'),
			narration: {
				en: 'Some series cannot be forecast. In a random walk each value is the last one plus a fresh random step, so the past holds no clue about the next step. Trained on each of {walks} such walks, the network misses the next value by {fit} on average on the training parts, {fitVs} repeating the last value ({persistFit}), but by {one} on the held-out parts, where repeating the last value misses by {persist}. It learned the noise, chance patterns that do not come back: overfitting.',
				zh: '有些序列是预测不了的。随机游走里，每个值都是上一个值加上一个新的随机步，所以过去不含任何关于下一步的线索。网络在 {walks} 条这样的游走上分别训练，它对下一个值的平均误差在训练部分是 {fit}，{fitVs}“重复最后一个值”（{persistFit}）；可在留出部分是 {one}，而“重复最后一个值”只差 {persist}。它学到的是噪声，也就是不会再出现的偶然规律：这叫过拟合。',
				vars: (_rt, at) => {
					const m = M(at);
					return { walks: m.walks, fit: m.avgFit, persistFit: m.avgPersistFit, fitVs: pick(m.avgFit < m.avgPersistFit, I('better than', '好于'), I('no better than', '并不好于')), one: m.avgOne, persist: m.avgPersist };
				}
			},
			look: {
				en: '{result} On a random walk the last value is the best guess, so which side wins is luck. The walks are the same for everyone (trained once, ahead of time); walk 1 is the Random walk preset.',
				zh: '{result}在随机游走上，最后一个值就是最好的猜测，所以谁赢全凭运气。这几条游走对每个人都一样（事先训练好了一次）；第 1 条就是“随机游走”预设。',
				vars: (_rt, at) => {
					const won: number[] = [];
					const lost: number[] = [];
					for (let i = 0; i <= at.index; i++) {
						const f = at.tape!.frame(i);
						if (f.phase === 'walk') (f.metrics!.one < f.metrics!.persist ? won : lost).push(f.sub!);
					}
					const w = walkList(won);
					const l = walkList(lost);
					const result = !won.length
						? I('The network loses to “repeat the last value” on every walk.', '网络在每一条游走上都输给了“重复最后一个值”。')
						: !lost.length
							? I('The network beats “repeat the last value” on every walk.', '网络在每一条游走上都胜过了“重复最后一个值”。')
							: I(`The network beats “repeat the last value” on ${w.en} and loses on ${l.en}, where it expects rises while the walk falls.`, `网络在${w.zh}上胜过“重复最后一个值”，在${l.zh}上输了：那里游走在往下走，它却预计会上涨。`);
					return { result };
				}
			}
		}
	],
	fixtures: [
		{ id: 'drawn', label: I('A hand-drawn curve', '一条手画的曲线'), raw: { y: Array.from({ length: 80 }, (_, i) => Math.round(100 * (0.8 * Math.sin(i / 5) + (i % 20 < 10 ? 0.3 : -0.3))) / 100) } }
	]
};

export default lesson;
