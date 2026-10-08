// kalman-filter: the guided lesson, ten steps. Steps 1-7 work through one cycle at second KDEMO of the 'cycle' source
// (the learner's track, the starting settings); 8-9 run the whole track with the learner's settings ('run'); 10 is the
// limits step on the fixed turn track ('turn' against 'turnQ'). Every number is a {placeholder} bound to a live frame or
// built in vars() from one; the other factual sentences are claims.test.ts. The predict step sets the question up and
// leaves the answer to predict.explain (lint AK412).
import type { At, ExplainerRuntime, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { I, KDEMO, N, QBIG, Q0, TURN_AT } from './track.ts';
import { doneOf, runCase, setting, turnStats } from './views/track/meta.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';

type Step = StepDef<ExplainerRuntime>;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const view = (show: string, extra: Record<string, unknown> = {}) => ({ key: 'kalman-filter/track', props: () => ({ show, ...extra }) });
const cyc = (phase: string) => ({ source: 'cycle', at: { phase, step: KDEMO }, provenance: 'live' as const });
const band = (variance: number) => 2 * Math.sqrt(variance);
/** a number inside a product or a difference: a negative one in parentheses, "(−2.29)" */
const par = (v: number) => (v < 0 ? I(`(${formatValue(v, undefined, 'en')})`, `(${formatValue(v, undefined, 'zh')})`) : v);

const steps: Step[] = [
	{
		id: 'noisy', source: 'cycle', at: { phase: 'start' }, view: view('noisy'), provenance: 'live',
		title: I('A noisy sensor', '有噪声的传感器'),
		narration: {
			en: 'A cart rolls along a straight rail. Once a second, for {n} seconds, a sensor reads its position, and every reading (an orange dot) is off by a random error. The dashed line is the true position, which only this page knows because the track is made up. The typical size of the readings’ errors, their root mean square, is {rmseZ} m.',
			zh: '一辆小车沿着笔直的轨道行驶。传感器每秒读一次它的位置，一共读 {n} 秒，每个读数（橙色圆点）都带着一个随机误差。虚线是真实位置：轨迹是编出来的，所以只有这个页面知道它。读数误差的典型大小（均方根）是 {rmseZ} 米。',
			vars: (_rt, at) => ({ n: m(at, 'n'), rmseZ: m(at, 'rmseZ') })
		},
		look: I('The orange dots scatter around the dashed line. In “Your input”, pick another track, draw your own or draw new sensor noise.', '橙色圆点散落在虚线两侧。可以在“你的输入”里换一条轨迹、自己画一条，或者换一组传感器噪声。'),
		terms: ['rmse']
	},
	{
		id: 'predict', ...cyc('predict'), view: view('predict'),
		title: I('First, predict where the cart will be', '先预测：小车会到哪里'),
		narration: {
			en: 'A Kalman filter keeps a best guess of the cart’s position p and velocity v. It started from the first reading and has made two moves every second since: predict, then update. At second {k0} its guess was p = {p} m, moving at v = {v} m/s. Its motion model says the velocity stays the same, so one second later it predicts p⁻ = {p} + {vx} × 1 = {pp} m (the minus sign means: before the reading arrives).',
			zh: '卡尔曼滤波始终保留着对小车位置 p 和速度 v 的最佳猜测。它从第一个读数出发，此后每秒做两个动作：先预测，再更新。第 {k0} 秒时，它猜小车在 p = {p} 米处，速度 v = {v} 米/秒。它的运动模型假定速度不变，所以预测一秒后小车在 p⁻ = {p} + {vx} × 1 = {pp} 米（右上角的减号表示：这是读数到来之前的位置）。',
			vars: (_rt, at) => ({ k0: m(at, 'k') - 1, p: m(at, 'p'), v: m(at, 'v'), vx: par(m(at, 'v')), pp: m(at, 'pp') })
		},
		look: I('The dashed arrow from the teal dot (the guess) to the hollow circle (the prediction), and the same sum in the formula.', '虚线箭头从青色圆点（猜测）指向空心圆（预测）；公式里是同一个加法。'),
		terms: ['kalman-filter', 'motion-model']
	},
	{
		id: 'grow', ...cyc('predict'), view: view('grow'),
		title: I('Predicting makes the filter less sure', '预测让滤波器更没把握'),
		narration: {
			en: 'The filter also tracks how unsure it is, as a variance P: the square of its typical error (the standard deviation). Predicting adds doubt: the velocity is only a guess, and the cart may change speed unexpectedly, by an amount set by Q = {q}. So the position’s variance grows from {P0} to {Pp}, and the band of ±2 standard deviations widens from ±{b0} to ±{bp} m.',
			zh: '滤波器还会记录自己有多没把握，用方差 P 表示：它是典型误差（标准差）的平方。预测会让不确定程度变大：速度只是估计出来的，小车还可能意外地加速或减速，这种变化有多大由 Q = {q} 设定。所以位置的方差从 {P0} 增大到 {Pp}，±2 个标准差的误差带从 ±{b0} 变宽到 ±{bp} 米。',
			vars: (_rt, at) => ({ q: setting(m(at, 'q')), P0: m(at, 'P00'), Pp: m(at, 'Pp00'), b0: band(m(at, 'P00')), bp: band(m(at, 'Pp00')) })
		},
		look: I('The two bars: the guess’s and the wider prediction’s. The formula adds up the doubts: the position’s, the velocity’s (p⁻ = p + v carries both errors), twice how the two move together, and the unexpected push. The ellipse draws both doubts at once, position across and velocity up: dashed before, solid after.', '两条竖线：猜测的误差带和更宽的预测误差带。公式把各处的不确定程度加起来：位置的、速度的（p⁻ = p + v 同时带着两者的误差）、两倍的协方差（两者一起变化的程度），再加上意外的推动。椭圆把两种不确定程度画在一起，横向是位置，纵向是速度：虚线是之前，实线是之后。'),
		terms: ['variance', 'process-noise']
	},
	{
		id: 'measure', ...cyc('measure'), view: view('measure'),
		title: I('A reading arrives', '读数到了'),
		narration: {
			en: 'Now the sensor reads z = {z} m (the orange dot), while the prediction said {pp} m. The gap y = z − p⁻ = {y} m is the surprise, called the innovation. Neither number is simply right: the reading’s variance is R = {r} (band ±{bz} m) and the prediction’s is P⁻ = {Pp} (band ±{bp} m).',
			zh: '这时传感器读到 z = {z} 米（橙色圆点），而预测是 {pp} 米。两者之差 y = z − p⁻ = {y} 米就是“意外”，术语叫新息。两个数都不一定准：读数的方差是 R = {r}（误差带 ±{bz} 米），预测的方差是 P⁻ = {Pp}（误差带 ±{bp} 米）。',
			vars: (_rt, at) => ({ z: m(at, 'z'), pp: m(at, 'pp'), y: m(at, 'y'), r: setting(m(at, 'r')), bz: band(m(at, 'r')), Pp: m(at, 'Pp00'), bp: band(m(at, 'Pp00')) })
		},
		look: I('The line marked y, between the hollow circle and the orange dot, is the gap; each of the two has its own band.', '空心圆和橙色圆点之间标着 y 的那条线就是差距；两者各有自己的误差带。'),
		terms: ['process-noise']
	},
	{
		id: 'gain', ...cyc('gain'), view: view('gain'),
		title: I('The Kalman gain: how much to trust the reading', '卡尔曼增益：读数该信几分'),
		narration: {
			en: 'The filter moves from the prediction towards the reading by a share K of the gap, called the Kalman gain: K = P⁻ / (P⁻ + R) = {Pp} / ({Pp} + {r}) = {K}. K = 0 would keep the prediction and K = 1 would jump to the reading. Guess first: where will the new estimate land?',
			zh: '滤波器从预测出发，朝读数的方向挪动两者差距的一部分，这个比例 K 叫作卡尔曼增益：K = P⁻ / (P⁻ + R) = {Pp} / ({Pp} + {r}) = {K}。K = 0 表示完全保留预测，K = 1 表示直接采用读数。先猜一猜：新的估计会落在哪里？',
			vars: (_rt, at) => ({ Pp: m(at, 'Pp00'), r: setting(m(at, 'r')), K: m(at, 'K') })
		},
		look: I('The gauge: where K sits between 0 (keep the prediction) and 1 (take the reading).', '刻度条：K 落在 0（保持预测）和 1（采用读数）之间的哪个位置。'),
		predict: {
			id: 'kalman-filter-gain',
			question: I('Where will the new estimate land?', '新的估计会落在哪里？'),
			options: [I('Closer to the prediction', '离预测更近'), I('Closer to the reading', '离读数更近'), I('Exactly halfway between them', '正好在两者中间')],
			answer: 0,
			explain: I('K is below one half because the prediction’s variance P⁻ is smaller than the sensor’s variance R: the filter trusts its prediction more, so it moves less than halfway towards the reading. If the prediction were the less certain one, K would be above one half.', 'K 小于一半，因为预测的方差 P⁻ 比传感器的方差 R 小：滤波器更相信自己的预测，所以朝读数挪不到一半。如果预测更不确定，K 就会大于一半。')
		},
		terms: ['kalman-gain']
	},
	{
		id: 'update', ...cyc('update'), view: view('update'),
		title: I('Then update the estimate', '再更新：修正估计'),
		narration: {
			en: 'The estimate moves the share K of the gap: p = p⁻ + K × y = {pp} + {K} × {yx} = {pn} m (the teal dot). The surprise also hints that the speed guess was off, so the velocity moves with its own gain {Kv}: from {vp} to {vn} m/s. The true position was {truth} m: the estimate is {eN} m off and the reading {eZ} m, {luck}',
			zh: '新的估计挪动差距的 K 倍：p = p⁻ + K × y = {pp} + {K} × {yx} = {pn} 米（青色圆点）。这个“意外”也说明速度可能猜得不准，所以速度也用它自己的增益 {Kv} 修正：从 {vp} 变为 {vn} 米/秒。真实位置是 {truth} 米：估计偏差 {eN} 米，读数偏差 {eZ} 米，{luck}',
			vars: (_rt, at) => {
				const eN = Math.abs(m(at, 'err')), eZ = Math.abs(m(at, 'errZ'));
				return { pp: m(at, 'pp'), K: m(at, 'K'), yx: par(m(at, 'y')), pn: m(at, 'p'), Kv: m(at, 'Kv'), vp: m(at, 'vp'), vn: m(at, 'v'), truth: m(at, 'truth'), eN, eZ, luck: pick(eN < eZ, I('so this time the filter is closer.', '这一次滤波器更接近真实位置。'), I('so the reading was luckier this time; step 8 scores the whole run, not one second.', '这一次读数运气更好；第 8 步会给整段运行打分，而不只看一秒。')) };
			}
		},
		look: I('The teal dot sits between the hollow circle and the orange dot, at the share K of the way. The short level mark at the same second is the true position.', '青色圆点落在空心圆和橙色圆点之间，位于 K 倍距离处。同一秒上那条短横线是真实位置。')
	},
	{
		id: 'shrink', ...cyc('update'), view: view('shrink'),
		title: I('The doubt shrinks', '不确定程度变小了'),
		narration: {
			en: 'Combining two uncertain guesses gives a surer one. The new variance is P = (1 − K) × P⁻ = (1 − {K}) × {Pp} = {P}, so the band shrinks to ±{b} m: narrower than the prediction’s ±{bp} m and even narrower than the reading’s ±{bz} m. The next second starts again with a prediction.',
			zh: '把两个都不太确定的猜测结合起来，会得到一个更可靠的估计。新的方差是 P = (1 − K) × P⁻ = (1 − {K}) × {Pp} = {P}，误差带因此缩小到 ±{b} 米：比预测的 ±{bp} 米窄，甚至比读数的 ±{bz} 米还窄。下一秒又从预测开始。',
			vars: (_rt, at) => ({ K: m(at, 'K'), Pp: m(at, 'Pp00'), P: m(at, 'P00'), b: band(m(at, 'P00')), bp: band(m(at, 'Pp00')), bz: 2 * m(at, 'sdz') })
		},
		look: {
			en: 'Three bars side by side: the prediction’s, the new, shortest one and the reading’s. The ellipse narrows more across (position) than up (velocity). The new P always equals K × R, here {K} × {r}.',
			zh: '三条竖线并排：预测的、新的（最短的那条）和读数的。椭圆在横向（位置）上缩得比纵向（速度）多。新的 P 总是等于 K × R，这里是 {K} × {r}。',
			vars: (_rt, at) => ({ K: m(at, 'K'), r: setting(m(at, 'sdz') ** 2) })
		},
		terms: ['variance']
	},
	{
		id: 'run', source: 'run', at: { phase: 'done' }, view: view('run'), provenance: 'live',
		play: { from: { phase: 'update', step: 1 }, to: { phase: 'done' }, phases: ['update', 'done'], interval: 120, unit: I('second', '时刻') },
		title: I('Run the whole track', '跑完整条轨迹'),
		narration: {
			en: 'The filter repeats predict and update every second, and the player replays it. Over the {n} seconds, with Q = {q} and R = {r}, the typical error (root mean square) is {kf} m for the filter, {raw} m for the raw readings and {ma} m for a moving average of the last {w} readings. {verdict}',
			zh: '滤波器每秒都重复“预测—更新”，播放条会逐秒重放这一过程。在这 {n} 秒里（Q = {q}，R = {r}），典型误差（均方根）分别是：滤波器 {kf} 米，原始读数 {raw} 米，最近 {w} 个读数的移动平均 {ma} 米。{verdict}',
			vars: (_rt, at) => {
				const d = at.frame?.metrics ?? {}, c = runCase(d), f = (v: number, l: 'en' | 'zh') => formatValue(v, undefined, l) ?? '…';
				const best = I(f(d.bestRmse, 'en'), f(d.bestRmse, 'zh')), win = d.rmseZ <= d.rmseMa ? I('the raw readings', '原始读数') : I('the moving average', '移动平均');
				const verdict = ![d.rmseKf, d.rmseZ, d.rmseMa, d.bestRmse].every(Number.isFinite) ? NaN
					: c === 'wins' ? I('The filter beats both: each reading keeps all its noise, and the average trails behind whenever the cart moves, because it averages where the cart was.', '滤波器胜过另外两者：每个读数都带着全部噪声，而只要小车在动，移动平均就会落后，因为它平均的是小车过去的位置。')
					: c === 'tune' ? I(`With these settings the filter does not win here, but the best pair on the sliders gets ${best.en} m and beats both (step 9).`, `用这组设置，滤波器在这里没有胜出；但滑块上最好的一组设置能做到 ${best.zh} 米，胜过另外两者（第 9 步）。`)
					: I(`Here even the best pair on the sliders (${best.en} m) loses to ${win.en}: the cart does not move the way the filter’s motion model expects (step 10).`, `在这里，即使滑块上最好的一组设置（${best.zh} 米）也比不过${win.zh}：小车的运动方式不符合滤波器的运动模型（第 10 步）。`);
				return { n: m(at, 'n'), q: setting(m(at, 'q')), r: setting(m(at, 'r')), kf: m(at, 'rmseKf'), raw: m(at, 'rmseZ'), ma: m(at, 'rmseMa'), w: m(at, 'w'), verdict };
			}
		},
		look: I('The teal line and its band against the blue moving average (faint: the seconds the player has not reached yet), and the three bars, which score the whole run.', '青色的线和它的误差带对比蓝色的移动平均线（颜色淡的部分：播放还没走到的秒），以及给整段运行打分的三条误差柱。'),
		terms: ['moving-average', 'rmse']
	},
	{
		id: 'tune', source: 'run', at: { phase: 'done' }, view: view('tune'), provenance: 'live',
		title: I('Your turn: tune Q and R', '轮到你了：调节 Q 和 R'),
		narration: {
			en: 'Q tells the filter how much the cart may change speed unexpectedly; R tells it how noisy the sensor is. A small R or a large Q makes it trust the readings and copy their noise; a large R or a small Q makes it trust its motion model and react late. Right now the error is {now} m, and the best pair of slider settings gets {best} m.',
			zh: 'Q 告诉滤波器，小车的速度可能意外变化多少；R 告诉它传感器的噪声有多大。R 小或 Q 大，它就更相信读数，把噪声也一并照搬；R 大或 Q 小，它就更相信运动模型，对变化反应迟缓。现在的误差是 {now} 米，滑块上最好的一组设置能做到 {best} 米。',
			vars: (_rt, at) => ({ now: m(at, 'rmseKf'), best: m(at, 'bestRmse') })
		},
		look: I('Your error and the best one under the sliders; the teal line on the plot changes with every move.', '滑块下方是你的误差和最好的误差；每动一下，图上的青色线都会跟着变。'),
		task: {
			prompt: I('Move the Q and R sliders until the filter’s error is at most 2% above the best pair’s.', '移动 Q 和 R 两个滑块，直到滤波器的误差比最好的组合高出不到 2%。'),
			done: (_rt, at) => m(at, 'rmseKf') <= 1.02 * m(at, 'bestRmse'),
			hint: I('Change one slider at a time and watch the error after each move; if it grows, go the other way.', '一次只动一个滑块，每动一下就看看误差；如果误差变大，就往反方向调。'),
			waiting: I('Not yet: your error is still more than 2% above the best pair’s.', '还没有：你的误差仍比最好的组合高出 2% 以上。')
		},
		terms: ['process-noise']
	},
	{
		id: 'limits', source: 'turn', at: { phase: 'done' }, view: view('turn', { from: TURN_AT - 12, to: TURN_AT + 16, turnAt: TURN_AT }), provenance: 'live', kindOf: 'limits',
		compare: { a: 'turn', b: 'turnQ', label: [I(`Starting setting Q = ${Q0}`, `起始设置 Q = ${Q0}`), I(`Larger Q = ${QBIG}`, `更大的 Q = ${QBIG}`)] },
		title: I('Limits: a wrong motion model', '局限：错误的运动模型'),
		narration: {
			en: 'A Kalman filter is only as good as its motion model. At second {tk} this cart turns back suddenly, but the model expects it to keep going: the estimate overshoots by up to {worst} m, and for {out} of the {after} seconds after the turn the truth lies outside the band, the filter sure and wrong. A larger Q = {q2} (second plot) recovers sooner ({out2} outside), but before the turn its error rises from {pre1} to {pre2} m.',
			zh: '卡尔曼滤波的好坏取决于它的运动模型。这辆小车在第 {tk} 秒突然掉头，可模型以为它会继续往前开：估计冲过了头，最多偏差 {worst} 米；掉头后的 {after} 秒里，有 {out} 秒真实位置落在误差带之外：滤波器很自信，却是错的。更大的 Q = {q2}（第二张图）恢复得更快（只有 {out2}在带外），但掉头之前的误差从 {pre1} 米升到了 {pre2} 米。',
			vars: (rt, at) => {
				const a = turnStats(at.tape, at.index, TURN_AT), bt = rt.sources.turnQ?.tape, b = turnStats(bt ?? undefined, (bt?.length ?? 0) - 1, TURN_AT);
				return { tk: TURN_AT, worst: a.worst, out: a.out, after: a.after, q2: setting(doneOf(bt ?? undefined).q ?? NaN), out2: plural(b.out, { one: '{n} second', many: '{n} seconds', zh: '{n} 秒' }), pre1: a.pre, pre2: b.pre };
			}
		},
		look: I('The band stays just as narrow after the turn: the filter’s doubt depends only on Q, R and time, never on how surprising the readings are. Real trackers watch the surprise y to notice a broken model.', '掉头之后误差带一点也没有变宽：滤波器的不确定程度只取决于 Q、R 和时间，从不取决于读数有多意外。实际的跟踪系统会盯着“意外” y 来发现模型出了问题。'),
		terms: ['motion-model']
	}
];

const lesson: LessonDef = {
	title: I('A Kalman filter, one prediction and one reading at a time', '卡尔曼滤波：一次预测，一次读数'),
	summary: I('Follow a cart through noisy sensor readings: predict with a motion model, weigh the reading with the Kalman gain, update, and see where the filter beats a moving average and where a sudden turn fools it.', '借助有噪声的传感器读数跟踪一辆小车：用运动模型预测，用卡尔曼增益权衡读数，更新估计；看看滤波器在哪里胜过移动平均，又在哪里被突然掉头骗过。'),
	steps,
	fixtures: [
		{ id: 'start', label: I('Start, cruise, stop (the page’s start)', '起步、匀速、停下（本页的起始输入）'), raw: { preset: 'stopgo', seed: 1 } },
		{ id: 'cruise', label: I('Steady speed', '匀速'), raw: { preset: 'cruise', seed: 1 } },
		{ id: 'wave', label: I('Back and forth, new noise', '来回摆动，换一组噪声'), raw: { preset: 'wave', seed: 5 } },
		{ id: 'turn', label: I('Sudden turn', '突然掉头'), raw: { preset: 'turn', seed: 1 } },
		{ id: 'tuned', label: I('Start track with the best pair Q = 0.1, R = 4', '起始轨迹，最好的组合 Q = 0.1、R = 4'), raw: { preset: 'stopgo', seed: 1, q: 0.1, r: 4 } },
		{ id: 'drawn', label: I('A drawn ramp, then flat', '画出的斜坡，然后走平'), raw: { y: Array.from({ length: N }, (_, t) => Math.min(30, 0.8 * t)), seed: 2, q: 0.001, r: 8 } }
	]
};

export default lesson;
