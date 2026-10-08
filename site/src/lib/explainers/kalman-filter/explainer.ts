// kalman-filter: the manifest. A 2-D (position + velocity) linear Kalman filter follows the noisy sensor readings of a
// cart on a straight rail. Kind C only: every stage is the process 'kf' (process.ts), run live in the browser on
// dataOf(input) and checked frame by frame against an independent Python run (pipeline/kalman-filter/reference.py).
//   cycle   the learner's track with the starting settings (the worked example of one predict-update cycle)
//   run     the learner's track with the learner's q and r (the whole run, the tuning task)
//   turn    the fixed "sudden turn" track with the starting settings, and turnQ with a large q (the limits step)
// No data, model or code from anyone else: the tracks and the noise are made in the browser (track.ts).
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { dataOf, I, QBIG, START } from './track.ts';

export const TURN = { preset: 'turn' as const, seed: 1 };
const on = (input: (rt: ExplainerRuntime) => unknown) => ({ kind: 'process' as const, process: 'kf', run: 'main' as const, seed: 1, input });

const manifest: ExplainerManifest = {
	slug: 'kalman-filter',
	type: 'model',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'track', start: START },
	sources: {
		cycle: on((rt) => dataOf(rt.input, { fixed: true })),
		run: on((rt) => dataOf(rt.input)),
		turn: on(() => dataOf(TURN, { fixed: true })),
		turnQ: on(() => ({ ...dataOf(TURN, { fixed: true }), q: QBIG }))
	},
	page: {
		goals: [
			I('Work through one predict-and-update cycle of a Kalman filter with its numbers: the prediction, the reading, the gain K = P⁻/(P⁻ + R), the new estimate.', '带着数字走完卡尔曼滤波的一轮“预测—更新”：预测、读数、增益 K = P⁻/(P⁻ + R)、新的估计。'),
			I('Say why the doubt grows when the filter predicts and shrinks when a reading arrives, and why the new estimate is surer than both the prediction and the reading.', '说出为什么预测时不确定程度会变大、读数到来时会变小，以及为什么新的估计比预测和读数都更可靠。'),
			I('Tune the process noise Q and the measurement noise R to bring the error down, and say what too small or too large values do.', '调节过程噪声 Q 和测量噪声 R 把误差降下来，并说出它们太小或太大会怎样。'),
			I('Explain why a wrong motion model (a sudden turn) makes the filter sure and wrong for a while.', '解释为什么错误的运动模型（突然掉头）会让滤波器有一阵子“很自信却是错的”。')
		],
		prequestion: {
			id: 'kalman-filter-pre',
			question: I('A cart moves along a rail and a noisy sensor reads its position once a second. What gives the best guess of where it is now?', '一辆小车沿轨道行驶，一个有噪声的传感器每秒读一次它的位置。怎样才能最好地猜出它现在在哪里？'),
			options: [I('The latest reading alone', '只看最新的读数'), I('The average of all readings so far', '把到目前为止的所有读数取平均'), I('A guess from how it was moving, corrected a little by each new reading', '根据它之前怎么动来猜，再用每个新读数稍作修正')],
			answer: 2,
			explain: I('The latest reading carries all of its noise, and an average of old readings trails behind a moving cart. A Kalman filter predicts from the motion, then corrects the prediction by a share of each new reading.', '最新的读数带着它全部的噪声，旧读数的平均值又会落后于一辆在动的小车。卡尔曼滤波先根据运动来预测，再用每个新读数按一定比例修正预测。')
		},
		tryThis: [
			{ id: 'wave', title: I('Back and forth', '来回摆动'), setup: { input: { preset: 'wave', seed: 1 } }, predict: I('On a cart that swings back and forth, will the starting settings or a larger Q do better?', '小车来回摆动时，是起始设置好，还是更大的 Q 更好？'), look: I('In the tuning step, compare the filter’s error with Q = 0.01 and with Q = 0.3.', '在调参那一步，比较 Q = 0.01 和 Q = 0.3 时滤波器的误差。'), goto: { step: 'tune' } },
			{ id: 'cruise', title: I('Steady speed', '匀速'), setup: { input: { preset: 'cruise', seed: 1 } }, predict: I('On a cart at a perfectly steady speed, should Q be small or large?', '小车完全匀速时，Q 应该小还是大？'), look: I('In the tuning step, try the smallest Q with a large R.', '在调参那一步，试试最小的 Q 加上较大的 R。'), goto: { step: 'tune' } }
		],
		quiz: [
			{ q: I('The prediction’s variance is P⁻ = 3 and the sensor’s is R = 1. What is the Kalman gain K?', '预测的方差 P⁻ = 3，传感器的方差 R = 1。卡尔曼增益 K 是多少？'), options: [I('0.75', '0.75'), I('0.25', '0.25'), I('3', '3')], answer: 0, explain: I('K = P⁻/(P⁻ + R) = 3/(3 + 1) = 0.75: the prediction is the less certain one, so the filter moves three quarters of the way to the reading.', 'K = P⁻/(P⁻ + R) = 3/(3 + 1) = 0.75：预测更不确定，所以滤波器朝读数挪四分之三的距离。') }
		],
		misconception: { claim: I('The filter should always follow the latest reading as closely as it can.', '滤波器应该总是尽量紧跟最新的读数。'), truth: I('Every reading carries noise. Following it closely copies the noise; the gain K weighs the reading against the prediction by how uncertain each one is.', '每个读数都带着噪声，紧跟读数就是把噪声也照搬过来；增益 K 按两者各自的不确定程度来权衡读数和预测。'), step: 'gain' },
		lessonNote: I('Ten steps on your own track: noisy readings, one predict-and-update cycle worked out with its numbers, the whole run against a moving average, tuning Q and R, and what a sudden turn does.', '十步，都在你自己的轨迹上：有噪声的读数、带数字的一轮“预测—更新”、整段运行与移动平均的对比、调节 Q 和 R，以及突然掉头会怎样。')
	},
	glossary: [
		{ id: 'kalman-filter', term: I('Kalman filter', '卡尔曼滤波'), def: I('A method that tracks a changing quantity from noisy measurements by repeating two steps: predict with a motion model, then correct the prediction by a share (the gain) of each new measurement.', '一种从有噪声的测量中跟踪变化量的方法，反复做两步：用运动模型预测，再按一定比例（增益）用新的测量修正预测。'), related: ['kalman-gain', 'motion-model'] },
		{ id: 'motion-model', term: I('motion model', '运动模型'), def: I('The filter’s rule for how the state changes in one time step; here: the position moves on by the velocity, and the velocity stays the same.', '滤波器描述状态在一个时间步内如何变化的规则；这里是：位置按速度往前走，速度保持不变。') },
		{ id: 'variance', term: I('variance and standard deviation', '方差和标准差'), def: I('The standard deviation is the typical size of an error; the variance is its square. When the errors follow a bell curve (a normal distribution), a band of ±2 standard deviations holds the true value about 95 times in 100.', '标准差是误差的典型大小，方差是它的平方。误差服从钟形的正态分布时，±2 个标准差的范围大约 100 次里有 95 次包含真实值。') },
		{ id: 'kalman-gain', term: I('Kalman gain', '卡尔曼增益'), def: I('K = P⁻/(P⁻ + R), with P⁻ the variance of the prediction and R that of the measurement: the share of the gap between the two that the filter moves; near 0 it keeps the prediction, near 1 it takes the measurement.', 'K = P⁻/(P⁻ + R)，其中 P⁻ 是预测的方差，R 是测量的方差：滤波器在预测和测量之间挪动的比例；接近 0 时保持预测，接近 1 时采用测量。'), related: ['variance'] },
		{ id: 'process-noise', term: I('process noise Q and measurement noise R', '过程噪声 Q 和测量噪声 R'), def: I('Q: how much the true motion may change in ways the motion model does not know. R: the variance of the sensor’s error. Both are settings the filter is given.', 'Q：真实运动可能以运动模型不知道的方式改变多少。R：传感器误差的方差。两者都是交给滤波器的设定。'), related: ['kalman-gain'] },
		{ id: 'rmse', term: I('root mean square error (RMSE)', '均方根误差（RMSE）'), def: I('Square every error, average the squares, take the square root: one number for the typical size of the errors.', '把每个误差平方，求平均，再开平方根：用一个数表示误差的典型大小。') },
		{ id: 'moving-average', term: I('moving average', '移动平均'), def: I('The average of the last few measurements, recomputed at every step; it smooths noise but trails behind a moving value.', '最近几次测量的平均值，每一步重新计算；它能抹平噪声，但会落后于一个在变化的值。') }
	],
	catalog: {
		shows: I('One predict and update cycle with real numbers: the prediction, the reading, the Kalman gain and the new estimate, with the uncertainty as a band and as a covariance ellipse; the whole run against the raw readings and a moving average; tuning Q and R; a sudden turn.', '带真实数字的一轮预测与更新：预测、读数、卡尔曼增益和新的估计，不确定程度画成误差带和协方差椭圆；整段运行与原始读数、移动平均的对比；调节 Q 和 R；突然掉头。'),
		runtime: I('Own TypeScript, live in the browser (checked against a Python run)', '自研 TypeScript，在浏览器中实时运行（已与 Python 运行核对）'),
		params: I('none learned: the settings Q and R, and a 2-number state with its 2 × 2 covariance', '没有学习的参数：设定 Q 和 R，以及 2 个数的状态和它的 2 × 2 协方差'),
		size: I('no files: the tracks and the sensor noise are made in your browser', '没有文件：轨迹和传感器噪声都在你的浏览器里生成')
	},
	credits: [
		{
			name: 'R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems” (Journal of Basic Engineering, 1960)',
			url: 'https://doi.org/10.1115/1.3662552',
			license: 'Article; ideas only (no text, figures, code or data used)',
			usedFor: 'the predict and update equations of the filter',
			attribution: 'Rudolf E. Kálmán'
		}
	]
};

export default manifest;
