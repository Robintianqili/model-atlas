// gaussian-process: the guided lesson, eleven steps on the 'gp' source (the learner's points and settings), the last on
// the fixed step data ('short' against 'long'). Every number is a {placeholder} bound to a live frame or built in vars()
// from one (a sentence that needs two points is chosen with pick(), its numbers filled from the frame); the other
// factual sentences are claims.test.ts. The predict step sets the question up and leaves the answer to predict.explain.
import type { At, ExplainerRuntime, I18n, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { BIG, I, LONG, SHORT } from './gp.ts';
import { phaseOf, setting } from './views/plot/meta.ts';
import { num, pick, plural } from '../../atlas/lesson/say.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';

type Step = StepDef<ExplainerRuntime>;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const view = (show: string) => ({ key: 'gaussian-process/plot', props: () => ({ show }) });
const gp = (phase: string, extra: Record<string, unknown> = {}) => ({ source: 'gp', at: { phase, ...extra }, provenance: 'live' as const });
/** a sentence with {name} slots filled as the narration prints numbers (for sentences chosen with pick) */
const fill = (s: I18n, v: Record<string, number>): I18n => {
	const f = (l: 'en' | 'zh', t: string) => t.replace(/\{(\w+)\}/g, (_, k: string) => formatValue(v[k], undefined, l) ?? '…');
	return { en: f('en', s.en), zh: f('zh', s.zh ?? s.en) };
};

/** how strongly a kernel value links two points, by its share r = k/σf² of the largest possible (the correlation) */
export const rel = (r: number): I18n => (r > 0.999 ? I('so their values move together', '它们的取值一起变化') : r > 0.5 ? I('so their values move almost together', '它们的取值几乎一起变化') : r > 0.1 ? I('a moderate link', '有一定的联系') : I('so they are nearly independent', '几乎互不相关'));

/** how far the learner's length-scale is from the best on the slider; the task counts a gap below CLOSE as found */
export const CLOSE = 0.005;
const verdict = (at: At): I18n | number => (!Number.isFinite(m(at, 'drop')) ? NaN : m(at, 'drop') <= CLOSE ? I('No length-scale on the slider does noticeably better: this one balances fit and complexity best.', '滑块上没有哪个长度尺度明显更好：这个长度尺度把拟合和复杂度平衡得最好。') : fill(I('The best length-scale on the slider, the one that balances fit and complexity, reaches log p(y) = {bestLml}, {drop} higher.', '滑块上最好的长度尺度（把拟合和复杂度平衡得最好的那个）能让 log p(y) 达到 {bestLml}，高出 {drop}。'), { bestLml: m(at, 'bestLml'), drop: m(at, 'drop') }));

const steps: Step[] = [
	{
		id: 'prior', ...gp('prior'), view: view('prior'),
		title: I('Before any data: a cloud of functions', '还没有数据：一团函数'),
		narration: {
			en: 'A Gaussian process is a probability distribution over functions. Before it uses any point, its best guess is 0 everywhere, and its band of ±2 standard deviations is ±{band0} at every x: twice the square root of the signal variance σf² = {sf2}. The three colored curves are functions drawn at random from this prior; how fast they wiggle is set by the length-scale ℓ = {ls}.',
			zh: '高斯过程是函数上的概率分布。在用上任何数据点之前，它的最佳猜测处处是 0，±2 个标准差的误差带在每个 x 处都是 ±{band0}：信号方差 σf² = {sf2} 的平方根的两倍。三条彩色曲线是从这个先验中随机抽取的函数；它们起伏得多快，由长度尺度 ℓ = {ls} 决定。',
			vars: (_rt, at) => ({ band0: m(at, 'band0'), sf2: setting(m(at, 'sf2')), ls: setting(m(at, 'ls')) })
		},
		look: I('The three colored curves inside the band. Move the ℓ slider: a shorter length-scale makes them wiggle faster.', '看误差带里的三条彩色曲线。拖动 ℓ 滑块：长度尺度越短，它们起伏得越快。'),
		terms: ['gaussian-process', 'variance']
	},
	{
		id: 'kernel', ...gp('kernel'), view: view('kernel'),
		title: I('The kernel: nearby points move together', '核函数：离得近的点一起变化'),
		narration: {
			en: 'What ties the function’s values together is the kernel, the covariance of the values at two places x and x′ (how strongly they move together): k(x, x′) = σf² exp(−d²/(2ℓ²)), where d is the distance between x and x′. {pairs}',
			zh: '把函数在各处的取值联系起来的是核函数，也就是 x 与 x′ 两处取值的协方差（它们一起变化的程度）：k(x, x′) = σf² exp(−d²/(2ℓ²))，其中 d 是 x 与 x′ 之间的距离。{pairs}',
			vars: (_rt, at) => {
				const v = Object.fromEntries(['iN', 'jN', 'dN', 'kN', 'iF', 'jF', 'dF', 'kF', 'sf2'].map((k) => [k, m(at, k)])), a = rel(v.kN / v.sf2), b = rel(v.kF / v.sf2);
				const one = I('With a single point there is no pair yet: add a second point to the plot.', '只有一个点时还没有点对：在图上再加一个点。');
				const same = I('All the points sit at the same x, so every pair is 0 apart and k = σf²: one value of the function has to explain them all. Add a point elsewhere.', '所有的点都在同一个 x 上，每一对都相距 0，k = σf²：函数在那里只有一个值，要同时解释所有的点。在别处再加一个点。');
				return { pairs: pick(v.dF > 0, fill(I(`The closest pair, points {iN} and {jN} (counting from 0), lie {dN} apart: k = {kN}, ${a.en}. The farthest pair, points {iF} and {jF}, lie {dF} apart: k = {kF}, ${b.en}.`, `最近的一对是编号 {iN} 和 {jN} 的点（从 0 数起），相距 {dN}：k = {kN}，${a.zh}。最远的一对是编号 {iF} 和 {jF} 的点，相距 {dF}：k = {kF}，${b.zh}。`), v), pick(m(at, 'n') > 1, same, one)) };
			}
		},
		look: I('The curve of k against the distance: the blue dot is the closest pair, the gray dot the farthest.', '看 k 随距离变化的曲线：蓝点是最近的一对，灰点是最远的一对。'),
		terms: ['rbf-kernel']
	},
	{
		id: 'first', ...gp('add', { step: 0 }), view: view('first'),
		title: I('One point pulls the curve', '一个点拉动曲线'),
		narration: {
			en: 'Now point 0 joins, at x₀ = {x0} with y₀ = {y0}. Right there the mean becomes σf²/(σf² + σn²) × y₀ = {m0}: almost y₀, because the noise variance σn² = {sn2} (how far a measured y may stray from the function) is small. At xₚ = {xp}, one length-scale away, the kernel is only {kp}, so the mean moves part of the way, to {mp}, and the band is ±{bp} instead of ±{b0}.',
			zh: '现在编号 0 的点加进来，位于 x₀ = {x0}，y₀ = {y0}。就在那里，均值变成 σf²/(σf² + σn²) × y₀ = {m0}：几乎等于 y₀，因为噪声方差 σn² = {sn2}（测得的 y 可能偏离函数多远）很小。在相距一个长度尺度的 xₚ = {xp} 处，核函数只有 {kp}，所以均值只被拉过去一部分，到 {mp}，误差带是 ±{bp}，而不是 ±{b0}。',
			vars: (_rt, at) => ({ x0: m(at, 'xi'), y0: m(at, 'yi'), m0: m(at, 'mAt'), sn2: setting(phaseOf(at.tape, 'prior').sn2 ?? NaN), xp: m(at, 'xp'), kp: m(at, 'kp'), mp: m(at, 'mp'), bp: 2 * m(at, 'sdp'), b0: 2 * m(at, 'sd0') })
		},
		look: I('The two vertical bars: the band at x₀, very short, and at xₚ, longer. Farther away the band is back at its full prior width.', '看两条竖线：x₀ 处的误差带很短，xₚ 处长一些。再远一些，误差带就回到先验的全宽。'),
		terms: ['gp-posterior']
	},
	{
		id: 'gap', ...gp('done'), view: view('gap'),
		title: I('Between two points', '两个点之间'),
		narration: {
			en: 'Right at a point the band is narrow, because the point pins the function there. {setup}',
			zh: '在数据点上，误差带很窄，因为这个点把函数固定在了那里。{setup}',
			vars: (_rt, at) => { const w = plural(m(at, 'gwl'), { one: '{n} length-scale', many: '{n} length-scales', zh: '{n} 个长度尺度' }) as I18n; return { setup: pick(Number.isFinite(m(at, 'gw')), fill(I(`All {n} points are in now, and the widest stretch without a point runs from x = {ga} to x = {gb}, ${w.en} wide, with a point at each end. Before you look: what does the band do between them?`, `现在 {n} 个点都加进来了，最宽的一段空白从 x = {ga} 到 x = {gb}，宽 ${w.zh}，两端各有一个点。先想一想：在它们之间，误差带会怎样？`), { n: m(at, 'n'), ga: m(at, 'ga'), gb: m(at, 'gb') }), pick(m(at, 'n') > 1, I('All the points sit at the same x, so there is no gap between two points yet: add a point elsewhere, then think about what the band does between two points.', '所有的点都在同一个 x 上，所以还没有两点之间的空白：在别处再加一个点，再想一想误差带在两个点之间会怎样。'), I('There is only one point, so there is no gap between two points yet: add a second point, then think about what the band does between them.', '现在只有一个点，所以还没有两点之间的空白：再加一个点，再想一想误差带在两点之间会怎样。'))) }; }
		},
		look: I('The shaded stretch is the widest gap; the three vertical bars show the band at its two ends and in its middle.', '阴影部分是最宽的空白；三条竖线分别是它两端和正中间的误差带。'),
		predict: {
			id: 'gaussian-process-gap',
			question: I('What does the band do between two neighbouring points?', '在相邻两个点之间，误差带会怎样？'),
			options: [I('It stays as narrow as at the points', '保持和点上一样窄'), I('It widens towards the middle (more in a wider gap), then narrows again at the next point', '越往中间越宽（空白越宽，宽得越多），到下一个点又变窄'), I('It is widest right next to the points', '紧挨着点的地方最宽')],
			answer: 1,
			explain: I('Each point pins the function only near itself, within a few length-scales. In the middle of a gap both points are farthest away, so the band is widest there. How wide it gets depends on the gap measured in length-scales: in a narrow gap it barely widens, in a gap of three or more length-scales it grows back to most of the prior band.', '每个点只在自己附近（几个长度尺度以内）把函数固定住。在空白的正中间，两个点都离得最远，所以误差带在那里最宽。它能变得多宽，取决于用长度尺度来量的空白宽度：空白很窄时几乎不变宽；空白有三个长度尺度或更宽时，误差带会恢复到先验宽度的大部分。')
		},
		terms: ['gp-posterior']
	},
	{
		id: 'points', ...gp('add', { last: true }), view: view('points'),
		play: { from: { phase: 'add', step: 0 }, to: { phase: 'add', last: true }, phases: ['add'], interval: 900, unit: I('point', '点') },
		title: I('The posterior, point by point', '后验，一个点一个点地更新'),
		narration: {
			en: 'Play adds the points one at a time, in the order they were added. Each new point pulls the mean towards itself and pinches the band around it, most of all right at the point. With {all} the average band over the plot has shrunk from ±{b0} to ±{bn}, and the mean passes within {res} of every point.',
			zh: '播放会按加入的顺序，一次加入一个点。每个新点都把均值拉向自己，并把周围的误差带收紧，在这个点上收得最紧。{all}后，整张图上误差带的平均宽度从 ±{b0} 缩小到 ±{bn}，均值与每个点的距离都不超过 {res}。',
			vars: (_rt, at) => ({ all: pick(m(at, 'used') > 1, fill(I('all {n} points', '{n} 个点全部加入'), { n: m(at, 'used') }), I('its one point', '唯一的一个点加入')), b0: 2 * m(at, 'sd0'), bn: m(at, 'avgBand'), res: m(at, 'maxRes') })
		},
		look: I('Watch the band pinch at each new point; the hollow points are not used yet.', '看误差带在每个新点处收紧；空心的点还没用上。'),
		terms: ['gp-posterior']
	},
	{
		id: 'matrix', ...gp('kernel'), view: view('matrix'),
		title: I('The kernel matrix and the weights', '核矩阵和权重'),
		narration: {
			en: 'Behind every curve is one table, K + σn² I: the kernel of every pair of points, with σf² + σn² = {kd} on the diagonal, {n} × {n} in all. Solving (K + σn² I) α = y gives one weight α per point, and the mean anywhere is the weighted sum m(x) = Σ αᵢ k(x, xᵢ). At x = 0 it is {mq}; the term largest in size comes from point {top} (counting from 0), at x = {topX}.',
			zh: '每条曲线背后都是同一张表 K + σn² I：每一对点之间的核函数值，对角线上是 σf² + σn² = {kd}，一共 {n} × {n} 个数。解方程 (K + σn² I) α = y，每个点得到一个权重 α；任意位置的均值都是加权和 m(x) = Σ αᵢ k(x, xᵢ)。在 x = 0 处它等于 {mq}；绝对值最大的一项来自编号 {top} 的点（从 0 数起），位于 x = {topX}。',
			vars: (_rt, at) => ({ kd: m(at, 'kd'), n: m(at, 'n'), mq: m(at, 'mq'), top: m(at, 'top'), topX: m(at, 'topX') })
		},
		look: { en: 'The heat map: a stronger color means a stronger link between two points; the bold row of the weight table is point {top}.', zh: '看热力图：颜色越浓，两个点之间的联系越强；权重表里加粗的那一行是编号 {top} 的点。', vars: (_rt, at) => ({ top: m(at, 'top') }) },
		terms: ['cholesky']
	},
	{
		id: 'samples', ...gp('sample'), view: view('samples'),
		title: I('Functions drawn from the posterior', '从后验中抽取的函数'),
		narration: {
			en: 'The posterior is again a distribution over functions, and the three colored curves are drawn from it at random. At x = {xNear}, next to point 0, they lie within {sNear} of each other. At x = {xFar}, the place on the plot farthest from every point, they spread {sFar} apart. The band summarizes all such functions at once.',
			zh: '后验仍然是函数上的分布，三条彩色曲线就是从中随机抽取的。在 x = {xNear}（编号 0 的点旁边），它们彼此相差不超过 {sNear}。在 x = {xFar}，也就是图上离所有点最远的地方，它们相差 {sFar}。误差带一次概括了所有这样的函数。',
			vars: (_rt, at) => ({ xNear: m(at, 'xNear'), sNear: m(at, 'spreadNear'), xFar: m(at, 'xFar'), sFar: m(at, 'spreadFar') })
		},
		look: I('The three colored curves: bundled together at the points, fanning out where there are none. “New samples” in your input draws others.', '看三条彩色曲线：在点附近挤成一束，在没有点的地方散开。在“你的输入”里点“换一组样本”可以抽取别的函数。'),
		terms: ['gp-posterior']
	},
	{
		id: 'evidence', ...gp('evidence'), view: view('evidence'),
		title: I('Which length-scale? Ask the marginal likelihood', '选哪个长度尺度？问边缘似然'),
		narration: {
			en: 'How probable are your y values under the model, averaged over every function it allows? That probability is the marginal likelihood p(y), and its logarithm log p(y) is a sum of three terms: how well the model fits the points ({fit}), a complexity term that is lower for more flexible models ({cpx}), and a constant ({cst}): {lml} for ℓ = {ls}. {verdict}',
			zh: '在这个模型下，把它允许的所有函数都平均掉，你的这些 y 值出现的概率有多大？这个概率叫作边缘似然 p(y)，它的对数 log p(y) 是三项之和：模型对数据点的拟合（{fit}）、模型越灵活就越低的复杂度项（{cpx}），以及一个常数（{cst}）：ℓ = {ls} 时合计 {lml}。{verdict}',
			vars: (_rt, at) => ({ fit: m(at, 'fit'), cpx: m(at, 'cpx'), cst: m(at, 'cst'), lml: m(at, 'lml'), ls: setting(m(at, 'ls')), verdict: verdict(at) })
		},
		look: I('The curve of log p(y) over the slider’s length-scales: yours is ringed, the best one filled.', '看 log p(y) 随滑块上长度尺度变化的曲线：带圈的是你的，实心大点是最好的。'),
		terms: ['marginal-likelihood']
	},
	{
		id: 'tune', ...gp('evidence'), view: view('tune'),
		title: I('Your turn: find a length-scale that fits', '轮到你了：找一个合适的长度尺度'),
		narration: {
			en: 'Move the ℓ slider and watch the band and the curve. Too short, and the mean wiggles between the points while the band balloons; too long, and the mean cannot follow them. Right now log p(y) is {lml}. {verdict}',
			zh: '拖动 ℓ 滑块，看误差带和曲线的变化。太短时，均值在点与点之间来回起伏，误差带也鼓起来；太长时，均值跟不上这些点。现在 log p(y) 是 {lml}。{verdict}',
			vars: (_rt, at) => ({ lml: m(at, 'lml'), verdict: verdict(at) })
		},
		look: I('The result line by the sliders compares your log p(y) with the best one.', '滑块旁那行结果把你的 log p(y) 和最好的作比较。'),
		task: {
			prompt: I('Move the ℓ slider to the length-scale with the highest log marginal likelihood (σf² may stay as it is).', '把 ℓ 滑块移到对数边缘似然最高的长度尺度（σf² 可以不动）。'),
			done: (_rt, at) => m(at, 'drop') <= CLOSE,
			hint: I('Try each position of the ℓ slider in turn and read log p(y) after every move.', '把 ℓ 滑块的每个位置都试一遍，每次都读一下 log p(y)。'),
			waiting: I('Not yet: another length-scale on the slider has a higher log p(y).', '还没有：滑块上还有一个长度尺度的 log p(y) 更高。')
		},
		terms: ['rbf-kernel', 'marginal-likelihood']
	},
	{
		id: 'cost', ...gp('done'), view: view('cost'), kindOf: 'limits',
		title: I('Limits: the cost grows with n³', '局限：计算量按 n³ 增长'),
		narration: {
			en: 'Every prediction rests on the whole n × n matrix, n being the number of points. {mine} Ten times as many points need a hundred times the memory and a thousand times the work: {big} points mean {memBig} numbers and about {opsBig} arithmetic steps, for one setting of ℓ.',
			zh: '每一次预测都离不开整个 n × n 的矩阵，n 是点的个数。{mine}点数变成十倍，内存就要一百倍，计算量要一千倍：{big} 个点意味着 {memBig} 个数和大约 {opsBig} 步运算，而这还只是一个 ℓ 取值的计算量。',
			vars: (_rt, at) => { const o = num(m(at, 'ops'), 'int') as I18n; return { mine: pick(m(at, 'n') > 1, fill(I(`For your {n} points it holds {mem} numbers, and its Cholesky factorization takes about n³/3 ≈ ${o.en} multiplications and subtractions.`, `对你的 {n} 个点，它有 {mem} 个数，它的 Cholesky 分解大约需要 n³/3 ≈ ${o.zh} 次乘法和减法。`), { n: m(at, 'n'), mem: m(at, 'mem') }), I('With your one point it is a single number, and its Cholesky factorization is one square root.', '你只有一个点，矩阵只有一个数，它的 Cholesky 分解就是开一次平方根。')), big: m(at, 'big'), memBig: m(at, 'memBig'), opsBig: m(at, 'opsBig') }; }
		},
		look: { en: 'The two lines of numbers with the plot: your data set, and one of {big} points.', zh: '看图旁的两行数字：你的数据集，以及一个 {big} 个点的数据集。', vars: (_rt, at) => ({ big: m(at, 'big') }) },
		terms: ['cholesky']
	},
	{
		id: 'choice', source: 'short', at: { phase: 'done' }, view: view('choice'), provenance: 'live', kindOf: 'limits',
		compare: { a: 'short', b: 'long', label: [I(`Short length-scale ℓ = ${SHORT}`, `较短的长度尺度 ℓ = ${SHORT}`), I(`Long length-scale ℓ = ${LONG}`, `较长的长度尺度 ℓ = ${LONG}`)] },
		title: I('Limits: one kernel, one smoothness', '局限：一个核，一种平滑程度'),
		narration: {
			en: 'The RBF kernel assumes the function is equally smooth everywhere, so data with a sudden step defeat it either way. With ℓ = {ls1} (first plot) the mean overshoots to {peak1}, though no point is above {ymax}, and the band swells to ±{band1} on average. With ℓ = {ls2} (second plot) the band is only ±{band2} on average, but the mean rounds off the step, misses the points beside it by up to {res2} and still overshoots to {peak2}.',
			zh: 'RBF 核假定函数处处同样平滑，所以遇到突然的台阶，怎么设都不行。ℓ = {ls1} 时（第一张图），均值冲到 {peak1}，可没有一个点高于 {ymax}；误差带平均鼓到 ±{band1}。ℓ = {ls2} 时（第二张图），误差带平均只有 ±{band2}，但均值把台阶磨圆了，台阶两侧的点最多偏差 {res2}，而且仍然冲到 {peak2}。',
			vars: (rt, at) => {
				const b = phaseOf(rt.sources.long?.tape ?? undefined, 'done');
				return { ls1: setting(m(at, 'ls')), peak1: m(at, 'peak'), ymax: m(at, 'ymax'), band1: m(at, 'avgBand'), ls2: setting(b.ls ?? NaN), band2: b.avgBand ?? NaN, res2: b.maxRes ?? NaN, peak2: b.peak ?? NaN };
			}
		},
		look: I('Compare the two plots at x = 0, where the step is: the overshoot just beside it, and the width of the band between the points.', '在台阶所在的 x = 0 附近比较两张图：紧挨着台阶的冲高，以及点与点之间误差带的宽度。'),
		terms: ['rbf-kernel']
	}
];

const lesson: LessonDef = {
	title: I('Gaussian process regression, one point at a time', '高斯过程回归：一次加一个点'),
	summary: I('Start from a cloud of functions, add your points one by one and watch the mean and its uncertainty band follow: the kernel, the kernel matrix, functions drawn from the posterior, the marginal likelihood for choosing the length-scale, and where the method runs out.', '从一团函数出发，把你的点一个一个加进去，看均值和它的误差带如何跟着变化：核函数、核矩阵、从后验中抽取的函数、用来挑选长度尺度的边缘似然，以及这个方法的局限。'),
	steps,
	fixtures: [
		{ id: 'start', label: I('Five points, one wide gap (the page’s start)', '五个点，中间一大段空白（本页的起始输入）'), raw: { preset: 'gap' } },
		{ id: 'tuned', label: I('The start points with the best length-scale ℓ = 2', '起始的点，最好的长度尺度 ℓ = 2'), raw: { preset: 'gap', ls: 2 } },
		{ id: 'wiggle', label: I('A fast wiggle, short length-scale, new samples', '快速起伏，较短的长度尺度，换一组样本'), raw: { preset: 'wiggle', ls: 0.5, seed: 4 } },
		{ id: 'smooth', label: I('A gentle curve, large signal variance', '平缓的曲线，较大的信号方差'), raw: { preset: 'smooth', ls: 3, sf2: 2 } },
		{ id: 'step', label: I('A sudden step', '突然的台阶'), raw: { preset: 'step' } },
		{ id: 'clicked', label: I('Three clicked points, small signal variance', '点出的三个点，较小的信号方差'), raw: { points: [{ x: -4, y: 1 }, { x: 0.3, y: -0.8 }, { x: 4.6, y: 2.5 }], sf2: 0.25, ls: 0.75 } }
	]
};

export default lesson;
