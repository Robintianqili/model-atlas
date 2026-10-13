// regularization: the guided lesson (ten steps; the concept page's primary path). Every number is live: the run source
// trains on the learner's knobs; methods, stop and drop train on the starting knobs (the same points and first weights).
//   split -> curves -> memorize -> predict-size (predict) -> early-stop -> l2 -> dropout (play) -> augment -> task
//   (task) -> too-much (limits)
import type { At } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmtInt } from '../../atlas/core/format.ts';
import { EVERY, PATIENCE, STEPS, TARGET, bestMiss, dec, metricsOf, paramCountOf, setting } from './data.ts';
import { STARTS } from './adapters/knobs/adapter.ts';
import { fieldPic } from './views/field/meta.ts';
import { unitsPic } from './views/units/meta.ts';

const L = (en: string, zh: string) => ({ en, zh });
const LABELS = {
	trainLoss: L('training loss', '训练损失'),
	testLoss: L('test loss', '测试损失')
};
/** the settings the task's hint names: the input panel's L2 and dropout buttons (claims.test.ts: each reaches the goal) */
const HINT_L2 = STARTS.find((s) => s.id === 'weight-decay')!.raw.l2;
const HINT_P = STARTS.find((s) => s.id === 'dropout')!.raw.dropout;
/** test points on the wrong side of the border at this frame */
const wrongOf = (at: At) => {
	const p = fieldPic({ at });
	return p && p.params ? p.wrong.filter(Boolean).length : NaN;
};

const lesson = defineLesson({
	title: L('Overfitting and regularization, step by step', '过拟合与正则化，分步课程'),
	summary: L(
		'Watch a big network learn few noisy points by heart while its test loss climbs, then fight it with live runs: a smaller network, early stopping, an L2 penalty, dropout and augmented data; close the gap yourself and overdo it.',
		'看一个大网络把少量有噪声的点背下来、测试损失却不断上升，再用实时训练来对付它：更小的网络、早停、L2 惩罚、dropout 和数据增强；自己动手缩小差距，最后看看用力过猛会怎样。'
	),
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'split',
			title: L('Training points and test points', '训练点和测试点'),
			source: 'run',
			at: 'first',
			view: { key: 'regularization/field', props: () => ({ field: false }) },
			provenance: 'live',
			terms: ['dataset-split', 'classifier'],
			narration: {
				en: 'Each point belongs to one of two moons, and a network must say which. It learns only from the {nTrain} training points. The other {nTest} points are the test points: it never trains on them, so they show how it does on new points. The noise knob is at {noise}: {scatter}',
				zh: '每个点属于两个月牙中的一个，一个神经网络要判断它属于哪一个。它只从 {nTrain} 个训练点中学习。另外 {nTest} 个点是测试点：网络从不在它们上面训练，所以它们能说明网络在新的点上表现如何。噪声旋钮现在是 {noise}：{scatter}',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const scatter =
						m.noise > 0
							? L('it scatters every point around its moon.', '它让每个点散布在自己的月牙周围。')
							: L('every point lies exactly on its moon.', '每个点都正好落在自己的月牙上。');
					return { nTrain: m.nTrain, nTest: m.nTest, noise: setting(m.noise), scatter };
				}
			},
			look: L('Filled dots are training points, rings are test points; the color is the moon.', '实心点是训练点，空心圈是测试点；颜色表示月牙。')
		},
		{
			id: 'curves',
			title: L('Training loss and test loss, side by side', '训练损失和测试损失，放在一起看'),
			source: 'run',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metrics: ['testLoss', 'trainLoss'], labels: LABELS, log: false }) },
			provenance: 'live',
			terms: ['loss', 'overfitting'],
			narration: {
				en: 'After {n} updates (training steps: small changes to its weights), the training loss is {train} and the test loss is {test}. The test loss was lowest, {best}, after {bestStep} updates, and {verdict}. A network that keeps getting better on its training points while it gets worse on new ones is overfitting.',
				zh: '对权重做了 {n} 次小的更新（也就是训练步）之后，训练损失是 {train}，测试损失是 {test}。测试损失在第 {bestStep} 次更新后最低，是 {best}，{verdict}。网络在训练点上越来越好、在新的点上却越来越差，这就是过拟合。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					// the verdict: the test loss climbed back up while the training loss fell further (overfitting), climbed
					// together with the training loss (a run held back too hard: not overfitting), or stayed close to its low
					const climbed = m.testLoss - m.best > 0.02;
					const verdict = !climbed
						? L('then stayed close to it', '之后一直停在它附近')
						: m.trainLoss < m.bestTrain
							? L('then climbed back up while the training loss fell further', '之后又升了回去，而训练损失还在继续下降')
							: L(
									'then climbed back up, and so did the training loss: this run is held back too hard to fit either set, which is not overfitting',
									'之后又升了回去，训练损失也一样：这次训练被约束得太狠，两组点都拟合不好，这不是过拟合'
								);
					return { n: m.step, train: dec(m.trainLoss), test: m.testLoss, best: m.best, bestStep: m.bestStep, verdict };
				}
			},
			look: L('The gap between the two curves at the end is how much worse the network does on new points.', '两条曲线最后的差距，就是网络在新的点上差了多少。')
		},
		{
			id: 'memorize',
			title: L('Where the border runs', '分界线怎么走'),
			source: 'run',
			at: 'last',
			view: 'regularization/field',
			provenance: 'live',
			terms: ['weight', 'parameter'],
			narration: {
				en: 'The network sorts {trainAcc|pct} of the training points and {testAcc|pct} of the test points correctly: {wrong} test points end up on the wrong side of its border. {why}',
				zh: '网络把 {trainAcc|pct} 的训练点和 {testAcc|pct} 的测试点分对了：有 {wrong} 个测试点落在分界线错误的一侧。{why}',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const P = paramCountOf(m.width);
					const n = fmtInt(m.nTrain);
					// the verdict: held back too hard (under 3 in 4 training points right); close shares (a gap of at most 2
					// points of accuracy); a gap without every training point right; every training point right and a gap, and
					// why (more weights than training points, or few points)
					let why: { en: string; zh: string };
					if (m.trainAcc < 0.75)
						why = L(
							'Even on its own training points it is often wrong: this network is held back too hard to follow the moons.',
							'连在自己的训练点上它也常常出错：这个网络被约束得太狠，分界线跟不上月牙。'
						);
					else if (m.trainAcc - m.testAcc <= 0.02)
						why = L('The two shares are close: this border follows the moons rather than single points.', '这两个比例很接近：这条分界线顺着月牙走，而不是绕着单个点。');
					else if (m.trainAcc < 1)
						why = L(
							'It does better on its training points than on new ones, though it has not learned every training point by heart.',
							'它在训练点上比在新的点上做得好，不过它也没有把每个训练点都背下来。'
						);
					else if (P > m.nTrain)
						why = L(
							`With ${fmtInt(P)} weights and biases for ${n} training points, it can bend its border around single points instead of following the two moons.`,
							`它有 ${fmtInt(P)} 个权重和偏置，却只有 ${n} 个训练点，所以能让分界线绕着单个点弯曲，而不是顺着两个月牙走。`
						);
					else
						why = L(
							`With only ${n} training points, the border is free to run wherever no training point lies, and there it misses the moons.`,
							`只有 ${n} 个训练点，分界线在没有训练点的地方可以随意走，在那里它就偏离了月牙。`
						);
					return { trainAcc: m.trainAcc, testAcc: m.testAcc, wrong: wrongOf(at), why };
				}
			},
			look: L('A second ring marks a test point the network gets wrong.', '外面多一个圈的，是网络分错的测试点。')
		},
		{
			id: 'predict-size',
			title: L('A smaller network on the same points', '同样的点，更小的网络'),
			source: 'methods',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: (_rt, at) => {
					const m = metricsOf(at);
					return {
						metrics: ['testLoss', 'testLossSmall', 'trainLoss', 'trainLossSmall'],
						labels: {
							testLoss: L(`test, ${m.width} units`, `测试，${m.width} 个单元`),
							testLossSmall: L(`test, ${m.widthSmall} units`, `测试，${m.widthSmall} 个单元`),
							trainLoss: L(`training, ${m.width} units`, `训练，${m.width} 个单元`),
							trainLossSmall: L(`training, ${m.widthSmall} units`, `训练，${m.widthSmall} 个单元`)
						}
					};
				}
			},
			provenance: 'live',
			predict: {
				id: 'regularization-size',
				question: L(
					'Both networks train on the same points for the same number of updates. Which one ends with the lower test loss?',
					'两个网络在同样的点上训练同样多次。哪一个最后的测试损失更低？'
				),
				options: [L('The small network', '小网络'), L('The big network', '大网络'), L('Both end about the same', '两个差不多')],
				answer: 0,
				explain: L(
					'The small one. The big network has weights to spare, so it fits the noise of its training points: its training loss goes almost to 0 while its test loss climbs. The small network cannot fit every point, so it has to follow the two moons, and that works better on new points.',
					'小网络。大网络的权重绰绰有余，所以连训练点里的噪声也拟合了进去：它的训练损失几乎降到 0，测试损失却在上升。小网络没法拟合每一个点，只能顺着两个月牙走，而这在新的点上效果更好。'
				)
			},
			narration: {
				en: 'Two networks learn from the same {nTrain} training points for {n} updates: one with {big} units in each hidden layer ({pBig} weights and biases), one with only {small} ({pSmall}). Answer first, then look at their four curves.',
				zh: '两个网络从同样的 {nTrain} 个训练点学习 {n} 次更新：一个每个隐藏层有 {big} 个单元（{pBig} 个权重和偏置），另一个只有 {small} 个（{pSmall} 个）。先回答，再看它们的四条曲线。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { nTrain: m.nTrain, n: m.step, big: m.width, small: m.widthSmall, pBig: paramCountOf(m.width), pSmall: paramCountOf(m.widthSmall) };
				}
			},
			look: L('Compare where the two test curves end; the training curves show why.', '比较两条测试曲线的终点；训练曲线能说明原因。')
		},
		{
			id: 'early-stop',
			title: L('Early stopping: quit while the test loss is low', '早停：趁测试损失还低时停下'),
			source: 'stop',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: (_rt, at) => ({ metrics: ['testLoss', 'trainLoss'], labels: LABELS, refs: [{ y: metricsOf(at).best, label: L('kept', '保留') }] })
			},
			provenance: 'live',
			terms: ['early-stopping', 'dataset-split'],
			narration: {
				en: 'Early stopping checks the test loss every {every} updates and remembers its lowest value, {best}, after {bestStep} updates. Once {patience} checks in a row bring no new low, it stops, here after {stop} updates, and goes back to the weights from update {bestStep}. Training on to update {steps} would have ended at a test loss of {full}. In practice a third set of points, the validation set, does this watching.',
				zh: '早停每隔 {every} 次更新检查一次测试损失，并记下最低值：第 {bestStep} 次更新后的 {best}。一旦连续 {patience} 次检查都没有创新低，它就停下（这里是在第 {stop} 次更新后），并退回到第 {bestStep} 次更新时的权重。如果一直训练到第 {steps} 次更新，测试损失会是 {full}。实际中，这件事由第三组点，也就是验证集来做。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { every: EVERY, best: m.best, bestStep: m.bestStep, patience: PATIENCE, stop: m.step, steps: STEPS, full: m.full };
				}
			},
			look: L('The curves end where training stopped; the dashed line is the test loss of the weights it keeps.', '曲线在训练停下的地方结束；虚线是保留下来的权重的测试损失。')
		},
		{
			id: 'l2',
			title: L('L2: a price on big weights', 'L2：给大权重标上价格'),
			source: 'methods',
			at: 'last',
			view: 'zoom.formula',
			provenance: 'live',
			terms: ['regularization', 'weight-decay'],
			narration: {
				en: 'Regularization holds a network back from fitting its training points too exactly. The L2 penalty does it with a second term in the loss: λ, a strength you choose, times Σw², the sum of every weight squared. With λ = {lam} the network ends with Σw² = {sq}, against {sq0} without the penalty, and a test loss of {test} instead of {test0}. The term\'s slope, 2λw, pulls every weight toward 0 at every update, which is why it is also called weight decay.',
				zh: '正则化就是约束网络，不让它把训练点拟合得太死。L2 惩罚的做法是给损失加上第二项：λ 乘以 Σw²，其中 λ 是你选定的强度，Σw² 是每个权重的平方之和。λ = {lam} 时，网络最后的 Σw² 是 {sq}，没有惩罚时是 {sq0}；测试损失是 {test}，而不是 {test0}。这一项的斜率 2λw 在每次更新时都把每个权重往 0 拉，所以它也叫权重衰减。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { lam: setting(m.l2L2), sq: m.sumSqL2, sq0: m.sumSq, test: m.testLossL2, test0: m.testLoss };
				}
			},
			look: L('The formula is the L2 run\'s loss at its last check: data loss plus λ × Σw², with its real numbers.', '公式是 L2 那次训练在最后一次检查时的损失：数据损失加上 λ × Σw²，用的是真实数值。')
		},
		{
			id: 'dropout',
			title: L('Dropout: units blink off at random', 'Dropout：单元随机熄灭'),
			source: 'drop',
			at: 'last',
			view: 'regularization/units',
			play: { from: 'first', to: 'last', interval: 450, unit: L('update', '次更新') },
			provenance: 'live',
			terms: ['dropout', 'neuron'],
			narration: {
				en: 'Dropout switches hidden units off at random, with a new choice at every update: each unit is off with probability p = {p}. At update {k}, the last one played, {off1} of the {w} units in hidden layer 1 and {off2} in layer 2 are off. The kept ones are multiplied by 1/(1 − p) = {scale}, so the next layer gets the same total on average. When the network is tested or used, nothing is switched off and nothing is scaled.',
				zh: 'Dropout 随机关掉一些隐藏单元，每次更新都重新挑选：每个单元被关掉的概率是 p = {p}。在播放的最后一次更新，也就是第 {k} 次更新中，隐藏层 1 的 {w} 个单元里有 {off1} 个被关掉，隐藏层 2 里有 {off2} 个。保留下来的单元乘以 1/(1 − p) = {scale}，这样下一层收到的总和平均不变。测试或使用网络时，不关掉任何单元，也不做缩放。',
				vars: (_rt, at) => {
					const u = unitsPic({ at });
					const m = metricsOf(at);
					return { p: setting(m.dropout), k: u?.step ?? NaN, off1: u?.off[0] ?? NaN, off2: u?.off[1] ?? NaN, w: u?.width ?? NaN, scale: u?.scale ?? NaN };
				}
			},
			look: L('Press play: a different set of units is off at every update.', '按下播放：每次更新关掉的单元都不一样。')
		},
		{
			id: 'augment',
			title: L('Data augmentation: never the same point twice', '数据增强：同一个点不会出现两次'),
			source: 'methods',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: () => ({ metrics: ['testLoss', 'testLossAug'], labels: { testLoss: L('test loss, plain', '测试损失，原样'), testLossAug: L('test loss, nudged points', '测试损失，挪动的点') } })
			},
			provenance: 'live',
			terms: ['data-augmentation'],
			narration: {
				en: 'Data augmentation makes new training examples out of the old ones. Here every training point is nudged by a small random amount, typically about σ = {sigma}, at every update, so the network never sees exactly the same point twice and has a hard time learning any single one by heart. Its test loss ends at {aug}, against {plain} without nudges. For drawings the nudges are shifts, resizing, turns and thicker or thinner strokes: TinyCNN was trained that way.',
				zh: '数据增强用原有的训练样本造出新的样本。这里每次更新时，每个训练点都被随机挪动一点，幅度通常在 σ = {sigma} 左右，所以网络从来不会两次看到完全相同的点，很难把某一个点背下来。它最后的测试损失是 {aug}，不挪动时是 {plain}。对手绘图来说，挪动就是平移、缩放、旋转和把笔画变粗或变细：TinyCNN 就是这样训练的。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { sigma: setting(m.augAug), aug: m.testLossAug, plain: m.testLoss };
				}
			},
			look: L('Two test curves from the same first weights: plain and with nudged points.', '同样的初始权重下的两条测试曲线：原样的点和挪动过的点。')
		},
		{
			id: 'task',
			title: L('Your turn: close the gap', '轮到你了：缩小差距'),
			source: 'run',
			at: 'last',
			view: 'regularization/field',
			provenance: 'live',
			task: {
				prompt: L(`Change the knobs until the test loss after all ${STEPS} updates is below ${TARGET}.`, `调整旋钮，直到全部 ${STEPS} 次更新之后的测试损失低于 ${TARGET}。`),
				done: (_rt, at) => {
					const m = metricsOf(at);
					return m.step === STEPS && m.testLoss < TARGET;
				},
				hint: L(
					`Try one change at a time: a λ of ${HINT_L2}, a p of ${HINT_P}, more training points or fewer units. The buttons of the input panel set some of them for you.`,
					`一次只改一样：λ 设为 ${HINT_L2}、p 设为 ${HINT_P}、更多训练点，或者更少的单元。输入面板上的按钮可以帮你设好其中几个。`
				),
				waiting: L('Not yet: the test loss at the end is still at or above the goal.', '还没有：最后的测试损失仍然不低于目标。'),
				doneText: L('Done: your network ends below the goal on points it never trained on.', '完成：你的网络在从没训练过的点上，最后低于目标。')
			},
			narration: {
				en: 'With your knobs the network ends at a training loss of {train} and a test loss of {test}. Every change of a knob trains it again from the start, here in your browser. More training points, a smaller network, a penalty λ or dropout p: each one closes the gap in its own way.',
				zh: '用你的旋钮设置，网络最后的训练损失是 {train}，测试损失是 {test}。每改动一个旋钮，网络都会在你的浏览器里从头重新训练。更多训练点、更小的网络、惩罚 λ 或 dropout p：每一种都用自己的方式缩小差距。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { train: dec(m.trainLoss), test: m.testLoss };
				}
			},
			look: L('The goal checks the test loss in the stage’s line of numbers; a second ring marks a test point the network still gets wrong.', '目标检查的是图中那行数字里的测试损失；外面多一个圈的，是仍然分错的测试点。')
		},
		{
			id: 'too-much',
			title: L('Too much of a good thing', '过犹不及'),
			source: 'methods',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: (_rt, at) => {
					const m = metricsOf(at);
					return {
						metrics: ['testLossHeavy', 'trainLossHeavy', 'testLossL2'],
						labels: {
							testLossHeavy: L(`test, λ = ${m.l2Heavy}`, `测试，λ = ${m.l2Heavy}`),
							trainLossHeavy: L(`training, λ = ${m.l2Heavy}`, `训练，λ = ${m.l2Heavy}`),
							testLossL2: L(`test, λ = ${m.l2L2}`, `测试，λ = ${m.l2L2}`)
						}
					};
				}
			},
			provenance: 'live',
			kindOf: 'limits',
			terms: ['underfitting'],
			narration: {
				en: 'Regularization is a trade. With λ = {lam} the penalty keeps every weight small (Σw² = {sq}): the training loss ends at {train} and the test loss at {test}, worse than the {good} of λ = {lamGood}. The border is too simple for the moons: the network underfits. And none of this lesson\'s runs sorts every test point: the best still gets {miss} of the {nTest} wrong, most of them closer to the other moon\'s training points than to their own.',
				zh: '正则化是一种取舍。λ = {lam} 时，惩罚把每个权重都压得很小（Σw² = {sq}）：训练损失最后是 {train}，测试损失是 {test}，比 λ = {lamGood} 时的 {good} 还差。分界线对两个月牙来说太简单了：网络欠拟合。而且这节课里没有哪次训练能分对所有测试点：最好的那次仍然分错了 {nTest} 个中的 {miss} 个，其中大多数离另一个月牙的训练点比离自己月牙的训练点更近。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { lam: setting(m.l2Heavy), sq: m.sumSqHeavy, train: m.trainLossHeavy, test: m.testLossHeavy, good: m.testLossL2, lamGood: setting(m.l2L2), miss: bestMiss(m), nTest: m.nTest };
				}
			},
			look: L('The two curves for the large λ end high and close together: it cannot even fit its training points.', '大 λ 的两条曲线最后都停在高处、彼此靠近：它连训练点都拟合不好。')
		}
	]
});

export default lesson;
