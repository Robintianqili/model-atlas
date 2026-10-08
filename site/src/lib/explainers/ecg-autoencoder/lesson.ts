// ecg-autoencoder: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (dims()), live
// values from the trace of the learner's beat (ecg.ts recOf), the test set's counts at the learner's threshold
// (countAt), the sweep's frames, and measured ones from facts.json ({fact:...}).
//   1 a beat is 140 numbers → 2 the encoder squeezes it into a 4-number code → 3 the decoder rebuilds it → 4 it only
//   learned normal beats (predict: what happens to an odd beat?) → 5 the error at every point → 6 the anomaly score →
//   7 choose a threshold (task) → 8 precision, recall and the ROC curve (the sweep) → 9 draw your own odd beat (task) →
//   10 limits: synthetic data, odd is not dangerous, not medical advice. Claims not computed live: claims.test.ts.
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { KINDS, KIND_LABEL } from './beats.ts';
import { GOAL, THR0, countAt, recOf, regionOf, testsetOf, thrOf } from './ecg.ts';
import { beatRaw, presetKind } from './adapters/series/adapter.ts';
import { SIZES } from './views/scores/meta.ts';

type RT = ExplainerRuntime;
const words = (v: I18n) => v as unknown as TmplVal;

/** structural numbers, read from the spec (never typed into the text) */
export function dims(rt: RT) {
	const out = (id: string) => Number(rt.spec?.layers.find((l) => l.id === id)?.outShape[0] ?? NaN);
	return { n: Number(rt.spec?.inputs?.[0]?.shape[0] ?? NaN), h: out('enc1'), c: out('code') };
}
const argmax = (a: ArrayLike<number>) => Array.from(a).reduce((b, v, i, x) => (v > x[b] ? i : b), 0);
/** the test set at the learner's threshold */
function counts(rt: RT) {
	const s = testsetOf(rt), thr = thrOf(rt.input);
	if (!s) return { thr, total: NaN, tp: NaN, fp: NaN, pos: NaN, neg: NaN, per: NaN, recall: NaN, precision: NaN };
	const a = countAt(s, thr);
	return { thr, total: s.scores.length, tp: a.tp, fp: a.fp, pos: a.pos, neg: a.neg, per: a.of[1], recall: a.recall, precision: a.precision };
}

const lesson: LessonDef = {
	title: { en: 'An autoencoder that flags odd heartbeats', zh: '会标出异常心跳的自编码器' },
	summary: {
		en: 'Follow a heartbeat through an autoencoder trained only on normal beats, see why odd beats come back changed, turn the error into a score, choose a threshold, read the ROC curve, and draw an odd beat yourself.',
		zh: '跟着一次心跳走过只用正常心跳训练的自编码器，看清为什么异常心跳被重建后会走样，把误差变成分数，选阈值，读懂 ROC 曲线，再亲手画一个异常心跳。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'A heartbeat is 140 numbers', zh: '一次心跳就是 140 个数' },
				view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'input' }) },
				narration: {
					en: 'An electrocardiogram (ECG) records the heart’s electrical activity as a trace. Here one heartbeat is {n} numbers, the height of the trace at {n} moments: a small bump (the P wave), a tall spike (the QRS) and a wide bump (the T wave). This beat is {what}; its tallest point is {top}, at point {at} (counting from 0).',
					zh: '心电图（ECG）把心脏的电活动记录成一条曲线。这里一次心跳是 {n} 个数，也就是曲线在 {n} 个时刻的高度：一个小鼓包（P 波）、一个高尖峰（QRS）和一个宽鼓包（T 波）。这次心跳是{what}；它的最高点是第 {at} 个点（从 0 数起），高 {top}。',
					vars: (rt) => {
						const r = recOf(rt), k = presetKind(rt.input), at = r ? argmax(r.x) : NaN;
						const what = k ? { en: `a preset beat (${KIND_LABEL[k].en})`, zh: `一个预设心跳（${KIND_LABEL[k].zh}）` } : { en: 'your own drawing', zh: '你自己画的' };
						return { ...dims(rt), what: words(what), at, top: r ? r.x[at] : NaN };
					}
				},
				look: { en: 'The grey bands show where each part usually is in these beats; the ring marks the tallest point.', zh: '灰色的带子标出这些心跳里各部分通常所在的位置；圆圈标出最高点。' }
			}
		},
		{
			id: 'encode',
			source: 'fwd',
			at: { layer: 'code' },
			select: () => ({ layer: 'code', unit: [0] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'The encoder squeezes it into 4 numbers', zh: '编码器把它挤成 4 个数' },
			narration: {
				en: 'The encoder squeezes. A layer of weighted sums followed by ReLU (keep positive numbers, turn negative ones into 0) turns the {n} heights into {h} numbers; a second layer of weighted sums turns those into just {c}, the code: ({c0}, {c1}, {c2}, {c3}). Everything the decoder will know about this beat has to pass through these {c} numbers, the bottleneck.',
				zh: '编码器负责“挤”。一层加权和再接 ReLU（正数保留，负数变成 0），把 {n} 个高度变成 {h} 个数；第二层加权和再把它们变成仅仅 {c} 个数，即编码：({c0}, {c1}, {c2}, {c3})。解码器对这次心跳能知道的一切，都必须穿过这 {c} 个数，也就是瓶颈。',
				vars: (rt) => {
					const c = recOf(rt)?.code ?? [];
					return { ...dims(rt), c0: c[0] ?? NaN, c1: c[1] ?? NaN, c2: c[2] ?? NaN, c3: c[3] ?? NaN };
				}
			},
			look: { en: 'The {h} products that add up to the first code number, plus the bias b, one more learned number.', zh: '加起来得到第一个编码数的 {h} 个乘积，再加上偏置 b，又一个学来的数。', vars: (rt) => dims(rt) },
			terms: ['dense-layer', 'relu', 'bias', 'bottleneck']
		},
		{
			id: 'decode',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'rebuild' }) },
			provenance: 'live',
			title: { en: 'The decoder rebuilds the beat', zh: '解码器重建出心跳' },
			narration: {
				en: 'The decoder runs the other way: a layer of weighted sums with ReLU grows the {c} code numbers into {h}, and a last layer of weighted sums (no ReLU, so heights can be negative) turns those into {n} heights, the rebuilt beat. Training asked for one thing only: make the rebuild equal the beat that came in, on {fact:trainN} normal beats. Here {how}; compare the tallest points, {top} in the beat and {topR} in its rebuild.',
				zh: '解码器反过来工作：一层加权和加 ReLU 把 {c} 个编码数变成 {h} 个，最后一层加权和（不接 ReLU，所以高度可以是负数）再把它们变成 {n} 个高度，也就是重建的心跳。训练只要求一件事：在 {fact:trainN} 个正常心跳上，让重建等于输入的心跳。这里{how}；比一比最高点：心跳里是 {top}，重建里是 {topR}。',
				vars: (rt) => {
					const r = recOf(rt);
					const how = r ? pick(r.score < THR0, { en: 'they nearly overlap', zh: '两者几乎重合' }, { en: 'they differ in places', zh: '两者有几处明显不同' }) : NaN;
					return { ...dims(rt), how, top: r ? Math.max(...r.x) : NaN, topR: r ? Math.max(...r.xr) : NaN };
				}
			},
			look: { en: 'Green: the beat that came in. Magenta: the rebuild from 4 numbers.', zh: '绿色：输入的心跳。品红色：从 4 个数重建出的心跳。' },
			terms: ['autoencoder', 'encoder-decoder']
		},
		{
			id: 'odd',
			source: 'data',
			at: 'last',
			view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'odd' }) },
			provenance: 'live',
			title: { en: 'It only learned normal beats', zh: '它只学过正常心跳' },
			predict: {
				id: 'ecg-autoencoder-odd',
				question: { en: 'Give the autoencoder a beat whose ST segment, the stretch between the spike and the T wave, is raised. What will it rebuild?', zh: '给自编码器一个 ST 段（尖峰和 T 波之间的那一段）抬高了的心跳。它会重建出什么？' },
				options: [
					{ en: 'The raised beat, as closely as it rebuilds normal beats', zh: '抬高的心跳，和重建正常心跳一样准' },
					{ en: 'A more normal-looking beat, with most of the raise missing', zh: '一个更像正常的心跳，大部分抬高都没了' },
					{ en: 'A flat line: it gives up', zh: '一条平线：它放弃了' }
				],
				answer: 1,
				explain: {
					en: 'It can only draw what its code learned to describe, and no training beat had a raised ST segment. So the rebuild looks like a normal beat, and the gap over the raised stretch is large. That gap is the signal: an autoencoder flags what it cannot rebuild.',
					zh: '它只能画出编码学会描述的东西，而训练心跳里没有一个 ST 段是抬高的。所以重建看起来像正常心跳，抬高那一段上的差距很大。这个差距就是信号：自编码器标记的是它重建不了的东西。'
				}
			},
			narration: {
				en: 'All {fact:trainN} training beats were normal; the first panel of the stage draws {g} of them. They vary: a taller or shorter spike, waves a little earlier or later, more or less noise. The code learned to describe this variation and nothing else. The beat in the second panel is odd: its ST segment, between the spike and the T wave, is raised.',
				zh: '全部 {fact:trainN} 个训练心跳都是正常的；第一张图画出了其中的 {g} 个。它们各有不同：尖峰高一点或矮一点，波形早一点或晚一点，噪声多一点或少一点。编码学会描述的就是这些变化，别的什么都没有。第二张图里的心跳是异常的：它的 ST 段，也就是尖峰和 T 波之间的那一段，被抬高了。',
				vars: (rt, at) => ({ g: (at.state as { beats?: { shape: number[] } } | undefined)?.beats?.shape[0] ?? NaN })
			},
			look: { en: 'In the second panel, the odd beat (green), its rebuild (magenta) and the gap between them.', zh: '第二张图里的异常心跳（绿色）、它的重建（品红色），以及两者之间的差距。' },
			terms: ['autoencoder', 'out-of-distribution']
		},
		{
			id: 'error',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'error' }) },
			provenance: 'live',
			title: { en: 'The error at every point', zh: '每个点上的误差' },
			narration: {
				en: 'Subtract, point by point: the error at each of the {n} points is the beat minus its rebuild. The bars under the plot show them, orange where the beat is above its rebuild and blue where it is below. Your largest error is {worst}, at point {at} (counting from 0), in the {region}.',
				zh: '逐点相减：{n} 个点上的每个误差，都是心跳减去它的重建。图下方的条形画出了这些误差：心跳高于重建处是橙色，低于重建处是蓝色。你最大的误差是 {worst}，在第 {at} 个点（从 0 数起），位于 {region}。',
				vars: (rt) => {
					const r = recOf(rt);
					return { ...dims(rt), worst: r ? r.e[r.worst] : NaN, at: r ? r.worst : NaN, region: r ? words(regionOf(r.worst)) : NaN };
				}
			},
			look: { en: 'The ringed bar is the largest error. Try the presets: where does each odd beat leave its errors?', zh: '圈出的条是最大的误差。试试各个预设：每种异常心跳把误差留在哪里？' },
			terms: ['autoencoder']
		},
		{
			id: 'score',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'score' }) },
			provenance: 'live',
			title: { en: 'One number: the anomaly score', zh: '一个数：异常分数' },
			narration: {
				en: 'One number for the whole beat: square each error (so signs do not cancel and big errors count more), average the {n} squares, then take the square root. That is the anomaly score, the typical size of an error; yours is {score}. On test beats the model never trained on, the median normal beat scores {fact:medianNormal} and the median odd beat {fact:medianOdd}: odd beats tend to score higher, but not always.',
				zh: '给整次心跳一个数：把每个误差平方（这样正负不会抵消，大误差也更受重视），对 {n} 个平方求平均，再开平方根。这就是异常分数，即误差的典型大小；你的是 {score}。在模型训练时没见过的测试心跳上，正常心跳分数的中位数是 {fact:medianNormal}，异常心跳是 {fact:medianOdd}：异常心跳的分数往往更高，但并不总是如此。',
				vars: (rt) => ({ ...dims(rt), score: recOf(rt)?.score ?? NaN })
			},
			look: { en: 'The formula with your numbers, and your score next to the test beats’ medians and the threshold.', zh: '代入你的数的公式，以及你的分数和测试心跳的中位数、阈值的对比。' },
			terms: ['anomaly-score', 'rmse']
		},
		{
			id: 'threshold',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/scores', props: () => ({ mode: 'hist' }) },
			provenance: 'precomputed',
			title: { en: 'Choose a threshold', zh: '选一个阈值' },
			narration: {
				en: 'Where does normal end? The histogram shows the scores of {total} test beats the model never trained on: {neg} normal and {pos} odd, {per} of each kind of anomaly. A beat that scores at or above the threshold is flagged. At {thr} the model catches {tp} odd beats and flags {fp} normal ones by mistake, the false alarms; the two piles overlap, so no threshold separates them completely.',
				zh: '“正常”到哪里为止？直方图画出了模型训练时没见过的 {total} 个测试心跳的分数：正常 {neg} 个，异常 {pos} 个，每种异常 {per} 个。分数达到或超过阈值的心跳会被标记。阈值为 {thr} 时，模型抓到 {tp} 个异常心跳，又错标了 {fp} 个正常心跳，这就是误报；两堆分数有重叠，所以没有哪个阈值能把它们完全分开。',
				vars: (rt) => counts(rt)
			},
			look: { en: 'Blue: normal test beats. Orange: odd ones. Everything right of the black line is flagged.', zh: '蓝色：正常测试心跳。橙色：异常测试心跳。黑线右边的都会被标记。' },
			task: {
				prompt: { en: 'Move the threshold until the model catches at least 150 of the 200 odd beats with no more than 15 false alarms.', zh: '移动阈值，让模型在 200 个异常心跳里至少抓到 150 个，同时误报不超过 15 个。' },
				done: (rt) => {
					const s = testsetOf(rt);
					if (!s) return false;
					const a = countAt(s, thrOf(rt.input));
					return a.tp >= GOAL.tp && a.fp <= GOAL.fp;
				},
				waiting: { en: 'Not yet: lower the threshold to catch more, and watch the false alarms.', zh: '还没有：调低阈值能抓到更多，同时留意误报。' },
				hint: { en: 'Move it a little at a time (the arrow keys move one step): each step left catches some odd beats and lets in some normal ones.', zh: '一点一点地移（方向键每次移一格）：每往左一格，都会多抓到一些异常心跳，也会放进一些正常心跳。' }
			},
			terms: ['decision-threshold']
		},
		{
			id: 'roc',
			source: 'sweep',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 25 },
			view: { key: 'ecg-autoencoder/scores', props: () => ({ mode: 'roc' }) },
			provenance: 'live',
			title: { en: 'Precision, recall and the ROC curve', zh: '精确率、召回率和 ROC 曲线' },
			narration: {
				en: 'The ROC curve shows every threshold at once: up is recall, the share of odd beats caught; right is the false-alarm rate, the share of normal beats flagged. Sliding the threshold down past an odd beat steps the curve up, past a normal beat right. At your threshold (the dot), recall is {rec|pct} and the false-alarm rate {fpr|pct}; precision, the share of flagged beats that are odd, is {prec|pct}. The area under the curve, {auc}, is the chance that a random odd beat outscores a random normal one.',
				zh: 'ROC 曲线把所有阈值一次画出来：往上是召回率，即抓到的异常心跳所占比例；往右是误报率，即被标记的正常心跳所占比例。阈值往下滑过一个异常心跳，曲线向上走一步；滑过一个正常心跳，曲线向右走一步。在你的阈值（圆点）处，召回率是 {rec|pct}，误报率是 {fpr|pct}；精确率，即被标记的心跳里真异常的比例，是 {prec|pct}。曲线下面积 {auc}，就是随机挑一个异常心跳和一个正常心跳，异常心跳得分更高的概率。',
				vars: (rt, at) => {
					const c = counts(rt);
					return { rec: c.recall, fpr: c.fp / c.neg, prec: c.precision, auc: at.frame?.metrics?.auc ?? NaN };
				}
			},
			look: { en: 'Press Play to watch the threshold slide down. The dashed diagonal is guessing.', zh: '按“播放”看阈值往下滑。虚线对角线代表瞎猜。' },
			terms: ['roc-curve', 'precision-recall']
		},
		{
			id: 'draw',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/beat', props: () => ({ mode: 'draw' }) },
			provenance: 'live',
			title: { en: 'Draw your own odd beat', zh: '画出你自己的异常心跳' },
			narration: {
				en: 'Now make an odd beat yourself. In the input box, drag across the curve to redraw a stretch: flatten the P wave, widen the spike, lift the stretch after it, or scribble some noise. This beat scores {score} against the threshold {thr}, so it {verdict}. Which change does the model notice most easily?',
				zh: '现在亲手做一个异常心跳。在输入框里沿着曲线拖动，重画其中一段：把 P 波抹平、把尖峰加宽、把它后面那段抬高，或者乱画一些噪声。这次心跳得分 {score}，阈值是 {thr}，所以它{verdict}。模型最容易注意到哪种改动？',
				vars: (rt) => {
					const r = recOf(rt), thr = thrOf(rt.input);
					return { score: r?.score ?? NaN, thr, verdict: r ? pick(r.score >= thr, { en: 'is flagged as odd', zh: '被标为异常' }, { en: 'passes as normal', zh: '被当作正常' }) : NaN };
				}
			},
			look: { en: 'The verdict under the plot updates as soon as you finish a stroke.', zh: '每画完一笔，图下方的结论就会更新。' },
			task: {
				prompt: { en: 'Redraw part of a beat until the model flags it as odd.', zh: '重画心跳的一部分，直到模型把它标为异常。' },
				done: (rt) => {
					const r = recOf(rt);
					return !!r && presetKind(rt.input) === null && r.score >= thrOf(rt.input);
				},
				waiting: { en: 'Not yet: draw on the beat in the input box until its score reaches the threshold.', zh: '还没有：在输入框里的心跳上画，直到它的分数达到阈值。' },
				doneText: { en: 'Flagged: the model could not rebuild what you drew.', zh: '已标记：模型重建不了你画的东西。' }
			},
			terms: ['anomaly-score']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'ecg-autoencoder/scores', props: () => ({ mode: 'limits' }) },
			provenance: 'precomputed',
			kindOf: 'limits',
			title: { en: 'Limits: synthetic beats, and odd is not dangerous', zh: '局限：合成的心跳，异常不等于危险' },
			narration: {
				en: 'These beats are synthetic bumps plus noise, far simpler than real ECGs, which this model has never seen. It flags what it cannot rebuild, not what is dangerous: a noisy recording scores high, while at the starting threshold {thr0}, {pPass} of {nP} beats with a shrunk or missing P wave and {stPass} of {nST} with a raised ST segment pass as normal. With {big} code numbers, normal beats are rebuilt more closely but odd beats stand out less: the AUC falls to {fact:c16Auc}. None of this is medical advice.',
				zh: '这些心跳是由鼓包加噪声合成的，比真实的心电图简单得多，而真实的心电图这个模型一次也没见过。它标记的是它重建不了的东西，而不是危险的东西：噪声大的记录分数很高，而在起始阈值 {thr0} 下，P 波变小或消失的 {nP} 个心跳里有 {pPass} 个、ST 段抬高的 {nST} 个里有 {stPass} 个被当作正常。编码用 {big} 个数时，正常心跳重建得更准，异常心跳却不那么显眼了：AUC 跌到 {fact:c16Auc}。这里的一切都不是医疗建议。',
				vars: (rt) => {
					const s = testsetOf(rt), a = s ? countAt(s, THR0) : null, P = KINDS.indexOf('noP'), ST = KINDS.indexOf('stUp');
					return { big: SIZES[SIZES.length - 1].c, thr0: THR0, nP: a ? a.of[P] : NaN, pPass: a ? a.of[P] - a.per[P] : NaN, nST: a ? a.of[ST] : NaN, stPass: a ? a.of[ST] - a.per[ST] : NaN };
				}
			},
			look: { en: 'Four code sizes, trained the same way: the rebuild error on normal beats keeps falling, but the AUC peaks at 4.', zh: '用同样方法训练的四种编码大小：正常心跳的重建误差一路下降，AUC 却在 4 处最高。' },
			terms: ['bottleneck', 'out-of-distribution']
		}
	],
	fixtures: [
		{ id: 'drawn-st', label: { en: 'the normal preset with its ST segment drawn higher, threshold 0.025', zh: '把正常预设的 ST 段画高，阈值 0.025' }, raw: { y: beatRaw('normal', 9).y!.map((v, i) => (i >= 60 && i < 76 ? v + 0.15 : v)), thr: 0.025 } }
	]
};

export default lesson;
