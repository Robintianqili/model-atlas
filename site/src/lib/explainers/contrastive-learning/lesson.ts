// contrastive-learning: the guided lesson (eleven steps; the concept page's primary path), in the page's order:
//   hook     pairs (task: add a pair of your own)
//   toy      cell -> softmax -> hot (predict) -> drag (task) -> gradient
//   real     train (play through the checkpoints of a training run) -> learned
//   compare  simclr -> invariance
//   pitfalls limits (false negatives)
// Every number is live: nce runs the learner's batch through the shipped TinyCLIP, progress through the checkpoints of a
// re-run of its training, simclr through its image tower; the re-run's test numbers come from facts.json.
import type { At, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { num, pick, plural } from '../../atlas/lesson/say.ts';
import { BATCHES, hasOwnPair, pairsOf, tauOf } from './adapters/batch/adapter.ts';
import { field, offMax, tauCurve } from './data.ts';
import { fmtSetting } from '../../atlas/core/format.ts';

/** a setting as set (0.2, not 0.200); NaN when missing, so lint still reports it */
export const st = (v: number) => (Number.isFinite(v) ? fmtSetting(v) : NaN);

const MATRIX = 'contrastive-learning/matrix';
const view = (mode: string) => ({ key: MATRIX, props: () => ({ mode }) });
/** metric `k` of the first frame of `phase` on the step's tape */
export function fm(at: At, phase: string, k: string): number {
	const tp = at.tape;
	const i = tp ? tp.find({ phase }) : -1;
	return i >= 0 ? Number(tp!.frame(i).metrics?.[k] ?? NaN) : NaN;
}
/** "after 30 of its 4,680 updates", or "after the last of its 4,680 updates" */
export function whenOf(u: number, last: number): I18n | number {
	const U = num(u, 'int'), T = num(last, 'int');
	if (typeof U !== 'object' || typeof T !== 'object') return NaN;
	return u === last ? { en: `after the last of its ${T.en} updates`, zh: `全部 ${T.zh} 次更新之后` } : { en: `after ${U.en} of its ${T.en} updates`, zh: `${T.zh} 次更新中的第 ${U.zh} 次之后` };
}
const capOf = (rt: { input: unknown }, j: number) => pairsOf(rt.input)[j]?.caption ?? '';
const q = (s: string): I18n => ({ en: `“${s}”`, zh: `“${s}”` });
/** the wrong cell (i, j) with the largest row share */
export function topWrong(at: At) {
	const P = field(at, 'P');
	if (!P) return { i: NaN, j: NaN, p: NaN };
	const b = offMax(P.data, P.shape[0]);
	return { i: b.i, j: b.j, p: b.v };
}

const lesson = defineLesson({
	title: { en: 'Contrastive learning, step by step', zh: '对比学习，分步课程' },
	summary: {
		en: 'Build a batch of drawings and captions, read TinyCLIP’s table of cosines, work out the InfoNCE loss and drag its temperature, watch training turn the table into a diagonal, then make SimCLR’s pairs from two distorted views and meet the false negatives the loss cannot see.',
		zh: '搭一批画和描述，读懂 TinyCLIP 的余弦表，算出 InfoNCE 损失并拖动它的温度，看训练把这张表变成一条对角线，再用两个变形视图造出 SimCLR 的正样本对，并认识损失看不见的伪负样本。'
	},
	fixtures: BATCHES.map((b) => ({ id: b.id, label: b.label, raw: b.raw })),
	steps: [
		{
			id: 'pairs',
			title: { en: 'A batch of pairs, and no labels', zh: '一批图文对，没有标签' },
			source: 'nce',
			at: { phase: 'scores' },
			view: view('cos'),
			provenance: 'live',
			terms: ['contrastive-learning', 'embedding', 'cosine-similarity', 'positive-pair'],
			narration: {
				en: 'Your batch has {n} pairs, each a drawing and the caption written for it. Nobody labels a drawing “circle”: the only signal is which caption came with which drawing. TinyCLIP turns every drawing and every caption into a vector of {d} numbers with length 1, and cell (i, j), counting from 0, is the cosine of drawing i and caption j. The true pairs on the diagonal average {diag}; the other cells average {off}.',
				zh: '你的这一批有 {n} 对，每对是一幅画和为它写的描述。没有人给画贴上“圆”这样的标签，唯一的信号是哪句描述和哪幅画在一起。TinyCLIP 把每幅画和每句描述都变成一个由 {d} 个数组成、长度为 1 的向量；格子 (i, j)（从 0 数起）就是画 i 与描述 j 的余弦。对角线上正确的对平均为 {diag}，其余格子平均为 {off}。',
				vars: (_rt, at) => ({ n: fm(at, 'scores', 'N'), d: fm(at, 'scores', 'D'), diag: fm(at, 'scores', 'diagMean'), off: fm(at, 'scores', 'offMean') })
			},
			look: { en: 'The outlined cells on the diagonal are the true pairs, the positives; every other cell is a negative.', zh: '对角线上带框的格子是正确的对，也就是正样本；其他每个格子都是负样本。' },
			task: {
				prompt: { en: 'Draw something in the side panel, give it a caption and add it to the batch: a new row and column appear. Rewriting a caption counts too.', zh: '在侧栏里画点什么，给它写一句描述并加入这一批：表里会多出一行和一列。改写一句已有的描述也算。' },
				done: (rt) => hasOwnPair(rt.input),
				waiting: { en: 'Not yet: the batch holds only the ready-made pairs.', zh: '还没有：这一批里只有现成的图文对。' },
				doneText: { en: 'Done: your pair is in the table.', zh: '完成：你的图文对已经在表里了。' }
			}
		},
		{
			id: 'cell',
			title: { en: 'One cell, worked out', zh: '算出一个格子' },
			source: 'nce',
			at: { phase: 'scores' },
			view: view('pair'),
			provenance: 'live',
			terms: ['dot-product'],
			narration: {
				en: 'Take cell (0, 0). Multiply the {d} numbers of drawing 0’s vector by those of caption 0’s, number by number, and add up the products: {cos}. Both vectors have length 1, so this sum is the cosine of the angle between them: 1 when they point the same way, 0 at right angles, −1 when they are opposite.',
				zh: '看格子 (0, 0)。把画 0 的向量和描述 0 的向量的 {d} 个数逐个相乘，再把乘积加起来：{cos}。两个向量的长度都是 1，所以这个和就是它们夹角的余弦：方向相同时是 1，互相垂直时是 0，方向相反时是 −1。',
				vars: (_rt, at) => ({ d: fm(at, 'scores', 'D'), cos: fm(at, 'scores', 'cos00') })
			},
			look: { en: 'Where both strips have the same colour, the product is positive and raises the cosine.', zh: '两条色带颜色相同的地方，乘积为正，会让余弦变大。' }
		},
		{
			id: 'softmax',
			title: { en: 'Each row is a multiple-choice question', zh: '每一行都是一道选择题' },
			source: 'nce',
			at: { phase: 'softmax' },
			view: view('P'),
			provenance: 'live',
			terms: ['temperature', 'softmax', 'cross-entropy', 'infonce'],
			narration: {
				en: 'Row 0 becomes a choice among the {n} captions: divide each cosine by the temperature τ = {tau}, then take a softmax. Drawing 0 gives its own caption a share of {p}, and the loss of that row is −log of that share: {l0}. InfoNCE averages this over every row, and over every column too (each caption choosing its drawing): {loss} for your batch.',
				zh: '第 0 行变成在 {n} 句描述里做选择：把每个余弦除以温度 τ = {tau}，再做 softmax。画 0 给它自己那句描述的份额是 {p}，这一行的损失就是这个份额的 −log：{l0}。InfoNCE 对每一行取平均，也对每一列取平均（每句描述挑选它的画）：你这一批的损失是 {loss}。',
				vars: (_rt, at) => ({ n: fm(at, 'scores', 'N'), tau: st(fm(at, 'scores', 'tau')), p: num(fm(at, 'softmax', 'p00'), 'pct'), l0: fm(at, 'softmax', 'l0') + 0 /* −log 1 is −0: print 0 */, loss: fm(at, 'softmax', 'loss') })
			},
			look: { en: 'The shares in each row add up to one; the toggle switches to the column softmax.', zh: '每一行的份额加起来等于 1；切换按钮可以改看列 softmax。' }
		},
		{
			id: 'hot',
			title: { en: 'Turn the temperature up', zh: '把温度调高' },
			source: 'nce',
			at: { phase: 'softmax' },
			view: view('tau'),
			provenance: 'live',
			predict: {
				id: 'contrastive-learning-hot',
				question: { en: 'As τ grows very large, what does the loss approach?', zh: '当 τ 变得非常大时，损失会趋近于什么？' },
				options: [
					{ en: '0, a perfect score', zh: '0，满分' },
					{ en: 'log N: the loss of a blind guess among the N captions', zh: 'log N：在 N 句描述里瞎猜的损失' },
					{ en: 'It grows without any limit', zh: '它会无限增大' }
				],
				answer: 1,
				explain: {
					en: 'A huge τ squeezes every score towards 0, so each row’s softmax becomes even: 1/N for every caption. The loss becomes −log(1/N) = log N whatever the model has learned: the dashed line in the stage.',
					zh: 'τ 很大时，每个分数都被压向 0，每一行的 softmax 变得平均：每句描述都是 1/N。不管模型学到了什么，损失都变成 −log(1/N) = log N，也就是舞台里的虚线。'
				}
			},
			narration: {
				en: 'At τ = {tau} your batch’s loss is {loss}. The temperature divides every cosine before the softmax, so it decides how sharp the choice in each row can be. Now picture τ ten or a hundred times larger. Answer first, then look at the curve.',
				zh: '在 τ = {tau} 时，你这一批的损失是 {loss}。在 softmax 之前，每个余弦都要除以温度，所以温度决定了每一行的选择能有多“尖锐”。现在想象 τ 变成十倍、一百倍。先回答，再看曲线。',
				vars: (_rt, at) => ({ tau: st(fm(at, 'scores', 'tau')), loss: fm(at, 'softmax', 'loss') })
			},
			look: { en: 'The dot is your τ; the dashed line is a blind guess, log N.', zh: '圆点是你的 τ；虚线是瞎猜的损失 log N。' }
		},
		{
			id: 'drag',
			title: { en: 'Drag the temperature', zh: '拖动温度' },
			source: 'nce',
			at: { phase: 'softmax' },
			view: view('tau'),
			provenance: 'live',
			task: {
				prompt: { en: 'Drag τ in the side panel to either end of the slider: 0.02 or lower, or 0.5 or higher.', zh: '在侧栏里把 τ 拖到滑块的任意一端：0.02 或更低，或者 0.5 或更高。' },
				done: (rt) => tauOf(rt.input) <= 0.02 || tauOf(rt.input) >= 0.5,
				hint: { en: 'Compare the loss at both ends with the dashed line.', zh: '把两端的损失和虚线比一比。' },
				waiting: { en: 'Not yet: τ is still in the middle of the slider.', zh: '还没有：τ 还在滑块中间。' },
				doneText: { en: 'Done: you have seen the loss at an end of the range.', zh: '完成：你看到了范围一端的损失。' }
			},
			narration: {
				en: 'At τ = {tau} the loss is {loss}, and a blind guess among {n} scores {chance}. {verdict}',
				zh: '在 τ = {tau} 时损失是 {loss}，在 {n} 个里瞎猜的损失是 {chance}。{verdict}',
				vars: (_rt, at) => {
					const C = field(at, 'C');
					const n = fm(at, 'scores', 'N');
					const c = C ? tauCurve(C.data, n) : null;
					const wrong = 2 * n - fm(at, 'softmax', 'rowRight') - fm(at, 'softmax', 'colRight');
					const best = c ? num(c.best.loss) : NaN, bestTau = c ? num(c.best.tau) : NaN;
					const t = (x: unknown) => (typeof x === 'object' && x ? (x as I18n) : { en: '…', zh: '…' });
					// wrong > 0 with the lowest point at the slider's end: ties (two equal captions) put a floor under the loss
					const floor = !!c && wrong > 0 && c.best.atEdge;
					return {
						tau: st(fm(at, 'scores', 'tau')), loss: fm(at, 'softmax', 'loss'), n, chance: fm(at, 'softmax', 'chance'),
						verdict: !c ? NaN : wrong === 0
							? { en: `Every drawing ranks its own caption first and every caption its own drawing, so the sharper the softmax, the lower the loss, all the way down to the slider’s lowest τ.`, zh: `每幅画都把自己的描述排第一，每句描述也都把自己的画排第一，所以 softmax 越尖锐，损失越低，一直到滑块最低的 τ。` }
							: floor
								? { en: `${wrong} of the ${2 * n} rows and columns rank a wrong partner first or tie with it, so the loss cannot fall to 0: even at the slider’s lowest τ it is ${t(best).en}.`, zh: `${2 * n} 个行和列中有 ${wrong} 个把错误的搭档排在第一，或者与它打平，所以损失降不到 0：即使在滑块最低的 τ，它仍是 ${t(best).zh}。` }
								: { en: `The loss is lowest, ${t(best).en}, at τ = ${t(bestTau).en}. Below that it climbs again: ${wrong} of the ${2 * n} rows and columns rank a wrong partner first or tie with it, and a sharp softmax punishes that hard.`, zh: `损失在 τ = ${t(bestTau).zh} 处最低，为 ${t(best).zh}。再往下它又会上升：${2 * n} 个行和列中有 ${wrong} 个把错误的搭档排在第一，或者与它打平，而尖锐的 softmax 会重罚这种错误。` }
					};
				}
			},
			look: { en: 'Watch the dot slide along the curve as you drag.', zh: '拖动时，看圆点沿着曲线滑动。' }
		},
		{
			id: 'gradient',
			title: { en: 'Pull the diagonal up, push the rest down', zh: '把对角线拉上来，把其余的推下去' },
			source: 'nce',
			at: { phase: 'grad' },
			view: view('G'),
			provenance: 'live',
			terms: ['gradient', 'hard-negative'],
			narration: {
				en: 'The gradient of the loss on each score says which way training moves it. The true pair (0, 0) gets {g00}: {sign} Every wrong pair gets a positive gradient: the shares the softmax wrongly gives it in its row and in its column, added and divided by 2N = {nn}, so it is pushed down. The hardest push here is drawing {gi} with caption {gj}, at {gmax}: the hardest negative.',
				zh: '损失对每个分数的梯度，说明训练会把它往哪边推。正确的对 (0, 0) 得到 {g00}：{sign}每个错误的对都得到正的梯度：softmax 在它所在的行和列里错给它的份额，相加后再除以 2N = {nn}，所以它被往下推。这里推得最狠的是画 {gi} 和描述 {gj}，梯度 {gmax}：它就是最难的负样本。',
				vars: (_rt, at) => ({ g00: fm(at, 'grad', 'g00') + 0, nn: { v: 2 * fm(at, 'scores', 'N'), fmt: 'int' }, sign: pick(fm(at, 'grad', 'g00') < -1e-9, { en: 'a negative number, so a step against the gradient raises that score.', zh: '一个负数，所以逆着梯度走一步会让这个分数变大。' }, { en: 'practically nothing, because the softmax already gives that pair almost all of its share.', zh: '微乎其微，因为 softmax 已经把几乎全部份额给了这一对。' }), gi: { v: fm(at, 'grad', 'gOffI'), fmt: 'int' }, gj: { v: fm(at, 'grad', 'gOffJ'), fmt: 'int' }, gmax: fm(at, 'grad', 'gOffMax') })
			},
			look: { en: 'Blue cells (negative gradient) are pulled up, orange cells (positive) pushed down.', zh: '蓝色格子（梯度为负）被往上拉，橙色格子（梯度为正）被往下推。' }
		},
		{
			id: 'train',
			title: { en: 'From noise to a diagonal', zh: '从噪声到对角线' },
			source: 'progress',
			at: 'last',
			view: view('progress'),
			play: { from: 'first', to: 'last', interval: 900, unit: { en: 'checkpoint', zh: '检查点 {n}' } },
			provenance: { mode: 'precomputed', note: { en: 'Weights saved during a re-run of TinyCLIP’s training (CPU, seed 0); your batch runs through them live.', zh: '权重保存自 TinyCLIP 的一次重新训练（CPU，种子 0）；你的这一批在浏览器里实时通过它们。' } },
			terms: ['checkpoint', 'batch'],
			narration: {
				en: 'These weights come from a TinyCLIP training run, saved {when}, with your batch run through them. {right} the highest cosine. The diagonal averages {diag}, the other cells {off}, and the loss at the run’s own τ = {tauk} is {loss}.',
				zh: '这些权重来自 TinyCLIP 的一次训练，保存于{when}，你的这一批通过它们算出了这张表。{right}排在余弦最高的位置。对角线平均为 {diag}，其余格子为 {off}，按这次训练自己的 τ = {tauk} 算，损失是 {loss}。',
				vars: (_rt, at) => {
					const m = (k: string) => Number(at.frame?.metrics?.[k] ?? NaN);
					const n = m('N');
					return {
						when: whenOf(m('update'), m('last')),
						right: plural(m('right'), { one: `{n} of the ${n} drawings gives its own caption`, many: `{n} of the ${n} drawings give their own caption`, zh: `${n} 幅画中有 {n} 幅把自己的描述` }, 'int'),
						diag: m('diag'), off: m('off'), tauk: m('tau'), loss: m('loss')
					};
				}
			},
			look: { en: 'Press play, or drag the stage’s slider from the first checkpoint: watch the outlined diagonal separate from the rest.', zh: '按播放，或者从第一个检查点开始拖动舞台的滑块：看带框的对角线如何和其余格子分开。' }
		},
		{
			id: 'learned',
			title: { en: 'The temperature is learned too', zh: '温度也是学出来的' },
			source: 'progress',
			at: 'last',
			view: view('progress'),
			provenance: { mode: 'precomputed', note: { en: 'Test numbers measured on the re-run’s checkpoints (pipeline/contrastive-learning/train.py).', zh: '测试数据是在这次重新训练的检查点上测得的（pipeline/contrastive-learning/train.py）。' } },
			terms: ['temperature'],
			narration: {
				en: 'CLIP does not fix τ: it learns s = 1/τ along with the other weights. This run started at τ = {tau0} and ended at {fact:tauLast}; the TinyCLIP on its own page learned {fact:tauClip}. In {fact:batches6|int} test batches of 6 pairs, a drawing’s own caption came first {acc0} of the time before training and {acc1} after.',
				zh: 'CLIP 不把 τ 固定下来：它把 s = 1/τ 和其他权重一起学。这次训练从 τ = {tau0} 开始，结束时是 {fact:tauLast}；TinyCLIP 模型页上的那个模型学到的是 {fact:tauClip}。在 {fact:batches6|int} 批测试图文对上（每批 6 对），训练前画自己的描述排第一的比例是 {acc0}，训练后是 {acc1}。',
				vars: (rt) => ({ tau0: st(Number(rt.num('fact:tauFirst').toPrecision(3))), acc0: num(rt.num('fact:acc6First'), 'pct'), acc1: num(rt.num('fact:acc6Last'), 'pct') })
			},
			look: { en: 'The τ in the stage’s readout line changes from checkpoint to checkpoint.', zh: '舞台读数行里的 τ 随检查点变化。' }
		},
		{
			id: 'simclr',
			title: { en: 'No captions at all: two views of one drawing', zh: '完全不用描述：同一幅画的两个视图' },
			source: 'simclr',
			at: { phase: 'loss' },
			view: view('simclr'),
			provenance: { mode: 'live', note: { en: 'TinyCLIP’s image tower stands in for a SimCLR encoder: it was trained with captions, not with this loss.', zh: '这里用 TinyCLIP 的图像塔代替 SimCLR 的编码器：它是用描述训练的，不是用这个损失训练的。' } },
			terms: ['data-augmentation', 'self-supervised'],
			narration: {
				en: 'SimCLR needs no captions. Each drawing is distorted twice, by a small random turn, stretch and shift, and the two views form the true pair; the other {neg} views are negatives. Through TinyCLIP’s image tower the two views of drawing 0 have a cosine of {pos}, while the closest views of two different drawings reach {negMax}. The same kind of loss over all {m} views, at τ = {tau}, is {loss}.',
				zh: 'SimCLR 不需要描述。每幅画被随机地稍微转一转、拉一拉、挪一挪，变形两次，这两个视图就是正确的一对；其余 {neg} 个视图都是负样本。经过 TinyCLIP 的图像塔，画 0 的两个视图余弦为 {pos}，而来自不同画的视图之间最高达到 {negMax}。对全部 {m} 个视图用同样的损失，在 τ = {tau} 时为 {loss}。',
				vars: (_rt, at) => ({ neg: { v: fm(at, 'views', 'M') - 2, fmt: 'int' }, pos: fm(at, 'views', 'pos0'), negMax: fm(at, 'views', 'negMax'), m: { v: fm(at, 'views', 'M'), fmt: 'int' }, tau: st(fm(at, 'loss', 'tau')), loss: fm(at, 'loss', 'loss') })
			},
			look: { en: 'Views 2i and 2i + 1 come from drawing i; the outlined cells mark each view’s partner.', zh: '视图 2i 和 2i + 1 来自画 i；带框的格子标出每个视图的搭档。' }
		},
		{
			id: 'invariance',
			title: { en: 'The distortions decide what is ignored', zh: '变形方式决定了忽略什么' },
			source: 'simclr',
			at: { phase: 'shift' },
			view: view('shift'),
			provenance: 'live',
			narration: {
				en: 'What the two views of a pair share, the encoder learns to keep; what differs, it learns to ignore. TinyCLIP’s captions name the place, such as “on the left”, so its image tower keeps the place: {verdict} If SimCLR trained with views moved across as true pairs, it would pull exactly those two vectors together, and the place would be lost.',
				zh: '一对视图的共同之处，就是编码器学会保留的东西；不同之处，它学会忽略。TinyCLIP 的描述会说出位置（比如“on the left”，即“在左边”），所以它的图像塔保留了位置：{verdict}如果 SimCLR 把移到另一边的视图当作正确的一对来训练，它就会把这两个向量拉到一起，位置信息也就丢了。',
				vars: (_rt, at) => {
					const s = fm(at, 'shift', 'shift0'), j = fm(at, 'shift', 'jitter0');
					const S = num(s), J = num(j);
					const t = (x: unknown) => (typeof x === 'object' && x ? (x as I18n) : { en: '…', zh: '…' });
					return {
						verdict: !Number.isFinite(s + j) ? NaN : pick(s < j - 0.05,
							{ en: `drawing 0 moved across has a cosine of ${t(S).en} with itself as drawn, while a small distortion keeps ${t(J).en}.`, zh: `画 0 移到另一边后，与原样的余弦只有 ${t(S).zh}，而小小的变形能保持 ${t(J).zh}。` },
							{ en: `here, though, drawing 0 moved across keeps a cosine of ${t(S).en} with itself, about as much as a small distortion (${t(J).en}), because it barely moves or has little ink.`, zh: `不过这里画 0 移到另一边后与原样的余弦仍有 ${t(S).zh}，和小小的变形差不多（${t(J).zh}），因为它几乎没动，或者墨迹很少。` })
					};
				}
			},
			look: { en: 'Compare the cosine of the distorted view with that of the moved one.', zh: '比较变形视图和移过去的那幅各自的余弦。' }
		},
		{
			id: 'limits',
			title: { en: 'What the loss cannot see: false negatives', zh: '损失看不见的东西：伪负样本' },
			source: 'nce',
			at: { phase: 'grad' },
			view: view('P'),
			provenance: 'live',
			kindOf: 'limits',
			terms: ['positive-pair'],
			narration: {
				en: 'InfoNCE treats every other caption in the batch as wrong, even one that also fits. Here drawing {i} gives caption {j}, {cap}, a share of {p}, and the loss pushes that score down with a gradient of {g}, just as for an unrelated caption. If that caption describes drawing {i} too, the push is a mistake: a false negative. A bigger batch holds more negatives, and more such accidents.',
				zh: 'InfoNCE 把同一批里其他每句描述都当作错的，哪怕它也说得通。这里画 {i} 给描述 {j}（{cap}）的份额是 {p}，损失会用 {g} 的梯度把这个分数往下推，和对待毫不相干的描述一样。如果这句描述其实也符合画 {i}，这一推就推错了：这是伪负样本。批越大，负样本越多，这样的意外也越多。',
				vars: (rt, at) => {
					const w = topWrong(at);
					const G = field(at, 'G');
					const n = G?.shape[0] ?? 0;
					return { i: { v: w.i, fmt: 'int' }, j: { v: w.j, fmt: 'int' }, cap: Number.isFinite(w.j) ? q(capOf(rt, w.j)) : NaN, p: num(w.p, 'pct'), g: G && Number.isFinite(w.i) ? G.data[w.i * n + w.j] : NaN };
				}
			},
			look: { en: 'Find the largest share outside the outlined diagonal.', zh: '找出带框对角线以外最大的份额。' }
		}
	]
});

export default lesson;
