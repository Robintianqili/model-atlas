// adversarial-tinycnn: the guided lesson (10 steps). Every number is a {placeholder}: live ones from TinyCNN's trace
// (step 1) and from the attack's frames (process.ts), measured ones from facts.json ({fact:...}, stats.py).
//   1 your drawing and its class → 2 the loss → 3 the gradient for every pixel (predict) → 4 keep the sign →
//   5 add ε·sign (slider) → 6 the flip → 7 PGD: small clipped steps → 8 how big must ε be (task) → 9 a defence →
//   10 limits: a white-box attack on a tiny model. Claims not computed live: claims.test.ts.
import type { At, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { TABLE_EPS, cls, epsText as E, nearSmallest, stateOf, type RT, type St } from './read.ts';

const M = (at: At) => at.frame?.metrics ?? {};
const w = (v: I18n) => v as unknown as TmplVal;
const view = (mode: string) => ({ key: 'adversarial-tinycnn/attack', props: () => ({ mode }) });
/** TinyCNN's answer on the learner's drawing, from the kit trace (step 1) */
function answer(rt: RT) {
	const p = rt.sources.fwd?.trace?.t.softmax?.data;
	if (!p) return { top: NaN, p: NaN, run: NaN, pr: NaN };
	const o = Array.from(p.keys()).sort((a, b) => p[b] - p[a] || a - b);
	return { top: w(cls(rt, o[0])), p: p[o[0]], run: w(cls(rt, o[1])), pr: p[o[1]] };
}
/** the side of TinyCNN's input grid (28), from its spec */
const side = (rt: RT) => Number(rt.sources.fwd?.model?.spec.inputs?.[0]?.shape[2] ?? NaN);
/** the share of empty-paper pixels (0) that one FGSM step turns grey */
function paperGreyed(at: At): number {
	const x = (at.state as St | undefined)?.x?.data, a = stateOf(at.tape, 'fgsm')?.adv?.data;
	if (!x || !a) return NaN;
	let paper = 0, grey = 0;
	for (let i = 0; i < x.length; i++) if (x[i] === 0) (paper++, a[i] > 0 && grey++);
	return paper ? grey / paper : 0;
}
/** class names of a frame's y, top answer and runner-up */
const names = (rt: RT, at: At) => ({ y: w(cls(rt, M(at).y)), top: w(cls(rt, M(at).top ?? M(at).y)) });
/** up to this ε the flip step calls the change faint (pixel values run from 0 to 1; beyond it the paper turns plainly grey) */
const FAINT = 0.1;

const lesson: LessonDef = {
	title: { en: 'Adversarial examples: change a drawing a little, flip the answer', zh: '对抗样本：把画改动一点点，让答案翻转' },
	summary: {
		en: 'Measure the loss of TinyCNN’s answer on your drawing, compute its gradient for every pixel, step along the sign (FGSM), repeat small clipped steps (PGD), find how big the change must be, and see the limits of a white-box attack.',
		zh: '在你的画上算出 TinyCNN 答案的损失，求出它对每个像素的梯度，沿着符号走一步（FGSM），再重复裁剪过的小步（PGD），找出改动要多大，并看清白盒攻击的局限。'
	},
	steps: [
		{
			id: 'drawing',
			source: 'fwd',
			at: { layer: 'softmax' },
			view: 'kit.probbars',
			provenance: 'live',
			title: { en: 'Your drawing and TinyCNN’s answer', zh: '你的画和 TinyCNN 的答案' },
			narration: {
				en: 'TinyCNN, the small convolutional network from its own page, reads your drawing as {side}×{side} numbers and answers {top} with {p|pct}; next comes {run} with {pr|pct}. This page tries to change that answer while changing the drawing as little as possible. A drawing changed like that is an adversarial example.',
				zh: 'TinyCNN 是在它自己那一页讲过的小卷积网络。它把你的画读成 {side}×{side} 个数，回答“{top}”，份额 {p|pct}；第二名是“{run}”，{pr|pct}。这一页要做的是：尽量少地改动这幅画，同时改掉这个答案。这样改过的画叫对抗样本。',
				vars: (rt) => ({ ...answer(rt), side: side(rt) })
			},
			look: { en: 'The ten class shares; the longest bar is the answer the attack will try to change.', zh: '十个类别的份额；最长的那条就是攻击要改掉的答案。' },
			terms: ['classifier', 'softmax', 'adversarial-example']
		},
		{
			id: 'loss',
			source: 'attack',
			at: { phase: 'clean' },
			view: 'zoom.formula',
			provenance: 'live',
			title: { en: 'How right is it? The loss', zh: '它有多对？损失' },
			narration: {
				en: 'Call TinyCNN’s answer y = {y}. Its loss is L = −log p(y) = −log {py|n4} = {loss} (log is the natural logarithm): near 0 when the model is sure, larger as p(y) falls. The attack wants this loss up. Once p({y}) drops below another class’s share, the answer flips.',
				zh: '把 TinyCNN 的答案记作 y =“{y}”。它的损失是 L = −log p(y) = −log {py|n4} = {loss}（log 是自然对数）：模型越有把握越接近 0，p(y) 越小损失越大。攻击就是要让这个损失变大。一旦 p(“{y}”) 掉到别的类别的份额以下，答案就翻转了。',
				vars: (rt, at) => ({ ...names(rt, at), py: M(at).py, loss: M(at).loss })
			},
			look: { en: 'The formula with your drawing’s numbers: p(y) and the loss it gives.', zh: '代入你这幅画的数的公式：p(y) 和它给出的损失。' },
			terms: ['loss', 'cross-entropy']
		},
		{
			id: 'grad',
			source: 'attack',
			at: { phase: 'grad' },
			view: view('grad'),
			provenance: 'live',
			title: { en: 'Which pixels would raise the loss? The gradient', zh: '哪些像素能让损失变大？梯度' },
			predict: {
				id: 'adversarial-tinycnn-grad',
				question: { en: 'Can a pixel of empty paper (value 0) have a gradient that is not 0?', zh: '一个空白纸上的像素（值为 0），梯度可能不是 0 吗？' },
				options: [
					{ en: 'No: it has no ink, so it cannot matter', zh: '不可能：它没有墨，所以无关紧要' },
					{ en: 'Yes: giving it a little ink can change the loss too', zh: '可能：给它加一点墨，同样会改变损失' },
					{ en: 'Only right next to a stroke', zh: '只在紧挨笔画的地方才可能' }
				],
				answer: 1,
				explain: {
					en: 'Every pixel feeds the first convolution, so the gradient reaches empty paper wherever the network’s units respond, near the strokes or not. That is why an attack changes the paper as well as the ink.',
					zh: '每个像素都进入第一层卷积，所以只要网络的单元对那里有反应，梯度就会到达空白的纸，不管离笔画近不近。这就是攻击既改墨也改纸的原因。'
				}
			},
			narration: {
				en: 'Backpropagation, the rule that computes the gradients used in training, does not have to stop at the weights: it can run all the way back to the input. The result g = ∂L/∂x has one number per pixel: how fast the loss grows when that pixel gets a little more ink (a negative number means the loss grows with a little less ink). The biggest in size is {gAt}, at row {row}, column {col} (counting from 0): {dir}.',
				zh: '反向传播（训练时用来算梯度的方法）不必停在权重上：它可以一路传回输入。结果 g = ∂L/∂x 每个像素一个数，表示这个像素多一点墨时损失增长得有多快（负数表示少一点墨时损失才增长）。绝对值最大的是 {gAt}，在第 {row} 行、第 {col} 列（从 0 数起）：{dir}。',
				vars: (_rt, at) => {
					const m = M(at);
					const dir = pick(m.gAt > 0, { en: 'a little more ink there raises the loss faster than anywhere else', zh: '在那里加一点墨，损失涨得比哪里都快' }, { en: 'taking a little ink away there raises the loss faster than anywhere else', zh: '在那里减一点墨，损失涨得比哪里都快' });
					return { gAt: m.gAt, row: m.row, col: m.col, dir: w(dir) };
				}
			},
			look: { en: 'Orange: more ink here raises the loss. Blue: less ink does. Hover a pixel for its number.', zh: '橙色：这里多一点墨损失就变大。蓝色：少一点墨损失变大。把指针移到像素上看它的数。' },
			terms: ['gradient', 'backpropagation']
		},
		{
			id: 'sign',
			source: 'attack',
			at: { phase: 'sign' },
			view: view('sign'),
			provenance: 'live',
			title: { en: 'Keep only the sign', zh: '只留下符号' },
			narration: {
				en: 'The fast gradient sign method (FGSM) throws the sizes away: sign(g) is +1 where more ink raises the loss ({nPos} pixels), −1 where less ink does ({nNeg}) and 0 where g is exactly 0 ({nZero}). Why? Each pixel may move by at most ε, and to a first approximation (treating the loss near your drawing as a straight slope) the most loss for that budget comes from moving every pixel the full ε its sign points to.',
				zh: '快速梯度符号法（FGSM）把大小扔掉：多一点墨能让损失变大的地方 sign(g) = +1（{nPos} 个像素），少一点墨能让损失变大的地方是 −1（{nNeg} 个），g 恰好为 0 的地方是 0（{nZero} 个）。为什么？每个像素最多只能动 ε；在一阶近似下（把你的画附近的损失看成一个笔直的斜坡），这个预算换来最多损失的走法，就是让每个像素都朝它的符号方向走满 ε。',
				vars: (_rt, at) => ({ nPos: M(at).nPos, nNeg: M(at).nNeg, nZero: M(at).nZero })
			},
			look: { en: 'Every pixel is now +1 (orange), −1 (blue) or 0: the size of the gradient is gone.', zh: '现在每个像素都是 +1（橙）、−1（蓝）或 0：梯度的大小没有了。' },
			terms: ['fgsm']
		},
		{
			id: 'step',
			source: 'attack',
			at: { phase: 'fgsm' },
			view: view('step'),
			provenance: 'live',
			title: { en: 'Add ε times the sign', zh: '加上 ε 乘以符号' },
			narration: {
				en: 'The step is x′ = clip(x + ε·sign(g), 0, 1) with ε = {eps}: every pixel moves by ε the way its sign points, and clip puts any value that would leave the range 0 to 1 back at the edge, so paper (0) can only gain ink and full ink (1) can only lose it. Here {clipped} pixels were clipped, so on average a pixel moves by {meanMove}. Change ε with the slider.',
				zh: '这一步是 x′ = clip(x + ε·sign(g), 0, 1)，ε = {eps}：每个像素都朝它的符号所指的方向移动 ε，clip 再把跑出 0 到 1 范围的值放回边上，所以纸（0）只能加墨，满墨（1）只能减墨。这里有 {clipped} 个像素被裁剪，所以平均每个像素移动 {meanMove}。用滑块改变 ε。',
				vars: (_rt, at) => ({ eps: E(M(at).eps), clipped: M(at).clipped, meanMove: M(at).meanMove })
			},
			look: { en: 'Your drawing, plus ε times the sign, gives the changed drawing x′.', zh: '你的画，加上 ε 乘以符号，得到改过的画 x′。' },
			terms: ['perturbation']
		},
		{
			id: 'flip',
			source: 'attack',
			at: { phase: 'fgsm' },
			view: view('flip'),
			provenance: 'live',
			title: { en: 'The drawing barely changes. Does the answer?', zh: '画几乎没变，答案呢？' },
			narration: {
				en: '{verdict} p({y}) went from {py0|pct} to {py|pct} and the loss from {loss0} to {loss}. {seen}',
				zh: '{verdict}p(“{y}”) 从 {py0|pct} 变成了 {py|pct}，损失从 {loss0} 变成了 {loss}。{seen}',
				vars: (rt, at) => {
					const m = M(at), n = names(rt, at), top = n.top as unknown as I18n, e = E(m.eps);
					const py0 = (at.state as { p0?: { data: ArrayLike<number> } } | undefined)?.p0?.data[m.y] ?? NaN;
					const verdict = pick(m.flipped === 1, { en: `The answer flipped: TinyCNN now says ${top.en}.`, zh: `答案翻转了：TinyCNN 现在说是“${top.zh}”。` }, { en: 'Not flipped yet: one step of this size is not enough for this drawing, and the next step tries harder.', zh: '还没有翻转：对这幅画来说，这么大的一步还不够，下一步会更用力。' });
					// what the learner sees depends on ε: nothing at 0, a faint pattern up to FAINT, plain grey paper beyond
					const seen: I18n = !(m.eps > 0)
						? { en: 'With ε = 0 no pixel moved at all, so nothing could change: go back one step and raise ε with the slider.', zh: '在 ε = 0 时没有一个像素动过，所以什么都不会变：回到上一步，用滑块把 ε 调大。' }
						: m.eps <= FAINT
							? { en: `Yet no pixel moved by more than ${e}, where full ink is 1: you would still read it as the same drawing. The change is a faint pattern spread over the whole grid, not a new stroke; the amplifier shows it.`, zh: `可没有一个像素的改动超过 ${e}（满墨是 1）：在你看来它还是那幅画。改动是铺满整个网格的一层淡淡的花纹，而不是新的一笔；放大器能把它显出来。` }
							: { en: `Here pixels moved by up to ${e}, where full ink is 1: that is no longer faint, and the empty paper turns plainly grey. Still, the change is a pattern spread over the whole grid, not a new stroke; the amplifier shows it.`, zh: `这里像素最多改动了 ${e}（满墨是 1）：这已经不算淡了，空白的纸明显变灰。不过改动仍是铺满整个网格的花纹，而不是新的一笔；放大器能把它显出来。` };
					return { ...n, verdict, seen: w(seen), py0, py: m.py, loss0: at.tape?.frame(0).metrics?.loss ?? NaN, loss: m.loss };
				}
			},
			look: { en: 'Your drawing and x′ side by side, the change between them amplified, and the class shares on each.', zh: '你的画和 x′ 并排，二者之间的改动被放大显示，还有各自的类别份额。' },
			terms: ['adversarial-example', 'perturbation']
		},
		{
			id: 'pgd',
			source: 'attack',
			at: { phase: 'pgd', last: true },
			view: view('pgd'),
			play: { from: { phase: 'pgd', step: 1 }, to: { phase: 'pgd', last: true }, interval: 700, unit: { en: 'step', zh: '步' } },
			provenance: 'live',
			title: { en: 'PGD: many small steps, clipped', zh: 'PGD：很多小步，再裁剪' },
			narration: {
				en: 'Projected gradient descent (PGD) takes {steps} steps of α = ε/4 = {alpha} instead of one. After each it recomputes the gradient at the new drawing and clips every pixel back into the allowed box: within {eps} of your drawing, and between 0 and 1. {verdict} On the test drawings at ε = {tableEps}, one FGSM step flips {fact:fgsm10|pct}, PGD {fact:pgd10|pct}.',
				zh: '投影梯度下降（PGD）不只走一步，而是走 {steps} 步，每步 α = ε/4 = {alpha}。每走一步，它就在新的画上重新算梯度，再把每个像素裁剪回允许的范围：离你的画不超过 {eps}，并且在 0 到 1 之间。{verdict}在测试画上，ε = {tableEps} 时，一步 FGSM 能翻转 {fact:fgsm10|pct}，PGD 能翻转 {fact:pgd10|pct}。',
				vars: (rt, at) => {
					const m = M(at), top = cls(rt, m.top);
					const verdict = pick(m.firstFlip > 0, { en: `It flipped at step ${m.firstFlip} and ends as ${top.en}.`, zh: `它在第 ${m.firstFlip} 步翻转，最后成了“${top.zh}”。` }, { en: 'Even PGD did not flip it at this ε; try a larger ε in the next step.', zh: '在这个 ε 下，连 PGD 也没能翻转它；下一步试试更大的 ε。' });
					return { steps: m.steps, alpha: E(m.alpha), eps: E(m.eps), verdict, tableEps: E(TABLE_EPS) };
				}
			},
			look: { en: 'Press Play: the drawing after each step, and p(y) against the best other class.', zh: '按“播放”：每一步之后的画，以及 p(y) 和最强的其他类别的对比。' },
			terms: ['pgd']
		},
		{
			id: 'budget',
			source: 'attack',
			at: { phase: 'scan' },
			view: view('budget'),
			provenance: 'live',
			title: { en: 'How big must ε be?', zh: 'ε 要多大？' },
			narration: {
				en: 'How little change is enough? With one FGSM step, find the smallest ε that still flips TinyCNN’s answer: now ε = {eps}, and the answer {now}. Over the test drawings, half flip with one FGSM step by ε = {minF}, and with PGD by ε = {minP}.',
				zh: '改多少就够了？只走一步 FGSM，找出仍能让 TinyCNN 答案翻转的最小 ε：现在 ε = {eps}，答案{now}。在测试画中，一半的画在 ε = {minF} 以内被一步 FGSM 翻转，用 PGD 则只要 ε = {minP}。',
				vars: (rt, at) => {
					const m = M(at);
					return { eps: E(m.eps), minF: E(rt.num('fact:minFgsmMedian')), minP: E(rt.num('fact:minPgdMedian')), now: pick(m.flipped === 1, { en: 'flips', zh: '翻转了' }, { en: 'stays', zh: '没变' }) };
				}
			},
			look: { en: 'Move the slider; once you are close, the curve shows p(y) for every ε on the grid.', zh: '拖动滑块；接近答案后，曲线会显示网格上每个 ε 的 p(y)。' },
			task: {
				prompt: { en: 'Set ε to the smallest value at which one FGSM step flips the answer (within 0.01 above it).', zh: '把 ε 调到一步 FGSM 就能让答案翻转的最小值（最多比它大 0.01）。' },
				done: (_rt, at) => nearSmallest(at.frame?.metrics),
				waiting: { en: 'Not yet: either the answer does not flip at this ε, or a smaller ε would flip it too.', zh: '还没有：要么这个 ε 下答案没翻转，要么更小的 ε 也能翻转它。' },
				hint: { en: 'Halve the range: if it flips, go lower; if not, go higher. A drawing that never flips by the end of the slider needs PGD or another drawing.', zh: '用二分法：翻转了就往小调，没翻转就往大调。如果滑到头也翻不了，就得用 PGD，或者换一幅画。' },
				doneText: { en: 'Found it: one step of this size is the least FGSM needs here.', zh: '找到了：这么大的一步，就是 FGSM 在这里最少需要的。' }
			},
			terms: ['fgsm', 'pgd']
		},
		{
			id: 'defend',
			source: 'attack',
			at: { phase: 'pgd', last: true },
			view: view('defend'),
			provenance: 'live',
			title: { en: 'A defence, in one line', zh: '一种防御，一句话说完' },
			narration: {
				en: 'The best-known defence is adversarial training: during training, attack each batch with PGD and train on the changed drawings with their right labels, so the model learns to answer {y} for x′ too. TinyCNN was trained without it. The planned concept page on adversarial robustness will show it on a small 2-D network.',
				zh: '最有名的防御是对抗训练：训练时用 PGD 攻击每一批数据，再用带正确标签的改过的画来训练，让模型对 x′ 也回答“{y}”。TinyCNN 训练时没有用它。计划中的“对抗鲁棒性”概念页会在一个小的二维网络上演示它。',
				vars: (rt, at) => ({ y: w(cls(rt, M(at).y)) })
			},
			look: { en: 'One training pair adversarial training would add: the changed drawing, with your drawing’s label.', zh: '对抗训练会加进来的一对训练数据：改过的画，配上你的画的标签。' },
			terms: ['adversarial-training', 'batch']
		},
		{
			id: 'limits',
			source: 'attack',
			at: { phase: 'scan' },
			view: view('limits'),
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'Limits: a white-box attack on a tiny model', zh: '局限：对一个小模型的白盒攻击' },
			narration: {
				en: 'This was a white-box attack: it used TinyCNN’s exact gradient. Of the PGD drawings made on TinyCNN at ε = {tableEps} that fool it, only {fact:transferVit|pct} also fool TinyViT, another model trained on the same drawings; attacking TinyViT with its own gradient flips {fact:vitPgd10|pct}. And on a {side}×{side} drawing the change has no texture to hide in: one FGSM step at ε = {eps} lays grey on {paper|pct} of the empty paper.',
				zh: '这是一次白盒攻击：它用的是 TinyCNN 精确的梯度。在 ε = {tableEps} 下对 TinyCNN 做出、并且骗过了它的 PGD 画中，只有 {fact:transferVit|pct} 也能骗过 TinyViT（用同样的画训练的另一个模型）；用 TinyViT 自己的梯度去攻击它，则能翻转 {fact:vitPgd10|pct}。另外，在 {side}×{side} 的小画上，改动没有纹理可以藏身：ε = {eps} 的一步 FGSM 会让 {paper|pct} 的空白纸变灰。',
				vars: (rt, at) => ({ tableEps: E(TABLE_EPS), eps: E(M(at).eps), paper: paperGreyed(at), side: side(rt) })
			},
			look: { en: 'The measured table: how often each attack flips TinyCNN’s test drawings, and what carries over to TinyViT.', zh: '实测表：每种攻击翻转 TinyCNN 测试画的比例，以及有多少能转移到 TinyViT。' },
			terms: ['white-box-attack']
		}
	],
	fixtures: [
		{ id: 'small-eps', label: { en: 'a square with ε = 0.01', zh: '一个方框，ε = 0.01' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 }, { x: 0.2, y: 0.2 }] }], eps: 0.01 } },
		{ id: 'big-eps', label: { en: 'a square with ε = 0.3', zh: '一个方框，ε = 0.3' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 }, { x: 0.2, y: 0.2 }] }], eps: 0.3 } },
		{ id: 'zero-eps', label: { en: 'a square with ε = 0', zh: '一个方框，ε = 0' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 }, { x: 0.2, y: 0.2 }] }], eps: 0 } }
	]
};

export default lesson;
