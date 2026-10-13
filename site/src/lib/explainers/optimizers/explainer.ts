// optimizers: the manifest of a CONCEPT package, routed at /train/optimizers/ (AutoConceptPage: goals, hook, the lesson,
// toy, real, compare, pitfalls, checkpoint, links). SGD, momentum, Adam and AdamW race on analytic 2-D loss surfaces
// (the kit's own optimizers), a learning-rate range test and a warm-up demo on a small live network, and gradient
// clipping on an exploding-gradient toy. Sources (process.ts):
//   valley      the race on the long valley from its fixed start, 80 updates            (process 'race', coarse)
//   valleyFine  the same race, one frame per racer per update, 12 updates                (process 'race', fine)
//   race        the race on the learner's landscape and start                            (process 'race', coarse)
//   league      the race on all three landscapes from their own starts                   (process 'league')
//   range       the range test: η from 0.001 to 10 in 100 steps on the 2-8-8-2 network   (process 'range')
//   sched       that network at a peak η of 4 with the learner's schedule, and at a constant 4 (process 'sched')
//   cliff       gradient descent with and without clipping next to a cliff               (process 'cliff')
// The page around the lesson is concept.ts.
import type { ExplainerManifest, I18n } from '../../atlas/types/index.ts';
import { RACE_STEPS, START, setupOf, surfaceOf } from './data.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const L = (en: string, zh: string): I18n => ({ en, zh });
const fixed = () => null;
const valleyStart = () => ({ surface: 'valley', x: surfaceOf('valley').start[0], y: surfaceOf('valley').start[1] });

const manifest: ConceptManifest = {
	slug: 'optimizers',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'setup', start: START },
	// none of these runs has a Python reference: the optimizers are the kit's (checked against torch.optim in the kit's own
	// tests) and the surfaces are formulas; their badge says "Live"
	sources: {
		valley: { kind: 'process', process: 'race', input: valleyStart, opt: { steps: RACE_STEPS }, reference: false },
		valleyFine: { kind: 'process', process: 'race', input: valleyStart, detail: 'fine', opt: { steps: 12 }, reference: false },
		race: { kind: 'process', process: 'race', input: (rt) => { const s = setupOf(rt.input); return { surface: s.surface, x: s.x, y: s.y }; }, opt: { steps: RACE_STEPS }, reference: false },
		league: { kind: 'process', process: 'league', input: fixed, opt: { steps: RACE_STEPS }, reference: false },
		range: { kind: 'process', process: 'range', input: fixed, opt: { steps: 100 }, reference: false },
		sched: { kind: 'process', process: 'sched', input: (rt) => { const s = setupOf(rt.input); return { schedule: s.schedule, warmup: s.warmup }; }, opt: { steps: 200 }, reference: false },
		cliff: { kind: 'process', process: 'cliff', input: (rt) => ({ clip: setupOf(rt.input).clip }), opt: { steps: 30 }, reference: false }
	},
	page: {
		goals: [
			L(
				'Say why plain gradient descent zig-zags in a narrow valley, and work out one update of momentum and one of Adam with the real numbers to see how each one fixes it.',
				'说明普通梯度下降为什么会在狭窄的山谷里来回折返，并用真实的数值算出动量法和 Adam 各自的一次更新，看它们怎样解决这个问题。'
			),
			L('Show with live runs that no optimizer wins on every landscape.', '用实时运行说明：没有哪个优化器在所有地形上都赢。'),
			L(
				'Read a learning-rate range test, and say what warm-up and a decay schedule do to the learning rate over training.',
				'读懂学习率区间测试，并说明预热和衰减调度在训练过程中怎样改变学习率。'
			),
			L('Say what gradient clipping changes in an update, the length and not the direction, and why that saves a run next to a cliff.', '说明梯度裁剪改变了更新中的什么（长度，而不是方向），以及它为什么能救下悬崖边的一次运行。')
		],
		prequestion: {
			id: 'optimizers-pre',
			question: L(
				'A model has two weights. The loss is very steep along one and almost flat along the other. Plain gradient descent moves both with one learning rate. What goes wrong?',
				'一个模型有两个权重。损失沿其中一个非常陡，沿另一个几乎是平的。普通梯度下降用同一个学习率移动这两个权重。会出什么问题？'
			),
			options: [
				L('The steep weight jumps back and forth while the flat one barely moves', '陡的那个权重来回跳，平的那个几乎不动'),
				L('Nothing: both reach the bottom at the same speed', '没有问题：两个权重以同样的速度到达谷底'),
				L('The flat weight overshoots while the steep one barely moves', '平的那个权重冲过头，陡的那个几乎不动')
			],
			answer: 0,
			explain: L(
				'The step is the learning rate times the slope. Along the steep weight the slope is big, so the step overshoots the bottom and lands on the other side; along the flat weight the slope is tiny, so the step is tiny. A learning rate small enough for the steep weight is far too small for the flat one.',
				'步长等于学习率乘以坡度。沿陡的权重坡度大，一步就冲过谷底、落到另一边；沿平的权重坡度很小，一步也很小。一个对陡的权重足够小的学习率，对平的权重就太小了。'
			)
		},
		quiz: [
			{
				q: L('Adam’s very first update, with η = 0.1, sees the gradient (0.2, −40). About how far does each weight move?', 'Adam 的第一次更新，η = 0.1，梯度是 (0.2, −40)。每个权重大约移动多少？'),
				options: [L('0.1 each: x down and y up', '各移动 0.1：x 减小，y 增大'), L('0.02 for x and 4 for y', 'x 移动 0.02，y 移动 4'), L('0.1 for x and 0 for y', 'x 移动 0.1，y 不动')],
				answer: 0,
				explain: L(
					'On the first update the bias-corrected averages are m̂ = g and √v̂ = |g|, so each weight moves by η·g / |g| = η against the sign of its gradient, whatever the gradient’s size.',
					'第一次更新时，修正偏差后的平均值是 m̂ = g、√v̂ = |g|，所以每个权重都逆着自己梯度的符号移动 η·g / |g| = η，与梯度的大小无关。'
				)
			},
			{
				q: L('A gradient (30, 40) is clipped to a length of at most 1. What is the clipped gradient?', '把梯度 (30, 40) 裁剪到长度不超过 1。裁剪后的梯度是什么？'),
				options: [L('(0.6, 0.8)', '(0.6, 0.8)'), L('(1, 1)', '(1, 1)'), L('(0.03, 0.04)', '(0.03, 0.04)')],
				answer: 0,
				explain: L('Its length is 50, so both numbers are divided by 50: the direction stays, the length becomes 1.', '它的长度是 50，所以两个数都除以 50：方向不变，长度变成 1。')
			},
			{
				q: L('A warm-up climbs linearly to a peak η of 4 over 20 steps, as η·(k + 1) / 20. What is η at step 0, the first step?', '预热在 20 步内线性升到峰值 η = 4，按 η·(k + 1) / 20 计算。第 0 步（第一步）的 η 是多少？'),
				options: [L('0.2', '0.2'), L('0', '0'), L('4', '4')],
				answer: 0,
				explain: L('4 × (0 + 1) / 20 = 0.2. It reaches 4 at step 19, the 20th step.', '4 × (0 + 1) / 20 = 0.2。到第 19 步（也就是第 20 步）达到 4。')
			}
		],
		misconception: {
			claim: L('Adam is always the best optimizer.', 'Adam 永远是最好的优化器。'),
			truth: L(
				'On this page’s three landscapes three different optimizers end lowest. Adam’s per-weight scaling wins the long valley; it does not help it over the bumps of the bumpy bowl, where momentum gets further.',
				'在本页的三个地形上，最后损失最低的是三个不同的优化器。Adam 按权重分别缩放，赢下了狭长山谷；但在坑洼的碗里，这帮不了它越过小坑，动量法走得更远。'
			),
			step: 'league'
		},
		lessonNote: L(
			'Eleven steps: four optimizers race down a valley; one update each of SGD, momentum and Adam with the real numbers; your own landscape; then the learning rate itself: a range test, warm-up and a schedule on a small network, and clipping next to a cliff.',
			'十一步：四个优化器沿山谷赛跑；用真实的数值看 SGD、动量法和 Adam 各一次更新；换成你自己的地形；然后看学习率本身：在一个小网络上做区间测试、预热和调度，最后在悬崖边看梯度裁剪。'
		)
	},
	glossary: [
		{ id: 'momentum', term: L('momentum', '动量法'), def: L('An optimizer that keeps a running sum of past gradients, the velocity v ← μ·v + g, and moves the weights by −η·v. Steps that agree add up; steps that flip sign partly cancel.', '一种优化器：它保留过去梯度的滑动累加，即速度 v ← μ·v + g，并让权重移动 −η·v。方向一致的步子会叠加，来回变号的步子会部分抵消。'), related: ['optimizer', 'gradient-descent'] },
		{ id: 'adam', term: L('Adam', 'Adam'), def: L('An optimizer that keeps two running averages per weight, of the gradient (m) and of its square (v), and moves each weight by η·m̂ / (√v̂ + ε): about η per step, whatever the gradient’s size.', '一种优化器：它为每个权重保留两个滑动平均，梯度的（m）和梯度平方的（v），并让每个权重移动 η·m̂ / (√v̂ + ε)：每步大约移动 η，与梯度的大小无关。'), related: ['optimizer', 'momentum'] },
		{ id: 'adamw', term: L('AdamW', 'AdamW'), def: L('Adam with decoupled weight decay: before each Adam step every weight shrinks by the factor 1 − η·λ, separately from the gradient.', '带解耦权重衰减的 Adam：每次 Adam 更新之前，每个权重先乘以 1 − η·λ 缩小一点，这与梯度无关。'), related: ['adam', 'regularization'] },
		{ id: 'lr-schedule', term: L('learning-rate schedule', '学习率调度'), def: L('A rule that sets the learning rate for every training step, such as constant, step decay or cosine decay.', '为每个训练步设定学习率的规则，例如恒定、阶梯衰减或余弦衰减。'), related: ['learning-rate'] },
		{ id: 'warm-up', term: L('warm-up', '预热'), def: L('Starting training with a small learning rate that climbs to its peak over the first steps.', '训练开始时先用较小的学习率，在最初若干步内逐渐升到峰值。'), related: ['lr-schedule', 'learning-rate'] },
		{ id: 'lr-range-test', term: L('learning-rate range test', '学习率区间测试'), def: L('A short run in which the learning rate grows a little every step; the loss curve shows which rates learn and where training blows up.', '一次短的运行，学习率每一步都调大一点；损失曲线显示哪些学习率能学习、从哪里开始训练崩溃。'), related: ['learning-rate'] },
		{ id: 'gradient-clipping', term: L('gradient clipping', '梯度裁剪'), def: L('Shrinking a gradient whose length is over a limit c down to length c before the update. The direction stays the same.', '在更新之前，把长度超过上限 c 的梯度缩短到长度 c。方向保持不变。'), related: ['gradient'] },
		{ id: 'saddle-point', term: L('saddle point', '鞍点'), def: L('A flat point that is a bottom in some directions and a top in others, like the middle of a horse saddle. The gradient there is 0.', '在某些方向上是谷底、在另一些方向上是山顶的平坦点，像马鞍的中间。那里的梯度为 0。'), related: ['gradient'] }
	],
	catalog: {
		interaction: L(
			'SGD, momentum, Adam and AdamW race on a long valley, a saddle and a bumpy bowl, with the optimizer state drawn as a momentum arrow and squared-gradient squares, and one update each of SGD, momentum and Adam worked out with the real numbers. Run a learning-rate range test on a small live network, pick a schedule with or without warm-up and watch it train, and toggle gradient clipping next to a cliff.',
			'SGD、动量法、Adam 和 AdamW 在狭长山谷、鞍点和坑洼的碗上赛跑，优化器状态画成动量箭头和梯度平方方块，并用真实的数值算出 SGD、动量法和 Adam 各一次更新。在一个实时训练的小网络上做学习率区间测试，选择带或不带预热的调度并观察训练，在悬崖边切换梯度裁剪。'
		),
		engine: L('Kit optimizers on analytic surfaces + live 2-8-8-2 MLP', '解析曲面上的内置优化器 + 实时训练的 2-8-8-2 MLP'),
		mode: 'live',
		links: ['mlp-playground', 'char-rnn']
	} as unknown as ExplainerManifest['catalog'],
	// No third-party data, model or code: the surfaces, the toy and the network are written here and the kit runs them.
	// The ideas are credited below (no text, figures or code used).
	credits: [
		{ name: 'Adam: A Method for Stochastic Optimization (Kingma and Ba, 2014)', url: 'https://arxiv.org/abs/1412.6980', license: 'Paper; ideas only (no text, figures or code used)', usedFor: 'The Adam update and its bias correction, as the kit implements it (the PyTorch form)', attribution: 'Diederik P. Kingma and Jimmy Ba' },
		{ name: 'Decoupled Weight Decay Regularization (Loshchilov and Hutter, 2017)', url: 'https://arxiv.org/abs/1711.05101', license: 'Paper; ideas only (no text, figures or code used)', usedFor: 'AdamW: weight decay applied to the weights directly, outside the Adam step', attribution: 'Ilya Loshchilov and Frank Hutter' },
		{ name: 'Cyclical Learning Rates for Training Neural Networks (Smith, 2015)', url: 'https://arxiv.org/abs/1506.01186', license: 'Paper; ideas only (no text, figures or code used)', usedFor: 'The learning-rate range test: raise the rate every step and read the loss curve', attribution: 'Leslie N. Smith' },
		{ name: 'On the difficulty of training recurrent neural networks (Pascanu, Mikolov and Bengio, 2013)', url: 'https://arxiv.org/abs/1211.5063', license: 'Paper; ideas only (no text, figures or code used)', usedFor: 'The exploding-gradient cliff of a one-unit recurrence and gradient clipping by norm', attribution: 'Razvan Pascanu, Tomas Mikolov and Yoshua Bengio' },
		{ name: 'Why Momentum Really Works (Goh, Distill 2017)', url: 'https://distill.pub/2017/momentum/', license: 'CC-BY-4.0; ideas only (no text, figures or code used)', usedFor: 'The idea of showing momentum on a long, narrow valley next to plain gradient descent', attribution: 'Gabriel Goh' }
	]
};

export default manifest;
