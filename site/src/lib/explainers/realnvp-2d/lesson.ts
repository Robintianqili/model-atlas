// realnvp-2d: the guided lesson, one idea per step. Every number is a {placeholder} computed from the frames that ran (or
// the learner's points); every sentence that is not live is tested in claims.test.ts.
import type { At, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { fmt3, fmtSetting, paren } from '../../atlas/core/format.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { bandShare, dataOf, LAYERS, logp, type Params } from './nvp.ts';
import { RINGS } from './explainer.ts';
import { GAP } from './process.ts';
import { SIDE } from './views/plane/meta.ts';

const L = (en: string, zh: string) => ({ en, zh });
type St = { params?: Params; grads?: Params; batch?: Record<string, Tensor>; step?: number };
const st = (at: At) => (at.state ?? {}) as St;
const metric = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const xy = (a: number, b: number) => `(${fmt3(a)}, ${fmt3(b)})`;
/** a setting as set (0.003, not 0.00300); a missing one stays NaN, so lint reports it */
const short = (v: number) => (Number.isFinite(v) ? fmtSetting(v) : NaN);
const plane = (mode: string, extra: Record<string, unknown> = {}) => ({ key: 'realnvp-2d/plane', props: () => ({ mode, ...extra }) });
const nParams = (p: Params | undefined) => Object.values(p ?? {}).reduce((s, t) => s + t.data.length, 0);
/** the metrics of the first frame of a phase in the step's tape */
const first = (at: At, phase: string) => (at.tape && at.tape.find({ phase }) >= 0 ? (at.tape.frame(at.tape.find({ phase })).metrics ?? {}) : {});
/** the weight with the largest gradient: key, [out, in] */
function largestGrad(at: At): { key: string; index: [number, number] } {
	let best = { key: 'c0.4.weight', index: [0, 0] as [number, number], g: -1 };
	for (const [key, t] of Object.entries(st(at).grads ?? {})) if (key.endsWith('.weight')) t.data.forEach((g, k) => Math.abs(g) > best.g && (best = { key, index: [Math.floor(k / t.shape[1]), k % t.shape[1]], g: Math.abs(g) }));
	return best;
}
/** the mean −log p of the frame's batch under the frame's weights (the loss after an update, on the same points) */
function batchLoss(at: At): number {
	const b = st(at).batch?.x;
	const p = st(at).params;
	if (!b || !p || !b.shape[0]) return NaN;
	return -logp(p, b.data, b.shape[0]).reduce((a, v) => a + v, 0) / b.shape[0];
}

const lesson: LessonDef = {
	title: L('RealNVP: invertible layers that warp the plane', 'RealNVP：一层层扭曲平面的可逆层'),
	summary: L(
		'Your points and a cloud of plain noise, one affine coupling layer worked out and undone, eight layers that bend your shape into the noise, the log-determinant and the exact log-likelihood, maximum-likelihood training, sampling by running the layers backwards, and why two separate rings are hard.',
		'你的点和一团普通的噪声，算出并撤销一个仿射耦合层，八层把你的形状弯成噪声，对数行列式和精确的对数似然，最大似然训练，把各层倒着跑来采样，以及为什么两个分开的圆环很难学。'
	),
	steps: [
		{
			id: 'data',
			source: 'train',
			at: 'first',
			view: plane('data'),
			provenance: 'live',
			title: L('Your points and plain noise', '你的点和普通的噪声'),
			narration: {
				en: 'Your shape: {pts}. A normalizing flow learns an invertible map between your points and plain noise, a round cloud of normal draws N(0, I): x and y independent, each with spread (standard deviation) 1. Run forward, the map says how likely any place is; run backwards, it turns fresh noise into new points like yours.',
				zh: '你的形状：{pts}。归一化流学的是一个可逆映射，一头连着你的点，另一头连着普通的噪声：一团正态分布 N(0, I) 的随机点，x 和 y 相互独立，标准差都是 1。正着跑，这个映射告诉你任何位置有多大可能；倒着跑，它把新的噪声变成像你的点一样的新点。',
				vars: (rt) => ({ pts: plural(dataOf(rt.input).n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' }) })
			},
			look: L('The green points; point 0 (counting from 0) is ringed: the next steps follow it. Pick another shape or paint your own in the input box.', '绿色的点；0 号点（从 0 数起）被圈出，后面几步都跟着它。可以在输入框里换一个形状，或者自己画。'),
			terms: ['generative-model', 'normalizing-flow', 'gaussian-noise']
		},
		{
			id: 'couple',
			source: 'flow',
			at: { phase: 'couple', step: 0 },
			view: plane('couple'),
			provenance: 'live',
			title: L('One coupling layer', '一个耦合层'),
			narration: {
				en: 'The flow, already trained on your points, is a stack of {K} coupling layers. Layer 0 keeps x and moves only y: a small network reads x and gives a log-stretch s and a shift t, then y′ = y·e^s + t. For point 0, x = {u} gives s = {s} and t = {t}, so y = {v} becomes {vp} × {es} + {tp} = {v2}.',
				zh: '这个流模型已经在你的点上训练好了，它由 {K} 个耦合层叠成。第 0 层保持 x 不变，只移动 y：一个小网络读入 x，给出对数伸缩量 s 和平移量 t，然后 y′ = y·e^s + t。对 0 号点，x = {u} 给出 s = {s}、t = {t}，于是 y = {v} 变成 {vp} × {es} + {tp} = {v2}。',
				vars: (_rt, at): Record<string, TmplVal> => ({ K: LAYERS, u: metric(at, 'u'), s: metric(at, 's'), t: metric(at, 't'), tp: paren(metric(at, 't')), v: metric(at, 'v'), vp: paren(metric(at, 'v')), es: metric(at, 'es'), v2: metric(at, 'v2') })
			},
			look: L('The grid: every vertical line slides and stretches as one piece, because s and t depend on x alone.', '看网格：每条竖线都整条地滑动和伸缩，因为 s 和 t 只取决于 x。'),
			terms: ['coupling-layer']
		},
		{
			id: 'invert',
			source: 'flow',
			at: { phase: 'couple', step: 0 },
			view: plane('couple', { back: true }),
			provenance: 'live',
			title: L('Undo the layer', '撤销这一层'),
			predict: {
				id: 'realnvp-2d-undo',
				question: L('How do you get y back from y′?', '怎样从 y′ 得回 y？'),
				options: [
					L('Run the small network backwards', '把小网络倒着跑'),
					L('Read s and t again from x, which did not change, then subtract t and divide by e^s', '用没变的 x 重新算出 s 和 t，再减去 t、除以 e^s'),
					L('You cannot: the layer loses information', '做不到：这一层丢掉了信息')
				],
				answer: 1,
				explain: L(
					'x passes through unchanged, so the same network gives the same s and t, and undoing the two moves in reverse order is plain arithmetic: y = (y′ − t)·e^−s. The small network is never inverted; that is why each layer keeps one coordinate as it is.',
					'x 原样穿过这一层，所以同一个网络给出同样的 s 和 t，再按相反的顺序撤销这两步，只是普通的算术：y = (y′ − t)·e^−s。小网络从来不用求逆；这正是每一层都保留一个坐标不动的原因。'
				)
			},
			narration: {
				en: 'Layer 0 moved point 0 from y = {v} to y′ = {v2}. To make new points, the flow must run the other way, from y′ back to y. The small network inside the layer has no inverse: it squeezes x through tanh and mixes many numbers.',
				zh: '第 0 层把 0 号点从 y = {v} 移到了 y′ = {v2}。要造出新的点，流模型得反过来跑：从 y′ 回到 y。可层里的小网络没有逆：它把 x 经过 tanh 压缩，又把许多数混在一起。',
				vars: (_rt, at): Record<string, TmplVal> => ({ v: metric(at, 'v'), v2: metric(at, 'v2') })
			},
			look: L('Before the stage shows the way back, decide how y′ becomes y again.', '在图中显示回去的路之前，先想好 y′ 怎样变回 y。'),
			terms: ['coupling-layer']
		},
		{
			id: 'stack',
			source: 'flow',
			at: { phase: 'couple', last: true },
			view: plane('warp'),
			play: { from: { phase: 'start' }, to: { phase: 'couple', last: true }, interval: 700, unit: L('layer', '层') },
			provenance: 'live',
			title: L('Layers take turns', '各层轮流来'),
			narration: {
				en: 'One layer can only slide points along one axis, so the layers take turns: even layers move y, odd layers move x, {K} layers in all. Layer by layer the flow pushes your points toward the noise: their spread goes from {sx0} to {sx1} in x and from {sy0} to {sy1} in y, {verdict}. The grid shows how each part of the plane was bent on the way.',
				zh: '一层只能沿一个方向滑动点，所以各层轮流来：偶数层移动 y，奇数层移动 x，一共 {K} 层。一层又一层，流模型把你的点推向噪声：x 方向的标准差从 {sx0} 变成 {sx1}，y 方向从 {sy0} 变成 {sy1}，{verdict}。网格显示平面的每一块在路上是怎样被弯过去的。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const s0 = first(at, 'start');
					const near = [metric(at, 'sx'), metric(at, 'sy')].every((v) => Math.abs(v - 1) < 0.25);
					return { K: LAYERS, sx0: s0.sx ?? NaN, sx1: metric(at, 'sx'), sy0: s0.sy ?? NaN, sy1: metric(at, 'sy'), verdict: pick(near, L('close to the noise’s 1: a round cloud', '接近噪声的 1：成了一团圆圆的云'), L('still away from the noise’s 1', '离噪声的 1 还远')) };
				}
			},
			look: L('Press Play: the grid over your points bends layer by layer; the dashed circles are radius 1 and 2 of the noise.', '按播放：覆盖你的点的网格一层层弯曲；虚线圆是噪声的半径 1 和 2。')
		},
		{
			id: 'logdet',
			source: 'flow',
			at: { phase: 'couple', last: true },
			view: plane('warp', { cell: true }),
			provenance: 'live',
			title: L('Stretching thins the density', '拉伸会稀释密度'),
			narration: {
				en: 'Each layer stretches its moving coordinate by e^s, so a small area around a point grows e^s times, and the density there must shrink by the same factor: the total probability stays 1. For point 0 the {K} values of s add up to Σs = {ld}, the log-determinant: a tiny area there grows e^Σs = {eld} times (the inset in the stage: a square of side {side} around it, before and after). One sum per point: that is why coupling layers are cheap.',
				zh: '每一层把它移动的坐标拉伸 e^s 倍，所以一个点周围的小面积变大 e^s 倍，那里的密度就必须缩小同样的倍数：总概率始终是 1。对 0 号点，{K} 个 s 加起来是 Σs = {ld}，这就是对数行列式：那里极小的一块面积会变大 e^Σs = {eld} 倍（图中的小插图画出了它周围边长 {side} 的小方块变换前后的样子）。每个点只要一次求和：耦合层便宜就便宜在这里。',
				vars: (_rt, at): Record<string, TmplVal> => ({ K: LAYERS, ld: metric(at, 'ld'), eld: Math.exp(metric(at, 'ld')), side: short(SIDE) })
			},
			look: L('The inset: the dashed square is the small square around point 0 before the layers, the filled shape the same square after them; the caption gives its measured area ratio.', '看小插图：虚线方块是 0 号点周围的小方块穿过各层之前的样子，填色的形状是它穿过之后的样子；图注给出量得的面积倍数。'),
			terms: ['log-determinant', 'density']
		},
		{
			id: 'likelihood',
			source: 'flow',
			at: { phase: 'density' },
			view: plane('density'),
			provenance: 'live',
			title: L('The exact log-likelihood', '精确的对数似然'),
			narration: {
				en: 'Now the likelihood of any point, with nothing approximated: push it through the layers to z, score z under the noise, log N(z) = −½|z|² − log 2π, and add the stretches: log p(x) = log N(z) + Σs. Point 0 lands at z = {z}: log N(z) = {logN}, plus Σs = {ld}, gives log p = {logp}. Averaged over all your points, log p is {mean}.',
				zh: '现在可以算任何一个点的似然，而且不用任何近似：把它推过各层得到 z，用噪声的分布给 z 打分，log N(z) = −½|z|² − log 2π，再加上各层的拉伸：log p(x) = log N(z) + Σs。0 号点落在 z = {z}：log N(z) = {logN}，加上 Σs = {ld}，得到 log p = {logp}。在你所有的点上取平均，log p 是 {mean}。',
				vars: (_rt, at): Record<string, TmplVal> => ({ z: xy(metric(at, 'zx'), metric(at, 'zy')), logN: metric(at, 'logN'), ld: metric(at, 'ld'), logp: metric(at, 'logp'), mean: -metric(at, 'nll') })
			},
			look: L('Stronger colour: higher density p(x) = e^(log p). It sits on your shape.', '颜色越浓，密度 p(x) = e^(log p) 越高。它落在你的形状上。'),
			terms: ['log-likelihood']
		},
		{
			id: 'step',
			source: 'sgd1',
			at: { phase: 'update', step: 0 },
			view: 'train-step',
			focus: (_rt, at) => [{ field: 'params', key: largestGrad(at).key, index: largestGrad(at).index, role: 'param' }],
			provenance: 'live',
			title: L('Training: make your points likely', '训练：让你的点更可能出现'),
			narration: {
				en: 'The loss is the mean of −log p over a batch of {b} of your points. Untrained, the flow does nothing (the last layer of every small network starts at 0, so z = x and Σs = 0): the loss is {loss|n4}. One gradient-descent step changes only {moved} of the {count} weights and biases, the last layers’: the zeros in them block the gradient to the earlier layers. The weight with the largest gradient goes {old} − {lr} × {g} = {new}; the batch’s loss falls by {drop}.',
				zh: '损失是一批 {b} 个点上 −log p 的平均。未训练的流模型什么也不做（每个小网络的最后一层一开始都是 0，所以 z = x、Σs = 0）：损失是 {loss|n4}。梯度下降的一步只改动了全部 {count} 个权重和偏置中的 {moved} 个，都在各个小网络的最后一层：那里的 0 把梯度挡住了，传不到前面的层。梯度最大的那个权重从 {old} 变成 {old} − {lr} × {g} = {new}；这一批点的损失下降了 {drop}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const best = largestGrad(at);
					const u = updateOf(at, best.key, best.index);
					const b = st(at).batch?.x;
					if (!b) return { loss: NaN };
					const moved = Object.values(st(at).grads ?? {}).reduce((n, t) => n + t.data.filter((g) => g !== 0).length, 0);
					return { b: b.shape[0], loss: metric(at, 'loss'), count: nParams(st(at).params), moved, old: u.old, lr: short(u.lr), g: paren(u.g), new: u.new, drop: metric(at, 'loss') - batchLoss(at) };
				}
			},
			look: L('The rule that ran: old weight − learning rate × gradient = new weight, for the weight the stage names.', '实际执行的规则：旧权重 − 学习率 × 梯度 = 新权重，作用在图中标出的那个权重上。'),
			terms: ['maximum-likelihood', 'loss', 'gradient-descent', 'learning-rate']
		},
		{
			id: 'train',
			source: 'train',
			at: { phase: 'snapshot', last: true },
			view: plane('learn'),
			play: { from: { phase: 'snapshot', step: 0 }, to: { phase: 'snapshot', last: true }, phases: ['snapshot'], interval: 200 },
			provenance: 'live',
			title: L('The density gathers on your shape', '密度聚到你的形状上'),
			narration: {
				en: 'Training repeats such steps {steps} times with Adam (a refined gradient descent) at learning rate {lr}, {b} points per step. Over all your points the mean −log p falls from {l0} to {l1}. The density, spread over the whole plane at the start, gathers onto your shape.',
				zh: '训练把这样的一步重复 {steps} 次，用的是 Adam（改进过的梯度下降），学习率 {lr}，每步 {b} 个点。在你所有的点上，−log p 的平均从 {l0} 降到 {l1}。一开始铺满整个平面的密度，慢慢聚到你的形状上。',
				vars: (_rt, at): Record<string, TmplVal> => ({ steps: st(at).step ?? NaN, lr: short(metric(at, 'lr')), b: (at.tape?.opt as { batch?: number } | undefined)?.batch ?? NaN, l0: first(at, 'init').nllAll ?? NaN, l1: metric(at, 'nllAll') })
			},
			look: L('Press Play to watch the density over the training snapshots; stronger colour means higher density.', '按播放，看密度在各个训练快照里怎样变化；颜色越浓，密度越高。'),
			task: {
				prompt: L('Your turn: pick another shape (or paint one) in the input box, and watch the flow retrain on it.', '轮到你了：在输入框里换一个形状（或者自己画），看流模型在它上面重新训练。'),
				done: (rt) => JSON.stringify(rt.input) !== JSON.stringify(presetRaw('smiley')) && rt.sources.train?.status === 'ready',
				waiting: L('Not yet: pick another shape, then wait for the training to finish.', '还没有：先换一个形状，再等训练结束。')
			},
			terms: ['adam']
		},
		{
			id: 'sample',
			source: 'sample',
			at: { phase: 'uncouple', last: true },
			view: plane('warp'),
			play: { from: { phase: 'noise' }, to: { phase: 'uncouple', last: true }, interval: 700, unit: L('layer undone', '撤销的层') },
			provenance: 'live',
			title: L('Sampling: run it backwards', '采样：倒着跑'),
			narration: {
				en: 'To make new points, draw z from the noise and undo the layers from {top} down to 0, each with y = (y′ − t)·e^−s. Sample 0 starts at z = {z0} and ends at {x0}. After all layers, the {n} samples lie on average {dist} from the nearest of your points, and {far} of them are farther than {r} from every one.',
				zh: '要造新的点，就从噪声里抽一个 z，再从第 {top} 层到第 0 层逐层撤销，每层用 y = (y′ − t)·e^−s。0 号样本从 z = {z0} 出发，最后到了 {x0}。撤销完所有层，{n} 个样本离你最近的那个点平均相距 {dist}，其中 {far} 离你的每个点都超过 {r}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const z = first(at, 'noise');
					return { top: LAYERS - 1, z0: xy(z.px ?? NaN, z.py ?? NaN), x0: xy(metric(at, 'px'), metric(at, 'py')), n: z.n ?? NaN, dist: metric(at, 'dist'), far: { v: metric(at, 'far'), fmt: 'pct' }, r: short(0.15) };
				}
			},
			look: L('Press Play: pink noise and the noise plane’s grid are pulled back layer by layer onto your green points.', '按播放：粉色的噪声和噪声平面上的网格被一层层拉回到绿色的点上。'),
		},
		{
			id: 'limits',
			source: 'ringSample',
			at: { phase: 'uncouple', last: true },
			view: plane('samples', { gap: true }),
			kindOf: 'limits',
			provenance: 'live',
			title: L('One blob cannot become two rings', '一团噪声变不成两个圆环'),
			narration: {
				en: 'A harder shape: two rings, one inside the other, with an empty gap between them; the same flow trains on them for {steps} steps. Each layer is smooth and invertible, so it can stretch the noise but never tear it into separate pieces: thin bridges of density stay across the gap. {sgap} of the samples land in the gap, against {dgap} of the ring points.',
				zh: '换一个更难的形状：两个圆环，一个套着一个，中间是空的；同样的流模型在它们上面训练 {steps} 步。每一层都是光滑可逆的，所以它能拉伸噪声，却不能把它撕成分开的几块：空隙上总留着细细的密度桥。{sgap} 的样本落在空隙里，而圆环上的点只有 {dgap}。',
				vars: (rt, at): Record<string, TmplVal> => {
					const d = dataOf(RINGS);
					const t = rt.sources.rings?.tape;
					return { steps: t?.length ? ((t.state(t.length - 1) as St).step ?? NaN) : NaN, sgap: { v: metric(at, 'gap'), fmt: 'pct' }, dgap: { v: bandShare(d.xy, d.n, GAP[0], GAP[1]), fmt: 'pct' } };
				}
			},
			look: L('The dashed circles mark the gap: hardly any ring point lies there, yet many pink samples do.', '虚线圆标出了空隙：那里几乎没有圆环上的点，却有不少粉色的样本。')
		},
		{
			id: 'compare',
			source: 'sample',
			at: { phase: 'uncouple', last: true },
			view: plane('samples'),
			provenance: 'live',
			title: L('Flows, flow matching and score matching', '流模型、流匹配和分数匹配'),
			narration: {
				en: 'Three ways to turn noise into points like yours. This flow made the {n} samples in the stage with one pass through {K} invertible layers each, and it gives the exact log p of any place. Flow matching (the flow-2d page) learns a velocity and moves noise along it in many small Euler steps, with no invertible layers. Score matching (the score-2d page) learns only the arrows ∇log p, never p itself, and samples with long Langevin chains.',
				zh: '把噪声变成像你的点一样的新点，有三种办法。这个流模型的 {n} 个样本，每个都只需穿过 {K} 个可逆层一遍，而且它能给出任何位置精确的 log p。流匹配（见 flow-2d 页面）学一个速度场，让噪声沿着它走许多欧拉小步，不需要可逆层。分数匹配（见 score-2d 页面）只学箭头 ∇log p，从不学 p 本身，靠很长的朗之万链来采样。',
				vars: (_rt, at): Record<string, TmplVal> => ({ n: first(at, 'noise').n ?? NaN, K: LAYERS })
			},
			look: L('Pink: the samples of this flow; green: your points.', '粉色：这个流模型的样本；绿色：你的点。'),
			terms: ['velocity-field', 'score-function']
		}
	],
	fixtures: [
		{ id: 'smiley', label: L('Smiley (the start)', '笑脸（起始输入）'), raw: presetRaw('smiley') },
		{ id: 'ring', label: L('Ring', '圆环'), raw: presetRaw('ring') },
		{
			id: 'painted',
			label: L('A few painted points', '几个手画的点'),
			raw: { points: [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6].flatMap((x, i) => [{ x, y: 0.5 - 0.1 * i }, { x, y: -0.5 + 0.1 * i }]) }
		}
	]
};

export default lesson;
