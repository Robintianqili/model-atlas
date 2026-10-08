// score-2d: the guided lesson, one idea per step. Every number is a {placeholder} computed from the frames that ran (or
// the learner's points); every sentence that is not live is tested in claims.test.ts.
import type { At, ExplainerRuntime, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { fmt3, fmtSetting, paren } from '../../atlas/core/format.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { dataOf, densityNear, denoise, exact, probeOf, SHOW, SIGMAS, SMAX, SMIN, type Params } from './score.ts';
import { followed, pairsOf, predOf } from './views/plane/meta.ts';

const L = (en: string, zh: string) => ({ en, zh });
type St = { params?: Params; grads?: Params; batch?: Record<string, Tensor>; step?: number };
const st = (at: At) => (at.state ?? {}) as St;
const metric = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const xy = (a: number, b: number) => `(${fmt3(a)}, ${fmt3(b)})`;
/** a set value (σ, α, a radius, a learning rate) as set: 0.16 and 0.0032, not 0.160 and 0.00320 (the kit's fmtSetting);
 *  a missing one stays NaN, so lint reports it as unresolved */
const short = (v: number) => (Number.isFinite(v) ? fmtSetting(v) : NaN);
const plane = (mode: string, extra: Record<string, unknown> = {}) => ({ key: 'score-2d/plane', props: () => ({ mode, ...extra }) });
const nParams = (p: Params | undefined) => Object.values(p ?? {}).reduce((s, t) => s + t.data.length, 0);
const nearest = (d: Points2D, x: number, y: number) => Array.from({ length: d.n }, (_, j) => Math.hypot(x - d.xy[2 * j], y - d.xy[2 * j + 1])).reduce((a, b) => Math.min(a, b), Infinity);
/** the density at your most crowded point (σ = 0.16): found with densityNear, then computed exactly; once per input */
const peaks = new WeakMap<Points2D, number>();
function peakOf(d: Points2D): number {
	if (!peaks.has(d)) {
		let [top, best] = [0, -1];
		for (let j = 0; j < d.n; j++) {
			const v = densityNear(d, d.xy[2 * j], d.xy[2 * j + 1], SHOW);
			if (v > best) [top, best] = [j, v];
		}
		peaks.set(d, exact(d, d.xy[2 * top], d.xy[2 * top + 1], SHOW).density);
	}
	return peaks.get(d)!;
}
/** the metrics of a Langevin source's last frame (its end-of-run numbers) and of its start frame */
function ends(rt: ExplainerRuntime, source: string) {
	const t = rt.sources[source]?.tape;
	const i = t ? t.find({ phase: 'move', last: true }) : -1;
	return t && i >= 0 ? { end: t.frame(i).metrics ?? {}, start: t.frame(t.find({ phase: 'start' })).metrics ?? {}, T: t.frame(i).step! + 1 } : null;
}
/** the weight with the largest gradient: key, [out, in] */
function largestGrad(at: At): { key: string; index: [number, number] } {
	let best = { key: 'net.0.weight', index: [0, 0] as [number, number], g: -1 };
	for (const [key, t] of Object.entries(st(at).grads ?? {})) if (key.endsWith('.weight')) t.data.forEach((g, k) => Math.abs(g) > best.g && (best = { key, index: [Math.floor(k / t.shape[1]), k % t.shape[1]], g: Math.abs(g) }));
	return best;
}
/** the mean of n recorded training losses from the start (first) or up to the step's frame (last) */
function meanLoss(at: At, n: number, from: 'first' | 'last'): number {
	const vs: number[] = [];
	for (let i = 0; at.tape && i <= at.index; i++) {
		const f = at.tape.frame(i);
		if ((f.phase === 'step' || f.phase === 'snapshot') && typeof f.metrics?.loss === 'number' && (f.step ?? 0) < (st(at).step ?? 0)) vs.push(f.metrics.loss);
	}
	const w = from === 'first' ? vs.slice(0, n) : vs.slice(-n);
	return w.length ? w.reduce((a, b) => a + b, 0) / w.length : NaN;
}
/** the mean squared error of the frame's weights on the frame's batch (the loss after an update, on the same copies) */
function batchLoss(at: At): number {
	const b = st(at).batch;
	const p = st(at).params;
	if (!b || !p) return NaN;
	const n = b.inp.shape[0];
	const pts = Float64Array.from({ length: 2 * n }, (_, k) => b.inp.data[3 * (k >> 1) + (k & 1)]);
	const d = denoise(p, pts, Float64Array.from({ length: n }, (_, i) => Math.exp(b.inp.data[3 * i + 2])), n);
	return d.reduce((s, v, k) => s + (v - b.target.data[k]) ** 2, 0) / (2 * n);
}
/** the example the lesson follows in the frame's batch, with the network's output for it (fine frames) */
function example(at: At) {
	const q = pairsOf({ at });
	if (!q) return null;
	const i = followed(q);
	const s = q.s[i];
	return { i, n: q.n, s, x: q.x[2 * i], y: q.x[2 * i + 1], ex: q.e[2 * i], ey: q.e[2 * i + 1], xt: q.xt[2 * i], yt: q.xt[2 * i + 1], tx: -q.e[2 * i] / s, ty: -q.e[2 * i + 1] / s, d: predOf({ at }, i) };
}

const lesson: LessonDef = {
	title: L('Score matching: arrows toward your points, and samplers that follow them', '分数匹配：指向你的点的箭头，以及跟着它们走的采样器'),
	summary: L(
		'The density of your points and its score, a network that learns the score from noisy copies, Langevin chains that follow it with a little noise, annealing from large to small noise, and where the learned score fails.',
		'你的点的密度和它的分数，一个从加噪副本学分数的网络，带一点噪声跟着分数走的朗之万链，从大噪声到小噪声的退火，以及学到的分数在哪里失灵。'
	),
	steps: [
		{
			id: 'data',
			source: 'train',
			at: 'first',
			view: plane('data'),
			provenance: 'live',
			title: L('Your points', '你的点'),
			narration: {
				en: 'Your shape: {pts}. A generative model learns to make new points that look as if they came from the same shape. This one learns, at every place, an arrow that points toward where your points are dense; a sampler then walks along the arrows.',
				zh: '你的形状：{pts}。生成模型学的是造出新的点，让它们看起来像是来自同一个形状。这个模型在每个位置学一个箭头，指向你的点密集的地方；然后采样器顺着箭头走。',
				vars: (rt) => ({ pts: plural(dataOf(rt.input).n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' }) })
			},
			look: L('The green points. Pick another shape or paint your own in the input box: every step follows.', '绿色的点。可以在输入框里换一个形状，或者自己画：后面每一步都会跟着变。'),
			terms: ['generative-model']
		},
		{
			id: 'density',
			source: 'train',
			at: 'first',
			view: plane('density'),
			provenance: 'live',
			title: L('Where points crowd, the density is high', '点越挤，密度越高'),
			narration: {
				en: 'Put a small bell of width σ = {s} on each of your points and average them: that is a density p(x), high where points crowd. At the ringed point, {d} from your nearest point, p = {pr}. At your most crowded point it reaches {pmax}. Farther out, {f} from your points, it is {pc}: tiny, but never exactly 0.',
				zh: '在你的每个点上放一个宽度 σ = {s} 的小钟形，再求平均：这就是密度 p(x)，点越挤的地方越高。圈出的点离你最近的点 {d}，那里 p = {pr}。在你的点最挤的地方，它达到 {pmax}。再往外，离你的点 {f} 的地方，它是 {pc}：极小，但不会正好是 0。',
				vars: (rt): Record<string, TmplVal> => {
					const d = dataOf(rt.input);
					const [px, py] = probeOf(d);
					return { s: short(SHOW), d: nearest(d, px, py), pr: exact(d, px, py, SHOW).density, pmax: peakOf(d), f: nearest(d, px + 0.7, py), pc: exact(d, px + 0.7, py, SHOW).density };
				}
			},
			look: L('Stronger colour: higher density. It follows your points, a little blurred.', '颜色越浓，密度越高。它跟着你的点走，只是稍微模糊了一点。'),
			terms: ['density']
		},
		{
			id: 'score',
			source: 'train',
			at: 'first',
			view: plane('score'),
			provenance: 'live',
			title: L('The score points uphill', '分数指向上坡'),
			predict: {
				id: 'score-2d-uphill',
				question: L('At the ringed point, which way does the score arrow go?', '圈出的点处的分数箭头指向哪里？'),
				options: [L('Toward the nearest part of your shape', '指向你的形状离它最近的部分'), L('Away from your points, toward emptier space', '背离你的点，指向更空的地方'), L('Nowhere: the density there is almost 0, so there is no arrow', '哪儿也不指：那里的密度几乎是 0，所以没有箭头')],
				answer: 0,
				explain: L(
					'Toward your shape. The score is a weighted average of (x_j − x) / σ² over your points, and the nearer a point, the larger its weight. Far out, log p falls like the side of a bowl: the farther you go, the longer the arrow.',
					'指向你的形状。分数是 (x_j − x) / σ² 在你的点上的加权平均，离它越近的点，权重越大。在远处，log p 像碗壁一样往下掉：走得越远，箭头越长。'
				)
			},
			narration: {
				en: 'The score is the gradient of log p: at every place, the direction in which the density grows fastest, and how fast. We take the log so that a place with a tiny density still has a clear direction. The ringed point lies {d} from your nearest point.',
				zh: '分数是 log p 的梯度：在每个位置，密度增长最快的方向，以及增长有多快。取对数是为了让密度极小的地方也有清楚的方向。圈出的点离你最近的点 {d}。',
				vars: (rt) => {
					const d = dataOf(rt.input);
					return { d: nearest(d, ...probeOf(d)) };
				}
			},
			look: L('Before the arrows appear: guess where the ringed point’s arrow goes.', '箭头出现之前，先猜圈出的点的箭头指向哪里。'),
			terms: ['score-function', 'gradient']
		},
		{
			id: 'noise',
			source: 'sgd1',
			at: { phase: 'forward', step: 0 },
			view: plane('pairs'),
			provenance: 'live',
			title: L('Add noise, learn the way back', '加噪声，学会回去的路'),
			narration: {
				en: 'Step 3 summed over all your points; with millions of images that is far too slow, so real models learn from noisy copies instead: denoising score matching. Your point x = {x}, a noise level σ = {s} (each copy draws its own) and noise ε = {e} give x̃ = x + σ·ε = {xt}. The arrow back, (x − x̃) / σ², is the target: {t}. Averaged over many copies, the best guess for this target is the score of your points blurred by σ.',
				zh: '第 3 步对你的每个点求和；图像有上百万张时，这个和慢得没法用，所以真正的模型改从加噪副本里学分数：这就是去噪分数匹配。你的点 x = {x}、噪声水平 σ = {s}（每个副本各抽一个）和噪声 ε = {e} 给出 x̃ = x + σ·ε = {xt}。回去的箭头 (x − x̃) / σ² 就是目标：{t}。对许多副本取平均，对这个目标最好的猜测正是你的点用 σ 模糊后的分数。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const q = example(at);
					return q ? { x: xy(q.x, q.y), s: q.s, e: xy(q.ex, q.ey), xt: xy(q.xt, q.yt), t: xy(q.tx, q.ty) } : { s: NaN };
				}
			},
			look: L('Grey: one batch of noisy copies, each joined to the point it came from. The ringed copy’s green arrow points back.', '灰色：一批加噪副本，每个都连着它来自的那个点。圈出的副本上，绿色箭头指回去。'),
			terms: ['denoising-score-matching', 'gaussian-noise']
		},
		{
			id: 'step',
			source: 'sgd1',
			at: { phase: 'update', step: 0 },
			view: 'train-step',
			focus: (_rt, at) => [{ field: 'params', key: largestGrad(at).key, index: largestGrad(at).index, role: 'param' }],
			provenance: 'live',
			title: L('One training step', '一步训练'),
			narration: {
				en: 'The network d reads the noisy copy’s two coordinates and log σ, and outputs two numbers. Its score is d / σ, and the target (x − x̃) / σ² equals −ε / σ, so d should be −ε = {me}; untrained it says {d}. The loss is the mean squared miss over all {b} copies: {loss|n4}. One gradient-descent step changes all {count} weights and biases (the one with the largest gradient goes {old} − {lr} × {g} = {new}), and on the same copies the loss falls by {drop}.',
				zh: '网络 d 读入加噪副本的两个坐标和 log σ，输出两个数。它的分数是 d / σ，而目标 (x − x̃) / σ² 等于 −ε / σ，所以 d 应该是 −ε = {me}；未训练时它给出 {d}。损失是全部 {b} 个副本的均方误差：{loss|n4}。梯度下降的一步改动全部 {count} 个权重和偏置（梯度最大的那个权重从 {old} 变成 {old} − {lr} × {g} = {new}），在同样的副本上，损失下降了 {drop}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const q = example(at);
					const best = largestGrad(at);
					const u = updateOf(at, best.key, best.index);
					const after = batchLoss(at);
					if (!q?.d) return { loss: NaN };
					return { me: xy(-q.ex, -q.ey), d: xy(q.d[0], q.d[1]), b: q.n, loss: metric(at, 'loss'), count: nParams(st(at).params), old: u.old, lr: short(u.lr), g: paren(u.g), new: u.new, drop: metric(at, 'loss') - after };
				}
			},
			look: L('The rule that ran: old weight − learning rate × gradient = new weight, for the weight the stage names.', '实际执行的规则：旧权重 − 学习率 × 梯度 = 新权重，作用在图中标出的那个权重上。'),
			terms: ['loss', 'mse', 'gradient-descent', 'learning-rate']
		},
		{
			id: 'field',
			source: 'train',
			at: { phase: 'snapshot', last: true },
			view: plane('field'),
			play: { from: { phase: 'snapshot', step: 0 }, to: { phase: 'snapshot', last: true }, phases: ['snapshot'], interval: 200 },
			provenance: 'live',
			title: L('The learned arrows', '学到的箭头'),
			narration: {
				en: 'Training repeats steps like this {steps} times, with Adam (a refined gradient descent) at learning rate {lr}, each on {b} fresh copies at noise levels between {smin} and {smax}. Averaged over {win} steps the loss falls from {l0} to {l1}{floor}. The arrows are the learned score at the slider’s σ.',
				zh: '训练把这样的一步重复 {steps} 次，用的是 Adam（改进过的梯度下降），学习率 {lr}，每次用 {b} 个新的副本，噪声水平在 {smin} 到 {smax} 之间。按 {win} 步取平均，损失从 {l0} 降到 {l1}{floor}。箭头是滑块所选 σ 下学到的分数。',
				vars: (rt, at): Record<string, TmplVal> => ({
					steps: st(at).step ?? NaN, lr: short(metric(at, 'lr')), b: (at.tape?.opt as { batch?: number } | undefined)?.batch ?? NaN, smin: short(SMIN), smax: SMAX, win: 50, l0: meanLoss(at, 50, 'first'), l1: meanLoss(at, 50, 'last'),
					floor: pick(dataOf(rt.input).n > 1, L('; it cannot reach 0, because one noisy point could have come from many of your points', '；它降不到 0，因为同一个加噪的点可能来自你的许多个点'), L('; with a single point every copy has only one way back', '；只有一个点时，每个副本都只有一条回去的路'))
				})
			},
			look: L('Slide σ: at the largest σ every arrow points toward the middle of your shape; at the smallest, toward its nearest part.', '拖动 σ：σ 最大时，每个箭头都指向你的形状的中间；σ 最小时，指向形状离它最近的部分。'),
			terms: ['optimizer']
		},
		{
			id: 'langevin',
			source: 'walk',
			at: { phase: 'move', last: true },
			view: plane('chains', { sliders: true }),
			play: { from: { phase: 'start' }, to: { phase: 'move', last: true }, interval: 60, unit: L('Langevin step', '朗之万步') },
			provenance: 'live',
			title: L('Langevin: follow the arrows, plus a little noise', '朗之万法：顺着箭头走，再加一点噪声'),
			narration: {
				en: 'Langevin dynamics turns the arrows into samples: start {n} chains anywhere and repeat x ← x + (α/2)·s(x) + √α·z, where s is the network’s score at σ = {s}, α = {a} is the step size and z is fresh noise each time. Chain 0’s first step: {p} + {ha} × {g} + {ra} × {z} = {q}. After {T} steps the chains lie on average {d1} from your points; they started {d0} away.',
				zh: '朗之万动力学把箭头变成样本：从任意位置出发放 {n} 条链，反复执行 x ← x + (α/2)·s(x) + √α·z，其中 s 是网络在 σ = {s} 时给出的分数，α = {a} 是步长，z 是每次重新抽的噪声。0 号链的第一步：{p} + {ha} × {g} + {ra} × {z} = {q}。走了 {T} 步之后，链离你的点平均 {d1}；出发时离 {d0}。',
				vars: (rt): Record<string, TmplVal> => {
					const r = ends(rt, 'walk');
					const t = rt.sources.walk?.tape;
					const m = t?.frame(t.find({ phase: 'move', step: 0 })).metrics;
					if (!r || !m) return { n: NaN };
					return { n: r.start.chains, s: short(m.sigma), a: short(m.alpha), p: xy(m.px, m.py), ha: short(m.alpha / 2), g: xy(m.sx, m.sy), ra: Math.sqrt(m.alpha), z: xy(m.zx, m.zy), q: xy(m.nx, m.ny), T: r.T, d1: r.end.dist, d0: r.start.dist };
				}
			},
			look: L('Pink: the chains and their paths; chain 0 is ringed. Move the sliders to re-run the chains with another step size or less noise.', '粉色：链和它们的路径；0 号链被圈出。拖动滑块，用别的步长或更少的噪声重新跑这些链。'),
			task: {
				prompt: L('Your turn: pick another shape (or paint one) in the input box, and watch the network relearn and the chains follow.', '轮到你了：在输入框里换一个形状（或者自己画），看网络重新学习、链跟着过去。'),
				done: (rt) => JSON.stringify(rt.input) !== JSON.stringify(presetRaw('smiley')) && rt.sources.walk?.status === 'ready',
				waiting: L('Not yet: pick another shape, then wait for the training and the chains to finish.', '还没有：先换一个形状，再等训练和链跑完。')
			},
			terms: ['langevin-dynamics']
		},
		{
			id: 'anneal',
			source: 'annealed',
			at: { phase: 'move', last: true },
			view: plane('chains'),
			compare: { a: 'plain', b: 'annealed', label: [L(`σ = ${SMIN} only`, `只用 σ = ${SMIN}`), L(`annealed: σ from ${SMAX} down to ${SMIN}`, `退火：σ 从 ${SMAX} 降到 ${SMIN}`)] },
			provenance: 'live',
			title: L('Annealing: from big noise to small', '退火：从大噪声到小噪声'),
			narration: {
				en: 'A small σ shows the fine shape, but there the network’s arrows are weak far from your points (the last step shows why). Annealed Langevin starts at σ = {smax}, where the arrows are smooth and nearly right everywhere, and halves σ every {T} steps down to {smin}, shrinking the step size α with σ². Of the chains run at σ = {smin} only, {pfar} are still more than {r} from your points after the same {total} steps; of the annealed ones, {afar}. {verdict}',
				zh: '小的 σ 能显出精细的形状，但在远离你的点的地方，网络在小 σ 时的箭头很弱（最后一步会说明原因）。退火朗之万法从 σ = {smax} 开始，那里的箭头平滑，而且处处都接近正确，然后每 {T} 步把 σ 减半，直到 {smin}，步长 α 也跟着 σ² 缩小。同样走 {total} 步，只用 σ = {smin} 的链有 {pfar} 离你的点仍超过 {r}；退火的链是 {afar}。{verdict}',
				vars: (rt): Record<string, TmplVal> => {
					const [p, a] = [ends(rt, 'plain'), ends(rt, 'annealed')];
					if (!p || !a) return { total: NaN };
					const [pf, af] = [p.end.far, a.end.far];
					return {
						smax: SMAX, smin: short(SMIN), r: short(0.2), T: a.T / SIGMAS.length, total: a.T, pfar: { v: pf, fmt: 'pct' }, afar: { v: af, fmt: 'pct' },
						verdict: pick(af < 0.6 * pf, L('Starting big brings far more of them in.', '从大噪声开始，走进来的链多得多。'), L('On this shape the small-σ arrows are strong enough: both get there.', '在这个形状上，小 σ 的箭头也够有力：两种都能走到。'))
					};
				}
			},
			look: L('The first panel uses the smallest σ for every step, the second anneals. Compare how many pink points are left far from the green shape.', '第一个面板每一步都用最小的 σ，第二个用退火。比一比有多少粉色点还留在离绿色形状很远的地方。'),
			terms: ['annealing']
		},
		{
			id: 'flow',
			source: 'annealed',
			at: { phase: 'move', last: true },
			view: plane('chains'),
			compare: { a: 'still', b: 'annealed', label: [L('annealed, without the noise term', '退火，去掉噪声项'), L('annealed, with it', '退火，保留噪声项')] },
			provenance: 'live',
			title: L('Why the noise, and how this compares with flow matching', '为什么要噪声，以及它和流匹配的比较'),
			narration: {
				en: 'Without the noise term, annealing only climbs: every chain heads uphill on p, and the chains end in a few tight bunches. At the end, {cs} of your points have a chain within {r}, against {ca} with noise{same}. Flow matching (the flow-2d page) needs no noise: its velocity carries the whole noise cloud onto the shape. A diffusion model (DDPM) predicts the noise ε; its guess ε̂ gives the score s = −ε̂ / σ, just like our d / σ.',
				zh: '去掉噪声项，退火就只剩往上爬：每条链都顺着 p 往高处走，最后挤成几小团。到最后，你的点中有 {cs} 在 {r} 之内有链，有噪声时是 {ca}{same}。流匹配（见 flow-2d 页面）不需要噪声：它的速度把整团噪声搬到形状上。扩散模型（DDPM）预测噪声 ε；用它的猜测 ε̂ 就得到分数 s = −ε̂ / σ，和我们的 d / σ 一样。',
				vars: (rt): Record<string, TmplVal> => {
					const [s, a] = [ends(rt, 'still'), ends(rt, 'annealed')];
					return s && a
						? { r: short(0.1), cs: { v: s.end.cover, fmt: 'pct' }, ca: { v: a.end.cover, fmt: 'pct' }, same: pick(s.end.cover >= a.end.cover, L(' (with so few points, both runs still reach every one)', '（点这么少，两种都能碰到每一个点）'), L('', '')) }
						: { cs: NaN };
				}
			},
			look: L('First panel: no noise, the chains end in a few tight bunches. Second panel: with noise, they spread over your shape.', '第一个面板：没有噪声，链最后挤成几小团。第二个面板：有噪声，它们铺满你的形状。'),
			terms: ['velocity-field']
		},
		{
			id: 'limits',
			source: 'limits',
			at: 'last',
			view: plane('limits'),
			kindOf: 'limits',
			provenance: 'live',
			title: L('Far from the data, the score is a guess', '远离数据，分数只是猜测'),
			narration: {
				en: 'At σ = {s} hardly any noisy copy lands more than 3σ = {r} from your points, yet such places are {share} of the plane: the network barely trained there. Its arrows still point roughly the right way (off by {ang}° on average) but are about {k} times too short: 1 to the right of your rightmost point the true score is {e1} long, the network’s {n1}. A chain started there at σ = {s} alone crawls about {k} times too slowly, which is why annealing starts big.',
				zh: '在 σ = {s} 时，几乎没有加噪副本会落到离你的点超过 3σ = {r} 的地方，可这样的地方占了平面的 {share}：网络在那里几乎没受过训练。它的箭头方向大致还对（平均偏 {ang}°），但长度只有真实的约 1/{k}：在你最右边的点再往右 1 处，真实的分数长 {e1}，网络给出的只有 {n1}。只用 σ = {s} 的链若从那里出发，速度只有应有的约 1/{k}，所以退火要从大 σ 开始。',
				vars: (_rt, at): Record<string, TmplVal> => ({ s: short(SMIN), r: short(3 * SMIN), share: { v: metric(at, 'share'), fmt: 'pct' }, ang: metric(at, 'angle'), k: metric(at, 'times'), e1: metric(at, 'exact1'), n1: metric(at, 'net1') })
			},
			look: L('Green: the true score at the smallest σ; pink: the network’s. Both lengths are on the same log scale, so a much shorter arrow stays visible: far out, the pink ones fall clearly short.', '绿色：最小的 σ 时真实的分数；粉色：网络给出的。两者的长度按同一个对数比例画，所以短得多的箭头也看得见：在远处，粉色箭头明显短一截。')
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
