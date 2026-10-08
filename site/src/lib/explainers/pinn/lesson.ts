// pinn: the guided lesson (design §7 outline), one idea per step. Every number is a {placeholder} computed from the runs
// in the learner's browser (or the process frames); every sentence that is not live is tested in claims.test.ts.
import type { At, Cursor, ExplainerRuntime, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { fmt3, minus } from '../../atlas/core/format.ts';
import { dampedOscillator } from '../../atlas/data/series.ts';
import { OSCILLATOR } from '../../atlas/adapters/series/adapter.ts';
import { C, K, N, SPLIT, derivs, meanSq, pointsOf } from './osc.ts';
import { hText, probe } from './process.ts';
// options known before a run (the sources set no other): the narration shows them while the source still trains
import { netAt, pinn, pinnWrongC, plain } from './trainer.ts';
import { withGradMode } from '../../atlas/autograd/var.ts';

const L = (en: string, zh: string) => ({ en, zh });
/** a setting as written (3.6, not 3.60); NaN stays NaN so lint reports it */
const setting = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? String(+v.toPrecision(6)) : NaN);
const metric = (at: At, k: string): number => at.frame?.metrics?.[k] ?? NaN;
function metricAt(at: At, c: Cursor, k: string): number {
	const i = at.tape ? at.tape.find(c) : -1;
	return i >= 0 ? (at.tape!.frame(i).metrics?.[k] ?? NaN) : NaN;
}
type Snap = { m: Record<string, number>; cv?: Tensor; opt: Record<string, number> };
/** a source's latest frame with weights: its metrics, its derived curves [4, N] and its options */
function snap(rt: ExplainerRuntime, source: string): Snap {
	const t = rt.sources[source]?.tape;
	let i = t ? t.length - 1 : -1;
	while (t && i > 0 && t.frame(i).phase !== 'snapshot') i--;
	if (!t || i < 0) return { m: {}, opt: {} };
	let cv: Tensor | undefined;
	try {
		cv = (t.state(i) as { curves?: Tensor }).curves;
	} catch {
		cv = undefined;
	}
	return { m: t.frame(i).metrics ?? {}, cv, opt: (t.opt ?? {}) as Record<string, number> };
}
const phys = (s: Snap) => (s.cv && s.cv.data.length === 4 * N ? meanSq(s.cv, 3) : NaN);
/** a ratio as the learner reads it: "4,590 times", "142 times", "35.5 times" (3 digits below 100, whole numbers above) */
const times = (r: number) => (r >= 100 ? Math.round(r) : r);
/** significant decimal digits of a float64, the number type the browser computes in (16) */
const DIGITS = Math.round(-Math.log10(Number.EPSILON));
/** −0 (the data loss of an input without points) as 0 */
const z0 = (x: number) => (x === 0 ? 0 : x);
const ratioText = (r: number) => (r >= 100 ? Math.round(r).toLocaleString('en-US') : fmt3(r));
const truthAt = (t: number) => dampedOscillator([t], OSCILLATOR)[0];
const fit = (props: Record<string, unknown> = {}) => ({ key: 'pinn/residual-strip', props: () => props });
const done = (rt: ExplainerRuntime, s: string) => {
	const t = rt.sources[s]?.tape as { length: number; done?: boolean } | null | undefined;
	return !!t && t.length > 0 && t.done !== false;
};

const lesson: LessonDef = {
	title: L('Physics-informed network: a spring, eight points and a law', '物理信息神经网络：一个弹簧、八个点和一条定律'),
	summary: L(
		'A plain network fits a few measured points of a damped spring and wanders off after them; a PINN, trained on the same points plus the law of the spring, carries the swing forward. Then the weight λ of the law, and what a wrong law does.',
		'普通网络能拟合阻尼弹簧的几个测量点，但在这些点之后就乱走；PINN 用同样的点再加上弹簧的定律来训练，能把振动延续下去。然后看定律的权重 λ，以及错误的定律会造成什么。'
	),
	steps: [
		{
			id: 'law',
			source: 'pinn',
			at: 'first',
			view: fit({ net: false, strip: false, points: false, split: false }),
			provenance: 'live',
			title: L('The law of a damped spring', '阻尼弹簧的定律'),
			narration: {
				en: 'A weight hangs on a spring and rubs against some friction. Pull it to u = 1 and let go: its position u(t) swings back and forth and dies down (the dashed curve). Physics gives the rule, for a mass of 1: u″ = −{k}·u − {c}·u′, where u′ is the velocity (how fast u changes) and u″ the acceleration (how fast u′ changes): the spring pulls back with {k}·u and the friction brakes with {c}·u′. With everything on one side: u″ + {c}·u′ + {k}·u = 0.',
				zh: '一个重物挂在弹簧上，还受到一点摩擦。把它拉到 u = 1 再松手：它的位置 u(t) 来回振动，并逐渐停下来（虚线）。物理给出的规律是（质量取 1）：u″ = −{k}·u − {c}·u′，其中 u′ 是速度（u 变化得多快），u″ 是加速度（u′ 变化得多快）：弹簧用 {k}·u 把它往回拉，摩擦用 {c}·u′ 让它减速。把所有项移到等号一边，就是 u″ + {c}·u′ + {k}·u = 0。',
				vars: () => ({ c: setting(C), k: setting(K) })
			},
			look: L('The dashed curve: every swing is smaller than the one before. That shrinking is the friction at work.', '看虚线：每一次摆动都比上一次小。这种缩小就是摩擦在起作用。'),
			terms: ['derivative']
		},
		{
			id: 'data',
			source: 'pinn',
			at: 'first',
			view: fit({ net: false, strip: false }),
			provenance: 'live',
			title: L('A few measured points', '几个测量点'),
			narration: {
				en: 'In an experiment you only measure a few positions, each a little off because of noise. {have} The points are your input, in the “Your input” box: add, drag or remove them there. The networks never see the true curve (dashed here, the line in the input box); they have to find it from the points.',
				zh: '在实验里，你只能测到几个位置，而且每个都因为噪声有点偏差。{have}这些点就是你的输入，在“你的输入”框里：可以添加、拖动或删除。网络从来看不到真实曲线（这里的虚线，也就是输入框里那条线），只能从这些点出发去找它。',
				vars: (rt) => {
					const p = pointsOf(rt.input);
					if (!p.n) return { have: L('Here there are none yet.', '这里还一个都没有。') };
					let noise = 0;
					for (let i = 0; i < p.n; i++) noise += Math.abs(p.u[i] - truthAt(p.t[i])) / p.n;
					const [a, b, e] = [fmt3(Math.min(...p.t)), fmt3(Math.max(...p.t)), fmt3(noise)];
					return {
						have: pick(
							p.n > 1,
							{ en: `Here: ${p.n} points between t = ${a} and t = ${b}, on average ${e} away from the true curve.`, zh: `这里有 ${p.n} 个点，在 t = ${a} 和 t = ${b} 之间，平均离真实曲线 ${e}。` },
							{ en: `Here: one point, at t = ${a}, ${e} away from the true curve.`, zh: `这里只有一个点，在 t = ${a}，离真实曲线 ${e}。` }
						)
					};
				}
			},
			look: L('Where the points are, and the long stretch of time where there are none.', '看这些点在哪里，以及后面那一大段没有任何点的时间。')
		},
		{
			id: 'plain',
			source: 'plain',
			at: 'last',
			view: fit({ strip: false }),
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 90 },
			provenance: 'live',
			title: L('A plain network learns the points', '普通网络学习这些点'),
			predict: {
				id: 'pinn-plain',
				question: L('After the last point there is no data. What will the plain network’s curve do there?', '最后一个点之后就没有数据了。普通网络的曲线在那里会怎样？'),
				options: [L('Keep swinging and die down, like the true curve', '继续摆动并逐渐停下，和真实曲线一样'), L('Go its own way: nothing pulls it towards the true curve', '自己乱走：没有任何东西把它拉向真实曲线'), L('Stop and stay at the height of the last point', '停下来，保持在最后一个点的高度')],
				answer: 1,
				explain: L(
					'Its loss only measures the gaps at the points, so after the last point every curve is equally good to it, and it wanders off. Nothing in the points says that the swing goes on.',
					'它的损失只量数据点上的差距，所以在最后一个点之后，任何曲线对它来说都一样好，于是它就乱走了。数据点里没有任何信息说振动还会继续。'
				)
			},
			narration: {
				en: 'Start with a plain network: one input (the time t), two hidden layers of {h} neurons (each squashes a weighted sum with tanh) and one output (the position u), {np} weights and biases in all. It learns from the points only: its loss, the data loss, is the average of the squared gaps between its curve and the points. Adam, an optimizer that adapts each weight’s step, ran {steps} steps and took the data loss from {d0} to {d1}{dec}.',
				zh: '先看一个普通网络：一个输入（时间 t），两个隐藏层各 {h} 个神经元（每个神经元用 tanh 压缩一个加权和），一个输出（位置 u），一共 {np} 个权重和偏置。它只从数据点学习：它的损失叫数据损失，是曲线与各个点之间差距的平方的平均。Adam 是一种会为每个权重调整步长的优化器，它训练了 {steps} 步，把数据损失从 {d0} 降到 {d1}{dec}。',
				vars: (_rt, at) => {
					const ps = Object.values(((at.state as { params?: Record<string, Tensor> })?.params ?? {}) as Record<string, Tensor>);
					const d1 = z0(metric(at, 'data'));
					// the first number in e-notation on the page: say once what it means (5.28e−5 = 0.0000528)
					const small = d1 >= 1e-7 && d1 < 1e-3;
					const dec = small ? d1.toFixed(2 - Math.floor(Math.log10(d1))) : '';
					return { h: (at.tape?.opt as { hidden?: number })?.hidden ?? plain.defaults.hidden, np: ps.length ? ps.reduce((s, t) => s + t.data.length, 0) : NaN, steps: at.frame?.step ?? NaN, d0: z0(metricAt(at, { phase: 'snapshot', step: 0 }, 'data')), d1, dec: pick(small, { en: `, that is ${dec}`, zh: `（也就是 ${dec}）` }, { en: '', zh: '' }) };
				}
			},
			look: L('Play: the curve moves to the points. Then look at what it does after them.', '播放：曲线向数据点靠拢。然后看看它在这些点之后做了什么。'),
			terms: ['mlp', 'loss', 'optimizer']
		},
		{
			id: 'deriv',
			source: 'probe',
			at: { phase: 'fd', step: 3 },
			// the step's own formula: each finite difference right above its autodiff value, with 8 digits so the learner sees
			// where they agree and where they part (the phase formula prints 3 digits: 1.14 and 1.14)
			view: {
				key: 'zoom.formula',
				props: (_rt, at) => {
					const v = (k: string) => (Number.isFinite(metric(at, k)) ? minus(String(+metric(at, k).toPrecision(8))) : '…');
					const t = Number.isFinite(metric(at, 't')) ? String(setting(metric(at, 't'))) : '…';
					return {
						tex: "\\begin{aligned} h &= {{h}} \\\\ u' &\\approx \\tfrac{u(t+h) - u(t-h)}{2h} \\\\ &= {{slope}} \\\\ u'_{\\text{autodiff}} &= {{slopeAD}} \\\\[6pt] u'' &\\approx \\tfrac{u(t+h) - 2u(t) + u(t-h)}{h^2} \\\\ &= {{bend}} \\\\ u''_{\\text{autodiff}} &= {{bendAD}} \\end{aligned}",
						values: { h: hText(metric(at, 'h')), slope: v('fd1'), slopeAD: v('d1'), bend: v('fd2'), bendAD: v('d2') },
						caption: { en: `The plain network at t = ${t}: finite differences and autodiff`, zh: `普通网络在 t = ${t} 处：有限差分与自动微分` }
					};
				}
			},
			provenance: 'live',
			title: L('Slope and bend by autodiff', '用自动微分求斜率和弯曲'),
			narration: {
				en: 'The law needs the slope u′ and the bend u″ of the plain network’s curve. Autodiff runs the chain rule backwards through the network (like backpropagation, but towards t) and gives them exactly: at t = {t}, u′ = {d1} and u″ = {d2}. Finite differences with a small step h = {h} agree to within {e1} and {e2}. At h = {hTiny} the bend sits in the last few of the {digits} digits the computer keeps for u, rounding takes over, and u″ is {eTiny} off; autodiff needs no step.',
				zh: '定律需要普通网络曲线的斜率 u′ 和弯曲 u″。自动微分把链式法则沿网络倒着走一遍（和反向传播一样，只不过是对 t 求导），给出精确值：在 t = {t}，u′ = {d1}，u″ = {d2}。用小步长 h = {h} 的有限差分与它们只差 {e1} 和 {e2}。到了 h = {hTiny}，弯曲只体现在电脑为 u 保存的 {digits} 位数字的最后几位上，舍入误差占了上风，u″ 差了 {eTiny}；自动微分则根本不需要步长。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const hs = (k: string) => (Number.isFinite(metric(at, k)) ? hText(metric(at, k)) : NaN);
					return { t: setting(metric(at, 't')), d1: metric(at, 'd1'), d2: metric(at, 'd2'), h: hs('h'), e1: metric(at, 'e1'), e2: metric(at, 'e2'), hTiny: hs('hTiny'), eTiny: metric(at, 'eTiny'), digits: DIGITS };
				}
			},
			look: {
				en: 'In the formula, each finite difference stands above the exact autodiff value: compare their digits. PyTorch’s autograd, on the same network, gives the same u′ and u″ to within a relative difference of {fact:derivParity}.',
				zh: '公式里，每个有限差分的结果下面紧跟着自动微分的精确值：比一比它们的数字。在同一个网络上，PyTorch 的自动微分算出的 u′ 和 u″ 与这里的相对差距不到 {fact:derivParity}。'
			},
			terms: ['autodiff', 'chain-rule', 'finite-difference', 'backpropagation']
		},
		{
			id: 'residual',
			source: 'plain',
			at: 'last',
			view: fit(),
			provenance: 'live',
			title: L('The residual: does the curve obey the law?', '残差：曲线遵守定律吗？'),
			narration: {
				en: 'Now put the plain network’s curve into the law. At t = {t}, step 4’s u″ and u′ and u = {u|n4} give ({d2} + {c} × {d1p} + {k} × {up}) ÷ {k} = {r}: {verdict}. This residual is 0 wherever the curve obeys the law (dividing by {k} makes it read in units of position); the strip shows it at {nT} times, and the average of its squares, the physics loss, is {phys}. The curve was never asked to obey the law: after t = {split} it is {err} from the true curve on average.',
				zh: '现在把普通网络的曲线代入定律。在 t = {t}，用第 4 步的 u″、u′ 和 u = {u|n4}：({d2} + {c} × {d1p} + {k} × {up}) ÷ {k} = {r}，{verdict}。曲线在哪里遵守定律，那里的这个残差就是 0（除以 {k} 是为了让它的单位和位置一样）；条带画出了 {nT} 个时刻上的残差，它的平方的平均，也就是物理损失，是 {phys}。从来没有人要求这条曲线遵守定律：t = {split} 之后，它与真实曲线平均相差 {err}。',
				vars: (_rt, at) => {
					const cv = (at.state as { curves?: Tensor })?.curves;
					const ok = !!cv && cv.data.length === 4 * N;
					// the residual at step 4's time, worked out with the same weights (the probe trains this network; claims.test.ts)
					const params = (at.state as { params?: Record<string, Tensor> })?.params;
					const t = probe.defaults.at;
					const d = params ? withGradMode(true, () => derivs(netAt(params, (at.tape?.opt ?? {}) as { hidden?: number }), [t], false)) : null;
					const [u, d1, d2] = d ? [d.u.data[0], d.du.data[0], d.d2u.data[0]] : [NaN, NaN, NaN];
					const r = (d2 + C * d1 + K * u) / K;
					const verdict = pick(Math.abs(r) >= 0.05, { en: 'not 0, so the law is broken there', zh: '不是 0，定律在那里被打破了' }, { en: 'close to 0: there the curve nearly obeys the law', zh: '接近 0：曲线在那里几乎遵守定律' });
					const par = (v: number, digits = 3) => (Number.isFinite(v) ? (v < 0 ? `(${fmt3(v, digits)})` : fmt3(v, digits)) : NaN);
					return { t: setting(t), u, d1, d2, d1p: par(d1), up: par(u, 4), r, verdict: Number.isFinite(r) ? verdict : NaN, nT: N, c: setting(C), k: setting(K), phys: ok ? meanSq(cv!, 3) : NaN, split: setting(SPLIT), err: metric(at, 'err') };
				}
			},
			look: L('The strip under the curve: one cell per time. A cell in the background color means the law holds there; strong color means it is broken.', '看曲线下方的条带：每个时刻一格。和背景同色表示那里遵守定律，颜色鲜明表示违反了定律。'),
			terms: ['residual']
		},
		{
			id: 'loss',
			source: 'pinn',
			at: 'last',
			view: 'pinn/loss-bars',
			play: { from: 'first', to: 'last', phases: ['snapshot'], interval: 90 },
			provenance: 'live',
			title: L('A PINN: three parts of one loss', 'PINN：一个损失的三个部分'),
			narration: {
				en: 'A physics-informed network (PINN) is the same network with the law added to its loss. Its loss has three parts: data, the squared gaps at the points; physics, the physics loss of step 5 (the average squared residual at the {nT} times), times a weight λ = {lam}; and start, u′(0)² ÷ {k}, which is 0 only when the curve starts at rest, as the weight did. From the same starting weights as the plain network, {steps} steps take them from {d0}, {p0} and {s0} to {d1}, {p1} and {s1}.',
				zh: '物理信息神经网络（PINN）就是同一个网络，只是把定律加进了损失。它的损失有三部分：数据，即数据点上差距的平方；物理，即第 5 步的物理损失（{nT} 个时刻上残差平方的平均），再乘上一个权重 λ = {lam}；起点，即 u′(0)² ÷ {k}，只有曲线从静止开始时它才是 0，而重物正是从静止松开的。它和普通网络从同样的初始权重出发，训练 {steps} 步后，三部分从 {d0}、{p0} 和 {s0} 降到 {d1}、{p1} 和 {s1}。',
				vars: (_rt, at) => {
					const lam = (at.tape?.opt as { lam?: number })?.lam ?? pinn.defaults.lam;
					const s0: Cursor = { phase: 'snapshot', step: 0 };
					return { nT: N, lam: setting(lam), k: setting(K), steps: at.frame?.step ?? NaN, d0: z0(metricAt(at, s0, 'data')), p0: lam * metricAt(at, s0, 'physics'), s0: metricAt(at, s0, 'start'), d1: z0(metric(at, 'data')), p1: lam * metric(at, 'physics'), s1: metric(at, 'start') };
				}
			},
			look: { en: 'Play: the data bar falls first, while the start bar even grows for a while; in the end all three are far below where they started. Log scale: each tick is {tick} times the one before.', zh: '播放：数据条先变短，起点条反而会先变长一阵；最后三条都远低于起点时的长度。对数刻度：每一格是前一格的 {tick} 倍。', vars: () => ({ tick: 100 }) },
			terms: ['pinn', 'loss']
		},
		{
			id: 'side',
			source: 'pinn',
			at: 'last',
			view: fit(),
			compare: { a: 'plain', b: 'pinn', label: [L('Plain network: the points only', '普通网络：只用数据点'), L('PINN: the points and the law', 'PINN：数据点加定律')] },
			provenance: 'live',
			title: L('Plain network and PINN side by side', '普通网络和 PINN 并排对比'),
			narration: {
				en: 'Both networks trained for {steps} steps from the same start. {atPoints}: data loss {dPlain} for the plain network, {dPinn} for the PINN. The strips tell them apart: the PINN’s physics loss is {pPinn}, the plain network’s {pPlain}, {ratio} times larger.',
				zh: '两个网络从同一个起点各训练了 {steps} 步。{atPoints}：普通网络的数据损失是 {dPlain}，PINN 是 {dPinn}。残差条把它们区分开来：PINN 的物理损失是 {pPinn}，普通网络是 {pPlain}，大了 {ratio} 倍。',
				vars: (rt) => {
					const [a, b] = [snap(rt, 'plain'), snap(rt, 'pinn')];
					const [da, db] = [z0(a.m.data ?? NaN), z0(b.m.data ?? NaN)];
					const atPoints = !pointsOf(rt.input).n ? L('There are no points to fit', '没有数据点可以拟合') : pick(Math.max(da, db) < 0.01, { en: 'At the points both do well', zh: '在数据点上，两者都不错' }, { en: 'At the points they differ', zh: '在数据点上，两者表现不同' });
					return { atPoints, steps: b.m.step ?? NaN, dPlain: da, dPinn: db, pPlain: phys(a), pPinn: phys(b), ratio: times(phys(a) / phys(b)) };
				}
			},
			look: L('Compare the two strips: one stays in the background color, the other is full of color.', '比较两条残差条：一条几乎和背景同色，另一条到处是颜色。')
		},
		{
			id: 'after',
			source: 'pinn',
			at: 'last',
			view: fit({ strip: false, gap: true }),
			compare: { a: 'plain', b: 'pinn', label: [L('Plain network', '普通网络'), L('PINN', 'PINN')] },
			provenance: 'live',
			title: L('After the data', '数据之后'),
			narration: {
				en: 'After t = {split} there are no points, and the two networks part ways. There the plain network is {ePlain} from the true curve on average, the PINN {ePinn}: the plain network’s gap is {ratio} times the PINN’s. {carry} Predicting outside the data like this is called extrapolation.',
				zh: 't = {split} 之后没有数据点，两个网络就分道扬镳了。在那一段，普通网络与真实曲线平均相差 {ePlain}，PINN 只差 {ePinn}：普通网络的差距是 PINN 的 {ratio} 倍。{carry}像这样在数据范围之外做预测，叫作外推。',
				vars: (rt) => {
					const [a, b] = [snap(rt, 'plain').m.err ?? NaN, snap(rt, 'pinn').m.err ?? NaN];
					// with no points nothing sets the size of the swing, and the PINN settles on u = 0 (claims.test.ts)
					const carry = pointsOf(rt.input).n
						? L('The law holds at every time, so it carries the swing on where the data stops.', '定律在每个时刻都成立，所以在数据停下的地方，它把振动延续了下去。')
						: L('With no points, though, nothing sets the size of the swing: the PINN settles on u = 0, a flat line that obeys the law too.', '不过没有数据点时，没有什么能定下振动的大小：PINN 停在 u = 0 上，这条平线同样遵守定律。');
					return { split: setting(SPLIT), ePlain: a, ePinn: b, ratio: times(a / b), carry };
				}
			},
			look: { en: 'The shaded band between each network’s curve and the dashed true curve: its average height after t = {split} is the gap under each plot.', zh: '每个网络的曲线和虚线（真实曲线）之间的阴影带：它在 t = {split} 之后的平均高度，就是图下方写的差距。', vars: () => ({ split: setting(SPLIT) }) },
			terms: ['extrapolation']
		},
		{
			id: 'lambda',
			source: 'pinn',
			at: 'last',
			view: fit({ choices: [['lam0', '0'], ['lam001', '0.01'], ['pinn', '1'], ['lam100', '100']].map(([source, v]) => ({ source, label: L(`λ = ${v}`, `λ = ${v}`) })) }),
			provenance: 'live',
			title: L('How much should the law count?', '定律应该占多大分量？'),
			narration: {
				en: 'λ decides how much the law counts against the points. Your PINN uses λ = {lam} and ends {err} from the true curve after t = {split}, with a data loss of {data}. Each button above the plot trains a new PINN from the same start with another λ; that takes a few seconds.',
				zh: 'λ 决定了定律和数据点相比占多大分量。你的 PINN 用的是 λ = {lam}，训练完后在 t = {split} 之后与真实曲线平均相差 {err}，数据损失是 {data}。图上方的每个按钮都会用另一个 λ、从同一个起点训练一个新的 PINN，需要几秒钟。',
				vars: (_rt, at) => ({ lam: setting((at.tape?.opt as { lam?: number })?.lam ?? pinn.defaults.lam), err: metric(at, 'err'), split: setting(SPLIT), data: z0(metric(at, 'data')) })
			},
			predict: {
				id: 'pinn-lambda',
				question: L('Before you try: what will a PINN with λ = 100 do?', '先别急着试：λ = 100 的 PINN 会怎样？'),
				options: [
					L('Fit even better: more of the law can only help', '拟合得更好：定律占得越多只会越好'),
					L('Let go of the points: its curve goes nearly flat', '放开数据点：曲线几乎变成一条平线'),
					L('Wander off after the points, like the plain network', '像普通网络一样，在数据点之后乱走')
				],
				answer: 1,
				explain: L(
					'With λ = 100 the law outweighs the points. A flat line u = 0 obeys the law too (u, u′ and u″ are all 0), so the curve sinks towards it and misses the points. λ = 0 is the opposite mistake: the law is switched off and the PINN wanders like the plain network.',
					'λ = 100 时，定律压过了数据点。一条 u = 0 的平线同样遵守定律（u、u′ 和 u″ 全是 0），所以曲线向它靠拢，不再穿过数据点。λ = 0 是相反的错误：定律被关掉了，PINN 像普通网络一样乱走。'
				)
			},
			task: {
				prompt: L(
					'Now train λ = 0 and λ = 100 with the buttons above the plot. Which one wanders off after the points, and which one no longer passes through them? Read the numbers under the plot.',
					'现在用图上方的按钮训练 λ = 0 和 λ = 100。哪一个在数据点之后乱走，哪一个不再穿过数据点？看图下方的数字。'
				),
				done: (rt) => done(rt, 'lam0') && done(rt, 'lam100'),
				waiting: L('Not yet: train both λ = 0 and λ = 100 with the buttons above the plot.', '还没有：用图上方的按钮把 λ = 0 和 λ = 100 都训练一遍。'),
				hint: L('With λ = 0 the law is switched off. With λ = 100 the law outweighs the points, and a flat line also obeys the law.', 'λ = 0 时定律被关掉了。λ = 100 时定律压过了数据点，而一条平线也是遵守定律的。')
			},
			look: L('The curve, the strip and the numbers under the plot, for each λ.', '对每个 λ，看曲线、残差条和图下方的数字。')
		},
		{
			id: 'wrong-c',
			source: 'wrongC',
			at: 'last',
			view: fit(),
			compare: { a: 'wrongC', b: 'pinn', label: [L('Wrong law: c = 0', '错误的定律：c = 0'), L(`Right law: c = ${setting(C)}`, `正确的定律：c = ${setting(C)}`)] },
			provenance: 'live',
			kindOf: 'breakIt',
			title: L('A wrong law gives a confident wrong curve', '错误的定律给出自信的错误曲线'),
			narration: {
				en: 'A PINN is only as good as its law. Give it the wrong friction, c = {cw} instead of {c}, as if the spring lost no energy: its loss still falls from {l0} to {l1}, and its curve looks just as smooth and sure. {rest}',
				zh: 'PINN 的好坏取决于它的定律。给它一个错误的摩擦系数，c = {cw} 而不是 {c}，就好像弹簧不损失能量：它的损失照样从 {l0} 降到 {l1}，曲线看起来照样平滑、有把握。{rest}',
				vars: (rt, at) => {
					const right = snap(rt, 'pinn').m;
					const [e, er, lr] = [metric(at, 'err'), right.err ?? NaN, right.loss ?? NaN];
					// with no points both laws settle on the same curve (claims.test.ts); the comparison needs data
					const rest = pointsOf(rt.input).n
						? {
								en: `Yet after t = ${SPLIT} it is ${fmt3(e)} from the true curve on average, ${ratioText(e / er)} times the right law’s ${fmt3(er)}. Its loss is higher than the right law’s ${fmt3(lr)}, but a real experiment has no right-law run to compare with.`,
								zh: `可是在 t = ${SPLIT} 之后，它与真实曲线平均相差 ${fmt3(e)}，是正确定律（${fmt3(er)}）的 ${ratioText(e / er)} 倍。它的损失确实比正确定律的 ${fmt3(lr)} 高，但真实的实验里并没有一个用正确定律训练的网络可以拿来比较。`
							}
						: L('With no points, though, both laws end on the same curve: add points to see them part.', '不过没有数据点时，两种定律最后得到同一条曲线：添加一些点，才能看到它们分开。');
					return { cw: setting((at.tape?.opt as { c?: number })?.c ?? pinnWrongC.defaults.c), c: setting(C), l0: metricAt(at, { phase: 'snapshot', step: 0 }, 'loss'), l1: metric(at, 'loss'), rest: Number.isFinite(e / er) && Number.isFinite(lr) ? rest : NaN };
				}
			},
			look: {
				en: 'After t = {split}: does each swing die down? The wrong law’s strip checks its curve against its own law, c = 0, and stays pale, far paler than the plain network’s in step 5: the curve obeys the wrong law.',
				zh: '看 t = {split} 之后：每个摆动会不会减弱？错误定律那一栏的残差条用它自己的定律（c = 0）来检查，颜色很浅，比第 5 步里普通网络的浅得多：这条曲线遵守的是错误的定律。',
				vars: () => ({ split: setting(SPLIT) })
			}
		}
	],
	fixtures: [{ id: 'outlier', label: L('One point far off the curve', '一个远离曲线的点'), raw: { preset: { name: 'oscillator', seed: 1 }, points: [{ t: 0.02, y: 1 }, { t: 0.11, y: 0.3 }, { t: 0.2, y: -0.45 }, { t: 0.25, y: 0.4 }, { t: 0.3, y: -0.58 }, { t: 0.39, y: -0.12 }] } }]
};

export default lesson;
