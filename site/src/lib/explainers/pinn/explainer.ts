// pinn: the manifest. A physics-informed neural network (PINN) for a damped oscillator, trained live in the browser
// (kind B), and the slope of a network checked by finite differences (kind C). Every source has seed 1, so every
// network starts from the same weights (rng.fork('init')).
//   pinn            trainer 'pinn': points + λ·law + start, λ = 1, 1,000 steps of Adam (lr 0.01)
//   plain           trainer 'plain': the same network on the points only
//   probe           process 'probe': the plain network's u′ and u″ at t = 0.7, by autodiff and by finite differences
//   lam0, lam001, lam100   the PINN with λ = 0, 0.01 and 100 (the λ task; trainers pinnLam0, pinnLam001, and pinn with lam 100)
//   wrongC          the PINN with a wrong law: c = 0, no friction (the breakIt step; trainer pinnWrongC)
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';

const L = (en: string, zh: string) => ({ en, zh });
const train = (trainer: string, opt: Record<string, unknown> = {}) => ({ kind: 'train' as const, trainer, opt: { seed: 1, ...opt } });
const START = seriesPresetRaw('oscillator');

const manifest: ExplainerManifest = {
	slug: 'pinn',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/pinn', auto: ['facts.json'] },
	input: { adapter: 'series', presets: 'presets.json', start: START, options: { mode: 'points', yRange: [-1.2, 1.4] } },
	sources: {
		pinn: train('pinn'),
		plain: train('plain'),
		probe: { kind: 'process', process: 'probe', run: 'worker', seed: 1 },
		lam0: train('pinnLam0'),
		lam001: train('pinnLam001'),
		lam100: train('pinn', { lam: 100 }),
		wrongC: train('pinnWrongC')
	},
	page: {
		goals: [
			L('Say what the residual u″ + c·u′ + k·u measures, and read from the residual strip where a curve breaks the law.', '说出残差 u″ + c·u′ + k·u 衡量的是什么，并从残差条上看出曲线在哪里违反了定律。'),
			L('Explain how a network gets u′ and u″ by autodiff, and check one against a finite difference.', '解释网络如何用自动微分得到 u′ 和 u″，并用有限差分核对其中一个。'),
			L('Compare a plain network and a PINN after the last data point, with numbers, and say what each part of the PINN loss pushes for.', '用数字比较普通网络和 PINN 在最后一个数据点之后的表现，并说出 PINN 损失的每一部分在推动什么。'),
			L('Predict what λ = 0, a very large λ and a wrong friction constant c do to the fit.', '预测 λ = 0、很大的 λ 和错误的摩擦系数 c 会对拟合产生什么影响。')
		],
		prequestion: {
			id: 'pinn-pre',
			question: L(
				'You know the law of a swinging spring exactly, and you have eight measured positions, all from the first 40% of the time. Can a network trained on them tell where the weight is at the very end?',
				'你确切知道一个弹簧振动的定律，还有八个测得的位置，全都来自前 40% 的时间。用它们训练的网络能说出重物在最后时刻的位置吗？'
			),
			options: [L('Yes, from the eight points alone', '能，只靠这八个点就行'), L('Only if it also has to obey the law', '只有它同时必须遵守定律才行'), L('No, that is impossible', '不能，这不可能')],
			answer: 1,
			explain: L(
				'The points say nothing about later times, so a network trained on them alone is free to do anything there. A network that must also obey the law at every time carries the swing forward: that is a physics-informed network.',
				'这些点对后面的时间什么也没说，所以只用它们训练的网络在那里可以随便乱走。如果网络还必须在每个时刻都遵守定律，它就能把振动延续下去：这就是物理信息神经网络。'
			)
		},
		tryThis: [
			{
				id: 'outlier',
				title: L('One bad measurement', '一个测坏了的点'),
				setup: { input: { ...START, points: [{ t: 0.02, y: 1 }, { t: 0.11, y: 0.3 }, { t: 0.2, y: -0.45 }, { t: 0.25, y: 0.4 }, { t: 0.3, y: -0.58 }, { t: 0.39, y: -0.12 }] } },
				predict: L('The point at t = 0.25 is far off the curve. Which network bends to reach it: the plain one or the PINN?', 't = 0.25 处的点远离曲线。哪个网络会弯过去够它：普通网络还是 PINN？'),
				look: L('Compare the two curves near t = 0.25, and the data bars of the two loss panels.', '比较两条曲线在 t = 0.25 附近的样子，以及两个损失面板里的数据条。'),
				goto: { step: 'side' }
			}
		],
		quiz: [
			{
				q: L('In the law u″ + c·u′ + k·u = 0, what is u″?', '在定律 u″ + c·u′ + k·u = 0 中，u″ 是什么？'),
				options: [L('The acceleration: how fast the velocity changes', '加速度：速度变化得有多快'), L('The velocity: how fast the position changes', '速度：位置变化得有多快'), L('The position at time t', '时刻 t 的位置')],
				answer: 0,
				explain: L('u′ is the velocity (the slope of u) and u″ the acceleration (the slope of u′, how much the curve bends).', 'u′ 是速度（u 的斜率），u″ 是加速度（u′ 的斜率，也就是曲线弯得有多厉害）。')
			}
		],
		misconception: {
			claim: L('A small loss means the PINN found the true curve.', '损失小，就说明 PINN 找到了真实曲线。'),
			truth: L('A PINN trusts its law. With the wrong friction its loss still falls far below where it started and its curve looks just as sure, yet after the data it is far from the truth.', 'PINN 信任它的定律。摩擦系数错了，它的损失照样比起点低得多，曲线看起来照样很有把握，但在数据之后却离真实情况很远。'),
			step: 'wrong-c'
		},
		lessonNote: L(
			'Ten steps: the law, the points, a plain network, slopes by autodiff, the residual, the three parts of the PINN loss, both networks side by side, after the data, a λ to choose and a wrong law. Everything trains live in your browser.',
			'十步：定律、数据点、普通网络、用自动微分求斜率、残差、PINN 损失的三个部分、两个网络并排对比、数据之后、选一个 λ，以及一个错误的定律。全部在你的浏览器里实时训练。'
		),
		sections: { titles: { training: L('Train the PINN yourself', '自己训练 PINN') } }
	},
	glossary: [
		{ id: 'derivative', term: L('derivative (u′, u″)', '导数（u′、u″）'), def: L('How fast a quantity changes: u′ is the slope of u (here the velocity), u″ the slope of u′ (the acceleration).', '一个量变化得有多快：u′ 是 u 的斜率（这里是速度），u″ 是 u′ 的斜率（加速度）。') },
		{ id: 'autodiff', term: L('automatic differentiation (autodiff)', '自动微分'), def: L('Computing exact derivatives of a program by the chain rule, step by step through its operations; backpropagation is one use of it.', '沿着程序的每一步运算用链式法则算出精确的导数；反向传播就是它的一种用法。'), related: ['backpropagation'] },
		{ id: 'chain-rule', term: L('chain rule', '链式法则'), def: L('How to get the slope of a function of a function: multiply the slopes of the steps. If u = f(g(t)), then u′ = f′(g(t)) · g′(t).', '求“函数套函数”的斜率的方法：把每一步的斜率乘起来。若 u = f(g(t))，则 u′ = f′(g(t)) · g′(t)。'), related: ['autodiff', 'backpropagation'] },
		{ id: 'finite-difference', term: L('finite difference', '有限差分'), def: L('An estimate of a slope from two nearby values: (u(t + h) − u(t − h)) / 2h for a small step h.', '用两个相近的值估计斜率：取一个小步长 h，算 (u(t + h) − u(t − h)) / 2h。'), related: ['derivative'] },
		{ id: 'residual', term: L('residual', '残差'), def: L('What is left when you put a curve into an equation that should give 0: here u″ + c·u′ + k·u. Zero means the curve obeys the law there.', '把曲线代入一个本应等于 0 的方程后剩下的部分：这里是 u″ + c·u′ + k·u。为 0 表示曲线在那里遵守定律。'), related: ['derivative'] },
		{ id: 'pinn', term: L('physics-informed neural network (PINN)', '物理信息神经网络（PINN）'), def: L('A network trained on data and, at the same time, on how badly its output breaks a known law (the residual).', '一种网络：训练时既看数据，也看它的输出违反已知定律的程度（残差）。'), related: ['residual', 'loss'] },
		{ id: 'extrapolation', term: L('extrapolation', '外推'), def: L('Predicting outside the range the data covers, such as after the last measured time.', '在数据覆盖的范围之外做预测，例如在最后一个测量时刻之后。') }
	],
	catalog: { runtime: L('Own autograd core in a worker (second derivatives by autodiff)', '自研自动微分内核，在 Web Worker 中运行（用自动微分求二阶导数）'), params: L('1 → 32 → 32 → 1 (1,153 parameters)', '1 → 32 → 32 → 1（1,153 个参数）') },
	// ideas only: the PINN loss (data + residual) and the damped-oscillator demo; no code, text or figures copied
	credits: [
		{
			name: 'Raissi, Perdikaris & Karniadakis, Physics-informed neural networks (J. Comput. Phys. 378, 2019)',
			url: 'https://doi.org/10.1016/j.jcp.2018.10.045',
			license: 'Journal article; ideas only (no text, figures or code used)',
			usedFor: 'The idea of a physics-informed loss: a data term plus the residual of the governing equation at collocation points, with derivatives by automatic differentiation',
			attribution: 'Maziar Raissi, Paris Perdikaris and George Em Karniadakis'
		},
		{
			name: 'Ben Moseley, harmonic-oscillator-pinn',
			url: 'https://github.com/benmoseley/harmonic-oscillator-pinn',
			license: 'MIT; ideas only (no code copied)',
			usedFor: 'The teaching setup: a damped harmonic oscillator, a plain network trained on early points against a PINN that extrapolates',
			attribution: 'Ben Moseley'
		}
	]
};

export default manifest;
