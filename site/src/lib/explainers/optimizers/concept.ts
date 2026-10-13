// optimizers: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef). hook
// (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { RACERS, distinct, metricsOf, setting, setupOf, surfaceOf } from './data.ts';
import { MOONS, PARAMS } from './net.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });

export const hook: I18n = L(
	'Most networks on this site learned with the same loop: compute the gradient, then take a step. The optimizer is the rule for that step, and the learning rate is how big it is. Here four common optimizers race on landscapes of just two weights, so you can watch every step, and then the learning rate itself gets tuned on a real network.',
	'这个网站上的大多数网络都是用同一个循环训练出来的：先求梯度，再走一步。优化器就是决定这一步怎么走的规则，学习率决定这一步有多大。这里四个常用的优化器在只有两个权重的地形上赛跑，你能看清每一步；然后在一个真实的网络上调学习率本身。'
);

export const toy: StepDef = {
	id: 'toy',
	title: L('Race on your landscape', '在你的地形上比赛'),
	source: 'race',
	at: 'last',
	view: { key: 'optimizers/surface', props: () => ({ glyph: 'adam' }) },
	play: { from: 'first', to: 'last', interval: 120, unit: L('update', '次更新') },
	provenance: 'live',
	narration: {
		en: 'On the {land} from x = {x}, y = {y}, after {n} updates: SGD {sgd}, momentum {mom}, Adam {adam}, AdamW {adamw}. Change the landscape or the start with the controls in this section and the race runs again; the squares show Adam’s √v̂ for each weight.',
		zh: '在{land}上，从 x = {x}、y = {y} 出发，更新 {n} 次之后：SGD {sgd}，动量法 {mom}，Adam {adam}，AdamW {adamw}。用这一部分的控件换一个地形或起点，比赛会重新开始；方块显示 Adam 每个权重的 √v̂。',
		vars: (rt, at) => {
			const m = metricsOf(at);
			const s = setupOf(rt.input);
			const [sgd, mom, adam, adamw] = distinct(RACERS.map((r) => m[`${r.key}Loss`]));
			return { land: surfaceOf(s.surface).name, x: s.x, y: s.y, n: m.step, sgd, mom, adam, adamw };
		}
	},
	look: L('Press Play to watch the four paths grow one update at a time.', '按播放，看四条路径一次更新一次更新地延伸。')
};

export const real: StepDef = {
	id: 'real',
	title: L('A range test on a real network', '在真实网络上做区间测试'),
	source: 'range',
	at: 'last',
	view: 'optimizers/range',
	provenance: 'live',
	narration: {
		en: 'The network has {params} weights and sorts {pts} points of two interleaved half-moons. In one run of {steps} steps η grows from {lo} to {hi}: the loss starts at {start}, is lowest, {low}, at η = {lowLr}, and is {end} after the last step.',
		zh: '这个网络有 {params} 个权重，给两个交错的半月形里的 {pts} 个点分类。在一次 {steps} 步的运行中，η 从 {lo} 增大到 {hi}：损失从 {start} 开始，在 η = {lowLr} 时最低（{low}），最后一步之后是 {end}。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			const f0 = at.tape?.frame(0).metrics ?? {};
			return { params: PARAMS, pts: MOONS.n, steps: m.step, lo: setting(f0.lr), hi: setting(at.tape?.frame(at.tape.length - 2).metrics?.lr), start: m.startLoss, low: m.lowLoss, lowLr: m.lowLr, end: m.endLoss };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: L('Three landscapes, three winners', '三个地形，三个赢家'),
	source: 'league',
	at: 'last',
	view: 'optimizers/league',
	provenance: 'live',
	narration: {
		en: 'Each of the {k} optimizers ran {n} updates on each landscape from that landscape’s own start. The table holds every final loss; read along a row to see which rule suited that landscape.',
		zh: '{k} 个优化器在每个地形上都从该地形自己的起点出发，各更新 {n} 次。表里是每一个最终损失；沿着一行读，就能看出哪种规则适合那个地形。',
		vars: (_rt, at) => ({ k: RACERS.length, n: metricsOf(at).step })
	}
};

export const pitfalls = [
	{
		claim: L('Momentum just makes the steps bigger.', '动量法只是把步子变大。'),
		truth: L(
			'It adds up past gradients. Where they agree, along the valley, the steps grow; where they flip sign, across it, they partly cancel. The velocity also carries the weights past the floor, so momentum swings across the valley before it settles.',
			'它把过去的梯度累加起来。梯度方向一致的地方（沿着山谷），步子变大；梯度来回变号的地方（横跨山谷），它们部分抵消。速度还会带着权重越过谷底，所以动量法在稳定下来之前会在山谷两侧来回摆动。'
		),
		step: 'momentum-update'
	},
	{
		claim: L('Warm-up is a detail you can leave out.', '预热是个可以省掉的细节。'),
		truth: L(
			'On this page’s network, at the schedule demo’s peak rate, every schedule without warm-up ends no better than a coin-flip guess, and the same schedules with warm-up train.',
			'在本页的网络上，以调度演示的峰值学习率训练，所有不带预热的调度最后都不比抛硬币瞎猜好，而同样的调度加上预热都能训练成功。'
		),
		step: 'warmup'
	},
	{
		claim: L('Gradient clipping changes where the update points.', '梯度裁剪会改变更新的方向。'),
		truth: L(
			'Clipping by norm only shortens a gradient that is too long; its direction stays. It stops one huge step from throwing the weights off a cliff, but it does not choose a good step size: the clipped run on this page still bounces across the narrow floor at the end.',
			'按范数裁剪只会缩短过长的梯度，方向不变。它能阻止一个巨大的步子把权重抛下悬崖，但不会替你选一个好的步长：本页里裁剪过的那次运行，最后仍在狭窄的谷底两侧来回弹跳。'
		),
		step: 'clip'
	}
];

export const further = [
	{ label: L('Why Momentum Really Works (Gabriel Goh, Distill)', '《Why Momentum Really Works》（Gabriel Goh，Distill）'), href: 'https://distill.pub/2017/momentum/' },
	{ label: L('Dive into Deep Learning: the optimization chapter (free online book)', '《动手学深度学习》：优化算法一章（免费在线书）'), href: 'https://d2l.ai/chapter_optimization/index.html' },
	{ label: L('Adam: A Method for Stochastic Optimization (Kingma and Ba)', '《Adam: A Method for Stochastic Optimization》（Kingma 与 Ba）'), href: 'https://arxiv.org/abs/1412.6980' },
	{ label: L('Cyclical Learning Rates, the range test (Leslie Smith)', '《Cyclical Learning Rates》，区间测试（Leslie Smith）'), href: 'https://arxiv.org/abs/1506.01186' },
	{ label: L('PyTorch: torch.optim and its learning-rate schedulers', 'PyTorch：torch.optim 及其学习率调度器'), href: 'https://pytorch.org/docs/stable/optim.html' }
];
