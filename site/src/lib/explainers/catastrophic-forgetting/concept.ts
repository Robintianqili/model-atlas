// catastrophic-forgetting: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { metricsOf, remedyName, settingsOf } from './data.ts';
import { REMEDIES, paramsOf } from './process.ts';
import { metricsAt, plane, settingsText } from './lesson.ts';

const join = (xs: I18n[], none: I18n): I18n => (xs.length ? { en: xs.map((x) => x.en).join(', '), zh: xs.map((x) => x.zh).join('、') } : none);

export const hook: I18n = {
	en: 'Fine-tune a network on something new and it can quietly lose what it knew before. Here a small network learns to sort one group of points, then a second group, and you can watch the first skill disappear, and then try to save it.',
	zh: '在新东西上微调一个网络，它可能会悄悄丢掉原来会的本领。这里一个小网络先学会给一组点分类，再学第二组；你可以看着第一项本领消失，然后想办法把它保住。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your settings, the whole run', zh: '你的设置，完整的一次运行' },
	source: 'mine',
	at: 'last',
	view: plane,
	play: { from: 'first', to: 'last', interval: 200, unit: { en: 'update', zh: '次更新' } },
	provenance: 'live',
	narration: {
		en: 'Your settings: {set}. After {nA} updates on task A the network gets A {a0|pct} right; after {nB} more on task B it gets A {accA|pct} and B {accB|pct} right. Change any setting and the run starts again.',
		zh: '你的设置：{set}。在任务 A 上更新 {nA} 次之后，网络在 A 上答对 {a0|pct}；再在任务 B 上更新 {nB} 次之后，它在 A 上答对 {accA|pct}，在 B 上答对 {accB|pct}。改动任何一个设置，运行都会重新开始。',
		vars: (rt, at) => {
			const m = metricsOf(at);
			return { set: settingsText(settingsOf(rt.input)), nA: m.stepsA, nB: m.stepsB, a0: metricsAt(at, { phase: 'switch' }).accA, accA: m.accA, accB: m.accB };
		}
	},
	look: { en: 'Play the run and watch the boundary in both groups: task A at negative x, task B at positive x.', zh: '播放这次运行，看两组点里的边界：任务 A 在 x 为负的一侧，任务 B 在 x 为正的一侧。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'A much wider network forgets just the same', zh: '宽得多的网络也一样会遗忘' },
	source: 'big',
	at: 'last',
	view: plane,
	provenance: 'live',
	narration: {
		en: 'The same two tasks with {big} weights and biases instead of {small}. After task A it gets A {a0|pct} right; after {nB} updates on task B it gets A {accA|pct} and B {accB|pct} right. More weights alone give B no reason to keep A\'s answers.',
		zh: '同样的两个任务，网络有 {big} 个权重和偏置，而不是 {small} 个。学完任务 A 后，它在 A 上答对 {a0|pct}；在任务 B 上更新 {nB} 次之后，它在 A 上答对 {accA|pct}，在 B 上答对 {accB|pct}。仅仅增加权重，并不会让 B 有理由保住 A 的答案。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { big: paramsOf(48), small: paramsOf(16), a0: metricsAt(at, { phase: 'switch' }).accA, nB: m.stepsB, accA: m.accA, accB: m.accB };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: "Task A's accuracy under each remedy", zh: '每种办法下任务 A 的准确率' },
	source: 'remedies',
	at: 'last',
	view: {
		key: 'loss-curve',
		props: () => ({
			metrics: REMEDIES.map((r) => `accA_${r.id}`),
			labels: Object.fromEntries(REMEDIES.map((r) => [`accA_${r.id}`, remedyName(r.set)]))
		})
	},
	provenance: 'live',
	narration: {
		en: 'One line per way of training on B, all from the same network after task A. After {n} updates on B, task A is still above one half with: {keep}. It fell below one half with: {lost}.',
		zh: '每条线是一种在 B 上训练的方式，起点都是学完任务 A 的同一个网络。在 B 上更新 {n} 次之后，任务 A 仍高于一半的有：{keep}。降到一半以下的有：{lost}。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			const none = { en: 'none', zh: '无' };
			return {
				n: m.step,
				keep: join(REMEDIES.filter((r) => m[`accA_${r.id}`] >= 0.5).map((r) => remedyName(r.set)), none),
				lost: join(REMEDIES.filter((r) => m[`accA_${r.id}`] < 0.5).map((r) => remedyName(r.set)), none)
			};
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'A smaller learning rate prevents forgetting.', zh: '学习率小一点就不会遗忘。' },
		truth: {
			en: 'It only slows it down. With nothing in the loss that asks for the old answers, a slower run reaches the same place later, and it learns the new task more slowly too.',
			zh: '它只会让遗忘变慢。只要损失里没有要求旧答案的项，慢一点的运行也会晚些到达同样的地方，而且新任务也学得更慢。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'Freezing the early layers keeps the old skill.', zh: '冻结前面的层就能保住旧本领。' },
		truth: {
			en: 'It keeps the old features, not the old decisions. Here the later layers still turn task A\'s answers over.',
			zh: '它保住的是旧的特征，而不是旧的判断。在这里，后面的层仍然会把任务 A 的答案翻过来。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'Replay is a free fix.', zh: '回放是没有代价的补救办法。' },
		truth: {
			en: 'It needs some of the old data kept around, which may be large or private, and every training batch grows by the stored points.',
			zh: '它需要保留一部分旧数据，而这些数据可能很大，也可能涉及隐私；而且每一批训练数据都会因存下的点而变大。'
		},
		step: 'replay'
	}
];

export const further = [
	{ label: { en: 'Kirkpatrick et al. (2017): Overcoming catastrophic forgetting in neural networks (EWC)', zh: 'Kirkpatrick 等（2017）：克服神经网络中的灾难性遗忘（EWC）' }, href: 'https://arxiv.org/abs/1612.00796' },
	{ label: { en: 'Parisi et al. (2019): Continual lifelong learning with neural networks, a review', zh: 'Parisi 等（2019）：神经网络的持续终身学习综述' }, href: 'https://arxiv.org/abs/1802.07569' },
	{ label: { en: 'Transfer learning on this site: reuse a trained network for a new task', zh: '本站的迁移学习：把训练好的网络用于新任务' }, href: '/train/transfer-learning/' },
	{ label: { en: 'The MLP playground: train a 2-D network yourself', zh: 'MLP 游乐场：自己训练一个二维网络' }, href: '/m/mlp-playground/' }
];
