// catastrophic-forgetting: the manifest of a CONCEPT package (K2), routed at /train/catastrophic-forgetting/ by the
// concept route (AutoConceptPage: goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links).
// A 2-16-16-2 network learns task A (the left group of points), then task B (the right group, the classes the other way
// round), and forgets A. The learner's input is how it trains on B: replay, a pull toward the old weights (EWC-lite), a
// frozen first layer, a smaller learning rate. Sources (all process 'sequence' or 'remedies', process.ts):
//   plain     A then B with no protection (ignores the input: the forgetting the first steps narrate)
//   mine      A then B with the learner's settings
//   pullRun   A then B with a pull of λ = 0.6 (ignores the input: the worked update of the pull step)
//   remedies  six ways to train on B, in lockstep, from the same network after A (ignores the input)
//   big       A then B, plain, with 48 units per hidden layer (ignores the input: the concept page's "real" panel)
// Every number is the kit's autograd and plain gradient descent in the learner's browser ("Live").
import type { ExplainerManifest } from '../../atlas/types/index.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const fixed = () => null;

const manifest: ConceptManifest = {
	slug: 'catastrophic-forgetting',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'settings', start: { replay: 0, pull: 0, freeze: false, slow: false } },
	sources: {
		// no Python reference: the frames are the kit's autograd (tested against PyTorch fp64 in the kit's own tests) and
		// plain gradient descent written out in process.ts, so the badge says "Live"
		plain: { kind: 'process', process: 'sequence', input: fixed, reference: false },
		mine: { kind: 'process', process: 'sequence', reference: false },
		pullRun: { kind: 'process', process: 'sequence', input: fixed, opt: { pull: 0.6 }, reference: false },
		remedies: { kind: 'process', process: 'remedies', input: fixed, reference: false },
		big: { kind: 'process', process: 'sequence', input: fixed, opt: { hidden: 48 }, reference: false }
	},
	page: {
		goals: [
			{
				en: 'Watch a network that has learned task A lose it while it learns task B, and say why: the two tasks share every weight, and a step down B\'s slope can be a step up A\'s.',
				zh: '看一个学会了任务 A 的网络在学任务 B 时把 A 忘掉，并说出原因：两个任务共用所有权重，沿 B 的斜率下坡的一步，可能正是 A 的上坡。'
			},
			{
				en: 'Predict, then check on live runs, what replay, a pull toward the old weights, a frozen first layer and a smaller learning rate do to both tasks.',
				zh: '先预测，再用实时运行来验证：回放、拉向旧权重、冻结第一层和更小的学习率，分别会让两个任务怎样。'
			},
			{
				en: 'Explain the trade-off between keeping the old task and learning the new one with the numbers of a pull that is too weak and one that is too strong.',
				zh: '用拉力太弱和太强时的数字，解释“保住旧任务”与“学会新任务”之间的权衡。'
			}
		],
		prequestion: {
			id: 'catastrophic-forgetting-pre',
			question: {
				en: 'A network sorts every point of task A correctly. Then it trains only on task B, whose points lie somewhere else. What happens to its answers on task A?',
				zh: '一个网络把任务 A 的每个点都分对了。然后它只在任务 B 上训练，而 B 的点在别的地方。它在任务 A 上的回答会怎样？'
			},
			options: [{ en: 'Nothing: A\'s points are not in B\'s data, so they are left alone', zh: '不变：A 的点不在 B 的数据里，所以不受影响' }, { en: 'They can get much worse, even all wrong', zh: '可能变差很多，甚至全错' }, { en: 'They get better, because the network keeps learning', zh: '变得更好，因为网络一直在学习' }],
			answer: 1,
			explain: {
				en: 'Every weight serves both tasks. Training on B moves the weights wherever B\'s loss falls fastest, and nothing in that loss asks to keep A\'s answers. In this lesson task A drops from all right to all wrong.',
				zh: '每个权重同时为两个任务服务。在 B 上训练会把权重推向 B 的损失下降最快的方向，而这个损失里没有任何一项要求保住 A 的答案。在这节课里，任务 A 从全对掉到全错。'
			}
		},
		quiz: [
			{
				q: { en: 'Which of the four remedies needs you to keep some of the old task\'s data?', zh: '四种补救办法中，哪一种需要保留一部分旧任务的数据？' },
				options: [
					{ en: 'Replay', zh: '回放' },
					{ en: 'The pull toward the old weights', zh: '拉向旧权重' },
					{ en: 'Freezing the first layer', zh: '冻结第一层' },
					{ en: 'A learning rate ten times smaller', zh: '把学习率缩小到十分之一' }
				],
				answer: 0,
				explain: {
					en: 'Replay trains on stored task-A points together with task B. The pull only needs the old weights; freezing and a smaller learning rate need nothing at all.',
					zh: '回放要把存下的任务 A 的点和任务 B 一起训练。拉力只需要旧的权重；冻结和更小的学习率则什么都不需要。'
				}
			},
			{
				q: { en: 'A pull toward the old weights is made so strong that task A stays right. What happens to task B?', zh: '把拉向旧权重的拉力调到足够强，让任务 A 保持正确。任务 B 会怎样？' },
				options: [
					{ en: 'It is learned just as well as without the pull', zh: '学得和没有拉力时一样好' },
					{ en: 'It is barely learned: the weights cannot move far enough', zh: '几乎学不会：权重没法移动到足够远' },
					{ en: 'It is learned faster', zh: '学得更快' }
				],
				answer: 1,
				explain: {
					en: 'Here, learning B needs large weight changes. A pull strong enough to keep A also holds back those changes, so B stays near chance or below.',
					zh: '在这里，学会 B 需要大幅改变权重。强到能保住 A 的拉力也拦住了这些改变，所以 B 停在随机猜测附近，甚至更差。'
				}
			}
		],
		misconception: {
			claim: { en: 'Fine-tuning on new data only adds to what a network knows.', zh: '在新数据上微调，只会给网络增加知识。' },
			truth: {
				en: 'The same weights hold the old skill and learn the new one. Each update for the new task also rewrites the old answers, and nothing in the new task\'s loss asks to keep them.',
				zh: '旧本领和新任务用的是同一组权重。为新任务做的每次更新也会改写旧的答案，而新任务的损失里没有任何一项要求保住它们。'
			},
			step: 'forget'
		},
		lessonNote: {
			en: 'Nine steps: learn task A, predict what task B does to it, watch it go, see why, then try four remedies and their price.',
			zh: '九步：学会任务 A，预测任务 B 会对它做什么，看它被忘掉，弄清原因，再试四种补救办法和它们的代价。'
		}
	},
	glossary: [
		{
			id: 'catastrophic-forgetting',
			term: { en: 'catastrophic forgetting', zh: '灾难性遗忘' },
			def: {
				en: 'When training a network on a new task wipes out what it had learned on an earlier one, because the same weights serve both.',
				zh: '在新任务上训练网络时，它在先前任务上学到的东西被抹掉了，因为两个任务用的是同一组权重。'
			},
			related: ['fine-tuning', 'replay', 'weight-pull']
		},
		{
			id: 'replay',
			term: { en: 'replay (rehearsal)', zh: '回放（复习）' },
			def: {
				en: 'Mixing stored examples of earlier tasks into the training batches of a new task, so the loss keeps asking for the old answers.',
				zh: '把先前任务存下的样本混进新任务的训练批次，让损失继续考查旧任务的答案。'
			},
			related: ['catastrophic-forgetting', 'batch']
		},
		{
			id: 'weight-pull',
			term: { en: 'pull toward the old weights (L2 to the old weights, EWC)', zh: '拉向旧权重（到旧权重的 L2、EWC）' },
			def: {
				en: 'An extra loss term λ/2 · Σ (w − w_old)² that pulls each weight back toward its value after the old task. Elastic weight consolidation (EWC) weighs each term by how much the old task needs that weight.',
				zh: '一个额外的损失项 λ/2 · Σ (w − w_old)²，把每个权重拉回旧任务结束时的值。弹性权重巩固（EWC）还按旧任务对每个权重的依赖程度给每一项加权。'
			},
			related: ['catastrophic-forgetting', 'regularization']
		}
	],
	credits: [
		{
			name: 'McCloskey and Cohen, Catastrophic Interference in Connectionist Networks (1989)',
			url: 'https://doi.org/10.1016/S0079-7421(08)60536-8',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the phenomenon the page shows: training on a new task interferes with an old one',
			attribution: 'Michael McCloskey, Neal J. Cohen'
		},
		{
			name: 'Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks (PNAS 2017)',
			url: 'https://arxiv.org/abs/1612.00796',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the pull toward the old weights (here with the same weight for every parameter, a simplified EWC)',
			attribution: 'James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, et al.'
		},
		{
			name: 'Robins, Catastrophic Forgetting, Rehearsal and Pseudorehearsal (Connection Science, 1995)',
			url: 'https://doi.org/10.1080/09540099550039318',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'replay (rehearsal): training on stored old examples together with the new task',
			attribution: 'Anthony Robins'
		}
	]
};

export default manifest;
