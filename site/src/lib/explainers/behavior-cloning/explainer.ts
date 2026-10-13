// behavior-cloning: the package manifest. Kinds B and C: a policy network learns state -> move from the learner's
// demonstrations by supervised learning, live in the package worker (trainer.ts), then drives on its own in a windy
// grid (process.ts, checked frame by frame against pipeline/behavior-cloning/reference.py). Every train source has seed
// 1, so they start from the same fresh network; the rollouts all meet the same gusts (seed EVAL.seed).
//   train   behavior cloning: 100 full-batch Adam updates on your demonstrations (the twin's parity source)
//   dagger  the same run, then 4 DAgger rounds of 25 updates: the policy drives, the expert labels the new cells
//   roll    the cloned policy (train's last weights) drives 20 windy episodes; the expert drives the same 20
//   droll   the same 20 episodes with the DAgger policy
import type { ExplainerManifest, ExplainerRuntime, Tape } from '../../atlas/types/index.ts';
import { DEFAULTS, EVAL } from './world.ts';
import { bc, DAGGER, datasetOf } from './trainer.ts';

const L = (en: string, zh: string) => ({ en, zh });
const H = bc.defaults.hidden;
/** The roll input of a finished train source: the learner's input, its last weights and its dataset's cells. */
export function rollInput(rt: ExplainerRuntime, src: 'train' | 'dagger') {
	const tape = rt.sources[src].tape as Tape | null, params = tape ? (tape.state(tape.length - 1) as { params: Record<string, unknown> }).params : {};
	return { raw: rt.input, params, cells: datasetOf(rt.input, tape, src === 'dagger' ? DAGGER : {}).s };
}

const manifest: ExplainerManifest = {
	slug: 'behavior-cloning',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'demos', start: DEFAULTS, presetsFromAdapter: true },
	sources: {
		train: { kind: 'train', trainer: 'bc', detail: 'coarse', opt: { seed: 1 } },
		dagger: { kind: 'train', trainer: 'bc', detail: 'coarse', opt: DAGGER },
		roll: { kind: 'process', process: 'roll', seed: EVAL.seed, after: 'train', input: (rt) => rollInput(rt, 'train') },
		droll: { kind: 'process', process: 'roll', seed: EVAL.seed, after: 'dagger', input: (rt) => rollInput(rt, 'dagger') }
	},
	page: {
		goals: [
			L('Turn demonstrations into a dataset of (state, move) pairs and say what the policy network learns from them.', '把示范变成（状态，动作）对的数据集，并说出策略网络从中学到了什么。'),
			L('Explain with a run you watched why a cloned policy that copies every demonstrated move can still fail: one gust, an unseen cell, then mistake after mistake.', '结合你看到的一次运行，解释为什么一个把每个示范动作都学会了的克隆策略仍会失败：一阵风，一个没见过的格子，然后错上加错。'),
			L('Describe one DAgger round (the policy drives, the expert labels the cells it visited, retrain) and predict what it fixes and what it cannot.', '描述 DAgger 的一轮（策略自己开，专家给它到过的格子标上动作，再训练），并预测它能修好什么、修不好什么。')
		],
		prequestion: {
			id: 'behavior-cloning-pre',
			question: L(
				'You show a robot how to cross a windy field once, and it copies every one of your moves perfectly. Will it now cross as reliably as you do?',
				'你给机器人示范了一次怎样穿过一片刮风的空地，它把你的每一步都学得一模一样。它现在能像你一样稳稳地穿过去吗？'
			),
			options: [L('Yes: it copies every move', '能：每一步它都照做'), L('No: once the wind pushes it somewhere you never went, it has nothing to copy', '不能：一旦风把它吹到你没去过的地方，它就没有可以照做的了')],
			answer: 1,
			explain: L(
				'Copying is only as good as the situations it saw. A gust puts the robot in a cell you never demonstrated, its guess there can push it further away, and the errors pile up. That is compounding error, and this page measures it.',
				'照着学，只在见过的情形里管用。一阵风把机器人吹到你从没示范过的格子，它在那里的猜测可能让它离得更远，错误就越积越多。这就是误差累积，本页会把它量出来。'
			)
		},
		tryThis: [
			{ id: 'calm', title: L('No wind', '没有风'), setup: { input: { ...DEFAULTS, wind: 0 } }, predict: L('Without gusts, will the cloned policy ever leave your path?', '没有阵风时，克隆的策略会离开你的路线吗？'), look: L('Look at how many of the episodes reach the goal, and at the moves on cells it never saw.', '看有多少个回合到达终点，以及它在没见过的格子上走了几步。'), goto: { step: 'drive' } },
			{ id: 'corner', title: L('Another map', '另一张地图'), setup: { preset: 'corner' }, predict: L('Now the start is top left and the goal sits low on the right edge, close to the pits. Will DAgger still close the gap to the expert?', '这次起点在左上角，终点在右边靠下、离陷阱很近的地方。DAgger 还能追上专家吗？'), look: L('Look at the success rates of cloning, DAgger and the expert.', '看克隆、DAgger 和专家的成功率。'), goto: { step: 'dagger' } }
		],
		quiz: [
			{ q: L('Behavior cloning trains the policy network with which kind of learning?', '行为克隆用哪种学习方式训练策略网络？'), options: [L('Supervised learning on (state, move) pairs', '在（状态，动作）对上做监督学习'), L('Rewards from the goal', '用终点给的奖励'), L('No training at all', '完全不训练')], answer: 0, explain: L('Each demonstrated cell is an input and the move made there is its label: a classifier with four classes, trained with cross-entropy.', '每个示范过的格子是一个输入，在那里走的那一步就是它的标签：一个四分类的分类器，用交叉熵训练。') },
			{ q: L('What does DAgger ask the expert for?', 'DAgger 向专家要什么？'), options: [L('More demonstrations from the start', '从起点再多示范几次'), L('The right move on the cells the learner’s own policy visited', '学习者自己的策略到过的格子上的正确动作'), L('A reward for each episode', '每个回合的奖励')], answer: 1, explain: L('The new labels come from where the learner actually goes, so the data covers the cells its mistakes lead to.', '新标签来自学习者真正走到的地方，所以数据覆盖了它的错误会把它带去的格子。') }
		],
		misconception: {
			claim: L('If the network gets every training example right, the cloned policy works.', '只要网络把每个训练样本都答对，克隆出来的策略就能用。'),
			truth: L(
				'Training accuracy only covers the cells you demonstrated. The policy chooses where it goes next, and a gust or one wrong guess takes it to cells the training data never covered.',
				'训练准确率只覆盖你示范过的格子。策略自己决定下一步去哪儿，一阵风或一次猜错就会把它带到训练数据从没覆盖过的格子。'
			),
			step: 'compound'
		},
		// the kit's Training section plots the cloning loss; the lesson's steps show the numbers that matter here (accuracy
		// on the demonstrations, then success in the wind)
		sections: { hide: ['training'] },
		lessonNote: L(
			'Ten steps: your demonstrations, the dataset, the network, training, the cloned policy, a prediction, the policy in the wind, compounding error, your fix, DAgger, and what neither can do.',
			'十步：你的示范、数据集、网络、训练、克隆出的策略、一次预测、风中的策略、误差累积、你的修补、DAgger，以及两者都做不到的事。'
		)
	},
	glossary: [
		{ id: 'imitation-learning', term: L('imitation learning', '模仿学习'), def: L('Learning a policy from an expert’s demonstrations instead of from rewards.', '从专家的示范中学习策略，而不是从奖励中学习。'), related: ['policy'] },
		{ id: 'demonstration', term: L('demonstration', '示范'), def: L('A recorded run of an expert: the states it was in and the action it took in each.', '专家的一次记录下来的运行：它经过的状态，以及在每个状态下采取的动作。'), related: ['imitation-learning', 'state'] },
		{ id: 'behavior-cloning', term: L('behavior cloning', '行为克隆'), def: L('Imitation learning as supervised learning: train a classifier from states to the expert’s actions on the demonstrations.', '把模仿学习当成监督学习：在示范数据上训练一个从状态到专家动作的分类器。'), related: ['demonstration', 'classifier'] },
		{ id: 'compounding-error', term: L('compounding error', '误差累积'), def: L('When a learned policy’s small mistakes take it to states its training data never covered, where it makes bigger ones.', '学到的策略犯了小错，被带到训练数据没覆盖过的状态，在那里犯更大的错。'), related: ['behavior-cloning', 'out-of-distribution'] },
		{ id: 'dagger', term: L('DAgger (dataset aggregation)', 'DAgger（数据集聚合）'), def: L('Let the learned policy drive, ask the expert for the right action in every state it visited, add those pairs to the dataset and retrain; repeat.', '让学到的策略自己开，请专家给它到过的每个状态标上正确的动作，把这些对加进数据集再训练；如此反复。'), related: ['behavior-cloning', 'compounding-error'] }
	],
	catalog: {
		runtime: L('Own autograd core in a worker', '自研自动微分内核，在 Web Worker 中运行'),
		params: L(`MLP 2 → ${H} → 4 (${7 * H + 4} parameters)`, `多层感知机 2 → ${H} → 4（${7 * H + 4} 个参数）`)
	},
	// The grid rules, the expert, the network and the views are written here. Behavior cloning and DAgger are from the
	// papers (ideas only, no code); the windy-field framing follows the usual "drifting off the road" picture of
	// compounding error (Ross & Bagnell 2010).
	credits: [
		{ name: 'Ross, Gordon & Bagnell (2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning', url: 'https://arxiv.org/abs/1011.0686', license: 'Paper; ideas only', usedFor: 'DAgger: let the learner drive, label the visited states with the expert, aggregate the datasets and retrain', attribution: 'Stéphane Ross, Geoffrey J. Gordon and J. Andrew Bagnell' },
		{ name: 'Ross & Bagnell (2010), Efficient Reductions for Imitation Learning', url: 'https://proceedings.mlr.press/v9/ross10a.html', license: 'Paper; ideas only', usedFor: 'Compounding error of behavior cloning: mistakes take the learner to states the expert never showed', attribution: 'Stéphane Ross and J. Andrew Bagnell' },
		{ name: 'Pomerleau (1988), ALVINN: An Autonomous Land Vehicle in a Neural Network', url: 'https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html', license: 'Paper; ideas only', usedFor: 'Behavior cloning: a neural network learns steering from a driver’s demonstrations', attribution: 'Dean A. Pomerleau' }
	]
};

export default manifest;
