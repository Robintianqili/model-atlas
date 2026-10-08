// cart-pole: the package manifest (design §4.9). Kinds B and C: a policy network learns to balance a pole by REINFORCE,
// live in the package worker (trainer.ts), and one episode is followed event by event (process.ts). Every source has
// seed 1, so they all start from the same fresh policy, and the episode process is exactly the first episode of round 0.
//   episode  process 'episode': the fresh policy's first episode from your start state, then its returns
//   train    the page's training: Adam lr 0.02, 10 episodes per round, 100 rounds (the twin's parity source)
//   fresh    the trainer with 0 rounds: only the fresh weights (p0 traces them)
//   p0       a kind-A trace of your start state through the fresh policy (zoom.weighted-sum)
//   step1    the first round slowed down: plain SGD (lr 0.5), fine detail (forward, loss, backward, update)
//   seeds    the same training with seeds 1 to 5 (the limits step)
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { DEFAULT_START } from './sim.ts';

const L = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'cart-pole',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'cart-pole', start: { start: DEFAULT_START } },
	sources: {
		episode: { kind: 'process', process: 'episode', run: 'main', seed: 1 },
		train: { kind: 'train', trainer: 'reinforce', detail: 'coarse', opt: { seed: 1 } },
		fresh: { kind: 'train', trainer: 'reinforce', detail: 'coarse', opt: { seed: 1, steps: 0 } },
		p0: { kind: 'live-model', of: 'fresh' },
		step1: { kind: 'train', trainer: 'reinforce', detail: 'fine', opt: { seed: 1, optimizer: 'sgd', lr: 0.5, steps: 1 } },
		seeds: { kind: 'process', process: 'seeds', run: 'worker', seed: 1 }
	},
	page: {
		goals: [
			L('Read the four numbers of the cart-pole state, and say what the policy network makes of them: a chance of pushing left.', '读懂倒立摆状态的四个数，并说出策略网络从中得到什么：向左推的概率。'),
			L('Compute a discounted return by hand, and say why the last steps before a fall get a negative advantage.', '亲手算出一个折扣回报，并说明为什么倒下前的最后几步得到负的优势。'),
			L('Explain one REINFORCE update: each step pulls its action up or down by the sign of its advantage, and the pulls add up to the gradient.', '解释一次 REINFORCE 更新：每一步按优势的正负把自己的动作往上拉或往下压，这些拉力加起来就是梯度。'),
			L('Predict how episode length changes with training, and why runs with different seeds differ.', '预测回合长度如何随训练变化，以及为什么不同随机种子的训练结果不同。')
		],
		prequestion: {
			id: 'cart-pole-pre',
			question: L(
				'Nobody tells the network which way to push. After each try it only finds out how long the pole stayed up. Can it still learn to balance?',
				'没有人告诉网络该往哪边推。每次尝试之后，它只知道杆子立了多久。它还能学会平衡吗？'
			),
			options: [
				L('No: without the right answer for each push it cannot learn', '不能：没有每一推的正确答案，它学不会'),
				L('Yes: it can make the pushes of long tries more likely and those of short tries less likely', '能：它可以让“立得久”的尝试里的推法更常出现，让“立得短”的更少出现')
			],
			answer: 1,
			explain: L(
				'That is reinforcement learning with a policy gradient (REINFORCE): the only feedback is a reward of +1 for every step the pole stays up, and the network shifts its chances toward the pushes that came before many more rewards.',
				'这就是用策略梯度（REINFORCE）做的强化学习：唯一的反馈是杆子每多立一步就得到 +1 的奖励，网络把概率往“后面跟着很多奖励”的推法上挪。'
			)
		},
		tryThis: [
			{
				id: 'lean',
				title: L('A steep start', '倾斜得很厉害的起点'),
				setup: { input: { start: [0, 0, 0.15, 0.5] } },
				predict: L('The pole starts leaning right and is already falling further right. Which way will the trained policy push first, and can it save the pole?', '杆子一开始就向右倾斜，而且已经在继续往右倒。训练好的策略第一推会往哪边？还能救回来吗？'),
				look: L('Look at the angle curve under the cart: does it come back toward 0 before it reaches the 12° line?', '看小车下面的角度曲线：它在碰到 12° 线之前有没有回到 0 附近？'),
				goto: { step: 'trained' }
			}
		],
		quiz: [
			{
				q: L('An episode lasts 3 steps and γ = 0.5. What is the return of its first step?', '一个回合持续 3 步，γ = 0.5。它第一步的回报是多少？'),
				options: [L('3', '3'), L('1.75', '1.75'), L('1', '1')],
				answer: 1,
				explain: L('Backwards: the last step has 1, the one before 1 + 0.5 × 1 = 1.5, the first 1 + 0.5 × 1.5 = 1.75.', '倒着算：最后一步是 1，前一步是 1 + 0.5 × 1 = 1.5，第一步是 1 + 0.5 × 1.5 = 1.75。')
			}
		],
		misconception: {
			claim: L('Every push in a long episode was a good push.', '一个长回合里的每一推都是好的推法。'),
			truth: L(
				'The update only shifts chances on average. A lucky long episode also contains bad pushes, and they get pulled up too; that noise is why training needs many episodes, and why two seeds can end up far apart.',
				'更新只是在平均意义上挪动概率。一个碰巧很长的回合里也有坏的推法，它们也会被往上拉；正是这种噪声让训练需要很多回合，也让两个随机种子的结果相差很远。'
			),
			step: 'seeds'
		},
		// the kit's Training section plots the policy-gradient loss, which does not fall as the policy improves (the advantages
		// are renormalized every round); lesson steps 'train' and 'trained' show what does: the episode length per round
		sections: { hide: ['training'] },
		lessonNote: L(
			'Twelve steps: the state, the policy, one episode event by event, its returns and advantages, one update slowed down, then training to the end and how much luck matters.',
			'十二步：状态、策略、一个回合逐个事件地走一遍、它的回报和优势、放慢的一次更新，然后训练到底，以及运气有多重要。'
		)
	},
	glossary: [
		{ id: 'state', term: L('state', '状态'), def: L('The numbers that describe the world at one moment; the policy decides from them.', '描述某一时刻世界的那组数；策略根据它们做决定。'), related: ['policy'] },
		{
			id: 'return',
			term: L('return (discounted)', '（折扣）回报'),
			def: L('The rewards from one step to the end of the episode, each later one multiplied by one more γ: G = r + γ × (G of the next step).', '从某一步到回合结束的奖励之和，越往后的奖励多乘一次 γ：G = r + γ ×（下一步的 G）。'),
			related: ['reward', 'discount']
		},
		{
			id: 'advantage',
			term: L('advantage', '优势'),
			def: L('How much better a step’s return is than usual: here (G − mean) / standard deviation over one round of episodes.', '某一步的回报比平常好多少：这里是 (G − 平均值) / 标准差，在一轮回合里计算。'),
			related: ['return']
		},
		{
			id: 'policy-gradient',
			term: L('policy gradient (REINFORCE)', '策略梯度（REINFORCE）'),
			def: L('Training a policy by making actions followed by a high return more likely and the others less likely.', '训练策略的一种方法：让后面跟着高回报的动作更可能出现，让其余的更少出现。'),
			related: ['policy', 'gradient']
		}
	],
	catalog: {
		runtime: L('Own autograd core in a worker', '自研自动微分内核，在 Web Worker 中运行'),
		params: L('4 → 16 → 1 policy (97 parameters)', '4 → 16 → 1 的策略网络（97 个参数）')
	},
	// The algorithm and the physics are written here. The idea of a REINFORCE cart-pole trained in the browser comes from the
	// TensorFlow.js example (no code copied); the physics from the classic paper, in the form OpenAI Gym states it.
	credits: [
		{
			name: 'TensorFlow.js examples: cart-pole (REINFORCE)',
			url: 'https://github.com/tensorflow/tfjs-examples/tree/master/cart-pole',
			license: 'Apache-2.0',
			usedFor: 'The idea and the setup (a policy network giving p(left), discounted and normalized rewards, training in the browser); reimplemented on the Model Atlas autograd core, no code copied',
			attribution: 'The TensorFlow.js authors, Google'
		},
		{
			name: 'Barto, Sutton & Anderson (1983), Neuronlike adaptive elements that can solve difficult learning control problems',
			url: 'https://doi.org/10.1109/TSMC.1983.6313077',
			license: 'Paper; ideas only (the cart-pole equations and constants, as OpenAI Gym states them)',
			usedFor: 'The cart-pole physics: masses, pole length, 10 N pushes, 0.02 s steps, the 12° and 2.4 m limits',
			attribution: 'Andrew G. Barto, Richard S. Sutton and Charles W. Anderson'
		}
	]
};

export default manifest;
