// snake-dqn: the package manifest (design §4.9). Kinds A and C: a Q-network trained offline by DQN
// (pipeline/snake-dqn/train.py; conv 3 → 16 → 32, linear 2,592 → 64 → 3 Q-values, 171,235 parameters) and the steps
// of DQN on the learner's own snake board, in TypeScript (process.ts).
//   fwd     the exported online network traced on the board (parity ≤ 1e-4 against PyTorch)
//   tgt     the target network of the same checkpoint (models/snake-dqn/target), traced on the board
//   dqn     one DQN step on the board: ε-greedy choice, reward, target, loss (same frames as pipeline/snake-dqn/reference.py)
//   replay  a replay buffer filled by the agent, and one batch drawn from it
//   agent   the trained agent playing a whole game on the learner's food seed
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { rawOf } from './adapters/snake-board/adapter.ts';
import { withWeights } from './process.ts';
import { weightsOf } from './qnet.ts';
import { L } from './rules.ts';

const w = (rt: ExplainerRuntime) => weightsOf(rt.model);
const wt = (rt: ExplainerRuntime) => weightsOf(rt.sources.tgt?.model);

const manifest: ExplainerManifest = {
	slug: 'snake-dqn',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/snake-dqn', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'snake-board', presets: 'presets.json', start: { seed: 7, moves: [1, 0] } },
	output: { view: 'snake-dqn/step', layer: 'out', props: { mode: 'q' } },
	sources: {
		fwd: { kind: 'trace' },
		tgt: { kind: 'trace', assets: 'models/snake-dqn/target' },
		dqn: { kind: 'process', process: 'dqn', seed: 4, input: (rt) => { const r = rawOf(rt.input); return withWeights({ seed: r.seed, moves: r.moves, draw: r.draw }, w(rt), wt(rt)); } },
		replay: { kind: 'process', process: 'replay', seed: 1, input: (rt) => withWeights({ seed: rawOf(rt.input).seed }, w(rt), wt(rt)) },
		agent: { kind: 'process', process: 'agent', input: (rt) => withWeights({ seed: rawOf(rt.input).seed }, w(rt)) }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			L('Read a snake board as the network’s 243 input numbers, and say what each of its three answers, Q(s, a), means.', '把贪吃蛇棋盘读成网络的 243 个输入数，并说出它的三个回答 Q(s, a) 各是什么意思。'),
			L('Follow ε-greedy with a real random draw: say when the agent explores and when it takes its best move.', '跟着一次真实的随机抽签走一遍 ε-贪心：说出智能体什么时候探索，什么时候走它最好的一步。'),
			L('Work out the target y = r + γ · max Q_target(s′, ·) and the loss (y − Q(s, a))² for one move, and say why y = r when the move ends the game.', '为一步算出目标值 y = r + γ · max Q_target(s′, ·) 和损失 (y − Q(s, a))²，并说明为什么这一步让游戏结束时 y = r。'),
			L('Explain why DQN learns from moves drawn at random from a replay buffer, against a target network that is copied only now and then.', '说明为什么 DQN 从回放缓冲区里随机抽旧步来学，而且对照的是一个只偶尔复制一次的目标网络。'),
			L('Read the training curve, and name two limits of this agent with measured numbers: a fixed small board, and a reward that shapes its behaviour.', '读懂训练曲线，并用实测数字说出这个智能体的两个局限：棋盘小而且固定，以及奖励会塑造它的行为。')
		],
		prequestion: {
			id: 'snake-dqn-pre',
			question: L('The snake only ever gets +1 for food and −1 for a crash. How can it learn to avoid a crash that becomes certain only a few moves later?', '蛇只会因为吃到食物得 +1，因为撞上东西得 −1。一个要再过几步才躲不掉的撞击，它怎么能学会提前避开？'),
			options: [L('Each move’s value counts the rewards that follow it, a little less for every move later, so the −1 leaks back to the moves that lead to it', '每一步的价值都算上它之后的奖励，越往后打得越多折扣，所以那个 −1 会一点点传回到导致它的那些步上'), L('Someone tells it the rules of the game', '有人把游戏规则告诉了它'), L('It remembers every board it has seen and the move that worked there', '它记住了见过的每一块棋盘，以及在那里管用的那一步')],
			answer: 0,
			explain: L('The network learns Q(s, a): the reward still to come after a move, with each later reward multiplied by γ once per move. A training target is r + γ times the best Q of the next board, so a −1 passes back one move per update, through every board that leads to it.', '网络学的是 Q(s, a)：走完一步之后还能得到的奖励，越往后的奖励每多一步就多乘一次 γ。训练目标是 r 加上 γ 乘以下一块棋盘上最好的 Q，所以一个 −1 每更新一次就往回传一步，传到通向它的每一块棋盘上。')
		},
		tryThis: [
			{ id: 'wall', title: L('Next to the wall', '贴着墙'), setup: { preset: 'wall' }, predict: L('Straight on hits the wall here. Will that move get the lowest Q-value of the three, and how low?', '这里直走会撞墙。这一步的 Q 值会是三个里最低的吗？会低到多少？'), look: L('The three arrows at the head and their numbers.', '蛇头上的三支箭头和它们的数。'), goto: { step: 'q' } },
			{ id: 'food', title: L('Food right ahead', '食物就在前面'), setup: { preset: 'food' }, predict: L('One move straight on eats the food. What will the target y be for that move: more or less than 1?', '直走一步就能吃到食物。这一步的目标值 y 会比 1 大还是比 1 小？'), look: L('The reward r and the number γ · max Q_target(s′) added to it.', '奖励 r，以及加在它上面的 γ · max Q_target(s′)。'), goto: { step: 'target' } }
		],
		quiz: [
			{
				q: L('A move gives r = 0, γ = 0.9, and the target network’s best Q on the next board is 2. What is the target y?', '某一步 r = 0，γ = 0.9，目标网络在下一块棋盘上最好的 Q 是 2。目标值 y 是多少？'),
				options: [L('1.8', '1.8'), L('2', '2'), L('2.9', '2.9')],
				answer: 0,
				explain: L('y = r + γ · max Q = 0 + 0.9 × 2 = 1.8.', 'y = r + γ · max Q = 0 + 0.9 × 2 = 1.8。')
			}
		],
		misconception: {
			claim: L('A Q-value is the reward the move will get.', 'Q 值就是这一步会得到的奖励。'),
			truth: L('It is the network’s guess of all the reward still to come after the move, each later reward discounted by γ per move. The move’s own reward is only the first part.', '它是网络对这一步之后还会得到的全部奖励的猜测，越往后的奖励每多一步就多打一次 γ 的折扣。这一步自己的奖励只是其中第一部分。'),
			step: 'q'
		},
		// the training log is lesson step 9 (a chart); the auto Training section would print raw keys
		sections: { hide: ['training'], intros: { output: L('The three Q-values on your board, drawn as arrows from the head: the move with the highest one is the move the trained agent makes.', '你的棋盘上的三个 Q 值，画成从蛇头出发的箭头：Q 值最高的那一步，就是训练好的智能体会走的一步。') } },
		lessonNote: L('Eleven steps: the board as numbers, three Q-values, ε-greedy with a real draw, the reward, the target and the loss for one move, the replay buffer, the target network, the training curve, a game against its score, and its limits.', '十一步：把棋盘变成数，三个 Q 值，带真实抽签的 ε-贪心，奖励，一步的目标值和损失，回放缓冲区，目标网络，训练曲线，和它比分数，以及它的局限。')
	},
	glossary: [
		{ id: 'dqn', term: L('deep Q-network (DQN)', '深度 Q 网络（DQN）'), def: L('A neural network that reads a state and outputs one Q-value per action, trained from the agent’s own moves with a replay buffer and a target network.', '读入一个状态、为每个动作输出一个 Q 值的神经网络；用智能体自己走过的步来训练，并配有回放缓冲区和目标网络。'), related: ['q-value'] },
		{ id: 'epsilon-greedy', term: L('ε-greedy', 'ε-贪心'), def: L('Take the action with the highest Q-value, except with chance ε, when a random action is taken instead.', '选 Q 值最高的动作，但有 ε 的机会改成随机选一个动作。'), related: ['exploration'] },
		{ id: 'td-target', term: L('TD target', 'TD 目标'), def: L('The number a Q-value is pulled towards: the reward just received plus γ times the best Q-value of the next state (just the reward if the episode ended).', 'Q 值被拉向的那个数：刚得到的奖励，加上 γ 乘以下一个状态里最好的 Q 值（如果回合结束了，就只有奖励）。'), related: ['q-value', 'discount'] },
		{ id: 'replay-buffer', term: L('replay buffer', '回放缓冲区'), def: L('A store of the agent’s recent moves (state, action, reward, next state); training draws random batches from it.', '存放智能体最近走过的步（状态、动作、奖励、下一个状态）；训练时从里面随机抽一批。') },
		{ id: 'target-network', term: L('target network', '目标网络'), def: L('A copy of the Q-network that is refreshed only every so many updates and is used to compute the targets, so they stay still between copies.', 'Q 网络的一个副本，每隔若干次更新才刷新一次，用来计算目标值，让目标值在两次复制之间保持不动。'), related: ['dqn'] },
		{ id: 'reward-shaping', term: L('reward shaping', '奖励塑形'), def: L('Adding extra rewards or penalties to steer learning; it can change what the agent ends up doing.', '额外加一些奖励或惩罚来引导学习；它会改变智能体最后学成的样子。'), related: ['reward'] }
	],
	catalog: {
		params: L('171,235 (two 3 × 3 convolutions with 16 and 32 channels, then 2,592 → 64 → 3 Q-values)', '171,235（两层 3 × 3 卷积，16 和 32 个通道，再 2,592 → 64 → 3 个 Q 值）'),
		size: L('669 KB of weights, plus 669 KB for the target network', '权重 669 KB，目标网络另有 669 KB'),
		runtime: L('Own JS forward pass (checked against PyTorch) and DQN steps in a worker', '自研 JS 前向计算（已与 PyTorch 核对），DQN 的各步在 Worker 线程中运行'),
		hook: L('Q-values drawn right on the snake board, then one DQN update worked out with real numbers.', '把 Q 值直接画在贪吃蛇棋盘上，再用真实数字算出一次 DQN 更新。'),
		shows: L('Your board as three planes of numbers, three Q-values as arrows, ε-greedy with a real draw, the reward, the target r + γ · max Q_target and the loss for one move, a replay buffer and a batch drawn from it, the target network’s staircase, the training curve, the agent’s game next to yours, and what a step penalty does to it.', '把你的棋盘变成三层数、画成箭头的三个 Q 值、带真实抽签的 ε-贪心、奖励、一步的目标值 r + γ · max Q_target 和损失、回放缓冲区以及从中抽出的一批、目标网络的阶梯、训练曲线、智能体的对局与你的对局并排，以及每步扣分对它的影响。')
	},
	// The method is DQN's (Mnih et al. 2015); the game, the board planes and the three relative moves follow the idea of the
	// TensorFlow.js snake-dqn example (Apache-2.0), reimplemented here: no code or weights were copied.
	credits: [
		{ name: 'Human-level control through deep reinforcement learning (Mnih et al., Nature 2015)', url: 'https://doi.org/10.1038/nature14236', license: 'Paper; ideas only (no code or figures used)', usedFor: 'The method: a Q-network trained from a replay buffer against a periodically copied target network, with ε-greedy exploration', attribution: 'Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. (DeepMind)' },
		{ name: 'TensorFlow.js examples: snake-dqn', url: 'https://github.com/tensorflow/tfjs-examples/tree/master/snake-dqn', license: 'Apache-2.0; idea only, reimplemented (no code or weights copied)', usedFor: 'The idea of the page: snake on a small board read as planes by a convolutional Q-network, with three moves relative to the heading', attribution: 'The TensorFlow.js authors (Google)' }
	]
};

export default manifest;
