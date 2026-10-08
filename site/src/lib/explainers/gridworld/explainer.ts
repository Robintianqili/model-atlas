// gridworld: the package manifest (design §4.9). Kind C only: every stage is a process run live in your browser
// (process.ts), checked frame by frame against an independent Python run (pipeline/gridworld/reference.py).
// The bandit sources share BANDIT_SEED, so their machines pay the same to every player (the race repeats each single run).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { BANDIT_SEED, BOARDS, DEFAULTS, GAMMAS, toRaw } from './world.ts';

const I = (en: string, zh: string) => ({ en, zh });
const bandit = (policy: string) => ({ kind: 'process' as const, process: 'bandit', run: 'main' as const, seed: BANDIT_SEED, opt: { policy } });

const manifest: ExplainerManifest = {
	slug: 'gridworld',
	type: 'model',
	kinds: ['process'],
	quality: 'reviewed',
	assets: { dir: 'models/gridworld', auto: [] },
	input: { adapter: 'gridworld', presets: 'presets.json', start: DEFAULTS },
	sources: {
		you: bandit('you'),
		greedy: bandit('greedy'),
		eps: bandit('eps'),
		ucb: bandit('ucb'),
		race: { kind: 'process', process: 'race', run: 'worker', seed: BANDIT_SEED },
		q: { kind: 'process', process: 'q', run: 'worker', seed: 12 },
		vi05: { kind: 'process', process: 'vi', run: 'main', opt: { gamma: GAMMAS[0] } },
		vi95: { kind: 'process', process: 'vi', run: 'main', opt: { gamma: GAMMAS[1] } }
	},
	page: {
		goals: [
			I('Explain, with a run you watched, why always picking the best-looking slot machine can get stuck, and how ε-greedy and UCB avoid it.', '结合你看到的一次运行，解释为什么总选看起来最好的老虎机会卡住，以及 ε-贪心和 UCB 怎样避免它。'),
			I('Compute one Q-learning update by hand: the target r + γ·max Q(s′), then Q ← Q + α·(target − Q).', '亲手算一次 Q 学习更新：先算目标值 r + γ·max Q(s′)，再算 Q ← Q + α·(目标值 − Q)。'),
			I('Say why values spread backward from the goal, read the policy from the arrows, and predict how γ changes a far cell’s value.', '说明价值为什么从终点往回扩散，从箭头读出策略，并预测 γ 如何改变远处格子的价值。')
		],
		prequestion: { id: 'gridworld-pre', question: I('You may pull two slot machines 100 times in total. A paid a coin on its first pull; B paid nothing on its first pull. What should you do?', '你一共可以拉两台老虎机 100 次。A 第一次就出了金币，B 第一次什么也没给。你该怎么做？'), options: [I('Pull A all the remaining times', '剩下的全拉 A'), I('Keep trying both for a while', '两台都再试一阵子'), I('Pull B all the remaining times', '剩下的全拉 B')], answer: 1, explain: I('One pull says very little about a machine. Trying both for a while (exploring) before settling on one (exploiting) is the trade-off this page is about.', '拉一次几乎说明不了什么。先把两台都试一阵子（探索），再选定一台（利用）：这就是本页要讲的取舍。') },
		tryThis: [
			{ id: 'cliff', title: I('The cliff', '悬崖'), setup: { input: toRaw({ ...DEFAULTS, rows: BOARDS.cliff }) }, predict: I('The shortest path runs right along a row of pits. Will the learned arrows hug the pits or keep their distance?', '最短的路线紧挨着一排陷阱。学到的箭头会贴着陷阱走，还是离它们远一点？'), look: I('Look at the arrows in the policy step, and at how many episodes fell in while it learned.', '看策略那一步的箭头，再看学习过程中有多少回合掉了进去。'), goto: { step: 'g-policy' } },
			{ id: 'no-explore', title: I('Never explore', '从不探索'), setup: { input: toRaw({ ...DEFAULTS, epsilon: 0 }) }, predict: I('With ε = 0 the agent never explores on purpose. Will it still find the goal?', 'ε = 0 时，智能体从不主动探索。它还能找到终点吗？'), look: I('Look at how many episodes reached the goal, and whether the arrows still take the shortest path.', '看有多少回合走到了终点，以及箭头是否还走最短路线。'), goto: { step: 'g-spread' } }
		],
		quiz: [
			{ q: I('A move reaches the goal (+1) and ends the episode. What is its Q-learning target?', '某一步走到了终点（+1），回合结束。它的 Q 学习目标值是多少？'), options: [I('1', '1'), I('1 + γ·max Q(s′)', '1 + γ·max Q(s′)'), I('0', '0')], answer: 0, explain: I('The episode ends there, so there is no next cell: the target is just r.', '回合在那里结束，没有下一个格子：目标值就是 r。') },
			{ q: I('Which strategy never explores on purpose?', '哪种策略从不主动探索？'), options: [I('greedy', '贪心'), I('ε-greedy', 'ε-贪心'), I('UCB', 'UCB')], answer: 0, explain: I('ε-greedy explores with chance ε, and UCB’s bonus pulls it back to machines it has tried little.', 'ε-贪心以 ε 的机会探索，UCB 的奖励分会把它拉回试得少的机器。') }
		],
		misconception: { claim: I('After reaching the goal once, the agent knows the way.', '只要到过一次终点，智能体就认得路了。'), truth: I('The first success changes one Q value, the move into the goal. The value needs many more episodes to spread back to the start.', '第一次成功只改变一个 Q 值：走进终点的那一步。价值还要再经过许多回合，才能传回起点。'), step: 'g-spread' },
		lessonNote: I('Seventeen steps: six on slot machines (bandits), eleven on a maze (Q-learning), all computed live in your browser.', '十七步：六步老虎机（多臂老虎机），十一步迷宫（Q 学习），全部在你的浏览器里实时计算。')
	},
	glossary: [
		{ id: 'agent', term: I('agent', '智能体'), def: I('The learner in reinforcement learning: it acts, sees what happens and gets rewards.', '强化学习中的学习者：它采取行动，观察结果，并得到奖励。') },
		{ id: 'reward', term: I('reward', '奖励'), def: I('A number the world gives after each action; the agent tries to collect as much as possible.', '每次行动后环境给出的一个数；智能体想尽量多拿。'), related: ['agent'] },
		{ id: 'episode', term: I('episode', '回合'), def: I('One try from the start until the task ends (a goal, a pit or a move limit).', '从起点开始直到任务结束（终点、陷阱或步数上限）的一次尝试。') },
		{ id: 'exploration', term: I('exploration vs exploitation', '探索与利用'), def: I('Trying actions to learn more about them, versus taking the action that looks best now.', '尝试一些动作以了解它们，与选择眼下看起来最好的动作之间的取舍。') },
		{ id: 'q-value', term: I('Q value, Q-table', 'Q 值、Q 表'), def: I('Q(s, a): the reward an agent expects in the long run after action a in state s; a Q-table stores one per state and action.', 'Q(s, a)：在状态 s 做动作 a 之后，长远来看期望得到的奖励；Q 表为每个状态和动作各存一个。'), related: ['reward'] },
		{ id: 'discount', term: I('discount factor γ', '折扣因子 γ'), def: I('A number between 0 and 1 that makes a reward one step later worth γ times as much.', '0 到 1 之间的数，让晚一步到手的奖励只值原来的 γ 倍。'), related: ['q-value'] },
		{ id: 'policy', term: I('policy', '策略'), def: I('The agent’s rule for which action to take in each state.', '智能体在每个状态下选哪个动作的规则。') },
		{ id: 'regret', term: I('regret', '遗憾'), def: I('The reward a player expects to miss compared with always taking the best action.', '与一直选最好的动作相比，玩家预计少得到的奖励。') }
	],
	catalog: { runtime: I('Own TypeScript, live in the browser (no neural network)', '自研 TypeScript，在浏览器中实时运行（无神经网络）'), params: I('none (a Q-table of 4 numbers per cell)', '无（Q 表：每个格子 4 个数）') },
	// No data, model or code from anyone else: the maze rules, the strategies and Q-learning are written here. The cliff
	// board and the order of ideas follow a textbook (ideas only, no text, figures or code).
	credits: [{ name: 'Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed., 2018)', url: 'http://incompleteideas.net/book/the-book-2nd.html', license: 'Book, CC BY-NC-ND 2.0; ideas only (no text, figures or code used)', usedFor: 'the cliff board (Example 6.6) and the order of ideas: bandits, then Q-learning', attribution: 'Richard S. Sutton and Andrew G. Barto' }]
};

export default manifest;
