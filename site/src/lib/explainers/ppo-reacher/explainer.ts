// ppo-reacher: the package manifest. Kinds B and C: PPO with GAE trains a Gaussian policy (and a value network) to move
// a two-link arm's fingertip to a target, live in the package worker (trainer.ts), and one episode is followed event by
// event (process.ts). Every source has seed 1, so all start from the same fresh networks, and the episode process is
// exactly episode 0 of the first batch.
//   episode  process 'episode': the fresh policy's first episode to your target, then its GAE advantages
//   iter     the first PPO iteration slowed down: 16 updates on one batch, fine detail (ratios, clipped samples)
//   train    the page's training: 100 iterations of 16 updates (Adam, learning rate 0.005), with your ε
//   pair     the task's comparison: the train source's run and the same training with ε = 0.2, in lockstep
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { DEFAULT_EPS, DEFAULT_TARGET, targetOf } from './sim.ts';

const L = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'ppo-reacher',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'reacher', start: { target: DEFAULT_TARGET, eps: DEFAULT_EPS }, presetsFromAdapter: true },
	sources: {
		episode: { kind: 'process', process: 'episode', seed: 1, input: (rt) => ({ target: targetOf(rt.input) }) },
		iter: { kind: 'train', trainer: 'ppo', detail: 'fine', opt: { seed: 1, steps: 16 } },
		train: { kind: 'train', trainer: 'ppo', detail: 'coarse', opt: { seed: 1 } },
		pair: { kind: 'process', process: 'pair', seed: 1, reference: false }
	},
	page: {
		goals: [
			L('Say what a Gaussian policy outputs for a continuous action, and how one torque is sampled from μ and σ.', '说出高斯策略对连续动作输出什么，以及怎样用 μ 和 σ 采样出一个力矩。'),
			L('Work out a TD error and a GAE advantage from rewards and a value network’s guesses.', '用奖励和价值网络的估计，算出一个 TD 误差和一个 GAE 优势。'),
			L('Explain the probability ratio and which samples PPO’s clip stops, and what that prevents.', '解释概率比率，说出 PPO 的截断会让哪些样本停下，以及这样能防止什么。'),
			L('Predict how a larger or smaller clip ε changes the stability of training, and check it.', '预测截断 ε 变大或变小时训练的稳定性会怎样变化，并亲自验证。')
		],
		prequestion: {
			id: 'ppo-reacher-pre',
			question: L(
				'Playing episodes is slow, so PPO learns from each batch of episodes 16 times instead of once. What is the danger of reusing the same batch?',
				'玩回合很费时间，所以 PPO 用每一批回合学 16 次，而不是 1 次。反复用同一批数据有什么危险？'
			),
			options: [L('None: more updates on the same data are always better', '没有危险：在同样的数据上多更新几次总是更好'), L('The policy can drift far from the one that played the episodes, and the batch stops describing it', '策略可能偏离玩这些回合的那个策略太远，这批数据就不再能描述它了')],
			answer: 1,
			explain: L(
				'The advantages say which actions were good for the policy that played them. After many updates the policy is a different one, and pushing further on old evidence can wreck it. PPO’s clip stops each sample from pulling once the policy has moved far enough on it.',
				'优势说明的是：对玩这些回合的那个策略来说，哪些动作好。更新多次以后，策略已经变了，再依据旧证据往前推，可能把它推坏。PPO 的截断让每个样本在策略对它挪得够远时就不再拉动。'
			)
		},
		quiz: [
			{
				q: L('The last two steps of an episode have TD errors δ = −1 and then δ = −1, and γλ = 0.9. What is the advantage A of the first of the two?', '一个回合的最后两步，TD 误差先后是 δ = −1 和 δ = −1，γλ = 0.9。这两步中前一步的优势 A 是多少？'),
				options: [L('−1', '−1'), L('−1.9', '−1.9'), L('−2', '−2')],
				answer: 1,
				explain: L('Backwards: the last step has A = δ = −1; the one before A = −1 + 0.9 × (−1) = −1.9.', '倒着算：最后一步 A = δ = −1；前一步 A = −1 + 0.9 × (−1) = −1.9。')
			}
		],
		misconception: {
			claim: L('Clipping keeps every probability ratio inside [1 − ε, 1 + ε].', '截断会把每个概率比率都限制在 [1 − ε, 1 + ε] 之内。'),
			truth: L(
				'Clipping only stops a sample from pulling further; the other samples’ updates still move the shared weights, so some ratios end up outside the range anyway. It slows the drift, it does not wall it in.',
				'截断只是让某个样本不再继续拉；其他样本的更新仍然在移动共享的权重，所以有些比率最后还是会跑到范围外面。它减慢了漂移，并没有把它围死。'
			),
			step: 'clip'
		},
		// the kit's Training section plots the PPO loss, which does not fall as the policy improves (the advantages are
		// renormalized every batch); lesson steps 'train' and 'epsilon' show what does: the average episode return
		sections: { hide: ['training'] },
		lessonNote: L(
			'Eleven steps: the arm, the Gaussian policy, one episode, the critic and GAE, the batch, the ratio and the clip, training, then changing ε and where the trained policy fails.',
			'十一步：机械臂、高斯策略、一个回合、评论家和 GAE、一批数据、比率和截断、训练，然后改变 ε，以及训练好的策略在哪里失败。'
		)
	},
	glossary: [
		{ id: 'gaussian-policy', term: L('Gaussian policy', '高斯策略'), def: L('A policy for continuous actions: a network gives the mean μ of each action, and the action is drawn from a bell curve of width σ around it.', '用于连续动作的策略：网络给出每个动作的均值 μ，动作从以它为中心、宽度为 σ 的钟形曲线里抽取。'), related: ['policy'] },
		{ id: 'value-network', term: L('value network (critic)', '价值网络（评论家）'), def: L('A network V(s) that guesses the discounted return still to come from a state; the policy (the actor) is judged against it.', '一个网络 V(s)，估计从某个状态起之后还能拿到的折扣回报；策略（演员）以它为参照来评判。'), related: ['return'] },
		{ id: 'td-error', term: L('TD error δ', 'TD 误差 δ'), def: L('δ = r + γ·V(s′) − V(s): how much better one step went than the critic expected.', 'δ = r + γ·V(s′) − V(s)：这一步比评论家预期的好多少。'), related: ['value-network', 'discount'] },
		{ id: 'gae', term: L('GAE (generalized advantage estimation)', 'GAE（广义优势估计）'), def: L('An advantage that adds up the TD errors of the steps that follow, each later one weighted by one more γλ: A_t = δ_t + γλ·A_(t+1).', '把后面各步的 TD 误差加起来得到的优势，越往后的多乘一次 γλ：A_t = δ_t + γλ·A_(t+1)。'), related: ['td-error'] },
		{ id: 'probability-ratio', term: L('probability ratio', '概率比率'), def: L('r = π_new(a | s) / π_old(a | s): how much more likely the updated policy makes an action than the policy that took it.', 'r = π_new(a | s) / π_old(a | s)：更新后的策略让某个动作的可能性，是当初做出这个动作的策略的多少倍。'), related: ['policy'] },
		{ id: 'ppo-clip', term: L('PPO clipped objective', 'PPO 截断目标'), def: L('min(r·A, clip(r, 1 − ε, 1 + ε)·A): a sample stops pulling once its ratio has moved past 1 ± ε in the direction its advantage wants.', 'min(r·A, clip(r, 1 − ε, 1 + ε)·A)：一个样本的比率朝它的优势想要的方向越过 1 ± ε 之后，就不再拉动。'), related: ['probability-ratio', 'gae'] }
	],
	catalog: {
		hook: L('Teach a two-link arm to reach your target with PPO, and see which samples its clip stops from pulling.', '用 PPO 教一条两节机械臂够到你的目标，看看它的截断让哪些样本不再拉动策略。'),
		shows: L('Your target for a two-link arm, the Gaussian policy’s bell curves over two torques, one sampled episode, the critic’s guesses with TD errors and GAE advantages worked backwards, a batch of 16 episodes split into minibatches, probability ratios and the clipped objective on real samples, the return over 100 iterations of training, your clip ε against ε = 0.2, and targets the arm never practised.', '你给两节机械臂设的目标、高斯策略在两个力矩上的钟形曲线、采样出的一个回合、评论家的估计以及倒着算出的 TD 误差和 GAE 优势、拆成小批量的 16 个回合、真实样本上的概率比率和截断目标、100 次迭代训练中的回报、你的截断 ε 与 ε = 0.2 的对比，以及机械臂从没练过的目标。'),
		runtime: L('Own autograd core in a worker', '自研自动微分内核，在 Web Worker 中运行'),
		params: L('Actor 6 → 32 → 2 plus 2 learned log σ (292 parameters) and critic 6 → 32 → 1 (257 parameters)', '演员网络 6 → 32 → 2，外加 2 个可学习的 log σ（292 个参数）；评论家网络 6 → 32 → 1（257 个参数）'),
		size: L('No files: the arm and both networks run in your browser', '没有文件：机械臂和两个网络都在你的浏览器里运行')
	},
	// The algorithm, the arm and the views are written here. PPO and GAE are from the papers (ideas only); the training setup
	// (separate actor and critic, a learned log-std, advantage normalization, minibatch epochs) follows CleanRL's
	// ppo_continuous_action (MIT), reimplemented on the Model Atlas autograd core; no code copied.
	credits: [
		{ name: 'Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal Policy Optimization Algorithms', url: 'https://arxiv.org/abs/1707.06347', license: 'Paper; ideas only',
			usedFor: 'The clipped objective min(r·A, clip(r, 1 − ε, 1 + ε)·A), several epochs of minibatch updates per batch', attribution: 'John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov' },
		{ name: 'Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-Dimensional Continuous Control Using Generalized Advantage Estimation', url: 'https://arxiv.org/abs/1506.02438', license: 'Paper; ideas only',
			usedFor: 'GAE: δ = r + γ·V(s′) − V(s), A_t = δ_t + γλ·A_(t+1)', attribution: 'John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan and Pieter Abbeel' },
		{ name: 'CleanRL: ppo_continuous_action.py', url: 'https://github.com/vwxyzjn/cleanrl', license: 'MIT',
			usedFor: 'The training recipe (separate actor and critic, a state-independent log-std, normalized advantages, shuffled minibatch epochs); reimplemented on the Model Atlas autograd core, no code copied', attribution: 'Shengyi Huang and the CleanRL contributors' }
	]
};

export default manifest;
