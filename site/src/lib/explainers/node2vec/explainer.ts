// node2vec: the manifest. Biased random walks on Zachary's karate club (kind C) become sentences, and skip-gram (the
// word2vec package's trainer, copied) learns one vector per member from them, live (kind B). Sources:
//   run    trainer 'sg' on the learner's graph, p and q (the map, the nearest members, the GCN link, the limits)
//   walks  process 'walks': every move of every walk with its chances (Python reference: pipeline/node2vec/reference.py)
//   one    the same run as `run` (same seed), fine detail, 1 step: the first training step slowed down
//   base   trainer 'sg' on the learner's graph with the start p and q: the task compares the learner's choice with it
// Twin of trainer 'sg': pipeline/node2vec/twin.py. The one measured number not computed live (the GCN page's result) is
// in facts.json (pipeline/node2vec/facts.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { BASE, START, WALK_SEED } from './walks.ts';

const L = (en: string, zh: string) => ({ en, zh });
/** the seed of the training runs (`run`, `one` and `base` start from the same weights) */
export const SEED = 3;

const manifest: ExplainerManifest = {
	slug: 'node2vec',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/node2vec', auto: ['facts.json'] },
	input: { adapter: 'walk-graph', start: START, presetsFromAdapter: true },
	sources: {
		run: { kind: 'train', trainer: 'sg', detail: 'coarse', opt: { seed: SEED } },
		walks: { kind: 'process', process: 'walks', seed: WALK_SEED },
		one: { kind: 'train', trainer: 'sg', detail: 'fine', opt: { seed: SEED, steps: 1 } },
		base: { kind: 'train', trainer: 'sg', detail: 'coarse', opt: { seed: SEED, ...BASE } }
	},
	page: {
		goals: [
			L('Work out the chances of one node2vec walk step from p, q and who is friends with whom.', '根据 p、q 和谁与谁是朋友，算出 node2vec 游走一步的各个概率。'),
			L('Explain how walks become (center, context) pairs and how skip-gram turns them into one vector per member.', '解释游走怎样变成（中心，上下文）训练对，skip-gram 又怎样把它们变成每个成员的一个向量。'),
			L('Predict and check how p and q change how far walks wander and which members end up nearest.', '预测并验证 p 和 q 怎样改变游走走多远、以及哪些成员最后离得最近。'),
			L('Say what node2vec cannot do that a GCN can: use member features, and place a new member without retraining.', '说出 node2vec 做不到而 GCN 能做到的事：利用成员的特征，以及不重新训练就安置一个新成员。')
		],
		prequestion: {
			id: 'node2vec-pre',
			question: L('node2vec learns a vector for every member of a club from nothing but random walks along the friendships. Which members end up with similar vectors?', 'node2vec 只靠沿着朋友关系的随机游走，为俱乐部的每个成员学一个向量。哪些成员最后会得到相似的向量？'),
			options: [L('Members with the same number of friends', '朋友数相同的成员'), L('Members that often turn up close together in the walks', '在游走中经常挨得很近的成员'), L('Members with nearby numbers (ids)', '编号相近的成员')],
			answer: 1,
			explain: L(
				'Skip-gram trains each member’s vector to predict the members near it in the walks. Members that often share a walk need the same predictions, so their vectors become alike; in a club that means friends and friends of friends, mostly on the same side.',
				'skip-gram 训练每个成员的向量去预测游走中它附近的成员。经常出现在同一段游走里的成员需要做出相同的预测，所以它们的向量会变得相像；在俱乐部里，这就是朋友和朋友的朋友，大多在同一方。'
			)
		},
		tryThis: [
			{
				id: 'close',
				title: L('Walks that stay close', '待在附近的游走'),
				setup: { input: { ...START, p: 0.25, q: 4 } },
				predict: L('Going back now weighs 4 and moving farther weighs 0.25. Will a walk end nearer to its start than with the start setting?', '现在走回去的权重是 4，走得更远的权重是 0.25。和开始的设置相比，游走结束时会离起点更近吗？'),
				look: L('Step 3 counts how many hops from its start a walk ends, on average.', '第 3 步数出游走结束时平均离起点几跳。'),
				goto: { step: 'sentence' }
			}
		],
		quiz: [
			{
				q: L('A walk is at member v and came from t. v has 4 friends: t, one friend of t, and two members t does not know. With p = 1 and q = 0.5, what is the chance of going back to t?', '游走在成员 v，是从 t 走来的。v 有 4 个朋友：t、t 的一个朋友，以及两个 t 不认识的成员。p = 1、q = 0.5 时，走回 t 的概率是多少？'),
				options: [L('1/4', '1/4'), L('1/6', '1/6'), L('1/3', '1/3')],
				answer: 1,
				explain: L('The weights are 1 (back, 1/p), 1 (a friend of t) and 2 + 2 (the others, 1/q each): 6 in all, so going back has 1/6.', '权重是 1（回去，1/p）、1（t 的朋友）和 2 + 2（其余两个，各 1/q）：一共 6，所以走回去的概率是 1/6。')
			}
		],
		misconception: {
			claim: L('node2vec must be told who is on which side to draw the two sides apart.', 'node2vec 必须被告知谁在哪一方，才能把两方分开。'),
			truth: L('It never sees the sides. Members of one side meet each other in the walks far more often than they meet the other side, and that alone pulls the map apart.', '它从来看不到谁在哪一方。同一方的成员在游走中相遇的次数远多于和另一方相遇，仅这一点就把图分开了。'),
			step: 'map'
		},
		lessonNote: L(
			'Ten steps: the club as a graph, one biased walk step with its chances, walks as sentences, their pairs, one skip-gram step with its numbers, the map of the members, your own p and q, nearest members, a comparison with the GCN, and the limits.',
			'十步：把俱乐部看成图，带着概率走一步有偏游走，游走就是句子，它们给出的训练对，带着数字算一步 skip-gram，成员向量图，你自己选的 p 和 q，最近的成员，和 GCN 比一比，最后是局限。'
		),
		metrics: {
			avgLoss: { label: L('average loss over all pairs', '所有训练对的平均损失'), fmt: 'n3' },
			pairs: { label: L('(center, context) pairs from the walks', '游走给出的（中心，上下文）训练对'), fmt: 'int' }
		},
		sections: { titles: { training: L('Train it yourself', '自己训练') } }
	},
	glossary: [
		{ id: 'graph-walk', term: L('random walk on a graph', '图上的随机游走'), def: L('Start at a node and hop to a randomly chosen neighbour, again and again; the nodes visited, in order, are the walk.', '从一个节点出发，一次又一次地随机跳到一个邻居；按顺序经过的节点就是这条游走。'), related: ['graph', 'neighbour'] },
		{ id: 'node2vec-pq', term: L('p and q (node2vec)', 'p 和 q（node2vec）'), def: L('The two knobs of a node2vec walk: going straight back weighs 1/p, moving to a node farther from the one it came from weighs 1/q, and a node next to both weighs 1.', 'node2vec 游走的两个旋钮：直接走回去的权重是 1/p，走到离来处更远的节点权重是 1/q，走到和两者都相邻的节点权重是 1。'), related: ['graph-walk'] },
		{ id: 'bfs-dfs', term: L('BFS and DFS (breadth-first, depth-first search)', 'BFS 和 DFS（广度优先、深度优先搜索）'), def: L('Two ways to explore a graph: breadth-first visits everything near the start before going farther; depth-first follows one path as far as it can.', '探索图的两种方式：广度优先先把起点附近的都走遍再往远处走；深度优先沿着一条路尽量往远处走。'), related: ['graph'] },
		{ id: 'node-embedding', term: L('node embedding', '节点嵌入'), def: L('A learned vector of numbers for each node of a graph, so that nodes that play alike parts get similar vectors.', '为图中每个节点学出的一个数字向量，使作用相似的节点得到相似的向量。'), related: ['embedding', 'graph'] }
	],
	catalog: {
		shows: L(
			'One biased walk step worked out from p and q, walks as sentences of member ids and their (center, context) pairs, one skip-gram step with real numbers, the PCA map of the member vectors by faction, your own p and q against the start setting, nearest members, the GCN’s result next to node2vec’s, and the limits.',
			'根据 p 和 q 算出的一步有偏游走、由成员编号组成句子的游走和它们的（中心，上下文）训练对、带真实数字的一步 skip-gram、按派别着色的成员向量 PCA 图、你自己的 p 和 q 与开始设置的对比、最近的成员、GCN 的结果和 node2vec 的对比，以及局限。'
		),
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		params: L('2 × 34 × 8 = 544 (two tables of one 8-number vector per member)', '2 × 34 × 8 = 544（两张表，每个成员一个 8 个数的向量）'),
		size: L('No weights shipped; trained live (one 0.3 KB facts file)', '不附带权重，实时训练（一个 0.3 KB 的事实文件）')
	},
	credits: [
		{ name: 'Zachary’s karate club (from NetworkX)', url: 'https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html', license: 'NetworkX: BSD-3-Clause; the graph is the published data of Zachary (1977)', usedFor: 'The graph: 34 members, 78 friendships (unweighted) and the side each member joined, used only to colour the map', attribution: 'W. W. Zachary, “An information flow model for conflict and fission in small groups”, Journal of Anthropological Research 33(4), 452–473 (1977); graph from NetworkX karate_club_graph()' },
		{ name: 'Grover & Leskovec, node2vec: Scalable Feature Learning for Networks (KDD 2016)', url: 'https://arxiv.org/abs/1607.00653', license: 'Paper; ideas only (no code used)', usedFor: 'The biased second-order walk with the return parameter p and the in-out parameter q, and the walks-to-skip-gram recipe (here with a full softmax instead of negative sampling)', attribution: 'Aditya Grover and Jure Leskovec' },
		{ name: 'Perozzi, Al-Rfou & Skiena, DeepWalk: Online Learning of Social Representations (KDD 2014)', url: 'https://arxiv.org/abs/1403.6652', license: 'Paper; ideas only (no code used)', usedFor: 'The idea of treating random walks on a graph as sentences for word2vec (node2vec with p = q = 1)', attribution: 'Bryan Perozzi, Rami Al-Rfou and Steven Skiena' },
		{ name: 'word2vec (skip-gram)', url: 'https://arxiv.org/abs/1301.3781', license: 'Paper (ideas only; no text, figures or code used)', usedFor: 'The skip-gram model this page trains on the walks (with a full softmax instead of the paper’s faster approximations)', attribution: 'Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space” (2013)' }
	]
};

export default manifest;
