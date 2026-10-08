// gcn: the package manifest (design §7). A graph convolutional network on Zachary's karate club, kinds A + B + C:
//   fwd     the exported GCN (pipeline/gcn: 34 -> 4 -> 2 graph layers, 2 -> 2 head; trained on members 0 and 33 only)
//           traced on the learner's graph: edit a friendship and every layer re-runs
//   mp      message passing into the member in focus, layer 1, one event at a time (process.ts; from the fwd trace)
//   train   the same network retrained live from the members the learner labels (trainer.ts; twin: pipeline/gcn/twin.py)
//   smooth  over-smoothing: the one-hot rows averaged with Â again and again (process.ts)
// Measured numbers (the 8-layer comparison, the training run) are in facts.json (pipeline/gcn/train.py via export.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { KARATE } from './graph.ts';
import { ISOLATED } from './adapters/graph/adapter.ts';
import { mpInput } from './process.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'gcn',
	type: 'model',
	kinds: ['trace', 'train', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/gcn', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'graph', start: KARATE },
	output: { view: 'gcn/graph', props: { mode: 'embed' } },
	sources: {
		fwd: { kind: 'trace' },
		mp: { kind: 'process', process: 'mp', input: mpInput },
		train: { kind: 'train', trainer: 'gcn', detail: 'coarse', opt: { seed: 1 } },
		smooth: { kind: 'process', process: 'smooth', opt: { rounds: 8 } }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			I('Find a member’s friends in the drawing of the club and in its row of the adjacency matrix A.', '在俱乐部的图里、也在邻接矩阵 A 的对应行里找出一个成员的朋友。'),
			I('Work out one entry of Â = D^−1/2 (A + I) D^−1/2 from two degrees, and say why a popular friend’s message counts less.', '用两个度算出 Â = D^−1/2 (A + I) D^−1/2 的一个元素，并说出为什么朋友很多的人发来的消息分量更小。'),
			I('Follow one member through a graph layer: the messages Âᵢⱼ × (X Wᵀ)ⱼ, their sum, the bias and tanh.', '跟着一个成员走完一个图卷积层：消息 Âᵢⱼ × (X Wᵀ)ⱼ、它们的和、偏置和 tanh。'),
			I('Explain how two labelled members are enough to sort most of the others, and predict what changes with other labels.', '解释为什么两个已标注的成员就足以把其余大多数人分对，并预测换成别的标注会怎样。'),
			I('Say what over-smoothing is and why graph networks usually have few layers.', '说出什么是过度平滑，以及为什么图网络通常只有几层。')
		],
		prequestion: {
			id: 'gcn-pre',
			question: I('A karate club of 34 members split into two factions. A network is told the side of only two members, the two leaders, and sees nothing else about the others except who is friends with whom. How many of the other 32 will it put on the right side?', '一个 34 人的空手道俱乐部分裂成两派。网络只被告知两个成员（两派的领头人）属于哪一派，关于其他人，它只知道谁和谁是朋友。其余 32 人中，它能把多少人分到正确的一派？'),
			options: [I('About half: without labels it can only guess', '大约一半：没有标签它只能猜'), I('Most of them: friends tend to be on the same side', '大多数：朋友往往在同一派'), I('None: it needs a label for every member', '一个也不行：它需要每个成员的标签')],
			answer: 1,
			explain: I('Friends tend to join the same side, and every graph layer mixes a member’s numbers with its friends’. So the two labels spread through the friendships: here the trained network gets all but one member right.', '朋友往往加入同一派，而每个图卷积层都会把一个成员的数和它朋友的数混在一起。于是两个标签顺着朋友关系传开：这里训练好的网络只分错了一个人。')
		},
		tryThis: [
			{
				id: 'alone',
				title: I('Member 11 alone', '成员 11 落单'),
				setup: { input: ISOLATED },
				predict: I('Member 11 loses his only friend. Whose messages will reach him now?', '成员 11 失去了唯一的朋友。现在谁的消息能传到他那里？'),
				look: I('Only his own: with no edges, his self-loop is the only message, with weight 1.', '只有他自己的：没有边时，自环是唯一的消息，权重是 1。'),
				goto: { step: 'messages' }
			}
		],
		quiz: [
			{
				q: I('Member i has 3 friends and member j has 8. They are friends. What is Âᵢⱼ?', '成员 i 有 3 个朋友，成员 j 有 8 个，他们俩是朋友。Âᵢⱼ 是多少？'),
				options: [I('1 / √(3 × 8)', '1 / √(3 × 8)'), I('1 / √(4 × 9) = 1/6', '1 / √(4 × 9) = 1/6'), I('1 / 11', '1 / 11')],
				answer: 1,
				explain: I('Each degree counts the friends plus the self-loop: 4 and 9, so Âᵢⱼ = 1 / √36 = 1/6.', '每个度都是朋友数加上自环：4 和 9，所以 Âᵢⱼ = 1 / √36 = 1/6。')
			}
		],
		misconception: {
			claim: I('A network needs a label for every member to learn which side it is on.', '网络必须知道每个成员的标签，才能学会他属于哪一派。'),
			truth: I('Here only two members are labelled; the friendships carry the information to the rest.', '这里只有两个成员有标签；朋友关系把信息带给了其余的人。'),
			step: 'embed'
		},
		lessonNote: I('Eleven steps: the club as a graph and a table, one member followed through a graph layer with real numbers, two layers, the factions in two numbers, then you pick the labelled members and the network retrains, and why more layers blur.', '十一步：把俱乐部看成图和表，带着真实数字跟一个成员走完一个图卷积层，再到两层，两个数里的两派；然后由你挑选已标注的成员让网络重新训练，最后看为什么层数多了会变模糊。'),
		sections: {
			intros: {
			training: I('Retrain the same 156-parameter network in your browser from the members labelled in your input, as in lesson step 10. The loss is the cross-entropy of the labelled members only.', '在你的浏览器里，用输入中已标注的成员重新训练同一个 156 参数的网络，和课程第 10 步一样。损失只计算已标注成员的交叉熵。'),
			output: I('Every member at the two numbers of the last graph layer, coloured by the faction it joined; the dashed line is where the head’s two scores tie.', '每个成员放在最后一个图卷积层给出的两个数的位置上，颜色表示它加入的派别；虚线是输出头两个分数相等的地方。')
			}
		}
	},
	glossary: [
		{ id: 'graph', term: I('graph (nodes and edges)', '图（节点和边）'), def: I('A set of things (nodes) and the pairs of them that are linked (edges), such as members and their friendships.', '一组事物（节点）以及它们之间有联系的成对关系（边），例如成员和他们的朋友关系。') },
		{ id: 'adjacency-matrix', term: I('adjacency matrix', '邻接矩阵'), def: I('A table with a row and a column per node and a 1 where two nodes are linked.', '每个节点占一行一列的表，两个节点相连的地方是 1。'), related: ['graph'] },
		{ id: 'degree', term: I('degree', '度'), def: I('How many edges a node has. In a GCN, d also counts the self-loop: friends + 1.', '一个节点有多少条边。在 GCN 里，d 还把自环算进去：朋友数 + 1。'), related: ['graph'] },
		{ id: 'one-hot', term: I('one-hot vector', '独热向量'), def: I('A row of 0s with a single 1 that says which item it is.', '一行 0 里只有一个 1，用来标明它是哪一个。') },
		{ id: 'message-passing', term: I('message passing', '消息传递'), def: I('Each node sends numbers along its edges, and each node combines the numbers it receives.', '每个节点沿着边发出一些数，每个节点把收到的数合在一起。'), related: ['graph'] },
		{ id: 'semi-supervised', term: I('semi-supervised learning', '半监督学习'), def: I('Learning from a few labelled examples plus many unlabelled ones, here linked by a graph.', '从少量有标签的样本和大量没有标签的样本中学习，这里它们由一张图连在一起。'), related: ['label'] },
		{ id: 'over-smoothing', term: I('over-smoothing', '过度平滑'), def: I('After many rounds of averaging with neighbours, all nodes end up with nearly the same numbers.', '和邻居平均很多轮之后，所有节点的数都变得几乎一样。'), related: ['message-passing'] }
	],
	catalog: {
		runtime: I('Own JS forward pass (checked against PyTorch) and live retraining', '自研 JS 前向计算（与 PyTorch 核对）和实时重新训练'),
		params: I('156 (34 → 4 → 2 graph layers, 2 → 2 head)', '156（34 → 4 → 2 图卷积层，2 → 2 输出头）'),
		size: I('0.6 KB of weights', '权重 0.6 KB')
	},
	credits: [
		{ name: 'Zachary’s karate club (from NetworkX)', url: 'https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html', license: 'NetworkX: BSD-3-Clause; the graph is the published data of Zachary (1977)', usedFor: 'The graph: 34 members, 78 friendships (unweighted) and the faction each member joined', attribution: 'W. W. Zachary, “An information flow model for conflict and fission in small groups”, Journal of Anthropological Research 33(4), 452–473 (1977); graph from NetworkX karate_club_graph()' },
		{ name: 'Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)', url: 'https://arxiv.org/abs/1609.02907', license: 'Paper; ideas only (no code used)', usedFor: 'The graph layer Â (H Wᵀ) + b with Â = D^−1/2 (A + I) D^−1/2, the two-layer semi-supervised setup, and the limits step’s remark that two or three layers work best (their appendix B on model depth)', attribution: 'Thomas N. Kipf and Max Welling' },
		{ name: 'Li, Han & Wu, Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning (AAAI 2018)', url: 'https://arxiv.org/abs/1801.07606', license: 'Paper; ideas only (no code used)', usedFor: 'The over-smoothing picture of the limits step: averaging with Â again and again makes the members’ rows alike', attribution: 'Qimai Li, Zhichao Han and Xiao-Ming Wu' }
	]
};

export default manifest;
