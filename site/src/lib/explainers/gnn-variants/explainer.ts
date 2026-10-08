// gnn-variants: the package manifest (design §5 catalog row "GAT / GraphSAGE / GIN on the same node"). Kinds A + C:
//   fwd   four exported networks side by side (pipeline/gnn-variants: GCN, GraphSAGE, GIN, GAT; each 34 -> 4 -> 2 graph
//         layers and a 2 -> 2 head, trained the same way on members 0 and 33 only), traced on the learner's graph
//   gat   GAT's attention into the member in focus, one event at a time (process.ts, from the fwd trace)
//   drop  attention is not an explanation: remove one friendship at a time and re-run the GAT (process.ts)
// Measured numbers (accuracy over 10 random starts) are in facts.json (pipeline/gnn-variants/train.py via export.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { ISOLATED, KARATE } from './graph.ts';
import { dropInput, gatInput } from './process.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'gnn-variants',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/gnn-variants', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'graph', start: KARATE },
	output: { view: 'gnn-variants/agg', props: { mode: 'embed' } },
	sources: {
		fwd: { kind: 'trace' },
		gat: { kind: 'process', process: 'gat', input: gatInput },
		drop: { kind: 'process', process: 'drop', input: dropInput, run: 'worker' }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			I('Name the three moves of a graph layer (gather, combine, update) and point to them in GCN, GraphSAGE, GIN and GAT.', '说出图网络层的三个动作（收集、合并、更新），并在 GCN、GraphSAGE、GIN 和 GAT 里分别指出它们。'),
			I('Give the weight each layer puts on a friend’s message: 1 / √(dᵢ dⱼ), 1 / k, 1 (with 1 + ε for the member itself) or a learned α.', '说出每种图网络层给一个朋友的消息多大权重：1 / √(dᵢ dⱼ)、1 / k、1（成员自己是 1 + ε），或者学来的 α。'),
			I('Work out one GAT attention weight from the scores: LeakyReLU, then a softmax over the member’s friends and itself.', '从分数算出 GAT 的一个注意力权重：先 LeakyReLU，再对成员的朋友和他自己做 softmax。'),
			I('Explain with an example why a sum can tell apart two groups of friends that an average cannot.', '举例说明为什么求和能区分两组朋友，而取平均却不能。'),
			I('Compare the four networks’ accuracy honestly (one run against many random starts) and test whether an attention weight explains an answer by removing the friendship it sits on.', '诚实地比较四个网络的准确率（一次运行和多次随机初始化的区别），并通过去掉一段朋友关系，检验它上面的注意力权重能不能解释答案。')
		],
		prequestion: {
			id: 'gnn-variants-pre',
			question: I('Four graph networks learn the two factions of a 34-member karate club from the same two labelled members: a plain GCN, GraphSAGE, GIN (built to tell more graphs apart) and GAT (with attention). Which sorts the other 32 members best on average?', '四个图网络都只从同样两个已标注的成员出发，去学一个 34 人空手道俱乐部的两派：普通的 GCN、GraphSAGE、GIN（为了区分更多的图而设计）和 GAT（带注意力）。平均来说，哪个能把其余 32 人分得最好？'),
			options: [I('GAT: attention picks the friends that matter', 'GAT：注意力会挑出要紧的朋友'), I('GIN: the most expressive layer wins', 'GIN：表达能力最强的赢'), I('The plain GCN', '普通的 GCN'), I('All four score the same', '四个一样好')],
			answer: 2,
			explain: I('On this small club with two labels, the plain GCN was the most accurate on average and the steadiest over random starts. More flexible layers have more ways to fit two labels without using the friendships well.', '在这个只有两个标签的小俱乐部上，普通的 GCN 平均最准，在不同的随机初始化之间也最稳定。更灵活的图网络层有更多办法去拟合那两个标签，却不一定好好利用朋友关系。')
		},
		tryThis: [ { id: 'alone', title: I('Member 11 alone', '成员 11 落单'), setup: { input: ISOLATED }, predict: I('Member 11 loses its only friend. What does each of the four layers combine for it now?', '成员 11 失去了唯一的朋友。现在四种图网络层分别为他合并了什么？'), look: I('GCN and GAT keep only the self-loop, with weight 1; GraphSAGE’s average of no friends is all zeros; GIN’s sum is just 1 + ε times its own row.', 'GCN 和 GAT 只剩自环，权重为 1；GraphSAGE 对零个朋友取平均，得到全 0；GIN 的和只是他自己的行乘以 1 + ε。'), goto: { step: 'side' } } ],
		quiz: [ { q: I('Member i has 4 friends. What weight does each friend’s row get in GraphSAGE’s average, and in GIN’s sum?', '成员 i 有 4 个朋友。在 GraphSAGE 的平均里，每个朋友的行权重是多少？在 GIN 的求和里呢？'), options: [I('1/4 and 1/4', '1/4 和 1/4'), I('1/4 and 1', '1/4 和 1'), I('1 and 1/4', '1 和 1/4')], answer: 1, explain: I('An average divides by the number of friends; a sum does not, so GIN keeps the count.', '平均要除以朋友的个数；求和不除，所以 GIN 记得有几个朋友。') } ],
		misconception: { claim: I('The friend with the largest attention weight is the friend the answer depends on most.', '注意力权重最大的朋友，就是答案最依赖的朋友。'), truth: I('Not always. In Zachary’s club, member 8 gives its largest layer-1 weight to member 0, yet removing that friendship barely moves its score, while removing the friendship with member 32, which gets little layer-1 weight, switches its side. Layer 2’s weights do favour member 32: a weight is a clue about one layer, and removing the input is the test.', '不一定。在 Zachary 的俱乐部里，成员 8 把第 1 层最大的权重给了成员 0，可去掉这段朋友关系，他的分数几乎不变；而和成员 32 的朋友关系在第 1 层只得到很小的权重，去掉它却让成员 8 换到了另一派。第 2 层的权重确实偏向成员 32：一个权重只是关于某一层的线索，去掉这个输入才是检验。'), step: 'limits' },
		lessonNote: I('Eleven steps: one member and its friends, what every graph layer does, GraphSAGE’s average, GIN’s sum, GAT’s scores, softmax and weighted sum with real numbers, the four rules side by side, their embeddings and accuracy, and why an attention weight is a clue to test, not an explanation.', '十一步：一个成员和他的朋友，每个图网络层都在做什么，GraphSAGE 的平均，GIN 的求和，带真实数字的 GAT 打分、softmax 和加权和，四种规则并排比较，它们的嵌入和准确率，以及为什么注意力权重只是需要检验的线索，而不是解释。'),
		sections: { intros: { output: I('Every member at the two numbers of each network’s last graph layer, coloured by the faction it joined; the dashed line is where that network’s head scores the two factions the same.', '每个成员放在每个网络最后一个图网络层给出的两个数的位置上，颜色表示他加入的派别；虚线是这个网络的输出头给两派打分相同的地方。') } 		}
	},
	glossary: [
		{ id: 'graph', term: I('graph (nodes and edges)', '图（节点和边）'), def: I('A set of things (nodes) and the pairs of them that are linked (edges), such as members and their friendships.', '一组事物（节点）以及它们之间有联系的成对关系（边），例如成员和他们的朋友关系。') },
		{ id: 'neighbour', term: I('neighbour', '邻居'), def: I('A node linked to a given node by an edge: here, a member’s friends.', '和某个节点之间有一条边的节点：这里就是一个成员的朋友。'), related: ['graph'] },
		{ id: 'degree', term: I('degree', '度'), def: I('How many edges a node has. GCN’s d also counts the self-loop: friends + 1.', '一个节点有多少条边。GCN 里的 d 还把自环算进去：朋友数 + 1。'), related: ['graph'] },
		{ id: 'one-hot', term: I('one-hot vector', '独热向量'), def: I('A row of 0s with a single 1 that says which item it is.', '一行 0 里只有一个 1，用来标明它是哪一个。') },
		{ id: 'message-passing', term: I('message passing', '消息传递'), def: I('Each node sends numbers along its edges, and each node combines the numbers it receives into its new row.', '每个节点沿着边发出一些数，每个节点把收到的数合并成自己的新行。'), related: ['graph'] },
		{ id: 'aggregation', term: I('aggregation', '聚合'), def: I('The rule that combines the messages a node receives into one: a weighted sum, an average, a plain sum or attention.', '把一个节点收到的消息合成一份的规则：加权和、平均、直接求和或注意力。'), related: ['message-passing'] },
		{ id: 'concatenation', term: I('concatenation', '拼接'), def: I('Placing two rows of numbers end to end to make one longer row.', '把两行数首尾相接，拼成一行更长的数。') },
		{ id: 'leaky-relu', term: I('LeakyReLU', 'LeakyReLU'), def: I('Like ReLU, but a negative number is multiplied by a small slope (here 0.2) instead of becoming 0.', '和 ReLU 类似，但负数不变成 0，而是乘以一个小斜率（这里是 0.2）。'), related: ['relu'] },
		{ id: 'semi-supervised', term: I('semi-supervised learning', '半监督学习'), def: I('Learning from a few labelled examples plus many unlabelled ones, here linked by a graph.', '从少量有标签的样本和大量没有标签的样本中学习，这里它们由一张图连在一起。'), related: ['label'] }
	],
	catalog: {
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（与 PyTorch 核对）'),
		params: I('808 in four networks (GCN 156, GraphSAGE 300, GIN 184, GAT 168)', '四个网络共 808 个（GCN 156、GraphSAGE 300、GIN 184、GAT 168）'),
		size: I('3.2 KB of weights', '权重 3.2 KB'),
		shows: I('One member’s friends weighted four ways (GCN, GraphSAGE, GIN, GAT attention per edge), the four embeddings and accuracies, and why an attention weight is a clue, not an explanation.', '一个成员的朋友被四种方式加权（GCN、GraphSAGE、GIN、每条边的 GAT 注意力），四种嵌入和准确率，以及为什么注意力权重是线索而不是解释。')
	},
	credits: [
		{ name: 'Zachary’s karate club (from NetworkX)', url: 'https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html', license: 'NetworkX: BSD-3-Clause; the graph is the published data of Zachary (1977)', usedFor: 'The graph: 34 members, 78 friendships (unweighted) and the faction each member joined', attribution: 'W. W. Zachary, “An information flow model for conflict and fission in small groups”, Journal of Anthropological Research 33(4), 452–473 (1977); graph from NetworkX karate_club_graph()' },
		{ name: 'Hamilton, Ying & Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)', url: 'https://arxiv.org/abs/1706.02216', license: 'Paper; ideas only (no code used)', usedFor: 'The GraphSAGE layer with the mean aggregator and concatenation', attribution: 'William L. Hamilton, Rex Ying and Jure Leskovec' },
		{ name: 'Xu, Hu, Leskovec & Jegelka, How Powerful are Graph Neural Networks? (ICLR 2019)', url: 'https://arxiv.org/abs/1810.00826', license: 'Paper; ideas only (no code used)', usedFor: 'The GIN layer (sum, 1 + ε, MLP) and why a sum tells apart more neighbourhoods than a mean', attribution: 'Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka' },
		{ name: 'Veličković et al., Graph Attention Networks (ICLR 2018)', url: 'https://arxiv.org/abs/1710.10903', license: 'Paper; ideas only (no code used)', usedFor: 'The GAT layer: LeakyReLU scores and a softmax over the neighbours (one head, as PyTorch Geometric’s GATConv)', attribution: 'Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio' },
		{ name: 'Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)', url: 'https://arxiv.org/abs/1609.02907', license: 'Paper; ideas only (no code used)', usedFor: 'The GCN layer Â (H Wᵀ) + b and the two-label semi-supervised setup', attribution: 'Thomas N. Kipf and Max Welling' },
		{ name: 'Jain & Wallace, Attention is not Explanation (NAACL 2019)', url: 'https://arxiv.org/abs/1902.10186', license: 'Paper; ideas only (no code used)', usedFor: 'The limits step: attention weights compared with the effect of removing an input', attribution: 'Sarthak Jain and Byron C. Wallace' }
	]
};

export default manifest;
