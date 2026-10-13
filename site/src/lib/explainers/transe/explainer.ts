// transe: the manifest. TransE embeddings trained live (kind B) on a small knowledge graph made in kg.ts, and link
// prediction one candidate at a time (kind C). Sources:
//   run   trainer 'transe' on the learner's facts (the map, the arrows, the question, the task, the limits)
//   one   the same run (same seed), fine detail, 1 step: the first step slowed down (the margin and the update)
//   base  trainer 'transe' on the start facts whatever the learner's are: the Portugal question, and the task's comparison
//   rank  process 'rank' on run's trained tables: the learner's question, every candidate scored in turn
// Twin of trainer 'transe': pipeline/transe/twin.py; reference of process 'rank': pipeline/transe/reference.py.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { inputOf, START } from './kg.ts';
import { tailsOf } from './space.ts';
import { rankInput } from './process.ts';

const L = (en: string, zh: string) => ({ en, zh });
/** the seed of the training runs (`run`, `one` and `base` start from the same weights) */
export const SEED = 3;

const manifest: ExplainerManifest = {
	slug: 'transe',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'kg-facts', start: START, presetsFromAdapter: true },
	sources: {
		run: { kind: 'train', trainer: 'transe', detail: 'coarse', opt: { seed: SEED } },
		one: { kind: 'train', trainer: 'transe', detail: 'fine', opt: { seed: SEED, steps: 1 } },
		base: { kind: 'train', trainer: 'transe', detail: 'coarse', opt: { seed: SEED, start: true } },
		rank: { kind: 'process', process: 'rank', after: 'run', input: (rt) => { const { facts, query: [h, r] } = inputOf(rt.input); return rankInput(rt.sources.run?.tape, h, r, tailsOf(facts, h, r)); } }
	},
	page: {
		goals: [
			L('Compute how plausible a fact is for TransE: the distance between head + relation and tail.', '算出 TransE 认为一个事实有多可信：头实体 + 关系 与尾实体之间的距离。'),
			L('Work out the margin ranking loss of a fact and its corrupted copy, and what one SGD step does to both.', '算出一个事实和它的损坏副本的间隔排序损失，以及一步 SGD 对两者做了什么。'),
			L('Answer a question (head, relation, ?) by ranking every entity, and explain where a guessed answer comes from.', '通过给所有实体排名来回答（头实体，关系，?）这样的问题，并解释一个猜出来的答案从哪里来。'),
			L('Say why one-to-many relations are hard for TransE, with numbers from the trained model.', '用训练好的模型里的数字，说明一对多的关系为什么让 TransE 为难。')
		],
		prequestion: {
			id: 'transe-pre',
			question: L('TransE stores Paris, France and the relation “capital of” as lists of numbers. What should Paris + “capital of” give, once it has learned?', 'TransE 把巴黎、法国和关系“首都所属”都存成一串数。学好以后，巴黎 + “首都所属” 应该得到什么？'),
			options: [L('A point close to France', '一个离法国很近的点'), L('A probability that the fact is true', '这个事实为真的概率'), L('The word “France”, looked up in a table', '在表里查到的单词“法国”')],
			answer: 0,
			explain: L(
				'Every entity is a point and every relation an arrow. Training moves them so that, for each known fact, the head plus the arrow lands near the tail: Paris + “capital of” ≈ France. Answering a question means looking for the entity nearest to where the arrow lands.',
				'每个实体是一个点，每个关系是一支箭头。训练会移动它们，让每个已知事实的头实体加上箭头落在尾实体附近：巴黎 + “首都所属” ≈ 法国。回答问题就是找离箭头落点最近的实体。'
			)
		},
		tryThis: [
			{
				id: 'pt',
				title: L('Tell it where Portugal is', '告诉它葡萄牙在哪里'),
				setup: { preset: 'with-pt' },
				predict: L('With the fact (Portugal, located in, Europe) added, which continent will Portugal + “located in” land nearest to?', '加上事实（葡萄牙，位于，欧洲）之后，葡萄牙 + “位于” 会落在哪个大洲附近？'),
				look: L('The task step ranks the four continents for your facts and for the start facts.', '任务那一步分别用你的事实和起始的事实给四个大洲排名。'),
				goto: { step: 'task' }
			}
		],
		quiz: [
			{
				q: L('A fact is at distance 0.3 and its corrupted copy at 0.9; the margin γ is 1. What is the margin ranking loss?', '一个事实的距离是 0.3，它的损坏副本是 0.9；间隔 γ 是 1。间隔排序损失是多少？'),
				options: [L('0', '0'), L('0.4', '0.4'), L('0.6', '0.6')],
				answer: 1,
				explain: L('max(0, γ + d(fact) − d(corrupted)) = max(0, 1 + 0.3 − 0.9) = 0.4: the copy is not yet a full margin farther away than the fact.', 'max(0, γ + d(事实) − d(损坏副本)) = max(0, 1 + 0.3 − 0.9) = 0.4：副本离得还不够远，没比事实多出一个完整的间隔。')
			}
		],
		misconception: {
			claim: L('TransE looks facts up: a question it was never told has no answer.', 'TransE 只会查事实：没告诉过它的问题就没有答案。'),
			truth: L('It always answers, with the entity nearest to head + relation, also when no fact says so; the guess comes from where the other facts pulled the head, which can be wrong.', '它总会回答：离头实体 + 关系最近的那个实体，哪怕没有任何事实这么说；这个猜测来自其他事实把头实体拉到了哪里，可能是错的。'),
			step: 'portugal'
		},
		lessonNote: L(
			'Ten steps: facts as triples, entities as points and relations as arrows, a corrupted copy and the margin, one SGD step with its numbers, training, the arrows on a map, ranking every candidate, a question no fact answers, your own fact, and why one-to-many relations are hard.',
			'十步：把事实写成三元组，实体是点、关系是箭头，损坏副本和间隔，带着数字的一步 SGD，训练，图上的箭头，给每个候选排名，一个没有事实回答的问题，你自己加的事实，以及一对多的关系为什么难。'
		),
		metrics: {
			avgLoss: { label: L('average loss over every fact and corruption', '所有事实和损坏副本上的平均损失'), fmt: 'n3' },
			hits1: { label: L('facts whose tail ranks first', '尾实体排第一的事实'), fmt: 'pct' },
			facts: { label: L('facts trained on', '训练用的事实'), fmt: 'int' }
		},
		trainingCurve: { metric: 'avgLoss', labels: { avgLoss: L('average loss (every fact and corruption)', '平均损失（所有事实和损坏副本）') } },
		sections: { titles: { training: L('Train it yourself', '自己训练') } }
	},
	glossary: [
		{ id: 'knowledge-graph', term: L('knowledge graph', '知识图谱'), def: L('A set of facts written as triples (head, relation, tail), such as (Paris, capital of, France): the entities are the nodes and each fact is a labelled arrow between two of them.', '一组写成三元组（头实体，关系，尾实体）的事实，比如（巴黎，首都所属，法国）：实体是节点，每个事实是两个节点之间一条带标签的箭头。'), related: ['graph'] },
		{ id: 'transe', term: L('TransE (translation embedding)', 'TransE（平移嵌入）'), def: L('A model that gives every entity a point and every relation an arrow, and calls a fact plausible when head + relation lands near tail: the distance ‖h + r − t‖ is small.', '给每个实体一个点、每个关系一支箭头的模型：头实体 + 关系落在尾实体附近，也就是距离 ‖h + r − t‖ 很小时，它就认为这个事实可信。'), related: ['knowledge-graph', 'embedding'] },
		{ id: 'corrupted-triple', term: L('corrupted triple', '损坏三元组'), def: L('A copy of a true fact with its head or its tail swapped for a random entity, used as a probably false example (a negative sample).', '把一个真实事实的头实体或尾实体换成随机实体得到的副本，当作一个多半为假的例子（负样本）。'), related: ['knowledge-graph'] },
		{ id: 'margin-ranking-loss', term: L('margin ranking loss', '间隔排序损失'), def: L('max(0, γ + d(fact) − d(corrupted)): zero once the corrupted copy is at least a margin γ farther away than the fact, positive otherwise.', 'max(0, γ + d(事实) − d(损坏副本))：损坏副本比事实至少远出一个间隔 γ 时为零，否则为正。'), related: ['loss'] },
		{ id: 'link-prediction', term: L('link prediction', '链接预测'), def: L('Answering (head, relation, ?) by scoring every entity as the tail and ranking them; the rank of the true tail measures the model.', '回答（头实体，关系，?）：给每个实体当尾实体打分并排名；真实尾实体的名次就是对模型的衡量。'), related: ['knowledge-graph'] },
		{ id: 'one-to-many', term: L('one-to-many relation', '一对多关系'), def: L('A relation where one head has several tails (Canada speaks English and French); many-to-one is the reverse (many countries located in Europe).', '一个头实体对应好几个尾实体的关系（加拿大讲英语和法语）；多对一则反过来（很多国家位于欧洲）。'), related: ['knowledge-graph'] }
	],
	catalog: {
		shows: L(
			'A small knowledge graph of countries, capitals, continents and languages; one margin-ranking step with a corrupted fact and real numbers; the entities (8 numbers each) and the relation arrows on a 2-D PCA map as TransE trains; every candidate ranked for a question; a question no fact answers before and after you add the fact; and why one-to-many relations are hard.',
			'一个由国家、首都、大洲和语言组成的小知识图谱；带着真实数字、用一个损坏事实算一步间隔排序损失；TransE 训练时，实体（每个 8 个数）和关系箭头在二维 PCA 图上的样子；为一个问题给所有候选排名；一个没有事实回答的问题，在你加上事实前后的答案；以及一对多关系为什么难。'
		),
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		params: L('(34 entities + 3 relations) × 8 = 296 numbers', '（34 个实体 + 3 个关系）× 8 = 296 个数'),
		size: L('No weights shipped; trained live', '不附带权重，实时训练')
	},
	credits: [
		{ name: 'Bordes et al., Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)', url: 'https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html', license: 'Paper; ideas only (no code, text or figures used)', usedFor: 'TransE: entities as points and relations as translations (h + r ≈ t), the L2 distance, the margin ranking loss with one corrupted triple per fact, and link prediction by ranking (here without the paper’s step that rescales entities to length 1)', attribution: 'Antoine Bordes, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston and Oksana Yakhnenko' },
		{ name: 'Wang et al., Knowledge Graph Embedding by Translating on Hyperplanes (AAAI 2014)', url: 'https://doi.org/10.1609/aaai.v28i1.8870', license: 'Paper; ideas only (no code, text or figures used)', usedFor: 'The analysis of why TransE struggles with one-to-many, many-to-one and many-to-many relations, which the limits step shows with live numbers', attribution: 'Zhen Wang, Jianwen Zhang, Jianlin Feng and Zheng Chen (TransH)' }
	]
};

export default manifest;
