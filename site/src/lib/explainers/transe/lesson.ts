// transe: the guided lesson. The facts, entities as points and relations as arrows, a corrupted copy and the margin
// (source `one`, step 0), one SGD step, training and the arrows (`run`), every candidate ranked (process `rank`), a
// question no start fact answers (`base`: the start facts whatever the learner's), the learner's own fact against the
// start facts, and the one-to-many limit. Every
// number is a {placeholder} from the frames that ran; the other sentences are tested in claims.test.ts.
import type { At, ExplainerRuntime, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { added, CAPITAL_OF, CONTINENT, ENT, factText, hasFact, id, inputOf, LOCATED_IN, nameOf, NE, ofType, PT_FACT, REL, SPEAKS, START, type Fact } from './kg.ts';
import { dist, lastVecs, pca2, ranked, spread, vecsAt, worked, type Vecs } from './space.ts';
import { GAMMA } from './views/step/meta.ts';
import { runFacts } from './trainer.ts';

type V = Record<string, TmplVal>;
const L = (en: string, zh: string) => ({ en, zh });
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const G = (rt: ExplainerRuntime) => inputOf(rt.input);
const space = (props: Record<string, unknown>) => ({ key: 'transe/space', props: () => props });
const [PT, BR, CA] = [id('Portugal'), id('Brazil'), id('Canada')];
const PT_Q = [PT, LOCATED_IN];
/** the nearest entity of a type to h + r, and its distance */
const top = (v: Vecs, h: number, r: number, kind: number) => { const c = ranked(v, h, r).filter((e) => ENT[e.x].type === kind)[0]; return { name: nameOf(c.x), d: c.d, x: c.x }; };
/** mean ‖h + r − t‖ over the facts of a relation, and over the same heads with a random other entity of the tails' type */
function fit(v: Vecs, fs: Fact[], r: number) {
	const own = fs.filter((f) => f[1] === r), other = own.flatMap(([h, , t]) => ofType(ENT[t].type).flatMap((x) => (x === t ? [] : [dist(v, h, r, x)])));
	return { fact: own.reduce((s, [h, , t]) => s + dist(v, h, r, t), 0) / Math.max(1, own.length), other: other.reduce((s, d) => s + d, 0) / Math.max(1, other.length), n: own.length };
}

const lesson: LessonDef = {
	title: L('TransE: facts as arrows between points', 'TransE：把事实变成点之间的箭头'),
	summary: L(
		'A small knowledge graph of countries, capitals, continents and languages; TransE learns a point for every entity and an arrow for every relation so that head + relation lands near tail, ranks every entity to answer a question, guesses where Portugal is, and shows why one-to-many relations are hard.',
		'一个由国家、首都、大洲和语言组成的小知识图谱；TransE 为每个实体学一个点、为每个关系学一支箭头，让头实体 + 关系落在尾实体附近；它给所有实体排名来回答问题，猜葡萄牙在哪里，并说明一对多的关系为什么难。'
	),
	steps: [
		{
			id: 'kg',
			source: 'run',
			at: 'first',
			view: { key: 'transe/kg', props: () => ({ mark: [id('Paris'), id('France')] }) },
			provenance: 'live',
			title: L('Facts as triples', '把事实写成三元组'),
			narration: {
				en: 'A knowledge graph stores facts as triples (head, relation, tail), such as (Paris, capital of, France). This one has {ne} entities of four types and {nf} facts in {nr} relations: {nc} “capital of”, {nl} “located in” and {ns} “speaks”. {pt}',
				zh: '知识图谱把事实存成三元组（头实体，关系，尾实体），比如（巴黎，首都所属，法国）。这个图谱有 {ne} 个实体，分四种类型，还有 {nf} 个事实，分属 {nr} 种关系：“首都所属” {nc} 个，“位于” {nl} 个，“讲” {ns} 个。{pt}',
				vars: (rt): V => {
					const fs = G(rt).facts, n = (r: number) => fs.filter((f) => f[1] === r).length;
					return { ne: NE, nf: fs.length, nr: REL.length, nc: n(CAPITAL_OF), nl: n(LOCATED_IN), ns: n(SPEAKS), pt: pick(hasFact(fs, PT_FACT), L('You have added which continent Portugal is in.', '你已经加上了葡萄牙位于哪个大洲。'), L('One fact is left out on purpose: no fact says which continent Portugal is in.', '有一个事实是故意漏掉的：没有任何事实说葡萄牙位于哪个大洲。')) };
				}
			},
			look: L('Each arrow runs from a head to a tail, coloured by its relation (the names are under the drawing); Paris and France are underlined.', '每条箭头从头实体指向尾实体，颜色代表关系（名称在图下方）；巴黎和法国加了下划线。'),
			terms: ['knowledge-graph', 'graph']
		},
		{
			id: 'points',
			source: 'run',
			at: 'first',
			view: space({ query: [id('Paris'), CAPITAL_OF], kind: 0 }),
			provenance: 'live',
			title: L('Entities are points, relations are arrows', '实体是点，关系是箭头'),
			narration: {
				en: 'TransE gives every entity a point of {dim} numbers and every relation an arrow of {dim} numbers. A fact (h, r, t) is plausible when h + r lands near t, so its score is the distance d = ‖h + r − t‖: the smaller, the better. Before training the points are random: Paris + “capital of” lands {d} from France, and the nearest country is {top}.',
				zh: 'TransE 给每个实体一个由 {dim} 个数组成的点，给每个关系一支由 {dim} 个数组成的箭头。h + r 落在 t 附近，事实（h，r，t）就可信，所以它的分数是距离 d = ‖h + r − t‖：越小越好。训练之前这些点是随机的：巴黎 + “首都所属” 落在离法国 {d} 远的地方，最近的国家是{top}。',
				vars: (rt, at): V => {
					const v = vecsAt(at.state);
					return v.E.length ? { dim: v.R[0].length, d: dist(v, id('Paris'), CAPITAL_OF, id('France')), top: top(v, id('Paris'), CAPITAL_OF, 0).name } : { dim: NaN };
				}
			},
			look: L('The ✕ is where Paris + “capital of” lands; the table ranks the countries by their distance in all 8 numbers: the five nearest, and France.', '✕ 是巴黎 + “首都所属” 的落点；表格按全部 8 个数算出的距离给国家排名：最近的五个，再加上法国。'),
			terms: ['transe', 'embedding']
		},
		{
			id: 'margin',
			source: 'one',
			at: { phase: 'update', step: 0 },
			view: { key: 'transe/step', props: () => ({ mode: 'margin' }) },
			provenance: 'live',
			title: L('A corrupted copy and a margin', '损坏副本和间隔'),
			narration: {
				en: 'Training also needs false facts. The first step (step 0) takes the fact {fact} and corrupts it by swapping its {side} for a random entity: {bad}. The margin ranking loss is max(0, γ + d(fact) − d(copy)) with the margin γ = {gamma}: here max(0, {gamma} + {dp} − {dn}) = {loss}.',
				zh: '训练还需要假的事实。第一步（第 0 步）取出事实{fact}，把它的{side}换成一个随机实体，得到损坏副本{bad}。间隔排序损失是 max(0, γ + d(事实) − d(副本))，间隔 γ = {gamma}：这里是 max(0, {gamma} + {dp} − {dn}) = {loss}。',
				vars: (_rt, at): V => {
					const w = worked(at, GAMMA);
					return w ? { fact: factText([w.h, w.r, w.t]), bad: factText([w.h2, w.r, w.t2]), side: w.tail ? L('tail', '尾实体') : L('head', '头实体'), gamma: String(w.gamma), dp: w.dPos, dn: w.dNeg, loss: w.loss } : { fact: NaN };
				}
			},
			look: L('First bar: the fact’s distance; second: the copy’s. The dashed line is d(fact) + γ; the outlined stretch, where the copy falls short of it, is the loss.', '第一条是事实的距离，第二条是副本的距离。虚线是 d(事实) + γ；副本没够到虚线的那一段（加了框）就是损失。'),
			predict: {
				id: 'transe-margin',
				question: L('Suppose the copy were already more than γ farther away than the fact. What would this training step change?', '假如副本已经比事实远出不止 γ。这一步训练会改变什么？'),
				options: [L('Nothing: the loss and every gradient are 0', '什么都不变：损失和所有梯度都是 0'), L('It still pulls the fact a little closer', '它仍然会把事实再拉近一点'), L('It pushes the copy even farther away', '它会把副本推得更远')],
				answer: 0,
				explain: L(
					'Once d(copy) ≥ d(fact) + γ, γ + d(fact) − d(copy) is at most 0, so max(0, …) is 0 and so is its gradient: the step moves nothing. Only copies that sit too close teach anything, and the margin γ says how far is far enough.',
					'一旦 d(副本) ≥ d(事实) + γ，γ + d(事实) − d(副本) 就不超过 0，所以 max(0, …) 是 0，它的梯度也是 0：这一步什么都不动。只有离得太近的副本才能教会模型东西，间隔 γ 规定了多远才算够远。'
				)
			},
			terms: ['corrupted-triple', 'margin-ranking-loss']
		},
		{
			id: 'update',
			source: 'one',
			at: { phase: 'update', step: 0 },
			view: { key: 'transe/step', props: () => ({ mode: 'update' }) },
			provenance: 'live',
			title: L('One SGD step, with its numbers', '带着数字走一步 SGD'),
			narration: {
				en: 'The gradient of a distance points along h + r − t, so one SGD step (learning rate {lr}) shifts h and r so that h + r moves toward t, shifts t toward h + r, and moves the copy’s points the opposite way. Number k = {k} (counting from 0) of {hn}’s point goes {old} − {lr} × ({g}) = {new}. The fact’s distance falls from {dp} to {dp1}, the copy’s rises from {dn} to {dn1}, and the loss drops by {drop}.',
				zh: '距离的梯度沿着 h + r − t 的方向，所以一步 SGD（学习率 {lr}）挪动 h 和 r，让 h + r 靠近 t，把 t 往 h + r 挪一点，副本的点则往相反方向挪。{hn}的点里编号 k = {k}（从 0 数起）的数变成 {old} − {lr} × ({g}) = {new}。事实的距离从 {dp} 降到 {dp1}，副本的从 {dn} 升到 {dn1}，损失下降了 {drop}。',
				vars: (_rt, at): V => {
					const w = worked(at, GAMMA);
					return w ? { lr: String(w.lr), k: w.k, hn: nameOf(w.h), old: w.old, g: w.g, new: w.new, dp: w.dPos, dp1: w.dPos1, dn: w.dNeg, dn1: w.dNeg1, drop: w.loss - w.loss1 } : { lr: NaN };
				}
			},
			look: L('The table has one row per number (k counts from 0): h, r, t and the swapped-in entity x before the step, the gradient of h and h after it; the outlined row is worked out in the text.', '表格每行一个数（k 从 0 数起）：这一步之前的 h、r、t 和换进来的实体 x，h 的梯度，以及这一步之后的 h；加框的那一行就是正文里算的那个数。'),
			terms: ['sgd', 'gradient']
		},
		{
			id: 'train',
			source: 'run',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metric: 'avgLoss', labels: { avgLoss: L('average loss over every fact and copy', '所有事实和副本上的平均损失') } }) },
			provenance: 'live',
			title: L('Repeat it thousands of times', '重复几千次'),
			narration: {
				en: 'Repeat that step {steps} times, each time on one fact drawn at random with a fresh corrupted copy. The loss averaged over every fact and every possible copy falls from {l0} to {l1}. The share of facts whose tail is the nearest entity to h + r (other true tails left out) rises from {h0|pct} to {h1|pct}.',
				zh: '把这一步重复 {steps} 次，每次随机抽一个事实，再配一个新的损坏副本。在所有事实和所有可能的副本上平均的损失从 {l0} 降到 {l1}。有多少事实的尾实体正好是离 h + r 最近的实体（其他正确的尾实体不算在内）？这个比例从 {h0|pct} 升到 {h1|pct}。',
				vars: (_rt, at): V => (at.tape ? { steps: m(at, 'step'), l0: at.tape.frame(0).metrics?.avgLoss ?? NaN, l1: m(at, 'avgLoss'), h0: at.tape.frame(0).metrics?.hits1 ?? NaN, h1: m(at, 'hits1') } : { steps: NaN })
			},
			look: L('The curve is the average loss after each block of steps; the readout gives its lowest value.', '曲线是每隔一段步数的平均损失；读数给出它的最低值。'),
			terms: ['loss', 'link-prediction']
		},
		{
			id: 'arrows',
			source: 'run',
			at: 'last',
			view: space({ rel: CAPITAL_OF, animate: true }),
			play: 'view',
			linger: 12000,
			provenance: 'live',
			title: L('One arrow for every capital', '每个首都用同一支箭头'),
			narration: {
				en: 'To draw points of {dim} numbers flat, PCA keeps the two directions they spread most in ({share|pct} of the spread); one relation’s arrow is drawn the same wherever it starts. Every capital gets the same “capital of” arrow, and it lands near its own country: {dcap} away on average, in all {dim} numbers, against {drand} from the other countries.',
				zh: '为了把 {dim} 个数的点画平，PCA 保留它们分布最开的两个方向（保留了 {share|pct} 的分布）；同一个关系的箭头不管从哪里出发，在图上都画得一样。每个首都用的都是同一支“首都所属”箭头，它落在自己国家附近：按全部 {dim} 个数算，平均相距 {dcap}，而离其他国家平均 {drand}。',
				vars: (rt, at): V => {
					const v = vecsAt(at.state);
					if (!v.E.length) return { dim: NaN };
					const f = fit(v, G(rt).facts, CAPITAL_OF);
					return { dim: v.R[0].length, share: pca2(v.E).share, dcap: f.n ? f.fact : NaN, drand: f.n ? f.other : NaN };
				}
			},
			look: L('Each arrow starts at a capital: follow it to its country. Replay the training to watch the points find their places.', '每支箭头从一个首都出发：顺着它找到它的国家。重放训练过程，看这些点怎样各就各位。'),
			terms: ['pca']
		},
		{
			id: 'rank',
			source: 'rank',
			at: 'last',
			view: 'transe/rank',
			play: { from: { phase: 'score', step: 0 }, to: 'last', interval: 220, unit: L('entity id {n}', '实体编号 {n}') },
			provenance: 'live',
			title: L('Answer a question: rank every entity', '回答问题：给每个实体排名'),
			narration: {
				en: 'To answer {qh} + “{qr}” → ?, TransE tries each of the {ne} entities as the tail and ranks them by distance: no lookup, and no idea of types, just the nearest points. The nearest is {t1} ({d1}), then {t2} ({d2}). {truth}',
				zh: '为了回答 {qh} + “{qr}” → ?，TransE 把 {ne} 个实体逐个当作尾实体试一遍，按距离排名：不是查表，也不管实体的类型，只是找最近的点。最近的是{t1}（{d1}），其次是{t2}（{d2}）。{truth}',
				vars: (rt, at): V => {
					const q = G(rt).query, rk = m(at, 'truthRank');
					return {
						qh: nameOf(q[0]), qr: REL[q[1]], ne: NE, t1: nameOf(m(at, 't1')), d1: m(at, 'd1'), t2: nameOf(m(at, 't2')), d2: m(at, 'd2'),
						truth: rk > 0 ? pick(rk === 1, { en: `The facts’ answer, ${nameOf(m(at, 'truth')).en}, ranks first.`, zh: `事实给出的答案${nameOf(m(at, 'truth')).zh}排在第一。` }, { en: `The facts’ answer, ${nameOf(m(at, 'truth')).en}, has rank ${rk} (other true tails left out).`, zh: `事实给出的答案${nameOf(m(at, 'truth')).zh}排在第 ${rk} 名（不算其他正确的尾实体）。` }) : L('No fact answers this question, yet it still gets a ranking.', '没有任何事实回答这个问题，但它照样得到一个排名。')
					};
				}
			},
			look: L('Play the step to score the candidates one at a time; the list keeps the nearest so far. Ask another question under “Your input”.', '播放这一步，逐个给候选打分；列表保留目前最近的几个。可以在“你的输入”下面换一个问题。'),
			terms: ['link-prediction']
		},
		{
			id: 'portugal',
			source: 'base',
			at: 'last',
			view: space({ query: PT_Q, kind: 2, ring: [BR, id('Portuguese')] }),
			provenance: 'live',
			title: L('A question no fact answers', '一个没有事实回答的问题'),
			narration: {
				en: 'Ask Portugal + “located in” → ?. The start facts never say where Portugal is: it has a capital, Lisbon, and it speaks Portuguese, like Brazil, whose continent the facts do give. Training placed Portugal’s point {dpb} from Brazil’s, against {dc} between two countries on average.',
				zh: '问一问：葡萄牙 + “位于” → ?。起始的事实从没说过葡萄牙在哪里：它有首都里斯本，和巴西一样讲葡萄牙语；而巴西在哪个大洲，事实里是有的。训练把葡萄牙的点放在离巴西 {dpb} 的地方，而两个国家平均相距 {dc}。',
				vars: (_rt, at): V => {
					const v = vecsAt(at.state);
					return v.E.length ? { dpb: spread(v, [PT, BR]), dc: spread(v, ofType(0)) } : { dpb: NaN };
				}
			},
			look: L('Portugal is ringed with its arrow to the ✕; Brazil and Portuguese are ringed too. The table ranks the four continents by distance.', '葡萄牙加了圈，它的箭头指向 ✕；巴西和葡萄牙语也加了圈。表格按距离给四个大洲排了名。'),
			predict: {
				id: 'transe-portugal',
				question: L('With the start facts, which continent will TransE put nearest to Portugal + “located in”?', '用起始的事实，TransE 会让哪个大洲离葡萄牙 + “位于” 最近？'),
				options: [L('Europe', '欧洲'), L('South America', '南美洲'), L('Nothing: without such a fact it cannot answer', '没有答案：没有这样的事实，它答不出来')],
				answer: 1,
				explain: L(
					'South America. Portugal + “speaks” and Brazil + “speaks” must both land on Portuguese, so training pulls the two countries’ points together, and Brazil + “located in” lands near South America. TransE always answers with the nearest point; here the guess is wrong.',
					'南美洲。葡萄牙 + “讲” 和巴西 + “讲” 都要落在葡萄牙语上，所以训练把两国的点拉到一起，而巴西 + “位于” 落在南美洲附近。TransE 总会给出最近的那个点作为答案；这一次它猜错了。'
				)
			},
			terms: ['link-prediction']
		},
		{
			id: 'task',
			source: 'run',
			at: 'last',
			view: space({ query: PT_Q, kind: 2 }),
			compare: { a: 'run', b: 'base', label: [L('Your facts', '你的事实'), L('The start facts', '起始的事实')] },
			provenance: 'live',
			title: L('Your turn: add the missing fact', '轮到你：把漏掉的事实加上'),
			narration: {
				en: 'Add a fact under “Your input” and TransE trains again on your facts. With your facts, the continent nearest to Portugal + “located in” is {top} ({d}); with the start facts, {top0} ({d0}). Portugal is now {dpb} from Brazil, against {dpb0} before: {verdict}',
				zh: '在“你的输入”下面加一个事实，TransE 就用你的事实重新训练。用你的事实，离葡萄牙 + “位于” 最近的大洲是{top}（{d}）；用起始的事实，是{top0}（{d0}）。葡萄牙现在离巴西 {dpb}，之前是 {dpb0}：{verdict}',
				vars: (rt): V => {
					const a = lastVecs(rt.sources.run?.tape), b = lastVecs(rt.sources.base?.tape);
					if (!a.E.length || !b.E.length) return { top: NaN };
					const ta = top(a, PT, LOCATED_IN, 2), tb = top(b, PT, LOCATED_IN, 2), same = !added(rt.input).length && runFacts(rt.input).length === START.facts.length;
					const verdict = same ? L('the same facts, so far the same answer.', '事实还是同一组，所以答案也一样。') : pick(ta.x !== tb.x, L('your facts changed the answer.', '你的事实改变了答案。'), L('the answer stayed, though the distances moved.', '答案没变，但距离变了。'));
					return { top: ta.name, d: ta.d, top0: tb.name, d0: tb.d, dpb: spread(a, [PT, BR]), dpb0: spread(b, [PT, BR]), verdict };
				}
			},
			look: L('First panel: TransE trained on your facts; second: on the start facts. Compare the ✕ and the tables.', '第一个面板：用你的事实训练的 TransE；第二个：用起始的事实。比较 ✕ 和表格。'),
			task: {
				prompt: L('Under “Your input”, add the fact Portugal · located in · Europe (or press “+ Portugal is in Europe”), then compare the two panels.', '在“你的输入”下面加上事实 葡萄牙 · 位于 · 欧洲（或者按“+ 葡萄牙位于欧洲”），然后比较两个面板。'),
				done: (rt) => added(rt.input).length > 0,
				hint: L('The row “Add a fact” under “Your input” has three menus and an Add button.', '“你的输入”下面的“加一个事实”一行有三个菜单和一个“加上”按钮。'),
				doneText: L('Done: the first panel now trains on your facts.', '完成了：第一个面板现在用你的事实训练。')
			},
			linger: 12000,
			terms: ['link-prediction']
		},
		{
			id: 'limits',
			source: 'run',
			at: 'last',
			view: space({ query: [CA, SPEAKS], kind: 3, ring: [id('English'), id('French')] }),
			provenance: 'live',
			kindOf: 'limits',
			title: L('Limits: one head, two tails', '局限：一个头实体，两个尾实体'),
			narration: {
				en: '{canada} that point lands {den} from English and {dfr} from French, and the two languages end {ef} apart, against {avg} for two languages on average. {m2o}: every country of a continent must land on it with the same arrow, so two of them end {within} apart, {closer} closer than two countries of different continents.',
				zh: '{canada}这个点离英语 {den}，离法语 {dfr}，两种语言最后相距 {ef}，而两种语言平均相距 {avg}。{m2o}：一个大洲的每个国家都要用同一支箭头落到这个大洲上，所以同一大洲的两个国家最后平均相距 {within}，比不同大洲的两个国家近 {closer}。',
				vars: (rt, at): V => {
					const v = vecsAt(at.state), both = [id('English'), id('French')].every((l) => hasFact(G(rt).facts, [CA, SPEAKS, l]));
					if (!v.E.length) return { den: NaN };
					const canada = pick(both, L('“Speaks” is one-to-many for Canada: its two tails, English and French, would both have to sit at the one point Canada + “speaks”. Training compromises:', '对加拿大来说，“讲”是一对多：它的两个尾实体英语和法语，都得落在加拿大 + “讲” 这一个点上。训练只能折中：'), L('Your facts no longer say that Canada speaks both English and French, so nothing forces a compromise at the point Canada + “speaks”:', '你的事实里已经没有“加拿大既讲英语又讲法语”，所以加拿大 + “讲” 这个点不需要折中：'));
					const C = ofType(0), pair = (same: boolean) => { let t = 0, n = 0; for (const a of C) for (const b of C) if (a < b && (CONTINENT[a] === CONTINENT[b]) === same) (t += spread(v, [a, b])), n++; return t / n; };
					const m2o = pick(both, L('Many-to-one squeezes too', '多对一也会挤压'), L('Many-to-one still squeezes', '多对一仍然会挤压'));
					return { canada, m2o, den: dist(v, CA, SPEAKS, id('English')), dfr: dist(v, CA, SPEAKS, id('French')), ef: spread(v, [id('English'), id('French')]), avg: spread(v, ofType(3)), within: pair(true), closer: pair(false) - pair(true) };
				}
			},
			look: L('Canada is ringed with its “speaks” arrow; English and French are ringed. The table gives the nearest languages to the ✕.', '加拿大加了圈，带着它的“讲”箭头；英语和法语也加了圈。表格列出离 ✕ 最近的语言。'),
			terms: ['one-to-many']
		}
	],
	fixtures: [
		{ id: 'start', label: L('The start: 33 facts, Paris + capital of', '起始：33 个事实，巴黎 + 首都所属'), raw: START },
		{ id: 'pt', label: L('With Portugal in Europe, asking Portugal + located in', '加上葡萄牙位于欧洲，问葡萄牙 + 位于'), raw: { facts: [...START.facts, PT_FACT], query: [PT, LOCATED_IN] } },
		{ id: 'odd', label: L('A strange fact added: (Tokyo, located in, Europe)', '加了一个奇怪的事实：（东京，位于，欧洲）'), raw: { facts: [...START.facts, [id('Tokyo'), LOCATED_IN, id('Europe')]], query: [id('Tokyo'), LOCATED_IN] } }
	]
};

export default lesson;
