// node2vec: the guided lesson. The club (process `walks`), one biased move with its chances, a walk as a sentence, its
// pairs, one skip-gram step (source `one`), the map (`run`), the learner's own p and q against the start setting (`base`),
// nearest members, the GCN link and the limits. Every number is a {placeholder} from the frames that ran (or facts.json);
// the other sentences are tested in claims.test.ts.
import type { At, ExplainerRuntime, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { FACTIONS, N } from '../../atlas/data/karate.ts';
import { exampleWalk, inputOf, LEN, runInput, START, START_P, START_Q, walksOf, WALKS, WIN } from './walks.ts';
import { EXAMPLE, WALK_FIRST, WALK_LAST } from './process.ts';
import { cosine, cutOf, factionStats, lastVecs, pca2, project, vecsAt, worked } from './vectors.ts';

type V = Record<string, TmplVal>;
const L = (en: string, zh: string) => ({ en, zh });
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const G = (rt: ExplainerRuntime) => inputOf(rt.input);
const map = (props: (rt: ExplainerRuntime) => Record<string, unknown> = () => ({})) => ({ key: 'node2vec/map', props: (rt: ExplainerRuntime) => props(rt) });
const walk = (mode: string) => ({ key: 'node2vec/walk', props: () => ({ mode }) });
/** the two leaders: Mr. Hi (member 0) and the Officer (member 33) */
const LEADERS = { hi: 0, off: N - 1 };
const side = (f: number) => (f ? L('the Officer’s', '干事一方') : L('Mr. Hi’s', 'Mr. Hi 一方'));
/** the largest group of members with the same friends (at least one friend); [] when no two members share theirs */
export function twins(raw: unknown): number[] {
	const nb = walksOf(raw).nb, by = new Map<string, number[]>();
	nb.forEach((f, i) => f.length && by.set(f.join(','), [...(by.get(f.join(',')) ?? []), i]));
	return [...by.values()].reduce((b, g) => (g.length > b.length ? g : b), [] as number[]).filter((_, k, g) => g.length > 1);
}
/** the run's walk statistics and how its vectors sit (factionStats), for the task step's two sides */
function sideOf(rt: ExplainerRuntime, source: string) {
	const W = walksOf(runInput(rt.input, source)), st = factionStats(rt.input, lastVecs(rt.sources[source]?.tape));
	return { far: W.far, hop: st?.hop ?? NaN, same: st?.same ?? NaN, ...runInput(rt.input, source) };
}

const lesson: LessonDef = {
	title: L('node2vec: vectors for the members of a club, from random walks', 'node2vec：从随机游走学出俱乐部成员的向量'),
	summary: L(
		'Biased random walks on Zachary’s karate club become sentences of member ids; skip-gram learns one vector per member from them, the two sides of the club fall apart on the map, and p and q decide how far the walks wander.',
		'在 Zachary 空手道俱乐部上做有偏随机游走，得到由成员编号组成的句子；skip-gram 从中为每个成员学一个向量，俱乐部的两方在图上分开，而 p 和 q 决定游走走多远。'
	),
	steps: [
		{
			id: 'club',
			source: 'walks',
			at: 'first',
			view: walk('club'),
			provenance: 'live',
			title: L('A karate club as a graph', '把空手道俱乐部看成一张图'),
			narration: {
				en: 'Zachary’s karate club: {n} members (the dots) and {e} friendships (the lines). After a quarrel the club split: {a} members followed Mr. Hi (member {hi}) and {b} followed the Officer (member {off}). node2vec will see only the friendships; the colours are the answer we check it against at the end.',
				zh: 'Zachary 的空手道俱乐部：{n} 个成员（点）和 {e} 段朋友关系（线）。一次争吵之后俱乐部分裂了：{a} 个成员跟了 Mr. Hi（成员 {hi}），{b} 个跟了干事（成员 {off}）。node2vec 只看得到朋友关系；颜色是我们最后用来检查它的答案。',
				vars: (rt) => ({ n: N, e: G(rt).edges.length, a: FACTIONS.filter((f) => f === 0).length, b: FACTIONS.filter((f) => f === 1).length, ...LEADERS })
			},
			look: L('Blue: Mr. Hi’s side; orange: the Officer’s. The outlined member is the member in focus (change it under “Your input”).', '蓝色：Mr. Hi 一方；橙色：干事一方。加粗边框的是关注的成员（在“你的输入”下面可以换）。'),
			terms: ['graph', 'neighbour']
		},
		{
			id: 'step',
			source: 'walks',
			at: EXAMPLE,
			view: walk('step'),
			provenance: 'live',
			title: L('One step of a biased random walk', '有偏随机游走的一步'),
			narration: {
				en: 'A random walk hops from member to member along friendships, and node2vec biases each hop with two settings, p and q. Here the walk is at member {v}, having come from {t}: going back to {t} weighs 1/p = {wb}, a friend that {t} also knows weighs 1, any other friend 1/q = {wf}. Divided by their total, {tot}, that gives back {pb|pct}, a friend of {t} {pn|pct}, farther {pf|pct}. A random number u = {u} times the total lands in member {x}’s stretch of the running sum.',
				zh: '随机游走沿着朋友关系从一个成员跳到另一个成员，node2vec 用两个设置 p 和 q 让每一跳有偏向。这里游走在成员 {v}，是从 {t} 走来的：走回 {t} 的权重是 1/p = {wb}，{t} 也认识的朋友权重是 1，其他朋友的权重是 1/q = {wf}。除以它们的总和 {tot}，就得到：回去 {pb|pct}，去 {t} 的朋友 {pn|pct}，更远 {pf|pct}。随机数 u = {u} 乘以总和，落在累计里成员 {x} 的那一段，于是选中成员 {x}。',
				vars: (rt, at) => ({ v: m(at, 'v'), t: m(at, 't'), wb: String(1 / G(rt).p), wf: String(1 / G(rt).q), tot: m(at, 'total'), pb: m(at, 'p0'), pn: m(at, 'p1'), pf: m(at, 'p2'), u: m(at, 'u'), x: m(at, 'x') })
			},
			look: L('The filled member is where the walk is, the half-filled one where it came from; each friend shows its weight (×). The table adds the weights up in a running sum; the line under it shows where u × total lands.', '实心的成员是游走所在的位置，半实心的是它的来处；每个朋友旁边标着权重（×）。表格把权重依次累加；表格下面那一行给出 u × 总和落在哪里。'),
			predict: {
				id: 'node2vec-bfs-dfs',
				question: L('A small q (below 1) makes every “farther” friend weigh more than 1. Do such walks act like breadth-first search (staying near where they started) or like depth-first search (wandering off)?', 'q 小于 1 时，每个“更远”的朋友权重都大于 1。这样的游走更像广度优先搜索（待在起点附近），还是更像深度优先搜索（越走越远）？'),
				options: [L('Like breadth-first search: they stay near the start', '像广度优先：待在起点附近'), L('Like depth-first search: they wander off', '像深度优先：越走越远'), L('Neither: q only changes the chance of going back', '都不像：q 只改变走回去的概率')],
				answer: 1,
				explain: L(
					'With q below 1 the walk keeps moving away from where it just was, deeper into the graph, like depth-first search. With q above 1 it prefers friends of the member it came from and circles near its start, like breadth-first search. p does the same for the way back: a small p makes going back likely.',
					'q 小于 1 时，游走总是离开刚才所在的地方，往图的深处走，像深度优先搜索。q 大于 1 时，它偏爱来处那个成员的朋友，在起点附近打转，像广度优先搜索。p 管的是回头：p 越小，越容易走回去。'
				)
			},
			terms: ['graph-walk', 'node2vec-pq', 'bfs-dfs']
		},
		{
			id: 'sentence',
			source: 'walks',
			at: WALK_LAST,
			view: walk('walks'),
			play: { from: WALK_FIRST, to: WALK_LAST, phases: ['move'], interval: 700, unit: L('move', '步'), by: 'sub' },
			provenance: 'live',
			title: L('A walk is a sentence', '一条游走就是一个句子'),
			narration: {
				en: 'Hop {hops} times and you have a walk of {len} members; node2vec walks {r} times from every member, {nw} walks in all. Written down, a walk is a sentence whose words are member ids, and from here on node2vec treats these sentences exactly as word2vec treats text. With p = {p} and q = {q}, a walk ends on average {far} hops from its start (by the shortest way along friendships) and meets {distinct} different members.',
				zh: '跳 {hops} 次，就得到一条有 {len} 个成员的游走；node2vec 从每个成员出发走 {r} 次，一共 {nw} 条游走。写下来，一条游走就是一个句子，句子里的词是成员编号；从这里开始，node2vec 对待这些句子的方式和 word2vec 对待文本完全一样。p = {p}、q = {q} 时，一条游走结束时平均离起点 {far} 跳（沿朋友关系走最短的路来数），遇到 {distinct} 个不同的成员。',
				vars: (rt) => { const W = walksOf(rt.input); return { hops: LEN - 1, len: LEN, r: WALKS, nw: W.walks.filter((w) => w.length > 1).length, p: String(W.p), q: String(W.q), far: W.far, distinct: W.distinct }; }
			},
			look: L('The walk grows move by move on the club (its newest move is thick); below it, every walk from the same member, one per round.', '游走在俱乐部图上一步一步变长（最新的一步加粗）；下面是从同一个成员出发的每一条游走，每轮一条。'),
			terms: ['graph-walk']
		},
		{
			id: 'pairs',
			source: 'walks',
			at: 'last',
			view: walk('pairs'),
			provenance: 'live',
			title: L('Pairs from the walks', '从游走得到训练对'),
			narration: {
				en: 'As in word2vec, each member of a walk is in turn the center, and every member up to {w} places before or after it is one of its context members: here member {c} gets {k} pairs. All the walks give {P} (center, context) pairs. Members that are often close in walks get many pairs together: member {c}’s most frequent other context member is {top}, {n} times.',
				zh: '和 word2vec 一样，游走中的每个成员轮流当中心，它前后各 {w} 个位置以内的成员都是它的上下文成员：这里成员 {c} 得到 {k} 个训练对。全部游走一共给出 {P} 个（中心，上下文）训练对。在游走中经常挨得近的成员会一起出现在很多训练对里：成员 {c} 最常见的其他上下文成员是 {top}，出现了 {n} 次。',
				vars: (rt): V => {
					const W = walksOf(rt.input), ex = exampleWalk(rt.input);
					if (ex < 0) return { w: NaN };
					const c = W.walks[ex][WIN], top = Array.from({ length: N }, (_, j) => j).filter((j) => j !== c).reduce((b, j) => (W.cooc[c * N + j] > W.cooc[c * N + b] ? j : b), c === 0 ? 1 : 0);
					return { w: WIN, c, k: 2 * WIN, P: W.pairs.length, top, n: W.cooc[c * N + top] };
				}
			},
			look: L('Under the club, the walk written as a sentence: the outlined member and its window (dashed) give its pairs; below them, its most frequent context members over all walks.', '俱乐部图下面是写成句子的游走：加框的成员和它的窗口（虚线）给出它的训练对；再下面是在全部游走中它最常见的上下文成员。'),
			terms: ['skip-gram', 'word-window']
		},
		{
			id: 'update',
			source: 'one',
			at: { phase: 'update', step: 0 },
			view: 'node2vec/update',
			provenance: 'live',
			title: L('One skip-gram step, with its numbers', '带着数字走一步 skip-gram'),
			narration: {
				en: 'Training is word2vec’s skip-gram on these pairs (the word2vec page goes slowly). For the pair ({c}, {o}), member {c}’s vector h ({d} numbers) scores every member by a dot product with that member’s output vector (each member has a second one): {o} scores {u0}, a softmax share of {p0|pct}, so the loss is −ln {p0|n4} = {l0|n4}. One SGD step moves every number against its gradient (how fast the loss grows with it): row k = {k} goes {old} − {lr} × ({g}) = {new}, and {o}’s score rises to {u1}.',
				zh: '训练就是 word2vec 的 skip-gram，用在这些训练对上（word2vec 那一页会放慢讲）。对训练对（{c}，{o}），成员 {c} 的向量 h（{d} 个数）和每个成员的输出向量（每个成员的第二个向量）做点积，得到这个成员的分数：{o} 得分 {u0}，softmax 份额 {p0|pct}，所以损失是 −ln {p0|n4} = {l0|n4}。一步 SGD 让每个数沿梯度（损失随这个数增长得多快）的反方向走一点：k = {k} 那一行是 {old} − {lr} × ({g}) = {new}，{o} 的分数升到 {u1}。',
				vars: (rt, at): V => {
					const w = worked(at);
					return w ? { c: w.c, o: w.o, d: w.h.length, u0: w.u[w.o], p0: w.p[w.o], l0: w.loss, k: w.k, old: w.old, lr: String(w.lr), g: w.g, new: w.new, u1: w.u1 } : { c: NaN };
				}
			},
			look: L('One row per number of h (k counts from 0): times the context member’s output vector, the products add up to the score; the outlined row is the number worked out in the text.', 'h 的每个数占一行（k 从 0 数起）：乘以上下文成员的输出向量，乘积加起来就是分数；加框的那一行就是正文里算出的那个数。'),
			terms: ['dot-product', 'softmax', 'sgd']
		},
		{
			id: 'map',
			source: 'run',
			at: 'last',
			view: map(() => ({ animate: true, cut: true })),
			play: 'view',
			linger: 12000,
			provenance: 'live',
			title: L('Repeat: the two sides fall apart', '重复下去：两方分开了'),
			narration: {
				en: 'Repeat that step {steps} times on random pairs: the average loss over all {P} pairs falls from {l0} to {l1}. To draw vectors of {d} numbers flat, PCA keeps the two directions they spread most in ({share|pct} of the spread). {apart} one vertical cut through the map puts {cut} of {n} members on their own side, and for {same} of {n} members the nearest member (by cosine similarity: how alike two vectors’ directions are) is on their side, though training never saw the sides.',
				zh: '在随机抽出的训练对上把这一步重复 {steps} 次：所有 {P} 个训练对的平均损失从 {l0} 降到 {l1}。为了把 {d} 个数的向量画平，PCA 保留它们分布最开的两个方向（保留了 {share|pct} 的分布）。{apart}在图上竖着切一刀，{n} 个成员中有 {cut} 个落在自己一方；有 {same} 个成员的最近成员（按余弦相似度，也就是两个向量方向有多一致）和自己同一方，尽管训练从来没看到过谁在哪一方。',
				vars: (rt, at): V => {
					const vs = vecsAt(at.state), st = factionStats(rt.input, vs);
					if (!st || !at.tape) return { steps: NaN };
					const p = pca2(vs);
					const cut = cutOf(project(vs, p)).ok, apart = pick(cut >= 30, L('The sides fall apart:', '两方分开了：'), L('Here the sides do not come apart cleanly:', '这里两方没有完全分开：'));
					return { steps: m(at, 'step'), P: walksOf(rt.input).pairs.length, l0: at.tape.frame(0).metrics?.avgLoss ?? NaN, l1: m(at, 'avgLoss'), d: vs[0].length, share: p.share, apart, cut, n: N, same: st.same };
				}
			},
			look: L('Each dot is a member’s vector, coloured by the side it joined. Replay the training to watch the sides move apart.', '每个点是一个成员的向量，颜色是它加入的那一方。重放训练过程，看两方怎样分开。'),
			terms: ['node-embedding', 'pca', 'cosine-similarity']
		},
		{
			id: 'pq',
			source: 'run',
			at: 'last',
			view: map(() => ({ stats: true })),
			compare: { a: 'run', b: 'base', label: [L('Your p and q', '你的 p 和 q'), L('The start: p = 1, q = 0.5', '开始时：p = 1，q = 0.5')] },
			provenance: 'live',
			title: L('Your turn: choose p and q', '轮到你：选择 p 和 q'),
			narration: {
				en: 'Large p and small q send walks far (like DFS); small p and large q keep them close (like BFS). Your walks (p = {p}, q = {q}) end {far} hops from their start, against {far0} with the start setting. A member’s 3 nearest members are on average {hop} hops away in the graph, against {hop0}: {verdict} Members whose nearest member is on their side: {same} of {n} with yours, {same0} with the start.',
				zh: 'p 大、q 小，游走就走得远（像 DFS）；p 小、q 大，游走就待在附近（像 BFS）。你的游走（p = {p}，q = {q}）结束时离起点 {far} 跳，开始的设置是 {far0} 跳。一个成员最近的 3 个成员在图中平均相隔 {hop} 跳，开始的设置是 {hop0} 跳：{verdict}最近的成员和自己同一方的：你的设置下 {n} 个成员中有 {same} 个，开始的设置下有 {same0} 个。',
				vars: (rt): V => {
					const a = sideOf(rt, 'run'), b = sideOf(rt, 'base'), same = a.p === b.p && a.q === b.q;
					const agree = (a.far - b.far) * (a.hop - b.hop) > 0;
					const verdict = same ? L('the same setting, so far the same walks.', '还是同一个设置，所以游走也一样。') : !agree ? L('here the difference is too small to show in the neighbours.', '这里的差别太小，在邻居上看不出来。') : pick(a.far < b.far, L('walks that stay closer give closer neighbours.', '游走待得越近，邻居也越近。'), L('walks that go farther give farther neighbours.', '游走走得越远，邻居也越远。'));
					return { p: String(a.p), q: String(a.q), far: a.far, far0: b.far, hop: a.hop, hop0: b.hop, verdict, same: a.same, same0: b.same, n: N };
				}
			},
			look: L('Left: your p and q; right: the start setting, on the same friendships. Compare where the two sides sit and how tight each side is.', '左边：你的 p 和 q；右边：开始时的设置，朋友关系相同。比较两方的位置，以及每一方聚得有多紧。'),
			task: {
				prompt: L('Change p and q with the buttons under “Your input” (try p = 0.25 with q = 4, then p = 4 with q = 0.25) and compare the two maps.', '用“你的输入”下面的按钮改变 p 和 q（先试 p = 0.25、q = 4，再试 p = 4、q = 0.25），比较两张图。'),
				done: (rt) => G(rt).p !== START_P || G(rt).q !== START_Q,
				hint: L('The rows “p (return)” and “q (in-out)” at the top of “Your input”.', '“你的输入”最上面的“p（返回）”和“q（进出）”两行。'),
				doneText: L('Done: the left map now trains on walks with your p and q.', '完成了：左边的图现在用你的 p 和 q 的游走来训练。')
			},
			linger: 12000,
			terms: ['node2vec-pq', 'bfs-dfs']
		},
		{
			id: 'nn',
			source: 'run',
			at: 'last',
			view: map((rt) => ({ focus: G(rt).focus, list: true })),
			provenance: 'live',
			title: L('Nearest members', '最近的成员'),
			narration: {
				en: 'The members nearest to member {f} (by cosine similarity of the whole vectors) are {list}. Of these 5, {fr}, {two}, and {same} ({side}). {note}',
				zh: '离成员 {f} 最近的成员（按整个向量的余弦相似度）是 {list}。这 5 个里，{fr}，{two}，{same}（{side}）。{note}',
				vars: (rt, at): V => {
					const f = G(rt).focus, st = factionStats(rt.input, vecsAt(at.state));
					if (!st) return { f: NaN };
					const near = st.nn[f].slice(0, 5), H = st.H[f], alone = !walksOf(rt.input).nb[f].length;
					const note = alone
						? L(`But member ${f} has no friends: no walk reaches it and no pair trains its vector, so it keeps its random start and its neighbours mean nothing.`, `但成员 ${f} 没有朋友：没有游走经过它，也没有训练对训练它的向量，所以它保持着随机的初始值，它的近邻没有意义。`)
						: L('Near in the vectors means often in the same walk windows, which here mostly means friends and friends of friends.', '向量里离得近，就是经常出现在同一个游走窗口里；在这里，这大多就是朋友和朋友的朋友。');
					const count = (n: number, one: string, many: string, zh: string) => plural(n, { one: `{n} ${one}`, many: `{n} ${many}`, zh: `{n} 个${zh}` });
					return {
						f,
						list: L(near.map((x) => x.j).join(', '), near.map((x) => x.j).join('、')),
						fr: count(near.filter((x) => H[x.j] === 1).length, 'is its friend', 'are its friends', '是它的朋友'),
						two: count(near.filter((x) => H[x.j] === 2).length, 'is a friend of a friend (two hops away)', 'are friends of friends (two hops away)', '是朋友的朋友（相隔两跳）'),
						same: count(near.filter((x) => FACTIONS[x.j] === FACTIONS[f]).length, 'is on its side', 'are on its side', '和它同一方'),
						side: side(FACTIONS[f]),
						note
					};
				}
			},
			look: L('The ringed member and dashed lines to its 3 nearest; the table lists its 5 nearest with their hops in the graph. Pick another member in focus under “Your input”.', '加圈的成员，以及连到它最近 3 个成员的虚线；表格列出最近的 5 个成员和它们在图中相隔几跳。可以在“你的输入”下面换一个关注的成员。'),
			terms: ['cosine-similarity', 'neighbour']
		},
		{
			id: 'gcn',
			source: 'run',
			at: 'last',
			view: map(() => ({ leaders: true })),
			provenance: 'live',
			title: L('Next to the GCN: walks instead of layers', '和 GCN 比一比：用游走，不用层'),
			narration: {
				en: 'The GCN page solved this club another way: told the sides of only members {hi} and {off}, it learned weights that mix each member’s numbers with its friends’, and sorted {fact:gcnRight} of the other {rest} right. node2vec never sees a side; it learns a vector of its own for every member, from the walks alone. Only now do we give it the same two labels: put each member with the leader whose vector is closer to its own (by cosine similarity), and that sorts {right} of {rest} right.',
				zh: 'GCN 那一页用另一种方法解决了这个俱乐部：只告诉它成员 {hi} 和 {off} 在哪一方，它学出一组权重，把每个成员的数和它朋友的数混在一起，并把其余 {rest} 人中的 {fact:gcnRight} 人分对了。node2vec 从来看不到谁在哪一方；它只靠游走，为每个成员单独学一个向量。直到现在我们才给它同样的两个标签：把每个成员归到向量和它更近（按余弦相似度）的那个领头人一方，这样 {rest} 人中有 {right} 人分对了。',
				vars: (rt, at): V => ({ right: factionStats(rt.input, vecsAt(at.state))?.right ?? NaN, rest: N - 2, ...LEADERS })
			},
			look: { en: 'Members {hi} and {off}, the leaders, are ringed; ✕ marks a member whose vector is closer to the other side’s leader.', zh: '两个领头人，成员 {hi} 和 {off}，加了圈；✕ 标出向量离另一方领头人更近的成员。', vars: () => LEADERS },
			terms: ['message-passing', 'semi-supervised']
		},
		{
			id: 'limits',
			source: 'run',
			at: 'last',
			view: map((rt) => ({ group: twins(rt.input) })),
			provenance: 'live',
			kindOf: 'limits',
			title: L('Limits: only the walks, and only these members', '局限：只有游走，也只有这些成员'),
			narration: {
				en: 'node2vec knows a member only by its walks. {twins} It ignores features (facts such as a member’s age). And it keeps one row per member it trained on: a new member has no vector until you walk and train again, while a GCN given real features applies its trained weights to a new member’s features and friends.',
				zh: 'node2vec 只通过游走来认识一个成员。{twins}它不看任何特征（成员的其他信息，比如年龄）。而且它只为训练过的成员各存一行：新成员在重新游走、重新训练之前没有向量；而有真实特征的 GCN 可以把训练好的权重直接用在新成员的特征和朋友上。',
				vars: (rt, at): V => {
					const g = twins(rt.input), vs = vecsAt(at.state), nb = walksOf(rt.input).nb;
					if (g.length < 2 || !vs.length) return { twins: L('Two members with the same friends would get alike vectors.', '朋友完全相同的两个成员会得到相像的向量。') };
					let s = 0, n = 0, all = 0, k = 0;
					for (let a = 0; a < g.length; a++) for (let b = a + 1; b < g.length; b++) (s += cosine(vs[g[a]], vs[g[b]])), n++;
					for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) (all += cosine(vs[a], vs[b])), k++;
					const c = fmt3(s / n), r = fmt3(all / k), fr = nb[g[0]], who = L(`Members ${g.join(', ')} have the same friends (${fr.join(' and ')}), so`, `成员 ${g.join('、')} 的朋友完全相同（${fr.join(' 和 ')}），所以`);
					// "alike" only with the claim test's margin (0.3 above two random members); a short run is not there yet
					const alike = s / n > all / k + 0.3;
					return {
						twins: pick(
							alike,
							L(`${who.en} their vectors end up alike: cosine similarity ${c} on average, against ${r} for two random members.`, `${who.zh}他们的向量最后很相像：平均余弦相似度 ${c}，而随便两个成员只有 ${r}。`),
							L(`${who.en} their vectors should end up alike, but are not yet: mean cosine similarity ${c}, against ${r} for two random members.`, `${who.zh}他们的向量本该变得相像，但这里还没有：平均余弦相似度 ${c}，随便两个成员是 ${r}。`)
						)
					};
				}
			},
			look: L('The ringed members have the same friends; see how close together their dots sit.', '加圈的成员朋友完全相同，看看他们的点挨得有多近。'),
			terms: ['node-embedding']
		}
	],
	fixtures: [
		{ id: 'start', label: L('The start: the club, p = 1, q = 0.5, member 8 in focus', '开始时：俱乐部，p = 1，q = 0.5，关注成员 8'), raw: START },
		{ id: 'bfs', label: L('BFS-like walks: p = 0.25, q = 4', '像 BFS 的游走：p = 0.25，q = 4'), raw: { edges: inputOf(null).edges, focus: 8, p: 0.25, q: 4 } },
		{ id: 'dfs', label: L('DFS-like walks: p = 4, q = 0.25, member 33 in focus', '像 DFS 的游走：p = 4，q = 0.25，关注成员 33'), raw: { edges: inputOf(null).edges, focus: 33, p: 4, q: 0.25 } }
	]
};

export default lesson;
