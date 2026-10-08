// gnn-variants: the guided lesson (catalog row: GAT / GraphSAGE / GIN on the same node; 11 steps). Every number is a
// {placeholder}: structure from the graph, live values from the fwd trace and the gat / drop frames, measured ones from
// facts.json. Claims that are not computed live are tested in claims.test.ts.
//   1 the member and its friends → 2 gather, combine, update (GCN) → 3 GraphSAGE: average → 4 GIN: sum + MLP (predict) →
//   5 GAT: scores (predict) → 6 softmax → 7 weighted sum → 8 four rules side by side → 9 four embeddings → 10 accuracy →
//   11 attention weights are clues, not explanations (limits, task; layer-1 and layer-2 α against removing a friendship)
// `duration` on the long steps without a predict, play or task: their reading time under Play (about 55 ms per character
// of the en narration; the kit's default stops at 16 s).
import type { At, ExplainerRuntime, I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { KARATE, LABELLED, LABELS, listOf, N, NAMES, VARIANTS } from './graph.ts';
import { focusOf, gcnRule, ginRule, rightOf, sageRule, sideOf, topFriend, vecText, weightOf } from './rules.ts';

type RT = ExplainerRuntime;
const I = (en: string, zh: string): I18n => ({ en, zh });
const [HI, OFF] = LABELLED;
const M = (at: At) => (at.frame?.metrics ?? {}) as Record<string, number>;
const f = (x: number) => fmt3(x), v3 = (v: number[]) => vecText(v, f);
const view = (mode: string) => ({ key: 'gnn-variants/agg', props: () => ({ mode }) });
const H = (rt: RT) => Number(rt.spec?.layers.find((l) => l.id === 'gat.gc1')?.outShape[1] ?? NaN);
/** a frame of the gat source by phase and step (the narration may quote one the cursor has passed) */
const gatFrame = (rt: RT, phase: string, step: number) => { const t = rt.sources.gat?.tape; const k = t ? t.find({ phase, step }) : -1; return ((k >= 0 ? t!.frame(k).metrics : undefined) ?? {}) as Record<string, number>; }; // prettier-ignore
/** true when the learner's friendships differ from Zachary's (the networks were trained on Zachary's) */
const edited_ = (rt: RT) => JSON.stringify(focusOf(rt).g.edges) !== JSON.stringify(KARATE.edges);
const sideText = (rt: RT, i: number): I18n => {
	const by = [0, 1].map((c) => VARIANTS.filter((v) => sideOf(rt, v, i) === c).map((v) => NAMES[v]));
	const side = [I('Mr. Hi’s side', 'Mr. Hi 一侧'), I('the Officer’s side', '干事一侧')];
	const joined = [I('Mr. Hi', 'Mr. Hi'), I('the Officer', '干事')][LABELS[i]];
	const c = by[0].length ? 0 : 1;
	if (!by[0].length || !by[1].length)
		return c === LABELS[i] ? I(`${side[c].en} in all four networks, the side it joined`, `在四个网络里都落在${side[c].zh}，也就是他加入的一方`) : I(`${side[c].en} in all four networks, though it joined ${joined.en}`, `在四个网络里都落在${side[c].zh}，可他实际加入的是 ${joined.zh} 一方`);
	const l = (k: number) => I(by[k].length > 1 ? `${by[k].slice(0, -1).join(', ')} and ${by[k].at(-1)}` : by[k][0], by[k].join('、'));
	return I(`${side[0].en} in ${l(0).en} and ${side[1].en} in ${l(1).en}; it joined ${joined.en}`, `在 ${l(0).zh} 里落在${side[0].zh}，在 ${l(1).zh} 里落在${side[1].zh}；他实际加入的是 ${joined.zh} 一方`);
};

const lesson: LessonDef = {
	title: I('GCN, GraphSAGE, GIN and GAT on the same member', '同一个成员上的 GCN、GraphSAGE、GIN 和 GAT'),
	summary: I('Follow one member of Zachary’s karate club through four kinds of graph layer: a GCN’s fixed weights, GraphSAGE’s average, GIN’s sum and GAT’s attention, with real numbers; then compare their embeddings and accuracy, and test whether attention explains anything.', '跟着 Zachary 空手道俱乐部的一个成员走过四种图网络层：GCN 的固定权重、GraphSAGE 的平均、GIN 的求和和 GAT 的注意力，全部带真实数字；再比较它们的嵌入和准确率，并检验注意力能不能解释什么。'),
	steps: [
		{
			id: 'club', duration: 20000, source: 'fwd', at: { layer: 'adj' }, view: view('club'), provenance: 'live', title: I('One member and its friends', '一个成员和他的朋友'),
			narration: {
				en: 'Wayne Zachary recorded which of the {n} members of a karate club were friends ({m} pairs{edited}), and the club later split between Mr. Hi (member {hi}) and the Officer (member {off}). Four graph networks were each told only which side these two members took, and must guess the side of every other member from the friendships. We follow member {focus}, {who}.',
				zh: 'Wayne Zachary 记录了一个空手道俱乐部 {n} 个成员中谁和谁是朋友（一共 {m} 对{edited}），后来俱乐部分裂成两派：Mr. Hi（成员 {hi}）一派和干事（成员 {off}）一派。四个图网络都只被告知这两个人各站在哪一派，要靠朋友关系去猜其余每个成员站在哪一派。我们跟着成员 {focus}：{who}。',
				vars: (rt) => {
					const { g, i, nb } = focusOf(rt);
					const edited = edited_(rt) ? I(`; your edited copy has ${g.edges.length}`, `；你改过的版本有 ${g.edges.length} 对`) : I('', '');
					const who = nb.length ? I(`who has ${nb.length === 1 ? 'one friend' : `${nb.length} friends`}: ${listOf(nb).en}`, `他有 ${nb.length} 个朋友（${listOf(nb).zh}）`) : I('who has no friends', '他没有朋友');
					return { n: N, m: KARATE.edges.length, edited, hi: HI, off: OFF, focus: i, who };
				}
			},
			look: I('Click any member to follow it instead: its friendships light up, and every later step is computed for it.', '点任意一个成员就改为跟随他：他的朋友关系会亮起来，后面每一步都会为他重新计算。'), terms: ['graph', 'neighbour']
		},
		{
			id: 'recipe', duration: 26000, source: 'fwd', at: { layer: 'gcn.act1' }, view: view('gcn'), provenance: 'live', title: I('What every graph layer does: gather, combine, update', '每个图网络层都在做的事：收集、合并、更新'),
			narration: {
				en: 'Each member starts as a one-hot row (all 0s, a 1 in its own column); multiplied by the layer’s learned weights W, it picks out one column, so member j’s message is column j of W: {h} numbers. A graph layer gathers the messages of member {focus} and its friends, combines them into one row, then updates it with a bias and tanh. GCN combines with fixed weights Â = 1/√(dᵢ dⱼ), where d counts a member’s friends plus itself: {selfw} for the member itself{fr}.',
				zh: '每个成员一开始是一行独热的数（全是 0，只有自己那一列是 1）；这一行乘以这一层学来的权重 W，正好挑出 W 的一列，所以成员 j 的消息就是 W 的第 j 列：{h} 个数。图网络层先收集成员 {focus} 和他朋友们的消息，把它们合并成一行，再加上偏置、用 tanh 更新。GCN 用固定权重 Â = 1/√(dᵢ dⱼ) 来合并，d 是一个成员的朋友数再加上他自己：成员自己是 {selfw}{fr}。',
				vars: (rt) => {
					const { i, nb } = focusOf(rt);
					const w = nb.map((j) => weightOf(rt, 'gcn', j) ?? NaN);
					const fr = nb.length ? I(`, ${f(Math.min(...w))} to ${f(Math.max(...w))} for its friends`, `，朋友们是 ${f(Math.min(...w))} 到 ${f(Math.max(...w))}`) : I('', '');
					return { h: H(rt), focus: i, selfw: weightOf(rt, 'gcn', i) ?? NaN, fr };
				}
			},
			look: I('Thicker edges carry more weight. Each table row is one sender’s message times its weight; the bold rows add them up, add the bias and apply tanh.', '边越粗，权重越大。表里每一行是一个发送者的消息乘以它的权重；加粗的几行把它们加起来，加上偏置，再用 tanh。'), terms: ['one-hot', 'message-passing', 'degree']
		},
		{
			id: 'sage', duration: 28000, source: 'fwd', at: { layer: 'sage.act1' }, view: view('sage'), provenance: 'live', title: I('GraphSAGE: average the friends, keep yourself apart', 'GraphSAGE：朋友取平均，自己单独放'),
			narration: {
				en: 'GraphSAGE combines with a plain average: {each}. The member’s own row is not mixed in: it is placed beside the average (concatenation), and one weight table reads both halves. The own half gives {own} and the friends’ half gives the average of their columns, {mean}; adding the two halves and the bias gives z = {z}, and tanh makes it {h}.',
				zh: 'GraphSAGE 用简单的平均来合并：{each}。成员自己的行不混进平均，而是放在平均值旁边（拼接），再由一张权重表同时读这两半。“自己”那一半给出 {own}，“朋友”那一半给出朋友们那几列的平均 {mean}；把两半相加再加上偏置得到 z = {z}，再经过 tanh 变成 {h}。',
				vars: (rt) => {
					const [{ i, nb }, R] = [focusOf(rt), sageRule(rt)];
					const k = nb.length;
					const each = k ? I(`each of member ${i}’s ${k === 1 ? 'one friend counts' : `${k} friends counts 1/${k} = ${f(1 / k)}`}, however many friends that friend has`, `成员 ${i} 的 ${k} 个朋友每个都占 1/${k} = ${f(1 / k)}，不管这个朋友自己有多少朋友`) : I(`member ${i} has no friends, so its average is all 0s`, `成员 ${i} 没有朋友，所以他的平均全是 0`);
					return { each, own: v3(R.own), mean: v3(R.mean), z: v3(R.z), h: v3(R.h) };
				}
			},
			look: I('Every friend’s edge carries the same weight. In the table, z = the “own” row + the average row + b.', '每个朋友的边权重都一样。表里的 z = “自己”那一行 + 平均那一行 + b。'), terms: ['concatenation', 'aggregation']
		},
		{
			id: 'gin', source: 'fwd', at: { layer: 'gin.act1' }, view: view('gin'), provenance: 'live', title: I('GIN: add up the friends, then an MLP', 'GIN：把朋友加起来，再接一个 MLP'),
			narration: {
				en: 'GIN adds the friends’ rows up without dividing, plus the member’s own row times 1 + ε, where ε = {eps} was learned. A small network inside the layer, an MLP (linear, ReLU, linear), then works on that sum. For member {focus}, the MLP’s first linear layer gives {onePlus} × the member’s own column of W₁ plus {cols}, plus a bias: {pre}; ReLU and the second linear layer make it {out}, and tanh {h}.',
				zh: 'GIN 把朋友们的行直接加起来，不做除法，再加上成员自己的行乘以 1 + ε，其中 ε = {eps} 是学来的。层里还有一个小网络，即 MLP（线性层、ReLU、线性层），接着处理这个和。对成员 {focus} 来说，MLP 的第一个线性层给出 {onePlus} × 成员自己那一列 W₁，加上{cols}，再加偏置：{pre}；经过 ReLU 和第二个线性层变成 {out}，再经过 tanh 变成 {h}。',
				vars: (rt) => {
					const [{ i, nb }, R] = [focusOf(rt), ginRule(rt)];
					const cols = nb.length ? I(`the columns of its ${nb.length === 1 ? 'one friend' : `${nb.length} friends`}`, `他 ${nb.length} 个朋友的那几列`) : I('nothing from friends (it has none)', '朋友们的列（他没有朋友，所以什么也不加）');
					return { eps: R.eps, focus: i, onePlus: 1 + R.eps, cols, pre: v3(R.pre), out: v3(R.out), h: v3(R.h) };
				}
			},
			predict: { id: 'gnn-variants-sum', question: I('Members A and B start from the same row. A has one friend and B has three, and all four friends send exactly the same row. Which combine gives A and B different results?', '成员 A 和 B 自己的行完全一样。A 有一个朋友，B 有三个，这四个朋友发来的行也完全一样。哪种合并方式会让 A 和 B 得到不同的结果？'), options: [I('GraphSAGE’s average', 'GraphSAGE 的平均'), I('GIN’s sum', 'GIN 的求和'), I('Both', '两种都会'), I('Neither', '两种都不会')], answer: 1, explain: I('The average of three equal rows is that same row, so GraphSAGE gives A and B the same result. The sum of three equal rows is three times that row, so GIN tells them apart: a sum remembers how many friends sent the same message. That is why GIN can tell apart more graphs than layers that average (Xu et al., 2019); the box under the table shows it on this club.', '三个相同的行取平均还是那一行，所以 GraphSAGE 给 A 和 B 的结果一样。三个相同的行加起来是那一行的三倍，所以 GIN 能把它们区分开：求和记得有多少个朋友发来了同样的消息。这就是 GIN 比取平均的图网络层能区分更多图的原因（Xu 等，2019）；表格下方的方框在这个俱乐部上演示了这一点。') },
			look: I('The box under the table: start every member from the same number and compare what the average and the sum make of it.', '看表格下方的方框：让每个成员都从同一个数出发，比较平均和求和分别得到什么。'), terms: ['mlp', 'aggregation']
		},
		{
			id: 'score', source: 'gat', at: { phase: 'score', last: true }, play: { from: { phase: 'score' }, to: { phase: 'score', last: true }, interval: 1100, unit: I('sender', '发送者') }, view: view('gat'), provenance: 'live', title: I('GAT: score each friend', 'GAT：给每个朋友打分'),
			narration: {
				en: 'GAT lets the network learn how much each friend counts. Every sender j of member {focus}, itself included, gets a score e = LeakyReLU(a_dst · z_{focus} + a_src · zⱼ): z is a member’s message (its column of W, as before), a_dst and a_src are two learned rows of {h} numbers, and “·” multiplies two rows number by number and adds up the products. LeakyReLU keeps a positive number and multiplies a negative one by {slope}. For {first} (the first row): LeakyReLU({sd} + {ss}) = {e}.',
				zh: 'GAT 让网络自己学每个朋友占多少分量。成员 {focus} 的每个发送者 j（包括他自己）都得到一个分数 e = LeakyReLU(a_dst · z_{focus} + a_src · zⱼ)：z 是一个成员的消息（和前面一样，是 W 中他那一列），a_dst 和 a_src 是两行学来的数，每行 {h} 个，“·”表示把两行数逐个相乘，再把乘积加起来。LeakyReLU 保留正数，把负数乘以 {slope}。对{first}（第一行）：LeakyReLU({sd} + {ss}) = {e}。',
				vars: (rt, at) => {
					const s = gatFrame(rt, 'score', 1); // the member itself: the first row of the table from the first frame of Play to the last
					return { focus: M(at).node, h: H(rt), slope: Number.isFinite(M(at).slope) ? String(M(at).slope) : NaN, first: I(`member ${s.j} itself`, `成员 ${s.j} 自己`), sd: s.sd, ss: s.ss < 0 ? `(${f(s.ss)})` : s.ss, e: s.e };
				}
			},
			predict: { id: 'gnn-variants-top', question: I('Which friend will GAT give the most weight to?', 'GAT 会给哪个朋友最大的权重？'), options: [I('The friend with the most friends of its own', '自己朋友最多的那个朋友'), I('The friend with the fewest friends, as GCN’s 1/√(dᵢ dⱼ) prefers', '自己朋友最少的那个朋友，就像 GCN 的 1/√(dᵢ dⱼ) 偏爱的那样'), I('No fixed rule: the friend with the highest learned score', '没有固定规则：学来的分数最高的那个朋友')], answer: 2, explain: I('GCN’s weights follow from the degrees alone. GAT’s come from learned scores of the members’ numbers, and the softmax in the next step keeps their order: the highest score gets the most weight. In Zachary’s club, member 8’s highest score goes to member 0, who has 16 friends, but member 33’s goes to member 19, who has only 3.', 'GCN 的权重只由度决定。GAT 的权重来自一条学来的打分规则，它按成员们的数给每个朋友打分；下一步的 softmax 会保持分数的大小顺序：分数最高的得到最大的权重。在 Zachary 的俱乐部里，成员 8 的最高分给了有 16 个朋友的成员 0，而成员 33 的最高分给了只有 3 个朋友的成员 19。') },
			look: I('Press Play: the senders are scored one at a time, each score appears on its edge, and the table adds the same a_dst · z to every row.', '按播放：发送者一个一个地被打分，每个分数显示在对应的边上；表里每一行都加上同一个 a_dst · z。'), terms: ['attention', 'leaky-relu']
		},
		{
			id: 'softmax', duration: 20000, source: 'gat', at: { phase: 'softmax' }, view: view('gat'), provenance: 'live', title: I('Softmax turns scores into weights', 'softmax 把分数变成权重'),
			narration: {
				en: 'A softmax turns {scores} into weights α between 0 and 1 that add up to 1: α = exp(score) ÷ (the sum of exp(score) over {senders}), where exp(x) = eˣ is always positive. {top} The member itself gets {self}; an even split would give each sender {even}.',
				zh: 'softmax 把{scores}变成介于 0 和 1 之间、加起来等于 1 的权重 α：α = exp(分数) ÷（{senders}的 exp(分数) 之和），其中 exp(x) = eˣ 总是正数。{top}成员自己得到 {self}；如果平均分配，每个发送者是 {even}。',
				vars: (rt, at) => {
					const m = M(at);
					const top = m.friends > 0 ? I(`The largest friend weight, ${f(m.aTop)}, goes to member ${m.top}, who has ${m.dTop} friends; in step 2, GCN gave that edge ${f(weightOf(rt, 'gcn', m.top) ?? NaN)}.`, `最大的朋友权重 ${f(m.aTop)} 给了成员 ${m.top}，他有 ${m.dTop} 个朋友；在第 2 步里，GCN 给这条边的权重是 ${f(weightOf(rt, 'gcn', m.top) ?? NaN)}。`) : I('With no friends, the member’s own score takes all the weight.', '没有朋友时，成员自己的分数拿走全部权重。');
					return { scores: plural(m.of, { one: 'the only score', many: 'the {n} scores', zh: '这 {n} 个分数' }), senders: plural(m.of, { one: 'its one sender', many: 'all {n} senders', zh: '所有 {n} 个发送者' }), top, self: m.self, even: m.even };
				}
			},
			look: I('The table lists each score, exp(score − max) and α; subtracting the largest score first keeps the numbers small and does not change α.', '表里列出每个分数、exp(分数 − 最大值) 和 α；先减去最大的分数可以让数保持不大，又不会改变 α。'), terms: ['softmax', 'attention']
		},
		{
			id: 'mix', source: 'gat', at: { phase: 'activate' }, play: { from: { phase: 'message' }, to: { phase: 'activate' }, interval: 1100, unit: I('message', '消息') }, view: view('gat'), provenance: 'live', title: I('A weighted sum of the messages', '消息的加权和'),
			narration: {
				en: 'Now each sender’s message z is multiplied by its weight α, and member {focus} adds the products up, number by number. {last}: {a} × {xj} = {mj}, and the sums are {sum}. Adding the bias {b} and applying tanh, GAT’s layer 1 gives member {focus} the row {h}.',
				zh: '现在每个发送者的消息 z 乘上它的权重 α，成员 {focus} 再把这些乘积逐个数地加起来。{last}：{a} × {xj} = {mj}，加起来是 {sum}。加上偏置 {b} 再经过 tanh，GAT 第 1 层给成员 {focus} 的行是 {h}。',
				vars: (rt, at) => {
					const m = M(at);
					const s = (at.tape && at.index > 0 ? at.tape.frame(at.index - 1).metrics : undefined) ?? {};
					const v = (p: string, q: Record<string, number>) => v3([0, 1, 2, 3].map((k) => q[`${p}${k}`]));
					const last = s.k > 1 ? I(`The last message comes from member ${s.j}`, `最后一条消息来自成员 ${s.j}`) : I(`The only message comes from member ${s.j} itself`, `唯一的一条消息来自成员 ${s.j} 自己`);
					return { focus: m.node, last, a: s.a, xj: v('x', s), mj: v('m', s), sum: v('s', m), b: v('b', m), h: v('h', m) };
				}
			},
			look: I('Press Play: the messages arrive one edge at a time, the line above the table shows the newest z, and the bold rows add them up, then the bias and tanh.', '按播放：消息一条边一条边地到达，表格上方那一行是最新到达的 z；加粗的几行把它们加起来，再加偏置、用 tanh。'), terms: ['attention']
		},
		{
			id: 'side', duration: 20000, source: 'fwd', at: { layer: 'gat.act1' }, view: view('side'), provenance: 'live', title: I('Four ways to combine, side by side', '四种合并方式并排比较'),
			narration: {
				en: 'Here are the four rules for member {focus} on the same edges: GCN’s fixed Â, GraphSAGE’s {sagew} with the member’s own row kept apart, GIN’s {gin}, and GAT’s learned α. The first three follow from who is friends with whom; only GAT’s weights depend on the members’ numbers. {tops}',
				zh: '这里是成员 {focus} 在同样几条边上的四种规则：GCN 的固定 Â，GraphSAGE 的{sagew}（成员自己的行单独放），GIN {gin}，以及 GAT 学来的 α。前三种只由谁和谁是朋友决定；只有 GAT 的权重取决于成员们的数。{tops}',
				vars: (rt) => {
					const { i, nb } = focusOf(rt);
					const [c, a] = [topFriend(rt, 'gcn'), topFriend(rt, 'gat')];
					const tops = !nb.length ? I(`Member ${i} has no friends, so every rule keeps only its own row.`, `成员 ${i} 没有朋友，所以每种规则都只剩他自己的行。`) : c === a ? I(`GCN and GAT both give their largest friend weight to member ${a}.`, `GCN 和 GAT 都把最大的朋友权重给了成员 ${a}。`) : I(`GCN’s largest friend weight goes to member ${c}, GAT’s to member ${a}.`, `GCN 最大的朋友权重给了成员 ${c}，GAT 的给了成员 ${a}。`);
					return { focus: i, sagew: nb.length ? I(`equal 1/${nb.length}`, `相等权重 1/${nb.length}`) : I('average of no friends (all 0s)', '空平均（没有朋友，全是 0）'), gin: nb.length ? I(`plain 1s with ${f(ginRule(rt).eps + 1)} for itself`, `直接相加的 1（自己是 ${f(ginRule(rt).eps + 1)}）`) : I(`${f(ginRule(rt).eps + 1)} for itself alone`, `只剩自己的 ${f(ginRule(rt).eps + 1)}`), tops };
				}
			},
			look: { en: 'Under each drawing: the row that rule gives member {focus} after layer 1. Click a friend to follow it.', zh: '每张图下面：这种规则在第 1 层之后给成员 {focus} 的行。点一个朋友就改为跟随他。', vars: (rt) => ({ focus: focusOf(rt).i }) }, terms: ['aggregation', 'attention']
		},
		{
			id: 'embed', duration: 27000, source: 'fwd', at: { layer: 'gat.head' }, view: view('embed'), provenance: 'live', title: I('Two numbers per member, four maps', '每个成员两个数，四张图'),
			narration: {
				en: 'A second graph layer applies each network’s rule again to the layer-1 rows and gives {e} numbers per member, drawn as a point coloured by the faction it joined. A last small layer, the head, turns them into one score per faction; the dashed line marks equal scores. All four were trained alike ({fact:steps} steps, only members {hi} and {off} labelled), yet each draws its own map. Member {focus} lands on {sides}.',
				zh: '每个网络的第二个图网络层对第 1 层得到的行再用一次自己的规则，给每个成员 {e} 个数，这里画成一个点，颜色是这个成员实际加入的派别。最后还有一个小层，叫输出头，它把这两个数变成每一派各一个分数；虚线就是两个分数相等的地方。四个网络的训练方法完全一样（{fact:steps} 步，只有成员 {hi} 和 {off} 有标签），可它们画出的图各不相同。成员 {focus} {sides}。',
				vars: (rt) => ({ e: Number(rt.spec?.layers.find((l) => l.id === 'gat.act2')?.outShape[1] ?? NaN), hi: HI, off: OFF, focus: focusOf(rt).i, sides: sideText(rt, focusOf(rt).i) })
			},
			look: { en: '✕ marks a member on the wrong side of its network’s line; the ring is member {focus}.', zh: '✕ 标出落在自己网络那条线错误一侧的成员；圆圈是成员 {focus}。', vars: (rt) => ({ focus: focusOf(rt).i }) }, terms: ['embedding', 'semi-supervised', 'head']
		},
		{
			id: 'accuracy', duration: 23000, source: 'fwd', at: { layer: 'gat.head' }, view: view('acc'), provenance: 'live', title: I('How many members each network gets right', '每个网络分对了多少人'),
			narration: {
				en: 'Of the {rest} members whose sides the networks never saw, GCN puts {gcn} on the right side, GraphSAGE {sage}, GIN {gin} and GAT {gat}. One run can be lucky: retrained from {fact:seeds} random starting weights, they averaged {fact:gcnMean|pct} (GCN), {fact:sageMean|pct} (GraphSAGE), {fact:ginMean|pct} (GIN) and {fact:gatMean|pct} (GAT). On this small, two-label club, the plain GCN did best, and GraphSAGE swung the most, from {fact:sageMin|pct}, worse than a coin flip, to {fact:sageMax|pct}.{edited}',
				zh: '在网络从没见过派别的 {rest} 个成员里，GCN 分对了 {gcn} 个，GraphSAGE {sage} 个，GIN {gin} 个，GAT {gat} 个。一次训练可能碰巧好，也可能碰巧差：从 {fact:seeds} 组随机初始权重重新训练，平均正确率是 {fact:gcnMean|pct}（GCN）、{fact:sageMean|pct}（GraphSAGE）、{fact:ginMean|pct}（GIN）和 {fact:gatMean|pct}（GAT）。在这个只有两个标签的小俱乐部上，普通的 GCN 最好，GraphSAGE 波动最大，从比抛硬币还差的 {fact:sageMin|pct} 到 {fact:sageMax|pct}。{edited}',
				vars: (rt) => ({ rest: rightOf(rt, 'gcn').rest, ...Object.fromEntries(VARIANTS.map((v) => [v, rightOf(rt, v).right])), edited: edited_(rt) ? I(' They are not retrained on your graph: each member’s own column of W keeps what Zachary’s friendships taught.', '网络没有在你的图上重新训练：每个成员自己那一列 W 仍然保留着训练时从 Zachary 的朋友关系里学到的东西。') : I('', '') })
			},
			look: I('Bars: the four trained networks on your graph. Short upright tick: the average over the random starts; thin line: from the worst start to the best.', '长条：四个训练好的网络在你的图上的结果。短竖线：多次随机初始化的平均；细横线：从最差的一次到最好的一次。'), terms: ['label', 'semi-supervised']
		},
		{
			id: 'limits', source: 'drop', at: 'last', view: view('drop'), provenance: 'live', kindOf: 'limits', title: I('Attention weights are clues, not explanations', '注意力权重是线索，不是解释'),
			narration: {
				en: 'Remove one friendship at a time, re-run the GAT and watch member {focus}’s score gap, the Officer’s score − Mr. Hi’s ({gap}: {side}). {story} {club}',
				zh: '一次去掉一段朋友关系，重新运行整个 GAT，看成员 {focus} 的分数差（干事的分数 − Mr. Hi 的；现在是 {gap}：{side}）怎么变。{story}{club}',
				vars: (_rt, at) => {
					const m = M(at);
					const l2 = m.same2 ? I(`, though layer 2 gives member ${m.topE} the largest α, ${f(m.a2TopE)}`, `，而第 2 层确实把最大的 α（${f(m.a2TopE)}）给了成员 ${m.topE}`) : I(`; in layer 2 the largest α, ${f(m.aTopA2)}, goes to member ${m.topA2}`, `；第 2 层最大的 α（${f(m.aTopA2)}）给了成员 ${m.topA2}`);
					const story = !m.friends ? I(`Member ${m.node} has no friendships to remove.`, `成员 ${m.node} 没有可以去掉的朋友关系。`)
						: m.friends === 1 ? I(`With one friend, member ${m.topA} has the largest α and the largest effect at once: removing that friendship moves the gap by ${f(m.dTopA)}.`, `只有一个朋友时，成员 ${m.topA} 既是 α 最大的，也是影响最大的：去掉这段朋友关系，分数差变化 ${f(m.dTopA)}。`)
						: m.same ? I(`Here they agree: member ${m.topA} gets layer 1’s largest α (${f(m.aTopA)}), and removing that friendship moves the gap most, by ${f(m.dTopA)}${m.same2 ? '; layer 2 agrees too' : `; in layer 2 the largest α, ${f(m.aTopA2)}, goes to member ${m.topA2}`}.`, `这里两者一致：成员 ${m.topA} 得到第 1 层最大的 α（${f(m.aTopA)}），去掉这段朋友关系时分数差变化也最大，为 ${f(m.dTopA)}${m.same2 ? '；第 2 层也一致' : `；第 2 层最大的 α（${f(m.aTopA2)}）给了成员 ${m.topA2}`}。`)
						: I(`Member ${m.topA} gets layer 1’s largest α (${f(m.aTopA)}), yet removing that friendship moves the gap by ${f(m.dTopA)}; removing the one with member ${m.topE} (α = ${f(m.aTopE)}) moves it by ${f(m.dTopE)}${m.dTopE * m.gap < 0 && Math.abs(m.dTopE) > Math.abs(m.gap) ? ` and switches member ${m.node}’s side` : ''}${l2.en}.`, `成员 ${m.topA} 得到第 1 层最大的 α（${f(m.aTopA)}），可去掉这段朋友关系，分数差变化 ${f(m.dTopA)}；去掉和成员 ${m.topE} 的朋友关系（α = ${f(m.aTopE)}），分数差变化 ${f(m.dTopE)}${m.dTopE * m.gap < 0 && Math.abs(m.dTopE) > Math.abs(m.gap) ? `，让成员 ${m.node} 换到了另一派` : ''}${l2.zh}。`);
					const club = m.members ? I(`Over the ${m.members} members with two or more friends, the largest-move friend has layer 1’s largest α for ${m.agree * 2 < m.members ? 'only ' : ''}${m.agree}, and layer 2’s for ${m.agree2}.`, `在有两个及以上朋友的 ${m.members} 个成员里，变化最大的朋友恰好是第 1 层 α 最大的朋友，这样的成员${m.agree * 2 < m.members ? '只有' : '有'} ${m.agree} 个；换成第 2 层的 α，有 ${m.agree2} 个。`) : I('No member has two or more friends, so there is nothing to compare across the club.', '没有哪个成员有两个及以上朋友，所以无法在整个俱乐部里比较。');
					return { focus: m.node, gap: m.gap, side: m.gap > 0 ? I('the Officer’s side', '干事一侧') : I('Mr. Hi’s side', 'Mr. Hi 一侧'), story, club };
				}
			},
			task: { prompt: I('Click members until you find one with two or more friends whose ★ (largest α in layer 1) and ◆ (largest move) mark the same friend.', '点击不同的成员，找到一个有两个及以上朋友、而且 ★（第 1 层 α 最大）和 ◆（变化最大）标在同一个朋友上的成员。'), done: (_rt, at) => M(at).same === 1 && M(at).friends >= 2, hint: I('Members with just two friends are a good place to start; “Member in focus” under “Your input” jumps to any member.', '可以先从只有两个朋友的成员找起；“你的输入”里的“关注的成员”可以直接跳到任何成员。'), waiting: I('Not yet: find a member with two or more friends whose ★ and ◆ mark the same friend.', '还没有：要找一个有两个及以上朋友、★ 和 ◆ 标在同一个朋友上的成员。') },
			look: I('★ largest α in layer 1, ◆ largest move. A weight says how much one layer mixed a message in, not how much the final answer needs it: it is a clue, and removing the friendship is the test.', '★ 第 1 层 α 最大，◆ 变化最大。权重说明的是一层把一条消息混进来多少，而不是最终答案有多需要它：它是线索，去掉这段朋友关系才是检验。'), terms: ['attention']
		}
	],
	fixtures: [
		{ id: 'club', label: I('Zachary’s club, member 8 in focus', 'Zachary 的俱乐部，关注成员 8'), raw: KARATE },
		{ id: 'none', label: I('no friendships at all', '完全没有朋友关系'), raw: { ...KARATE, edges: [] } },
		{ id: 'bridge', label: I('one more friendship, 16–33', '多一段朋友关系：16–33'), raw: { ...KARATE, edges: [...KARATE.edges, [16, 33]] } },
		{ id: 'hub', label: I('focus on the Officer, member 33', '关注干事，即成员 33'), raw: { ...KARATE, focus: 33 } },
		{ id: 'leader', label: I('focus on Mr. Hi, member 0', '关注 Mr. Hi，即成员 0'), raw: { ...KARATE, focus: 0 } }
	]
};

export default lesson;
