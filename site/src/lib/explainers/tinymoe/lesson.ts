// tinymoe: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (cfg()), live values from
// the trace of the learner's text ('fwd') and from the dispatch process, measured ones from facts.json ({fact:...}).
// Claims that are not computed live are tested in claims.test.ts.
//   1 experts instead of one MLP → 2 the router's scores → 3 softmax → 4 the top 2 (predict) → 5 the weighted sum →
//   6 who handles what (predict) → 7 the balancing loss → 8 capacity (limits, task) → 9 a dense twin → 10 the next character
import type { At, I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtSetting, paren } from '../../atlas/core/format.ts';
import { cfg,chances, charOf, dispatchAll, focusT, idsOf, routeOf, row, spaceExpert, T, type RT } from './routing.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
/** A list of numbers, joined as each language writes a list. */
const list = (v: string[]): I18n => ({ en: v.join(', '), zh: v.join('、') });
const num = (x: number) => (Number.isFinite(x) ? x.toFixed(2).replace('-', '−') : '…');

/** Block 1, the followed character: its route, and softmax worked out for the runner-up expert b. */
function r1(rt: RT) {
	const { z, p, t, c, a, b, pa, pb, ga, gb } = routeOf(rt, 1, focusT(rt));
	const r = { t, c, a, b, pa, pb, ga, gb };
	const M = Math.max(...z);
	const ex = z.map((v) => Math.exp(v - M));
	const sum = ex.reduce((a, b) => a + b, 0);
	return { ...r, ...cfg(rt), za: z[r.a] ?? NaN, zb: z[r.b] ?? NaN, zs: list(z.map(num)), ps: list(p.map((v) => `${(100 * v).toFixed(1)}%`)), exb: ex[r.b] ?? NaN, sum };
}

/** Step 5: the number of the followed character that the MoE layer changes most, and its weighted sum. */
function mixOf(rt: RT) {
	const r = r1(rt);
	const y = row(rt, 'B1.moe', r.t);
	const dd = y.reduce((bi, v, i) => (Math.abs(v) > Math.abs(y[bi]) ? i : bi), 0);
	const ex = (e: number) => T(rt, 'B1.moe#e')?.data[(r.t * r.E + e) * r.d + dd] ?? NaN;
	const pr = (v: number) => (Number.isFinite(v) ? paren(v) : NaN);
	return { ...r, dd, xa: pr(ex(r.a)), xb: pr(ex(r.b)), y: y[dd] ?? NaN };
}

/** Step 6: block 2's space expert (held-out stats) and what it took in the learner's text. */
function whoOf(rt: RT) {
	const s = spaceExpert(rt, 2);
	const ids = idsOf(rt);
	let [sp, nsp, total] = [0, 0, 0];
	ids.forEach((_, t) => {
		const took = (row(rt, 'B2.moe#gate', t)[s.e] ?? 0) > 0 ? 1 : 0;
		total += took;
		if (charOf(rt, t) === '␣') (nsp++, (sp += took));
	});
	const n = ids.length;
	const sps = nsp === 1 ? 'space' : 'spaces';
	const chs = n === 1 ? 'character' : 'characters';
	const yours = pick(nsp > 0, { en: `In your text it took ${sp} of ${nsp} ${sps} and ${total} of your ${n} ${chs} in all.`, zh: `在你的文字里，它接下了 ${nsp} 个空格中的 ${sp} 个，总共接下你 ${n} 个字符中的 ${total} 个。` }, { en: `Your text has no space; it took ${total} of your ${n} ${chs}.`, zh: `你的文字里没有空格；它接下了你 ${n} 个字符中的 ${total} 个。` });
	return { e: s.e, yours };
}

/** Steps 7 and 8: the metrics of the step's dispatch frame (the last one: the balancing loss). */
const metricsOf = (at: At) => (at.frame?.metrics ?? {}) as Record<string, number>;

/** The capacity rule on block 1 of the learner's text (the same as the dispatch process), for the task. */
export const drops = (rt: RT) => (idsOf(rt).length ? dispatchAll(chances(rt, 1), cfg(rt).K, 1) : null);

const lesson: LessonDef = {
	title: L('Tiny mixture of experts: each character picks 2 of 4 experts', '小型混合专家：每个字符从 4 个专家中选 2 个'),
	summary: L(
		'Follow one character through a tiny Transformer whose MLPs are mixtures of 4 experts: the router scores the experts, softmax turns the scores into chances, the top 2 run and their outputs are mixed by the gates; then see which characters each expert takes, why training balances the load, what a capacity limit drops, and how the model compares with a dense MLP of about the same active size.',
		'跟着一个字符走过一个小 Transformer，它的 MLP 换成了 4 个专家的混合：路由器给专家打分，softmax 把分数变成概率，前 2 名运行，它们的输出按门控混合；接着看每个专家接哪些字符、训练为什么要均衡负载、容量限制会丢掉什么，以及它和每个字符用到的参数差不多的稠密 MLP 比起来如何。'
	),
	steps: [
		{
			id: 'experts',
			source: 'fwd',
			at: { layer: 'B1.moe' },
			view: { key: 'tinymoe/route', props: () => ({ mode: 'grid' }) },
			provenance: 'live',
			title: L('Four experts instead of one MLP', '用四个专家代替一个 MLP'),
			narration: {
				en: 'In an ordinary Transformer every character passes through the same MLP in each block. Here each of the {L} blocks holds {E} smaller MLPs, the experts, and a router that sends each character to just {K} of them. The model has {fact:params|int} parameters, but one character uses only {fact:active|int}.',
				zh: '在普通的 Transformer 里，每一块中所有字符都经过同一个 MLP。这里 {L} 块中的每一块都有 {E} 个更小的 MLP，也就是专家，还有一个路由器，只把每个字符送给其中 {K} 个。模型共有 {fact:params|int} 个参数，但一个字符只用到 {fact:active|int} 个。',
				vars: (rt) => cfg(rt)
			},
			look: { en: 'Under each character: the {K} experts that block 1 (top row) and block 2 (bottom row) chose.', zh: '每个字符下面：第 1 块（上一行）和第 2 块（下一行）选中的 {K} 个专家。', vars: (rt) => cfg(rt) },
			terms: ['mixture-of-experts', 'mlp', 'transformer', 'parameter']
		},
		{
			id: 'scores',
			source: 'fwd',
			at: { layer: 'B1.router' },
			select: (rt) => ({ layer: 'B1.router', unit: [Math.max(0, focusT(rt)), Math.max(0, r1(rt).a)] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: L('The router scores every expert', '路由器给每个专家打分'),
			narration: {
				en: 'Follow “{c}”, the last character of your text, into block 1. After layer normalization the character is {d} numbers; the router, a linear layer without a bias, multiplies them by {d} learned weights per expert and adds up the products. Expert {a} scores {za}; the {E} scores are {zs}.',
				zh: '跟着你文字的最后一个字符“{c}”进入第 1 块。经过层归一化后，这个字符是 {d} 个数；路由器是一个没有偏置的线性层，它把这些数和每个专家的 {d} 个学来的权重逐个相乘，再把乘积加起来。专家 {a} 得分 {za}；{E} 个分数是 {zs}。',
				vars: (rt) => r1(rt)
			},
			look: { en: 'The {d} products for expert {a} add up to its score, {za}.', zh: '专家 {a} 的 {d} 个乘积加起来就是它的分数 {za}。', vars: (rt) => r1(rt) },
			terms: ['router', 'layer-norm', 'logit']
		},
		{
			id: 'chances',
			source: 'fwd',
			at: { layer: 'B1.route' },
			view: { key: 'tinymoe/route', props: () => ({ mode: 'token', block: 1, softmax: true }) },
			provenance: 'live',
			title: L('Softmax turns scores into chances', 'softmax 把分数变成概率'),
			narration: {
				en: 'Softmax turns the {E} scores into chances that add up to 1: e to the power of each score minus the largest, divided by the sum of all {E} results. For expert {b}, the runner-up: e to the power {zb} − {za} is {exb|n4}, and {exb|n4} ÷ {sum|n4} = {pb|pct}. The chances of experts 0 to 3 are {ps}.',
				zh: 'softmax 把 {E} 个分数变成加起来等于 1 的概率：每个分数先减去最大的分数，求 e 的这个次方，再除以全部 {E} 个结果的和。排第二的专家 {b}：e 的 {zb} − {za} 次方是 {exb|n4}，{exb|n4} ÷ {sum|n4} = {pb|pct}。专家 0 到 3 的概率是 {ps}。',
				vars: (rt) => r1(rt)
			},
			look: { en: 'The third column holds e^(score − largest); each chance is its row’s number divided by their sum.', zh: '第三列是 e^(分数 − 最大分数)；每个概率就是这一行的数除以这一列的和。' },
			terms: ['softmax']
		},
		{
			id: 'pick',
			source: 'fwd',
			at: { layer: 'B1.moe' },
			view: { key: 'tinymoe/route', props: () => ({ mode: 'token', block: 1 }) },
			provenance: 'live',
			title: L('Which experts run?', '哪些专家运行？'),
			narration: {
				en: 'Every expert now has a chance for “{c}”. Running an expert costs as much as running a small MLP. Which of the {E} run, and how much each one counts, is the router’s rule.',
				zh: '现在每个专家对“{c}”都有一个概率。运行一个专家的开销和运行一个小 MLP 一样。{E} 个专家里哪些运行、每个算多少分量，由路由器的规则决定。',
				vars: (rt) => r1(rt)
			},
			look: { en: 'The table lists each expert’s score, chance and gate.', zh: '表里列出每个专家的分数、概率和门控。' },
			predict: {
				id: 'tinymoe-pick',
				question: L('The router has a chance for each of the 4 experts. Which experts actually run for this character?', '路由器给 4 个专家各算了一个概率。对这个字符，真正运行的是哪些专家？'),
				options: [L('All four, each weighted by its chance', '全部四个，按各自的概率加权'), L('The two with the largest chances', '概率最大的两个'), L('Two drawn at random, likely ones more often', '随机抽两个，概率大的更常被抽到'), L('Only the one with the largest chance', '只有概率最大的那一个')],
				answer: 1,
				explain: L('Top-2 routing keeps the two largest chances and skips the other experts entirely: they do not run for this character, and that is what saves the work. The two kept chances are rescaled so that they add up to 1; these are the gates.', '前 2 名路由保留最大的两个概率，其余专家完全跳过：它们不为这个字符运行，省下的正是这部分计算。保留的两个概率会重新缩放，让它们加起来等于 1，这就是门控。')
			},
			terms: ['router', 'mixture-of-experts']
		},
		{
			id: 'mix',
			source: 'fwd',
			at: { layer: 'B1.moe' },
			select: (rt) => ({ layer: 'B1.moe', unit: [Math.max(0, focusT(rt)), mixOf(rt).dd] }),
			view: { key: 'zoom.weighted-sum', props: () => ({ names: { x: 'E', w: 'g', p: 'g·E' } }) },
			provenance: 'live',
			title: L('Two experts run, the gates mix them', '两个专家运行，门控把它们混合'),
			narration: {
				en: 'Experts {a} and {b} run: each is a small MLP ({d} → {H} → {d} with ReLU) that turns the {d} numbers of “{c}” into {d} new ones. Their gates are their chances, {pa|n3} and {pb|n3}, rescaled to add up to 1: {ga} and {gb}. For number {dd} of the output (counting from 0): {ga} × {xa} + {gb} × {xb} = {y}.',
				zh: '专家 {a} 和 {b} 运行：每个都是一个小 MLP（{d} → {H} → {d}，用 ReLU），把“{c}”的 {d} 个数变成 {d} 个新数。它们的门控是把两者的概率 {pa|n3} 和 {pb|n3} 重新缩放到和为 1 的结果：{ga} 和 {gb}。对输出里编号 {dd} 的数（从 0 数起）：{ga} × {xa} + {gb} × {xb} = {y}。',
				vars: (rt) => mixOf(rt)
			},
			look: { en: 'Two products only: the other {rest} experts add nothing and cost nothing.', zh: '只有两个乘积：另外 {rest} 个专家什么也不加，也不花任何计算。', vars: (rt) => ({ rest: cfg(rt).E - cfg(rt).K }) },
			terms: ['relu', 'mlp']
		},
		{
			id: 'who',
			source: 'fwd',
			at: { layer: 'B2.moe' },
			view: { key: 'tinymoe/route', props: (rt) => ({ mode: 'grid', mark: { block: 2, expert: spaceExpert(rt, 2).e } }) },
			provenance: 'live',
			title: L('Who handles what', '谁负责什么'),
			narration: {
				en: 'Nobody told the router how to split the work, and each character is routed on its own. Yet a trained router is far from random: one expert of block 2, expert {e}, took every space of the held-out stories. {yours}',
				zh: '没有人告诉路由器怎样分工，每个字符也都是单独分派的。可训练好的路由器远不是随机的：第 2 块的专家 {e} 接下了没见过的故事里的每一个空格。{yours}',
				vars: (rt) => whoOf(rt)
			},
			look: { en: 'Outlined: the characters block 2 sent to expert {e}.', zh: '加框的是第 2 块送给专家 {e} 的字符。', vars: (rt) => whoOf(rt) },
			predict: {
				id: 'tinymoe-who',
				question: L('Besides spaces, which characters do you expect this expert of block 2 to take?', '除了空格，你觉得第 2 块的这个专家还会接哪些字符？'),
				options: [L('Periods, commas and new lines', '句号、逗号和换行'), L('The vowels', '元音字母'), L('Capital letters', '大写字母'), L('Nothing else: one expert per character', '别的都不接：一个字符一个专家')],
				answer: 0,
				explain: L('On held-out stories it takes every space, period and comma and almost every new line: the characters that end a word or a sentence. The experts split the work by the kind of character, which is what a character-level model can see.', '在没见过的故事里，它接下了每一个空格、句号和逗号，以及几乎每一个换行：也就是结束一个词或一句话的字符。专家们按字符的种类分工，这正是字符级模型能看到的东西。')
			},
			terms: ['router']
		},
		{
			id: 'balance',
			source: 'dispatch',
			at: 'last',
			view: { key: 'tinymoe/route', props: () => ({ mode: 'load', block: 1, capacity: false }) },
			provenance: 'live',
			title: L('Training keeps the load balanced', '训练让负载保持均衡'),
			narration: {
				en: 'Nothing in the next-character loss stops the router from sending almost everything to one or two experts and wasting the rest. So training adds a balancing loss, {E} × Σ f × P, times {alpha}: f is an expert’s share of all the slots (each character fills {K}), P its mean chance. It is about 1 when the work is spread evenly and grows when one expert gets a large share and a large chance. On your text block 1 scores {aux|n4}.',
				zh: '下一个字符的损失并不阻止路由器把几乎所有字符都送给一两个专家，让其余专家闲着。所以训练时要加一个负载均衡损失 {E} × Σ f × P，再乘以 {alpha}：f 是专家分到的位置份额（每个字符占 {K} 个位置），P 是它的平均概率。活分得均匀时它约为 1；某个专家既分得多、概率又大时，它就变大。在你的文字上，第 1 块的值是 {aux|n4}。',
				vars: (rt, at) => ({ ...cfg(rt), aux: metricsOf(at).aux ?? NaN, alpha: fmtSetting(rt.num('fact:alpha')) })
			},
			look: { en: 'The table gives f, P and f × P for each expert; its last line adds them up and multiplies by {E}.', zh: '表里是每个专家的 f、P 和 f × P；最后一行把它们加起来再乘以 {E}。', vars: (rt) => cfg(rt) },
			terms: ['load-balancing-loss', 'loss']
		},
		{
			id: 'capacity',
			source: 'dispatch',
			at: 'last',
			view: { key: 'tinymoe/route', props: () => ({ mode: 'load', block: 1, table: false }) },
			play: { from: 'first', to: 'last', interval: 120, unit: L('character', '第 {n} 个字符') },
			provenance: 'live',
			kindOf: 'limits',
			title: L('When an expert is full', '当专家满了'),
			narration: {
				en: 'Many large MoE systems give every expert a fixed number of slots per batch so that the work fits the hardware: here {cap} each, just enough if the {slots} slots of your {chars} were split evenly. A slot that reaches a full expert is dropped, and the character misses that expert. Block 1 dropped {dropped} of {slots} slots of your text; on held-out stories it drops {fact:b1Drop1_0|pct}, and {fact:b1Drop1_25|pct} with a quarter more room.',
				zh: '很多大的 MoE 系统给每个专家每批固定数量的位置，好让计算装得进硬件：这里每个专家 {cap} 个，你的 {chars}共有 {slots} 个位置，平均分配时刚好够用。送到已满专家的位置会被丢掉，这个字符就少了那个专家。第 1 块在你的文字上丢掉了 {slots} 个位置中的 {dropped} 个；在没见过的故事上它丢掉 {fact:b1Drop1_0|pct}，多给四分之一的空间时丢掉 {fact:b1Drop1_25|pct}。',
				vars: (rt, at) => {
					const m = metricsOf(at);
					const n = idsOf(rt).length;
					return { cap: m.cap ?? NaN, chars: plural(n, { one: '{n} character', many: '{n} characters', zh: '{n} 个字符' }), dropped: m.dropped ?? NaN, slots: (m.k ?? NaN) * (m.T ?? NaN) };
				}
			},
			look: { en: 'Press Play: the bars fill up to the dashed line, and every × under a character is a dropped slot.', zh: '按播放：条形一直涨到虚线，字符下面的每个 × 都是一个被丢掉的位置。' },
			task: {
				prompt: L('Overload an expert: pick “One letter, over and over”, or type one letter many times, until at least 2 in 5 of the slots are dropped.', '让一个专家超载：选“同一个字母，一遍又一遍”，或者把一个字母打很多遍，直到至少五分之二的位置被丢掉。'),
				done: (rt) => {
					const d = drops(rt);
					return !!d && d.dropped / d.slots >= 0.4;
				},
				waiting: L('Not yet: fewer than 2 in 5 slots are dropped.', '还没有：被丢掉的位置不到五分之二。'),
				doneText: L('Done: when most slots go to the same two experts, those two fill up and the rest of the slots sent to them are dropped.', '完成：大多数位置都送给同样两个专家时，这两个专家很快就满了，之后送给它们的位置都被丢掉。')
			},
			terms: ['expert-capacity']
		},
		{
			id: 'dense',
			source: 'fwd',
			at: 'last',
			view: { key: 'tinymoe/route', props: () => ({ mode: 'dense' }) },
			provenance: 'precomputed',
			title: L('Is it better than one MLP?', '它比一个 MLP 更好吗？'),
			narration: {
				en: 'A dense twin replaces each mixture with one MLP of {Hd} hidden units: about as many parameters per character as two experts, {fact:denseParams|int} in all. Trained the same way, its cross-entropy loss on held-out stories was {fact:valLossDense|n4} nats per character (lower is better); the MoE, with {fact:params|int} parameters, scored {fact:valLoss|n4}, {verdict}. Without the balancing loss the MoE scored {fact:valLossNoaux|n4}, and block 2’s busiest expert took {fact:b2NoauxBusiest|pct} of the slots instead of {fact:b2Busiest|pct}.',
				zh: '稠密对照模型把每个专家混合换成一个有 {Hd} 个隐藏单元的 MLP：每个字符用到的参数和两个专家差不多，共 {fact:denseParams|int} 个。用同样的方法训练后，它在没见过的故事上的交叉熵损失是每个字符 {fact:valLossDense|n4} 奈特（越低越好）；有 {fact:params|int} 个参数的 MoE 是 {fact:valLoss|n4}，{verdict}。不加均衡损失时，MoE 是 {fact:valLossNoaux|n4}，第 2 块最忙的专家接下了 {fact:b2NoauxBusiest|pct} 的位置，而不是 {fact:b2Busiest|pct}。',
				vars: (rt) => {
					const gap = rt.num('fact:valLossDense') - rt.num('fact:valLoss');
					return { ...cfg(rt), verdict: pick(gap > 0, L(`lower by ${gap.toFixed(4)}`, `低了 ${gap.toFixed(4)}`), L(`higher by ${(-gap).toFixed(4)}`, `高了 ${(-gap).toFixed(4)}`)) };
				}
			},
			look: { en: 'The bars compare all parameters with the ones one character uses; the last column is the measured loss.', zh: '条形比较全部参数和一个字符用到的参数；最后一列是测得的损失。' },
			terms: ['cross-entropy', 'parameter']
		},
		{
			id: 'next',
			source: 'fwd',
			at: 'last',
			view: 'kit.probbars',
			provenance: 'live',
			title: L('All of it serves one guess', '这一切都为了一次猜测'),
			narration: {
				en: 'After both blocks, a last layer normalization and a linear layer score all {V} characters, and softmax turns the scores into chances for the next one. After “{c}” the model expects “{top.label}” with {top.p|pct}, then “{runnerUp.label}” with {runnerUp.p|pct}.',
				zh: '两块之后，最后一次层归一化和一个线性层给全部 {V} 个字符打分，softmax 再把分数变成下一个字符的概率。在“{c}”之后，模型最期待“{top.label}”，概率 {top.p|pct}；其次是“{runnerUp.label}”，概率 {runnerUp.p|pct}。',
				vars: (rt) => ({ c: r1(rt).c, V: rt.num('fact:vocab') })
			},
			look: { en: 'The longest bar is the model’s top guess for the character after your text.', zh: '最长的那一条是模型对你文字之后那个字符的第一猜测。' },
			terms: ['next-token-prediction', 'softmax']
		}
	]
};

export default lesson;
