// hmm: the guided lesson, ten steps on the learner's days and tables. 1 the model (tables), 2 making up days ('sample'),
// 3-5 the forward algorithm ('fwd': one guessed path, one cell worked out, the whole lattice), 6-8 Viterbi ('vit': one
// cell, the trace back, the task), 9 the limits step ('spells': the Markov assumption), 10 the comparison with the
// char-RNN (facts.json). Every number is a {placeholder} bound to a live frame or built in vars() from one, or a
// {fact:...}; the other factual sentences are claims.test.ts. A one-day input has no day 1: steps 3, 4 and 6 then say
// so in a sentence of their own (built in vars()).
import type { At, ExplainerRuntime, I18n, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { ACTS, DID, FREE, I, obsOf, percentOf, START_OBS, STATES } from './hmm.ts';
import { daysText, latOf, pathText } from './views/lattice/meta.ts';

type Step = StepDef<ExplainerRuntime>;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
/** a table entry as set (0.75, not 0.750) */
const q = (v: number) => (Number.isFinite(v) ? String(Math.round(v * 100) / 100) : NaN);
/** a number as the narration prints it, in both languages (for sentences built in vars) */
const fv = (v: number): I18n => I(formatValue(v, undefined, 'en') ?? '…', formatValue(v, undefined, 'zh') ?? '…');
const name = (s: number): I18n | number => STATES[s] ?? NaN;
const lattice = (show: string) => ({ key: 'hmm/lattice', props: () => ({ show }) });
const model = (show: string) => ({ key: 'hmm/model', props: () => ({ show }) });
const memory = (show: string) => ({ key: 'hmm/memory', props: () => ({ show }) });
const day = I('day', '第 {n} 天');
/** the worked-out cell: day 1, rainy (a one-day input has none: then day 0, rainy) */
const cellAt = (rt: ExplainerRuntime) => (obsOf(rt.input).length > 1 ? { phase: 'cell', step: 1, sub: 1 } : { phase: 'init', step: 0, sub: 1 });
/** the activities as a list: "walk, walk, shop" */
const acts = (rt: ExplainerRuntime): I18n | number => {
	const o = obsOf(rt.input);
	return o.length ? I(o.map((x) => ACTS[x].en).join(', '), o.map((x) => ACTS[x].zh).join('、')) : NaN;
};

const steps: Step[] = [
	{
		id: 'story', source: 'fwd', at: 'first', view: model('story'), provenance: 'live', title: I('Hidden weather, seen activities', '看不见的天气，看得见的活动'),
		narration: {
			en: 'A friend far away tells you only what she did each day: {acts}; her weather stays hidden, but it shapes what she does. A hidden Markov model writes this as two tables and a start row. The transition table says how the weather moves: a sunny day is followed by another with probability {aSS}. The emission table says what she does in each weather: she walks with probability {bSW} on a sunny day and {bRW} on a rainy one.',
			zh: '远方的朋友只告诉你她每天做了什么：{acts}；她那里的天气是看不见的，但天气影响她做什么。隐马尔可夫模型用两张表和一行初始概率把这件事写下来。转移表说明天气怎样变化：晴天之后还是晴天的概率是 {aSS}。发射表说明每种天气下她做什么：晴天她有 {bSW} 的概率去散步，雨天只有 {bRW}。',
			vars: (rt) => {
				const tb = percentOf(rt.input);
				return { acts: acts(rt), aSS: q(tb.A[0][0] / 100), bSW: q(tb.B[0][0] / 100), bRW: q(tb.B[1][0] / 100) };
			}
		},
		look: I('The two circles are the hidden weather states, the arrows carry the transition probabilities and the bars the emission probabilities; her days have a question mark where the weather should be.', '两个圆圈是隐藏的天气状态，箭头上是转移概率，横条是发射概率；她的每一天下面，天气的位置是一个问号。'), terms: ['hidden-markov-model', 'transition-table', 'emission-table']
	},
	{
		id: 'sample', source: 'sample', at: { phase: 'day', step: 0 }, view: model('sample'), provenance: 'live',
		play: { from: { phase: 'day', step: 0 }, to: { phase: 'done' }, interval: 700, unit: day }, title: I('Run forward, the model makes up days', '让模型往下走，它会编出日子'),
		narration: {
			en: 'Run forward, the model is a recipe for making up days: draw a random number u between 0 and 1; for day 0 it is {u}, inside the {s0} part of the start row, so day 0 is {s0}. A second draw, {v}, picks her activity from that weather’s emission row: she {did}. Every later day draws its weather from the transition row of the day before, and Play makes up {n} days.',
			zh: '让模型自己往下走，它就是一份编造日子的配方：抽一个 0 到 1 之间的随机数 u，第 0 天抽到 {u}，落在初始那一行里“{s0}”的区间，所以第 0 天是{s0}。再抽一个数 {v}，从这种天气的发射那一行里选出她的活动：她{did}。之后每一天，都从前一天天气的转移那一行里抽天气，播放会编出 {n} 天。',
			vars: (_rt, at) => ({ u: m(at, 'u'), s0: name(m(at, 's')), v: m(at, 'v'), did: DID[m(at, 'act')] ?? NaN, n: at.tape && at.tape.length ? ((at.tape.opt as { days?: number })?.days ?? NaN) : NaN })
		},
		look: I('The two number lines: each draw lands in one interval of its row, and the interval’s width is that row’s probability.', '两条数轴：每次抽到的数落在这一行的某个区间里，区间的宽度就是这一行里的概率。'), terms: ['sampling']
	},
	{
		id: 'path', source: 'fwd', at: 'first', view: lattice('guess'), provenance: 'live', title: I('One guess of the weather, one product', '猜一条天气路径，算一个乘积'),
		narration: {
			en: 'Back to her real days, with a first guess: each day gets the weather under which her activity is likelier. The chance that this whole weather path happens together with her activities is a product of table entries: {pi} × {b0} for day 0 (start, then emission){rest}, {joint} in all. But this is only one of many weather paths: how many are there?',
			zh: '回到她真实的日子，先猜一次：每天都选那种让她当天的活动更可能发生的天气。这整条天气路径和她的活动一起发生的概率，是表格里各项的乘积：第 0 天是 {pi} × {b0}（先初始，再发射）{rest}，合计 {joint}。可这只是众多天气路径中的一条：一共有多少条？',
			vars: (_rt, at) => ({ pi: q(m(at, 'pi')), b0: q(m(at, 'b0')), joint: m(at, 'joint'), rest: m(at, 'n') > 1 ? I(`, then × ${q(m(at, 'a1'))} × ${q(m(at, 'b1'))} for day 1 (a transition, then an emission), and so on`, `，第 1 天再乘 ${q(m(at, 'a1'))} × ${q(m(at, 'b1'))}（先转移，再发射），依此类推`) : I('', '') })
		},
		look: I('The outlined cells and arrows are the guessed path; the formula multiplies along it.', '加了框的格子和箭头就是猜的这条路径；公式沿着它一路相乘。'),
		predict: { id: 'hmm-paths', question: I('For n days and two kinds of weather, how many different weather paths are there?', '一共 n 天、两种天气，有多少条不同的天气路径？'), answer: 2,
			options: [I('n: one per day', 'n 条：每天一条'), I('2 × n: two per day', '2 × n 条：每天两条'), I('2ⁿ: two choices on every day, multiplied', '2ⁿ 条：每天两种选择，连乘起来')],
			explain: I('Every day can be sunny or rainy whatever the other days are, so the choices multiply: 2 × 2 × … × 2. Ten days already give more than a thousand paths, each with its own product, so listing them all soon becomes hopeless.', '每天都可以是晴天或雨天，与其他天无关，所以选择要连乘：2 × 2 × … × 2。十天就已经有一千多条路径，每条都有自己的乘积，一条条列出来很快就行不通了。') },
		terms: ['emission-table']
	},
	{
		id: 'forward', source: 'fwd', at: cellAt, view: lattice('forward'), provenance: 'live', title: I('The forward algorithm: one cell', '前向算法：一个格子'),
		narration: {
			en: 'The forward algorithm adds up every path without listing them. Each cell α holds the probability of her activities so far, over all paths that end in that weather on that day. A day-0 cell is the start probability times the emission. {cell}',
			zh: '前向算法不用一条条列出路径，就能把所有路径加起来。每个格子 α 存的是“到这一天为止她的活动”的概率，汇总了所有在这一天以这种天气结尾的路径。第 0 天的格子是初始概率乘发射概率。{cell}',
			vars: (_rt, at) => ({
				cell: at.frame?.phase === 'cell'
					? I(`Every later cell sums its two ways in, each a cell of the day before times a transition, then multiplies by the emission; for day 1, rainy: α₁(Rainy) = (${fv(m(at, 'p0')).en} × ${q(m(at, 'a0'))} + ${fv(m(at, 'p1')).en} × ${q(m(at, 'a1'))}) × ${q(m(at, 'b'))} = ${fv(m(at, 'alpha')).en}.`, `之后的每个格子，把两条进来的路（前一天的格子乘转移概率）加起来，再乘发射概率；以第 1 天的雨天为例：α₁(雨天) = (${fv(m(at, 'p0')).zh} × ${q(m(at, 'a0'))} + ${fv(m(at, 'p1')).zh} × ${q(m(at, 'a1'))}) × ${q(m(at, 'b'))} = ${fv(m(at, 'alpha')).zh}。`)
					: I('With a single day there is no day before, so these two cells are the whole lattice.', '只有一天，就没有“前一天”，所以这两个格子就是整张网格。')
			})
		},
		look: I('The dashed cell is the one being filled and the two thick arrows are its ways in; the formula multiplies each by its transition probability.', '虚线框的格子正在填写，指向它的两条粗箭头就是进来的两条路；公式把每条路乘上各自的转移概率。'), terms: ['forward-algorithm']
	},
	{
		id: 'fill', source: 'fwd', at: 'last', view: lattice('fill'), provenance: 'live',
		play: { from: { phase: 'init', step: 0, sub: 0 }, to: { phase: 'end' }, interval: 450, unit: day }, title: I('The whole lattice, and the probability of everything she did', '整张网格，以及她所做一切的概率'),
		narration: {
			en: 'Play fills the whole lattice: {cells} cells of a few products each, instead of {paths} products, one per path. The last column adds up to {ev}, the probability of all her activities with every weather path counted. {shrink} Given all she did, her last day was sunny with probability {sunny|pct}.',
			zh: '播放会填满整张网格：一共 {cells} 个格子，每个只要几次乘法，而不必为 {paths} 条路径各算一个乘积。最后一列加起来是 {ev}，也就是把所有天气路径都算进去之后，她全部活动的概率。{shrink}根据她做过的所有事，最后一天是晴天的概率是 {sunny|pct}。',
			vars: (_rt, at) => ({ cells: 2 * m(at, 'n'), paths: m(at, 'paths'), ev: m(at, 'evidence'), sunny: m(at, 's0'), shrink: pick(m(at, 'n') > 1, I('The numbers shrink day by day because each multiplies more probabilities below 1.', '数字一天比一天小，因为每多一天就要多乘一些小于 1 的概率。'), I('With more days the numbers would shrink, each day multiplying more probabilities below 1.', '天数越多，数字就越小，因为每多一天就要多乘一些小于 1 的概率。')) })
		},
		look: I('A cell’s color is its share of its day (stronger color, larger share); the formula adds up the last column.', '格子的颜色表示它在当天所占的比例（颜色越浓，比例越大）；公式把最后一列加起来。'), terms: ['forward-algorithm']
	},
	{
		id: 'viterbi', source: 'vit', at: cellAt, view: lattice('viterbi'), provenance: 'live', title: I('Viterbi: keep only the best way in', '维特比算法：只留下最好的那条路'),
		narration: {
			en: 'Now the question changes: not the total, but the single most likely weather path. The Viterbi algorithm fills the same lattice with one change: each cell δ keeps only the larger of its ways in (max instead of sum) and remembers which one won, a back-pointer. {cell}',
			zh: '现在问题变了：不再求总和，而是找出最可能的那一条天气路径。维特比算法填的是同一张网格，只改一处：每个格子 δ 只保留进来的路中较大的那条（用最大值代替求和），并记住是哪条胜出，这就是回溯指针。{cell}',
			vars: (_rt, at) => ({
				cell: at.frame?.phase === 'cell'
					? I(`For day 1, rainy: δ₁(Rainy) = max(${fv(m(at, 'c0')).en}, ${fv(m(at, 'c1')).en}) × ${q(m(at, 'b'))} = ${fv(m(at, 'delta')).en}, and the pointer goes back to ${STATES[m(at, 'from')]?.en ?? '…'}.`, `第 1 天的雨天：δ₁(雨天) = max(${fv(m(at, 'c0')).zh}, ${fv(m(at, 'c1')).zh}) × ${q(m(at, 'b'))} = ${fv(m(at, 'delta')).zh}，指针指回${STATES[m(at, 'from')]?.zh ?? '…'}。`)
					: I('With a single day, the two day-0 cells are all there is.', '只有一天时，第 0 天的两个格子就是全部。')
			})
		},
		look: I('The thick arrow into the dashed cell is the winner; the dashed arrow lost and is forgotten.', '指向虚线框格子的粗箭头是胜出的路；虚线箭头输了，被丢掉。'),
		predict: { id: 'hmm-survivors', question: I('Each cell keeps one way in. After any day, how many partial weather paths are still in the running?', '每个格子只留一条进来的路。任何一天过后，还有多少条“半截”天气路径在竞争？'), answer: 1,
			options: [I('Just one: the best so far', '只有一条：目前最好的那条'), I('One per weather state, so two', '每种天气一条，所以是两条'), I('All of them: 2ᵗ after t days', '全部：t 天后有 2ᵗ 条')],
			explain: I('One per cell: whatever comes after a day only cares which weather that day had, so a worse path into the same cell can never become the best and is dropped. That is why Viterbi needs about days × states² products, not one per path.', '每个格子一条：一天之后发生什么，只取决于这一天是什么天气，所以进入同一格子的较差路径永远不可能变成最好的，可以直接丢掉。正因为如此，维特比算法只需要大约“天数 × 状态数²”次乘法，而不是每条路径算一遍。') },
		terms: ['viterbi']
	},
	{
		id: 'back', source: 'vit', at: 'last', view: lattice('back'), provenance: 'live',
		play: { from: { phase: 'end' }, to: { phase: 'done' }, interval: 600, unit: day }, title: I('Trace back the best path', '沿回溯指针找回最好的路径'),
		narration: {
			en: 'The best cell of the last day has δ = {pBest}. Following the back-pointers from it to day 0 gives the decoded weather: {path}. This one path holds {share|pct} of the probability of her activities: it is the likeliest single path, {sure}. {verdict}',
			zh: '最后一天最好的格子 δ = {pBest}。从它出发沿回溯指针一直走回第 0 天，就得到解码出的天气：{path}。这一条路径占了她全部活动概率的 {share|pct}：它是最可能的单条路径，{sure}。{verdict}',
			vars: (_rt, at) => {
				const L = latOf(at.tape, at.index), dd = m(at, 'diff'), when = daysText(L.path, L.naive);
				const verdict = !Number.isFinite(dd) ? NaN : dd > 0
					? I(`It differs from reading each day’s activity alone on ${dd === 1 ? 'day' : 'days'} ${when.en}: there, the transitions to and from the neighbouring days outweigh the hint of a single activity.`, `它和逐天只看活动的结果在第 ${when.zh} 天不同：在那里，和前后几天之间的转移概率，比这一天的活动所给的提示更有分量。`)
					: I('It agrees with reading each day’s activity alone.', '它和逐天只看活动的结果一致。');
				const sure = pick(m(at, 'share') < 0.5, I('yet far from certain', '但远谈不上确定'), I('and here more likely than all the other paths together', '而且在这里，它比其余所有路径加起来还要可能'));
				return { sure: Number.isFinite(m(at, 'share')) ? sure : NaN, pBest: m(at, 'pBest'), path: L.path.length && L.path.every((s) => s >= 0) ? pathText(L.path) : NaN, share: m(at, 'share'), verdict };
			}
		},
		look: I('The outlined cells and thick arrows are the decoded path; the bottom row is the weather each activity suggests on its own, and the days where the two rows disagree are outlined.', '加框的格子和粗箭头是解码出的路径；最下面一行是只看每天的活动得出的天气，两行不一致的日子加了框。'), terms: ['viterbi']
	},
	{
		id: 'task', source: 'vit', at: 'last', view: lattice('task'), provenance: 'live', title: I('Your turn: make Viterbi change its mind', '轮到你了：让维特比改变主意'),
		narration: {
			en: 'Viterbi balances two pulls. The transition table rewards keeping the same weather; the emission table rewards the weather that fits each activity. Move the sliders in the stage until the decoded weather changes on at least one day. Right now it differs from the starting tables’ answer on {changed} of {n}.',
			zh: '维特比算法在两股力量之间权衡：转移表奖励天气保持不变，发射表奖励最贴合每天活动的天气。拖动舞台里的滑块，直到至少有一天的解码天气发生变化。现在，它和起始表格的答案在 {n} 天中有 {changed} 天不同。',
			vars: (_rt, at) => ({ changed: m(at, 'changed'), n: plural(m(at, 'n'), { one: '{n} day', many: '{n} days', zh: '{n}' }) })
		},
		look: I('The first path row follows your sliders; days where it differs from the starting tables’ row are outlined.', '第一行路径随你的滑块变化；和起始表格那一行不同的日子会加上框。'),
		task: { prompt: I('Move the sliders until Viterbi’s path differs from the starting tables’ path on at least one day.', '拖动滑块，直到维特比路径至少有一天和起始表格的路径不同。'), done: (_rt, at) => m(at, 'changed') > 0,
			hint: I('Make the weather stickier (raise Sunny → Sunny and Rainy → Rainy), or make cleaning likelier on sunny days.', '让天气更“黏”（调高 晴天 → 晴天 和 雨天 → 雨天），或者让晴天打扫的概率更高。'), waiting: I('Not yet: the decoded weather is still the same as with the starting tables.', '还没有：解码出的天气和用起始表格时还是一样。'),
			doneText: I('Done: the decoded weather changed. The step’s numbers now describe your tables.', '完成：解码出的天气变了。这一步的数字现在描述的是你的表格。') },
		terms: ['transition-table', 'emission-table']
	},
	{
		id: 'limits', source: 'spells', at: 'last', view: memory('limits'), provenance: 'live', kindOf: 'limits', title: I('Limits: the Markov assumption', '局限：马尔可夫假设'),
		narration: {
			en: 'An HMM assumes tomorrow’s weather depends only on today’s, not on how long it has rained. Take {n} made-up days in which every rainy spell lasts exactly three days, and count its transitions: rain stays rain on {aRR} of rainy days, which gets the average spell right ({mean|int} days). Yet that counted table makes a one-day spell the likeliest ({p1}) and gives exactly three days only {p3}.',
			zh: '隐马尔可夫模型假定明天的天气只取决于今天，而与已经下了几天雨无关。拿 {n} 天编出来的日子，其中每段连雨天都正好 3 天，数一数其中的转移：雨天之后还是雨天的比例是 {aRR}，由此得出的连雨天平均长度也对得上（{mean|int} 天）。可按这张数出来的表，最可能的连雨天只有 1 天（{p1}），正好 3 天的只占 {p3}。',
			vars: (_rt, at) => {
				const c = at.tape && at.tape.length ? (at.tape.frame(0).metrics ?? {}) : {};
				return { n: c.n ?? NaN, aRR: m(at, 'aRR'), mean: m(at, 'meanModel'), p1: m(at, 'p1'), p3: m(at, 'p3') };
			}
		},
		look: I('In the bar chart, every real spell sits at three days, while the bars of the counted table fall from one day on.', '在柱状图里，实际的连雨天全都在 3 天处，而按数出来的表算出的柱子从 1 天开始一路下降。'), terms: ['markov-assumption']
	},
	{
		id: 'rnn', source: 'spells', at: 'last', view: memory('rnn'), provenance: 'precomputed', title: I('Compare: a recurrent network’s memory', '对比：循环神经网络的记忆'),
		narration: {
			en: 'An HMM could count rainy days only with more states (Rainy-1, Rainy-2, Rainy-3). A recurrent network carries many numbers instead: this site’s char-RNN passes {fact:charRnnHidden} numbers in its hidden state h, and as many in its cell state c, from character to character, and its unit {fact:counterUnit} (counting from 0) learned to count the letters of the current word (correlation {fact:counterR}). The price: {fact:charRnnParams} parameters learned from data, against the {free} free numbers of this HMM’s tables (a row’s last entry is 1 minus the rest), all readable.',
			zh: '隐马尔可夫模型要数雨下了几天，只能增加状态（雨天第 1 天、第 2 天、第 3 天）。循环神经网络则带着许多数往下传：本站的 char-RNN 每读一个字符，就把隐藏状态 h 里的 {fact:charRnnHidden} 个数和细胞状态 c 里同样多的数传下去，其中单元 {fact:counterUnit}（从 0 数起）学会了数当前单词已有几个字母（相关系数 {fact:counterR}）。代价是：{fact:charRnnParams} 个从数据中学出来的参数，而这个隐马尔可夫模型的表格里只有 {free} 个自由数（每行最后一项等于 1 减去其余各项），每个都能直接读懂。',
			vars: () => ({ free: FREE })
		},
		look: I('The two cards side by side; the link opens the char-RNN page.', '两张卡片并排对比；链接会打开 char-RNN 页面。'), terms: ['recurrent-network', 'lstm', 'hidden-state', 'cell-state']
	}
];

const lesson: LessonDef = {
	title: I('A hidden Markov model: guessing the weather from what she did', '隐马尔可夫模型：从她做的事猜天气'),
	summary: I('Two tables describe hidden weather and the activities it causes. Make up days with them, add up every weather path with the forward algorithm, find the best one with Viterbi, change the tables until Viterbi changes its mind, and see what the Markov assumption cannot express.', '两张表描述了隐藏的天气和它引起的活动。用它们编出日子，用前向算法把所有天气路径加起来，用维特比算法找出最好的那条，修改表格直到维特比改变主意，再看看马尔可夫假设表达不了什么。'),
	steps,
	fixtures: [
		{ id: 'start', label: I('Eight mixed days (the page’s start)', '有晴有雨的八天（本页的起始输入）'), raw: { obs: START_OBS } },
		{ id: 'clean-sunny', label: I('Same days, cleaning likelier on sunny days', '同样的日子，晴天更常打扫'), raw: { obs: START_OBS, B: [[40, 25, 35], [10, 30, 60]] } },
		{ id: 'two', label: I('Two days: clean, then walk', '两天：先打扫，再散步'), raw: { obs: [2, 0] } }
	]
};

export default lesson;
