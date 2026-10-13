// hmm: the manifest. A two-state hidden Markov model (the weather, Sunny or Rainy) explains the activities a friend
// reports (walk, shop, clean). Kind C only: every stage is a process of process.ts run live in the browser on the
// learner's activities and tables, checked frame by frame against an independent numpy run (pipeline/hmm/reference.py).
//   fwd      the forward algorithm on the learner's days and tables (one guessed path, then the lattice of α)
//   vit      Viterbi on the same input (δ, back-pointers, the trace back, the comparison with the starting tables)
//   sample   the learner's tables used as a story generator: SAMPLE_DAYS made-up days (seeded)
//   spells   the limits step: transition counts of a made-up stretch where rain lasts exactly three days (fixed input)
// facts.json (pipeline/hmm/facts.py) copies four measured numbers of the char-RNN page for the comparison step.
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { dataOf, I, SAMPLE_DAYS, SPELL_MAX, START, YEAR } from './hmm.ts';

const on = (process: string, input: (rt: ExplainerRuntime) => unknown, opt?: Record<string, unknown>) => ({ kind: 'process' as const, process, seed: 1, input, ...(opt ? { opt } : {}) });
/** the learner's tables as probabilities (the sampling source ignores the activities, so it keeps its run when they change) */
const tables = (rt: ExplainerRuntime) => {
	const { pi, A, B } = dataOf(rt.input);
	return { pi, A, B };
};

const manifest: ExplainerManifest = {
	slug: 'hmm',
	type: 'model',
	kinds: ['process'],
	quality: 'reviewed',
	assets: { dir: 'models/hmm', auto: ['facts.json'] },
	input: { adapter: 'days', start: START, presetsFromAdapter: true },
	sources: {
		fwd: on('forward', (rt) => dataOf(rt.input)),
		vit: on('viterbi', (rt) => dataOf(rt.input)),
		sample: on('sample', tables, { days: SAMPLE_DAYS }),
		spells: on('spells', () => ({ year: YEAR, maxLen: SPELL_MAX }))
	},
	page: {
		goals: [
			I('Name the two tables of a hidden Markov model (transition and emission) and use them to make up a sequence of hidden states and observations.', '说出隐马尔可夫模型的两张表（转移表和发射表），并用它们编出一串隐藏状态和观测。'),
			I('Fill one cell of the forward lattice by hand: the sum over the ways in, times the emission probability, and say why the lattice needs days × states² products instead of one per path.', '亲手填出前向网格的一个格子：把所有进来的路加起来，再乘发射概率；并说出为什么网格只需要“天数 × 状态数²”次乘法，而不必每条路径算一遍。'),
			I('Decode the most likely weather with Viterbi (max instead of sum, back-pointers, trace back) and explain why it can disagree with reading each day on its own.', '用维特比算法解码最可能的天气（用最大值代替求和、记下回溯指针、再回溯），并解释为什么它会和逐天单独判断的结果不同。'),
			I('Change the tables until Viterbi changes its answer, and say which pull (staying in the same state, or fitting each observation) won.', '修改表格直到维特比改变答案，并说出是哪一股力量（保持同一状态，还是贴合每天的观测）占了上风。'),
			I('Explain what the Markov assumption cannot express (a rain spell that always lasts three days) and how a recurrent network differs.', '解释马尔可夫假设表达不了什么（总是连下三天的雨），以及循环神经网络有何不同。')
		],
		prequestion: {
			id: 'hmm-pre',
			question: I('A friend far away tells you only what she did each day: walk, shop or clean. Can you say anything about her weather?', '远方的朋友只告诉你她每天做了什么：散步、购物或打扫。你能推断出她那里的天气吗？'),
			options: [I('No: the weather is never observed, so nothing can be said', '不能：天气从来没被观测到，所以什么也说不了'), I('Yes, exactly: each activity tells the weather for sure', '能，而且很确定：每种活动都准确对应一种天气'), I('Yes, as probabilities: activities hint at the weather, and weather tends to last', '能，但只能说概率：活动透露天气的线索，而且天气往往会持续')],
			answer: 2,
			explain: I('A walk is likelier on a sunny day and cleaning on a rainy one, and today’s weather tends to be tomorrow’s. A hidden Markov model turns both hints into numbers and weighs every possible weather sequence.', '晴天更可能去散步，雨天更可能在家打扫；而且今天的天气往往延续到明天。隐马尔可夫模型把这两种线索都变成数字，并权衡每一种可能的天气序列。')
		},
		tryThis: [
			{ id: 'wet', title: I('A wet spell', '连着下雨'), setup: { preset: 'wet' }, predict: I('Her days run walk, walk, shop, clean, clean, shop, clean, walk, walk. Will Viterbi call both shop days sunny, as shopping alone suggests?', '她的九天是：散步、散步、购物、打扫、打扫、购物、打扫、散步、散步。维特比会像只看购物那样，把两个购物日都判为晴天吗？'), look: I('In the trace-back step, compare the decoded path with the activity-alone row.', '在回溯那一步，对比解码出的路径和“只看活动”那一行。'), goto: { step: 'back' } },
			{ id: 'sticky', title: I('Weather that rarely changes', '很少变化的天气'), setup: { preset: 'sticky' }, predict: I('If the weather keeps for days on end, will the shop day between two clean days still be called rainy?', '如果天气一连多天不变，夹在两个打扫日之间的那个购物日还会被判为雨天吗？'), look: I('In the trace-back step, find the shop days and their decoded weather.', '在回溯那一步，找到购物的那几天，看它们被解码成什么天气。'), goto: { step: 'back' } }
		],
		quiz: [
			{ q: I('Two hidden states, observed for 10 days: how many weather paths are there, and how many cells does the forward lattice have?', '两个隐藏状态，观测 10 天：一共有多少条天气路径？前向网格有多少个格子？'), options: [I('1,024 paths and 20 cells', '1,024 条路径，20 个格子'), I('20 paths and 20 cells', '20 条路径，20 个格子'), I('1,024 paths and 1,024 cells', '1,024 条路径，1,024 个格子')], answer: 0, explain: I('Each day has 2 choices, so 2¹⁰ = 1,024 paths; the lattice has one cell per day and state, 10 × 2 = 20, and each cell sums only 2 ways in.', '每天有 2 种选择，所以有 2¹⁰ = 1,024 条路径；网格每天每个状态一个格子，10 × 2 = 20 个，每个格子只需加 2 条进来的路。') }
		],
		misconception: { claim: I('The most likely weather sequence is just the most likely weather of each day, one day at a time.', '最可能的天气序列，就是每天各自最可能的天气连起来。'), truth: I('Days are linked: the transition table makes a lone change of weather costly, so the best whole path can overrule what one day’s activity suggests. Viterbi scores whole paths.', '各天是相互关联的：转移表让天气单独变一天的代价很高，所以最好的整条路径可能推翻某一天的活动所给的提示。维特比算法给整条路径打分。'), step: 'back' },
		lessonNote: I('Ten steps on your own days and tables: the two tables, making up days, one path’s probability, the forward lattice, Viterbi with its back-pointers, changing the tables until Viterbi changes its mind, the Markov assumption, and a recurrent network for comparison.', '十步，都在你自己的日子和表格上：两张表、编出日子、一条路径的概率、前向网格、带回溯指针的维特比算法、改表格直到维特比改变主意、马尔可夫假设，以及和循环神经网络的对比。')
	},
	glossary: [
		{ id: 'hidden-markov-model', term: I('hidden Markov model (HMM)', '隐马尔可夫模型（HMM）'), def: I('A model of a sequence in which an unseen state (here the weather) changes from step to step by a transition table, and each observation (here the activity) is drawn from an emission table of the current state.', '一种序列模型：一个看不见的状态（这里是天气）按转移表逐步变化，每个观测（这里是活动）按当前状态的发射表抽出。'), related: ['transition-table', 'emission-table'] },
		{ id: 'transition-table', term: I('transition table', '转移表'), def: I('The probabilities of the next hidden state given the current one: row i, column j is P(tomorrow j | today i); each row sums to 1.', '给定当前隐藏状态时下一个状态的概率：第 i 行第 j 列是 P(明天 j | 今天 i)；每行加起来等于 1。') },
		{ id: 'emission-table', term: I('emission table', '发射表'), def: I('The probabilities of each observation given the hidden state: row j, column k is P(activity k | weather j); each row sums to 1.', '给定隐藏状态时各个观测的概率：第 j 行第 k 列是 P(活动 k | 天气 j)；每行加起来等于 1。') },
		{ id: 'forward-algorithm', term: I('forward algorithm', '前向算法'), def: I('Fills a lattice day by day: each cell α is the probability of the observations so far ending in that state, the sum over the cells of the day before (times the transition) times the emission. The last column sums to the probability of all the observations.', '逐天填满一张网格：每个格子 α 是“到目前为止的观测，且当天处在这个状态”的概率，等于前一天各格子乘转移概率之和，再乘发射概率。最后一列之和就是全部观测的概率。'), related: ['hidden-markov-model'] },
		{ id: 'viterbi', term: I('Viterbi algorithm', '维特比算法'), def: I('The forward lattice with max in place of the sum: each cell keeps the probability of the single best path into it and a back-pointer to where that path came from; tracing the pointers back from the best last cell gives the most likely hidden sequence.', '把前向网格里的求和换成取最大值：每个格子保留进入它的最优单条路径的概率，以及指向这条路径来处的回溯指针；从最后一天最好的格子沿指针往回走，就得到最可能的隐藏序列。'), related: ['forward-algorithm'] },
		{ id: 'markov-assumption', term: I('Markov assumption', '马尔可夫假设'), def: I('The next state depends only on the current state, not on how long it has lasted or what came before.', '下一个状态只取决于当前状态，与它已持续多久、之前发生过什么无关。') }
	],
	catalog: {
		hook: I('Guess the hidden weather behind a friend’s activities, with the forward and Viterbi algorithms filling their lattices on your own days.', '根据朋友的活动猜出背后隐藏的天气：前向算法和维特比算法在你自己的日子上逐格填出网格。'),
		shows: I('The transition and emission tables (editable), made-up days drawn from them, one path’s probability, the forward lattice cell by cell, Viterbi with back-pointers and the trace back, and what the Markov assumption cannot express, next to a recurrent network.', '可编辑的转移表和发射表、由它们编出的日子、一条路径的概率、逐格填写的前向网格、带回溯指针和回溯过程的维特比算法，以及马尔可夫假设表达不了的东西，并与循环神经网络对比。'),
		runtime: I('Own TypeScript, live in the browser (checked against a Python run)', '自研 TypeScript，在浏览器中实时运行（已与 Python 运行核对）'),
		params: I('7 free numbers in three small tables, set by you (nothing is learned)', '三张小表中的 7 个自由数，由你设定（没有学习）'),
		size: I('one small facts file; the tables and the days are made in your browser', '一个很小的事实文件；表格和日子都在你的浏览器里生成'),
		prereqs: []
	},
	credits: [
		{ name: 'L. R. Rabiner, “A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition” (Proceedings of the IEEE, 1989)', url: 'https://doi.org/10.1109/5.18626', license: 'Article; ideas only (no text, figures, code or data used)', usedFor: 'the forward and Viterbi recursions and their notation (α, δ, ψ)', attribution: 'Lawrence R. Rabiner' },
		{ name: 'A. J. Viterbi, “Error bounds for convolutional codes and an asymptotically optimum decoding algorithm” (IEEE Transactions on Information Theory, 1967)', url: 'https://doi.org/10.1109/TIT.1967.1054010', license: 'Article; ideas only (no text, figures, code or data used)', usedFor: 'the Viterbi algorithm', attribution: 'Andrew J. Viterbi' },
		{ name: 'Wikipedia, “Hidden Markov model” (the example of a distant friend whose activities depend on the weather)', url: 'https://en.wikipedia.org/wiki/Hidden_Markov_model', license: 'CC BY-SA 4.0; the idea of the example only (no text, figures or numbers used: the tables here are our own)', usedFor: 'the setting of the story: hidden weather behind walk, shop and clean', attribution: 'Wikipedia contributors' }
	]
};

export default manifest;
