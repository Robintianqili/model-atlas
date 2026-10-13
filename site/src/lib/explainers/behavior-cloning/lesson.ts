// behavior-cloning: the guided lesson, ten steps. Every number is a {placeholder} bound to a live frame or computed from
// the learner's input and the sources' tapes in vars(); the other sentences are claims.test.ts. A predict step's
// narration never gives its answer away (it shows before the learner answers): the stage and predict.explain do.
import type { At, ExplainerRuntime, I18n, LessonDef, StepDef, Tape, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { DAGGER, bc, datasetOf } from './trainer.ts';
import { EVAL, OPEN, featuresOf, pairsOf, policyAt, toRaw, worldOf } from './world.ts';
import { boardOf } from './views/board/meta.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
type Step = StepDef<ExplainerRuntime>;
const tapeOf = (rt: ExplainerRuntime, s: string) => rt.sources[s]?.tape as Tape | null | undefined;
const lastM = (t: Tape | null | undefined) => (t?.length ? (t.frame(t.length - 1).metrics ?? {}) : {});
const board = (show: string, more: Record<string, unknown> = {}) => ({ key: 'behavior-cloning/board', props: () => ({ show, ...more }) });
const demosOf = (rt: ExplainerRuntime) => plural(toRaw(rt.input).demos.length, { one: '{n} demonstration', many: '{n} demonstrations', zh: '{n} 条示范' });
const pairs = (rt: ExplainerRuntime) => pairsOf(toRaw(rt.input)).s.length;
const cellsN = (n: number) => plural(n, { one: '{n} cell', many: '{n} cells', zh: '{n} 个格子' });
const pairsN = (n: number) => plural(n, { one: '{n} pair', many: '{n} pairs', zh: '{n} 对' });
/** the roll tape's totals (its last frame, 'done') */
const totals = (at: At) => (at.frame?.phase === 'done' ? at.frame.metrics ?? {} : {});

const steps: Step[] = [
	{
		id: 'demos',
		source: 'train',
		at: 'first',
		view: board('demos,data'),
		provenance: 'live',
		title: I('Learning from demonstrations', '从示范中学习'),
		narration: {
			en: 'Imitation learning needs no rewards: an expert shows the way and the learner copies it. You are the expert here: the stage draws your {demos} on the field, which runs from the start S to the goal G. Each cell of a demonstration, paired with the move made there, is one training example: {pairs} in all.',
			zh: '模仿学习不需要奖励：专家示范怎么走，学习者照着学。这里你就是专家：画面里画出了你的 {demos}；这片空地的起点是 S，终点是 G。示范经过的每个格子，配上在那里走的那一步，就是一个训练样本：一共 {pairs} 个。',
			vars: (rt) => ({ demos: demosOf(rt), pairs: pairs(rt) })
		},
		look: I('Look at the outlined cells along your path: one per (cell, move) pair.', '看你的路线上加了外框的格子：每个对应一个（格子，动作）对。'),
		terms: ['imitation-learning', 'demonstration']
	},
	{
		id: 'net',
		source: 'train',
		at: 'first',
		view: board('probe,data', { first: true }),
		provenance: 'live',
		title: I('A policy network', '策略网络'),
		narration: {
			en: 'The policy is a small network. It reads a cell as two numbers, its column and row scaled to the range −1 to 1, passes them through {hidden} tanh units and gives each of the four moves a score; softmax turns the scores into probabilities. With its random starting weights it gives the outlined cell up {p0}, right {p1}, down {p2}, left {p3}.',
			zh: '策略是一个小网络。它把一个格子看成两个数：缩放到 −1 到 1 之间的列号和行号；这两个数经过 {hidden} 个 tanh 单元，给四个动作各打一个分；softmax 再把分数变成概率。用随机的初始权重，它给外框格子的概率是：上 {p0}，右 {p1}，下 {p2}，左 {p3}。',
			vars: (rt, at) => {
				const g = toRaw(rt.input), P = at.tape?.length ? (at.tape.state(0) as { params?: Record<string, { shape: number[]; data: ArrayLike<number> }> }).params : undefined;
				const p = P ? policyAt(P, featuresOf(worldOf(g.map), g.probe)).p : [NaN, NaN, NaN, NaN];
				return { hidden: bc.defaults.hidden, p0: p[0], p1: p[1], p2: p[2], p3: p[3] };
			}
		},
		look: I('Look at the four bars: probabilities that add up to 1. The arrow buttons move the outlined cell.', '看那四根条：加起来等于 1 的概率。箭头按钮可以移动外框格子。'),
		terms: ['policy', 'mlp', 'activation-function', 'softmax']
	},
	{
		id: 'train',
		source: 'train',
		at: 'last',
		view: { key: 'loss-curve', props: () => ({ metric: 'loss' }) },
		provenance: 'live',
		title: I('Cloning is supervised learning', '克隆就是监督学习'),
		narration: {
			en: 'Behavior cloning treats every pair as a labelled example: the cell is the input, your move is the label. Cross-entropy measures how unlikely the network finds your moves, and {steps} Adam updates on {all} took it from {l0} to {l1}. The network now picks your move on {right} of your {pairs}.',
			zh: '行为克隆把每一对都当成带标签的样本：格子是输入，你走的那一步是标签。交叉熵衡量的是：在网络看来，你的动作有多不可能。在{all}上做了 {steps} 次 Adam 更新，交叉熵从 {l0} 降到 {l1}。现在，在你的 {pairs}中，网络有 {right} 对选的和你一样。',
			vars: (rt, at) => {
				const t = at.tape, m = lastM(t);
				const all = pick(m.n === 1, I('your 1 pair', '你的这 1 对'), I(`all ${m.n} pairs`, `全部 ${m.n} 对`));
				return { steps: m.step, all, pairs: pairsN(m.n), l0: t?.length ? (t.frame(1).metrics?.loss ?? NaN) : NaN, l1: m.loss, right: Math.round(m.acc * m.n) };
			}
		},
		look: I('Look at the loss curve: it falls toward 0 as the network learns your moves.', '看损失曲线：网络学会你的动作时，它降向 0。'),
		terms: ['behavior-cloning', 'cross-entropy', 'label', 'adam']
	},
	{
		id: 'policy',
		source: 'train',
		at: 'last',
		view: board('demos,data,policy,wrong'),
		provenance: 'live',
		title: I('The cloned policy, cell by cell', '克隆出的策略，一格一格看'),
		narration: {
			en: 'Ask the trained network for its move on every open cell and you get the arrows in the stage. On the {seen} you demonstrated, {copies}. The other {unseen} were never in the data, so the network extrapolates from the cells it knows, and {wrong}.',
			zh: '让训练好的网络在每个空地格子上选一个动作，就得到了画面中的箭头。在你示范过的 {seen}上，{copies}。其余 {unseen}从没出现在数据里，网络只能从认识的格子往外推；{wrong}。',
			vars: (rt) => {
				const b = boardOf(rt, {}), open = b.W.cell.filter((c) => c === OPEN).length;
				const m = lastM(tapeOf(rt, 'train')), right = Math.round(m.acc * m.n);
				const copies = pick(right === m.n, I('it copies you', '它照你的做'), I(`it picks your move on ${right} of your ${m.n} pairs`, `它在你的 ${m.n} 对中的 ${right} 对上选了你的动作`));
				const bad = b.wrong.filter((s) => !b.dataCells.has(s)).length;
				const wrong = pick(bad > 0, I(`on ${bad} of them its arrow leads no closer to the goal`, `其中 ${bad} 个格子上，它的箭头不能让智能体更接近终点`), I('here every arrow still leads closer to the goal', '这些格子上的箭头都还能让智能体更接近终点'));
				return { seen: cellsN(b.dataCells.size), copies, unseen: cellsN(open - b.dataCells.size), wrong: b.moves ? wrong : NaN };
			}
		},
		look: I('Look at the marked arrows below your path: where do they point?', '看你路线下方那些标出来的箭头：它们指向哪里？'),
		terms: ['extrapolation', 'out-of-distribution']
	},
	{
		id: 'predict',
		source: 'roll',
		at: 'last',
		view: board('demos,policy,roll'),
		provenance: 'live',
		title: I('Where will it fail?', '它会在哪里出错？'),
		narration: {
			en: 'Now the cloned policy drives alone, {episodes} times from S. On every move a gust may blow it one cell down instead of its move (chance {wind}), and the bottom row is pits. Training copied your moves on {right} of your {pairs}. Where will it go wrong?',
			zh: '现在克隆出的策略自己开，从 S 出发跑 {episodes} 次。每走一步，都可能有一阵风把它往下吹一格、顶替它这一步（概率 {wind}），而最下面一行是陷阱。训练后，在你的 {pairs}中，它有 {right} 对照你的做。它会在哪里出错？',
			vars: (rt) => {
				const m = lastM(tapeOf(rt, 'train'));
				return { episodes: EVAL.episodes, wind: fmtSetting(toRaw(rt.input).wind), right: Math.round(m.acc * m.n), pairs: pairsN(m.n) };
			}
		},
		look: I('Look at where the failed episodes leave your path.', '看失败的回合是在哪里离开你的路线的。'),
		predict: {
			id: 'behavior-cloning-where',
			question: I('Where does the cloned policy go wrong?', '克隆出的策略在哪里出错？'),
			options: [I('On the cells you demonstrated', '在你示范过的格子上'), I('After a gust pushes it below your path, on cells it never saw', '阵风把它推到你的路线下方之后，在它从没见过的格子上'), I('Nowhere: it copies you perfectly', '哪里都不会：它完全照你的做')],
			answer: 1,
			explain: I(
				'On your cells it does what you did. A gust can put it on a cell below your path that was never in the data; its arrow there is only an extrapolation, often along the path instead of back up, and the next gust pushes it lower still. The lines in the stage are the episodes.',
				'在你的格子上，它照你的做。一阵风可能把它吹到你路线下方、数据里从没有过的格子；那里的箭头只是往外推出来的，常常顺着路线方向指，而不是往回指，下一阵风再把它往下推。画面中的线条就是这些回合。'
			)
		},
		terms: ['episode']
	},
	{
		id: 'drive',
		source: 'roll',
		at: 'last',
		view: board('data,policy,roll'),
		provenance: 'live',
		title: I('Copying is not enough', '光会照做还不够'),
		narration: {
			en: 'The cloned policy reached the goal in {goals} of {episodes} episodes; the expert, meeting exactly the same gusts, in {exGoals}. {on} {off}',
			zh: '克隆出的策略在 {episodes} 个回合中有 {goals} 个到达终点；专家遇到完全相同的阵风，到达了 {exGoals} 个。{on}{off}',
			vars: (_rt, at) => {
				const m = totals(at), worse = m.offMoves > 0 && m.onMoves > 0 && m.offGood / m.offMoves < m.onGood / m.onMoves;
				const on = m.onMoves > 0 ? I(`On cells in the dataset, ${m.onGood} of the ${m.onMoves} moves it chose headed toward the goal.`, `在数据集里的格子上，它选的 ${m.onMoves} 步中有 ${m.onGood} 步是朝终点去的；`) : I('It never stood on a cell from the dataset.', '它从没站在数据集里的格子上；');
				const off = m.offMoves === 0 ? I('It never left the cells in the dataset.', '它也从没离开过数据集里的格子。')
					: m.onMoves === 0 ? I(`Of the ${m.offMoves} moves it chose on cells it never saw, ${m.offGood} headed toward the goal.`, `在它没见过的格子上，它选的 ${m.offMoves} 步中有 ${m.offGood} 步是朝终点去的。`)
					: I(`On cells it never saw, ${worse ? 'only ' : ''}${m.offGood} of ${m.offMoves} did.`, `在它没见过的格子上，${m.offMoves} 步中${worse ? '只有' : '有'} ${m.offGood} 步是这样。`);
				return { goals: m.goals, episodes: m.episodes, exGoals: m.exGoals, on, off };
			}
		},
		look: I('Look at the cells with the strongest color: where does the policy spend its moves?', '看颜色最浓的格子：策略的步子都花在了哪里？'),
		terms: ['rollout']
	},
	{
		id: 'compound',
		source: 'roll',
		// the first episode the policy fails while the expert, with the same gusts, gets through; else the last episode
		at: { phase: 'end', where: (f) => (f.metrics!.outcome !== 1 && f.metrics!.exOutcome === 1) || f.metrics!.e === EVAL.episodes - 1 },
		view: board('data,policy,roll', { only: true }),
		provenance: 'live',
		title: I('Compounding error, one episode', '误差累积：看一个回合'),
		narration: {
			en: 'Follow episode {ep}. {gust} It made {off} of its {len} moves on cells it had never seen, and {end}. {moral}',
			zh: '跟着第 {ep} 个回合看。{gust}它的 {len} 步中有 {off} 步走在从没见过的格子上，{end}。{moral}',
			vars: (_rt, at) => {
				const m = at.frame?.metrics ?? {}, t = at.tape;
				let g1 = -1;
				for (let i = at.index - 1; t && i >= 0 && t.frame(i).phase === 'move' && t.frame(i).metrics!.e === m.e; i--) if (t.frame(i).metrics!.gust) g1 = t.frame(i).metrics!.t;
				// firstOff: the first move made from a cell outside the dataset (the episode may come back to your cells later)
				const g = g1 < 0 ? I('No gust hit it', '这一回合没有遇到阵风') : I(`The first gust came on move ${g1 + 1}`, `第一阵风出现在第 ${g1 + 1} 步`);
				const gust: TmplVal = m.firstOff < 0
					? I(`${g.en}, and it never left your cells.`, `${g.zh}，它也始终没有离开你的格子。`)
					: I(`${g.en}; move ${m.firstOff + 1} was its first from a cell you never showed.`, `${g.zh}；第 ${m.firstOff + 1} 步是它第一次从你没示范过的格子出发。`);
				const end = m.outcome === 1 ? I('it still reached the goal', '它最后还是到了终点') : m.outcome === -1 ? I('it fell into a pit', '最后掉进了陷阱') : I('it ran out of moves', '最后步数用完了');
				const moral = pick(m.outcome !== 1 && m.exOutcome === 1,
					I('The expert got through the same gusts: one push led to unfamiliar cells, unfamiliar cells to wrong moves, and wrong moves to stranger cells still, so the errors compounded.', '专家顶着同样的阵风走到了终点：一次推挤把它带到陌生的格子，陌生的格子带来错误的动作，错误的动作又把它带到更陌生的地方，误差就这样越积越多。'),
					I('No episode failed where the expert got through, so there is no compounding error to follow here.', '没有哪个回合是专家走通了而它失败的，所以这里没有可以跟踪的误差累积。'));
				return { ep: m.e + 1, gust, off: m.off, len: m.len, end, moral };
			}
		},
		look: I('Look at the one episode in the stage: where it leaves the outlined cells, and what happens next.', '看画面中的这一个回合：它在哪里离开加了外框的格子，之后又发生了什么。'),
		terms: ['compounding-error']
	},
	{
		id: 'fix',
		source: 'roll',
		at: 'last',
		view: board('demos,data,policy,roll,edit'),
		provenance: 'live',
		title: I('Your fix: show it how to recover', '你的修补：教它怎样回来'),
		narration: {
			en: 'Wherever nobody showed the policy what to do, it can only guess. Add a demonstration that starts on a cell below your path and leads back up to the goal; the page trains again and replays the same gusty episodes. With {demos} it reaches the goal in {goals} of {episodes} episodes (the expert: {exGoals}).',
			zh: '凡是没人教过该怎么办的地方，策略都只能猜。加一条示范：从你路线下方的一个格子出发，一路回到终点；页面会重新训练，再跑一遍同样的有风回合。用 {demos}，它在 {episodes} 个回合中有 {goals} 个到达终点（专家：{exGoals}）。',
			vars: (rt, at) => {
				const m = totals(at);
				return { demos: demosOf(rt), goals: m.goals, episodes: m.episodes, exGoals: m.exGoals };
			}
		},
		look: I('Look at the arrows below your path after you add a demonstration.', '加了示范之后，看你路线下方的箭头。'),
		task: {
			prompt: I('Record a demonstration from a cell below your path: tap the cell in the recording grid in the stage, then use the arrow buttons until you reach the goal.', '从你路线下方的一个格子录一条示范：在画面里的录制网格中点那个格子，然后用箭头按钮一直走到终点。'),
			done: (rt) => toRaw(rt.input).demos.length >= 2,
			waiting: I('Not yet: add a demonstration that starts below your path.', '还没有：加一条从你路线下方出发的示范。'),
			doneText: I('Done: the stage shows the policy trained with your new demonstration.', '完成：画面显示的是用你新示范训练出的策略。')
		},
		terms: ['demonstration']
	},
	{
		id: 'dagger',
		source: 'droll',
		at: 'last',
		view: board('data,policy,roll', { policy: 'dagger', roll: 'droll' }),
		provenance: 'live',
		title: I('DAgger: let it drive, ask the expert', 'DAgger：让它自己开，再问专家'),
		narration: {
			en: 'DAgger finds those cells for you. In each round the policy drives {ep} gusty episodes, the expert labels each visited cell the dataset lacks with its own move, and training goes on; {grew}. The new policy reaches the goal in {goals} of {episodes} episodes, the expert in {exGoals}.',
			zh: 'DAgger 会替你找出这些格子。每一轮里，策略先在有风的环境里跑 {ep} 个回合，专家给它到过、而数据集里还没有的每个格子标上专家自己的动作，然后接着训练；{grew}。新策略在 {episodes} 个回合中有 {goals} 个到达终点，专家是 {exGoals} 个。',
			vars: (rt, at) => {
				const m = totals(at), D = datasetOf(rt.input, tapeOf(rt, 'dagger'), DAGGER), n0 = D.from.filter((r) => r === 0).length, n1 = D.s.length, R = D.rounds.length;
				const grew = pick(n1 > n0, I(`${R} rounds grew the dataset from ${n0} to ${n1} pairs`, `${R} 轮之后，数据集从 ${n0} 对增加到 ${n1} 对`), I(`in ${R} rounds the policy never left your cells, so nothing was added`, `${R} 轮里策略从没离开你的格子，所以什么也没加`));
				return { ep: bc.defaults.episodes, grew, goals: m.goals, episodes: m.episodes, exGoals: m.exGoals };
			}
		},
		look: I('Look at the cells with dashed outlines: the expert labelled them because the policy went there.', '看虚线外框的格子：因为策略去过那里，专家才给它们标了动作。'),
		terms: ['dagger']
	},
	{
		id: 'limits',
		source: 'droll',
		at: 'last',
		kindOf: 'limits',
		view: board('data,policy,wrong', { policy: 'dagger' }),
		provenance: 'live',
		title: I('What it still cannot do', '它仍然做不到的事'),
		narration: {
			en: '{asked} It only labels the cells the policy reaches: {unlabelled} of the {open} open cells never got a label, and their arrows are still guesses{wrongPart}. And it copies the expert rather than beating it: {goals} of {episodes} episodes against the expert’s {exGoals}, and without rewards nothing tells it what better would be.',
			zh: '{asked}它只给策略会走到的格子标注：{open} 个空地格子中，有 {unlabelled} 个从没得到标签，上面的箭头仍然是猜的{wrongPart}。而且它是照着专家做，而不是超过专家：{episodes} 个回合中到达终点 {goals} 个，专家是 {exGoals} 个；没有奖励，就没有什么能告诉它怎样算更好。',
			vars: (rt, at) => {
				const b = boardOf(rt, { policy: 'dagger' }), m = totals(at), q = b.added.size;
				const free = b.W.cell.map((c, s) => (c === OPEN && !b.dataCells.has(s) ? s : -1)).filter((s) => s >= 0), bad = free.filter((s) => b.wrong.includes(s)).length;
				const asked = pick(q > 0, I(`DAgger asked the expert about ${q} ${q === 1 ? 'cell' : 'cells'} you never chose to show, so a real expert would have to stay and answer.`, `DAgger 就 ${q} 个你从没打算示范的格子询问了专家，所以真实的专家得一直守着回答。`), I('Here DAgger never needed the expert: the policy stayed on your cells.', '这里 DAgger 一次也没用上专家：策略始终待在你的格子上。'));
				const wrongPart = pick(bad > 0, I(`; ${bad} of them lead no closer to the goal`, `，其中 ${bad} 个不能让智能体更接近终点`), I('', ''));
				return { asked, unlabelled: free.length, open: b.W.cell.filter((c) => c === OPEN).length, wrongPart, goals: m.goals, episodes: m.episodes, exGoals: m.exGoals };
			}
		},
		look: I('Look at the cells without an outline: nobody labelled them, so their arrows are the network’s guesses.', '看没有外框的格子：没人给它们标过动作，上面的箭头都是网络猜的。'),
		terms: ['reward']
	}
];

const lesson: LessonDef = {
	title: I('Learning by copying: behavior cloning and DAgger', '照着学：行为克隆与 DAgger'),
	summary: I(
		'Demonstrate a path across a windy field, train a policy network to copy your moves, watch it fail once a gust pushes it where you never went, then fix it with your own recovery demonstrations or with DAgger.',
		'在刮风的空地上示范一条路线，训练策略网络照着你的动作学，看它在阵风把它吹到你没去过的地方后怎样失败，再用你自己的纠正示范或 DAgger 把它修好。'
	),
	steps
};

export default lesson;
