// snake-dqn: the guided lesson (design §4.6). Eleven steps on the learner's board: the board as numbers and the three
// Q-values (trace), one DQN step slowed down (ε-greedy draw, reward, target, loss: process dqn), the replay buffer
// (process replay), the target network and the training curve (precomputed log), the agent's game against the learner's
// (process agent + task), limits (measured facts). Every number is live or from facts.json.
import type { At, ExplainerRuntime, I18n, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { posOf } from './adapters/snake-board/adapter.ts';
import { ACTION, ENDS, L, SIZE, safe } from './rules.ts';
import { best, dec2, logOf, m, num, qOf, stateOf, yourGame } from './picture.ts';

type V = number | string | I18n;
/** A sentence with {name} or {name|fmt} filled the way a placeholder prints (NaN, so lint reports it, when one is missing). */
function S(t: I18n, v: Record<string, V>): I18n | number {
	const f = (s: string, lang: 'en' | 'zh') => s.replace(/\{(\w+)(?:\|(\w+))?\}/g, (_, k: string, fmt?: string) => formatValue(v[k], fmt, lang) ?? '\u0000');
	const en = f(t.en, 'en'), zh = f(t.zh ?? t.en, 'zh');
	return en.includes('\u0000') || zh.includes('\u0000') ? NaN : { en, zh };
}
const view = (mode: string) => ({ key: 'snake-dqn/step', props: () => ({ mode }) });
const act = (a: number) => ACTION[a] ?? NaN;
/** A move's name at the start of a sentence ("Turn left would crash …"). */
const Act = (a: number) => (ACTION[a] ? { en: ACTION[a].en[0].toUpperCase() + ACTION[a].en.slice(1), zh: ACTION[a].zh } : NaN);
const T = (xs: number[]) => ({ shape: [xs.length], data: Float64Array.from(xs) });
/** The agent's finished game (process agent, last frame). */
const agentEnd = (rt: ExplainerRuntime) => { const tp = rt.sources.agent?.tape, f = tp?.length ? tp.frame(tp.length - 1) : null; return f?.phase === 'end' ? f.metrics! : null; };
const why = (code: number) => ENDS[['', 'wall', 'self', 'full', 'hungry', 'cap'][code]] ?? L('it reached the move limit', '到了步数上限');

const steps: StepDef[] = [
	{
		id: 'board', source: 'fwd', at: 'last', view: view('input'), provenance: 'live', terms: ['state'],
		title: L('The board as numbers', '把棋盘变成数'),
		narration: {
			en: 'The network never sees a picture: it reads three planes (grids) of 9 × 9 numbers, {n|int} in all. The head plane has a single 1 on the head, the food plane a single 1 on the food. On the body plane the cell right behind the head holds {neck}, and the numbers fall to {tail} at the tail: the tail’s cell is the next to become free, and the largest body number marks where the head just came from, so which way the snake is heading.',
			zh: '网络看不到图片：它读的是三层（三个网格）9 × 9 的数，一共 {n|int} 个。蛇头层只在蛇头处有一个 1，食物层只在食物处有一个 1。身体层里，紧跟在蛇头后面的格子是 {neck}，越往尾巴数越小，到尾巴是 {tail}：尾巴的格子是下一个会空出来的；最大的那个身体数标出蛇头刚从哪里来，也就是蛇正朝哪边走。',
			vars: (rt) => { const n = posOf(rt.input).snake.length; return { n: 243, neck: dec2((n - 1) / n), tail: dec2(1 / n) }; }
		},
		look: L('The three small grids: one number per cell (· is 0).', '三个小网格：每个格子一个数（· 表示 0）。')
	},
	{
		id: 'q', source: 'fwd', at: 'last', view: view('q'), provenance: 'live', terms: ['q-value', 'dqn', 'convolution', 'discount'],
		title: L('Three answers: a Q-value for each move', '三个回答：每一步一个 Q 值'),
		narration: {
			en: 'After two convolutions and a hidden layer of {hidden}, out come three numbers, one per move: {qs}. Each is Q(s, a) for board s and move a: the reward the network expects from here on if the snake makes that move, a reward k moves later counted γ^k as much (γ = {gamma}). The agent takes the highest, {best}. {crash}',
			zh: '经过两层卷积和一个 {hidden} 个单元的隐藏层，出来三个数，每一步一个：{qs}。每个数都是 Q(s, a)，s 是棋盘，a 是这一步：如果蛇走这一步，网络预计从这里开始总共还能得到多少奖励，k 步之后的奖励只按 γ^k 算（γ = {gamma}）。智能体走最高的那一步：{best}。{crash}',
			vars: (rt) => {
				const q = qOf(rt), g = posOf(rt.input);
				if (!q) return { hidden: NaN, qs: NaN, best: NaN, crash: NaN, gamma: NaN };
				const bad = [0, 1, 2].filter((a) => !safe(g, a));
				const v = bad.length ? q[bad[0]] : 0;
				const crash = !bad.length ? L('No move crashes at once here.', '这里没有哪一步会立刻撞上。')
					: v < -0.5 ? S(L('{move} would crash at once (reward −1), and it gets {v}, below the safe moves.', '{move}会立刻撞上（奖励 −1），网络给它 {v}，比不会撞的走法都低。'), { move: Act(bad[0]), v })
						: S(L('{move} would crash at once (reward −1), yet it gets {v}: below the safe moves, but far off, for a Q-value is only a guess.', '{move}会立刻撞上（奖励 −1），可网络给它 {v}：虽然比不会撞的走法低，却差得很远，因为 Q 值只是猜测。'), { move: Act(bad[0]), v });
				const p = q.map((x) => formatValue(x, undefined, 'en')), tie = new Set(p).size < 3;
				const qs = S(tie ? L('straight on {q0}, left {q1}, right {q2} (two of them differ only after the third digit)', '直走 {q0}，左转 {q1}，右转 {q2}（其中两个要到第四位数字才有差别）') : L('straight on {q0}, left {q1}, right {q2}', '直走 {q0}，左转 {q1}，右转 {q2}'), { q0: q[0], q1: q[1], q2: q[2] });
				return { hidden: rt.spec?.layers.find((l) => l.id === 'fc1')?.outShape[0] ?? NaN, qs, best: act(best(q)), crash, gamma: fmtSetting(rt.num('fact:gamma')) };
			}
		},
		look: L('The arrows from the head: each ends where that move would take the head, labelled with its Q-value. Left and right are the snake’s own, as it faces.', '从蛇头出发的箭头：每支箭头指向这一步会把蛇头带到的格子，标着它的 Q 值。左转和右转是按蛇自己面朝的方向算的。')
	},
	{
		id: 'act', source: 'dqn', at: { phase: 'act' }, view: view('act'), provenance: 'live', terms: ['epsilon-greedy', 'exploration'],
		title: L('Choose a move: ε-greedy', '选一步：ε-贪心'),
		narration: {
			en: 'While it learns, the agent must sometimes try moves it does not rate highest, or it never finds out they are better. Before each move it draws a random number u between 0 and 1: below ε = {eps} it explores, drawing again to take one of the three moves at random, and otherwise it takes the move with the highest Q. Here u = {u}, so {what}. In training ε fell from 1 (all random) to {epsEnd} over the first {fact:epsMoves|int} moves; this step uses {eps}.',
			zh: '在学习的时候，智能体有时必须试试它评分不是最高的走法，否则永远不会发现它们其实更好。每走一步之前，它先抽一个 0 到 1 之间的随机数 u：小于 ε = {eps} 就去探索，再抽一次，从三步里随机选一步；否则就走 Q 最高的那一步。这里 u = {u}，所以{what}。训练时，ε 在前 {fact:epsMoves|int} 步里从 1（全是随机走）降到了 {epsEnd}；这一步用的是 {eps}。',
			vars: (rt, at) => ({ eps: fmtSetting(m(at, 'eps')), epsEnd: fmtSetting(rt.num('fact:epsEnd')), u: m(at, 'u'), what: m(at, 'explore') ? S(L('it explores and takes {a}', '它去探索，选了{a}'), { a: act(m(at, 'a')) }) : S(L('it takes the best move, {a}', '它走了最好的一步：{a}'), { a: act(m(at, 'a')) }) })
		},
		look: L('The line from 0 to 1: u lands left of ε (explore) or right of it (best move). “Draw u again” makes a new draw.', '从 0 到 1 的线：u 落在 ε 左边（探索）还是右边（走最好的一步）。“再抽一次 u”会重新抽。')
	},
	{
		id: 'reward', source: 'dqn', at: { phase: 'reward' }, view: view('reward'), provenance: 'live', terms: ['reward'],
		title: L('The move and its reward', '这一步和它的奖励'),
		narration: {
			en: 'The snake makes the move: {a}. {what} The reward is r = {r}. This game has only two rewards, +1 for food and −1 for a crash; every other move gives 0. That is all the network is ever told about what is good.',
			zh: '蛇走了这一步：{a}。{what}奖励 r = {r}。这个游戏只有两种奖励：吃到食物 +1，撞上东西 −1；其他每一步都是 0。关于什么是好，网络得到的信息就只有这些。',
			vars: (rt, at) => ({ a: act(m(at, 'a')), r: fmtSetting(m(at, 'r')), what: m(at, 'ate') ? L('Its head lands on the food: it grows by one, and new food appears.', '蛇头落在食物上：它长了一节，新的食物出现了。') : m(at, 'end') === 1 ? L('Its head hits the wall: the game is over.', '蛇头撞上了墙：这一局结束了。') : m(at, 'end') === 2 ? L('Its head runs into its body: the game is over.', '蛇头撞上了自己的身体：这一局结束了。') : L('No food, no crash: the snake just moves one cell.', '没吃到食物，也没撞上：蛇只是往前移动一格。') })
		},
		look: L('The board before (s) and after (s′) the move, and the reward.', '走之前（s）和走之后（s′）的棋盘，以及奖励。')
	},
	{
		id: 'target', source: 'dqn', at: { phase: 'target' }, view: view('target'), provenance: 'live', terms: ['td-target', 'discount'],
		title: L('The target: reward now plus what comes next', '目标值：眼前的奖励加上之后的价值'),
		narration: {
			en: 'Was that move as good as the network thought? DQN builds a target y from what just happened: the reward r plus what the new board s′ is still worth, γ times its best Q-value. That best Q is read from a second copy of the network, the target network (step 8 says why). {here} Guess first: what is y for a move that ends the game?',
			zh: '这一步真像网络想的那么好吗？DQN 用刚刚发生的事算出一个目标值 y：奖励 r，加上新棋盘 s′ 还值多少，也就是 γ 乘以它上面最好的 Q 值。这个最好的 Q 值是从网络的另一个副本，也就是目标网络读出来的（第 8 步会说为什么）。{here}先猜一猜：如果这一步让游戏结束了，y 是多少？',
			vars: (rt, at) => ({ here: m(at, 'terminal') ? L('Here the move ended the game; the stage shows what that does to y.', '这里这一步让游戏结束了；舞台上会显示这对 y 意味着什么。') : S(L('Here r = {r} and the best Q on s′ is {tmax}, so y = {r} + {gamma} × {tmax} = {y}.', '这里 r = {r}，s′ 上最好的 Q 是 {tmax}，所以 y = {r} + {gamma} × {tmax} = {y}。'), { r: fmtSetting(m(at, 'r')), tmax: m(at, 'tmax'), gamma: fmtSetting(m(at, 'gamma')), y: m(at, 'y') }) })
		},
		look: L('The new board s′ with the target network’s Q-values, and the formula with the numbers filled in.', '新棋盘 s′ 上目标网络给出的 Q 值，以及代入了数字的公式。'),
		predict: {
			id: 'snake-dqn-terminal',
			question: L('When a move ends the game (the snake crashes), what is the target y?', '如果这一步让游戏结束了（蛇撞上了），目标值 y 是多少？'),
			options: [L('r + γ · max Q_target(s′, ·), the same as for any other move', 'r + γ · max Q_target(s′, ·)，和别的步一样'), L('Just r: after the end no more reward can come', '只有 r：结束之后不会再有奖励'), L('0: a finished game is worth nothing', '0：结束了的一局什么也不值')],
			answer: 1,
			explain: L('After a crash nothing can follow, so nothing is added: y = r = −1. That −1 is what teaches the network that the boards just before a crash are bad, and from there it leaks back, one move per update, through γ · max Q.', '撞上之后不会再有任何事发生，所以什么也不加：y = r = −1。正是这个 −1 让网络知道撞上之前的那些棋盘不好；之后它再通过 γ · max Q 一次更新往回传一步。')
		}
	},
	{
		id: 'loss', source: 'dqn', at: { phase: 'loss' }, view: view('loss'), provenance: 'live', terms: ['loss', 'gradient-descent'],
		title: L('The loss for this one move', '这一步的损失'),
		narration: {
			en: 'Before the move the network said Q(s, {a}) = {qa}; the target is {y}. The difference δ = y − Q = {delta}, and the loss for this move is δ² = {loss}. Training takes a small gradient step on all the weights to make the loss smaller, which pulls Q(s, {a}) {dir}, towards y. The two moves that were not taken get no target from this move.',
			zh: '走之前网络说 Q(s, {a}) = {qa}；目标值是 {y}。两者之差 δ = y − Q = {delta}，这一步的损失是 δ² = {loss}。训练会对所有权重走一小步梯度下降，让损失变小，也就是把 Q(s, {a}) 往{dir}拉，拉向 y。没被走的另外两步，这一步不给它们目标值。',
			vars: (rt, at) => ({ a: act(m(at, 'a')), qa: m(at, 'qa'), y: m(at, 'y'), delta: num(m(at, 'delta')), loss: num(m(at, 'loss')), dir: m(at, 'delta') >= 0 ? L('up', '上') : L('down', '下') })
		},
		look: L('The two formulas: δ first, then its square.', '两个公式：先是 δ，再是它的平方。')
	},
	{
		id: 'replay', source: 'replay', at: 'last', view: 'snake-dqn/replay', provenance: 'live', terms: ['replay-buffer'],
		play: { from: 'first', to: 'last', interval: 600, unit: L('move', '步') },
		title: L('A replay buffer: learn from random old moves', '回放缓冲区：从随机抽出的旧步里学'),
		narration: {
			en: 'Learning from each move right after it happens works badly: one move’s board is almost the same as the next, so the network would chase whatever it saw last and forget the rest. DQN stores its moves in a replay buffer and learns from random batches of them. Here the agent plays {fill} moves (ε = {eps}) into {cap} slots, overwriting the oldest, then draws {batch}: their mean loss is {mean}. Training used {fact:buffer|int} slots and batches of {fact:batch}.',
			zh: '每走一步就马上拿它来学，效果不好：相邻两步的棋盘几乎一样，网络会一直追着刚看到的东西，忘了别的。DQN 把走过的步存进回放缓冲区，再从里面随机抽一批来学。这里智能体走了 {fill} 步（ε = {eps}），存进 {cap} 个格位，最旧的会被覆盖，然后抽出 {batch} 步：它们的平均损失是 {mean}。训练时用的是 {fact:buffer|int} 个格位，每批 {fact:batch} 步。',
			vars: (rt, at) => ({ fill: rt.num('opt.fill'), eps: fmtSetting(rt.num('opt.eps')), cap: rt.num('opt.cap'), batch: rt.num('opt.batch'), mean: m(at, 'mean') })
		},
		look: L('The chips fill up and wrap around; after the draw, the table works out each drawn move’s target and loss as in steps 5 and 6.', '小块一个个填满，满了再从头覆盖；抽完之后，表格像第 5、6 步那样算出每个被抽中的步的目标值和损失。')
	},
	{
		id: 'target-net', source: 'fwd', at: 'last', provenance: 'precomputed', terms: ['target-network'],
		view: {
			key: 'curve1d',
			props: (rt) => {
				const p = logOf(rt)?.probe;
				if (!p) return {};
				return { points: null, xLabel: L('training update', '训练更新次数'), yLabel: L('Q on one fixed board', '一块固定棋盘上的 Q'), legend: true, series: [{ y: T(p.online), x: T(p.updates), kind: 'data', label: L('online network: changes with every update (thin)', '在线网络：每次更新都在变（细线）') }, { y: T(p.target), x: T(p.updates), kind: 'model', label: L('target network: a copy, refreshed every few hundred updates (thick)', '目标网络：一个副本，每几百次更新刷新一次（粗线）') }] };
			}
		},
		title: L('The target network: a slowly copied network', '目标网络：一个慢慢复制的网络'),
		narration: {
			en: 'The targets come from the network itself: if the network that makes them changed with every update, each update would also move its own target, like chasing your own shadow, and training can swing out of control. So DQN keeps a copy, the target network, uses it for every target, and refreshes it only every {fact:sync|int} updates (gradient steps). The chart follows both on one fixed board during training. This page’s networks are from update {fact:ckUpdates|int}, {fact:ckSinceSync|int} updates after the last copy.',
			zh: '目标值是网络自己算出来的：如果算目标值的网络每次更新都在变，那每次更新也会挪动它自己的目标，就像追自己的影子，训练可能会失控。所以 DQN 留一个副本，也就是目标网络，所有目标值都用它来算，而且每 {fact:sync|int} 次更新（梯度下降步）才刷新一次。图里跟踪了训练中两者在一块固定棋盘上的输出。本页用的网络来自第 {fact:ckUpdates|int} 次更新，那时距离上一次复制已经过了 {fact:ckSinceSync|int} 次更新。'
		},
		look: L('The thin curve wiggles with every update; the thick one stays flat, then jumps to it at each refresh: a staircase.', '细线随着每次更新抖动；粗线保持不动，每次刷新时跳到细线的位置：像一级级台阶。')
	},
	{
		id: 'curve', source: 'fwd', at: 'last', provenance: 'precomputed', terms: ['episode'],
		view: {
			key: 'curve1d',
			props: (rt) => {
				const lg = logOf(rt);
				if (!lg) return {};
				return { points: null, xLabel: L('moves played in training', '训练中走过的步数'), yLabel: L('food eaten per game', '每局吃到的食物'), legend: true, series: [{ y: T(lg.score), x: T(lg.moves), kind: 'model', label: L('test games, always the best move (ε = 0)', '测试对局：总走最好的一步（ε = 0）') }, { y: T(lg.train), x: T(lg.trainMoves), kind: 'truth', label: L('training games, with ε-greedy (dashed)', '训练对局：用 ε-贪心（虚线）') }] };
			}
		},
		title: L('Training: from random moves to over twenty food', '训练：从乱走到吃二十多个'),
		narration: {
			en: 'Training played {fact:trainMoves|int} moves, about {fact:minutes} minutes on this machine. Every {fact:evalEvery|int} moves the agent played {fact:evalGames} test games without exploring. A player that moves at random eats {fact:randomMean} food per game; the agent passed half its best ({fact:evalHalf}) by move {fact:halfMoves|int} and peaked at {fact:evalBest} after {fact:ckMoves|int} moves, the checkpoint this page uses. The training games (dashed) score lower because they still explore.',
			zh: '训练一共走了 {fact:trainMoves|int} 步，在这台机器上大约用了 {fact:minutes} 分钟。每走 {fact:evalEvery|int} 步，智能体就不带探索地玩 {fact:evalGames} 局测试。随便乱走的玩家每局吃 {fact:randomMean} 个食物；智能体到第 {fact:halfMoves|int} 步就超过了它最好成绩的一半（{fact:evalHalf}），在第 {fact:ckMoves|int} 步达到最高的 {fact:evalBest}，本页用的就是这个检查点。训练对局（虚线）分数低一些，因为它们还在探索。'
		},
		look: L('The solid curve rises fast, then levels off above twenty; the dashed curve lags while ε is still large.', '实线升得很快，然后在二十多的地方变平；虚线在 ε 还大的时候落在后面。')
	},
	{
		id: 'watch', source: 'agent', at: 'last', view: 'snake-dqn/watch', provenance: 'live', linger: 8000,
		play: { from: 'first', to: 'last', interval: 90, unit: L('move', '步') },
		title: L('Watch it play, then beat its score', '看它玩，再超过它的分数'),
		narration: {
			en: 'Here is the trained agent on your food seed, always taking its highest Q: it ate {score} in {moves} moves, then {why}. Over {fact:agentGames|int} games, one on each seed from 1 up, it ate {fact:agentMean} on average (step 9’s {fact:evalBest} came from only {fact:evalGames} games) and {fact:agentMax} at most. Now beat it: play the same seed with the buttons. The first food is in the same place, and every later food is drawn with the same random numbers.',
			zh: '这是训练好的智能体在你的食物种子上玩的一局，每一步都走 Q 最高的：它走了 {moves} 步，吃了 {score} 个，然后{why}。在从 1 开始的 {fact:agentGames|int} 个种子上各玩一局，它平均吃 {fact:agentMean} 个（第 9 步的 {fact:evalBest} 只来自 {fact:evalGames} 局），最多吃 {fact:agentMax} 个。现在来打败它：用按钮在同一个种子上玩一局。第一个食物在同一个位置，之后的每个食物都用同样的随机数来抽。',
			vars: (rt) => { const e = agentEnd(rt); return e ? { score: e.score, moves: e.moves, why: why(e.why) } : { score: NaN, moves: NaN, why: NaN }; }
		},
		look: L('The agent’s board, move by move, with its Q-values at the head; and your board on the same seed, steered with the buttons under it.', '智能体的棋盘一步一步地放，蛇头上标着它的 Q 值；你的棋盘用同一个种子，用它下面的按钮操纵。'),
		task: {
			prompt: L('Play your own game on the same seed with the buttons under your board, and try to eat more than the agent.', '用你的棋盘下面的按钮在同一个种子上玩一局，试着比智能体吃得多。'),
			done: (rt) => { const y = yourGame(rt.input), e = agentEnd(rt); return y.done || (!!e && y.g.score > e.score); },
			waiting: L('Not yet: keep steering until you beat its score or your game ends.', '还没有：继续操纵，直到你超过它的分数，或者你的这一局结束。'),
			doneText: L('Done: compare your score with the agent’s beside the boards.', '完成：在棋盘旁边比一比你和智能体的分数。')
		}
	},
	{
		id: 'limits', source: 'fwd', at: 'last', provenance: 'precomputed', kindOf: 'limits', terms: ['reward-shaping'],
		view: {
			key: 'curve1d',
			props: (rt) => {
				const lg = logOf(rt), h = (rt.spec as unknown as { hurrylog?: { moves: number[]; score: number[] } } | null)?.hurrylog;
				if (!lg || !h) return {};
				return { points: null, xLabel: L('moves played in training', '训练中走过的步数'), yLabel: L('food eaten per test game', '每局测试吃到的食物'), legend: true, series: [{ y: T(lg.score), x: T(lg.moves), kind: 'model', label: L('rewards +1 food, −1 crash (this page)', '奖励：食物 +1，撞上 −1（本页）') }, { y: T(h.score), x: T(h.moves), kind: 'truth', label: L('the same, plus −0.1 for every move (dashed)', '同上，再加每步 −0.1（虚线）') }] };
			}
		},
		title: L('Limits: a small fixed board, and rewards that shape it', '局限：小而固定的棋盘，以及塑造它的奖励'),
		narration: {
			en: 'The network reads exactly 3 × 9 × 9 numbers, so a {big} × {big} board would need new weights and new training. Even here it never filled the board ({fact:maxFood} food) in {fact:agentGames|int} games. Rewards shape it too: a second agent also got {fact:hurryStep} per move, to make it hurry; it ate {fact:hurryMean} per game instead of {fact:agentMean}, in {fact:hurryMoves} moves instead of {fact:agentMoves}. Each move now costs and a crash ends the costs, so it risks more: on {fact:hurryAvoidable|pct} of its crashes another move was safe, against {fact:agentAvoidable|pct}.',
			zh: '网络只能读正好 3 × 9 × 9 个数，所以换成 {big} × {big} 的棋盘，就得换新权重、重新训练。就算在这里，{fact:agentGames|int} 局里它也一次都没把棋盘填满（要吃 {fact:maxFood} 个食物）。奖励也会塑造它：另一个智能体每走一步还会得到 {fact:hurryStep} 的奖励，本意是让它快点；它每局只吃 {fact:hurryMean} 个，而不是 {fact:agentMean} 个，每局只走 {fact:hurryMoves} 步，而不是 {fact:agentMoves} 步。现在每一步都扣分，而撞上之后就不再扣了，所以它更敢冒险：在它撞上的那些时刻，有 {fact:hurryAvoidable|pct} 换一种走法本来不会撞，原来那个智能体是 {fact:agentAvoidable|pct}。',
			vars: () => ({ big: SIZE + 1 })
		},
		look: L('The two curves: the same network and training, only the reward differs.', '两条曲线：网络和训练都一样，只有奖励不同。')
	}
];

const lesson: LessonDef = {
	title: L('How a deep Q-network learns to play snake', '深度 Q 网络怎样学会玩贪吃蛇'),
	summary: L('A small network reads your snake board and rates each move; one DQN step is worked out with real numbers (ε-greedy, reward, target, loss), then the replay buffer, the target network, training, and a game against the agent.', '一个小网络读你的贪吃蛇棋盘，给每一步打分；用真实数字算一步 DQN（ε-贪心、奖励、目标值、损失），再看回放缓冲区、目标网络、训练，并和智能体比一局。'),
	steps,
	fixtures: [
		{ id: 'explore', label: L('a draw that explores', '抽到探索的一次'), raw: { seed: 7, moves: [1, 0], draw: 2 } },
		{ id: 'played', label: L('a game played to the end', '玩完的一局'), raw: { seed: 7, moves: [1, 0], game: [0, 0, 0, 0, 0] } },
		{ id: 'crashed', label: L('a crashed snake', '撞了的蛇'), raw: { seed: 7, moves: [0, 0, 0, 0, 0] }, expect: { empty: true } }
	]
};

export default lesson;
