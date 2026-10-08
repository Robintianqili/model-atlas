// gridworld: the guided lesson, six steps on slot machines (bandits) and eleven on a maze (Q-learning). Every number is a
// {placeholder} bound to a live frame or computed from your maze in vars(); the other sentences are claims.test.ts. A
// predict step's narration never gives its answer away (it shows before the learner answers): the stage does.
import type { At, Cursor, ExplainerRuntime, Frame, I18n, LessonDef, StepDef, TmplVal } from '../../atlas/types/index.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { record } from '../../atlas/process/record.ts';
import { bandit } from './process.ts';
import { ARM_NAMES, BANDIT_SEED, BEST, BOARDS, DIR_NAMES, PROBS, GAMMAS, goalDistance, toRaw, worldOf } from './world.ts';

export const PULL_GOAL = 10;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
/** vars() may return text in both languages: the narration prints an I18n as it is (lesson/narrate.ts formatValue) */
const say = (x: I18n): TmplVal => ({ v: x as unknown as number });
/** a setting (γ, ε, α, a win chance) as the learner set it: 0.1, not 0.100 */
const set = (v: number) => String(Number(v.toFixed(4)));
/** an average of 0/1 payouts, as short as it is exact to 3 decimals: 1, 0.5, 0.667 */
const avg = (v: number) => String(Number(v.toFixed(3)));
/** a number inside a sentence built in vars(), printed as a placeholder without a format prints it: 0, −1, 0.450 */
const num = (v: number) => formatValue(v, undefined, 'en') ?? '…';
const dist = (rt: ExplainerRuntime) => goalDistance(worldOf(rt.input));
const frames = (at: At) => (at.tape ? Array.from({ length: at.index + 1 }, (_, i) => at.tape!.frame(i)) : []);
const I = (en: string, zh: string): I18n => ({ en, zh });
type Step = StepDef<ExplainerRuntime>;
/** one step: id, source, cursor, view key ('arms' or 'board') with its props, title, narration, look and the rest */
function S(id: string, source: string, at: Step['at'], view: 'arms' | 'board', props: Record<string, unknown>, title: I18n, narration: Step['narration'], look: I18n, more: Partial<Step> = {}): Step {
	return { id, source, at, view: { key: `gridworld/${view === 'arms' ? 'arm-bars' : 'grid-board'}`, props: () => props }, provenance: 'live', title, narration, look, ...more };
}
/** greedy's whole run (it reads no input): the pulls per machine, for the step after its predict */
let greedyN: number[] | null = null;
const greedy = () => (greedyN ??= ((t) => Array.from((t.state(t.length - 1) as { N: { data: ArrayLike<number> } }).N.data))(record(bandit, null, { policy: 'greedy' }, { seed: BANDIT_SEED })));
/** the UCB pick to show: every machine tried, not the best average, and ahead of the next score by at least 0.1 */
const ucbShowcase = (f: Frame) => {
	const v = f.metrics!, s = [0, 1, 2, 3].map((k) => v[`q${k}`] + v[`b${k}`]);
	return v.isBest === 0 && [0, 1, 2, 3].every((k) => v[`n${k}`] > 0) && s.every((x, k) => k === v.arm || s[v.arm] - x >= 0.1);
};
/** the update the target and update steps show: the first one that reuses a learned value, else the first that pays */
const updateCursor = (rt: ExplainerRuntime): Cursor => (dist(rt) >= 0 ? { phase: 'update', where: (f) => f.metrics!.maxNext > 0 } : { phase: 'update', where: (f) => f.metrics!.r !== 0 });

const lesson: LessonDef = {
	title: I('Learning by trial and error: slot machines, then a maze', '在试错中学习：先玩老虎机，再走迷宫'),
	summary: I('Pull slot machines yourself, see why always picking the best-looking one gets stuck, then watch Q-learning solve your own maze one update at a time.', '先亲手拉老虎机，看看为什么总选看起来最好的会卡住；再看 Q 学习一次更新一次地解开你自己的迷宫。'),
	steps: [
		S('b-pull', 'you', 'last', 'arms', { mode: 'you' }, I('Pull the slot machines yourself', '亲手拉老虎机'), {
			en: 'Here are four slot machines, A to D. Each pull pays 1 coin or nothing, and every machine has its own hidden chance of paying. Pull {goal|int} times, any machines you like: so far you have pulled {n|int} times and won {wins|int} coins.',
			zh: '这里有四台老虎机，A 到 D。每拉一次，要么得到 1 枚金币，要么什么也没有；每台机器出金币的机会各不相同，而且是隐藏的。随便挑机器，一共拉 {goal|int} 次：到现在你拉了 {n|int} 次，赢了 {wins|int} 枚金币。',
			vars: (rt, at) => ({ goal: PULL_GOAL, n: toRaw(rt.input).pulls.length, wins: at.frame?.phase === 'update' ? m(at, 'wins') : 0 })
		}, I('Each bar is a machine’s average payout so far.', '每根柱子是一台机器到目前为止的平均回报。'), {
			task: { prompt: I('Pull ten times with the Pull buttons.', '用“拉”按钮一共拉十次。'), done: (rt) => toRaw(rt.input).pulls.length >= PULL_GOAL, hint: I('The buttons under each machine, or the machines in “Your input”.', '每台机器下面的按钮，或者“你的输入”里的机器按钮。') }
		}),
		S('b-average', 'you', 'last', 'arms', { mode: 'you', formula: true }, I('Keep a running average', '边拉边算平均值'), {
			en: 'To judge a machine, keep its average payout Q: the coins it paid divided by the times you pulled it. You need not store every result: after each pull, Q moves part of the way toward the new reward r, Q ← Q + (r − Q) / n, where n counts that machine’s pulls. {last}',
			zh: '要评价一台机器，就记下它的平均回报 Q：它给的金币数除以你拉它的次数。不必记住每一次的结果：每拉一次，Q 就朝新的奖励 r 挪一段，Q ← Q + (r − Q) / n，其中 n 是这台机器被拉的次数。{last}',
			vars: (_rt, at) => ({ last: say(at.frame?.phase === 'update' ? m(at, 'old') === m(at, 'new') && m(at, 'cnt') > 1
						? I(`Your last pull, machine ${ARM_NAMES[m(at, 'arm')]}, paid ${m(at, 'r')}, the same as its average, so the average stayed at ${avg(m(at, 'new'))}.`, `你上一次拉的是机器 ${ARM_NAMES[m(at, 'arm')]}，得到 ${m(at, 'r')}，正好等于它的平均值，所以平均值保持 ${avg(m(at, 'new'))} 不变。`)
						: I(`Your last pull, machine ${ARM_NAMES[m(at, 'arm')]}, paid ${m(at, 'r')}, so its average went from ${avg(m(at, 'old'))} to ${avg(m(at, 'new'))}.`, `你上一次拉的是机器 ${ARM_NAMES[m(at, 'arm')]}，得到 ${m(at, 'r')}，所以它的平均值从 ${avg(m(at, 'old'))} 变成了 ${avg(m(at, 'new'))}。`) : I('You have not pulled yet, so every average is still 0.', '你还没有拉，所以每个平均值都还是 0。')) })
		}, I('The formula under the bars, filled in with your last pull.', '柱子下面的公式，代入的是你上一次拉的结果。')),
		S('b-greedy', 'greedy', 'last', 'arms', { mode: 'agent', reveal: true }, I('Greedy gets stuck', '贪心会卡住'), {
			en: 'A greedy player always pulls the machine with the highest average so far. At the start all four averages were 0, so it picked {first} at random, and that first pull paid {r1|int} coin: {first}’s average became {q1} while the other three stayed at 0. Make your guess, then see where all {T|int} of its pulls went.',
			zh: '贪心玩家总是拉目前平均值最高的机器。一开始四台的平均值都是 0，所以它随机选了 {first}，而这第一下就给了 {r1|int} 枚金币：{first} 的平均值变成了 {q1}，其余三台还是 0。先猜一猜，再看它的 {T|int} 次都拉了哪台机器。',
			vars: (_rt, at) => { const f = at.tape!.frame(2).metrics!; return { first: ARM_NAMES[f.arm], r1: f.r, q1: f.new, T: m(at, 'n') }; }
		}, I('The pulls under each machine, and the line across each bar: that machine’s hidden win chance.', '每台机器下面的拉动次数，以及每根柱子上的横线：这台机器隐藏的中奖机会。'), {
			predict: { id: 'gridworld-greedy', question: I('A greedy player always pulls the machine with the highest average so far, starting with every average at 0. Will it find the best machine?', '贪心玩家总是拉目前平均值最高的机器，一开始所有平均值都是 0。它能找到最好的机器吗？'), options: [I('Yes, sooner or later', '能，早晚会找到'), I('Not necessarily: one lucky pull can trap it', '不一定：一次走运就可能把它困住'), I('It pulls all four equally often', '它会把四台拉得一样多')], answer: 1, explain: I('A machine that paid on its first pull has an average above 0, while untried machines stay at 0, so greedy may never look at them again.', '第一次就出金币的机器，平均值会高于 0，而没试过的机器一直是 0，所以贪心可能再也不会去看它们。') }
		}),
		S('b-eps', 'eps', { phase: 'choose', where: (f) => f.metrics!.explore === 1 && f.metrics!.isBest === 0 }, 'arms', { mode: 'agent' }, I('ε-greedy: explore now and then', 'ε-贪心：时不时探索一下'), {
			en: 'Greedy put {nTop|int} of its {gT|int} pulls into {top} and never tried {best}, the best machine. The fix is to explore a little: before each pull, ε-greedy draws a random number u between 0 and 1; if u is below ε = {eps}, it explores and pulls a machine at random, otherwise it exploits, like greedy. At pull {n|int}, u = {u|n2} was below {eps}, so it pulled {arm} instead of the best-looking {look}.',
			zh: '贪心把 {gT|int} 次里的 {nTop|int} 次都给了 {top}，从没试过最好的机器 {best}。解决办法是偶尔探索一下：每次拉之前，ε-贪心先抽一个 0 到 1 之间的随机数 u；如果 u 小于 ε = {eps}，它就探索，随机拉一台机器，否则就利用，像贪心一样拉平均值最高的。第 {n|int} 次时 u = {u|n2}，小于 {eps}，所以它拉了 {arm}，而不是看起来最好的 {look}。',
			vars: (_rt, at) => { const q = [0, 1, 2, 3].map((k) => m(at, `q${k}`)), g = greedy(), top = g.indexOf(Math.max(...g)); return { nTop: g[top], gT: g.reduce((a, b) => a + b, 0), top: ARM_NAMES[top], best: ARM_NAMES[BEST], eps: set(m(at, 'eps')), n: m(at, 'n'), u: m(at, 'u'), arm: ARM_NAMES[m(at, 'arm')], look: ARM_NAMES.filter((_, k) => q[k] === Math.max(...q)).join('/') }; }
		}, I('Where the random number u falls on the gauge: below ε or above it.', '随机数 u 落在量尺的哪里：ε 的左边还是右边。')),
		S('b-ucb', 'ucb', { phase: 'score', where: ucbShowcase }, 'arms', { mode: 'agent', formula: true }, I('UCB: a bonus for the less tried', 'UCB：给试得少的机器加分'), {
			en: 'UCB (upper confidence bound) adds a bonus to each average: c·√(ln t / n), with c = {c}, t = {t|int} pulls so far and n the machine’s own pulls (ln, the natural logarithm, grows slowly), so the fewer pulls, the bigger the bonus. At pull {pull|int}, {arm} had {cnt|int} pulls: average {q|n2} plus bonus {bonus|n2} makes {score|n3}. {rival} had the best average, {rq|n2}, but a smaller bonus, so it scored only {rs|n3}, and UCB pulled {arm}.',
			zh: 'UCB（置信上界）给每个平均值加一个奖励分：c·√(ln t / n)，其中 c = {c}，t = {t|int} 是到目前为止的总次数，n 是这台机器自己被拉的次数（ln 是自然对数，增长得很慢），所以拉得越少，奖励分越大。第 {pull|int} 次时，{arm} 只被拉过 {cnt|int} 次：平均值 {q|n2} 加上奖励分 {bonus|n2}，得 {score|n3}。{rival} 的平均值最高，是 {rq|n2}，但奖励分小，只得 {rs|n3}，所以 UCB 拉了 {arm}。',
			vars: (_rt, at) => { const q = [0, 1, 2, 3].map((k) => m(at, `q${k}`)), r = q.indexOf(Math.max(...q)); return { c: set(m(at, 'c')), t: m(at, 't'), pull: m(at, 'n'), arm: ARM_NAMES[m(at, 'arm')], cnt: m(at, 'cnt'), q: m(at, 'q'), bonus: m(at, `b${m(at, 'arm')}`), score: m(at, 'score'), rival: ARM_NAMES[r], rq: q[r], rs: q[r] + m(at, `b${r}`) }; }
		}, I('The striped part of each bar: the bonus, biggest for the least-pulled machine.', '每根柱子的斜纹部分：奖励分，拉得最少的机器最大。')),
		S('b-regret', 'race', 'last', 'arms', { mode: 'regret', reveal: true }, I('Compare the strategies: regret', '比较各种策略：遗憾'), {
			en: 'Regret counts the coins a player can expect to miss compared with always pulling the best machine: each pull of a machine with win chance p costs {pBest} − p. After {T|int} pulls on the same machines, greedy’s regret is {r0|n3}, ε-greedy’s {r1|n3}, UCB’s {r2|n3} and Thompson sampling’s {r3|n3}. Thompson sampling keeps a range of plausible win chances for each machine, draws one guess from each range and pulls the highest guess.',
			zh: '遗憾衡量的是：与一直拉最好的机器相比，玩家预计少赢了多少金币；每拉一次中奖机会为 p 的机器，就记 {pBest} − p。在同样的机器上拉 {T|int} 次后，贪心的遗憾是 {r0|n3}，ε-贪心是 {r1|n3}，UCB 是 {r2|n3}，汤普森采样是 {r3|n3}。汤普森采样为每台机器保留一个“中奖机会大概是多少”的范围，从每个范围里随机抽一个猜测值，再拉猜测值最高的那台。',
			vars: (_rt, at) => ({ pBest: set(PROBS[BEST]), T: m(at, 'n'), r0: m(at, 'r0'), r1: m(at, 'r1'), r2: m(at, 'r2'), r3: m(at, 'r3') })
		}, I('Which line keeps climbing in a straight line?', '哪条线一直直线上升？')),
		S('g-world', 'q', 'first', 'board', { show: 'world' }, I('A maze to learn', '一个要学会走的迷宫'), {
			en: 'Now a harder problem: a maze. An agent, the learner in reinforcement learning, starts at S and can move up, right, down or left; walls and the edge block it. The goal pays +1 and the pit −1, and either one ends the episode, one try from start to finish; every other move pays 0. {reach}',
			zh: '现在换一个更难的问题：迷宫。智能体（强化学习里的学习者）从 S 出发，可以向上、右、下、左走；墙和边界会挡住它。走到终点得 +1，掉进陷阱得 −1，两者都会结束这一回合（从起点到结束的一次尝试）；其他每一步都得 0。{reach}',
			vars: (rt) => ({ reach: say(dist(rt) >= 0 ? I(`The shortest path from S to the goal takes ${dist(rt)} moves.`, `从 S 到终点最短要走 ${dist(rt)} 步。`) : I('Here no path leads from S to the goal.', '这里没有任何路能从 S 通到终点。')) })
		}, I('S, the goal (+1), the pit (−1) and the walls.', 'S、终点（+1）、陷阱（−1）和墙。')),
		S('g-qtable', 'q', 'first', 'board', { show: 'tri' }, I('The Q-table: four numbers per cell', 'Q 表：每个格子四个数'), {
			en: 'Nobody tells the agent the rules; it learns from what happens, as you did with the slot machines. It keeps a Q-table: for every cell s and move a, one number Q(s, a), the reward it expects in the long run after making that move there. Your maze has {open|int} open cells, so the table holds {entries|int} numbers, all 0 at the start, drawn as four triangles per cell.',
			zh: '没有人告诉智能体规则；它像你拉老虎机那样，从发生的事情里学习。它维护一张 Q 表：对每个格子 s 和每个动作 a 记一个数 Q(s, a)，表示在那里做这个动作之后，长远来看能期望得到多少奖励。你的迷宫有 {open|int} 个空格子，所以这张表有 {entries|int} 个数，一开始全是 0，每个格子画成四个三角形。',
			vars: (_rt, at) => ({ open: m(at, 'open'), entries: m(at, 'entries') })
		}, I('No triangle has a color yet: every Q is still 0.', '还没有哪个三角形有颜色：所有 Q 都还是 0。')),
		S('g-choose', 'q', { phase: 'act', step: 0, sub: 0 }, 'board', { show: 'tri,act' }, I('Explore or exploit: u against ε', '探索还是利用：u 与 ε 比较'), {
			en: 'Each move begins with the same choice as ε-greedy on the slot machines: draw u between 0 and 1; if u is below ε = {eps}, explore with a random move, otherwise exploit and take the move with the highest Q, breaking ties at random. Here u = {u|n2}, so the agent {kind} and goes {dir}.',
			zh: '每一步都从和 ε-贪心拉老虎机一样的选择开始：抽一个 0 到 1 之间的 u；如果 u 小于 ε = {eps}，就探索，随机走一步；否则就利用，选 Q 最大的动作，并列时随机挑一个。这里 u = {u|n2}，所以智能体选择{kind}，向{dir}走。',
			vars: (_rt, at) => ({ eps: set(m(at, 'eps')), u: m(at, 'u'), kind: say(m(at, 'explore') ? I('explores', '探索') : I('exploits', '利用')), dir: say(DIR_NAMES[m(at, 'a')]) })
		}, I('The gauge under the board, and the arrow at the agent.', '棋盘下面的量尺，以及智能体身上的箭头。'), {
			predict: { id: 'gridworld-first-move', question: I('All four Q values of the start cell are 0. Which move will the agent make first?', '起点格子的四个 Q 值都是 0。智能体第一步会怎么走？'), options: [I('Up, because it is listed first', '向上，因为它排在第一个'), I('A random one', '随机走一步'), I('The one toward the goal', '朝终点的方向')], answer: 1, explain: I('No move looks better than another yet: if the agent exploits, it breaks the tie at random, and if it explores, it picks at random anyway.', '现在没有哪个动作看起来更好：如果智能体选择利用，就在并列的动作里随机挑；如果选择探索，本来就是随机挑。') }
		}),
		S('g-reward', 'q', { phase: 'act', step: 0, sub: 0 }, 'board', { show: 'tri,move' }, I('Move and receive a reward', '移动，得到奖励'), {
			en: 'The world answers each move with the next cell s′ and a reward r. The agent went from cell ({x}, {y}) to ({x2}, {y2}) (column, row, counting from 0) and received r = {r}. {note}',
			zh: '环境对每一步给出两样东西：下一个格子 s′ 和奖励 r。智能体从格子 ({x}, {y}) 走到 ({x2}, {y2})（列、行，从 0 数起），得到 r = {r}。{note}',
			vars: (_rt, at) => ({ x: m(at, 'x'), y: m(at, 'y'), x2: m(at, 'x2'), y2: m(at, 'y2'), r: m(at, 'r'), note: say(m(at, 'bump') ? I('It bumped into a wall or the edge, so it stayed where it was.', '它撞上了墙或边界，所以留在原地。') : m(at, 'r') === 0 ? I('Most moves pay 0, so one move alone tells the agent little.', '大多数移动都得 0，所以单独一步告诉智能体的东西很少。') : I('That move ended the episode at once.', '这一步直接结束了回合。')) })
		}, I('The dashed ring: where the agent was; the dot: where it is now.', '虚线圈：智能体原来的位置；实心点：它现在的位置。')),
		S('g-target', 'q', updateCursor, 'board', { show: 'tri,update', formula: 'target' }, I('The target: r plus γ times the best next Q', '目标值：r 加上 γ 乘以下一格的最大 Q'), {
			en: 'To update Q(s, a), the agent builds a target: the reward r plus γ times the best Q of the new cell s′; the discount γ = {gamma} makes a later reward worth a little less. In episode {ep|int}, it moved {dir} and got r = {r}. {calc}',
			zh: '为了更新 Q(s, a)，智能体先算一个目标值：奖励 r，加上 γ 乘以新格子 s′ 的最大 Q；折扣因子 γ = {gamma} 会给晚到手的奖励打个折。在第 {ep|int} 个回合，它向{dir}走，得到 r = {r}。{calc}',
			vars: (_rt, at) => { const [r, g, mx, tg] = [m(at, 'r'), set(m(at, 'gamma')), m(at, 'maxNext'), m(at, 'target')]; const src = frames(at).slice(0, -1).filter((f) => f.phase === 'update' && f.metrics!.s === m(at, 's2') && f.metrics!.new === mx).pop(); const ep = src ? (src.sub ?? 0) + 1 : 0, end = r > 0 ? I('reached the goal', '走到了终点') : I('fell into the pit', '掉进了陷阱'); return { gamma: g, ep: (at.frame?.sub ?? NaN) + 1, dir: say(DIR_NAMES[m(at, 'a')]), r, calc: say(m(at, 'done') ? I(`That move ${end.en} and ended the episode, so there is no next cell: the target is just r = ${num(tg)}.`, `这一步${end.zh}，回合结束，没有下一个格子：目标值就是 r = ${num(tg)}。`) : I(`It landed in a cell whose best Q was ${num(mx)}, so the target is ${num(r)} + ${g} × ${num(mx)} = ${num(tg)}.${ep ? ` That ${num(mx)} was learned in episode ${ep}: this is how the goal’s reward travels backward.` : ''}`, `它进入的格子最大 Q 是 ${num(mx)}，所以目标值是 ${num(r)} + ${g} × ${num(mx)} = ${num(tg)}。${ep ? `这个 ${num(mx)} 是在第 ${ep} 个回合学到的：终点的奖励就是这样一格一格往回传的。` : ''}`)) }; }
		}, I('The dashed square: the new cell s′, whose best Q enters the target.', '虚线方框：新格子 s′，它的最大 Q 进入目标值。')),
		S('g-update', 'q', updateCursor, 'board', { show: 'tri,update', formula: 'update' }, I('Update: move Q part of the way', '更新：让 Q 朝目标值走一段'), {
			en: 'Then Q(s, a) moves part of the way toward the target: Q ← Q + α·(target − Q). With the learning rate α = {alpha}, that is {old} + {alpha} × ({target} − {old}) = {new}. A bigger α trusts the newest experience more; a smaller one averages over many visits, like the running average of the slot machines.',
			zh: '然后 Q(s, a) 朝目标值走一段：Q ← Q + α·(目标值 − Q)。学习率 α = {alpha}，算出来是 {old} + {alpha} × ({target} − {old}) = {new}。α 越大，越相信最新的经验；α 越小，越像在许多次经历上取平均，就像老虎机的平均值那样。',
			vars: (_rt, at) => ({ alpha: set(m(at, 'alpha')), old: m(at, 'old'), target: m(at, 'target'), new: m(at, 'new') })
		}, I('The triangle with the bold edge: the one Q value this update changes.', '粗边框的三角形：这次更新改变的那一个 Q 值。'), { terms: ['learning-rate'] }),
		S('g-episode', 'q', { phase: 'end', step: 0 }, 'board', { show: 'tri,path' }, I('One episode', '一个回合'), {
			en: 'Repeat choose, move and update until the episode ends: at the goal, in the pit, or after {maxSteps|int} moves. The first episode {how} after {moves|int} moves. With every Q at 0 the agent wandered at random, and {changed}',
			zh: '不断重复“选择、移动、更新”，直到回合结束：走到终点、掉进陷阱，或者走满 {maxSteps|int} 步。第一个回合走了 {moves|int} 步，{how}。所有 Q 都是 0，智能体只能乱走，{changed}',
			vars: (_rt, at) => { const n = m(at, 'moves'), end = m(at, 'how') > 0 ? I('into the goal', '走进终点') : I('into the pit', '掉进陷阱'); const changed = frames(at).filter((f) => f.phase === 'update' && f.sub === 0 && f.metrics!.new !== f.metrics!.old).length; return { maxSteps: at.tape?.frame(0).metrics?.maxSteps ?? NaN, moves: n, how: say([I('fell into the pit', '最后掉进了陷阱'), I('stopped at the move limit', '到了步数上限才停下'), I('reached the goal', '最后走到了终点')][m(at, 'how') + 1]), changed: say(changed ? I(`only one of its ${n} updates changed a Q value: the last, the move ${end.en}. Every earlier move paid 0 and led to a cell whose Q values were all 0, so its target was 0.`, `${n} 次更新里只有 1 次改变了 Q 值：最后那一步，${end.zh}。之前每一步都得 0，走到的格子 Q 值也全是 0，所以目标值都是 0。`) : I(`none of its ${n} updates changed a Q value: every move paid 0 and led to a cell whose Q values were all 0, so every target was 0.`, `${n} 次更新没有一次改变 Q 值：每一步都得 0，走到的格子 Q 值也全是 0，所以目标值都是 0。`)) }; }
		}, I('The trail: every cell the agent has visited so far; each frame of the replay is one move.', '轨迹：智能体到目前为止走过的每个格子；回放的每一帧是一步。'), { play: { from: { phase: 'act', step: 0, sub: 0 }, to: { phase: 'end', step: 0 }, phases: ['act'] } }),
		S('g-spread', 'q', { phase: 'end', last: true }, 'board', { show: 'heat,arrows' }, I('Values spread backward', '价值一格一格往回传'), {
			en: 'Now the agent plays all {episodes|int} episodes, and each cell is colored by its best Q, the value of its best move. {spread} After the last episode, the start’s best Q is {vStart|n3}, and {goals|int} of the {episodes|int} episodes reached the goal.',
			zh: '现在智能体把 {episodes|int} 个回合全部走完，每个格子按它的最大 Q（最佳动作的价值）上色。{spread}最后一个回合之后，起点的最大 Q 是 {vStart|n3}，{episodes|int} 个回合里有 {goals|int} 个走到了终点。',
			vars: (rt, at) => ({ episodes: m(at, 'ep') + 1, vStart: m(at, 'vStart'), goals: m(at, 'goals'), spread: say(dist(rt) >= 0 ? I('Make your guess, then watch the replay, one frame per episode: where does color appear first, and which way does it spread?', '先猜一猜，再看回放（每一帧是一个回合）：颜色最先出现在哪里，又往哪个方向扩散？') : I('Here no episode can reach the goal, so no positive value ever appears.', '这里没有哪个回合能走到终点，所以永远不会出现正的价值。')) })
		}, I('Where color appears first, and which way it spreads.', '颜色最先出现在哪里，又往哪个方向扩散。'), {
			play: { from: { phase: 'end', step: 0 }, to: { phase: 'end', last: true }, phases: ['end'] },
			predict: { id: 'gridworld-spread', question: I('Over many episodes, which cells will get a positive value first?', '经过许多回合，哪些格子会最先得到正的价值？'), options: [I('The cells next to the goal', '终点旁边的格子'), I('The cells next to the start', '起点旁边的格子'), I('All cells at once', '所有格子同时')], answer: 0, explain: I('A move gains value only when it lands on the goal or on a cell that already has value, so value starts at the goal and moves backward, one cell at a time.', '一个动作只有走到终点，或者走到已经有价值的格子，才会得到价值，所以价值从终点开始，一格一格往回扩散。') }
		}),
		S('g-policy', 'q', { phase: 'end', last: true }, 'board', { show: 'heat,arrows,answer,edit' }, I('The policy: follow the arrows', '策略：跟着箭头走'), {
			en: 'In each cell the arrow points to the move with the highest Q; together, the arrows are the agent’s policy, its rule for what to do where. {path} {check}',
			zh: '每个格子里的箭头指向 Q 最大的动作；所有箭头合起来就是智能体的策略，也就是“在哪里该怎么走”的规则。{path}{check}',
			vars: (rt, at) => ({ check: say(m(at, 'decided') > 0 ? I(`Value iteration, a method that knows the rules and computes every cell’s exact value, agrees with ${m(at, 'agree')} of the ${m(at, 'decided')} arrows.`, `价值迭代是一种知道规则、能算出每个格子精确价值的方法，它认同 ${m(at, 'decided')} 个箭头中的 ${m(at, 'agree')} 个。`) : I('No cell has a single best move yet, so there is no arrow to check against value iteration, a method that knows the rules and computes every cell’s exact value.', '还没有哪个格子有唯一的最佳动作，所以没有箭头可以拿去和价值迭代对照；价值迭代是一种知道规则、能算出每个格子精确价值的方法。')),
				path: say(m(at, 'len') > 0 ? I(`Following them from S reaches the goal in ${m(at, 'len')} moves; the shortest path takes ${dist(rt)}.`, `从 S 出发跟着箭头走，${m(at, 'len')} 步到达终点；最短路线是 ${dist(rt)} 步。`) : dist(rt) < 0 ? I('No path leads from S to the goal, so no arrows can lead there.', '没有路能从 S 通到终点，所以箭头也不可能通到那里。') : I('Following them from S does not reach the goal yet; more episodes (under “Your input”) give the agent more time to learn.', '从 S 出发跟着箭头走，还到不了终点；多给几个回合（在“你的输入”里调）能让它学得更久。')) })
		}, I('The arrows from S: do they lead straight to the goal?', '从 S 出发的箭头：它们是不是直通终点？'), {
			task: { prompt: I('Change the maze: put a wall on the agent’s path or move the goal, and watch it relearn.', '改一改迷宫：在智能体的路线上放一堵墙，或者挪动终点，看它重新学习。'), done: (rt) => !Object.values(BOARDS).some((b) => b.join() === toRaw(rt.input).rows.join()), hint: I('Pick a tool under the board (Wall, Goal…), then tap a cell.', '在棋盘下面选一个工具（墙、终点……），再点一个格子。') }
		}),
		S('g-gamma', 'vi05', 'last', 'board', { show: 'heat', pair: ['vi05', 'vi95'] }, I('γ: how far ahead the agent looks', 'γ：智能体往前看多远'), {
			en: 'The discount γ sets how far ahead the agent looks: a reward one move later counts only γ times as much. Both boards are your maze solved exactly by value iteration, with γ = {g1} and with γ = {g2}. {far}',
			zh: '折扣因子 γ 决定智能体往前看多远：晚一步到手的奖励只算 γ 倍。两块棋盘都是你的迷宫，由价值迭代精确求解，一块 γ = {g1}，一块 γ = {g2}。{far}',
			vars: (rt) => { const d = dist(rt), k = d - 1; const far = d < 0 ? I('Here the goal cannot be reached, so the start is worth 0 with any γ; the Classic maze shows the difference.', '这里到不了终点，所以不管 γ 是多少，起点的价值都是 0；换成“经典”迷宫就能看出差别。') : d === 1 ? I('The goal is right next to the start, so the start is worth the whole reward, 1, with any γ.', '终点就在起点旁边，所以不管 γ 是多少，起点的价值都是完整的奖励 1。') : I(`The start is ${d} moves from the goal: the last move collects the reward, and ${k === 1 ? 'the one move' : `each of the ${k} moves`} before it multiplies it by γ, so the start is worth ${k === 1 ? 'γ itself' : `γ to the power ${k}`}. Guess first, then read the start’s value under each board.`, `起点离终点 ${d} 步：最后一步拿到奖励，之前的${k === 1 ? '那一步' : ` ${k} 步每一步都`}要把它乘一次 γ，所以起点的价值是${k === 1 ? ' γ 本身' : ` γ 的 ${k} 次方`}。先猜一猜，再看每块棋盘下面起点的价值。`); return { g1: set(GAMMAS[0]), g2: set(GAMMAS[1]), far: say(far) }; }
		}, I('The colors far from the goal on each board.', '两块棋盘上，离终点远的格子的颜色。'), {
			predict: { id: 'gridworld-gamma', question: I('With γ = 0.5 instead of 0.95, what happens to the value of a cell many moves away from the goal?', '把 γ 从 0.95 换成 0.5，离终点很多步远的格子，价值会怎样？'), options: [I('It stays about the same', '差不多不变'), I('It becomes much smaller', '变小很多'), I('It becomes larger', '变大')], answer: 1, explain: I('Every move between the cell and the goal multiplies the goal’s reward by γ once, and 0.5 shrinks it much faster than 0.95.', '格子和终点之间每隔一步，终点的奖励就要乘一次 γ；0.5 让它缩小得比 0.95 快得多。') }
		}),
		S('g-limits', 'q', { phase: 'end', last: true }, 'board', { show: 'heat,arrows,edit' }, I('Break it: a goal it cannot reach', '弄坏它：一个到不了的终点'), {
			en: 'Q-learning only learns from rewards it actually meets. Wall the goal off and every episode ends in the pit or runs out of moves, so no positive value can appear. {now} A table also needs one row per cell: {open|int} cells are easy, but a game seen through camera images has far too many states for a table, so deep Q-networks replace the table with a neural network.',
			zh: 'Q 学习只能从它真正遇到的奖励中学习。把终点用墙围住，每个回合都只会掉进陷阱或走满步数，所以永远不会出现正的价值。{now}而且这张表要给每个格子留一行：{open|int} 个格子还好办，但如果智能体看到的是摄像头画面，状态多到表格根本装不下，所以深度 Q 网络用神经网络代替了这张表。',
			vars: (rt, at) => ({ open: at.tape?.frame(0).metrics?.open ?? NaN, now: say(dist(rt) < 0 ? I(`Right now the goal cannot be reached, and after ${m(at, 'ep') + 1} episodes the start’s best Q is ${num(m(at, 'vStart'))}.`, `现在终点到不了；${m(at, 'ep') + 1} 个回合之后，起点的最大 Q 是 ${num(m(at, 'vStart'))}。`) : I(`Right now the goal can still be reached in ${dist(rt)} moves.`, `现在终点还能到达，最短 ${dist(rt)} 步。`)) })
		}, I('The colors and arrows after you close the goal off.', '把终点围起来之后的颜色和箭头。'), {
			kindOf: 'breakIt',
			task: { prompt: I('Wall off the goal completely. What happens to the colors and the arrows?', '用墙把终点完全围住。颜色和箭头会怎样？'), done: (rt) => dist(rt) < 0, hint: I('Choose Wall under the board, then tap the open cells next to the goal.', '在棋盘下面选“墙”，再点终点旁边的空格子。') }
		})
	]
};

export default lesson;
