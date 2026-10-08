// alphazero-ttt: the guided lesson (design §4.6). Eleven steps on the learner's position: the board as numbers (trace),
// the priors (a guess first) and the value (trace), one simulation of the search slowed down (select, expand, back up:
// simulation 2, the first one that knows something), the tree growing (predict + Play), the move it plays, a game
// against it (task), how self-play trained it (precomputed log), limits. Every number is live or from facts.json.
import type { At, ExplainerRuntime, I18n, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { rawOf } from './adapters/ttt-board/adapter.ts';
import { boardAt, gameOf, kidsOf, netOf, posOf, stateOf } from './picture.ts';
import { CELL, L, RESULT, filled, legal, markOf, moveValue, solve, toMove, treeSize, urgent } from './rules.ts';

type V = number | string | I18n;
/** A sentence with {name} or {name|fmt} filled the way a placeholder prints (NaN, so lint reports it, when one is missing). */
function S(t: I18n, v: Record<string, V>): I18n | number {
	const f = (s: string, lang: 'en' | 'zh') => s.replace(/\{(\w+)(?:\|(\w+))?\}/g, (_, k: string, fmt?: string) => formatValue(v[k], fmt, lang) ?? '\u0000');
	const en = f(t.en, 'en'), zh = f(t.zh ?? t.en, 'zh');
	return en.includes('\u0000') || zh.includes('\u0000') ? NaN : { en, zh };
}
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const me = (rt: ExplainerRuntime) => markOf(toMove(posOf(rt)));
const board = (mode: string) => ({ key: 'alphazero-ttt/board', props: () => ({ mode }) });
/** The root's child with the largest prior (simulation 1 always tries it first): its cell, prior, visits and Q. */
function fav(at: At) {
	const s = stateOf(at);
	if (!s) return null;
	const k = kidsOf(s, 0).reduce((a, x) => (s.P.data[x] > s.P.data[a] ? x : a));
	return { cell: s.move.data[k], p: s.P.data[k], n: s.N.data[k], q: s.N.data[k] > 0 ? s.W.data[k] / s.N.data[k] : 0, k };
}
/** How a Q (or a value) reads for the player it belongs to. */
const feel = (q: number, who: string) =>
	q <= -0.5 ? S(L('which looks like a loss for {who}', '看起来对 {who} 是输棋'), { who }) : q >= 0.5 ? S(L('which looks like a win for {who}', '看起来对 {who} 是赢棋'), { who }) : L('which looks about even', '看起来胜负难分');
const treeOf = (rt: ExplainerRuntime) => treeSize(posOf(rt));
const sim2 = (phase: string) => ({ phase, step: 2, ...(phase === 'select' ? { sub: 0 } : {}) });

const steps: StepDef[] = [
	{
		id: 'board', source: 'fwd', at: 'last', view: board('input'), provenance: 'live',
		title: L('The board as numbers', '把棋盘变成数'),
		narration: {
			en: 'It is {me}’s turn. The network never sees X or O: it reads nine numbers, one per cell, row by row from the top left: +1 where {me} has played, −1 where {other} has, 0 for an empty cell. It always reads the board from the side of the player to move, so for {other} the same board gives the same numbers with every sign flipped. A second list, the mask, is 0 for an empty cell and −∞ for a taken one ({taken}).',
			zh: '轮到 {me} 走。网络看不到 X 和 O：它读到的是九个数，每个格子一个，从左上角开始一行一行地排：{me} 下过的地方记 +1，{other} 下过的地方记 −1，空格记 0。它总是站在轮到走棋的一方来读棋盘，所以换成 {other} 来读，同一块棋盘得到的是同样的数，只是正负号全都反过来。另一组数叫掩码：空格记 0，已经占了的格子记 −∞（{taken}）。',
			vars: (rt) => { const b = posOf(rt), k = filled(b); return { me: markOf(toMove(b)), other: markOf(-toMove(b)), taken: k ? S(L('{k} taken here', '这里占了 {k} 个'), { k }) : L('none is taken yet', '现在还没有格子被占') }; }
		},
		look: L('The small number in each cell, and the same nine numbers in a row beside the board.', '每个格子里的小数字，以及棋盘旁边排成一行的同样九个数。')
	},
	{
		id: 'priors', source: 'fwd', at: 'last', view: board('prior'), provenance: 'live', terms: ['prior', 'softmax'],
		// under Play: after the guess, time to read the priors and the changed narration (the default is 4 s)
		linger: 12000,
		title: L('First answer: a prior for each move', '第一个回答：每一步的先验'),
		narration: {
			en: 'The network’s first answer comes from its policy head: a prior for each move, a number between 0 and 1. It gives every cell a score, the mask sets the taken cells to −∞, and {split}, so a taken cell gets exactly 0. A high prior means “this move looks good at first sight”. {after}',
			zh: '网络的第一个回答来自它的策略头：每一步一个先验，是 0 到 1 之间的数。它先给每个格子打一个分，掩码把已占格子的分设成 −∞，再{split}，所以已占的格子正好得 0。先验高，意思是“这一步乍一看不错”。{after}',
			vars: (rt) => {
				const n = netOf(rt), g = rawOf(rt.input).guess, k = legal(posOf(rt)).length;
				const free = k > 1 ? S(L('a softmax turns the scores into {k} chances that add up to 1', '用 softmax 把分数变成 {k} 个加起来等于 1 的概率'), { k }) : L('a softmax gives the only empty cell a chance of exactly 1', '用 softmax 给唯一的空格正好 1 的概率');
				if (!n || g === null || g === undefined) return { split: free, after: L('Guess first: which empty cell will it like most?', '先猜一猜：它最喜欢哪个空格？') };
				return { split: free, after: g === n.top ? S(L('You guessed its favourite: {cell}, with a prior of {p}.', '你猜中了它最喜欢的格子：{cell}，先验是 {p}。'), { cell: CELL[g], p: n.p[g] }) : S(L('Its favourite is {top}, with {pt}; your guess, {cell}, has {p}.', '它最喜欢的是{top}，先验 {pt}；你猜的{cell}是 {p}。'), { top: CELL[n.top], pt: n.p[n.top], cell: CELL[g], p: n.p[g] }) };
			}
		},
		task: { prompt: L('Guess before you look: tap the empty cell on the board that you think the network likes most.', '先猜再看：在棋盘上点一个你觉得网络最喜欢的空格。'), done: (rt) => rawOf(rt.input).guess !== null, waiting: L('Not yet: tap an empty cell on the board. The priors appear after your guess.', '还没有：在棋盘上点一个空格。猜完之后先验才会出现。') }
	},
	{
		id: 'value', source: 'fwd', at: 'last', view: board('value'), provenance: 'live', terms: ['value-head'],
		title: L('Second answer: the value', '第二个回答：价值'),
		narration: {
			en: 'The second answer comes from the value head: one number v between −1 and +1, the network’s guess of how this game will end for {me}. +1 means a sure win, −1 a sure loss and 0 a draw; here v = {v}, {lean}. Tic-tac-toe is small enough to check exactly: with perfect play from both sides, this position is {truth} for {me}. The value is only a quick guess from one look, and the search will check it.',
			zh: '第二个回答来自价值头：一个在 −1 和 +1 之间的数 v，是网络对“这盘棋对 {me} 来说会怎样结束”的猜测。+1 表示稳赢，−1 表示稳输，0 表示平局；这里 v = {v}，{lean}。井字棋小到可以精确地查：如果双方都下得完美，这个局面对 {me} 来说是{truth}。价值只是看一眼得出的快速猜测，接下来的搜索会检查它。',
			vars: (rt) => {
				const v = netOf(rt)?.v ?? NaN;
				const lean = v <= -0.25 ? L('which leans towards a loss', '偏向输') : v >= 0.25 ? L('which leans towards a win', '偏向赢') : L('which is close to a draw', '接近平局');
				return { me: me(rt), v, lean, truth: RESULT[solve(posOf(rt)) + 1] };
			}
		},
		look: L('The filled dot (the network’s guess) and the dashed ring (perfect play) on the scale.', '刻度上的实心点（网络的猜测）和虚线圈（完美下法的结果）。')
	},
	{
		id: 'select', source: 'search', at: sim2('select'), view: 'alphazero-ttt/tree', provenance: 'live', terms: ['mcts', 'puct', 'exploration'],
		title: L('One simulation, part 1: select with PUCT', '一次模拟（一）：用 PUCT 选择'),
		narration: {
			en: 'The search grows a tree of positions one simulation at a time: a simulation walks down from your board, adds one new position and carries its value back up. Going down, it takes the move with the highest score U = Q + c·P·√N / (1 + n): Q is how well the move has done so far, and the bonus grows with its prior P and shrinks as its visits n grow (N: visits of the position, c = {c}). {first} {pick}',
			zh: '搜索一次模拟一次模拟地把局面长成一棵树：一次模拟从你摆的棋盘往下走，添上一个新局面，再把它的价值传回来。往下走时，它选分数 U = Q + c·P·√N / (1 + n) 最高的那一步：Q 是这一步到目前为止表现得多好；奖励分随先验 P 变大，随它的访问次数 n 增加而变小（N 是这个局面的访问次数，c = {c}）。{first}{pick}',
			vars: (rt, at) => {
				const v = { sim: m(at, 'sim'), cell: CELL[m(at, 'cell')], c: String(m(at, 'c')), p: m(at, 'p'), N: m(at, 'N'), n: m(at, 'n'), q: m(at, 'q'), u: m(at, 'u'), bonus: m(at, 'bonus') };
				const f = fav(at), who = me(rt);
				const first = f ? S(L('In simulation 1 every Q was still 0, so it simply tried the network’s favourite, {top}; now its Q = {q}, {feel}.', '第 1 次模拟时所有 Q 都还是 0，所以它直接试了网络最喜欢的{top}；现在它的 Q = {q}，{feel}。'), { top: CELL[f.cell], q: f.q, feel: feel(f.q, who) }) : NaN;
				const pick = f && f.cell === m(at, 'cell')
					? S(L('In simulation {sim} it picks {cell} again: Q = {q}, plus a bonus of {c} × {p} × √{N} / (1 + {n}) = {bonus}, gives U = {u}, the highest score.', '第 {sim} 次模拟它又选了{cell}：Q = {q}，加上奖励分 {c} × {p} × √{N} / (1 + {n}) = {bonus}，U = {u}，是最高分。'), v)
					: v.n === 0
						? S(L('In simulation {sim} it picks {cell}: never tried, so Q = 0, but its bonus {c} × {p} × √{N} / (1 + 0) = {u} is the highest score.', '第 {sim} 次模拟它选了{cell}：这一步还没试过，所以 Q = 0，但它的奖励分 {c} × {p} × √{N} / (1 + 0) = {u} 是最高分。'), v)
						: S(L('In simulation {sim} it picks {cell}: Q = {q} after {n} visits, plus a bonus of {bonus}, gives U = {u}, the highest score.', '第 {sim} 次模拟它选了{cell}：试了 {n} 次后 Q = {q}，加上奖励分 {bonus}，U = {u}，是最高分。'), v);
				return { c: v.c, first, pick };
			}
		},
		look: L('The table under the tree: the highlighted row has the largest U, and the formula works it out. N counts the visits of your board so far (the top box of the tree); the network’s first look at it counts as one.', '树下面的表格：高亮的那一行 U 最大，公式把它算了出来。N 是你的棋盘到目前为止被访问的次数（树最上面那个方块里的数）；网络第一次看它也算一次。')
	},
	{
		id: 'expand', source: 'search', at: sim2('expand'), view: 'alphazero-ttt/tree', provenance: 'live', terms: ['value-head'],
		title: L('One simulation, part 2: expand and evaluate', '一次模拟（二）：展开并评估'),
		narration: {
			en: 'Following the highest scores, the path stops at a position the tree has not seen before: {cell} has just been played, and {mover} is to move. {eval}',
			zh: '顺着最高分往下走，路径停在一个树里还没有的局面：刚下在{cell}，轮到 {mover} 走。{eval}',
			vars: (rt, at) => {
				const s = stateOf(at), nb = s ? boardAt(s, posOf(rt), m(at, 'node')) : null, mover = m(at, 'mover');
				const u = nb && !m(at, 'terminal') ? urgent(nb, m(at, 'top')) : 0;
				const why = u > 0 ? L(', which wins at once', '，这一步能直接赢') : u < 0 ? S(L(', which blocks {opp}’s three in a row', '，这一步堵住了 {opp} 的三连'), { opp: markOf(-mover) }) : '';
				const v = { mover: markOf(mover), v: m(at, 'v'), kids: m(at, 'kids'), top: CELL[m(at, 'top')], topP: m(at, 'topP'), res: RESULT[m(at, 'v') + 1], why };
				return { cell: CELL[m(at, 'cell')], mover: v.mover, eval: m(at, 'terminal') ? S(L('The game is already over there: {res} for {mover}. So the search takes the real result, v = {v}, and needs no network.', '那里棋局已经结束：对 {mover} 来说是{res}。所以搜索直接用真实结果 v = {v}，不需要网络。'), v)
					: S(L('The network looks at it once. Its value v = {v} stands in for playing the game to the end, and its priors give the {kids} new branches their P (the largest, {topP}, goes to {top}{why}).', '网络看它一眼。它的价值 v = {v} 代替了把这盘棋下到底；它的先验给 {kids} 个新分支各自的 P（最大的 {topP} 给了{top}{why}）。'), v) };
			}
		},
		look: L('The board under the tree: the new position and, if the game goes on there, the priors of its empty cells.', '树下面的棋盘：新局面；如果棋局还没结束，空格上标着先验。')
	},
	{
		id: 'backup', source: 'search', at: sim2('backup'), view: 'alphazero-ttt/tree', provenance: 'live', terms: ['q-value'],
		title: L('One simulation, part 3: back up', '一次模拟（三）：回传'),
		narration: {
			en: 'Now the value v = {v} travels back up the path to your board, and every position on the way gets one more visit: n goes up by 1. The value was measured for {mover}, the player to move at the new position, but each Q on the path belongs to whoever chose that move. Guess first: what is added to the move just above the new position?',
			zh: '现在价值 v = {v} 沿着路径往回传，一直传到你摆的棋盘；路上每个局面都多一次访问：n 加 1。这个价值是对 {mover}（新局面里轮到走棋的一方）量出来的，而路径上每个 Q 都属于做出那一步选择的一方。先猜一猜：新局面正上方的那一步，加上的是什么？',
			vars: (rt, at) => { const s = stateOf(at); return { v: m(at, 'v'), mover: s ? markOf(toMove(boardAt(s, posOf(rt), m(at, 'leaf')))) : NaN }; }
		},
		look: L('The table: every row’s n goes up by 1; the new Q of the moves on the path shows once you have answered.', '表格：每一行的 n 都加 1；路径上那些走法的新 Q，要等你回答之后才显示。'),
		predict: {
			id: 'alphazero-ttt-backup',
			question: L('The value v was measured for the player to move at the new position. What is added to the move just above it, a move the other player chose?', 'v 是对新局面里轮到走棋的一方量出来的。它正上方的那一步是另一方选的，那一步加上的是什么？'),
			options: [L('v itself: a value is a value', 'v 本身：价值就是价值'), L('−v: what is good for one player is equally bad for the other', '−v：对一方有多好，对另一方就有多坏'), L('v / 2: a value fades as it goes up', 'v / 2：价值越往上传越弱')],
			answer: 1,
			explain: L('In tic-tac-toe one player’s gain is the other’s loss, so the sign flips at every level on the way up: −v for the move into the new position, +v one level higher, and so on. That keeps every Q meaning “how good this move was for the player who made it”. The table shows n and Q before and after.', '井字棋里一方的得就是另一方的失，所以价值每往上一层就变一次号：走进新局面的那一步加 −v，再上一层加 +v，依此类推。这样每个 Q 始终表示“这一步对下这步棋的一方有多好”。表格列出了 n 和 Q 前后的变化。')
		}
	},
	{
		id: 'grow', source: 'search', at: { phase: 'backup', step: 1 }, view: 'alphazero-ttt/tree', provenance: 'live', terms: ['mcts'],
		// K1: the stage rests on the cursor until Play; simulation 1, so the dimmed stage under the predict shows no answer
		play: { from: { phase: 'backup', step: 1 }, to: 'last', phases: ['backup'], unit: L('simulation', '模拟') },
		title: L('Many simulations: the tree grows', '很多次模拟：树长起来了'),
		narration: {
			en: 'The search repeats select, expand and back up {opt.sims} times, and every simulation starts again at your board. Each simulation adds 1 to every box on its path, and a box appears the first time a simulation reaches its position, so the tree grows where the simulations go. Guess first, then press Play or drag the slider: where will the search spend most of its simulations?',
			zh: '搜索把选择、展开、回传重复 {opt.sims} 次，每次模拟都从你摆的棋盘重新出发。每次模拟都给路径上的每个方块加 1；一个局面第一次被模拟走到时，就多出一个方块，所以模拟往哪里去，树就往哪里长。先猜一猜，再按“播放”或拖动滑块：搜索会把大部分模拟花在哪里？'
		},
		look: L('The second row of the tree: the width of each move’s box is its share of the visits so far.', '树的第二行：每一步的方块宽度，就是它在目前所有访问里所占的份额。'),
		predict: {
			id: 'alphazero-ttt-visits',
			question: L('Where will the search spend most of its simulations?', '搜索会把大部分模拟花在哪里？'),
			options: [L('Evenly on every empty cell, to be fair', '平均分给每个空格，以示公平'), L('On the moves that keep looking good once they are tried', '花在试过之后一直显得不错的走法上'), L('Mostly on the moves with the lowest priors, to check them', '主要花在先验最低的走法上，好去检查它们')],
			answer: 1,
			explain: L('Each simulation goes to the move with the best score U. A move that keeps doing well keeps winning that contest, a bad one drops out once its Q is known, and the bonus slowly brings back moves that were never checked. When one move is clearly best, its box grows wide; when several moves are about equally good, as on the empty board, they share the simulations.', '每次模拟都走向分数 U 最高的那一步。一直表现好的走法会一直赢下这场比较；差的走法一旦 Q 被查出来就退出了；奖励分则会慢慢把还没检查过的走法拉回来。如果有一步明显最好，它的方块就会变得很宽；如果几步差不多一样好（比如在空棋盘上），它们就分享这些模拟。')
		}
	},
	{
		id: 'pick', source: 'search', at: 'last', view: board('pick'), provenance: 'live',
		title: L('Visits versus priors: the move it plays', '访问次数对先验：它下的那一步'),
		narration: {
			en: 'After {sims} simulations the search plays the move with the most visits: {most}, with {n} of them. {verdict} Checked against perfect play, {most} {quality}. Over all {fact:openPositions|int} positions with a move to make, the network’s favourite is a perfect move on {fact:priorPerfect|pct} of them and the search’s move on {fact:searchPerfect|pct}: the search fixes {fact:fixedBySearch} of the network’s mistakes and makes {fact:brokenBySearch} new ones.',
			zh: '{sims} 次模拟之后，搜索下访问次数最多的那一步：{most}，访问了 {n} 次。{verdict}和完美下法对照，{most}{quality}。在所有 {fact:openPositions|int} 个还有棋可走的局面里，网络最喜欢的那一步是完美一步的占 {fact:priorPerfect|pct}，搜索下的那一步占 {fact:searchPerfect|pct}：搜索改正了网络的 {fact:fixedBySearch} 个错误，自己新犯了 {fact:brokenBySearch} 个。',
			vars: (rt, at) => {
				const b = posOf(rt), cell = m(at, 'cell'), best = solve(b), got = Number.isFinite(cell) ? moveValue(b, cell) : NaN, who = me(rt);
				const verdict = m(at, 'agree') ? S(L('That is also the network’s favourite: its prior was {topP}.', '它也是网络最喜欢的那一步：先验是 {topP}。'), { topP: m(at, 'topP') })
					: m(at, 'topN') === m(at, 'n') ? S(L('The network’s favourite, {top} (prior {topP}), got just as many; on a tie the search takes the cell that comes first, row by row from the top left.', '网络最喜欢的{top}（先验 {topP}）也得到同样多的次数；打平时，搜索取从左上角一行一行数过来排在前面的格子。'), { top: CELL[m(at, 'top')], topP: m(at, 'topP') })
					: S(L('The network’s favourite, {top} with a prior of {topP}, got only {topN}: the search overruled it.', '网络最喜欢的{top}（先验 {topP}）只得到 {topN} 次：搜索推翻了它。'), { top: CELL[m(at, 'top')], topP: m(at, 'topP'), topN: m(at, 'topN') });
				const quality = got === best ? S(L('keeps the best result possible here, {res} for {me}', '保住了这里能得到的最好结果：对 {me} 来说是{res}'), { res: RESULT[best + 1], me: who }) : S(L('is a mistake: it turns {res} into {worse} for {me}', '是一步错棋：对 {me} 来说，它把{res}变成了{worse}'), { res: RESULT[best + 1], worse: RESULT[got + 1], me: who });
				return { sims: m(at, 'sims'), most: CELL[cell], n: m(at, 'n'), verdict, quality };
			}
		},
		look: L('The table beside the board: priors on the left, visits on the right. A move keeps getting visits only while what the search finds below it keeps looking good.', '棋盘旁边的表格：左边是先验，右边是访问次数。只有当搜索在一步棋下面查到的局面一直显得不错，这一步才会一直得到访问。')
	},
	{
		id: 'play', source: 'game', at: 'last', view: 'alphazero-ttt/play', provenance: 'live', linger: 8000,
		title: L('Play a game against it', '和它下一盘'),
		narration: {
			en: 'Now test it. You play X and move first, or let the search start; after each of your moves it runs a new search of {sims} simulations and plays its most-visited move. Can you beat it? We let it play every possible opponent, as X and as O; because it always answers a position the same way, that makes only {fact:searchGames|int} different games, and it lost {fact:searchLosses} of them.',
			zh: '现在来考考它。你执 X 先走，也可以让搜索先走；你每走一步，它都重新做一次 {sims} 次模拟的搜索，然后下访问最多的那一步。你能赢它吗？我们让它和所有可能的对手各下了一遍，执 X、执 O 都有；因为同一个局面它总是走同一步，一共只有 {fact:searchGames|int} 盘不同的棋，它输了 {fact:searchLosses} 盘。',
			vars: (rt) => ({ sims: rt.sources.game?.tape?.frame(0).metrics?.sims ?? NaN })
		},
		task: {
			prompt: L('Play one game to the end against the search: click a cell on the board to make your move.', '和搜索下完一整盘：点棋盘上的格子来走棋。'),
			done: (rt) => gameOf(rt)?.over === true,
			hint: L('The buttons beside the board start a new game or let the search move first.', '棋盘旁边的按钮可以新开一盘，或者让搜索先走。'),
			waiting: L('Not yet: the game is still going. Keep moving until someone has three in a row or the board is full.', '还没有：这盘棋还没下完。继续走，直到有一方连成三个或者棋盘下满。')
		}
	},
	{
		id: 'selfplay', source: 'fwd', at: 'last', provenance: 'precomputed', terms: ['self-play'],
		view: {
			key: 'curve1d',
			props: (rt) => {
				const sp = (rt.spec as unknown as { selfplay?: Record<string, number[]> } | null)?.selfplay;
				const T = (xs: number[]) => ({ shape: [xs.length], data: Float64Array.from(xs) });
				if (!sp) return {};
				const r = sp.round, from1 = (k: string) => T(sp[k].slice(1));
				return { points: null, xLabel: L('training round', '训练轮次'), yRange: [0, 1], series: [{ y: T(sp.acc), x: T(r), kind: 'model', label: L('share of open positions where its favourite move is perfect', '最喜欢的一步是完美一步的局面占比') }, { y: from1('lossV'), x: T(r.slice(1)), kind: 'truth', label: L('value loss: squared error against the game results (dashed)', '价值损失：与对局结果的平方误差（虚线）') }] };
			}
		},
		title: L('How self-play trained the network', '自我对弈怎样训练了这个网络'),
		narration: {
			en: 'Nobody showed the network a single good move. It played {fact:selfplayGames|int} games against itself, choosing every move with this same search plus a little randomness for variety. Each position became a training example: the visit shares as the target for the priors, and the game’s final result as the target for the value. Over {fact:rounds} rounds, the share of open positions where its favourite move is perfect went from {fact:accStart|pct} (several moves are often perfect, so even random picks get more than half right) to {fact:accEnd|pct}.',
			zh: '没有人给网络看过哪怕一步好棋。它自己和自己下了 {fact:selfplayGames|int} 盘，每一步都用同样的搜索来选，只加了一点随机，让对局有变化。每个局面都成了一个训练样本：访问次数的占比是先验的目标，这盘棋的最终结果是价值的目标。经过 {fact:rounds} 轮，在所有还有棋可走的局面里，它最喜欢的那一步是完美一步的比例从 {fact:accStart|pct}（很多局面有好几步完美的棋，所以就算随便挑，也能挑对一半以上）变成了 {fact:accEnd|pct}。'
		},
		look: L('The top curve rises fast, peaks early and then sags a little, probably because later games are mostly well played and rarely reach odd positions. The dashed curve is the value loss.', '最上面的曲线升得很快，很早就到了顶，之后略有回落，可能是因为后来的对局大多下得不错，很少走到奇怪的局面。虚线是价值损失。')
	},
	{
		id: 'limits', source: 'search', at: 'last', view: { key: 'alphazero-ttt/tree', props: () => ({ limits: true }) }, provenance: 'live', kindOf: 'limits',
		title: L('Limits: tic-tac-toe is tiny', '局限：井字棋太小了'),
		narration: {
			en: 'Tic-tac-toe is tiny: {fact:positions|int} different positions in all, and even counting a position again for every order of moves that reaches it, the game tree from your board has {tree|int}, so a plain program can check every line; that is how this page knows the perfect moves. The search let the network look at only {nets} of them. Go has {fact:goPositions} positions: AlphaZero ran {fact:azSims} simulations per move and {fact:azTpus|int} TPUs for its self-play, and its search still saw a vanishing sliver. There, almost everything rests on the network’s guesses.',
			zh: '井字棋太小了：一共只有 {fact:positions|int} 个不同的局面；就算同一个局面每被一种走法顺序走到就重算一次，从你摆的棋盘出发的整棵博弈树也就 {tree|int} 个局面，普通程序就能把每条路线都查一遍，这个页面正是这样知道哪一步完美的。这次搜索只让网络看了其中 {nets} 个。围棋有{fact:goPositions} 个局面：AlphaZero 每走一步做 {fact:azSims} 次模拟，用 {fact:azTpus|int} 块 TPU 来下自我对弈，它的搜索看到的仍然只是微不足道的一小片。在那里，几乎一切都要靠网络的猜测。',
			vars: (rt, at) => ({ tree: treeOf(rt), nets: m(at, 'nets') })
		},
		look: L('The note under the tree: positions in this tree against the whole game tree from your board.', '树下面的说明：这棵树里的局面数，和从你的棋盘出发的整棵博弈树比一比。')
	}
];

const lesson: LessonDef = {
	title: L('How AlphaZero picks a move', 'AlphaZero 怎样选一步棋'),
	summary: L('A tiny policy/value network looks at your tic-tac-toe board, then a tree search checks its first impression with 50 simulations; you play against it and see how self-play trained it.', '一个小小的策略/价值网络先看你的井字棋棋盘，再由树搜索用 50 次模拟检查它的第一印象；你和它下一盘，并看看自我对弈是怎样训练它的。'),
	steps,
	fixtures: [
		{ id: 'guessed', label: L('the start position after a guess', '猜过之后的开局局面'), raw: { moves: [2, 3, 7], guess: 4 } },
		{ id: 'game-over', label: L('a finished game against the search', '和搜索下完的一盘'), raw: { moves: [2, 3, 7], game: [4, 0, 1, 2, 3, 5, 6, 7, 8] } },
		{ id: 'search-first', label: L('the search moves first', '搜索先走'), raw: { moves: [4, 0], game: [2], aiFirst: true } },
		{ id: 'over', label: L('a board where X has already won', 'X 已经赢了的棋盘'), raw: { moves: [0, 3, 1, 4, 2] }, expect: { empty: true } }
	]
};

export default lesson;
