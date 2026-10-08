// alphazero-ttt: the package manifest (design §4.9). Kinds A and C: a policy/value network trained offline by self-play
// (pipeline/alphazero-ttt/train.py; 9 → 96 → 96 → 9 priors + 1 value, 11,242 parameters) and Monte Carlo tree search
// with PUCT on top of it, in TypeScript (search.ts), on the learner's own position.
//   fwd     the exported network traced on the position (priors and value; parity ≤ 1e-4 against PyTorch)
//   search  the search from that position, event by event (process.ts; same frames as pipeline/alphazero-ttt/reference.py)
//   game    the learner's game against the search (50 simulations per move)
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { rawOf } from './adapters/ttt-board/adapter.ts';
import { withWeights } from './process.ts';
import { L } from './rules.ts';
import { weightsOf } from './search.ts';

type R = { input: unknown; model: Parameters<typeof weightsOf>[0] | null };
const w = (rt: R) => (rt.model ? weightsOf(rt.model) : null);

const manifest: ExplainerManifest = {
	slug: 'alphazero-ttt',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed', // teaching review: notes/review-alphazero-ttt.md
	assets: { dir: 'models/alphazero-ttt', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'ttt-board', presets: 'presets.json', start: { moves: [2, 3, 7] } },
	output: { view: 'kit.probbars', layer: 'prior' },
	sources: {
		fwd: { kind: 'trace' },
		search: { kind: 'process', process: 'search', run: 'main', input: (rt) => withWeights({ moves: rawOf(rt.input).moves }, w(rt)) },
		game: { kind: 'process', process: 'game', run: 'main', input: (rt) => withWeights({ you: rawOf(rt.input).game ?? [], aiFirst: !!rawOf(rt.input).aiFirst }, w(rt)) }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			L('Read a board as the network’s nine input numbers, and say what its two answers mean: a prior for each move and a value from −1 to +1.', '把棋盘读成网络的九个输入数，并说出它的两个回答是什么意思：每一步的先验，以及一个在 −1 和 +1 之间的价值。'),
			L('Work out one PUCT score U = Q + c·P·√N / (1 + n) by hand, and say why a move never tried can beat a move already tried.', '亲手算出一个 PUCT 分数 U = Q + c·P·√N / (1 + n)，并说明为什么一步没试过的棋能胜过试过的棋。'),
			L('Follow one simulation through select, expand and back up, and say why the value changes sign at every level.', '跟着一次模拟走完选择、展开和回传，并说明为什么价值每往上一层就变一次号。'),
			L('Explain with the measured numbers why the most-visited move after 50 simulations beats the network’s favourite move.', '用实测的数字说明：为什么 50 次模拟后访问最多的那一步，比网络最看好的那一步更好。'),
			L('Describe how self-play turns visit counts and game results into training targets.', '说出自我对弈怎样把访问次数和对局结果变成训练目标。')
		],
		prequestion: {
			id: 'alphazero-ttt-pre',
			question: L('AlphaZero has a network that already guesses good moves. Why does it still search, trying moves ahead in its head, before every move?', 'AlphaZero 已经有一个会猜好棋的网络。为什么每走一步之前，它还要搜索，也就是在脑子里往后试几步？'),
			options: [L('The network is only a quick guess; looking ahead checks the guess and fixes some of its mistakes', '网络只是一个快速的猜测；往后看能检查这个猜测，改正它的一部分错误'), L('The search is how it learns the rules of the game', '搜索是它学会游戏规则的方式'), L('It does not need to: the network alone already plays perfectly', '其实不需要：光靠网络就已经下得很完美')],
			answer: 0,
			explain: L('The network gives a first impression in one quick pass. On most positions its favourite move is a perfect one, but not on all of them. The search spends a few dozen looks ahead to check that impression, and the lesson measures how many mistakes it fixes.', '网络算一遍，就给出第一印象。在大多数局面里，它最看好的那一步是完美的一步，但不是全部。搜索再往后看几十次来检查这个印象；这节课会量出它改正了多少错误。')
		},
		tryThis: [
			{ id: 'empty', title: L('The empty board', '空棋盘'), setup: { preset: 'empty' }, predict: L('With perfect play every first move draws. Will 50 simulations pile up on one move, or spread out?', '如果双方都下得完美，第一步下在哪里都是平局。50 次模拟会堆在某一步上，还是分散开？'), look: L('The visit counts on the board, next to the priors.', '棋盘上的访问次数，和旁边的先验比一比。'), goto: { step: 'pick' } },
			{ id: 'chance', title: L('A chance for O', 'O 的机会'), setup: { preset: 'chance' }, predict: L('O can force a win here. Will the network’s favourite move find it, or only the search?', '这里 O 能强行赢下来。网络最看好的那一步能找到它吗，还是只有搜索能？'), look: L('Which move has the largest prior, and which one ends with the most visits.', '哪一步的先验最大，哪一步最后访问最多。'), goto: { step: 'pick' } }
		],
		quiz: [
			{
				q: L('A move has Q = 0.2, prior P = 0.4 and n = 2 visits; its parent has N = 9 visits and c = 1.5. What is its score U?', '某一步 Q = 0.2，先验 P = 0.4，被访问 n = 2 次；它的父节点访问了 N = 9 次，c = 1.5。它的分数 U 是多少？'),
				options: [L('0.8', '0.8'), L('0.6', '0.6'), L('1.4', '1.4')],
				answer: 0,
				explain: L('U = 0.2 + 1.5 × 0.4 × √9 / (1 + 2) = 0.2 + 1.5 × 0.4 × 3 / 3 = 0.2 + 0.6 = 0.8.', 'U = 0.2 + 1.5 × 0.4 × √9 / (1 + 2) = 0.2 + 1.5 × 0.4 × 3 / 3 = 0.2 + 0.6 = 0.8。')
			}
		],
		misconception: {
			claim: L('The search tries every possible continuation, like checking the whole game tree.', '搜索会把所有可能的走法都试一遍，就像把整棵博弈树查一遍。'),
			truth: L('It looks at a few dozen positions and follows the most promising lines. The priors decide where to look, and the value head stands in for playing each line to the end.', '它只看几十个局面，顺着最有希望的路线往下走。先验决定往哪里看，价值头则代替把每条路线下到底。'),
			step: 'limits'
		},
		// the self-play log is lesson step 10 (a chart); the auto Training section would print its raw metric key
		sections: { hide: ['training'], intros: { output: L('The priors: the network’s chance for each cell, before any search, largest first. Its favourite is not always the move it plays: the search may still choose another (lesson step 8).', '先验：搜索之前，网络给每个格子的概率，从大到小排列。它最喜欢的一步不一定就是它下的那一步：搜索最后仍可能选别的棋（课程第 8 步）。') } },
		lessonNote: L('Eleven steps: the board as numbers, the network’s two answers, one simulation of the search slowed down, what 50 simulations change, a game against it, how self-play trained it, and its limits.', '十一步：把棋盘变成数，网络的两个回答，放慢的一次搜索模拟，50 次模拟改变了什么，和它下一盘，自我对弈怎样训练它，以及它的局限。')
	},
	glossary: [
		{ id: 'mcts', term: L('Monte Carlo tree search (MCTS)', '蒙特卡洛树搜索（MCTS）'), def: L('Looking ahead by growing a tree of positions one simulation at a time, spending more simulations on the moves that look best so far.', '一次一次地模拟，把局面长成一棵树来往后看；看起来越好的走法，分到的模拟越多。'), related: ['policy'] },
		{ id: 'prior', term: L('prior (policy head)', '先验（策略头）'), def: L('The network’s first-impression chance for each legal move, before any search; the chances add up to 1.', '搜索之前，网络给每一步合法走法的第一印象概率；这些概率加起来等于 1。'), related: ['policy', 'softmax'] },
		{ id: 'value-head', term: L('value (value head)', '价值（价值头）'), def: L('The network’s guess of how the game will end for the player to move, from −1 (a loss) to +1 (a win).', '网络对“这盘棋对走棋一方会怎样结束”的猜测，从 −1（输）到 +1（赢）。'), related: ['head'] },
		{ id: 'puct', term: L('PUCT score', 'PUCT 分数'), def: L('U = Q + c·P·√N / (1 + n): how good a move looks so far (Q) plus a bonus for a high prior P and few visits n.', 'U = Q + c·P·√N / (1 + n)：一步棋目前看起来有多好（Q），加上一份奖励分，先验 P 越高、访问次数 n 越少，奖励越大。'), related: ['exploration', 'q-value'] },
		{ id: 'self-play', term: L('self-play', '自我对弈'), def: L('Training by letting the program play against itself; the games are the only training data.', '让程序自己和自己下棋来训练；这些对局就是全部的训练数据。'), related: ['reward'] }
	],
	catalog: {
		params: L('11,242 (9 → 96 → 96, then 9 priors and 1 value)', '11,242（9 → 96 → 96，再输出 9 个先验和 1 个价值）'),
		size: L('44 KB of weights', '权重 44 KB'),
		runtime: L('Own JS forward pass (checked against PyTorch) and tree search', '自研 JS 前向计算（已与 PyTorch 核对）和树搜索'),
		hook: L('Guess a tiny network’s favourite move, watch a tree search check it, then play against the search.', '先猜一个小网络最看好的一步，看树搜索怎样检查它，再和搜索下一盘。'),
		shows: L('Your board as nine numbers and a mask, the priors and the value against perfect play, one PUCT simulation slowed down (select with live numbers, expand, back up with the sign flip), the tree growing over 50 simulations, visits versus priors, a game against the search, and the self-play training log.', '把你的棋盘变成九个数和一个掩码、先验和价值（与完美下法对照）、放慢的一次 PUCT 模拟（带实时数字的选择、展开、带变号的回传）、50 次模拟中长大的树、访问次数对先验、和搜索下一盘，以及自我对弈的训练记录。')
	},
	// The method is AlphaZero's (and AlphaGo Zero's PUCT and self-play targets); the code, the network and the training are
	// written here. Facts about AlphaZero in the limits step come from the paper; Go's position count from Tromp.
	credits: [
		{ name: 'AlphaZero (Silver et al., Science 2018)', url: 'https://doi.org/10.1126/science.aar6404', license: 'Paper; ideas only (no code or figures used)', usedFor: 'The method: a policy/value network trained by self-play, Monte Carlo tree search with the PUCT rule, visit counts as the policy target; its simulation and TPU counts in the limits step', attribution: 'David Silver, Thomas Hubert, Julian Schrittwieser et al. (DeepMind)' },
		{ name: 'AlphaGo Zero (Silver et al., Nature 2017)', url: 'https://doi.org/10.1038/nature24270', license: 'Paper; ideas only (no code or figures used)', usedFor: 'The search details: select, expand and evaluate, back up; Dirichlet noise at the root and the eight board symmetries in training', attribution: 'David Silver, Julian Schrittwieser, Karen Simonyan et al. (DeepMind)' },
		{ name: 'The number of legal Go positions (Tromp, 2016)', url: 'https://tromp.github.io/go/legal.html', license: 'Web page; one number quoted', usedFor: 'The count of legal Go positions in the limits step', attribution: 'John Tromp' }
	]
};

export default manifest;
