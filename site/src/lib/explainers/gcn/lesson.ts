// gcn: the guided lesson (design §7 outline, 11 steps). Every number is a {placeholder}: structure from the graph and
// the spec, live values from the fwd trace, the mp / smooth frames and the train source, measured ones from facts.json.
//   1 graph + adjacency (edit a friendship) → 2 one-hot features → 3 self-loops → 4 Â with live degrees (predict) →
//   5 X Wᵀ → 6 messages into the member in focus (play) → 7 bias + tanh → 8 two hops (predict) → 9 the 2-D embedding by
//   faction → 10 pick the labelled members and retrain (task) → 11 over-smoothing (limits).
// Claims that are not computed live are tested in claims.test.ts.
import type { At, ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { graphOf, hops, KARATE, LABELS, N, neighbours, sorted } from './graph.ts';
import { retrained } from './views/graph/meta.ts';

type RT = ExplainerRuntime;
const I = (en: string, zh: string): I18n => ({ en, zh });
const [HI, OFF] = KARATE.labelled;
const G = (rt: RT) => graphOf(rt.input);
const M = (at: At) => (at.frame?.metrics ?? {}) as Record<string, number>;
/** "(0.12, −0.3, …)" from metrics p0, p1, …; NaN when a number is missing, so lint reports it */
const vec = (m: Record<string, number>, p: string, n = 4) => {
	const v = Array.from({ length: n }, (_, f) => m[`${p}${f}`]);
	return v.every(Number.isFinite) ? `(${v.map((x) => fmt3(x)).join(', ')})` : NaN;
};
const friends = (n: number) => plural(n, { one: '{n} friend', many: '{n} friends', zh: '{n} 个朋友' });
const dim = (rt: RT, id: string) => Number(rt.spec?.layers.find((l) => l.id === id)?.outShape[1] ?? NaN);
const view = (mode: string) => ({ key: 'gcn/graph', props: () => ({ mode }) });

const lesson: LessonDef = {
	title: I('A graph convolutional network on a karate club', '空手道俱乐部上的图卷积网络'),
	summary: I('Follow one member of Zachary’s karate club through a graph convolutional network: friendships as a table, messages from friends with real numbers, two layers, the two factions in two numbers, then label members yourself and retrain.', '跟着 Zachary 空手道俱乐部的一个成员走过图卷积网络：把朋友关系写成表，带着真实数字看朋友发来的消息，再到两层、用两个数分出两派，最后自己标注成员并重新训练。'),
	steps: [
		{
			id: 'graph', source: 'fwd', at: { layer: 'adj' }, view: view('adj'), provenance: 'live',
			title: I('A club as a graph', '把俱乐部看成一张图'),
			narration: {
				en: 'Wayne Zachary recorded which of the {n} members of a university karate club were friends: {m0} pairs{edited}. The club then split between the instructor, Mr. Hi (member {hi}), and the Officer (member {off}). As a graph, each member is a node and each friendship an edge; the adjacency matrix A has a 1 in row i, column j when i and j are friends, so row {focus} marks the {k} of member {focus}.',
				zh: 'Wayne Zachary 记录了一个大学空手道俱乐部 {n} 个成员之间谁和谁是朋友：一共 {m0} 对{edited}。后来俱乐部分裂成两派：教练 Mr. Hi（成员 {hi}）一派和干事（成员 {off}）一派。画成图时，每个成员是一个节点，每段朋友关系是一条边；当 i 和 j 是朋友时，邻接矩阵 A 的第 i 行第 j 列是 1，所以第 {focus} 行标出了成员 {focus} 的 {k}。',
				vars: (rt) => {
					const g = G(rt);
					const edited = JSON.stringify(g.edges) === JSON.stringify(KARATE.edges) ? I('', '') : I(`; your edited copy has ${g.edges.length}`, `；你改过的版本有 ${g.edges.length} 对`);
					return { n: N, m0: KARATE.edges.length, edited, hi: HI, off: OFF, focus: g.focus, k: friends(neighbours(g, g.focus).length) };
				}
			},
			look: I('Hover a member: its friendships light up in both pictures. Click a square of A to add or remove a friendship (every later step re-computes), or click a member to follow it instead.', '把鼠标移到一个成员上：它的朋友关系在两张图里同时亮起。点 A 的一个格子可以加上或去掉一段朋友关系（后面每一步都会重新计算），点一个成员则改为跟随它。'),
			terms: ['graph', 'adjacency-matrix']
		},
		{
			id: 'features', source: 'fwd', at: { layer: 'x' }, view: view('x'), provenance: 'live',
			title: I('What each member knows: only who it is', '每个成员只知道自己是谁'),
			narration: {
				en: 'A network also needs some numbers about each node, its features. Here it gets nothing about the members themselves: member i gets a row of {n} numbers, all 0 except a 1 in column i, a one-hot row. So whatever the network learns about the factions must come from the friendships.',
				zh: '网络还需要每个节点自身的一些数，叫作特征。这里它对成员本人一无所知：成员 i 得到一行 {n} 个数，除了第 i 列是 1，其余全是 0，这叫独热行。所以网络学到的关于两派的一切，都只能来自朋友关系。',
				vars: () => ({ n: N })
			},
			look: I('The diagonal of 1s: each member’s own column.', '对角线上的 1：每个成员自己的那一列。'),
			terms: ['one-hot']
		},
		{
			id: 'loops', source: 'mp', at: { phase: 'loop' }, view: view('mp'), provenance: 'live',
			title: I('Every member also listens to itself', '每个成员也听自己的'),
			narration: {
				en: 'A graph layer lets every member collect numbers from its friends; let us follow member {focus} through layer 1. First a 1 is added on the diagonal of A, giving A + I (I has 1s on its diagonal), so each member’s own row counts too: a self-loop. Member {focus} has {k}, so it listens to {d} including itself; d is called its degree.',
				zh: '图卷积层让每个成员从朋友那里收集一些数；我们跟着成员 {focus} 走过第 1 层。先在 A 的对角线上加 1，得到 A + I（I 的对角线上都是 1），这样每个成员自己的行也算进去：这叫自环。成员 {focus} 有 {k}，所以连同自己，它一共听取 {d}的消息；d 叫作它的度。',
				vars: (_rt, at) => ({ focus: M(at).node, k: friends(M(at).friends), d: plural(M(at).d, { one: 'd = {n} member', many: 'd = {n} members', zh: 'd = {n} 个成员' }) })
			},
			look: I('In the table A + I, the 1s on the diagonal are the self-loops and the outlined row is the member in focus; the line under the drawing counts its d.', '在表 A + I 里，对角线上的 1 就是自环，加框的那一行是关注的成员；图下方那一行数出了它的 d。'),
			terms: ['degree']
		},
		{
			id: 'norm', source: 'mp', at: { phase: 'norm' }, view: view('mp'), provenance: 'live',
			title: I('How much each message counts: Â', '每条消息占多少分量：Â'),
			narration: {
				en: 'If member {focus} simply added up its friends’ rows, members with many friends would get huge sums. So each edge i–j gets the weight Âᵢⱼ = 1 / √(dᵢ × dⱼ), using the degrees at both ends, and the self-loop gets 1 / dᵢ = {self}. {range}',
				zh: '如果成员 {focus} 只是把朋友们的行直接加起来，朋友多的成员就会得到很大的和。所以每条边 i–j 得到权重 Âᵢⱼ = 1 / √(dᵢ × dⱼ)，用的是两端的度；自环得到 1 / dᵢ = {self}。{range}',
				vars: (_rt, at) => {
					const m = M(at);
					const range = pick(m.friends > 0, I(`Its friends’ degrees run from ${m.dLow} to ${m.dHub}.`, `它的朋友们的度从 ${m.dLow} 到 ${m.dHub} 不等。`), I('It has no friends, so the self-loop is its only edge.', '它没有朋友，所以自环是它唯一的边。'));
					return { focus: m.node, self: m.self, range };
				}
			},
			predict: {
				id: 'gcn-weight',
				question: I('Take a friend who has many friends of its own and a friend who has few. Whose message gets the larger weight Âᵢⱼ?', '一个朋友自己有很多朋友，另一个朋友的朋友很少。谁发来的消息权重 Âᵢⱼ 更大？'),
				options: [I('The friend with many friends', '朋友多的那个'), I('The friend with few friends', '朋友少的那个'), I('Both get the same weight', '两个一样大')],
				answer: 1,
				explain: I('Âᵢⱼ divides by √dⱼ, so the more friends j has, the smaller its weight: a popular member’s voice is spread over many listeners. The numbers above the friends in the drawing are their weights.', 'Âᵢⱼ 要除以 √dⱼ，所以 j 的朋友越多，它的权重越小：受欢迎的成员的声音被分给了很多听众。图中朋友上方的数就是它们的权重。')
			},
			look: I('Each row of the table works out one weight from two degrees, 1 / √(dᵢ × dⱼ); the same weights float above the members in the drawing.', '表的每一行用两个度算出一个权重：1 / √(dᵢ × dⱼ)；同样的权重也标在图中成员的上方。'),
			terms: ['degree']
		},
		{
			id: 'transform', source: 'mp', at: { phase: 'transform' }, view: view('mp'), provenance: 'live',
			title: I('Each member’s own message: X Wᵀ', '每个成员自己的消息：X Wᵀ'),
			narration: {
				en: 'Next, every member’s one-hot row is multiplied by the same learned weights of layer 1: a table W of {h} × {n} numbers, used as Wᵀ, that is with its rows and columns swapped ({n} × {h}). A one-hot row with its 1 in column j, times a table, just picks out row j of the table. So member j’s row of X Wᵀ is column j of W: {h} learned numbers. For member {focus} they are {x}.',
				zh: '接着，每个成员的独热行都乘以第 1 层同一组学来的权重：一张 {h} × {n} 的表 W，用的是 Wᵀ，也就是把它的行和列互换（{n} × {h}）。如果一行独热行的 1 在第 j 列，它乘一张表就只是挑出这张表的第 j 行。所以 X Wᵀ 中成员 j 的那一行就是 W 的第 j 列：{h} 个学来的数。成员 {focus} 的这几个数是 {x}。',
				vars: (rt, at) => ({ h: dim(rt, 'gc1'), n: N, focus: M(at).node, x: vec(M(at), 'x') })
			},
			look: I('Each row of the table: the 4 numbers one member will send to its friends (only the member in focus and its friends are listed).', '表的每一行：一个成员要发给朋友们的 4 个数（只列出关注的成员和它的朋友）。'),
			terms: ['weight']
		},
		{
			// The narration works out the first message (the member's own): the scrubber starts there and the table keeps it.
			// The cursor stays at the last message: re-resolving it after a new focus records the whole range (the kit
			// ensures play.to only when the stage mounts; with the cursor at the first message the range stopped short).
			id: 'messages', source: 'mp', at: { phase: 'message', last: true }, play: { from: { phase: 'message' }, to: { phase: 'message', last: true }, interval: 1100, unit: I('message', '消息') }, view: view('mp'),
			provenance: 'live',
			title: I('Messages into the member in focus', '流入关注成员的消息'),
			narration: {
				en: 'Now every member with an edge into member {focus}, itself included, sends its {h} numbers multiplied by that edge’s weight in Â: a message. The first is its own: {a} × {xj} = {mj}. {rest}',
				zh: '现在，每个有边连到成员 {focus} 的成员（包括它自己）都把自己的 {h} 个数乘上这条边在 Â 里的权重发过来：这就是一条消息。第一条是它自己的：{a} × {xj} = {mj}。{rest}',
				vars: (rt, at) => {
					const tape = at.tape ?? rt.sources.mp?.tape, i1 = tape ? tape.find({ phase: 'message' }) : -1;
					const m = (i1 >= 0 ? tape!.frame(i1).metrics : {}) as Record<string, number>, more = m.of - 1;
					// prettier-ignore
					const rest = more > 1 ? I(`The other ${more} follow one edge at a time, and the row Σ adds all ${m.of} up, number by number.`, `其余 ${more} 条沿着边一条一条地到达，Σ 那一行把全部 ${m.of} 条消息按位置逐个相加。`) : more === 1 ? I('The other one follows along the edge, and the row Σ adds the two up, number by number.', '另一条沿着边到达，Σ 那一行把这两条消息按位置逐个相加。') : I('It has no friends, so this is its only message, and Σ is just this message.', '它没有朋友，所以这是它唯一的消息，Σ 就等于这条消息。');
					return { focus: m.node, h: dim(rt, 'gc1'), a: m.a, xj: vec(m, 'x'), mj: vec(m, 'm'), rest };
				}
			},
			look: I('Press Play: the thick edge brings the newest message, the line under the table works it out, and Σ is the sum so far.', '按播放：粗线送来最新的一条消息，表下面那一行算出它，Σ 是到目前为止的和。'),
			terms: ['message-passing']
		},
		{
			id: 'tanh', source: 'mp', at: { phase: 'activate' }, view: view('mp'), provenance: 'live',
			title: I('Add a bias, then tanh', '加上偏置，再用 tanh'),
			narration: {
				en: 'All the messages add up to Σ = {s}. A learned bias b is added, giving z = {z}, and tanh squashes every number into the range from −1 to 1. So after layer 1, member {focus} is described by {h} new numbers, {hv}, {mix}. The same recipe runs for all {n} members at once: tanh(Â X Wᵀ + b).',
				zh: '所有消息加起来是 Σ = {s}。再加上一个学来的偏置 b，得到 z = {z}，然后用 tanh 把每个数压到 −1 到 1 之间。于是经过第 1 层，成员 {focus} 由 {h} 个新数 {hv} 来描述，{mix}。所有 {n} 个成员同时按这个办法计算：tanh(Â X Wᵀ + b)。',
				vars: (rt, at) => ({ s: vec(M(at), 's'), z: vec(M(at), 'z'), focus: M(at).node, h: dim(rt, 'gc1'), hv: vec(M(at), 'h'), n: N, mix: pick(neighbours(G(rt), M(at).node).length > 0, I('which mix its own column of W with its friends’', '它们混合了 W 中它自己的那一列和它朋友们的那几列'), I('which come from its own column of W alone, since no friend sends anything', '它们只来自 W 中它自己的那一列，因为没有朋友发来任何东西')) })
			},
			look: { en: 'The last row is member {focus}’s output of layer 1.', zh: '最后一行就是成员 {focus} 在第 1 层的输出。', vars: (_rt, at) => ({ focus: M(at).node }) },
			terms: ['bias', 'activation-function']
		},
		{
			id: 'hops', source: 'fwd', at: { layer: 'act2' }, view: view('hops'), provenance: 'live',
			title: I('Layer 2: friends of friends', '第 2 层：朋友的朋友'),
			narration: {
				en: 'Layer 2 repeats the recipe on these rows with its own weights and ends with {e} numbers per member; member {focus} gets {ev}. But a friend’s row from layer 1 already holds numbers from that friend’s own friends.',
				zh: '第 2 层用自己的权重，在这些行上把同样的步骤再做一遍，最后每个成员得到 {e} 个数；成员 {focus} 得到 {ev}。但要注意，一个朋友在第 1 层得到的行，已经含有那个朋友自己的朋友们的数。',
				vars: (rt) => {
					const g = G(rt);
					const t = rt.sources.fwd?.trace?.t.act2;
					return { e: dim(rt, 'act2'), focus: g.focus, ev: t ? vec({ v0: t.data[g.focus * 2], v1: t.data[g.focus * 2 + 1] }, 'v', 2) : NaN };
				}
			},
			predict: {
				id: 'gcn-hops',
				question: I('After two graph layers, can the message of a member three friendships away reach the member in focus?', '经过两个图卷积层，隔着三段朋友关系的成员发出的消息，能传到关注的成员那里吗？'),
				options: [I('Yes: after one layer everyone is mixed in', '能：一层之后所有人都混进来了'), I('No: messages travel at most two friendships', '不能：消息最多走两段朋友关系'), I('Only its direct friends get through', '只有直接的朋友能传到')],
				answer: 1,
				explain: I(
					'Each layer carries messages one edge further: layer 1 brings in the friends, layer 2 the friends of friends. A message from three steps away would need a third layer. (Its friendships still count a little: they raise the degree d of a friend of a friend, and so change the weights Â.)',
					'每一层把消息往外多传一条边：第 1 层带来朋友，第 2 层带来朋友的朋友。隔着三步的消息需要第三层才能传到。（它的朋友关系仍有一点影响：它们会增加朋友的朋友的度 d，从而改变权重 Â。）'
				)
			},
			look: I('The rings of colour: the member in focus, its friends, their friends. The line under the drawing counts who reached it and gives its 2 numbers.', '一圈圈的颜色：关注的成员、它的朋友、朋友的朋友。图下方那一行数出了有哪些人的消息传到了它，并给出它的 2 个数。'),
			terms: ['layer']
		},
		{
			id: 'embed', source: 'fwd', at: { layer: 'probs' }, view: view('embed'), provenance: 'live',
			title: I('Two numbers per member: the factions part ways', '每个成员两个数：两派分开了'),
			narration: {
				en: 'Each member’s {e} numbers place it on this square, a 2-D embedding; the colours show the faction each member really joined. The head, a linear layer, draws one straight line and puts the members on one side with Mr. Hi, the rest with the Officer. Trained with only members {hi} and {off} labelled, the network sorts {right} of the {rest} other members correctly{wrong}.',
				zh: '每个成员的 {e} 个数把它放在这个方块里的一个位置上，这叫二维嵌入；颜色表示每个成员实际加入了哪一派。输出头是一个线性层，它画出一条直线，把线一侧的成员归给 Mr. Hi，另一侧归给干事。网络训练时只知道成员 {hi} 和 {off} 的标签，其余 {rest} 个成员中它分对了 {right} 个{wrong}。',
				vars: (rt) => {
					const g = G(rt);
					const z = rt.sources.fwd?.trace?.t.head?.data;
					if (!z) return { e: NaN } as Record<string, TmplVal>;
					const s = sorted(z, KARATE.labelled);
					const w = s.wrong[0];
					const other = w === undefined ? 0 : neighbours(g, w).filter((j) => LABELS[j] !== LABELS[w]).length;
					const wrong =
						s.wrong.length === 0
							? I('', '')
							: s.wrong.length === 1
								? I(`; it gets member ${w} wrong, and ${other} of that member’s ${neighbours(g, w).length} friends joined the other faction`, `；它分错了成员 ${w}，而这个成员的 ${neighbours(g, w).length} 个朋友里有 ${other} 个加入了另一派`)
								: I(`; it gets ${s.wrong.length} members wrong`, `；有 ${s.wrong.length} 个成员分错了`);
					return { e: dim(rt, 'act2'), hi: HI, off: OFF, right: s.right, rest: s.rest, wrong };
				}
			},
			look: I('Dashed rings: the two labelled members. ✕: on the wrong side of the line. Near the two corners many dots lie on top of each other, as the line under the square counts.', '虚线圈：两个已标注的成员。✕：落在线的错误一侧。在两个角落附近，很多点叠在一起，方块下面那一行数出了有多少个。'),
			terms: ['embedding', 'semi-supervised', 'head']
		},
		{
			id: 'retrain', source: 'train', at: 'last', view: view('pick'), provenance: 'live',
			title: I('Your turn: choose the labelled members', '轮到你：选择已标注的成员'),
			narration: {
				en: 'So far the labels came from members {hi} and {off}. Here a fresh network with the same {p} parameters starts from random numbers and trains in your browser for {steps} steps. Training shrinks its loss, a number that measures how wrong it is about the members labelled now ({labs}) only. It then sorts {right} of the other {rest} into the faction they joined{same}.',
				zh: '到目前为止，标签来自成员 {hi} 和 {off}。这里一个同样有 {p} 个参数的新网络从随机数开始，在你的浏览器里训练 {steps} 步。训练让它的损失变小：这个数只衡量它对现在已标注的成员（{labs}）错了多少。训练之后，它把其余 {rest} 个成员中的 {right} 个分到了他们实际加入的一派{same}。',
				vars: (rt, at) => {
					const r = retrained({ rt, at });
					const l = G(rt).labelled;
					// both labels from one faction: the loss never sees the other one; count who ends up on the labelled side
					const side = l.length === 2 && LABELS[l[0]] === LABELS[l[1]] ? LABELS[l[0]] : -1;
					const on = r && side >= 0 ? LABELS.map((_, i) => i).filter((i) => !l.includes(i) && (r.z[i * 2 + 1] > r.z[i * 2] ? 1 : 0) === side).length : 0;
					const same = side < 0 ? I('', '') : I(`; both labels come from one faction, so the loss never sees the other one, and it puts ${on} of them on that side`, `；两个标签都来自同一派，损失从来看不到另一派，于是它把其中 ${on} 个人分到了这一派`);
					return { hi: HI, off: OFF, p: Number(rt.spec?.params ?? NaN), steps: r?.step ?? NaN, labs: l.length ? I(l.join(', '), l.join('、')) : I('none', '无'), right: r?.right ?? NaN, rest: r?.rest ?? NaN, same };
				}
			},
			task: {
				prompt: I('Click members to label two others, one from each faction, and watch the network retrain. Then try two members of the same faction.', '点击成员，给另外两个人加标签（每派一个），看网络重新训练。再试试同一派的两个人。'),
				// done when both labels are new members (the waiting line asks for two members other than 0 and 33) and the
				// network has finished retraining on them, so Play does not move on before the new answer is on the stage
				done: (rt, at) => ((l) => l.length === 2 && !l.includes(HI) && !l.includes(OFF))(G(rt).labelled) && at?.frame?.step !== undefined && at.frame.step === (at.tape?.opt as { steps?: number } | null | undefined)?.steps,
				hint: I('A click on a third member drops the oldest label. With two labels from one faction the loss never sees the other faction.', '点第三个成员时，最早的标签会被去掉。如果两个标签都来自同一派，损失就永远看不到另一派。'),
				waiting: I('Not yet: label two members other than 0 and 33.', '还没有：给 0 和 33 以外的两个成员加标签。')
			},
			// time to read the new count and to try two members of one faction before Play moves on (measured at 390 px: the
			// step moves on 14 s after the second tap; the retraining takes about 2.5 s of that)
			linger: 14000,
			terms: ['label', 'loss', 'parameter']
		},
		{
			id: 'smooth', source: 'smooth', at: 'last', play: { from: 'first', to: 'last', interval: 900, unit: I('round', '轮') }, view: view('smooth'), provenance: 'live', kindOf: 'limits',
			title: I('Why not eight layers? Over-smoothing', '为什么不用八层？过度平滑'),
			narration: {
				en: 'Every layer averages each member with its friends once more. Average the one-hot rows with Â {k} times, with no weights, and the two leaders’ rows (members {hi} and {off}) {grow}. This blurring is called over-smoothing, and it is one reason graph networks on bigger graphs usually keep two or three layers. On this small club it barely matters: over {fact:seeds} random starts, trained 8-layer networks sorted {fact:deepAcc|pct} of the unlabelled members on average, 2-layer ones {fact:shallowAcc|pct}.',
				zh: '每一层都会把每个成员和它的朋友再平均一次。把独热行用 Â 平均 {k} 次（中间没有权重），两个领头人（成员 {hi} 和 {off}）的行{grow}。这种变模糊叫作过度平滑，也是在更大的图上，图网络通常只用两三层的原因之一。在这个小俱乐部上它影响不大：各用 {fact:seeds} 次随机初始化训练，8 层网络对未标注成员的平均正确率是 {fact:deepAcc|pct}，2 层网络是 {fact:shallowAcc|pct}。',
				vars: (rt, at) => {
					const tape = rt.sources.smooth?.tape;
					const i2 = tape ? tape.find({ phase: 'round', step: 2 }) : -1;
					const lead = M(at).leaders;
					const lead2 = i2 >= 0 ? tape!.frame(i2).metrics!.leaders : NaN;
					const k = M(at).k;
					if (!Number.isFinite(lead) || !Number.isFinite(lead2)) return { k, hi: HI, off: OFF, grow: NaN };
					const sim = I('their cosine similarity (1 means the same direction)', '它们的余弦相似度（1 表示方向完全相同）');
					// prettier-ignore
					const grow = lead > lead2 + 1e-9 ? I(`grow alike: ${sim.en} goes from ${fmt3(lead2)} after 2 rounds to ${fmt3(lead)} after ${k}, and the colours fade`, `变得越来越像：${sim.zh}从 2 轮后的 ${fmt3(lead2)} 升到 ${k} 轮后的 ${fmt3(lead)}，颜色也越来越淡`)
						: hops(G(rt), HI)[OFF] < 0 ? I(`stay unlike: no chain of friendships links them, so ${sim.en} stays ${fmt3(lead)}`, `始终不像：没有一串朋友关系把他们连起来，所以${sim.zh}一直是 ${fmt3(lead)}`) : I(`change: ${sim.en} is ${fmt3(lead2)} after 2 rounds and ${fmt3(lead)} after ${k}`, `有变化：${sim.zh}在 2 轮后是 ${fmt3(lead2)}，${k} 轮后是 ${fmt3(lead)}`);
					return { k, hi: HI, off: OFF, grow };
				}
			},
			look: I('Press Play: round by round, colour spreads from the two leaders, then fades as everyone drifts to the same mix. The bars under the drawing show the leaders’ similarity after each round.', '按播放：一轮一轮地，颜色先从两个领头人那里扩散开，然后随着大家都趋向同一种混合而变淡。图下方的柱子是每一轮之后两个领头人的相似度。'),
			terms: ['over-smoothing']
		}
	],
	fixtures: [
		{ id: 'club', label: I('Zachary’s club as recorded', 'Zachary 记录的俱乐部'), raw: KARATE },
		{ id: 'none', label: I('no friendships at all', '完全没有朋友关系'), raw: { ...KARATE, edges: [] } },
		{ id: 'bridge', label: I('one more friendship, 16–33', '多一段朋友关系：16–33'), raw: { ...KARATE, edges: [...KARATE.edges, [16, 33]] } },
		{ id: 'same-side', label: I('members 1 and 2 labelled (both with Mr. Hi), focus on 0', '标注成员 1 和 2（都在 Mr. Hi 一派），关注成员 0'), raw: { ...KARATE, labelled: [1, 2], focus: 0 } },
		{ id: 'hub', label: I('focus on the Officer, member 33', '关注干事，即成员 33'), raw: { ...KARATE, focus: 33 } }
	]
};

export default lesson;
