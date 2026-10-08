// classic-ml: the guided lesson, twelve steps in three chapters (kNN 3, decision tree 4, k-means 5). Every number is a
// {placeholder} bound to a live frame, or built in vars() from one; the other factual sentences are claims.test.ts.
// Predict steps set the question up and leave the answer to the stage and predict.explain (lint AK412).
import type { At, ExplainerRuntime, Frame, I18n, LessonDef, StepDef } from '../../atlas/types/index.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { minus } from '../../atlas/core/format.ts';
import { plural } from '../../atlas/lesson/say.ts';
import { CENTROIDS, className, dataOf, DMAX, I, KS, rawOf, START } from './ml.ts';

type Step = StepDef<ExplainerRuntime>;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const frames = (at: At, phase: string): Frame[] => (at.tape ? Array.from({ length: at.tape.length }, (_, i) => at.tape!.frame(i)).filter((f) => f.phase === phase) : []);
/** a number inside a sentence built here, as a placeholder without a format prints it; a setting as set: −0.3 */
const n3 = (v: number) => formatValue(v, undefined, 'en') ?? '…';
const co = (v: number) => minus(String(Number(v.toFixed(2))));
const pct = (v: number) => formatValue(v, 'pct', 'en') ?? '…';
/** "a, b and c" / "a、b和c" (zh: or every part joined by `zsep`, "a，b，c") */
const list = (xs: I18n[], zsep?: string): I18n => ({ en: xs.length > 1 ? `${xs.slice(0, -1).map((x) => x.en).join(', ')} and ${xs[xs.length - 1].en}` : (xs[0]?.en ?? ''), zh: zsep || xs.length < 2 ? xs.map((x) => x.zh).join(zsep) : `${xs.slice(0, -1).map((x) => x.zh).join('、')}和${xs[xs.length - 1].zh}` });
/** a coordinate in a sentence: 3 decimals at most, never e-notation (−0.001, 0.5) */
const xy = (v: number) => minus(String(Number(v.toFixed(3))));
const canvas = (show: string, extra: Record<string, unknown> = {}) => ({ key: 'classic-ml/canvas', props: () => ({ show, ...extra }) });
const pts = (n: number) => plural(n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' });
/** the shallowest depth with the best held-out accuracy (null when nothing was held out) */
export function bestDepth(at: At): number | null {
	const ds = frames(at, 'depth').map((f) => f.metrics!).filter((x) => x.testAcc >= 0);
	return ds.length ? ds.reduce((b, x) => (x.testAcc > b.testAcc ? x : b)).d : null;
}

const steps: Step[] = [
	{
		id: 'knn-dist', source: 'knn', at: { phase: 'distances' }, view: canvas('knn', { formula: true }), provenance: 'live',
		title: I('A new point: measure its distance to every point', '一个新点：量出它到每个点的距离'),
		narration: {
			en: 'Your canvas holds {pts} in {cls}, and a new point, the ×, arrives at ({qx}, {qy}). k nearest neighbours (kNN) will guess its class from the points around it, so first it measures the straight-line distance from the new point to every point. The nearest is point {i0} (counting from 0), just {d0} away, and it is in {c0}.',
			zh: '你的画布上有 {pts}，分成 {cls}；一个新点 × 落在 ({qx}, {qy})。k 近邻（kNN）要根据周围的点来猜它的类别，所以先量出新点到每个点的直线距离。最近的是第 {i0} 号点（从 0 数起），距离只有 {d0}，属于{c0}。',
			vars: (_rt, at) => ({ pts: pts(m(at, 'n')), cls: plural(new Set(Array.from((at.tape?.state(0) as { label?: { data: ArrayLike<number> } })?.label?.data ?? [])).size || NaN, { one: '{n} class', many: '{n} classes', zh: '{n} 类' }), qx: co(m(at, 'qx')), qy: co(m(at, 'qy')), i0: m(at, 'i0'), d0: m(at, 'd0'), c0: className(m(at, 'c0')) })
		},
		look: I('The formula, worked out for the nearest point (x, y); the letter q marks the new point ×. Click the plane to move the new point.', '公式代入的是最近的那个点 (x, y)；下标 q 表示新点 ×。点击平面可以移动新点。'),
		terms: ['knn']
	},
	{
		id: 'knn-vote', source: 'knn', at: { phase: 'vote' }, view: canvas('knn'), provenance: 'live',
		title: I('The k nearest points vote', '最近的 k 个点投票'),
		narration: {
			en: 'Now kNN sorts the distances and keeps the k = {k} nearest {pts}{few}, {within}. {each}: {votes}, so kNN answers {pred}. {note}There is no training at all: the stored points are the whole model.',
			zh: '接着 kNN 把距离从小到大排好，只留最近的 k = {k} {pts}{few}，{within}。{each}：{votes}，所以 kNN 的回答是{pred}。{note}这里完全没有训练：存下来的点就是整个模型。',
			vars: (_rt, at) => {
				const v = Object.keys(at.frame?.metrics ?? {}).filter((k) => /^v\d$/.test(k)).map((k) => [Number(k.slice(1)), m(at, k)]).filter(([, n]) => n > 0);
				const c0 = at.tape?.frame(1).metrics?.c0 ?? NaN, pred = m(at, 'pred'), k = m(at, 'k'), asked = m(at, 'kAsked'), r = n3(m(at, 'r'));
				const note = m(at, 'tie') ? I('The vote is a tie, and a tie goes to the smaller class number. ', '票数打平时，归给编号小的类别。') : k > 1 && c0 !== pred ? I(`The single nearest point is in class ${c0}, but it is outvoted. `, `最近的那一个点属于第 ${c0} 类，但它被投票压过了。`) : I('', '');
				return {
					k, pts: plural(k, { one: 'point', many: 'points', zh: '个点' }), votes: list(v.map(([c, n]) => I(`${n} for class ${c}`, `第 ${c} 类 ${n} 票`)), '，'), pred: className(pred), note,
					few: asked > k ? I(` (you asked for k = ${asked}, but there are only ${k} points)`, `（你选了 k = ${asked}，可一共只有 ${k} 个点）`) : I('', ''),
					within: k === 1 ? I(`which lies on the dashed circle of radius ${r}`, `它就在半径 ${r} 的虚线圆上`) : I(`all within the dashed circle of radius ${r}`, `它们都在半径 ${r} 的虚线圆以内（最远的那个在圆上）`),
					each: k === 1 ? I('It votes for its own class', '它给自己的类别投一票') : I('Each one votes for its own class', '每个点给自己的类别投一票')
				};
			}
		},
		look: I('The dashed circle, the vote bars and the nearest points in the table. Try the k slider: when does the answer change?', '虚线圆、投票条和表里最近的点。拖一拖 k 滑块：答案在什么时候改变？')
	},
	{
		id: 'knn-map', source: 'map', at: { phase: 'map', last: true }, view: canvas('map'), provenance: 'live',
		play: { from: { phase: 'map', step: 1 }, to: { phase: 'map', last: true }, interval: 1500 },
		title: I('Ask about every spot: the decision map', '问遍每一处：决策图'),
		narration: {
			en: 'Ask kNN about every spot of the plane and colour each spot by its answer: that is its decision map, and the line where the colour changes is the decision boundary. The replay draws the map for k = {ks}. Guess first: which k leaves small islands around single stray points?',
			zh: '让 kNN 回答平面上每一处，再按回答给每一处上色：这就是它的决策图，颜色变化的那条线就是决策边界。回放会依次画出 k = {ks} 的决策图。先猜一猜：哪个 k 会在个别“跑错地方”的点周围留下小岛？',
			vars: (rt) => {
				const ks = KS.filter((k) => k <= dataOf(rt.input).label.length);
				return { ks: { en: list(ks.map((k) => I(String(k), String(k)))).en, zh: ks.join('、') } };
			}
		},
		look: I('The islands on the map, and the count of separate patches for each k in the chips.', '图上的小岛，以及标签里每个 k 的独立区域数。'),
		predict: {
			id: 'classic-ml-knn-map',
			question: I('Which k gives a decision map with small islands around single stray points?', '哪个 k 画出的决策图，会在个别“跑错地方”的点周围留下小岛？'),
			options: [I('k = 1', 'k = 1'), I('A large k, such as 25', '大的 k，比如 25'), I('k does not change the map', 'k 不会改变这张图')],
			answer: 0,
			explain: I('With k = 1 every spot takes the class of its one nearest point, so a stray point of one class claims a little island inside the other. With a larger k its neighbours outvote it and the boundary gets smoother: compare the patch counts for each k.', 'k = 1 时，每一处都取离它最近的那一个点的类别，所以一个跑错地方的点会在另一类的地盘里占出一个小岛。k 大一些，它就会被邻居投票压过，边界也变得更平滑：比一比每个 k 的区域数。')
		},
		terms: ['decision-boundary']
	},
	{
		id: 'tree-gini', source: 'tree', at: { phase: 'gini' }, view: canvas('gini', { formula: true }), provenance: 'live',
		title: I('A decision tree: how mixed are the points?', '决策树：这些点混得有多乱？'),
		narration: {
			en: 'A decision tree classifies with yes/no questions such as “is x ≤ t?”. {hold} To judge questions it measures how mixed points are with the Gini impurity G = 1 − Σ p² (p is each class’s share; Σ adds over the classes). {range}; for {tp}, G = {gini}.',
			zh: '决策树用“x ≤ t 吗？”这样的是非题来分类。{hold}要评判问题，它需要一把尺子来量一组点混得有多乱：基尼不纯度 G = 1 − Σ p²（p 是每个类别的占比，Σ 表示把各类加起来）。{range}；{tp}算出 G = {gini}。',
			vars: (_rt, at) => {
				const h = at.tape?.frame(0).metrics ?? {}, C = h.classes ?? NaN;
				const range = C === 1 ? I('G is 0 when all the points share one class', '所有点都属于同一类时，G 为 0')
					: C === 2 ? I('G is 0 for one class and 0.5, its largest, for two classes half and half', '只有一类时 G 是 0，两类各占一半时最大，是 0.5')
					: I(`G is 0 for one class and ${n3(1 - 1 / C)}, its largest, for ${C} classes in equal shares`, `只有一类时 G 是 0，${C} 类各占一样多时最大，是 ${n3(1 - 1 / C)}`);
				return { range, tp: plural(m(at, 'm'), { one: 'the one training point', many: 'the {n} training points', zh: '{n} 个训练点' }), gini: m(at, 'gini'), hold: h.nTest ? I(`First we hold back ${h.nTest} of your ${h.n} points, drawn as rings: the tree never learns from them, so later they show how it does on new points.`, `先从你的 ${h.n} 个点里留出 ${h.nTest} 个（画成空心圈）：树不会用它们来学习，所以之后它们能说明树在新点上表现如何。`) : I('With a single point there is nothing to hold back for testing.', '只有一个点，没有可以留出来做检验的点。') };
			}
		},
		look: I('Solid points train the tree; rings are held out. The formula uses the class counts of the training points.', '实心点用来训练树，空心圈是留出点。公式用的是训练点里各类的点数。'),
		terms: ['decision-tree', 'gini', 'dataset-split']
	},
	{
		id: 'tree-split', source: 'tree', at: { phase: 'node', step: 0 }, view: canvas('split', { formula: true }), provenance: 'live',
		title: I('The best first question', '最好的第一个问题'),
		narration: {
			en: 'The tree tries every cut halfway between two neighbouring points, on x and on y, and scores each cut by the Gini of its two sides, weighted by their sizes; the chart shows every score. {best}',
			zh: '树会试遍所有切法：在 x 上和 y 上，每两个相邻点的正中间都切一刀，再用两侧的基尼不纯度按点数加权来给每一刀打分；图表画出了每个分数。{best}',
			vars: (_rt, at) => {
				const q = `${m(at, 'axis') === 0 ? 'x' : 'y'}\u00a0≤\u00a0${n3(m(at, 'thr'))}`, [nL, nR, gL, gR, w, g, gain] = ['nL', 'nR', 'gL', 'gR', 'w', 'gini', 'gain'].map((k) => n3(m(at, k)));
				return { best: m(at, 'axis') >= 0 ? I(`The lowest score belongs to “${q}?”: ${nL} points answer yes with G = ${gL}, ${nR} answer no with G = ${gR}, weighted ${w}. That is ${gain} below the ${g} before the cut, the biggest drop of any cut.`, `得分最低的是“${q} 吗？”：${nL} 个点答“是”，G = ${gL}；${nR} 个点答“否”，G = ${gR}；加权后是 ${w}。比切之前的 ${g} 低了 ${gain}，是所有切法里降得最多的。`) : m(at, 'm') === 1 ? I('There is a single training point, so there is nothing to cut: the tree stays a single leaf.', '只有一个训练点，没有什么可切的：这棵树只有一片叶子。') : I(`${m(at, 'm') === 2 ? 'Both training points are' : `All ${m(at, 'm')} training points are`} in one class or on one spot, so no cut can make them purer: the tree stays a single leaf.`, `这 ${m(at, 'm')} 个训练点都属于同一类或挤在同一处，怎么切都不会更纯：这棵树只有一片叶子。`) };
			}
		},
		look: I('The chart: across is where a cut lies, up is its weighted Gini. The lowest point of either line is the cut the tree picks, drawn on the plane.', '图表：横向是这一刀切在哪里，纵向是它的加权基尼。两条线的最低点就是树选中的那一刀，画在平面上。')
	},
	{
		id: 'tree-grow', source: 'tree', at: { phase: 'node', last: true }, view: canvas('grow'), provenance: 'live',
		play: { from: { phase: 'node', step: 0 }, to: { phase: 'node', last: true }, phases: ['node'], unit: I('node', '节点') },
		title: I('Ask again inside each box', '在每个方框里接着问'),
		narration: {
			en: 'The cut splits the plane into two boxes, and the tree asks the best question again inside each box, then inside the new boxes, one box at a time. A box stops when it holds a single class, when no cut makes it purer, or when it sits {dmax} questions deep; then it is a leaf that answers with its most common class. Your tree asked {splits} and ended with {leaves}.',
			zh: '这一刀把平面分成两个方框，树在每个方框里再问一个最好的问题，然后在新的方框里接着问，一次问一个方框。方框里只剩一类、怎么切都不会更纯，或者已经问到第 {dmax} 层时，就停下来，成为一片叶子，回答它里面最常见的类别。你的树一共问了 {splits}，最后有 {leaves}。',
			vars: (_rt, at) => {
				const s = frames(at, 'node').filter((f) => f.metrics!.axis >= 0).length;
				return { dmax: DMAX, splits: plural(s, { one: '{n} question', many: '{n} questions', zh: '{n} 个问题' }), leaves: plural(s + 1, { one: '{n} leaf', many: '{n} leaves', zh: '{n} 片叶子' }) };
			}
		},
		look: I('Each new cut on the plane, and the same question as a node of the tree below it (the left branch answers yes).', '平面上的每一刀，以及下面那棵树里对应的节点（左分支回答“是”）。'),
		terms: ['decision-tree']
	},
	{
		id: 'tree-depth', source: 'tree', at: { phase: 'depth', last: true }, view: canvas('depth'), provenance: 'live',
		title: I('How deep? Overfitting', '多深才好？过拟合'),
		narration: {
			en: 'Every extra level of questions fits the training points at least as well as before: at depth {dmax} the tree gets {top|pct} of them right. {over} {now}',
			zh: '每多一层问题，树在训练点上的成绩只会持平或变好：到深度 {dmax}，它答对 {top|pct} 的训练点。{over}{now}',
			vars: (rt, at) => {
				const d = rawOf(rt.input).depth ?? DMAX, ds = frames(at, 'depth'), g = ds.find((f) => f.step === d)?.metrics ?? {}, b = bestDepth(at);
				const over = b !== null && m(at, 'testAcc') < (ds.find((f) => f.step === b)?.metrics?.testAcc ?? NaN)
					? I('But the deepest boxes wrap single stray points, which is memorising noise instead of learning the shape: overfitting. The held-out points show it, since some shallower tree does better on them.', '可最深的那些方框是在给个别跑错地方的点单独画框，这是在记噪声，而不是学形状：这就是过拟合。留出点暴露了这一点：某棵更浅的树在它们身上表现更好。')
					: I(`On these points the held-out accuracy does not drop at depth ${DMAX}. On noisier points it does: the deepest boxes then wrap single stray points, memorising noise instead of the shape, which is called overfitting.`, `在这些点上，深度 ${DMAX} 的留出点准确率并没有下降。点的噪声更大时就会下降：最深的那些方框会给个别跑错地方的点单独画框，记住的是噪声而不是形状，这叫过拟合。`);
				const train = pct(g.trainAcc ?? NaN);
				const now = g.testAcc >= 0
					? I(`At your depth, ${d}, the tree gets ${train} of the training points and ${pct(g.testAcc)} of the held-out points right.`, `在你选的深度 ${d}，树答对 ${train} 的训练点、${pct(g.testAcc)} 的留出点。`)
					: I(`At your depth, ${d}, it gets ${train} of the training points right; no point was held out.`, `在你选的深度 ${d}，它答对 ${train} 的训练点；没有留出点。`);
				return { over, dmax: DMAX, top: m(at, 'trainAcc'), now };
			}
		},
		look: I('Solid line: training accuracy; dashed: held out. Picking the depth with the held-out points makes them a validation set: a fair final score needs a third set used for nothing else.', '实线是训练准确率，虚线是留出点准确率。用留出点来挑深度，它们就成了验证集：要得到公平的最终成绩，还需要一份另外留着、什么都不参与的测试集。'),
		task: {
			prompt: I('Move the depth slider to the depth where the held-out accuracy (dashed line) is highest.', '把深度滑块拖到留出点准确率（虚线）最高的那个深度。'),
			done: (rt, at) => bestDepth(at) === null || rawOf(rt.input).depth === bestDepth(at),
			hint: I('The slider is under the chart. If two depths tie, take the shallower one: the simpler tree.', '滑块在图表下面。如果两个深度一样高，选浅的那个：树更简单。'),
			waiting: I('Not yet: move the slider to the highest point of the dashed line (if two depths tie, the shallower one).', '还没有：把滑块拖到虚线的最高点（如果两个深度一样高，选浅的那个）。')
		},
		terms: ['overfitting', 'dataset-split']
	},
	{
		id: 'km-init', source: 'km', at: { phase: 'init' }, view: canvas('km'), provenance: 'live',
		title: I('k-means: groups without labels', 'k-means：没有标签也能分组'),
		narration: {
			en: 'Now forget the colours: k-means sees only the positions of your {pts}, and must split them into {groups}, called clusters, on its own. It starts from {cents}, placed on points picked at random: {picks}.',
			zh: '现在忘掉颜色：k-means 只看得到你这 {pts}的位置，要自己把它们分成 {groups}，叫作簇。它从 {cents}出发，放在随机挑出的点上：{picks}。',
			vars: (_rt, at) => {
				const k = m(at, 'k'), p = Array.from({ length: k }, (_, j) => m(at, `p${j}`));
				return { pts: pts(m(at, 'n')), groups: plural(k, { one: 'k = 1 group', many: 'k = {n} groups', zh: 'k = {n} 组' }), cents: plural(k, { one: '1 centre, called a centroid (A)', many: `{n} centres, called centroids (${CENTROIDS.slice(0, k).join(', ')})`, zh: `{n} 个中心（叫作质心：${CENTROIDS.slice(0, k).join('、')}）` }), picks: list(p.map((i, j) => I(`${CENTROIDS[j]} on point ${i}`, `${CENTROIDS[j]} 在第 ${i} 号点`)), '，') };
			}
		},
		look: I('The lettered squares: the centroids. The points are gray: no group yet.', '带字母的方块就是质心。点是灰色的：还没有分组。'),
		terms: ['clustering', 'k-means', 'centroid']
	},
	{
		id: 'km-assign', source: 'km', at: { phase: 'assign', step: 0 }, view: canvas('km'), provenance: 'live',
		title: I('Assign: every point joins its nearest centroid', '分配：每个点归入最近的质心'),
		narration: {
			en: 'Step one: every point joins the centroid it is closest to. Point {p} (counting from 0), ringed at ({px}, {py}), lies {dists}, so it joins {j}{tie}. {all}, and the colours now show {groups}.',
			zh: '第一步：每个点归入离它最近的质心。第 {p} 号点（从 0 数起，画了圈）在 ({px}, {py})，{dists}，所以它归入 {j}{tie}。{all}，现在颜色显示出 {groups}。',
			vars: (_rt, at) => {
				const k = at.tape?.frame(0).metrics?.k ?? NaN, ds = Array.from({ length: k }, (_, j) => m(at, `d${j}`)), tie = ds.filter((d) => d === Math.min(...ds)).length > 1;
				return { p: m(at, 'p'), px: xy(m(at, 'px')), py: xy(m(at, 'py')), dists: list(ds.map((d, j) => I(`${n3(d)} from ${CENTROIDS[j]}`, `离 ${CENTROIDS[j]} ${n3(d)}`)), '，'), j: CENTROIDS[m(at, 'j')] ?? '…', tie: tie ? I(' (a tie goes to the earlier letter)', '（距离一样时，归给字母靠前的那个）') : I('', ''), all: plural(at.tape?.frame(0).metrics?.n ?? NaN, { one: 'It is the only point', many: 'All {n} points do the same', zh: '全部 {n} 个点都这样做' }), groups: plural(k, { one: 'one group', many: 'the {n} groups', zh: '{n} 个组' }) };
			}
		},
		look: I('The lines from the ringed point to every centroid, and its distance to each one (the nearest is highlighted).', '从画圈的点连到每个质心的线，以及它到每个质心的距离（最近的那个高亮）。')
	},
	{
		id: 'km-update', source: 'km', at: { phase: 'update', step: 0 }, view: canvas('km', { formula: true }), provenance: 'live',
		title: I('Update: each centroid moves to the middle of its group', '更新：质心移到自己那组的中间'),
		narration: {
			en: 'Step two: every centroid moves to the mean of the points that joined it, the middle of its group. {calc} {moves}',
			zh: '第二步：每个质心移到归入它的那些点的平均位置，也就是这组点的中间。{calc}{moves}',
			vars: (_rt, at) => {
				const cnt = m(at, 'cnt'), [ox, oy, nx, ny] = ['ox', 'oy', 'nx', 'ny'].map((k) => xy(m(at, k))), stays = m(at, 'ox') === m(at, 'nx') && m(at, 'oy') === m(at, 'ny'), moved = m(at, 'moved');
				const calc = !cnt ? I('No point joined A this round, so A stays where it is.', '这一轮没有点归入 A，所以 A 留在原地。')
					: cnt === 1 ? (stays ? I(`A has one point and already sits on it, at (${nx}, ${ny}), so A stays.`, `A 组只有一个点，A 本来就在这个点上 (${nx}, ${ny})，所以 A 不动。`) : I(`A has one point, so A moves onto it, from (${ox}, ${oy}) to (${nx}, ${ny}).`, `A 组只有一个点，所以 A 移到这个点上：从 (${ox}, ${oy}) 移到 (${nx}, ${ny})。`))
					: stays ? I(`A has ${cnt} points, and their mean, (${nx}, ${ny}), is where A already is, so A stays.`, `A 组有 ${cnt} 个点，它们的平均位置 (${nx}, ${ny}) 正是 A 现在的位置，所以 A 不动。`)
					: I(`A has ${cnt} points: their x values average ${nx} and their y values ${ny}, so A moves from (${ox}, ${oy}) to (${nx}, ${ny}).`, `A 组有 ${cnt} 个点：它们 x 的平均是 ${nx}，y 的平均是 ${ny}，所以 A 从 (${ox}, ${oy}) 移到 (${nx}, ${ny})。`);
				return { calc, moves: moved > 0 ? I(`The dashed lines show each move; the longest is ${n3(moved)}.`, `虚线画出了每个质心的移动；最长的一段是 ${n3(moved)}。`) : I('No centroid moves this time.', '这一次没有质心移动。') };
			}
		},
		look: I('The dashed lines from where each centroid was to where it is now; the formula is A’s mean.', '虚线从每个质心原来的位置连到现在的位置；公式算的是 A 的平均位置。')
	},
	{
		id: 'km-repeat', source: 'km', at: { phase: 'done' }, view: canvas('km', { chart: true }), provenance: 'live',
		play: { from: { phase: 'init' }, to: { phase: 'done' }, phases: ['assign', 'update', 'done'], unit: I('round', '轮次') },
		title: I('Repeat until nothing changes', '重复，直到不再变化'),
		narration: {
			en: 'k-means now repeats the two steps, assign and update, until a whole round passes in which no point changes its centroid. The chart tracks one number: the total squared distance from every point to its own centroid, which k-means tries to make small. Guess first: how does that number move from step to step?',
			zh: 'k-means 接着反复做这两步：分配、更新，直到整整一轮没有一个点换质心。图表只追踪一个数：每个点到自己质心的距离的平方，全部加起来，k-means 想让它尽量小。先猜一猜：这个数每一步会怎样变化？'
		},
		look: I('The chart of the total squared distance after each step, and the colours changing round by round (the player counts rounds from 0).', '每一步之后总平方距离的图表，以及颜色一轮轮的变化（播放条上的轮次从 0 数起）。'),
		predict: {
			id: 'classic-ml-km-total',
			question: I('From one step to the next, what happens to the total squared distance from the points to their centroids?', '从一步到下一步，各点到自己质心的总平方距离会怎样？'),
			options: [I('It never goes up', '它从不上升'), I('It goes up and down', '它时升时降'), I('It falls all the way to 0', '它一直降到 0')],
			answer: 0,
			explain: I('Joining a nearer centroid can only shorten a point’s distance, and the mean is the spot with the smallest total squared distance to a group, so neither step can raise the total. It levels off when nothing changes; it is 0 only if every point sits on a centroid.', '换到更近的质心只会让点的距离变短，而平均位置正是到这组点的总平方距离最小的地方，所以两步都不会让总数变大。没有点再变化时它就停住；只有每个点都正好在质心上，它才会是 0。')
		}
	},
	{
		id: 'km-limits', source: 'rings', at: 'last', view: canvas('rings'), provenance: 'live', kindOf: 'limits',
		title: I('Limits: k must be given, and borders are straight', '局限：k 要事先给定，边界是直的'),
		narration: {
			en: 'k-means has two blind spots, shown here on a disc inside a ring. It needs k in advance and cannot tell which k is right: the total squared distance just keeps falling as k grows, from {i1} at k = 1 to {iK} at k = {kTop}. And the border between two centroids is a straight line (dashed), so with k = 2 it cuts both shapes in half: group A gets {a0} ring and {a1} disc points, group B {b0} and {b1}.',
			zh: 'k-means 有两个盲点，这里用“圆环里的一团点”来看。它要事先给定 k，自己却分辨不出哪个 k 才对：k 越大，总平方距离就一直往下降，从 k = 1 时的 {i1} 降到 k = {kTop} 时的 {iK}。而且两个质心之间的边界是一条直线（虚线），所以 k = 2 时它把圆环和圆盘都拦腰切成两半：A 组有 {a0} 个圆环点和 {a1} 个圆盘点，B 组有 {b0} 个和 {b1} 个。',
			vars: (_rt, at) => {
				const sw = frames(at, 'sweep').map((f) => f.metrics!), mix = frames(at, 'done')[0]?.metrics ?? {};
				return { i1: sw[0]?.inertia ?? NaN, iK: sw[sw.length - 1]?.inertia ?? NaN, kTop: sw.length || NaN, a0: mix.m0 ?? NaN, a1: mix.m1 ?? NaN, b0: mix.m2 ?? NaN, b1: mix.m3 ?? NaN };
			}
		},
		look: I('The dashed border: on each side of it one colour covers half of the ring and half of the disc. The bars show the total squared distance for each k.', '虚线边界：它两侧各有一种颜色，各占半个圆环和半个圆盘。横条显示每个 k 的总平方距离。')
	}
];

const lesson: LessonDef = {
	title: I('Three classic algorithms on your own points: kNN, a decision tree, k-means', '在你自己的点上看三种经典算法：kNN、决策树、k-means'),
	summary: I('Draw a dataset, then watch k nearest neighbours vote, a decision tree pick its questions by Gini impurity and k-means find groups, every number computed live on your points.', '画一个数据集，然后看 k 近邻投票、决策树用基尼不纯度挑问题、k-means 自己找出分组，每个数都在你的点上实时算出。'),
	steps,
	fixtures: [
		{ id: 'start', label: I('Noisy moons (the page’s start)', '有噪声的两个月牙（本页的起始输入）'), raw: START },
		{ id: 'moons', label: I('Two moons (little noise)', '两个月牙（噪声小）'), raw: { preset: { name: 'moons', n: 200, seed: 1 } } },
		{ id: 'circles', label: I('Circle in a ring', '环中圆'), raw: { preset: { name: 'circles', n: 200, seed: 1 }, kc: 2 } },
		{ id: 'xor', label: I('XOR quadrants, k-means with 4', '异或象限，k-means 取 4'), raw: { preset: { name: 'xor', n: 200, seed: 1 }, kc: 4, q: { x: 0.9, y: -0.9 } } },
		{ id: 'three', label: I('Three painted classes, k = 1, depth 2', '画的三类点，k = 1，深度 2'), raw: { points: [[-0.8, 0.6, 0], [-0.6, 0.8, 0], [-0.7, 0.5, 0], [0.7, 0.7, 1], [0.6, 0.5, 1], [0.8, 0.6, 1], [0, -0.7, 2], [0.1, -0.6, 2], [-0.1, -0.8, 2], [0.05, 0.1, 2]].map(([x, y, c]) => ({ x, y, c })), k: 1, depth: 2, kc: 6 } },
		{ id: 'twins', label: I('Two points on the same spot', '两个点在同一个位置'), raw: { points: [{ x: 0.2, y: 0.2, c: 0 }, { x: 0.2, y: 0.2, c: 1 }, { x: -0.5, y: 0.1, c: 1 }], k: 25, kc: 3 } }
	]
};

export default lesson;
