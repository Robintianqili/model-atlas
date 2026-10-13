// pointnet: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec and the weights,
// live values from the trace of the learner's points and from the same model on variants of them (cloud.ts), measured
// ones from facts.json ({fact:...}).
//   1 points → table (centre, scale) → 2 does order matter? (predict) → 3 the shared first layer, one point → 4 features
//   per point → 5 the max, one feature → 6 critical points (the critical rows alone give the same scores) → 7 remove
//   the critical points until the answer changes (task) → 8 classifier → 9 answer → 10 one stray point (limits).
import type { ExplainerRuntime, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3, paren } from '../../atlas/core/format.ts';
import { T, argmax, crit, label, maxDiff, variants } from './cloud.ts';
import { cloudOf, normalize } from './shapes.ts';
import type { CloudRaw } from './adapters/points2d/adapter.ts';

type RT = ExplainerRuntime;
/** structural numbers, read from the spec */
function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const n = Number(T(rt, 'input')?.shape[0] ?? NaN);
	return { n, h: Number(L('fc1')?.outShape[1] ?? NaN), F: Number(L('fc2')?.outShape[1] ?? NaN), h3: Number(L('fc3')?.outShape[0] ?? NaN), K: Array.isArray(rt.spec?.classes) ? rt.spec.classes.length : NaN };
}
const top = (rt: RT) => {
	const p = T(rt, 'softmax')?.data;
	return p ? argmax(p) : -1;
};
/** the point the lesson follows: the one that wins the most features of the max */
const strongest = (rt: RT) => crit(rt)?.strongest ?? 0;
const ROWS = { one: '{n} row', many: '{n} rows', zh: '{n} 行' };
const PTS = { one: '{n} point', many: '{n} points', zh: '{n} 个点' };
const cutOf = (rt: RT) => (rt.input as CloudRaw | null)?.cut;

const lesson: LessonDef = {
	title: { en: 'PointNet: a shape from loose points, in any order', zh: 'PointNet：从任意顺序的散点认出形状' },
	summary: {
		en: 'Your points become a table, every point goes through the same small network alone, a max keeps the strongest value of each feature, and a few critical points end up deciding the answer.',
		zh: '你的点变成一张数表，每个点单独经过同一个小网络，取最大值只保留每个特征最强的那个值，最后由少数几个关键点决定答案。'
	},
	steps: [
		{
			id: 'points',
			source: 'fwd',
			at: 'first',
			view: { key: 'pointnet/cloud', props: () => ({ mode: 'points' }) },
			provenance: 'live',
			title: { en: 'Your points become a table', zh: '你的点变成一张数表' },
			narration: {
				en: 'A point cloud is just a set of points. The model gets yours as a table of {rows} and two columns, x and y, in the order you placed them. First the page moves them so the middle of the box around them sits at 0, then scales them by {scale} {to}. A big and a small drawing of one shape now give similar numbers.',
				zh: '点云就是一组点。模型拿到的是一张 {rows}、2 列（x 和 y）的数表，行的顺序就是你放下这些点的顺序。页面先把它们平移，让包住它们的方框的中心落在 0，再乘以 {scale}，{to}。这样，同一个形状画大画小，得到的数都差不多。',
				vars: (rt) => {
					const m = normalize(cloudOf(rt.input).flatMap((q) => [q.x, q.y]));
					return {
						rows: m.n ? plural(m.n, ROWS) : NaN,
						scale: m.n ? m.scale : NaN,
						// all points on one spot (one point, or one point drawn again and again): nothing to scale
						to: pick(m.radius > 1e-9, { en: 'so the farthest point is at distance 1 from 0', zh: '让最远的点到 0 的距离为 1' }, { en: '(all your points sit on one spot, so there is no size to scale)', zh: '（你的点都在同一处，没有大小可缩放）' })
					};
				}
			},
			look: { en: 'The cross marks 0 and the dashed circle has radius 1: the farthest point touches it.', zh: '十字标出 0，虚线圆的半径是 1：最远的点正好碰到它。' },
			terms: ['point-cloud']
		},
		{
			id: 'order',
			source: 'fwd',
			at: 'first',
			view: { key: 'pointnet/cloud', props: () => ({ mode: 'order' }) },
			provenance: 'live',
			title: { en: 'Does the order of the points matter?', zh: '点的顺序重要吗？' },
			predict: {
				id: 'pointnet-order',
				question: { en: 'The network reads the shuffled table instead of yours. What happens to its answer?', zh: '网络改读打乱后的数表，而不是你原来的那张。它的答案会怎样？' },
				options: [
					{ en: 'It changes completely: the rows are different', zh: '完全改变：每一行都不同了' },
					{ en: 'It changes a little', zh: '变一点点' },
					{ en: 'It stays exactly the same', zh: '完全不变' }
				],
				answer: 2,
				explain: {
					en: 'Exactly the same, to the last digit. Each point goes through the network alone, and a later step keeps, for each of the network’s outputs, only its largest value over all points: a maximum does not care about order. A network that read all the numbers as one long list would see a new input.',
					zh: '完全一样，连最后一位都不差。每个点单独经过网络，后面有一步对网络的每个输出只保留它在所有点上的最大值：最大值不在乎顺序。如果网络把所有的数排成一长串来读，打乱之后它看到的就是一个新的输入。'
				}
			},
			narration: {
				en: 'A shape has no first point: you could have drawn the same outline starting anywhere, so the rows could come in any order. The page shuffles your table: {moved} of the {pts} land in another row. In the stage, a thin line joins the points in table order, once for your order and once for the shuffled one.',
				zh: '形状没有“第一个点”：同一个轮廓你可以从任何地方开始画，所以这些行可以按任何顺序排列。页面把你的数表打乱：{pts}里有 {moved} 个换到了别的行。画面里，一条细线按表中的顺序把点连起来：一次按你的顺序，一次按打乱后的顺序。',
				vars: (rt) => {
					const v = variants(rt);
					return { pts: plural(dims(rt).n, PTS), moved: v ? v.order.filter((p, i) => p !== i).length : NaN };
				}
			},
			look: { en: 'Compare the answers under the two drawings.', zh: '比较两幅图下面的答案。' },
			terms: ['permutation-invariance']
		},
		{
			id: 'shared',
			source: 'fwd',
			at: { layer: 'fc1' },
			select: (rt) => {
				const r = T(rt, 'fc1');
				const i = strongest(rt);
				if (!r) return { layer: 'fc1', unit: [0, 0] };
				const H = r.shape[1];
				return { layer: 'fc1', unit: [i, argmax(r.data.subarray(i * H, (i + 1) * H))] };
			},
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'Every point through the same small network', zh: '每个点都过同一个小网络' },
			narration: {
				en: 'Each point now goes alone through a small MLP whose weights are shared by all points. Its first layer turns x and y into {h} numbers, each a weighted sum w₁·x + w₂·y + b. For point {i}, number {f} (both counting from 0) is {sum} = {v}. Every other point uses the same three numbers, so this layer has {pw} weights and biases however many points you draw.',
				zh: '现在每个点单独经过一个小 MLP，所有点共用它的权重。第一层把 x 和 y 变成 {h} 个数，每个都是加权和 w₁·x + w₂·y + b。对点 {i}，编号 {f} 的数（都从 0 数起）是 {sum} = {v}。其他每个点都用这同样的三个数，所以不管你画多少个点，这一层都只有 {pw} 个权重和偏置。',
				vars: (rt) => {
					const u = rt.sel.unit ?? [];
					const W = rt.model?.weights['fc1.weight']?.data, B = rt.model?.weights['fc1.bias']?.data;
					const x = T(rt, 'input')?.data, r = T(rt, 'fc1');
					const [i, f] = u;
					const d = dims(rt);
					const sum = W && B && x ? `${fmt3(W[2 * f])} × ${paren(x[2 * i])} + ${paren(W[2 * f + 1])} × ${paren(x[2 * i + 1])} + ${paren(B[f])}` : NaN;
					return { h: d.h, i, f, sum, pw: 3 * d.h, v: r ? r.data[i * d.h + f] : NaN };
				}
			},
			look: { en: 'Two products and a bias: that is all one point contributes to this number.', zh: '两个乘积加一个偏置：这就是一个点对这个数的全部贡献。' },
			terms: ['shared-mlp', 'mlp', 'weight', 'bias']
		},
		{
			id: 'features',
			source: 'fwd',
			at: { layer: 'relu2' },
			select: (rt) => ({ layer: 'relu2', unit: [strongest(rt), 0] }),
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'A row of features for every point', zh: '每个点一行特征' },
			narration: {
				en: 'A ReLU keeps the positive numbers and turns the rest into 0. A second shared layer and another ReLU give every point {F} numbers, its features, so the table is now {rows} of {F}; {layer.relu2.zeroShare|pct} of its numbers are 0. Still no row has looked at any other: a row gives the same features in any table that contains it.',
				zh: 'ReLU 保留正数，把其余的数变成 0。再经过一个共享层和一个 ReLU，每个点得到 {F} 个数，也就是它的特征，于是数表变成 {rows}、每行 {F} 个数；其中 {layer.relu2.zeroShare|pct} 是 0。到这里，还没有哪一行看过别的行：同一行放在任何数表里，得到的特征都一样。',
				vars: (rt) => ({ ...dims(rt), rows: plural(dims(rt).n, ROWS) })
			},
			look: { en: 'Each row is one point and each column one feature; the zeros are the uncoloured cells.', zh: '每一行是一个点，每一列是一个特征；为 0 的格子没有颜色。' },
			terms: ['relu']
		},
		{
			id: 'max',
			source: 'fwd',
			at: { layer: 'pool' },
			view: { key: 'pointnet/cloud', props: (rt) => ({ mode: 'feature', f: bestFeature(rt) }) },
			provenance: 'live',
			title: { en: 'Keep the largest value of each feature', zh: '每个特征只留最大值' },
			narration: {
				en: 'Now the table is squeezed into one row: for each of the {F} features, keep only the largest value over all {pts}. This max pooling is the step that ignores order. In the stage, colour shows feature {f} (counting from 0) at every point; its largest value, {m}, comes from point {i}. This row of {F} numbers is all the classifier will see of your shape.',
				zh: '现在把整张表压成一行：对 {F} 个特征中的每一个，只保留它在全部 {pts}上的最大值。这一步叫最大池化，正是它让点的顺序变得无关紧要。画面里，颜色表示每个点上特征 {f}（从 0 数起）的值；它的最大值 {m} 来自点 {i}。分类器对你的形状所知道的，就只有这一行 {F} 个数。',
				vars: (rt) => {
					const c = crit(rt);
					const f = bestFeature(rt);
					return { ...dims(rt), pts: plural(dims(rt).n, PTS), f, m: c ? c.max[f] : NaN, i: c ? c.win[f] : NaN };
				}
			},
			look: { en: 'Step through the features with ◀ ▶ and watch where the ringed winner lands.', zh: '用 ◀ ▶ 逐个查看特征，看加圈的赢家落在哪里。' },
			terms: ['pooling']
		},
		{
			id: 'critical',
			source: 'fwd',
			at: { layer: 'pool' },
			view: { key: 'pointnet/cloud', props: () => ({ mode: 'critical' }) },
			provenance: 'live',
			title: { en: 'The critical points', zh: '关键点' },
			narration: {
				en: '{liveSay}; {deadSay}. The critical points are the ones that win something: {k} of your {pts}. Feed the network only their {krows} and its {K} probabilities do not change at all: the largest difference is {diff}.',
				zh: '{liveSay}；{deadSay}。赢得了特征的点就是关键点：你的 {pts}里有 {k} 个。只把这 {krows}喂给网络，{K} 个概率一点都不变：最大差是 {diff}。',
				vars: (rt) => {
					const c = crit(rt);
					const v = variants(rt);
					// verb agreement at 1 and a clause for no dead feature (both read right however many there are)
					const liveSay = c ? plural(c.F - c.dead, { one: `{n} of the ${c.F} features is above 0 somewhere and is won by one point`, many: `{n} of the ${c.F} features are above 0 somewhere, and each is won by one point`, zh: `${c.F} 个特征中有 {n} 个在某处大于 0，每个都由一个点赢得` }) : NaN;
					const deadSay = !c ? NaN : c.dead === 0 ? { en: 'every feature has a winner', zh: '每个特征都有赢家' } : plural(c.dead, { one: 'the other {n} is 0 at every point and has no winner', many: 'the other {n} are 0 at every point and have no winner', zh: '其余 {n} 个在每个点上都是 0，没有赢家' });
					return { ...dims(rt), pts: plural(dims(rt).n, PTS), krows: c ? plural(c.pts.length, ROWS) : NaN, liveSay, deadSay, k: c?.pts.length ?? NaN, diff: maxDiff(T(rt, 'softmax')?.data, v?.only?.p) };
				}
			},
			look: { en: 'The large dots are the critical points; the larger, the more features a point wins.', zh: '大圆点是关键点；越大，赢的特征越多。' },
			terms: ['critical-point']
		},
		{
			id: 'cut',
			source: 'fwd',
			at: { layer: 'pool' },
			view: { key: 'pointnet/cloud', props: () => ({ mode: 'critical', task: true }) },
			provenance: 'live',
			title: { en: 'Remove critical points until the answer changes', zh: '删掉关键点，直到答案改变' },
			narration: {
				en: 'Erasing a point that is not critical leaves every maximum where it was, as long as the centring and scaling of your points stay the same. Erasing the critical points leaves no hole: each feature passes to the next-best point, so new critical points take over and the answer often holds. On the test shapes, removing a random half of the points changed the answer only {fact:halfFlip|pct} of the time. Now it says {now}.',
				zh: '擦掉一个不是关键点的点，只要点的平移和缩放不变，每个最大值都原封不动。擦掉关键点也不会留下空缺：每个特征的最大值改由次大的那个点给出，于是新的关键点接替上来，答案常常保持不变。在测试形状上，随机去掉一半的点，答案只在 {fact:halfFlip|pct} 的情况下改变。现在它的答案是{now}。',
				vars: (rt) => ({ now: top(rt) >= 0 ? label(top(rt)) : NaN })
			},
			task: {
				prompt: { en: 'Press “Remove the critical points” in the stage until the answer changes. Each press erases the critical points shown; then the page centres and scales the rest again.', zh: '在画面里按“删掉这些关键点”，直到答案改变。每按一次就擦掉画面里的关键点，然后页面把剩下的点重新居中、缩放。' },
				done: (rt) => {
					const c = cutOf(rt);
					return !!c && top(rt) >= 0 && top(rt) !== c.cls;
				},
				waiting: { en: 'Not yet: the answer is still the same.', zh: '还没有：答案还是原来那个。' },
				doneText: { en: 'The answer changed. “Put them back” restores your points.', zh: '答案变了。按“放回去”可以恢复你的点。' }
			},
			look: { en: 'After each press: the new critical points, and the answer under the drawing.', zh: '每按一次之后：新的关键点，以及图下方的答案。' },
			terms: ['critical-point']
		},
		{
			id: 'head',
			source: 'fwd',
			at: { layer: 'fc4' },
			select: (rt) => ({ layer: 'fc4', unit: [Math.max(0, top(rt))] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'From one row to five scores', zh: '从一行数到五个分数' },
			narration: {
				en: 'A second small MLP, the classifier, reads the {F} pooled numbers: {h3} weighted sums with a ReLU, then one score per shape. The score of {lab} is the sum of {h3} products plus a bias: {s}, the largest of the {K} scores.',
				zh: '第二个小 MLP，也就是分类器，读取池化后的 {F} 个数：先是 {h3} 个加权和再加 ReLU，然后每个形状一个分数。{lab}的分数是 {h3} 个乘积之和再加偏置：{s}，是 {K} 个分数中最大的。',
				vars: (rt) => {
					const t = top(rt);
					return { ...dims(rt), lab: t >= 0 ? label(t) : NaN, s: t >= 0 ? (T(rt, 'fc4')?.data[t] ?? NaN) : NaN };
				}
			},
			look: { en: 'The products with the largest bars decide the score.', zh: '条最长的那些乘积决定了这个分数。' },
			terms: ['classifier', 'logit']
		},
		{
			id: 'output',
			source: 'fwd',
			at: { layer: 'softmax' },
			view: 'kit.probbars',
			provenance: 'live',
			title: { en: 'The answer', zh: '答案' },
			narration: {
				en: 'Softmax turns the {K} scores into shares that add up to 1: {top.label} gets {top.p|pct}, then {runnerUp.label} with {runnerUp.p|pct}. On {fact:testN} test shapes the network was right {fact:testAcc|pct} of the time. It can only answer with one of these {K} shapes, so a heart or a letter still gets one of them.',
				zh: 'Softmax 把 {K} 个分数变成加起来为 1 的份额：{top.label}得到 {top.p|pct}，其次是{runnerUp.label}，{runnerUp.p|pct}。在 {fact:testN} 个测试形状上，网络答对了 {fact:testAcc|pct}。它只能从这 {K} 个形状里选一个回答，所以一颗心或一个字母也会被归入其中之一。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The longest bar is the answer.', zh: '最长的那根条就是答案。' },
			terms: ['softmax']
		},
		{
			id: 'stray',
			source: 'fwd',
			at: { layer: 'softmax' },
			view: { key: 'pointnet/cloud', props: () => ({ mode: 'stray' }) },
			provenance: 'live',
			kindOf: 'breakIt',
			title: { en: 'One stray point', zh: '一个离群点' },
			narration: {
				en: 'Removing points at random rarely changes PointNet’s answer, but adding one can. Put one stray point {fact:stray} radii to the right of the centre: it becomes the right edge of the box, so the centre moves toward it, your shape shrinks to {shrink|pct} of its size and the maxima shift. The answer {verb} {lab} ({p|pct}). On the test shapes, such a point changed the answer {fact:strayFlip|pct} of the time.',
				zh: '随机去掉一些点，很少会改变 PointNet 的答案；加一个点却可能改变。在中心右边 {fact:stray} 个半径处放一个离群点：它成了方框的右边缘，于是中心向它移动，你的形状缩小到原来的 {shrink|pct}，那些最大值也跟着变。答案{verb}{lab}（{p|pct}）。在测试形状上，这样一个点有 {fact:strayFlip|pct} 的时候改变了答案。',
				vars: (rt) => {
					const v = variants(rt);
					const s = v?.stray;
					return {
						shrink: v ? v.scale : NaN,
						verb: s ? pick(s.top !== top(rt), { en: 'changes to', zh: '变成了' }, { en: 'stays', zh: '仍然是' }) : NaN,
						lab: s ? label(s.top) : NaN,
						p: s ? s.p[s.top] : NaN
					};
				}
			},
			look: { en: 'Compare the size of the shape and the answer in the two drawings.', zh: '比较两幅图里形状的大小和答案。' },
		}
	]
};

/** the feature the max step shows: the one whose winner wins the most features (the strongest point's first live one) */
function bestFeature(rt: RT) {
	const c = crit(rt);
	if (!c) return 0;
	const f = c.win.indexOf(c.strongest);
	return f >= 0 ? f : 0;
}

export default lesson;
