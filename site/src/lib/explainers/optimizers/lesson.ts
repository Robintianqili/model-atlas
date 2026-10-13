// optimizers: the guided lesson (eleven steps; the concept page's primary path). Every number is live: the race runs the
// kit's optimizers on the surfaces of data.ts, the range test and the schedules train the 2-8-8-2 network of net.ts.
//   race -> sgd -> momentum (predict) -> momentum-update -> adam -> yours (task) -> league (limits) -> range -> schedule
//   (task) -> warmup -> clip (breakIt, task)
import type { At, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { num, pick } from '../../atlas/lesson/say.ts';
import { CLIFF, RACERS, RANGE, SCHED, SCHEDULE_NAMES, START, RACE_SURFACES, distinct, leagueKey, metricsOf, setting, setupOf, surfaceOf, type RacerKey, type SurfaceId } from './data.ts';
import { STARTS } from './adapters/setup/adapter.ts';
import { MOONS, PARAMS } from './net.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const lrOf = (k: RacerKey) => setting(RACERS.find((r) => r.key === k)!.lr);
const valley = surfaceOf('valley');
/** the metrics of frame i of the step's own tape (NaN-filled when it is not recorded yet) */
const frameM = (at: At, i: number): Record<string, number> => (at.tape && i >= 0 && i < at.tape.length ? (at.tape.frame(i).metrics ?? {}) : {});
/** a racer's name after Chinese text: a Latin name (SGD, Adam) gets a space before it, as everywhere else in the zh */
const after = (n: I18n): I18n => ({ en: n.en, zh: /^[A-Za-z]/.test(n.zh ?? '') ? ` ${n.zh}` : n.zh });
/** the racer with the lowest loss at a race frame */
const winner = (m: Record<string, number>): I18n => {
	let best = RACERS[0];
	for (const r of RACERS) if (m[`${r.key}Loss`] < m[`${best.key}Loss`]) best = r;
	return after(best.name);
};
const schedName = (raw: unknown): I18n => {
	const s = setupOf(raw);
	const n = SCHEDULE_NAMES[s.schedule];
	return s.warmup ? L(`${n.en} with warm-up`, `带预热的${n.zh}`) : L(`${n.en} without warm-up`, `不带预热的${n.zh}`);
};
/** the warm-up verdict: how many times lower the loss ends than the cold start's */
const times = (r: number): I18n => {
	const v = num(r);
	if (typeof v === 'number') return L('…', '…');
	return L(`Starting small got it through: its loss ends ${v.en} times lower.`, `从小步起步让它挺了过来：最后的损失低了 ${v.zh} 倍。`);
};
const surfaceView = (props: Record<string, unknown>) => ({ key: 'optimizers/surface', props: () => props });

const lesson = defineLesson({
	title: L('Optimizers, learning-rate schedules and warm-up, step by step', '优化器、学习率调度与预热，分步课程'),
	summary: L(
		'Four optimizers race down a long valley; work out one update of SGD, momentum and Adam with the real numbers, race on your own landscape, see that no optimizer wins everywhere, then run a learning-rate range test, rescue a too-large rate with warm-up and stop an exploding gradient with clipping.',
		'四个优化器沿狭长山谷赛跑；用真实的数值算出 SGD、动量法和 Adam 的一次更新，在你自己的地形上比赛，看到没有哪个优化器处处都赢；然后做学习率区间测试，用预热救下过大的学习率，再用梯度裁剪拦住梯度爆炸。'
	),
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'race',
			title: L('Four optimizers, one valley', '四个优化器，同一个山谷'),
			source: 'valley',
			at: 'last',
			view: surfaceView({}),
			play: { from: 'first', to: 'last', interval: 120, unit: L('update', '次更新') },
			provenance: 'live',
			terms: ['optimizer', 'learning-rate', 'momentum', 'adam', 'adamw'],
			narration: {
				en: 'Four optimizers start from the same point of a long, narrow valley, each with its own learning rate η, and take {n} updates. They end at a loss of {sgd} for SGD, {mom} for momentum, {adam} for Adam and {adamw} for AdamW; the bottom is 0. The gradients are the same for all four: only the rule that turns them into steps differs.',
				zh: '四个优化器从狭长山谷中的同一个点出发，各用自己的学习率 η，各更新 {n} 次。最后的损失：SGD 是 {sgd}，动量法是 {mom}，Adam 是 {adam}，AdamW 是 {adamw}；谷底是 0。四者面对的梯度完全一样，不同的只是把梯度变成步子的规则。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { n: m.step, sgd: m.sgdLoss, mom: m.momentumLoss, adam: m.adamLoss, adamw: m.adamwLoss };
				}
			},
			look: L('Follow each colored path from the ring at the start; the cross marks the bottom.', '从起点的圆环出发，沿每条彩色路径看；叉号标出谷底。')
		},
		{
			id: 'sgd',
			title: L('SGD: one learning rate for a steep and a flat direction', 'SGD：陡和平两个方向共用一个学习率'),
			source: 'valley',
			at: { phase: 'step', step: 10 },
			view: surfaceView({ show: ['sgd'] }),
			provenance: 'live',
			terms: ['sgd', 'gradient'],
			narration: {
				en: 'SGD, here plain gradient descent (there are no batches to sample), moves each weight by −η × its slope. At the start the slope across the valley (y) is {gy}, {ratio} times steeper than along it (x), where it is {gx}. With η = {lr} the first update takes y from {y} to {y1}, past the floor, and x only from {x} to {x1}. After {n} updates x has crawled to {xn}; the bottom is at x = {xb}.',
				zh: 'SGD 在这里就是普通的梯度下降（没有要抽样的小批次）：每个权重移动 −η × 它的坡度。起点处横跨山谷（y 方向）的坡度是 {gy}，比沿山谷（x 方向）的坡度 {gx} 陡 {ratio} 倍。η = {lr} 时，第一次更新让 y 从 {y} 变成 {y1}，越过了谷底；x 只从 {x} 变成 {x1}。更新 {n} 次之后，x 才慢慢挪到 {xn}；谷底在 x = {xb}。',
				vars: (_rt, at) => {
					const [gx, gy] = valley.grad(valley.start[0], valley.start[1]);
					const f1 = frameM(at, 1);
					const m = metricsOf(at);
					return { gx, gy, ratio: Math.abs(gy / gx), lr: lrOf('sgd'), x: valley.start[0], y: valley.start[1], x1: f1.sgdX, y1: f1.sgdY, n: m.step, xn: m.sgdX, xb: valley.min[0][0] };
				}
			},
			look: L('The path flips from one side of the valley to the other while it creeps along it.', '路径在山谷两侧来回翻转，同时沿山谷慢慢爬行。')
		},
		{
			id: 'momentum',
			title: L('Momentum: a running sum of gradients', '动量法：梯度的滑动累加'),
			source: 'valley',
			at: { phase: 'step', step: 20 },
			view: surfaceView({ show: ['sgd', 'momentum'], glyph: 'momentum' }),
			provenance: 'live',
			terms: ['momentum'],
			predict: {
				id: 'optimizers-momentum',
				question: L('After 20 updates, where is momentum compared with SGD?', '更新 20 次之后，动量法和 SGD 相比在哪里？'),
				options: [
					L('Further along the valley, but still swinging across it', '沿山谷走得更远，但仍在山谷两侧来回摆动'),
					L('About as far as SGD, on the same path', '和 SGD 差不多远，路径也一样'),
					L('Behind SGD: the old gradients hold it back', '落在 SGD 后面：旧的梯度拖住了它')
				],
				answer: 0,
				explain: L(
					'Along the valley every gradient points the same way, so the velocity piles them up and grows to many gradients’ worth. Across the valley the gradient changes sign, so the old velocity and the new gradient partly cancel; but the velocity carries the weights past the floor a few times, like a ball rolling side to side, before the swing dies down.',
					'沿着山谷，每个梯度都指向同一个方向，所以速度把它们累加起来，变成好几个梯度那么大。横跨山谷时梯度会变号，旧的速度和新的梯度部分抵消；但速度还是会带着权重几次越过谷底，像一个左右滚动的球，摆动要过一阵才平息。'
				)
			},
			narration: {
				en: 'Momentum keeps a velocity v: each update v ← μ·v + g with μ = {mu}, then the weights move by −η·v with η = {lr}. Along the valley the gradients all point the same way; across it, they change sign each time y crosses the floor. Answer first, then compare the two paths.',
				zh: '动量法保留一个速度 v：每次更新先算 v ← μ·v + g（μ = {mu}），再让权重移动 −η·v（η = {lr}）。沿着山谷，梯度都指向同一个方向；横跨山谷时，y 每越过一次谷底，梯度就变一次号。先回答，再比较两条路径。',
				vars: () => ({ mu: setting(0.9), lr: lrOf('momentum') })
			},
			look: L('The solid arrow is momentum’s last step; the dashed one is the step its gradient alone would take.', '实线箭头是动量法的上一步；虚线箭头是只用它的梯度时会走的一步。')
		},
		{
			id: 'momentum-update',
			title: L('One momentum update, with the real numbers', '动量法的一次更新，代入真实数值'),
			source: 'valleyFine',
			at: { phase: 'momentum', step: 9 },
			view: surfaceView({ show: ['momentum'], glyph: 'momentum', formula: true }),
			provenance: 'live',
			narration: {
				en: 'This is update {k} (counting from 0), with μ = {mu}. Along the valley the new slope {gx} {sx} the velocity {bx}, so μ·v + g = {bx1}, larger in size than the old velocity: {ratio} times the slope. Across it the slope {gy} {sy} the velocity {by}, so the two partly cancel and μ·v + g is only {by1}. The weights then move by −η·v.',
				zh: '这是编号 {k} 的更新（从 0 数起），μ = {mu}。沿着山谷，新的坡度 {gx} 和速度 {bx} {sx}，所以 μ·v + g = {bx1}，绝对值比原来的速度还大：是坡度的 {ratio} 倍。横跨山谷时，坡度 {gy} 和速度 {by} {sy}，两者部分抵消，μ·v + g 只有 {by1}。然后权重移动 −η·v。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const same = L('has the same sign as', '同号');
					const opp = L('has the opposite sign to', '异号');
					return { k: m.step, mu: setting(m.mu), gx: m.gx, bx: m.bx, bx1: m.bx1, ratio: m.bx1 / m.gx, gy: m.gy, by: m.by, by1: m.by1, sx: pick(m.gx * m.bx > 0, same, opp), sy: pick(m.gy * m.by > 0, same, opp) };
				}
			},
			look: L('Read the formula row by row: first the velocity, then the weights.', '逐行读公式：先算速度，再算权重。')
		},
		{
			id: 'adam',
			title: L('Adam: every weight gets its own step size', 'Adam：每个权重都有自己的步长'),
			source: 'valleyFine',
			at: { phase: 'adam', step: 0 },
			view: surfaceView({ show: ['adam'], glyph: 'adam', formula: true }),
			provenance: 'live',
			terms: ['adam'],
			narration: {
				en: 'Adam keeps two running averages per weight: m of the gradients and v of their squares; m̂ and v̂ correct them for starting at 0. Each weight moves by η·m̂ / √v̂, so a steep weight is divided by a big number and a flat one by a small one. On this first update the slope across, {gy}, is {ratio} times the slope along, {gx}, yet x moves by {dx} and y by {dy}: η = {lr} each, against the sign of its slope.',
				zh: 'Adam 为每个权重保留两个滑动平均：梯度的 m 和梯度平方的 v；m̂ 和 v̂ 修正了它们从 0 起步带来的偏差。每个权重移动 η·m̂ / √v̂，所以陡的权重除以一个大数，平的权重除以一个小数。在这第一次更新中，横跨方向的坡度 {gy} 是沿山谷方向 {gx} 的 {ratio} 倍，但 x 移动了 {dx}，y 移动了 {dy}：各移动 η = {lr}，方向与各自坡度的符号相反。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { gy: m.gy, gx: m.gx, ratio: Math.abs(m.gy / m.gx), dx: m.dx, dy: m.dy, lr: setting(m.lr) };
				}
			},
			look: L('The two squares have sides √v̂ of x and of y: Adam divides each weight’s step by its own side.', '两个方块的边长分别是 x 和 y 的 √v̂：Adam 用各自的边长去除每个权重的步子。')
		},
		{
			id: 'yours',
			title: L('Your landscape', '你的地形'),
			source: 'race',
			at: 'last',
			view: surfaceView({}),
			play: { from: 'first', to: 'last', interval: 120, unit: L('update', '次更新') },
			provenance: 'live',
			terms: ['saddle-point'],
			task: {
				prompt: L('Pick another landscape or move the start in the side panel.', '在侧栏里换一个地形，或移动起点。'),
				done: (rt) => {
					const s = setupOf(rt.input);
					return s.surface !== START.surface || s.x !== START.x || s.y !== START.y;
				},
				hint: L('Try the saddle with the start exactly on y = 0, or the bumpy bowl.', '试试把起点正好放在 y = 0 的鞍点，或者坑洼的碗。'),
				waiting: L('Not yet: the race still runs on the long valley from its first start.', '还没有：比赛还在狭长山谷的原始起点上进行。'),
				doneText: L('Done: the race now runs on your landscape.', '完成：比赛现在在你的地形上进行。')
			},
			narration: {
				en: 'This race runs on the {land} from x = {x}, y = {y}. After {n} updates the losses are {sgd} for SGD, {mom} for momentum, {adam} for Adam and {adamw} for AdamW: {best} ends lowest. {note}',
				zh: '这场比赛在{land}上进行，起点 x = {x}，y = {y}。更新 {n} 次之后，损失分别是：SGD {sgd}，动量法 {mom}，Adam {adam}，AdamW {adamw}；最低的是{best}。{note}',
				vars: (rt, at) => {
					const m = metricsOf(at);
					const s = setupOf(rt.input);
					const stuck = s.surface === 'saddle' && RACERS.every((r) => m[`${r.key}Y`] === 0);
					const [sgd, mom, adam, adamw] = distinct(RACERS.map((r) => m[`${r.key}Loss`]));
					return {
						land: surfaceOf(s.surface).name,
						x: s.x,
						y: s.y,
						n: m.step,
						sgd,
						mom,
						adam,
						adamw,
						best: winner(m),
						note: pick(
							stuck,
							L('None of them left the ridge y = 0: the slope across is exactly 0 there, so nothing pushes them off the saddle.', '它们都没有离开 y = 0 这条脊线：那里横跨方向的坡度正好是 0，没有任何力量把它们推离鞍点。'),
							L('AdamW is Adam that also shrinks every weight a little towards 0 each update, separately from the gradient.', 'AdamW 就是 Adam，只是每次更新还把每个权重向 0 收缩一点，这和梯度无关。')
						)
					};
				}
			},
			look: L('Change the landscape or the start: all four racers restart from it.', '换一个地形或起点：四个选手都会从那里重新出发。')
		},
		{
			id: 'league',
			title: L('No optimizer wins everywhere', '没有哪个优化器处处都赢'),
			source: 'league',
			at: 'last',
			view: 'optimizers/league',
			provenance: 'live',
			kindOf: 'limits',
			narration: {
				en: 'The same four optimizers, each from its landscape’s own start, {n} updates each. On the long valley {v} ends lowest, at {vl}. On the saddle {s} does, at {sl}. On the bumpy bowl {b} ends lowest, at {bl}: its velocity carries it over dips that stop the others, so which optimizer is best depends on the landscape and on its settings.',
				zh: '同样四个优化器，各从每个地形自己的起点出发，各更新 {n} 次。在狭长山谷上，最低的是{v}，损失 {vl}。在鞍点上，最低的是{s}，损失 {sl}。在坑洼的碗里，最低的是{b}，损失 {bl}：它的速度带着它越过了困住其他选手的小坑。所以哪个优化器最好，取决于地形和它的设置。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const row = (s: SurfaceId) => {
						const best = RACERS.reduce((a, r) => (m[leagueKey(s, r.key)] < m[leagueKey(s, a.key)] ? r : a));
						return { name: after(best.name), loss: m[leagueKey(s, best.key)] };
					};
					const [v, s, b] = RACE_SURFACES.map(row);
					return { n: m.step, v: v.name, vl: v.loss, s: s.name, sl: s.loss, b: b.name, bl: b.loss };
				}
			},
			look: L('Each row’s lowest loss is outlined and marked “lowest”.', '每一行最低的损失有边框，并标着“最低”。')
		},
		{
			id: 'range',
			title: L('Find a learning rate: the range test', '找一个学习率：区间测试'),
			source: 'range',
			at: 'last',
			view: 'optimizers/range',
			provenance: 'live',
			terms: ['lr-range-test'],
			narration: {
				en: 'Now a real network: {params} weights sorting {pts} points, trained once with SGD while η grows from {lo} to {hi}, a little every step. The loss falls fastest at η = {steep}, is lowest, {low}, at η = {lowLr}, and climbs back above its start at η = {blow}. A common reading picks a rate on the falling stretch, between {steep} and {lowLr}.',
				zh: '现在换成一个真实的网络：{params} 个权重，给 {pts} 个点分类。用 SGD 训练一次，同时让 η 每一步都调大一点，从 {lo} 一直到 {hi}。损失在 η = {steep} 时下降最快，在 η = {lowLr} 时最低（{low}），到 η = {blow} 时又回升到起始值以上。一种常见的读法是在下降的那一段里挑一个学习率，也就是 {steep} 到 {lowLr} 之间。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { params: PARAMS, pts: MOONS.n, lo: setting(RANGE.lo), hi: setting(RANGE.hi), steep: m.steepLr, low: m.lowLoss, lowLr: m.lowLr, blow: m.blowLr };
				}
			},
			look: L('Both axes are logarithmic: each labelled line along η is ten times the one before it.', '两个轴都是对数刻度：η 轴上每条标了数的竖线都是前一条的十倍。')
		},
		{
			id: 'schedule',
			title: L('A schedule sets η for every step', '调度为每一步设定 η'),
			source: 'sched',
			// the last training step (the 'end' frame after it holds only the final losses, so the curve's readout would be empty)
			at: { phase: 'step', step: SCHED.steps - 1 },
			view: { key: 'loss-curve', props: (rt) => ({ metrics: ['lr', 'lrBase'], log: false, labels: { lr: schedName(rt.input), lrBase: L(`constant η = ${SCHED.peak}`, `恒定 η = ${SCHED.peak}`) } }) },
			provenance: 'live',
			terms: ['lr-schedule', 'warm-up'],
			task: {
				prompt: L('Turn warm-up on in the side panel.', '在侧栏里打开预热。'),
				done: (rt) => setupOf(rt.input).warmup,
				hint: L('Any schedule works; warm-up only changes the first 20 steps.', '任何调度都可以；预热只改变前 20 步。'),
				waiting: L('Not yet: warm-up is off.', '还没有：预热是关着的。'),
				doneText: L('Done: the first 20 steps now climb to the peak.', '完成：前 20 步现在会逐步升到峰值。')
			},
			narration: {
				en: 'Next, the same network at a peak η of {peak}, just below where the range test blew up. Your schedule, {name}, sets η to {lr0} at step 0, {lrMid} at step {mid} and {lrEnd} at the last step. Warm-up climbs to the peak over the first {warm} steps; a decay schedule lowers η as training goes on.',
				zh: '接下来，同一个网络以 η = {peak} 为峰值，刚好低于区间测试开始崩溃的地方。你的调度是{name}：第 0 步 η = {lr0}，第 {mid} 步 η = {lrMid}，最后一步 η = {lrEnd}。预热在前 {warm} 步内逐步升到峰值；衰减调度则随着训练进行降低 η。',
				vars: (rt, at) => {
					const mid = SCHED.steps / 2;
					return { peak: setting(SCHED.peak), name: schedName(rt.input), lr0: frameM(at, 0).lr, mid, lrMid: frameM(at, mid).lr, lrEnd: frameM(at, SCHED.steps - 1).lr, warm: SCHED.warm };
				}
			},
			look: L('The two lines are η at every training step: yours and the constant one.', '两条线是每个训练步的 η：你的调度和恒定的那条。')
		},
		{
			id: 'warmup',
			title: L('Warm-up rescues a rate that breaks a cold start', '预热救下一个从冷启动就会崩溃的学习率'),
			source: 'sched',
			at: 'last',
			view: { key: 'loss-curve', props: (rt) => ({ metrics: ['loss', 'lossBase'], log: true, labels: { loss: schedName(rt.input), lossBase: L(`constant η = ${SCHED.peak}`, `恒定 η = ${SCHED.peak}`) } }) },
			provenance: 'live',
			narration: {
				en: 'The range test came close to η = {peak} only after many smaller steps; from a cold start at a constant {peak} the loss ends at {base|n4} (a coin-flip guess scores ln 2 = {coin|n4}). Your schedule, {name}, ends at {mine|n4}. {verdict}',
				zh: '区间测试是先走了很多更小的步子，才接近 η = {peak} 的；如果从冷启动开始就用恒定的 {peak}，损失最后是 {base|n4}（像抛硬币一样瞎猜的得分是 ln 2 = {coin|n4}）。你的调度（{name}）最后是 {mine|n4}。{verdict}',
				vars: (rt, at) => {
					const m = metricsOf(at);
					const s = setupOf(rt.input);
					return {
						peak: setting(SCHED.peak),
						base: m.lossBase,
						coin: Math.LN2,
						name: schedName(rt.input),
						mine: m.loss,
						verdict:
							s.warmup && m.loss < m.lossBase / 10
								? times(m.lossBase / m.loss)
								: s.warmup
									? L('Here even the warm-up does not save it.', '在这里，即使有预热也没能救下它。')
									: L('Without warm-up the first steps already use the full rate: turn warm-up on and compare.', '没有预热，第一步就用上了完整的学习率：打开预热再比较一下。')
					};
				}
			},
			look: L('The loss axis is logarithmic; compare where the two lines end.', '损失轴是对数刻度；比较两条线的终点。')
		},
		{
			id: 'clip',
			title: L('A cliff, and gradient clipping', '悬崖与梯度裁剪'),
			source: 'cliff',
			at: 'last',
			view: surfaceView({}),
			provenance: 'live',
			kindOf: 'breakIt',
			terms: ['gradient-clipping'],
			task: {
				prompt: L('Turn gradient clipping on in the side panel.', '在侧栏里打开梯度裁剪。'),
				done: (rt) => setupOf(rt.input).clip,
				waiting: L('Not yet: clipping is off.', '还没有：裁剪是关着的。'),
				doneText: L('Done: your run now clips its gradients.', '完成：你的运行现在会裁剪梯度。')
			},
			narration: {
				en: 'Here the loss comes from h, a number multiplied by w {T} times (h ← w·h + b), so past w = 1 it rises like a cliff. Without clipping, update {k} (counting from 0) lands at w = {w} where the gradient is {norm} long, and the next step throws the loss to {big}. Clipping shrinks any gradient longer than {c} to length {c}, keeping its direction, so each step is at most {maxStep} long and the loss gets as low as {low}. {mine}',
				zh: '这里的损失来自 h，一个被 w 连乘 {T} 次的数（h ← w·h + b），所以 w 一过 1，损失就像悬崖一样陡升。不裁剪时，编号 {k} 的更新（从 0 数起）落在 w = {w}，那里的梯度长度是 {norm}，下一步就把损失抛到 {big}。裁剪把长度超过 {c} 的梯度缩短到 {c}，方向不变，所以每一步最长 {maxStep}，损失最低降到 {low}。{mine}',
				vars: (rt, at) => {
					const m = metricsOf(at);
					const g = m.plainGone;
					const before = frameM(at, g - 1);
					const after = frameM(at, g);
					let low = Infinity;
					for (let i = 0; at.tape && i < at.tape.length && i <= at.index; i++) low = Math.min(low, frameM(at, i).clipLoss);
					return {
						T: CLIFF.T,
						k: g - 2,
						w: before.plainW,
						norm: after.plainNorm,
						big: after.plainLoss,
						c: setting(CLIFF.maxNorm),
						maxStep: setting(CLIFF.lr * CLIFF.maxNorm),
						low,
						mine: pick(setupOf(rt.input).clip, L('Your run, the solid path, clips.', '你的运行（实线）开启了裁剪。'), L('Your run, the solid path, does not clip yet.', '你的运行（实线）还没有裁剪。'))
					};
				}
			},
			look: L('Solid: your run; dashed: the other setting. A path that leaves the window was thrown off the cliff.', '实线是你的运行，虚线是另一种设置。离开窗口的路径就是被抛下了悬崖。')
		}
	]
});

export default lesson;
