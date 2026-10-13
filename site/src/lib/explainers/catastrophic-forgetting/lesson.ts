// catastrophic-forgetting: the guided lesson (nine steps; the concept page's primary path). Every number is live: the
// processes train the network in the learner's browser with the kit's autograd.
//   learn-a -> predict -> forget (play) -> curves -> why -> replay (task) -> pull -> tradeoff -> limits
import type { At, I18n, Tape } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { ALL_LABEL, ALL_XY, CLASSES, N_TASK, metricsOf, remedyName, replayCount, setting, settingsOf, type Settings } from './data.ts';
import { REMEDIES, paramsOf } from './process.ts';
import { PRESETS } from './adapters/settings/adapter.ts';

/** The plane with both tasks' points over the network's answer (the kit's plane2d view). */
export const plane = { key: 'plane2d', props: () => ({ points: ALL_XY, labels: ALL_LABEL, grid: 'grid', classes: CLASSES }) };
const SMALL = paramsOf(16);
/** The metrics of the frame at a cursor of the step's own tape ({} until recorded). */
export function metricsAt(at: At, c: Parameters<Tape['find']>[0]): Record<string, number> {
	const t = at.tape;
	const i = t ? t.find(c) : -1;
	return t && i >= 0 ? (t.frame(i).metrics ?? {}) : {};
}
/** The learner's settings as one phrase: "Replay 25% with 20 stored task-A points". */
export function settingsText(s: Settings): I18n {
	const name = remedyName(s);
	const k = replayCount(s.replay);
	const en = name.en.charAt(0).toLowerCase() + name.en.slice(1);
	return k > 0 ? { en: `${en}, with ${k} stored task-A points`, zh: `${name.zh}，存下 ${k} 个任务 A 的点` } : { en, zh: name.zh };
}
const updates = (n: number) => plural(n, { one: '{n} update', many: '{n} updates', zh: '{n} 次更新' });
const remedy = (id: string) => REMEDIES.find((r) => r.id === id)!.set;

const lesson = defineLesson({
	title: { en: 'Catastrophic forgetting, step by step', zh: '灾难性遗忘，分步课程' },
	summary: {
		en: 'Train a small network on task A, then on task B, and watch A disappear; see why the two tasks fight over the same weights, then try replay, a pull toward the old weights, freezing and a smaller learning rate, and what each one costs.',
		zh: '先在任务 A 上训练一个小网络，再训练任务 B，看着 A 被忘掉；弄清两个任务为什么在争同一组权重，再试试回放、拉向旧权重、冻结和更小的学习率，以及每种办法的代价。'
	},
	fixtures: PRESETS.map((p) => ({ id: p.id, label: p.label, raw: p.raw })),
	steps: [
		{
			id: 'learn-a',
			title: { en: 'Task A: sort the first group', zh: '任务 A：给第一组点分类' },
			source: 'plain',
			at: { phase: 'switch' },
			view: plane,
			provenance: 'live',
			terms: ['mlp', 'gradient-descent', 'decision-boundary'],
			narration: {
				en: "Task A is the group of points at negative x: class 1 above the middle line y = 0, class 0 below it. A network with {params} weights and biases, after {n} updates of gradient descent on A, gets {accA|pct} of A's {nA} points right. It has never seen task B, the group at positive x, and gets {accB|pct} of it right.",
				zh: '任务 A 是 x 为负的那组点：中线 y = 0 以上是第 1 类，以下是第 0 类。一个有 {params} 个权重和偏置的网络在 A 上用梯度下降更新了 {n} 次，A 的 {nA} 个点答对 {accA|pct}。它从没见过任务 B，也就是 x 为正的那组点，在 B 上答对 {accB|pct}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { params: SMALL, n: m.step, nA: N_TASK, accA: m.accA, accB: m.accB };
				}
			},
			look: {
				en: "The background is the network's answer at every place. The line between its two colors, the decision boundary, crosses the whole plane, through B's group too.",
				zh: '背景是网络在每个位置的回答。两种颜色之间的分界线就是决策边界，它横穿整个平面，也穿过 B 的那组点。'
			}
		},
		{
			id: 'predict',
			title: { en: 'Now train on task B only', zh: '现在只在任务 B 上训练' },
			source: 'plain',
			at: { phase: 'switch' },
			view: plane,
			provenance: 'live',
			terms: ['fine-tuning', 'learning-rate'],
			predict: {
				id: 'catastrophic-forgetting-b-on-a',
				question: { en: 'While the network learns task B, what happens to its answers on task A?', zh: '网络学习任务 B 的时候，它在任务 A 上的回答会怎样？' },
				options: [
					{ en: "They stay right: A's points are not in B's training data", zh: '保持正确：A 的点不在 B 的训练数据里' },
					{ en: 'Many of them flip to wrong, perhaps all', zh: '很多会变成错的，甚至全部' },
					{ en: 'They get better as well, since the network keeps learning', zh: '也会变好，因为网络一直在学习' }
				],
				answer: 1,
				explain: {
					en: "B's loss only measures B's points. Gradient descent moves every weight wherever that loss falls fastest, and the boundary in A's group moves with them: nothing in B's loss asks for A's answers to stay.",
					zh: 'B 的损失只衡量 B 的点。梯度下降把每个权重推向这个损失下降最快的方向，A 那组点里的边界也跟着移动：B 的损失里没有任何一项要求 A 的答案保持不变。'
				}
			},
			narration: {
				en: "Task B has the classes the other way round: class 1 below the middle line. Fine-tuning on B means the same network now trains only on B's {nB} points, for {stepsB} updates at the same learning rate, {lr}. Answer first; the next step plays the run.",
				zh: '任务 B 的类别正好反过来：中线以下是第 1 类。在 B 上微调，就是让同一个网络只用 B 的 {nB} 个点训练，用同样的学习率 {lr} 更新 {stepsB} 次。先回答，下一步会播放这次运行。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { nB: N_TASK, stepsB: m.stepsB, lr: setting(m.lrB) };
				}
			},
			look: { en: "A's group is all sorted correctly for now; B's group is sorted the wrong way.", zh: '目前 A 那组点全部分对了；B 那组点被分反了。' }
		},
		{
			id: 'forget',
			title: { en: 'Watch task A disappear', zh: '看着任务 A 消失' },
			source: 'plain',
			at: 'last',
			view: plane,
			play: { from: { phase: 'switch' }, to: 'last', interval: 300, unit: { en: 'update', zh: '次更新' } },
			provenance: 'live',
			terms: ['catastrophic-forgetting'],
			narration: {
				en: "Each frame is {every} more updates on task B. After {nB} on B, B is {accB|pct} right, and A has gone from {a0|pct} to {accA|pct}. {verdict}",
				zh: '每一帧是在任务 B 上多更新 {every} 次。在 B 上做了 {nB}之后，B 答对 {accB|pct}，A 则从 {a0|pct} 变成 {accA|pct}。{verdict}',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const a0 = metricsAt(at, { phase: 'switch' }).accA;
					return {
						every: m.every,
						nB: updates(m.stepB),
						accB: m.accB,
						a0,
						accA: m.accA,
						verdict: pick(
							m.accA < 0.5,
							{ en: "In A's group the boundary has turned over: the network now answers A's points with B's rule. This is catastrophic forgetting.", zh: '在 A 那组点里，边界翻了过来：网络现在用 B 的规则回答 A 的点。这就是灾难性遗忘。' },
							{ en: "So far most of A's points are still on their own side of the boundary.", zh: '到目前为止，A 的大多数点还在自己那一侧。' }
						)
					};
				}
			},
			look: { en: "Watch the boundary in A's group turn over while B's group gets sorted.", zh: '看 A 那组点里的边界怎样翻转，同时 B 那组点被分好。' }
		},
		{
			id: 'curves',
			title: { en: 'Both tasks over the whole run', zh: '整次运行中的两个任务' },
			source: 'plain',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: () => ({
					metrics: ['accA', 'accB'],
					labels: { accA: { en: 'task A right', zh: '任务 A 答对' }, accB: { en: 'task B right', zh: '任务 B 答对' } },
					refs: [{ y: 0.5, label: { en: 'coin flip', zh: '抛硬币' } }]
				})
			},
			provenance: 'live',
			narration: {
				en: 'The whole run on one chart. Task A reached {aTop|pct} after {learnA}. Once B starts, A falls below one half within {fall} on B, {verdict} while B climbs from {b0|pct} to {bEnd|pct}. A ends at {aEnd|pct}.',
				zh: '整次运行画在一张图上。任务 A 用了 {learnA}达到 {aTop|pct}。B 一开始，A 在 B 上只用了 {fall}就掉到一半以下，{verdict}而 B 从 {b0|pct} 升到 {bEnd|pct}。A 最后是 {aEnd|pct}。',
				vars: (_rt, at) => {
					const sw = metricsAt(at, { phase: 'switch' });
					// the first frame at the top of task A (at the latest the switch), the first frame of B with A below one half
					const learnA = metricsAt(at, { where: (f) => f.phase === 'switch' || (f.metrics?.accA ?? -1) >= sw.accA }).step;
					const fall = sw.accA1 < 0.5 ? 1 : metricsAt(at, { phase: 'learnB', where: (f) => (f.metrics?.accA ?? 1) < 0.5 }).stepB;
					return {
						learnA: updates(learnA),
						aTop: sw.accA,
						fall: updates(fall),
						b0: sw.accB,
						bEnd: metricsOf(at).accB,
						aEnd: metricsOf(at).accA,
						verdict: pick(
							fall <= learnA,
							{ en: 'no slower than it was learned,', zh: '一点也不比学会它慢，' },
							{ en: 'more slowly than it was learned,', zh: '比学会它慢，' }
						)
					};
				}
			},
			look: { en: 'The dashed line at one half is a coin flip; task A ends below it, so the network is not guessing but sure and wrong.', zh: '一半处的虚线相当于抛硬币；任务 A 最后落在它下面，说明网络不是在瞎猜，而是很确定地答错。' }
		},
		{
			id: 'why',
			title: { en: "Why: B's downhill is A's uphill", zh: '原因：B 的下坡就是 A 的上坡' },
			source: 'plain',
			at: { phase: 'switch' },
			view: 'zoom.formula',
			provenance: 'live',
			terms: ['gradient', 'cosine-similarity', 'loss'],
			narration: {
				en: "At the end of task A, take the gradient of A's loss and the gradient of B's loss: two lists of {params} slopes, one per weight. Their cosine is {cos}, {verdict} B's first update cuts B's loss from {lb0} to {lb1} and raises A's loss from {la0} to {la1}.",
				zh: '在任务 A 结束时，分别求 A 的损失和 B 的损失的梯度：两组各 {params} 个斜率，每个权重一个。它们的余弦是 {cos}，{verdict}B 的第一次更新把 B 的损失从 {lb0} 降到 {lb1}，同时把 A 的损失从 {la0} 抬到 {la1}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return {
						params: SMALL,
						cos: m.cos,
						verdict: pick(
							m.cos < 0,
							{ en: 'below 0: they point in opposing directions, so a step downhill for B is a step uphill for A.', zh: '小于 0：两者方向相反，所以对 B 是下坡的一步，对 A 就是上坡。' },
							{ en: 'above 0: they point roughly the same way.', zh: '大于 0：两者方向大致相同。' }
						),
						lb0: m.lossB,
						lb1: m.lossB1,
						la0: m.lossA,
						la1: m.lossA1
					};
				}
			},
			look: { en: 'The formula divides the dot product of the two gradients by their lengths: −1 is exactly opposite, 1 the same direction.', zh: '公式把两个梯度的点积除以它们的长度：−1 表示完全相反，1 表示方向相同。' }
		},
		{
			id: 'replay',
			title: { en: 'Remedy 1: replay a few old points', zh: '补救一：回放少量旧数据' },
			source: 'mine',
			at: 'last',
			view: plane,
			provenance: 'live',
			terms: ['replay', 'batch'],
			predict: {
				id: 'catastrophic-forgetting-replay',
				question: { en: 'Replay at 10% stores only a handful of task-A points. How much of task A will the network keep?', zh: '10% 的回放只存了少量任务 A 的点。网络能保住多少任务 A？' },
				options: [
					{ en: 'Almost none: a handful of points cannot outweigh all of B', zh: '几乎保不住：少量点抵不过整个 B' },
					{ en: 'About a tenth of it', zh: '大约十分之一' },
					{ en: 'Nearly all of it', zh: '几乎全部' }
				],
				answer: 2,
				explain: {
					en: "Each stored point puts a piece of A's loss back into every batch. To lower it the network must find a boundary that bends at x = 0, one that sorts both groups, and such a boundary exists, so a handful of points is enough here.",
					zh: '每个存下的点都会把 A 的一部分损失放回每一批数据里。要让它下降，网络必须找到一条在 x = 0 处拐弯、能同时分好两组点的边界；这样的边界是存在的，所以这里少量点就够了。'
				}
			},
			task: {
				prompt: { en: 'Set the replay share to 10% or more in the side panel (or press Replay), then look at the boundary in A\'s group.', zh: '在侧栏把回放比例调到 10% 或更高（或者按“回放”），然后看 A 那组点里的边界。' },
				done: (rt) => settingsOf(rt.input).replay >= 0.1,
				hint: { en: 'The replay slider is the first control in the side panel and in the toy below the lesson.', zh: '回放滑块是侧栏里的第一个控件，课程下方的玩具实验里也有。' },
				waiting: { en: 'Not yet: the replay share is still below the goal.', zh: '还没有：回放比例仍低于目标。' },
				doneText: { en: 'Done: the run restarted with replay.', zh: '完成：这次运行已经带着回放重新开始。' }
			},
			narration: {
				en: "Replay keeps a small memory of task-A points and mixes it into every training batch of task B, so the loss also asks for A's answers. Your run: {set}. After task B it gets task A {accA|pct} right and task B {accB|pct}.",
				zh: '回放会存下少量任务 A 的点，把它们混进任务 B 的每一批训练数据里，这样损失里也有一部分在考查 A 的答案。你的这次运行：{set}。学完任务 B 之后，它在任务 A 上答对 {accA|pct}，在任务 B 上答对 {accB|pct}。',
				vars: (rt, at) => {
					const m = metricsOf(at);
					return { set: settingsText(settingsOf(rt.input)), accA: m.accA, accB: m.accB };
				}
			},
			look: { en: "Compare the boundary in A's group with the one at the end of the forgetting step.", zh: '把 A 那组点里的边界和“看着任务 A 消失”那一步结束时的边界比一比。' }
		},
		{
			id: 'pull',
			title: { en: 'Remedy 2: pull toward the old weights', zh: '补救二：拉向旧权重' },
			source: 'pullRun',
			at: { phase: 'learnB', step: 350 },
			view: 'zoom.formula',
			provenance: 'live',
			terms: ['weight-pull'],
			narration: {
				en: "This remedy needs no old data: keep the weights w_A from the end of task A and add λ/2 · Σ (w − w_A)² to B's loss, a pull of strength λ = {lam}. Its slope adds λ × (w − w_A) to each weight's gradient. Take the weight that has moved furthest, {wa} after task A and {w} after {nB} on B: in this update B's slope alone moves it by {push|signed}, the pull by {back|signed}, so it ends at {w1}.",
				zh: '这种办法不需要旧数据：记下任务 A 结束时的权重 w_A，在 B 的损失上加 λ/2 · Σ (w − w_A)²，λ 是拉力的强度，这里 λ = {lam}。它的斜率给每个权重的梯度加上 λ × (w − w_A)。看移动得最远的那个权重：任务 A 结束时是 {wa}，在 B 上做了 {nB}之后是 {w}。这次更新中，单靠 B 的斜率让它移动 {push|signed}，拉力让它移动 {back|signed}，最后它变成 {w1}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return {
						lam: setting(m.pull),
						wa: m.pa,
						w: m.pw,
						nB: updates(m.stepB - 1),
						push: -m.lrB * m.pg,
						back: -m.lrB * m.pp,
						w1: m.pw1
					};
				}
			},
			look: { en: "The formula's first three lines add B's slope and the pull, which points back toward w_A; the last two lines are the update.", zh: '公式前三行是 B 的斜率加上拉力，拉力指回 w_A；最后两行是这次更新。' }
		},
		{
			id: 'tradeoff',
			title: { en: 'Six ways to train on B, side by side', zh: '在 B 上训练的六种做法并排比较' },
			source: 'remedies',
			at: 'last',
			view: 'catastrophic-forgetting/remedies',
			provenance: 'live',
			narration: {
				en: 'Six copies of the network after task A, each trained on B another way. Replay with only {k} stored points keeps task A at {rA|pct} and still learns B to {rB|pct}. The pull is a seesaw: at λ = {lw} task A ends at {wA|pct}; at λ = {ls} A stays at {sA|pct}, but B only reaches {sB|pct}.',
				zh: '把学完任务 A 的网络复制六份，每份用不同的方式在 B 上训练。回放只存了 {k} 个点，就把任务 A 保持在 {rA|pct}，同时 B 也学到 {rB|pct}。拉力像跷跷板：λ = {lw} 时任务 A 最后只剩 {wA|pct}；λ = {ls} 时 A 保持在 {sA|pct}，但 B 只到 {sB|pct}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return {
						k: replayCount(remedy('replay').replay),
						rA: m.accA_replay,
						rB: m.accB_replay,
						lw: setting(remedy('pullWeak').pull),
						wA: m.accA_pullWeak,
						ls: setting(remedy('pull').pull),
						sA: m.accA_pull,
						sB: m.accB_pull
					};
				}
			},
			look: { en: 'Read the two bars of each row: a good remedy keeps both long.', zh: '看每一行的两根条：好的办法能让两根都很长。' }
		},
		{
			id: 'limits',
			title: { en: 'Two remedies that do not work here', zh: '在这里不管用的两种办法' },
			source: 'remedies',
			at: 'last',
			kindOf: 'limits',
			view: {
				key: 'loss-curve',
				props: () => ({
					metrics: ['accA_plain', 'accA_freeze', 'accA_slow'],
					labels: { accA_plain: remedyName(remedy('plain')), accA_freeze: remedyName(remedy('freeze')), accA_slow: remedyName(remedy('slow')) }
				})
			},
			provenance: 'live',
			terms: ['frozen-layer', 'learning-rate'],
			narration: {
				en: 'Freezing the first layer keeps the features task A built, yet A ends at {fA|pct}: the answer is decided in the later layers, and B rewrites them. A learning rate ten times smaller only slows the slide: A is still {s50|pct} right after {n50} on B, but {sEnd|pct} after {n}.',
				zh: '冻结第一层保住了任务 A 学到的特征，但 A 最后只剩 {fA|pct}：答案是由后面的层决定的，而 B 改写了这些层。把学习率缩小到十分之一只是让下滑变慢：在 B 上做了 {n50}之后 A 还答对 {s50|pct}，但做了 {n}之后只剩 {sEnd|pct}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const mid = metricsAt(at, { step: 50 });
					return { fA: m.accA_freeze, s50: mid.accA_slow, n50: updates(mid.step), sEnd: m.accA_slow, n: updates(m.step) };
				}
			},
			look: { en: "Each line is task A's accuracy during training on B: the slow one falls later and ends nearly as low.", zh: '每条线是在 B 上训练时任务 A 的准确率：慢速的那条掉得晚，最后却几乎一样低。' }
		}
	]
});

export default lesson;
