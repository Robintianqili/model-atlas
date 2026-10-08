// gan-2d: the guided lesson, one idea per step. Every number is a {placeholder} computed from frames that ran (scene.ts
// reads the same numbers the arena view draws); every sentence that is not live is tested in claims.test.ts.
import type { At, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { presetOf } from './data.ts';
import { mean, MODE_MIN, MODE_R, N_SHOW, NEAR, SPAN } from './net.ts';
import { dLossParts, gLossOf, lateRange, sceneOf, type Scene } from './scene.ts';

const L = (en: string, zh: string) => ({ en, zh });
/** a number, or a dash when there is nothing to measure (no points; `why` false: a bare dash, after the first one) */
const val = (v: number, why = true): TmplVal => (Number.isFinite(v) ? v : why ? L('— (no points)', '—（没有点）') : '—');
const arena = (show: string[], extra: Record<string, unknown> = {}) => ({ key: 'gan-2d/arena', props: () => ({ show, ...extra }) });
const setting = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);
const optOf = (at: At) => (at.tape?.opt ?? {}) as Record<string, unknown>;
/** a metric's recorded values on the step's tape, up to the cursor */
const valuesOf = (at: At, k: string) => Array.from(at.tape?.metric(k) ?? []).slice(0, at.index + 1).filter(Number.isFinite);
/** D can hardly tell real from generated: its mean scores for the two differ by less than 0.02 */
const unsure = (s: Scene | null) => !!s && Math.abs(s.dReal - s.dFake) < 0.02;
/** the width of the hidden layers that ran */
const widthOf = (s: Scene) => s.params['gen.0.weight']?.shape[0] ?? NaN;

const lesson: LessonDef = {
	title: L('2-D GAN: two networks play a game', '二维 GAN：两个网络的博弈'),
	summary: L(
		'A generator turns noise into points, a discriminator judges them against your points, and they train in turns. One step of each slowed down, the whole game, how it differs from flow matching, and its best-known failure.',
		'生成器把噪声变成点，判别器拿它们和你的点比较、打分，两者轮流训练。放慢看各自的一步，再看整场博弈、它和流匹配有什么不同，以及它最有名的失败方式。'
	),
	steps: [
		{
			id: 'real',
			source: 'train',
			at: 'first',
			view: arena(['real']),
			provenance: 'live',
			title: L('The real data', '真实数据'),
			narration: {
				en: 'A GAN (generative adversarial network) learns to make new points that look like a set of examples. Here the examples are your {pts} (filled dots): the real data. The GAN never gets a rule for where they lie, only the examples themselves.',
				zh: 'GAN（生成对抗网络）学着造出和一组样例相像的新点。这里的样例就是你的 {pts}（实心圆点）：真实数据。GAN 从来拿不到“点在哪里”的规则，只有这些样例本身。',
				vars: (rt, at) => ({ pts: plural(sceneOf(rt, at, 'train')?.real.n ?? NaN, { one: '{n} point', many: '{n} points', zh: '{n} 个点' }) })
			},
			look: L('The filled dots. Brush more in the input box, or pick another set there; the whole lesson reruns on them.', '看实心圆点。可以在输入框里再画一些点，或在那里换一组点；整节课都会用它们重新计算。'),
			terms: ['gan']
		},
		{
			id: 'noise',
			source: 'train',
			at: 'first',
			view: arena(['noise', 'real']),
			provenance: 'live',
			title: L('Noise goes in', '输入是噪声'),
			narration: {
				en: 'The first network, the generator, starts from noise. For each point it will make, it draws two random numbers z₁ and z₂, each anywhere between −1 and 1 with equal chance: a random spot in the square marked “noise z”. Here are {nz} such draws. A new draw gives a new point, so a GAN can make as many points as you like.',
				zh: '第一个网络叫生成器，它从噪声出发。每造一个点，它先抽两个随机数 z₁ 和 z₂，各自落在 −1 到 1 之间的任何位置，机会均等：也就是标着“噪声 z”的方块里的一个随机位置。这里画出了 {nz} 次这样的抽取。抽一次新的就得到一个新点，所以 GAN 想造多少点都可以。',
				vars: () => ({ nz: N_SHOW })
			},
			look: L('The dots in the noise square: the noise. Nothing is on the plane yet except your points.', '看噪声方块里的点：这就是噪声。平面上暂时只有你的点。'),
			terms: ['noise']
		},
		{
			id: 'generator',
			source: 'train',
			at: 'first',
			view: arena(['noise', 'real', 'fake', 'sheet'], { one: 0 }),
			provenance: 'live',
			title: L('The generator turns noise into points', '生成器把噪声变成点'),
			narration: {
				en: 'The generator G is a small network: 2 numbers in, two hidden layers of {h} neurons, 2 numbers out, a point (x, y). The ringed noise point z = ({z1}, {z2}) becomes G(z) = ({x}, {y}). Sent through G as well, the pink grid lines of the noise square become G’s sheet: it shows where G puts the whole square. Untrained, G squeezes all of it into a patch about {w} wide around ({cx}, {cy}); training has to spread it over your points.',
				zh: '生成器 G 是一个小网络：输入 2 个数，经过两层各 {h} 个神经元的隐藏层，输出 2 个数，也就是一个点 (x, y)。圈出的噪声点 z = ({z1}, {z2}) 变成了 G(z) = ({x}, {y})。把噪声方块里的粉色网格线也送进 G，就得到 G 的“布”：它显示 G 把整个方块放到了哪里。还没训练时，G 把整个方块挤进 ({cx}, {cy}) 附近一块大约 {w} 宽的小区域；训练要做的，就是把它铺到你的点上。',
				vars: (rt, at): Record<string, TmplVal> => {
					const s = sceneOf(rt, at, 'train');
					if (!s) return { h: NaN };
					const P = s.fake.length / 2;
					const xs = Array.from({ length: P }, (_, j) => s.fake[2 * j]);
					const ys = Array.from({ length: P }, (_, j) => s.fake[2 * j + 1]);
					const w = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
					return { h: widthOf(s), z1: s.z[0], z2: s.z[1], x: s.fake[0], y: s.fake[1], w, cx: mean(xs), cy: mean(ys) };
				}
			},
			look: L('The ringed dot in the noise square and the ringed point on the plane: one noise point and the point G makes from it. The pink patch on the plane is G’s sheet, squeezed small for now; the green dots are your points, where G should end up.', '看噪声方块里圈出的点和平面上圈出的点：一个噪声点，以及 G 用它造出的点。平面上那一小块粉色就是 G 的“布”，现在还被挤得很小；绿色点是你的点，也就是 G 最终应该去的地方。'),
			terms: ['generator', 'mlp']
		},
		{
			id: 'discriminator',
			source: 'train',
			at: 'first',
			view: arena(['map', 'real', 'fake']),
			provenance: 'live',
			title: L('The discriminator scores every place', '判别器给每个位置打分'),
			narration: {
				en: 'The second network, the discriminator D (2 numbers in, two hidden layers of {h} neurons, 1 number out), looks at one point and outputs a number between 0 and 1: how sure it is that the point is real. The background shows D at every place: green where it says “real”, pink where it says “fake”, a line where it says exactly ½. Untrained, it is unsure everywhere: it gives your points {dr} and G’s points {df} on average.',
				zh: '第二个网络是判别器 D（输入 2 个数，经过两层各 {h} 个神经元的隐藏层，输出 1 个数）。它看一个点，输出 0 到 1 之间的一个数：它有多确定这个点是真的。背景画出了 D 在每个位置的回答：绿色表示它说“真”，粉色表示它说“假”，线上恰好是 ½。还没训练时，它处处拿不准：给你的点平均 {dr} 分，给 G 的点平均 {df} 分。',
				vars: (rt, at) => {
					const s = sceneOf(rt, at, 'train');
					return { h: s ? widthOf(s) : NaN, dr: val(s?.dReal ?? NaN), df: val(s?.dFake ?? NaN) };
				}
			},
			look: L('The faint colours: D has learned nothing yet, so its answer is close to ½ everywhere. The ½ line just falls where D’s random starting weights put it.', '看很淡的颜色：D 还什么都没学，所以它的回答处处接近 ½。那条 ½ 线只是落在 D 随机的初始权重碰巧给出的位置。'),
			terms: ['discriminator']
		},
		{
			id: 'd-step',
			source: 'slow',
			at: { phase: 'update', step: 0 },
			view: arena(['map', 'real', 'fake']),
			provenance: 'live',
			title: L('D’s turn: one step with numbers', '轮到 D：把一步训练算一遍'),
			narration: {
				en: 'The networks take turns; D goes first, on a batch of {real} and {fake}. Its loss has two parts (ln: the natural logarithm): the average of −ln D(x) over the real points, {a}, small when they score near 1, plus the average of −ln(1 − D(x)) over the generated ones, {b}, small when they score near 0; in all {loss|n4}. One Adam step on D’s weights lowers it by {drop} to {after|n4}: on this batch, real points now average {r1|n4} (were {r0|n4}), generated ones {f1|n4} (were {f0|n4}).',
				zh: '两个网络轮流训练，D 先来，用的是一批数据：{real}和 {fake}。它的损失由两部分相加（ln 是自然对数）：真实点上 −ln D(x) 的平均值 {a}，真实点得分接近 1 时它就小；加上生成点上 −ln(1 − D(x)) 的平均值 {b}，生成点得分接近 0 时它就小；合计 {loss|n4}。用 Adam 优化器对 D 的权重走一步，这个损失下降了 {drop}，变成 {after|n4}：在这批数据上，真实点现在平均 {r1|n4} 分（原来 {r0|n4}），生成点 {f1|n4} 分（原来 {f0|n4}）。',
				vars: (rt, at): Record<string, TmplVal> => {
					const s = sceneOf(rt, at, 'slow');
					if (!s?.before || !s.fakeBefore) return { a: NaN };
					const nR = s.realShown.length / 2;
					const [a, b] = nR ? dLossParts(s.before, s.realShown, s.fakeBefore) : [NaN, NaN];
					const [a1, b1] = nR ? dLossParts(s.params, s.realShown, s.fake) : [NaN, NaN];
					return {
						real: plural(nR, { one: '{n} real point', many: '{n} real points', zh: '{n} 个真实点' }),
						fake: plural(s.fake.length / 2, { one: '{n} generated point', many: '{n} generated points', zh: '{n} 个生成的点' }),
						a: val(a), b: val(b, false), loss: val(a + b, false), after: val(a1 + b1, false), drop: val(a + b - a1 - b1, false),
						r0: val(s.dRealBefore, false), r1: val(s.dReal, false), f0: val(s.dFakeBefore, false), f1: val(s.dFake, false)
					};
				}
			},
			look: L('Only this batch is drawn. Dashed: the ½ line before the step; solid: after. D says nearly ½ everywhere, so even a small step can move the line a long way; the averages under the picture (before → after) show how small this one was.', '图上只画了这一批点。虚线是这一步之前的 ½ 线，实线是之后的。D 处处都说接近 ½，所以哪怕很小的一步，也可能让这条线移动很远；图下方的平均分（之前 → 之后）显示这一步其实有多小。'),
			terms: ['loss', 'optimizer']
		},
		{
			id: 'g-step',
			source: 'slow',
			at: { phase: 'update', step: 1 },
			view: arena(['map', 'real', 'fake', 'arrows', 'move', 'zoom']),
			provenance: 'live',
			title: L('G’s turn: one step, with D held still', '轮到 G：D 不动，G 走一步'),
			predict: {
				id: 'gan-2d-g-step',
				question: L('Now G takes one step and D’s weights stay fixed. Which way will the generated points move?', '现在 G 走一步，D 的权重保持不动。生成的点会往哪里移动？'),
				options: [L('Straight towards the nearest real point', '直奔最近的真实点'), L('Roughly along the arrows: towards where D says “real”', '大致顺着箭头：朝 D 说“真”的地方'), L('Nowhere: G changes weights, not points', '哪儿也不去：G 改的是权重，不是点')],
				answer: 1,
				explain: L(
					'G never sees the real points. All it gets is the gradient of D’s score, drawn as arrows. Changing G’s weights to lower −ln D(G(z)) moves the generated points roughly along their arrows, into places D finds more real. G chases D’s map, not your data.',
					'G 从来看不到真实的点。它得到的只有 D 的分数的梯度，也就是图上的箭头。为了降低 −ln D(G(z)) 而改动 G 的权重，生成的点就会大致顺着各自的箭头移动，移向 D 认为更像真的地方。G 追的是 D 的地图，而不是你的数据。'
				)
			},
			narration: {
				en: 'Now it is G’s turn, and D’s weights stay fixed. G’s loss is the average of −ln D(x) over its generated points x = G(z), {g0|n4}: small when D scores them near 1. Each arrow starts at a generated point and shows the direction that raises D’s score fastest there. One Adam step on G’s weights makes the loss {g1|n4}, and each short line shows where a point went.',
				zh: '现在轮到 G，D 的权重保持不动。G 的损失是生成点 x = G(z) 上 −ln D(x) 的平均值，为 {g0|n4}：D 给生成点的分数接近 1 时，它就小。每个箭头从一个生成点出发，指向在那里让 D 的分数升得最快的方向。用 Adam 对 G 的权重走一步，损失变成 {g1|n4}，每条短线显示一个点移到了哪里。',
				vars: (rt, at) => {
					const s = sceneOf(rt, at, 'slow');
					return s?.before && s.fakeBefore ? { g0: gLossOf(s.before, s.fakeBefore), g1: gLossOf(s.params, s.fake) } : { g0: NaN, g1: NaN };
				}
			},
			look: L('The framed box, magnified in the second picture: the arrows, and the short lines from where each point was to where it is now. Under the picture: a count that checks your answer.', '看框出的区域（另一张图是它放大后的样子）：箭头，以及从每个点原来的位置连到现在位置的短线。图下方有一个计数，可以用来核对你的答案。'),
			terms: ['gradient', 'generator']
		},
		{
			id: 'watch',
			source: 'train',
			at: 'last',
			view: arena(['map', 'real', 'fake', 'sheet'], { reach: true }),
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 120 },
			provenance: 'live',
			title: L('Take turns, and watch', '轮流训练，看它怎么变'),
			narration: {
				en: 'Training repeats the two turns, D step then G step, {pairs} times each; G learns more slowly than D (learning rate {lrG}, D’s {lrD}). Watch the bent sheet spread over your points while D’s map keeps changing. At the end, {recall|pct} of your points have a generated point within {near} (the picture is {wide} wide), and {precision|pct} of the generated points lie that close to a real one. D gives real points {r|n4} and generated ones {f|n4}: {verdict}.',
				zh: '训练把这两步不断重复：D 走一步，G 走一步，各 {pairs} 次；G 学得比 D 慢（G 的学习率是 {lrG}，D 的是 {lrD}）。看那张弯曲的“布”在你的点上铺开，而 D 的地图一直在变。最后，{recall|pct} 的真实点在距离 {near}（整张图宽 {wide}）以内能找到生成的点；{precision|pct} 的生成点离某个真实点也这么近。D 给真实点 {r|n4} 分，给生成点 {f|n4} 分：{verdict}。',
				vars: (rt, at) => {
					const s = sceneOf(rt, at, 'train');
					const o = optOf(at);
					const steps = at.frame?.step ?? NaN;
					return {
						pairs: Math.floor(steps / 2),
						lrD: setting(o.lr),
						lrG: setting(o.lrG),
						recall: val(s?.recall ?? NaN),
						precision: val(s?.precision ?? NaN),
						near: setting(NEAR),
						wide: setting(2 * SPAN),
						r: val(s?.dReal ?? NaN),
						f: val(s?.dFake ?? NaN),
						verdict: !s?.real.n ? L('there are no real points to compare', '没有真实点可比') : pick(unsure(s), L('nearly the same: it can hardly tell real from generated any more', '几乎一样，它已经很难分清真假了'), L('it can still tell them apart', '它还分得清'))
					};
				}
			},
			look: L('The pink sheet stretching over the green dots, and the background fading towards ½ where the two kinds of points mix.', '看粉色的“布”在绿色点上铺开，以及两种点混在一起的地方，背景渐渐褪向 ½。'),
			terms: ['learning-rate']
		},
		{
			id: 'game',
			source: 'train',
			at: 'last',
			view: 'gan-2d/game',
			provenance: 'live',
			title: L('A game, not a slope: compare with flow matching', '是博弈，不是下坡：和流匹配比一比'),
			narration: {
				en: 'Most networks, like flow matching, lower one loss: the lower, the better. A GAN is a game: if D cannot tell real from fake, it says ½ everywhere, and D’s loss is 2 ln 2 = {eq} (ln 2 per kind of point), G’s is ln 2 = {half}. Over the last {k} steps they average {dl|n4} and {gl|n4}. Yet D’s loss started at {d0|n4}, near 2 ln 2, while G made only a small patch: these curves cannot show how good G’s points are; step 7’s picture can.',
				zh: '大多数网络（比如流匹配）只降低一个损失：越低越好。GAN 却是一场博弈：如果 D 分不清真假，它就处处说 ½，这时 D 的损失是 2 ln 2 = {eq}（每种点各贡献 ln 2），G 的损失是 ln 2 = {half}。最后 {k} 步里，它们的平均值是 {dl|n4} 和 {gl|n4}。可是在第 0 步，G 造的点还只是一小块，D 的损失就已经是 {d0|n4}，接近 2 ln 2：所以这两条曲线看不出 G 的点有多好，第 7 步的图才看得出。',
				vars: (rt, at) => {
					const none = !sceneOf(rt, at, 'train')?.real.n;
					return { eq: 2 * Math.LN2, half: Math.LN2, k: 200, dl: val(none ? NaN : mean(valuesOf(at, 'dLoss').slice(-200))), gl: val(none ? NaN : mean(valuesOf(at, 'gLoss').slice(-200))), d0: val(none ? NaN : (valuesOf(at, 'dLoss')[0] ?? NaN)) };
				}
			},
			look: L('Where the two curves start and where they end. The link under them opens the flow matching page.', '看两条曲线从哪里开始、在哪里结束。曲线下方的链接通往流匹配那一页。'),
			terms: ['loss']
		},
		{
			id: 'collapse',
			source: 'same',
			at: 'last',
			view: arena(['map', 'real', 'fake', 'blobs'], { choices: [{ source: 'same', label: L('G as fast as D', 'G 和 D 一样快') }, { source: 'slower', label: L('G 5 times slower (this page’s setting)', 'G 慢 5 倍（本页的设置）') }] }),
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 120 },
			provenance: 'live',
			kindOf: 'limits',
			title: L('When it fails: mode collapse', '失败的时候：模式坍缩'),
			narration: {
				en: 'A well-known way GANs fail is mode collapse. The data here is eight blobs, the same for everyone, and G learns as fast as D (learning rate {lr} for both). G piles its points onto a few blobs; D learns to call those places fake, so G jumps to other blobs, and so on. At the end of training it covers {k} of the 8 blobs; from step {from} on, it covered between {lo} and {hi}.',
				zh: 'GAN 有一种很有名的失败方式，叫模式坍缩。这里的数据是八团点，每个人看到的都一样，而且 G 学得和 D 一样快（两者的学习率都是 {lr}）。G 把点堆到少数几团上；D 学会把那些地方判为假，于是 G 跳到别的团，如此反复。训练结束时，它覆盖了 8 团中的 {k} 团；从第 {from} 步起，它覆盖的团数在 {lo} 到 {hi} 之间。',
				vars: (rt, at) => {
					const late = lateRange(at.tape, 'modes', at.index);
					return { lr: setting(optOf(at).lrG), k: sceneOf(rt, at, 'same')?.modes ?? NaN, from: late.from, lo: late.lo, hi: late.hi };
				}
			},
			task: {
				prompt: L('Slow G down: click “G 5 times slower” right above the picture, then compare the blob counts of the two runs under the picture.', '让 G 慢下来：点图片正上方的“G 慢 5 倍”，再比较图下方两次运行覆盖的团数。'),
				done: (rt) => rt.sources.slower?.status === 'ready',
				hint: L('The two buttons sit between this box and the picture.', '两个按钮就在这个框和图片之间。')
			},
			look: {
				en: 'Solid circles are blobs with at least {min} generated points within {r} of their centre; dashed ones are missed. Generated points that fly off the picture are counted under it.',
				zh: '实线圆圈表示中心 {r} 以内至少有 {min} 个生成点的团；虚线圆圈表示没覆盖到的团。飞出图外的生成点在图下方计数。',
				vars: () => ({ min: MODE_MIN, r: setting(MODE_R) })
			},
			terms: ['mode-collapse']
		}
	],
	// inputs every step must survive, besides the adapter's edge fixtures (no points, one point, one class)
	fixtures: (['blob', 'two', 'eight', 'smiley'] as const).map((id) => ({ id, label: L(`Preset: ${id}`, `预设：${id}`), raw: presetOf(id) }))
};

export default lesson;
