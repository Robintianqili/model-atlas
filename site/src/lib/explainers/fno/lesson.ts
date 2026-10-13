// fno: the guided lesson (11 steps). Every number is a {placeholder}: structure from the spec (dims()), live values from
// the trace of the learner's curve and the solver (fno.ts), the solver's frames, and measured ones ({fact:...}).
//   1 a curve is 64 numbers on a ring -> 2 the solver's 50 steps -> 3 lift -> 4 FFT -> 5 keep the lowest modes (predict)
//   -> 6 complex weights -> 7 inverse FFT + pointwise path + GELU -> 8 prediction vs solver -> 9 the learned filter ->
//   10 how many modes does the curve need (task) -> 11 limits. Claims not computed live: claims.test.ts.
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { fmtSetting } from '../../atlas/core/format.ts';
import { num, pick, plural } from '../../atlas/lesson/say.ts';
import { AMP, STEPS, exactGain, fdGain, presetY } from './heat.ts';
import { CH, GOAL, T, argmax, col, errByKeep, fewest, heights, learnedGain, liftOf, mixMode, stateOf, taller } from './fno.ts';
import { K, presetOf } from './adapters/series/adapter.ts';
import { TALL } from './views/field/meta.ts';

type RT = ExplainerRuntime;
const words = (v: I18n) => v as unknown as TmplVal;
const view = (mode: string) => ({ key: 'fno/field', props: () => ({ mode }) });
const NaNs = { err: NaN, at: NaN };
/** a number as {name} prints it, in both languages (a string, so it can sit inside a composed phrase) */
const n3 = (v: number) => { const x = num(v); return typeof x === 'number' ? { en: String(x), zh: String(x) } : x; };
/** "+ 0.0320" / "− 0.0320": a shift written after a product, with its sign as the operator */
const signed = (v: number): TmplVal => (Number.isFinite(v) ? { en: `${v < 0 ? '−' : '+'} ${n3(Math.abs(v)).en}`, zh: `${v < 0 ? '−' : '+'} ${n3(Math.abs(v)).zh}` } : NaN);

/** structural numbers, read from the spec (never typed into the text) */
export function dims(rt: RT) {
	const out = (id: string, i = 0) => Number(rt.spec?.layers.find((l) => l.id === id)?.outShape[i] ?? NaN);
	return { n: out('lift'), w: out('lift', 1), m: out('f0.fft'), K: out('f0.keep'), steps: STEPS };
}
/** where GELU changes channel c of layer 1 most */
export function geluAt(rt: RT) {
	const c = CH(rt), s = col(T(rt, 'f0.sum'), c), g = col(T(rt, 'f0.gelu'), c), at = argmax(s.map((v, j) => Math.abs(v - g[j])));
	return { at: s.length ? at : NaN, sum: s[at] ?? NaN, g: g[at] ?? NaN };
}

const lesson: LessonDef = {
	title: { en: 'A neural operator that learned how heat spreads', zh: '学会热量如何扩散的神经算子' },
	summary: {
		en: 'Watch a solver spread heat around a ring in 50 small steps, then follow your curve through a Fourier neural operator that jumps to the same answer in one pass: FFT, the kept modes, learned complex weights, inverse FFT, a pointwise path; then compare the filter it learned with the solver’s, find how many modes your curve needs, and see where it fails.',
		zh: '先看求解器用 50 个小步让热量沿环扩散，再跟着你的曲线走过一个傅里叶神经算子，它一次就跳到同样的答案：FFT、保留的模式、学来的复数权重、逆 FFT、逐点通路；然后把它学到的滤波器和求解器的比一比，找出你的曲线需要多少个模式，最后看看它在哪里会出错。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'A temperature curve is 64 numbers', zh: '一条温度曲线就是 64 个数' },
				view: view('input'),
				narration: {
					en: 'Picture a thin metal ring with a temperature at each of {n} points; point {last} sits next to point 0, so the curve wraps around. This curve is {what}; its hottest point is {top}, at point {at} (counting from 0). The heat equation says how it changes: heat flows from hot to cold, so bumps flatten.',
					zh: '想象一个细细的金属环，环上 {n} 个点各有一个温度；点 {last} 挨着点 0，所以曲线首尾相连。这条曲线是{what}；最热的是点 {at}（从 0 数起），温度 {top}。热传导方程描述它怎样变化：热量从热处流向冷处，所以鼓包会变平。',
					vars: (rt) => {
						const s = stateOf(rt), p = presetOf(rt.input), at = s ? argmax(s.u0) : NaN;
						return { ...dims(rt), last: dims(rt).n - 1, what: words(p ? { en: `a preset (${p.en.toLowerCase()})`, zh: `一个预设（${p.zh}）` } : { en: 'your own drawing', zh: '你自己画的' }), at, top: s ? s.u0[at] : NaN };
					}
				},
				look: { en: 'Draw on the curve in the input box, or pick a preset: every step follows your curve.', zh: '在输入框里的曲线上画，或者选一个预设：后面每一步都跟着你的曲线走。' },
				terms: ['heat-equation']
			}
		},
		{
			id: 'solver',
			source: 'solve',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 80 },
			view: view('solver'),
			provenance: 'live',
			title: { en: 'The solver: 50 small steps', zh: '求解器：50 个小步' },
			narration: {
				en: 'A numerical solver follows the heat equation in small time steps: at every step, each point moves a little toward the average of its two neighbours. After {frame.step} steps the steepest rise between two neighbouring points is {frame.steep}, down from {steep0}, and the curve spans {frame.spread} from lowest to highest, down from {spread0}. Narrow bumps and sharp corners flatten first; wide hills sink slowly.',
				zh: '数值求解器把热传导方程拆成许多小时间步：每一步，每个点都向两边邻居的平均值靠拢一点。{frame.step} 步之后，相邻两点之间最陡的落差是 {frame.steep}，原来是 {steep0}；曲线从最低点到最高点相差 {frame.spread}，原来是 {spread0}。窄鼓包和锐利的拐角最先变平；宽的山丘下沉得很慢。',
				vars: (rt) => { const s = rt.sources.solve?.tape, m = s?.length ? s.frame(0).metrics : undefined; return { spread0: Number(m?.spread ?? NaN), steep0: Number(m?.steep ?? NaN) }; }
			},
			look: { en: 'Press Play to watch every step; the dotted curve is where it started.', zh: '按“播放”观看每一步；点线是起始曲线。' },
			terms: ['numerical-solver', 'finite-difference']
		},
		{
			id: 'lift',
			source: 'fwd',
			at: { layer: 'lift' },
			view: view('lift'),
			provenance: 'live',
			title: { en: 'Lift: 4 channels per point', zh: '升维：每个点 4 个通道' },
			narration: {
				en: 'Now the neural operator, which should jump from the starting curve to the solver’s answer in one pass. First a pointwise linear layer, the lift, turns each height into {w} numbers, the channels. Every channel is a scaled and shifted copy of your curve: channel {ch} (counting from 0) is {a} × height {b}. The lesson follows channel {ch}, the one scaled most.',
				zh: '现在轮到神经算子，它要从起始曲线一次跳到求解器的答案。第一步是逐点的线性层，即升维：把每个高度变成 {w} 个数，也就是通道。每个通道都是你的曲线缩放并平移后的副本：通道 {ch}（从 0 数起）等于 {a} × 高度 {b}。本课跟踪通道 {ch}，它被放大得最多。',
				vars: (rt) => {
					const ch = CH(rt), l = liftOf(rt)[ch];
					return { ...dims(rt), ch, a: l?.a ?? NaN, b: signed(l?.b ?? NaN) };
				}
			},
			look: { en: 'Four curves with the same shape: only their scale and height differ.', zh: '四条形状相同的曲线：只有缩放和高低不同。' },
			terms: ['neural-operator', 'channel', 'dense-layer']
		},
		{
			id: 'fft',
			source: 'fwd',
			at: { layer: 'f0.fft' },
			view: view('fft'),
			provenance: 'live',
			title: { en: 'The FFT: the curve as waves', zh: 'FFT：把曲线看成波' },
			narration: {
				en: 'The first Fourier layer starts with the FFT. It rewrites channel {ch}’s {n} values as {m} waves, the Fourier modes: mode k goes k times around the ring, and mode 0 is the average. The bars show how tall each wave is; the tallest is mode {kmax}, {hmax} high. Wide shapes live in the low modes, sharp corners and wiggles also need high ones.',
				zh: '第一个傅里叶层从 FFT 开始。它把通道 {ch} 的 {n} 个值改写成 {m} 个波，即傅里叶模式：模式 k 绕环 k 圈，模式 0 是平均值。条形表示每个波有多高；最高的是模式 {kmax}，高 {hmax}。宽的形状靠低模式就够了，锐利的拐角和小抖动还需要高模式。',
				vars: (rt) => {
					const ch = CH(rt), h = heights(T(rt, 'f0.fft'), ch), k = h.length ? argmax(h) : NaN;
					return { ...dims(rt), ch, kmax: k, hmax: h[k] ?? NaN };
				}
			},
			look: { en: 'One bar per mode, from mode 0 on the left to the highest mode on the right.', zh: '每个模式一根条，从左边的模式 0 到右边最高的模式。' },
			terms: ['fft', 'fourier-mode', 'fourier-layer']
		},
		{
			id: 'keep',
			source: 'fwd',
			at: { layer: 'f0.keep' },
			view: view('keep'),
			provenance: 'live',
			title: { en: 'Keep only the lowest modes', zh: '只保留最低的模式' },
			predict: {
				id: 'fno-keep',
				question: { en: 'The layer now drops every mode above its cut. What does dropping the high modes do to the channel?', zh: '这一层接着把截断以上的模式全部丢掉。丢掉高模式会让这个通道怎样？' },
				options: [
					{ en: 'It loses fine detail: corners round off and small wiggles disappear', zh: '失去细节：拐角变圆，小抖动消失' },
					{ en: 'It moves sideways around the ring', zh: '沿着环向旁边移动' },
					{ en: 'It shrinks evenly and keeps its shape', zh: '整体均匀缩小，形状不变' }
				],
				answer: 0,
				explain: {
					en: 'The high modes are the short waves, so without them only the smooth outline is left: see the rebuilt channel in the stage. For heat this costs almost nothing, because the solver shrinks those short waves to nearly zero in its 50 steps: they were about to vanish anyway.',
					zh: '高模式就是那些短波，没有它们，只剩下平滑的轮廓：看舞台上重建的通道。对热传导来说这几乎没有代价，因为求解器在 50 步里会把这些短波缩到几乎为零：它们本来就要消失。'
				}
			},
			narration: {
				en: 'Next the layer keeps only the lowest {keep} of the {m} modes; the dashed bars are dropped. This cut is what makes a Fourier layer small: it has learned weights for {K} modes only, whatever the number of points. The second chart in the stage rebuilds channel {ch} from the kept modes alone.',
				zh: '接着，这一层只保留 {m} 个模式中最低的 {keep} 个；虚线条被丢掉了。正是这一刀让傅里叶层很小：不管有多少个点，它只为 {K} 个模式学权重。舞台上的第二张图只用保留的模式重建通道 {ch}。',
				vars: (rt) => ({ ...dims(rt), keep: stateOf(rt)?.keep ?? NaN, ch: CH(rt) })
			},
			look: { en: 'Solid bars go on; dashed bars are dropped.', zh: '实心条继续往下走；虚线条被丢掉。' },
			terms: ['fourier-mode']
		},
		{
			id: 'mix',
			source: 'fwd',
			at: { layer: 'f0.mix' },
			select: (rt) => ({ layer: 'f0.mix', unit: [mixMode(rt), CH(rt), 0] }),
			view: { key: 'zoom.weighted-sum', props: () => ({ names: { x: 'a | b', w: 'c | −d', p: 'a·c | −b·d' } }) },
			provenance: 'live',
			title: { en: 'Multiply by learned complex weights', zh: '乘以学来的复数权重' },
			narration: {
				en: 'Each kept mode is two numbers, a and b, read as one complex number a + bi. For every pair of channels the layer multiplies it by a learned complex weight c + di; the real part of the product is a·c − b·d. For {which}, output channel {ch} adds these real parts over the {w} input channels: {nt} products, sum {y}. Scaling a mode sets how much of that wave survives; the imaginary part shifts it around the ring.',
				zh: '每个保留的模式是两个数 a 和 b，合起来看成一个复数 a + bi。对每一对通道，这一层把它乘以一个学来的复数权重 c + di；乘积的实部是 a·c − b·d。对于{which}，输出通道 {ch} 把 {w} 个输入通道的这些实部加起来：{nt} 个乘积，和为 {y}。把一个模式放大或缩小，决定这个波留下多少；虚部则让它沿环移动。',
				vars: (rt) => {
					const t = T(rt, 'f0.mix'), ch = CH(rt), k = mixMode(rt);
					const which = k === 0
						? { en: 'mode 0, the average (no kept wave has any height here)', zh: '模式 0（平均值；这里保留的波高度都是 0）' }
						: { en: `mode ${k}, the tallest kept wave`, zh: `模式 ${k}（保留下来的最高的波）` };
					return { ...dims(rt), ch, which: words(which), nt: 2 * dims(rt).w, y: t ? t.data[(k * t.shape[1] + ch) * 2] : NaN };
				}
			},
			look: { en: 'Each row is one product: a·c, or b times −d, for one input channel.', zh: '每一行是一个乘积：某个输入通道的 a·c，或 b 乘以 −d。' },
			terms: ['weight', 'fourier-mode']
		},
		{
			id: 'back',
			source: 'fwd',
			at: { layer: 'f0.gelu' },
			view: view('back'),
			provenance: 'live',
			title: { en: 'Back to points, plus a pointwise path', zh: '变回点，再加上逐点通路' },
			narration: {
				en: 'The inverse FFT adds the mixed waves back up into {n} values per channel: a smooth curve, since only {left} left. A second path skips the Fourier part: a pointwise linear layer that mixes the {w} channels at each point. The two paths are added, and GELU bends the sum; at point {at}, channel {ch} goes from {sum} to {g}.',
				zh: '逆 FFT 把混合后的波加回去，每个通道变回 {n} 个值：因为只剩 {left}，所以是一条平滑的曲线。第二条通路绕开傅里叶部分：一个逐点的线性层，在每个点上混合 {w} 个通道。两条通路相加，再由 GELU 把和弯一弯；在点 {at}，通道 {ch} 从 {sum} 变成 {g}。',
				vars: (rt) => ({ ...dims(rt), left: plural(stateOf(rt)?.keep ?? NaN, { one: '{n} mode is', many: '{n} modes are', zh: '{n} 个模式' }), ch: CH(rt), ...geluAt(rt) })
			},
			look: { en: 'The dotted curve is the sum of the two paths; the blue curve is after GELU.', zh: '点线是两条通路之和；蓝色曲线是 GELU 之后的结果。' },
			terms: ['fourier-layer', 'activation-function']
		},
		{
			id: 'predict',
			source: 'fwd',
			at: { layer: 'proj' },
			view: view('compare'),
			provenance: 'live',
			title: { en: 'The prediction against the solver', zh: '预测与求解器对比' },
			narration: {
				en: 'A second Fourier layer repeats these steps with its own weights and no GELU, and a last pointwise layer turns the {w} channels into one height per point: the prediction, one pass instead of {steps} solver steps. Its largest difference from the solver is {err}, at point {at}. On {fact:testN} test curves the typical error was {fact:testRel|pct} of the curve’s size.',
				zh: '第二个傅里叶层用自己的权重重复这些步骤（没有 GELU），最后一个逐点层把 {w} 个通道变成每个点一个高度：这就是预测，只算一次，而不是求解器的 {steps} 步。它和求解器的最大差距是 {err}，在点 {at}。在 {fact:testN} 条测试曲线上，典型误差是曲线大小的 {fact:testRel|pct}。',
				vars: (rt) => ({ ...dims(rt), ...(stateOf(rt) ?? NaNs) })
			},
			look: { en: 'Green: the solver’s answer. Magenta dashes: the network’s, in one pass.', zh: '绿色：求解器的答案。品红色虚线：网络一次算出的答案。' },
			terms: ['neural-operator']
		},
		{
			id: 'filter',
			source: 'fwd',
			at: { layer: 'proj' },
			view: view('gain'),
			provenance: 'live',
			title: { en: 'What it learned: a filter', zh: '它学到的：一个滤波器' },
			narration: {
				en: 'Feed the network a single wave and measure how much of it comes out: that is its filter. Bars are the solver’s shares, dots the network’s: mode {k1} keeps {s1|n4} in the solver and {l1|n4} in the network, mode {k8} keeps {s8|n4} and {l8|n4}. Nobody told the network that short waves fade; it learned this filter from {fact:trainN} examples.',
				zh: '给网络输入单独一个波，量一量出来还剩多少：这就是它的滤波器。条形是求解器保留的比例，圆点是网络的：模式 {k1} 在求解器里保留 {s1|n4}，在网络里保留 {l1|n4}；模式 {k8} 分别是 {s8|n4} 和 {l8|n4}。没人告诉网络短波会消失；这个滤波器是它从 {fact:trainN} 个例子里学来的。',
				vars: (rt) => {
					const g = rt.model ? learnedGain(rt.model) : [];
					return { k1: 1, k8: 8, s1: fdGain(1), l1: g[1] ?? NaN, s8: fdGain(8), l8: g[8] ?? NaN };
				}
			},
			look: { en: 'The dots sit on the bars. The dashed line marks the last mode with weights.', zh: '圆点都落在条形上。虚线标出有权重的最后一个模式。' },
			terms: ['fourier-mode']
		},
		{
			id: 'modes',
			source: 'fwd',
			at: { layer: 'proj' },
			view: view('modes'),
			provenance: 'live',
			title: { en: 'How many modes does your curve need?', zh: '你的曲线需要多少个模式？' },
			narration: {
				en: 'The slider sets how many modes both Fourier layers keep. With {kept} the largest error is {err}. Fewer modes make a cheaper layer, but every low mode you drop is a wave the solver still keeps. The bars show the largest error for every choice, on a log scale; the dotted line is the target, {goal}.',
				zh: '滑块决定两个傅里叶层保留多少个模式。保留 {kept}时，最大误差是 {err}。模式越少，这一层越省，但每丢掉一个低模式，就丢掉一个求解器仍然保留的波。条形画出每种选择的最大误差（对数刻度）；点线是目标 {goal}。',
				vars: (rt) => ({ kept: plural(stateOf(rt)?.keep ?? NaN, { one: '{n} mode', many: '{n} modes', zh: '{n} 个模式' }), err: stateOf(rt)?.err ?? NaN, goal: fmtSetting(GOAL) })
			},
			look: { en: 'The bar in a different color is your choice; move the slider and watch the prediction.', zh: '颜色不同的那根条是你的选择；移动滑块，看预测怎么变。' },
			task: {
				prompt: { en: 'Move the slider to the fewest modes that keep the largest error below the dotted line.', zh: '把滑块移到能让最大误差低于点线的最少模式数。' },
				done: (rt) => {
					const s = stateOf(rt);
					// a drawing unlike the training curves may miss the target with every choice: then all the modes count as done
					const f = s && rt.model ? fewest(errByKeep(rt.model, s.u0)) : -1;
					return !!s && f >= 0 && s.keep === (f > 0 ? f : K);
				},
				waiting: { en: 'Not yet: find the first bar below the dotted line and move the slider there (if no bar is below it, keep all the modes).', zh: '还没有：找到点线下方的第一根条，把滑块移到那里（如果没有一根条低于点线，就保留全部模式）。' },
				doneText: { en: 'Done: the fewest modes that reach the target on your curve, or all of them when no choice does.', zh: '完成：这是你的曲线达到目标所需的最少模式数；如果哪种选择都达不到，就是全部模式。' },
				hint: { en: 'The arrow keys move the slider one mode at a time.', zh: '方向键每次把滑块移动一个模式。' }
			},
			terms: ['fourier-mode']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'proj' },
			view: view('limits'),
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'Limits: taller curves and a 64-point grid', zh: '局限：更高的曲线和 64 个点的网格' },
			narration: {
				en: 'Heat does not care how tall a curve is: {tall} times taller in, {tall} times taller out. The network never saw a curve taller than {amp}; your curve made {tall} times taller gets a largest error of {errT}, {vs}. And it learned the {n}-point solver, not the true equation: at mode {k8} the solver keeps {fd8} of the wave where the exact equation keeps {ex8}. Nothing finer than one grid point can be drawn at all.',
				zh: '热传导不在乎曲线有多高：输入高 {tall} 倍，输出也高 {tall} 倍。网络从没见过高于 {amp} 的曲线；把你的曲线加高 {tall} 倍，最大误差变成 {errT}，{vs}。而且它学的是 {n} 个点的求解器，不是真正的方程：在模式 {k8}，求解器保留这个波的 {fd8}，精确方程保留 {ex8}。比一个网格点更细的东西根本画不出来。',
				vars: (rt) => {
					const errT = taller(rt, TALL)?.err ?? NaN, err = stateOf(rt)?.err ?? NaN, r = errT / err;
					if (!Number.isFinite(r)) return { ...dims(rt), tall: TALL, amp: fmtSetting(AMP), errT, vs: NaN, k8: 8, fd8: fdGain(8), ex8: exactGain(8) };
					const e = n3(err), x = n3(r);
					// the error grows by more than the curve: the network did not learn that heat scales with height
					const vs = pick(r > TALL,
						{ en: `${x.en} times your own curve’s ${e.en}, though the curve grew only ${TALL} times`, zh: `是原曲线误差 ${e.zh} 的 ${x.zh} 倍，而曲线只加高了 ${TALL} 倍` },
						{ en: `against ${e.en} for your own curve`, zh: `原曲线是 ${e.zh}` });
					return { ...dims(rt), tall: TALL, amp: fmtSetting(AMP), errT, vs: words(vs), k8: 8, fd8: fdGain(8), ex8: exactGain(8) };
				}
			},
			look: { en: 'Your curve made taller: compare the network (magenta dashes) with the solver (green).', zh: '加高后的曲线：比较网络（品红色虚线）和求解器（绿色）。' },
			terms: ['out-of-distribution', 'finite-difference']
		}
	],
	fixtures: [
		{ id: 'square-keep4', label: { en: 'the square pulse with 4 modes kept', zh: '方波脉冲，保留 4 个模式' }, raw: { ic: 'square', y: presetY('square'), keep: 4 } },
		{ id: 'drawn', label: { en: 'the bump with a step drawn in', zh: '在鼓包上画了一个台阶' }, raw: { y: presetY('bump').map((v, i) => (i >= 8 && i < 20 ? v - 0.8 : v)) } }
	]
};

export default lesson;
