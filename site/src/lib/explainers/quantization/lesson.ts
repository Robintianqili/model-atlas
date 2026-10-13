// quantization: the guided lesson (eleven steps; the concept page's primary path). Every number is live: the ledger
// sources round the learner's own eight values (quant.ts), the sweep source quantizes TinyGPT's weights in the browser and
// scores the result; the PyTorch numbers on 64,000 tokens come from facts.json.
//   formats -> int8 -> predict (int4) -> int4 -> affine -> outlier (task, breakIt) -> fixes -> layer -> granularity
//   -> whole -> limits
import type { At, ExplainerRuntime, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { plural } from '../../atlas/lesson/say.ts';
import { fmtNum } from '../../atlas/core/format.ts';
import { formatsOf, hasOutlier, ledger, medianAbs, OUTLIER_MAKE, rawOf, WEIGHT_BLOCK } from './quant.ts';
import { FOCUS, matrixNames } from './process.ts';
import { PRESETS } from './adapters/values/adapter.ts';
import { show } from './views/formats/meta.ts';

const NaNs = new Proxy({} as Record<string, number>, { get: () => NaN });
/** The step's frame metrics (NaN for anything missing, so lint reports it). */
const M = (at: At): Record<string, number> => ({ ...NaNs, ...(at.frame?.metrics ?? {}) }) as Record<string, number>;
/** Metrics of another frame of the step's tape, by phase and step. */
function F(at: At, phase: string, step = 0): Record<string, number> {
	const i = at.tape?.find({ phase, step }) ?? -1;
	return i >= 0 ? (at.tape!.frame(i).metrics ?? {}) : {};
}
const valsAt = (at: At) => Array.from(((at.state as { x?: { data: ArrayLike<number> } } | undefined)?.x?.data ?? []) as ArrayLike<number>);
/** "3 nonzero values are" / "no nonzero value is" (zh: "有 3 个非零的值" / "没有非零的值"), before "stored as 0". */
const zerosText = (n: number) =>
	n === 0 ? { en: 'no nonzero value is', zh: '没有非零的值' } : plural(n, { one: '{n} nonzero value is', many: '{n} nonzero values are', zh: '有 {n} 个非零的值' });
const MB = (b: number) => b / 1e6;

/** The integer formats' sentence of step 'formats' (x outside −1..1 is clipped to the end of the range). */
function intsText(x: number, q8: number, q4: number, v8: string, v4: string): I18n {
	const inside = Math.abs(x) <= 1;
	return {
		en: `int8 and int4 store a whole-number code on a range you choose, here −1 to 1${inside ? '' : ', so x is clipped to the end of the range'}: the codes ${q8} and ${q4} stand for ${v8} and ${v4}.`.replace(/-(\d)/g, '−$1'),
		zh: `int8 和 int4 在你选定的范围（这里是 −1 到 1）上存一个整数编码${inside ? '' : '，x 超出范围，被截断到范围的端点'}：编码 ${q8} 和 ${q4} 分别代表 ${v8} 和 ${v4}。`.replace(/-(\d)/g, '−$1')
	};
}

/** TinyGPT's quantized layer: its size, largest |w| and standard deviation (from the trace source's weights). */
function focusStats(rt: ExplainerRuntime) {
	const w = rt.sources.gpt?.model?.weights?.[FOCUS]?.data;
	if (!w) return { n: NaN, amax: NaN, std: NaN };
	let s = 0;
	let amax = 0;
	for (let i = 0; i < w.length; i++) {
		s += w[i];
		amax = Math.max(amax, Math.abs(w[i]));
	}
	const mean = s / w.length;
	let v = 0;
	for (let i = 0; i < w.length; i++) v += (w[i] - mean) ** 2;
	return { n: w.length, amax, std: Math.sqrt(v / w.length) };
}

const lesson = defineLesson({
	title: { en: 'Quantization, step by step', zh: '量化，分步课程' },
	summary: {
		en: 'Look at one number bit by bit, round eight numbers to int8 and int4 by hand, watch one outlier wreck the scale, then quantize a real TinyGPT in your browser and measure what it costs.',
		zh: '逐位查看一个数，亲手把八个数舍入成 int8 和 int4，看一个离群值怎样毁掉缩放比例，再在浏览器里量化一个真实的 TinyGPT，测量它的代价。'
	},
	fixtures: [
		...PRESETS.slice(1).map((p) => ({ id: p.id, label: p.label, raw: { x: 0.1, vals: p.vals } })),
		{ id: 'big-x', label: { en: 'x too big for fp16', zh: 'x 超出 fp16 的范围' }, raw: { x: 70000, vals: PRESETS[0].vals } }
	],
	steps: [
		{
			id: 'formats',
			title: { en: 'One number, five formats', zh: '一个数，五种格式' },
			source: 'formats',
			at: 'first',
			view: { key: 'quantization/formats', props: () => ({ flip: true }) },
			provenance: 'live',
			terms: ['precision', 'floating-point'],
			narration: {
				en: 'A float stores a number as a sign, an exponent (how big) and a fraction (its leading digits). fp32 keeps x = {x} as {v32}; {f16} bf16 keeps the eight exponent bits of fp32 but only seven fraction bits: {vbf}. {ints}',
				zh: '浮点数把一个数存成三部分：符号、指数（决定大小）和尾数（有效数字）。fp32 把 x = {x} 存成 {v32}；{f16}bf16 保留了 fp32 的八个指数位，但只有七个尾数位：{vbf}。{ints}',
				vars: (_rt, at) => {
					const x = M(at).x;
					if (!Number.isFinite(x)) return { x: NaN, v32: NaN, f16: NaN, vbf: NaN, ints: NaN };
					const [f32, f16, bf, i8, i4] = formatsOf(x);
					const fin = Number.isFinite(f16.value);
					return {
						x: show(x, 8),
						v32: show(f32.value, 9),
						f16: fin
							? { en: `fp16, with half the bits, keeps ${show(f16.value, 6)}.`, zh: `fp16 只有一半的位数，存成 ${show(f16.value, 6)}。` }
							: { en: 'fp16 cannot reach it and stores infinity.', zh: 'fp16 表示不了这么大的数，存成了无穷大。' },
						vbf: show(bf.value, 6),
						ints: intsText(x, i8.code ?? NaN, i4.code ?? NaN, show(i8.value, 6), show(i4.value, 6))
					};
				}
			},
			look: { en: 'Press any fp32 bit in the stage to flip it: an exponent bit multiplies or divides x by a power of 2, a fraction bit changes it by less than a factor of 2.', zh: '按一下舞台里 fp32 的任意一位就能翻转它：翻转指数位会让 x 乘以或除以 2 的某个幂，翻转尾数位带来的变化不到 2 倍。' }
		},
		{
			id: 'int8',
			title: { en: 'Eight numbers, one scale: int8', zh: '八个数，一个缩放比例：int8' },
			source: 'int8',
			at: { phase: 'round', step: 0 },
			view: 'quantization/ledger',
			provenance: 'live',
			terms: ['quantization', 'quant-scale'],
			narration: {
				en: 'To store your eight values as int8 codes, absmax (absolute maximum) quantization gives them one shared scale. {scale} Each value becomes the code q = round(x / s). Value 0 (counting from 0) is {x}: {x} / {s} = {r}, which rounds to {q} and stands for q × s = {xh}, an error of {err}.',
				zh: '要把你的八个值存成 int8 编码，absmax（绝对值最大值）量化让它们共用一个缩放比例。{scale}每个值变成编码 q = round(x / s)。编号 0 的值（从 0 数起）是 {x}：{x} / {s} = {r}，舍入成 {q}，代表 q × s = {xh}，误差是 {err}。',
				vars: (_rt, at) => {
					const m = M(at);
					const f0 = F(at, 'scale');
					const amax = f0.amax ?? NaN;
					const scale: I18n =
						amax > 0
							? { en: `s = max|x| / ${f0.qmax} = ${fmtNum(amax)} / ${f0.qmax} = ${fmtNum(f0.s)}, so the largest value gets the code ${f0.qmax}.`, zh: `s = max|x| / ${f0.qmax} = ${fmtNum(amax)} / ${f0.qmax} = ${fmtNum(f0.s)}，于是最大的值得到编码 ${f0.qmax}。` }
							: { en: 'All your values are 0, so any scale works: the ledger uses s = 1.', zh: '你的值全是 0，任何缩放比例都行：账本用 s = 1。' };
					return { scale, x: m.x, s: m.s, r: m.r, q: m.q, xh: m.xh, err: m.err };
				}
			},
			look: { en: 'The outlined row is value 0. On the ruler each tick is one code, and every dot moves to its nearest tick.', zh: '带框的那一行是编号 0 的值。尺子上每个刻度是一个编码，每个点都移到离它最近的刻度。' }
		},
		{
			id: 'predict',
			title: { en: 'Fewer bits: guess the cost of int4', zh: '更少的位数：先猜猜 int4 的代价' },
			source: 'int4',
			at: 'last',
			view: { key: 'quantization/ledger', props: () => ({ alt: true }) },
			provenance: 'live',
			predict: {
				id: 'quantization-int4-step',
				question: {
					en: 'int4 has sixteen codes, and absmax uses the fifteen from −7 to 7. Compared with int8, how much wider is the step s between two codes?',
					zh: 'int4 有十六个编码，absmax 用的是 −7 到 7 这十五个。和 int8 相比，两个编码之间的步长 s 宽了多少？'
				},
				options: [
					{ en: 'About twice as wide', zh: '大约宽一倍' },
					{ en: 'About 18 times as wide', zh: '大约宽 18 倍' },
					{ en: 'About 128 times as wide', zh: '大约宽 128 倍' }
				],
				answer: 1,
				explain: {
					en: 's = max|x| / 7 instead of max|x| / 127, so each step is 127 / 7 ≈ 18 times wider. A value is never more than half a step from its code, so the largest possible error grows by the same factor.',
					zh: 's 从 max|x| / 127 变成 max|x| / 7，所以每一步宽了 127 / 7 ≈ 18 倍。一个值离它的编码最多半步，所以可能的最大误差也按同样的倍数增大。'
				}
			},
			narration: {
				en: 'Going from int8 to int4 halves the bits of every code. Your int8 ledger had a largest error of {e8}. Make your guess before the stage shows the int4 ledger next to it.',
				zh: '从 int8 换成 int4，每个编码的位数减半。你的 int8 账本最大误差是 {e8}。先做出猜测，舞台再把 int4 账本和它并排显示出来。',
				vars: (_rt, at) => ({ e8: M(at).altMaxErr })
			},
			look: { en: 'After you answer, compare the two rulers: the gap between neighbouring ticks is the step s.', zh: '回答之后，对比两把尺子：相邻刻度之间的距离就是步长 s。' }
		},
		{
			id: 'int4',
			title: { en: 'int4: wider steps, bigger errors', zh: 'int4：步长更宽，误差更大' },
			source: 'int4',
			at: 'last',
			view: { key: 'quantization/ledger', props: () => ({ alt: true }) },
			provenance: 'live',
			narration: {
				en: '{scales} The largest rounding error {grew}, and {zeros} now stored as 0. In exchange, each code takes half the bits of an int8 code.',
				zh: '{scales}最大舍入误差{grew}，现在{zeros}被存成 0。换来的是每个编码只占 int8 编码一半的位数。',
				vars: (_rt, at) => {
					const m = M(at);
					const [s4, s8, e4, e8] = [fmtNum(m.s), fmtNum(m.altS), fmtNum(m.maxErr), fmtNum(m.altMaxErr)];
					const allZero = !((F(at, 'scale').amax ?? NaN) > 0);
					const scales: I18n = allZero
						? { en: 'All your values are 0, so both ledgers use the placeholder scale s = 1.', zh: '你的值全是 0，所以两个账本都用占位的缩放比例 s = 1。' }
						: { en: `With int4 the scale is s = ${s4}, against ${s8} for int8: ${fmtNum(m.s / m.altS)} times wider.`, zh: `用 int4 时缩放比例是 s = ${s4}，而 int8 是 ${s8}：宽了 ${fmtNum(m.s / m.altS)} 倍。` };
					const grew: I18n =
						m.maxErr > m.altMaxErr
							? { en: `grew from ${e8} to ${e4}`, zh: `从 ${e8} 增大到 ${e4}` }
							: m.maxErr === 0
								? { en: `is ${e4}: every value sits exactly on an int4 code`, zh: `是 ${e4}：每个值都正好落在一个 int4 编码上` }
								: { en: `is ${e4}, against ${e8} for int8`, zh: `是 ${e4}，而 int8 是 ${e8}` };
					return Number.isFinite(m.s) && Number.isFinite(m.altS) && Number.isFinite(m.maxErr) && Number.isFinite(m.altMaxErr) ? { scales, grew, zeros: zerosText(m.zeros) } : { scales: NaN, grew: NaN, zeros: NaN };
				}
			},
			look: { en: 'On the two rulers the int4 ticks lie far apart. Column error holds int4’s errors, error′ int8’s.', zh: '两把尺子上，int4 的刻度隔得很远。“误差”一列是 int4 的误差，“误差′”一列是 int8 的。' }
		},
		{
			id: 'affine',
			title: { en: 'All-positive values: affine with a zero point', zh: '全是正数：带零点的仿射量化' },
			source: 'affine',
			at: 'last',
			view: { key: 'quantization/ledger', props: () => ({ alt: true }) },
			provenance: 'live',
			terms: ['zero-point', 'relu'],
			narration: {
				en: 'After a ReLU no value is below 0, so absmax wastes its negative codes. Affine quantization spreads all {levels} codes, 0 to {qhi}, over the range {lo} to {hi}: s = {s}. The zero point is the code that stands for 0; this range starts at 0, so it is the lowest code, z = {z}. On the sizes |x| of your values, the largest error is {eAff}, against {eAbs} with absmax.',
				zh: '经过 ReLU 之后没有值小于 0，absmax 的负数编码就浪费了。仿射量化把全部 {levels} 个编码（0 到 {qhi}）铺在 {lo} 到 {hi} 这个范围上：s = {s}。零点是代表 0 的那个编码；这个范围从 0 开始，所以零点就是最小的编码 z = {z}。对你这些值的绝对值 |x|，最大误差是 {eAff}，而 absmax 是 {eAbs}。',
				vars: (_rt, at) => {
					const m = M(at);
					const f0 = F(at, 'scale');
					return { levels: m.levels, qhi: m.levels - 1, lo: f0.lo ?? NaN, hi: f0.hi ?? NaN, s: m.s, z: f0.z ?? NaN, eAff: m.maxErr, eAbs: m.altMaxErr };
				}
			},
			look: { en: 'Two rulers: the affine ticks start at 0 and fill the range, while no value lands on a negative absmax tick.', zh: '两把尺子：仿射的刻度从 0 开始铺满整个范围，而 absmax 的负数刻度上一个值也没有。' }
		},
		{
			id: 'outlier',
			title: { en: 'One outlier wrecks the scale', zh: '一个离群值毁掉缩放比例' },
			source: 'outlier',
			at: 'last',
			view: 'quantization/ledger',
			provenance: 'live',
			kindOf: 'breakIt',
			terms: ['outlier'],
			task: {
				prompt: { en: 'Make one value far bigger than the others: type 6 into one of the eight fields, or start from the outlier set of values in the input widget.', zh: '让一个值比其他值大得多：在八个输入框之一里填 6，或者在输入控件里选用“离群值”那组值。' },
				done: (rt) => hasOutlier(rawOf(rt.input).vals),
				hint: { en: 'An outlier here is a value at least eight times the median size of the eight.', zh: '这里的离群值是指至少为八个值的中位数大小八倍的值。' },
				waiting: { en: 'Not yet: no value is eight times the median size.', zh: '还没有：没有哪个值达到中位数大小的八倍。' },
				doneText: { en: 'Done: one value now sets the scale for all eight.', zh: '完成：现在由一个值决定八个值共用的缩放比例。' }
			},
			narration: {
				en: 'One scale serves all eight values, so the largest one sets it. Your largest, {big} (value {k}, counting from 0), {cmp}. So s = {s}, {zeros} stored as 0, and the other seven values are off by {rest} on average.',
				zh: '八个值共用一个缩放比例，所以由最大的值决定。你最大的值是 {big}（编号 {k}，从 0 数起），{cmp}。于是 s = {s}，{zeros}被存成 0，其余七个值平均偏差 {rest}。',
				vars: (_rt, at) => {
					const m = M(at);
					const cmp: I18n =
						m.med > 0
							? { en: `is ${fmtNum(Math.abs(m.bigX) / m.med)} times the median size ${fmtNum(m.med)}`, zh: `是中位数大小 ${fmtNum(m.med)} 的 ${fmtNum(Math.abs(m.bigX) / m.med)} 倍` }
							: { en: 'stands beside values of which at least half are 0', zh: '而这些值里至少一半是 0' };
					return { big: m.bigX, k: m.big, cmp, s: m.s, zeros: zerosText(m.zeros), rest: m.restMean };
				}
			},
			look: { en: 'On the ruler the ticks spread out to reach the largest value, and the small values crowd around the tick at 0.', zh: '尺子上的刻度为了够到最大的值而拉得很开，小的值都挤在 0 那个刻度周围。' }
		},
		{
			id: 'fixes',
			title: { en: 'Two fixes: clip the range, or one scale per block', zh: '两种补救：截断范围，或每块一个缩放比例' },
			source: 'fixes',
			at: 'last',
			view: { key: 'quantization/ledger', props: () => ({ alt: true }) },
			provenance: 'live',
			terms: ['quant-granularity'],
			narration: {
				en: '{which} Clipping at the second-largest size, {clipAt}, gives the other seven a mean error of {cRest} but stores the outlier as {cOut}, off by {cBig}. Blocks of four give each half its own scale: the outlier coarsens only its own block, and the other seven are off by {bRest}. The price is a second scale, stored in fp32: {bpv} bits per value instead of {bpv1}.',
				zh: '{which}在第二大的绝对值 {clipAt} 处截断，其余七个值的平均误差是 {cRest}，但离群值被存成 {cOut}，偏差 {cBig}。改成每四个值一块、每块各用一个缩放比例：离群值只让自己那一块的刻度变粗，其余七个值平均偏差 {bRest}。代价是多存一个 fp32 的缩放比例：每个值 {bpv} 位，而不是 {bpv1} 位。',
				vars: (rt, at) => {
					const m = M(at);
					const x = valsAt(at);
					const C = x.length ? ledger(x, { bits: 4, clip: 'second' }) : null;
					const raw = rawOf(rt.input).vals;
					const [k, big] = [String(m.big), fmtNum(m.bigX)];
					const times = medianAbs(raw) > 0 ? { en: `, ${OUTLIER_MAKE} times the median size`, zh: `，是中位数大小的 ${OUTLIER_MAKE} 倍` } : { en: '', zh: '' };
					const which: I18n = hasOutlier(raw)
						? { en: `Your value ${k} (counting from 0), ${big}, is the outlier.`, zh: `编号 ${k} 的值（从 0 数起）${big} 就是离群值。` }
						: { en: `Your values have no outlier, so this step moves value ${k} (counting from 0) out to ${big}${times.en}.`, zh: `你的值里没有离群值，所以这一步把编号 ${k} 的值（从 0 数起）移到 ${big}${times.zh}。` };
					if (!Number.isFinite(m.big) || !Number.isFinite(m.bigX)) return { which: NaN, clipAt: NaN, cRest: NaN, cOut: NaN, cBig: NaN, bRest: NaN, bpv: NaN, bpv1: NaN };
					return { which, clipAt: C?.blocks[0].hi ?? NaN, cRest: m.altRestMean, cOut: C?.xh[m.big] ?? NaN, cBig: m.altBigErr, bRest: m.restMean, bpv: m.bitsPerValue, bpv1: m.altBits };
				}
			},
			look: { en: 'Column error is blocks of four, error′ is clipping: compare the outlier’s row with the other rows.', zh: '“误差”一列是每四个一块，“误差′”一列是截断：对比离群值那一行和其他行。' }
		},
		{
			id: 'layer',
			title: { en: 'A real layer: TinyGPT’s first MLP weights', zh: '真实的一层：TinyGPT 第一个 MLP 层的权重' },
			source: 'sweep',
			at: { phase: 'layer', step: 0 },
			view: { key: 'quantization/gpt', props: () => ({ hist: true, probe: false }) },
			provenance: 'live',
			terms: ['perplexity', 'weight'],
			narration: {
				en: 'TinyGPT’s layer L0.mlp.fc holds {n} weights, most of them near 0: the largest size, {amax}, is {ratio} times their standard deviation. With one int8 scale for the whole layer, the most likely next token (top-1) stays the same at {agree|pct} of {tokens} positions in four stories. Perplexity (roughly how many tokens the model hesitates between; lower is better) moves from {ppl0|n4} to {ppl|n4}, too small a change to trust on so few tokens: on {fact:bigTokens}, PyTorch measures {fact:pplLay8t|n4} against {fact:bigPpl0|n4}.',
				zh: 'TinyGPT 的 L0.mlp.fc 层有 {n} 个权重，大多数都在 0 附近：最大的绝对值 {amax} 是它们标准差的 {ratio} 倍。整层只用一个 int8 缩放比例时，在四个故事的 {tokens} 个位置中，有 {agree|pct} 的位置上最可能的下一个词元（top-1）保持不变。困惑度（大致是模型每一步在多少个词元之间犹豫，越低越好）从 {ppl0|n4} 变成 {ppl|n4}，这么小的变化在这么少的词元上说明不了什么：在 {fact:bigTokens} 个词元上，PyTorch 测得 {fact:pplLay8t|n4}，而 fp32 是 {fact:bigPpl0|n4}。',
				vars: (rt, at) => {
					const m = M(at);
					const w = focusStats(rt);
					return { n: w.n, amax: w.amax, ratio: w.amax / w.std, tokens: m.tokens, ppl0: m.ppl0, ppl: m.ppl, agree: m.agree };
				}
			},
			look: { en: 'The dashed lines are the int8 levels of one scale for the layer: so many that they nearly fill the plot, so every weight has a level close by.', zh: '虚线是整层一个缩放比例时 int8 的各个刻度：多到几乎铺满整张图，所以每个权重附近都有一个刻度。' }
		},
		{
			id: 'granularity',
			title: { en: 'int4 on one layer: per tensor, per channel, per block', zh: '一层用 int4：按张量、按通道、分块' },
			source: 'sweep',
			at: { phase: 'layer', step: 3 },
			view: { key: 'quantization/gpt', props: () => ({ table: 'layer', hist: true, probe: false }) },
			provenance: 'live',
			terms: ['quant-granularity', 'kl-divergence'],
			narration: {
				en: 'At int4 with one scale for the layer, perplexity jumps to {pT}, and the KL divergence from fp32, the average gap between the two next-token distributions, is {klT}. One scale per output channel, a row of {cols} weights, brings perplexity to {pC} (KL {klC}); one per block of {blk} weights, to {pB} (KL {klB}). Smaller groups fit the sizes of their own weights.',
				zh: '用 int4 且整层一个缩放比例时，困惑度跳到 {pT}，与 fp32 的 KL 散度（两个下一词元分布之间的平均差距）是 {klT}。每个输出通道（一行 {cols} 个权重）一个缩放比例，困惑度降到 {pC}（KL {klC}）；每 {blk} 个权重一块，则是 {pB}（KL {klB}）。组越小，刻度越贴合组内权重的大小。',
				vars: (_rt, at) => {
					const t = F(at, 'layer', 1);
					const c = F(at, 'layer', 2);
					const b = F(at, 'layer', 3);
					return { pT: t.ppl ?? NaN, klT: t.kl ?? NaN, cols: M(at).fCols, pC: c.ppl ?? NaN, klC: c.kl ?? NaN, blk: WEIGHT_BLOCK, pB: b.ppl ?? NaN, klB: b.kl ?? NaN };
				}
			},
			look: { en: 'The table has a row per setting for this one layer; the dashed lines now mark the fifteen int4 levels of one scale.', zh: '表格里这一层的每种设置各占一行；虚线现在标出的是整层一个缩放比例时 int4 的十五个刻度。' }
		},
		{
			id: 'whole',
			title: { en: 'The whole model: int8 or int4, and the size', zh: '整个模型：int8 还是 int4，以及大小' },
			source: 'sweep',
			at: 'last',
			view: { key: 'quantization/gpt', props: () => ({ table: 'all', probe: false }) },
			provenance: 'live',
			terms: ['embedding'],
			narration: {
				en: 'Now all {nm} weight matrices are quantized. int8 per channel keeps the KL divergence at {kl8c} and shrinks the model from {mb0} MB to {mb8c} MB. int4 per block of {blk} costs more, KL {kl4b} and perplexity {p4b} against {ppl0}, for {mb4b} MB. Not eight times smaller than fp32: the token embedding, {emb|pct} of the parameters, stays fp32 on this page.',
				zh: '现在全部 {nm} 个权重矩阵都被量化了。int8 按通道时 KL 散度只有 {kl8c}，模型从 {mb0} MB 缩小到 {mb8c} MB。int4 每 {blk} 个一块代价更大：KL {kl4b}，困惑度 {p4b}（fp32 为 {ppl0}），换来 {mb4b} MB。并没有小到 fp32 的八分之一：占全部参数 {emb|pct} 的词元嵌入在本页仍保持 fp32。',
				vars: (rt, at) => {
					const c8 = F(at, 'all', 1);
					const b4 = F(at, 'all', 5);
					const w = rt.sources.gpt?.model?.weights;
					const emb = w?.['tok_emb.weight']?.data.length ?? NaN;
					return {
						nm: w ? matrixNames(w).length : NaN,
						kl8c: c8.kl ?? NaN,
						mb0: MB(M(at).bytes0),
						mb8c: MB(c8.bytes ?? NaN),
						blk: WEIGHT_BLOCK,
						kl4b: b4.kl ?? NaN,
						p4b: b4.ppl ?? NaN,
						ppl0: M(at).ppl0,
						mb4b: MB(b4.bytes ?? NaN),
						emb: emb / M(at).params
					};
				}
			},
			look: { en: 'Read the size and KL columns together: every smaller file in the table pays in KL.', zh: '把“模型大小”和“KL 散度”两列对照着看：表里文件越小，KL 越大。' }
		},
		{
			id: 'limits',
			title: { en: 'Where it breaks, and what this page does not show', zh: '哪里会坏，以及本页没有展示什么' },
			source: 'sweep',
			at: { phase: 'all', step: 3 },
			view: 'quantization/gpt',
			provenance: 'live',
			kindOf: 'limits',
			narration: {
				en: 'int4 with one scale per matrix breaks TinyGPT: perplexity goes from {ppl0} to {ppl}, and the top-1 next token matches fp32 at only {agree|pct} of the positions. These live numbers come from {tokens} tokens; on {fact:bigTokens} tokens, PyTorch measures {fact:pplAll4t} for the same setting, against {fact:bigPpl0} in fp32. Each code is turned back into a float before it is multiplied, so the page shows the cost in accuracy, not a speed-up.',
				zh: 'int4 且每个矩阵只用一个缩放比例时，TinyGPT 坏掉了：困惑度从 {ppl0} 变成 {ppl}，只有 {agree|pct} 的位置上 top-1 下一个词元与 fp32 相同。这些实时数字来自 {tokens} 个词元；在 {fact:bigTokens} 个词元上，PyTorch 测得同一设置的困惑度是 {fact:pplAll4t}，而 fp32 是 {fact:bigPpl0}。每个编码在相乘之前都先换回浮点数，所以本页展示的是精度上的代价，而不是速度上的提升。',
				vars: (_rt, at) => {
					const m = M(at);
					return { ppl0: m.ppl0, ppl: m.ppl, agree: m.agree, tokens: m.tokens };
				}
			},
			look: { en: 'In the probe, the line after the bars names the token the quantized model now puts first.', zh: '在探针句里，柱状图后面的那一行写出了量化后的模型现在排第一的词元。' }
		}
	]
});

export default lesson;
