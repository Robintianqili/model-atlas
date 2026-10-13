// number-formats: the guided lesson (eleven steps; the concept page's primary path). Every number is live: the formats
// sources store the learner's x (or a fixed x) in nine formats (fmt.ts), the sweep source casts TinyGPT's weights in the
// browser and scores the result, the sum source adds small updates in fp32, bf16 and fp16; the gradient numbers come
// from facts.json (pipeline/number-formats/prepare.py).
//   fields -> gap -> predict (70,000 in 16 bits) -> range16 (task) -> fp8 -> tiny -> ints -> weights -> cast -> mixed
//   -> limits
import type { At, ExplainerRuntime, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmt3, fmtPct, fmtSetting } from '../../atlas/core/format.ts';
import { cast, finiteCount, FMTS, gapAtOne, INT_RANGE, largest, qmaxOf, rawOf, rowOf, smallest, smallestNormal, type FloatFmt, type FmtId } from './fmt.ts';
import { CASTS, castIndex } from './process.ts';
import { DIGITS, show, sup } from './views/bits/meta.ts';

const NaNs = new Proxy({} as Record<string, number>, { get: () => NaN });
/** The step's frame metrics (NaN for anything missing, so lint reports it). */
const M = (at: At): Record<string, number> => ({ ...NaNs, ...(at.frame?.metrics ?? {}) }) as Record<string, number>;
/** Metrics of another frame of the step's tape, by phase and step (NaN for anything missing). */
function F(at: At, phase: string, step = 0): Record<string, number> {
	const i = at.tape?.find({ phase, step }) ?? -1;
	return { ...NaNs, ...(i >= 0 ? (at.tape!.frame(i).metrics ?? {}) : {}) } as Record<string, number>;
}
const fl = (id: FmtId) => FMTS[id] as FloatFmt;
/** A number in a format's own precision, as text. */
const sv = (id: FmtId, v: number) => show(v, DIGITS[id]);
/** What a format stores for x, as text: its value, or what happened instead (infinity, not a number). In zh a number
 * starts with a space and a word does not, so '存成{v}' reads right for both. */
function stored(id: FmtId, x: number): I18n | number {
	if (!Number.isFinite(x)) return NaN;
	const v = cast(id, x);
	if (Number.isNaN(v)) return { en: 'the code for “not a number”', zh: '“非数”' };
	if (!Number.isFinite(v)) return { en: v > 0 ? 'infinity (an overflow)' : 'minus infinity (an overflow)', zh: v > 0 ? '无穷大（上溢）' : '负无穷大（上溢）' };
	if (v === 0 && x !== 0) return { en: '0 (an underflow)', zh: ' 0（下溢）' };
	// The zh templates put no space before a stored value: numbers bring their own, Chinese words need none.
	return { en: sv(id, v), zh: ` ${sv(id, v)}` };
}
/** A relative error |v − x| / |x| (NaN for x = 0 or a value that is not finite). */
const relErr = (id: FmtId, x: number) => {
	const v = cast(id, x);
	return x !== 0 && Number.isFinite(v) ? Math.abs(v - x) / Math.abs(x) : NaN;
};
/** A relative error as a percent with three significant digits (fmtPct would print <0.1%). */
const pct3 = (r: number) => (Number.isFinite(r) ? `${fmt3(r * 100)}%` : NaN);
const noX = <T extends string>(keys: T[]) => Object.fromEntries(keys.map((k) => [k, NaN])) as Record<T, number>;

const lesson = defineLesson({
	title: { en: 'Number formats, step by step', zh: '数值格式，分步课程' },
	summary: {
		en: 'Click the bits of one number in fp32, fp16, bf16, FP8, FP4, int8, int4 and NF4, see where each format runs out of range or precision, cast a real TinyGPT to each, and see why mixed precision keeps fp32 master weights.',
		zh: '点击一个数在 fp32、fp16、bf16、FP8、FP4、int8、int4 和 NF4 中的各个比特位，看每种格式在哪里耗尽范围或精度，把真实的 TinyGPT 换成每种格式，再看混合精度为什么要保留 fp32 主权重。'
	},
	fixtures: [
		{ id: 'big-x', label: { en: 'x too big for fp16', zh: 'x 超出 fp16 的范围' }, raw: { x: 70000 } },
		{ id: 'third', label: { en: 'x = 1/3', zh: 'x = 1/3' }, raw: { x: 1 / 3 } },
		{ id: 'gradient', label: { en: 'a gradient-sized x', zh: '梯度大小的 x' }, raw: { x: 3e-6 } }
	],
	steps: [
		{
			id: 'fields',
			title: { en: 'One number, three fields', zh: '一个数，三个字段' },
			source: 'formats',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp32'], detail: 'fp32', flip: true }) },
			provenance: 'live',
			terms: ['floating-point', 'precision'],
			narration: {
				en: 'fp32 stores your x = {x} in {n} bits: one sign bit, {e} exponent bits and {m} fraction bits. {how} Together they make {v}, the number fp32 actually keeps.',
				zh: 'fp32 用 {n} 位存下你的 x = {x}：一个符号位、{e} 个指数位和 {m} 个尾数位。{how}合起来就是 {v}，这才是 fp32 真正存下的数。',
				vars: (_rt, at) => {
					const x = M(at).x;
					if (!Number.isFinite(x)) return noX(['x', 'n', 'e', 'm', 'how', 'v']);
					const f = fl('fp32');
					const r = rowOf('fp32', x);
					const sig = show(r.sig ?? NaN, 9);
					const how: I18n =
						r.E === 0
							? r.value === 0
								? { en: 'Every exponent and fraction bit is 0, which means zero.', zh: '指数位和尾数位全是 0，表示零。' }
								: { en: `The exponent field is 0, so there is no leading 1: a subnormal, 2${sup(1 - f.bias)} × ${sig}.`, zh: `指数字段是 0，所以开头没有那个 1：这是一个次正规数，2${sup(1 - f.bias)} × ${sig}。` }
							: {
									en: `The exponent field reads ${r.E}; minus the bias ${f.bias}, that is a factor 2${sup(r.exp ?? NaN)}. The fraction bits add ${show((r.sig ?? NaN) - 1, 9)} to a leading 1, giving ${sig}.`,
									zh: `指数字段读作 ${r.E}，减去偏置 ${f.bias}，得到因子 2${sup(r.exp ?? NaN)}。尾数位在开头的 1 上再加 ${show((r.sig ?? NaN) - 1, 9)}，得到 ${sig}。`
								};
					return { x: show(x, 9), n: 1 + f.e + f.m, e: f.e, m: f.m, how, v: show(r.value, 9) };
				}
			},
			look: { en: 'Press any bit in the stage to flip it: the colours mark the sign, the exponent and the fraction.', zh: '按一下舞台里的任意一位就能翻转它：颜色分别标出符号、指数和尾数。' }
		},
		{
			id: 'gap',
			title: { en: 'The gap to the next number', zh: '到下一个数的间隔' },
			source: 'formats',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp32', 'fp16'], detail: 'fp32', flip: true }) },
			provenance: 'live',
			terms: ['ulp'],
			narration: {
				en: 'Each power of 2 holds the same {steps} fp32 numbers, so the gap between neighbours grows with the number. {gap} {err} {g16}',
				zh: '每个 2 的幂区间里都有同样多的 {steps} 个 fp32 数，所以相邻两数之间的间隔随数值增大。{gap}{err}{g16}',
				vars: (_rt, at) => {
					const x = M(at).x;
					if (!Number.isFinite(x)) return noX(['steps', 'gap', 'err', 'g16']);
					const r = rowOf('fp32', x);
					const r16 = rowOf('fp16', x);
					const v = sv('fp32', r.value);
					const gap: I18n =
						r.value === 0
							? { en: `x is 0; the nearest nonzero fp32 number is ${show(smallest('fp32'), 3)}.`, zh: `x 是 0；离它最近的非零 fp32 数是 ${show(smallest('fp32'), 3)}。` }
							: Number.isFinite(r.gap)
								? { en: `Next to ${v}, away from 0, the gap is ${show(r.gap, 3)}: ${fmt3(r.gap / Math.abs(r.value))} times the number.`, zh: `在 ${v} 旁边（远离 0 的一侧），间隔是 ${show(r.gap, 3)}：是这个数的 ${fmt3(r.gap / Math.abs(r.value))} 倍。` }
								: { en: `${v} is fp32’s largest number: nothing comes after it.`, zh: `${v} 是 fp32 最大的数：后面没有数了。` };
					const e = Math.abs(r.value - x);
					const err: I18n =
						e === 0
							? { en: 'Your x sits exactly on an fp32 number, so it is stored without error.', zh: '你的 x 正好落在一个 fp32 数上，所以存下来没有误差。' }
							: { en: `Your x is rounded to the nearer neighbour, off by ${show(e, 3)}, never more than half a gap.`, zh: `你的 x 被舍入到较近的那个邻居，偏差 ${show(e, 3)}，永远不超过半个间隔。` };
					const g16: I18n = !Number.isFinite(r16.value)
						? { en: 'fp16 cannot hold x at all: it overflows.', zh: 'fp16 根本存不下 x：它上溢了。' }
						: r16.value === 0 && x !== 0
							? { en: 'fp16 cannot hold x at all: it underflows to 0.', zh: 'fp16 根本存不下 x：它下溢成 0。' }
							: Number.isFinite(r16.gap)
								? { en: `fp16, with ${fl('fp16').m} fraction bits, has a gap of ${show(r16.gap, 3)} here.`, zh: `fp16 只有 ${fl('fp16').m} 个尾数位，在这里的间隔是 ${show(r16.gap, 3)}。` }
								: { en: `In fp16, x is the largest number, ${sv('fp16', r16.value)}.`, zh: `在 fp16 里，x 是最大的数 ${sv('fp16', r16.value)}。` };
					return { steps: `2${sup(fl('fp32').m)}`, gap, err, g16 };
				}
			},
			look: { en: 'The ruler in the stage marks the stored value, its two neighbours and x itself.', zh: '舞台里的尺子标出了存下的值、它两侧的邻居，以及 x 本身。' }
		},
		{
			id: 'predict',
			title: { en: 'Two 16-bit formats: guess first', zh: '两种 16 位格式：先猜一猜' },
			source: 'big',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp32', 'fp16', 'bf16'] }) },
			provenance: 'live',
			predict: {
				id: 'number-formats-16bit',
				question: {
					en: 'Store 70,000 in fp16 (1 + 5 + 10 bits) and in bf16 (1 + 8 + 7 bits). What happens?',
					zh: '把 70,000 分别存进 fp16（1 + 5 + 10 位）和 bf16（1 + 8 + 7 位）。会怎样？'
				},
				options: [
					{ en: 'Both store about 70,000, rounded a little', zh: '两者都存成约 70,000，只是稍有舍入' },
					{ en: 'fp16 overflows to infinity; bf16 stores about 70,000', zh: 'fp16 上溢成无穷大；bf16 存成约 70,000' },
					{ en: 'bf16 overflows to infinity; fp16 stores about 70,000', zh: 'bf16 上溢成无穷大；fp16 存成约 70,000' }
				],
				answer: 1,
				explain: {
					en: 'fp16’s five exponent bits end at 65,504, so 70,000 becomes infinity. bf16 has fp32’s eight exponent bits, so 70,000 is far inside its range, but with seven fraction bits its steps there are 512 wide: it stores 70,144.',
					zh: 'fp16 的五个指数位最大只到 65,504，所以 70,000 变成了无穷大。bf16 有和 fp32 一样的八个指数位，70,000 远在它的范围之内，但只有七个尾数位，在这里的步长是 512：它存成 70,144。'
				}
			},
			narration: {
				en: 'This step uses a fixed number, x = {x}. Both {n16}-bit formats keep one sign bit; fp16 splits the rest into {e16} exponent and {m16} fraction bits, bf16 into {ebf} and {mbf}. Decide what each one stores before the stage shows it.',
				zh: '这一步用一个固定的数 x = {x}。两种 {n16} 位格式都留一个符号位；fp16 把其余的位分成 {e16} 个指数位和 {m16} 个尾数位，bf16 分成 {ebf} 个和 {mbf} 个。先判断每种格式会存下什么，再看舞台。',
				vars: (_rt, at) => ({ x: show(M(at).x, 9), n16: 1 + fl('fp16').e + fl('fp16').m, e16: fl('fp16').e, m16: fl('fp16').m, ebf: fl('bf16').e, mbf: fl('bf16').m })
			},
			look: { en: 'After you answer, compare the fp16 and bf16 rows with the fp32 row.', zh: '回答之后，把 fp16 和 bf16 两行与 fp32 那一行对比。' }
		},
		{
			id: 'range16',
			title: { en: 'Range or precision: fp16 against bf16', zh: '范围还是精度：fp16 对比 bf16' },
			source: 'formats',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp32', 'fp16', 'bf16'], flip: true }) },
			provenance: 'live',
			terms: ['overflow'],
			task: {
				prompt: {
					en: 'Make fp16 overflow while bf16 still holds x: flip a high exponent bit of x in the stage, or type a number above fp16’s largest in the input widget.',
					zh: '让 fp16 上溢、而 bf16 仍能存下 x：在舞台里翻转 x 的一个高位指数位，或者在输入控件里填一个比 fp16 最大值还大的数。'
				},
				done: (rt: ExplainerRuntime) => {
					const x = rawOf(rt.input).x;
					return !Number.isFinite(cast('fp16', x)) && Number.isFinite(cast('bf16', x));
				},
				hint: { en: 'In the fp32 row, bit 1 (counting from 0) is the highest exponent bit.', zh: '在 fp32 那一行，第 1 位（从 0 数起）是最高的指数位。' },
				waiting: { en: 'Not yet: fp16 still holds x.', zh: '还没有：fp16 仍然存得下 x。' },
				doneText: { en: 'Done: fp16 overflows to infinity, bf16 still stores x.', zh: '完成：fp16 上溢成无穷大，bf16 仍然存得下 x。' }
			},
			narration: {
				en: 'Exponent bits buy range; fraction bits buy precision. For your x, fp16 stores {v16} and bf16 stores {vbf}. Near 1, fp16’s steps are {ratio} times finer, but it ends at {max16}, while bf16 reaches {maxbf}, about as far as fp32.',
				zh: '指数位换来范围，尾数位换来精度。对你的 x，fp16 存成{v16}，bf16 存成{vbf}。在 1 附近，fp16 的步长细 {ratio} 倍，但最大只到 {max16}；bf16 能到 {maxbf}，和 fp32 差不多远。',
				vars: (_rt, at) => {
					const x = M(at).x;
					return {
						v16: stored('fp16', x),
						vbf: stored('bf16', x),
						ratio: gapAtOne('bf16') / gapAtOne('fp16'),
						max16: largest('fp16'),
						maxbf: largest('bf16')
					};
				}
			},
			look: { en: 'Compare the error of the fp16 and bf16 rows, then push x past fp16’s largest number.', zh: '对比 fp16 和 bf16 两行的误差，然后把 x 推过 fp16 的最大值。' }
		},
		{
			id: 'fp8',
			title: { en: 'Eight bits and four: FP8 and FP4', zh: '八位和四位：FP8 与 FP4' },
			source: 'formats',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['e4m3', 'e5m2', 'fp4'], detail: 'e4m3', flip: true }) },
			provenance: 'live',
			terms: ['fp8'],
			narration: {
				en: 'Eight bits split two ways. e4m3 keeps {m43} fraction bits, a gap of {g43} at 1, and ends at {max43}; e5m2 gives one fraction bit to the exponent: a gap of {g52}, up to {max52}. In e4m3 your x becomes {v43}; in e5m2, {v52}. FP4 e2m1 has only {n4} values, the largest {max4}: x becomes {v4}.',
				zh: '八个位有两种分法。e4m3 保留 {m43} 个尾数位，1 处的间隔是 {g43}，最大到 {max43}；e5m2 把一个尾数位让给指数：间隔 {g52}，最大到 {max52}。你的 x 在 e4m3 里变成{v43}，在 e5m2 里变成{v52}。FP4 e2m1 只有 {n4} 个值，最大是 {max4}：x 变成{v4}。',
				vars: (_rt, at) => {
					const x = M(at).x;
					return {
						m43: fl('e4m3').m,
						g43: gapAtOne('e4m3'),
						max43: largest('e4m3'),
						g52: gapAtOne('e5m2'),
						max52: largest('e5m2'),
						v43: stored('e4m3', x),
						v52: stored('e5m2', x),
						n4: finiteCount('fp4'),
						max4: largest('fp4'),
						v4: stored('fp4', x)
					};
				}
			},
			look: { en: 'Flip the e4m3 bits in the stage: with so few fraction bits, the value jumps in coarse steps.', zh: '在舞台里翻转 e4m3 的各位：尾数位这么少，数值一跳就是一大步。' }
		},
		{
			id: 'tiny',
			title: { en: 'Small numbers: subnormals and underflow', zh: '很小的数：次正规数与下溢' },
			source: 'tiny',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp16', 'bf16', 'e4m3', 'e5m2', 'fp4'], detail: 'e4m3' }) },
			provenance: 'live',
			terms: ['subnormal'],
			narration: {
				en: 'Small numbers meet the other edge. Below a format’s smallest normal number, {min43} in e4m3, the exponent field is 0 and the leading 1 is gone: a subnormal, with fewer bits left. For a fixed x = {x}, e4m3 stores {v43}, off by {rel43|pct}, and FP4 stores {v4}. fp16 and bf16 are off by only {e16} and {ebf}.',
				zh: '很小的数会碰到另一条边界。低于格式的最小正规数（e4m3 是 {min43}）时，指数字段为 0，开头的 1 没有了：这就是次正规数，剩下的有效位更少。对固定的 x = {x}，e4m3 存成{v43}，偏差 {rel43|pct}；FP4 存成{v4}。fp16 和 bf16 的偏差只有 {e16} 和 {ebf}。',
				vars: (_rt, at) => {
					const x = M(at).x;
					return { min43: smallestNormal('e4m3'), x: show(x, 9), v43: stored('e4m3', x), rel43: relErr('e4m3', x), v4: stored('fp4', x), e16: pct3(relErr('fp16', x)), ebf: pct3(relErr('bf16', x)) };
				}
			},
			look: { en: 'The worked line under the rows shows e4m3’s exponent field at 0: no leading 1.', zh: '各行下面的算式显示 e4m3 的指数字段为 0：开头没有那个 1。' }
		},
		{
			id: 'ints',
			title: { en: 'No exponent: int8, int4 and NF4', zh: '没有指数：int8、int4 与 NF4' },
			source: 'formats',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['int8', 'int4', 'nf4'], detail: 'int8', flip: true }) },
			provenance: 'live',
			terms: ['quant-scale', 'nf4'],
			narration: {
				en: 'Integer codes have no exponent: a code q stands for q × s, so the steps are even, and the scale s sets the range, here −{R} to {R} (s = 1/{q8} for int8, 1/{q4} for int4). NF4’s {nnf} levels crowd near 0, where most weights lie. Your x: int8 {v8}, int4 {v4}, NF4 {vnf}. {more}',
				zh: '整数编码没有指数：编码 q 代表 q × s，所以步长均匀，由缩放比例 s 决定范围，这里是 −{R} 到 {R}（int8 的 s = 1/{q8}，int4 的 s = 1/{q4}）。NF4 的 {nnf} 个刻度挤在 0 附近，正是大多数权重所在的地方。你的 x：int8 {v8}，int4 {v4}，NF4 {vnf}。{more}',
				vars: (_rt, at) => {
					const x = M(at).x;
					if (!Number.isFinite(x)) return noX(['R', 'q8', 'q4', 'nnf', 'v8', 'v4', 'vnf', 'more']);
					const e = (id: FmtId) => show(Math.abs(cast(id, x) - x), 3);
					const more: I18n =
						Math.abs(x) > INT_RANGE
							? { en: 'x is outside the range, so all three clip it to the end.', zh: 'x 超出了范围，所以三者都把它截断到端点。' }
							: { en: `The errors: ${e('int8')}, ${e('int4')} and ${e('nf4')}.`, zh: `误差分别是 ${e('int8')}、${e('int4')} 和 ${e('nf4')}。` };
					return { R: INT_RANGE, q8: qmaxOf(8), q4: qmaxOf(4), nnf: finiteCount('nf4'), v8: sv('int8', cast('int8', x)), v4: sv('int4', cast('int4', x)), vnf: sv('nf4', cast('nf4', x)), more };
				}
			},
			look: { en: 'Flip the int8 code’s bits in the stage: each bit adds or removes a fixed multiple of s.', zh: '在舞台里翻转 int8 编码的各位：每一位都增加或减少 s 的固定倍数。' }
		},
		{
			id: 'weights',
			title: { en: 'Real numbers: TinyGPT’s weights', zh: '真实的数：TinyGPT 的权重' },
			source: 'sweep',
			at: { phase: 'fp32', step: 0 },
			view: { key: 'number-formats/range', props: () => ({ formats: ['fp32', 'fp16', 'bf16', 'e5m2', 'e4m3', 'fp4'] }) },
			provenance: 'live',
			terms: ['weight'],
			narration: {
				en: 'Now real numbers: the {n} weights of TinyGPT’s {nm} weight matrices, counted by size in the stage. Their median size is {med} and the largest {amax}. Without a scale, e4m3 stores {sub43|pct} of them as subnormals and rounds {zero43|pct} to 0; FP4 rounds {zero4|pct} to 0. {wide}',
				zh: '现在看真实的数：TinyGPT 的 {nm} 个权重矩阵共有 {n} 个权重，舞台里按大小统计。它们大小的中位数是 {med}，最大是 {amax}。不加缩放时，e4m3 把其中 {sub43|pct} 存成次正规数，把 {zero43|pct} 舍入成 0；FP4 把 {zero4|pct} 舍入成 0。{wide}',
				vars: (_rt, at) => {
					const m = M(at);
					const lost = m.zero_fp16 + m.zero_bf16;
					const wide: I18n | number = !Number.isFinite(lost)
						? NaN
						: lost === 0
							? { en: 'fp16 and bf16 round none of them to 0.', zh: 'fp16 和 bf16 一个也没有舍入成 0。' }
							: { en: `fp16 rounds ${fmtPct(m.zero_fp16)} to 0, bf16 ${fmtPct(m.zero_bf16)}.`, zh: `fp16 把 ${fmtPct(m.zero_fp16)} 舍入成 0，bf16 是 ${fmtPct(m.zero_bf16)}。` };
					return { n: m.n, nm: m.nm, med: m.wMed, amax: m.amax, sub43: m.sub_e4m3, zero43: m.zero_e4m3, zero4: m.zero_fp4, wide };
				}
			},
			look: { en: 'The histogram in the stage counts TinyGPT’s weights per power of 2: compare it with where each format’s normal range starts.', zh: '舞台里的直方图按 2 的幂统计 TinyGPT 的权重个数：把它和每种格式正规数范围的起点对比。' }
		},
		{
			id: 'cast',
			title: { en: 'A real model in every format', zh: '一个真实的模型，换成每种格式' },
			source: 'sweep',
			at: 'last',
			view: { key: 'number-formats/gpt', props: () => ({ table: true, probe: false }) },
			provenance: 'live',
			terms: ['perplexity'],
			narration: {
				en: 'Cast every weight matrix and score TinyGPT on {tokens} tokens of four stories: fp32 has a perplexity of {ppl0|n4}. bf16 and fp16 change it by only {dbf|signed} and {d16|signed}; e4m3 and e5m2 give {p43|n4} and {p52|n4}. FP4 without a scale wrecks it: {p4}. A scale per block of {b32} brings FP4 back to {p4b}; with a scale per block of {b64}, NF4 gives {pnf} and int4 {pi4}.',
				zh: '把每个权重矩阵换成某种格式，在四个故事的 {tokens} 个词元上评测 TinyGPT：fp32 的困惑度是 {ppl0|n4}。bf16 和 fp16 只让它变化了 {dbf|signed} 和 {d16|signed}；e4m3 和 e5m2 分别是 {p43|n4} 和 {p52|n4}。不加缩放的 FP4 把模型彻底搞坏：{p4}。每 {b32} 个一块、各配缩放比例，FP4 回到 {p4b}；每 {b64} 个一块时，NF4 是 {pnf}，int4 是 {pi4}。',
				vars: (_rt, at) => {
					const p = (id: string) => F(at, 'cast', castIndex(id)).ppl;
					return { tokens: M(at).tokens, ppl0: M(at).ppl0, dbf: p('bf16') - M(at).ppl0, d16: p('fp16') - M(at).ppl0, p43: p('e4m3'), p52: p('e5m2'), p4: p('fp4'), b32: CASTS[castIndex('fp4-b')].block ?? NaN, p4b: p('fp4-b'), b64: CASTS[castIndex('nf4-b')].block ?? NaN, pnf: p('nf4-b'), pi4: p('int4-b') };
				}
			},
			look: { en: 'Read the perplexity column against the bits-per-weight column: every scale is one more fp32 number to store.', zh: '把“困惑度”一列和“每个权重的位数”一列对照着看：每个缩放比例都要多存一个 fp32 数。' }
		},
		{
			id: 'mixed',
			title: { en: 'Why mixed precision keeps fp32 master weights', zh: '混合精度为什么要保留 fp32 主权重' },
			source: 'sum',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: () => ({
					metrics: ['d32', 'dbf', 'd16', 'dmix'],
					labels: {
						d32: { en: 'fp32: change', zh: 'fp32：变化量' },
						dbf: { en: 'bf16: change', zh: 'bf16：变化量' },
						d16: { en: 'fp16: change', zh: 'fp16：变化量' },
						dmix: { en: 'bf16 copy of an fp32 master: change', zh: 'fp32 主权重的 bf16 副本：变化量' }
					}
				})
			},
			provenance: 'live',
			terms: ['mixed-precision'],
			narration: {
				en: 'Training adds many small updates to each weight. Here a weight of {w0} gets {n} updates of {u}; in fp32 it ends at {w32}. In bf16 the gap at 1 is {gbf}, so every update, less than half a gap, rounds away and the weight stays at {wbf}. Mixed precision keeps an fp32 master copy that collects the updates; its bf16 copy ends at {wmix}. fp16, with finer steps, reaches {w16}{why16}.',
				zh: '训练会给每个权重加上许多次小的更新。这里一个值为 {w0} 的权重加了 {n} 次 {u}；用 fp32 时最终是 {w32}。bf16 在 1 处的间隔是 {gbf}，每次更新都不到半个间隔，被舍入掉了，权重一直停在 {wbf}。混合精度保留一份 fp32 主权重来积累这些更新；它的 bf16 副本最终是 {wmix}。步长更细的 fp16 能到 {w16}{why16}。',
				vars: (_rt, at) => {
					const m = M(at);
					const f0 = F(at, 'update', 0);
					const u = (m.exact - f0.exact) / m.step;
					const g16 = gapAtOne('fp16');
					const why16: I18n | string = Math.abs((m.fp16 - f0.fp16) / m.step - g16) < 1e-12 ? { en: `: each update rounds to exactly one gap, ${show(g16, 3)}`, zh: `：每次更新都被舍入成正好一个间隔 ${show(g16, 3)}` } : '';
					return { w0: f0.fp32, n: m.step, u: fmtSetting(Number(u.toPrecision(6))), w32: show(m.fp32, 7), gbf: gapAtOne('bf16'), wbf: show(m.bf16, 7), wmix: show(m.mixed, 7), w16: show(m.fp16, 7), why16 };
				}
			},
			look: { en: 'The stage plots how far each weight has moved from its start: the bf16 line stays at 0 while the other three climb.', zh: '舞台画出每个权重离起点移动了多少：bf16 那条线一直停在 0，另外三条在上升。' }
		},
		{
			id: 'limits',
			title: { en: 'The edges, and what this page does not show', zh: '边界，以及本页没有展示什么' },
			source: 'huge',
			at: 'first',
			view: { key: 'number-formats/bits', props: () => ({ formats: ['fp16', 'bf16', 'e4m3', 'e5m2', 'fp4', 'int8'] }) },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['loss-scaling'],
			narration: {
				en: 'Every format has edges. A fixed x = {x} is above e4m3’s largest value, {max43}, and PyTorch’s cast gives {v43}, which spreads through every sum it enters; FP4 sticks at {v4}. Gradients sit at the other edge: in one backward pass of TinyGPT, {fact:gradZero_fp16|pct} of the gradient entries become 0 in fp16, and loss scaling by {fact:gradScale} cuts that to {fact:gradZeroScaled_fp16|pct}. This page casts stored numbers; it runs no low-precision arithmetic.',
				zh: '每种格式都有边界。固定的 x = {x} 超过了 e4m3 的最大值 {max43}，PyTorch 的转换得到{v43}，它会扩散到它参与的每一个求和里；FP4 停在{v4}。梯度在另一条边界上：TinyGPT 的一次反向传播中，有 {fact:gradZero_fp16|pct} 的梯度元素在 fp16 里变成 0，把损失放大 {fact:gradScale} 倍后，这个比例降到 {fact:gradZeroScaled_fp16|pct}。本页只是把数换成各种格式存储，并没有真的用低精度做运算。',
				vars: (_rt, at) => {
					const x = M(at).x;
					return { x: show(x, 9), max43: largest('e4m3'), v43: stored('e4m3', x), v4: stored('fp4', x) };
				}
			},
			look: { en: 'Read the status of each row in the stage: overflow, NaN (not a number), stuck at the largest, clipped.', zh: '看舞台里每一行的状态：上溢、NaN（非数）、停在最大值、截断。' }
		}
	]
});

export default lesson;
