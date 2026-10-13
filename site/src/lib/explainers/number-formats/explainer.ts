// number-formats: the manifest of a CONCEPT package (K2): routed at /train/number-formats/ by the concept route, which
// renders AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). One number in nine
// formats, bit by bit (the learner clicks the bits), the range and precision of each, TinyGPT's real weights cast to them,
// and the small updates that mixed precision protects. Sources:
//   formats  the learner's x in every format                                              (process 'formats')
//   big      x = 70,000, beyond fp16's range (the predict step)                           (process 'formats')
//   tiny     x = 0.001, a small weight: subnormal or 0 in the 8- and 4-bit floats          (process 'formats')
//   huge     x = 500, beyond e4m3's range (the limits step)                               (process 'formats')
//   gpt      TinyGPT's spec and weights (static/models/tinygpt, read only); no forward pass here  (trace, inputs: ids)
//   sweep    TinyGPT with its 24 weight matrices cast to bf16, fp16, FP8, FP4, int4 and NF4, scored on four stories
//                                                                                          (process 'sweep')
//   sum      one weight, 100 updates of 0.001 in fp32, bf16, fp16 and with an fp32 master  (process 'sum')
// The page around the lesson is concept.ts; the arithmetic is fmt.ts.
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { BIG_X, HUGE_X, rawOf, START, TINY_X } from './fmt.ts';
import type { GptInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

/** TinyGPT's spec and weights for the sweep, copied once per loaded model (each tensor gets its own buffer, so the
 *  worker receives 1.46 MB, and the same object every time, so a new learner input keeps the sweep's run). */
const copies = new WeakMap<object, GptInput>();
export function gptInput(rt: ExplainerRuntime): GptInput {
	const m = rt.sources.gpt?.model;
	if (!m) return null;
	let c = copies.get(m);
	if (!c) {
		const weights: Record<string, Tensor> = {};
		for (const [k, t] of Object.entries(m.weights)) weights[k] = { shape: t.shape.slice(), data: (t.data as Float32Array).slice() };
		c = { spec: JSON.parse(JSON.stringify(m.spec)), weights };
		copies.set(m, c);
	}
	return c;
}

const manifest: ConceptManifest = {
	slug: 'number-formats',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'number', start: START },
	assets: { dir: 'models/number-formats', auto: ['facts.json'] },
	sources: {
		formats: { kind: 'process', process: 'formats', input: (rt) => rawOf(rt.input).x },
		big: { kind: 'process', process: 'formats', input: () => BIG_X },
		tiny: { kind: 'process', process: 'formats', input: () => TINY_X },
		huge: { kind: 'process', process: 'formats', input: () => HUGE_X },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['ids'] },
		sweep: { kind: 'process', process: 'sweep', input: gptInput },
		sum: { kind: 'process', process: 'sum', input: () => null }
	},
	page: {
		goals: [
			{
				en: 'Read a float’s bits as sign, exponent and fraction, and work out the number they stand for.',
				zh: '把浮点数的各位读成符号、指数和尾数，并算出它们代表的数。'
			},
			{
				en: 'Explain why the gap between neighbouring floats grows with the number, and what rounding, overflow and underflow do.',
				zh: '解释为什么相邻浮点数之间的间隔随数值增大，以及舍入、上溢和下溢分别会发生什么。'
			},
			{
				en: 'Compare fp32, fp16, bf16, FP8, FP4, int8, int4 and NF4 by range and precision, on one number and on a real model’s weights.',
				zh: '在一个数和一个真实模型的权重上，按范围和精度比较 fp32、fp16、bf16、FP8、FP4、int8、int4 和 NF4。'
			},
			{
				en: 'Explain why mixed-precision training keeps an fp32 copy of the weights and scales the loss.',
				zh: '解释混合精度训练为什么要保留一份 fp32 权重，并对损失做缩放。'
			}
		],
		prequestion: {
			id: 'number-formats-pre',
			question: {
				en: 'fp16 and bf16 both store a number in 16 bits. How do they differ?',
				zh: 'fp16 和 bf16 都用 16 位存一个数。它们有什么不同？'
			},
			options: [
				{ en: 'They hold the same numbers; only the hardware differs', zh: '它们能表示的数完全一样，只是硬件不同' },
				{ en: 'bf16 reaches far larger numbers; fp16 has finer steps', zh: 'bf16 能表示大得多的数；fp16 的刻度更细' },
				{ en: 'fp16 reaches far larger numbers; bf16 has finer steps', zh: 'fp16 能表示大得多的数；bf16 的刻度更细' }
			],
			answer: 1,
			explain: {
				en: 'bf16 keeps fp32’s eight exponent bits, so it reaches about as far as fp32, but only seven fraction bits. fp16 has five exponent bits and stops at 65,504, but its ten fraction bits make its steps eight times finer.',
				zh: 'bf16 保留了 fp32 的八个指数位，所以能表示的范围和 fp32 差不多，但只有七个尾数位。fp16 只有五个指数位，最大只到 65,504，但它有十个尾数位，刻度细了八倍。'
			}
		},
		quiz: [
			{
				q: { en: 'FP8 e4m3 has 3 fraction bits. What is the gap between 1 and the next e4m3 number?', zh: 'FP8 e4m3 有 3 个尾数位。1 和下一个 e4m3 数之间的间隔是多少？' },
				options: [
					{ en: '0.125 (one eighth)', zh: '0.125（八分之一）' },
					{ en: '0.001', zh: '0.001' },
					{ en: '0.5', zh: '0.5' }
				],
				answer: 0,
				explain: {
					en: 'Between 1 and 2 the exponent is fixed, and 3 fraction bits split that interval into 2³ = 8 equal steps of 1/8.',
					zh: '在 1 到 2 之间指数不变，3 个尾数位把这一段分成 2³ = 8 个相等的步长，每步 1/8。'
				}
			},
			{
				q: { en: 'Why do mixed-precision optimizers keep an fp32 copy of the weights?', zh: '混合精度的优化器为什么要保留一份 fp32 权重？' },
				options: [
					{ en: 'A small update added to a bf16 weight can round away completely', zh: '加到 bf16 权重上的小更新可能被完全舍入掉' },
					{ en: 'fp32 arithmetic is faster than bf16', zh: 'fp32 运算比 bf16 快' },
					{ en: 'bf16 cannot store negative weights', zh: 'bf16 存不了负的权重' }
				],
				answer: 0,
				explain: {
					en: 'An update smaller than half the gap at the weight leaves a bf16 weight unchanged, step after step. The fp32 master copy has gaps 65,536 times finer, so it collects the updates; the fast bf16 copy is made from it.',
					zh: '小于权重处间隔一半的更新，会让 bf16 权重一步又一步地保持不变。fp32 主权重的间隔细 65,536 倍，能把这些更新积累下来；用于快速计算的 bf16 副本再从它得到。'
				}
			}
		],
		misconception: {
			claim: { en: 'Fewer bits just means fewer decimal places.', zh: '位数少了，只不过是小数位少了。' },
			truth: {
				en: 'A float splits its bits between range (exponent) and precision (fraction). bf16 reaches as far as fp32 with fewer digits; fp16 has more digits but overflows above 65,504; an integer or NF4 code needs a scale before it covers any range at all.',
				zh: '浮点数把位数分给范围（指数）和精度（尾数）。bf16 的范围和 fp32 一样远，只是有效数字少；fp16 的有效数字多，但超过 65,504 就上溢；整数或 NF4 编码必须配一个缩放比例，才能覆盖某个范围。'
			},
			step: 'range16'
		},
		lessonNote: {
			en: 'Eleven steps: one number bit by bit, its neighbours, fp16 against bf16, the 8- and 4-bit formats, a real TinyGPT cast to each, and why mixed precision keeps fp32 master weights.',
			zh: '十一步：逐位看一个数，看它的邻居，比较 fp16 和 bf16，再看 8 位和 4 位格式，把真实的 TinyGPT 换成每种格式，最后看混合精度为什么要保留 fp32 主权重。'
		}
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'the evaluation text of TinyGPT with cast weights: four validation stories in the browser (the quantization page’s evaluation set), 1,000 validation windows and one gradient batch in PyTorch (TinyGPT’s tokenized copy of the validation split)',
			attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)'
		},
		{
			name: 'bitsandbytes NF4 data type (the 16 NF4 levels)',
			url: 'https://github.com/bitsandbytes-foundation/bitsandbytes',
			license: 'MIT',
			usedFor: 'the table of the 16 NF4 levels (fmt.ts NF4), recomputed from normal quantiles in pipeline/number-formats/flib.py; the rounding code is written fresh',
			attribution: 'QLoRA (Dettmers, Pagnoni, Holtzman and Zettlemoyer, 2023) and the bitsandbytes contributors'
		},
		{
			name: 'FP8 formats for deep learning; OCP Microscaling (MX) formats (ideas)',
			url: 'https://arxiv.org/abs/2209.05433',
			license: 'ideas only (no code or text used)',
			usedFor: 'the e4m3 and e5m2 encodings (e4m3 without infinity, one NaN code) and FP4 e2m1 with a scale per block of 32; code written fresh and checked against PyTorch’s float8 dtypes',
			attribution: 'Micikevicius et al. (2022); Open Compute Project MX specification v1.0 (2023)'
		},
		{
			name: 'Mixed precision training (ideas)',
			url: 'https://arxiv.org/abs/1710.03740',
			license: 'ideas only (no code or text used)',
			usedFor: 'fp32 master weights and loss scaling, shown on one weight and on TinyGPT’s gradients; code written fresh',
			attribution: 'Micikevicius et al. (2018)'
		},
		{
			name: 'Quantization from the ground up (ideas)',
			url: 'https://ngrok.com/blog/quantization',
			license: 'ideas only (no code or text used; no licence stated)',
			usedFor: 'bit-level format widgets in which the learner flips bits; code written fresh',
			attribution: 'Sam Rose, ngrok blog (2026)'
		}
	],
	// The concept's catalog entry (read only by `atlas promote`; the frozen type lacks a concept's fields, hence the cast).
	catalog: {
		interaction: {
			en: 'Click the bits of one number in FP32, FP16, BF16, FP8 (e4m3, e5m2), FP4, INT8, INT4 and NF4 and watch the stored value, the rounding error and the gap to the next number change; cast TinyGPT’s real weights to each format and measure its perplexity; see why mixed precision keeps fp32 master weights.',
			zh: '点击一个数在 FP32、FP16、BF16、FP8（e4m3、e5m2）、FP4、INT8、INT4 和 NF4 中的各个比特位，看存下的值、舍入误差和到下一个数的间隔怎样变化；把 TinyGPT 的真实权重转换成每种格式，测出它的困惑度；再看混合精度为什么要保留 fp32 主权重。'
		},
		mode: 'live',
		links: ['tinygpt']
	} as unknown as ExplainerManifest['catalog'],
	glossary: [
		{
			id: 'ulp',
			term: { en: 'gap to the next number (ulp)', zh: '到下一个数的间隔（ulp）' },
			def: {
				en: 'The distance from a stored number to its neighbour in the same format, the “unit in the last place”. A float’s gap doubles at every power of 2, so it stays about the same relative to the number.',
				zh: '一个存储的数到同一格式里相邻的数之间的距离，也叫“最后一位的单位”（ulp）。浮点数的间隔每过一个 2 的幂就翻倍，所以相对于数本身的大小基本不变。'
			},
			related: ['floating-point', 'precision']
		},
		{
			id: 'subnormal',
			term: { en: 'subnormal number and underflow', zh: '次正规数与下溢' },
			def: {
				en: 'Below a float’s smallest normal number, the exponent field is 0 and the leading 1 is dropped: these subnormal numbers fill the gap down to 0 with fewer significant bits. A number below half the smallest subnormal underflows to 0.',
				zh: '小于浮点格式最小正规数的数，指数字段为 0，开头的那个 1 被省掉：这些次正规数用更少的有效位填补到 0 之间的空隙。小于最小次正规数一半的数会下溢成 0。'
			},
			related: ['floating-point', 'overflow']
		},
		{
			id: 'overflow',
			term: { en: 'overflow', zh: '上溢' },
			def: {
				en: 'A number too large for a format. fp32, fp16, bf16 and e5m2 store it as infinity; e4m3 has no infinity, and PyTorch’s cast gives NaN; FP4 and integer codes stop at their largest value.',
				zh: '一个数大到格式存不下。fp32、fp16、bf16 和 e5m2 把它存成无穷大；e4m3 没有无穷大，PyTorch 的转换会得到 NaN；FP4 和整数编码则停在最大值上。'
			},
			related: ['subnormal', 'floating-point']
		},
		{
			id: 'fp8',
			term: { en: 'FP8 (e4m3 and e5m2)', zh: 'FP8（e4m3 与 e5m2）' },
			def: {
				en: 'Two 8-bit float formats: e4m3 (4 exponent, 3 fraction bits, up to 448) is the more precise, e5m2 (5 exponent, 2 fraction bits, up to 57,344) reaches further. Both are usually used with a scale per tensor or block.',
				zh: '两种 8 位浮点格式：e4m3（4 个指数位、3 个尾数位，最大 448）更精确，e5m2（5 个指数位、2 个尾数位，最大 57,344）范围更大。两者通常都要配合按张量或分块的缩放比例使用。'
			},
			related: ['floating-point', 'quant-scale']
		},
		{
			id: 'nf4',
			term: { en: 'NF4 (4-bit NormalFloat)', zh: 'NF4（4 位 NormalFloat）' },
			def: {
				en: 'A 4-bit code that indexes 16 fixed levels between −1 and 1, placed at quantiles of a normal distribution, so they are dense near 0 where most weights are. Each block of weights gets a scale (its largest size). Introduced with QLoRA.',
				zh: '一种 4 位编码，指向 −1 到 1 之间 16 个固定的刻度。这些刻度取在正态分布的分位点上，所以在 0 附近最密，正好是大多数权重所在的地方。每一块权重配一个缩放比例（这一块里最大的绝对值）。随 QLoRA 一起提出。'
			},
			related: ['quantization', 'quant-scale', 'lora']
		},
		{
			id: 'mixed-precision',
			term: { en: 'mixed-precision training', zh: '混合精度训练' },
			def: {
				en: 'Training that computes in a 16-bit (or 8-bit) format for speed and memory while keeping a master copy of the weights, and the sums of updates, in fp32 so that small updates are not rounded away.',
				zh: '为了速度和内存，用 16 位（或 8 位）格式计算，同时把权重的主副本和更新的累加保留在 fp32 里，免得小的更新被舍入掉。'
			},
			related: ['precision', 'loss-scaling']
		},
		{
			id: 'loss-scaling',
			term: { en: 'loss scaling', zh: '损失缩放' },
			def: {
				en: 'Multiplying the loss, and so every gradient, by a constant such as 1,024 before a backward pass in fp16, so that small gradients do not underflow to 0; the gradients are divided by it again in fp32 before the update.',
				zh: '在用 fp16 做反向传播之前，把损失（从而每个梯度）乘上一个常数，比如 1,024，使小的梯度不至于下溢成 0；更新之前再在 fp32 里除回去。'
			},
			related: ['mixed-precision', 'subnormal', 'gradient']
		}
	]
};

export default manifest;
