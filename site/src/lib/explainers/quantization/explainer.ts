// quantization: the manifest of a CONCEPT package (K2): routed at /train/quantization/ by the concept route, which renders
// AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). One number in five formats,
// a rounding ledger of eight numbers, then TinyGPT with its weights quantized in the browser. Sources:
//   formats  the learner's x in fp32, fp16, bf16, int8, int4                               (process 'formats')
//   int8     the learner's eight values rounded to int8 with one absmax scale              (process 'ledger')
//   int4     the same at int4, with int8 as the second setting                            (process 'ledger')
//   affine   the sizes |x| of the eight values (all ≥ 0), affine int4 against absmax int4  (process 'ledger')
//   outlier  the learner's values at int4 (the outlier task)                               (process 'ledger')
//   fixes    the values with an outlier (quant.ts withOutlier), int4 in blocks of four, against clipping at the
//            second-largest size                                                          (process 'ledger')
//   toy      the learner's values; the toy view picks its own settings                    (process 'ledger')
//   gpt      TinyGPT's spec and weights (static/models/tinygpt, read only); no forward pass here  (trace, inputs: ids)
//   sweep    TinyGPT with one layer, then every weight matrix, quantized, scored on four stories (process 'sweep')
// The page around the lesson is concept.ts; the arithmetic is quant.ts.
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { absAll, rawOf, START, withOutlier } from './quant.ts';
import type { GptInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const vals = (rt: ExplainerRuntime) => rawOf(rt.input).vals;

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
	slug: 'quantization',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'values', start: START },
	assets: { dir: 'models/quantization', auto: ['facts.json'] },
	sources: {
		formats: { kind: 'process', process: 'formats', input: (rt) => rawOf(rt.input).x },
		int8: { kind: 'process', process: 'ledger', input: vals, opt: { bits: 8 } },
		int4: { kind: 'process', process: 'ledger', input: vals, opt: { bits: 4, alt: { bits: 8 } } },
		affine: { kind: 'process', process: 'ledger', input: (rt) => absAll(vals(rt)), opt: { bits: 4, scheme: 'affine', alt: { scheme: 'absmax' } } },
		outlier: { kind: 'process', process: 'ledger', input: vals, opt: { bits: 4 } },
		fixes: { kind: 'process', process: 'ledger', input: (rt) => withOutlier(vals(rt)), opt: { bits: 4, block: 4, alt: { block: 8, clip: 'second' } } },
		toy: { kind: 'process', process: 'ledger', input: vals, opt: { bits: 4 } },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['ids'] },
		sweep: { kind: 'process', process: 'sweep', input: gptInput }
	},
	page: {
		goals: [
			{
				en: 'Read a float’s bits as sign, exponent and fraction, and say what fp16 and bf16 give up compared with fp32.',
				zh: '把浮点数的各位读成符号、指数和尾数，并说出 fp16 和 bf16 相比 fp32 放弃了什么。'
			},
			{
				en: 'Quantize eight numbers to int8 and int4 by hand: pick the scale, round, and compute each error.',
				zh: '亲手把八个数量化成 int8 和 int4：确定缩放比例、舍入，再算出每个误差。'
			},
			{
				en: 'Explain why one outlier wrecks a shared scale, and how clipping, per-channel scales and blocks limit the damage.',
				zh: '解释为什么一个离群值会毁掉共用的缩放比例，以及截断、按通道缩放和分块如何减轻损害。'
			},
			{
				en: 'Judge a quantized TinyGPT by perplexity, KL divergence, top-1 agreement and model size.',
				zh: '用困惑度、KL 散度、top-1 一致率和模型大小来评价量化后的 TinyGPT。'
			}
		],
		prequestion: {
			id: 'quantization-pre',
			question: {
				en: 'A model’s weights are stored with 4 bits each instead of 32. What happens to the model?',
				zh: '把模型的每个权重从 32 位改成 4 位来存储，模型会怎样？'
			},
			options: [
				{ en: 'Nothing: the weights are only rounded a little', zh: '没什么变化：权重只是被稍微舍入了一点' },
				{ en: 'It gets smaller and somewhat worse; how much worse depends on how the rounding is set up', zh: '它变小了，也变差了一些；差多少取决于舍入方式的设置' },
				{ en: 'It stops working: 4 bits cannot hold a weight', zh: '它不能用了：4 位存不下一个权重' }
			],
			answer: 1,
			explain: {
				en: 'Four bits hold only sixteen different codes, so every weight is rounded to one of sixteen levels. With one scale for a whole matrix, TinyGPT on this page gets much worse; with one scale per row or per small block, it loses far less.',
				zh: '4 位只能表示 16 个不同的编码，所以每个权重都被舍入到 16 个刻度之一。整个矩阵只用一个缩放比例时，本页的 TinyGPT 明显变差；每行或每个小块各用一个缩放比例时，损失就小得多。'
			}
		},
		quiz: [
			{
				q: { en: 'Eight values, the largest of size 2, are stored as int8 with absmax. What is the scale s?', zh: '八个值中绝对值最大的是 2，用 absmax 存成 int8。缩放比例 s 是多少？' },
				options: [
					{ en: '2 / 127', zh: '2 / 127' },
					{ en: '2 / 255', zh: '2 / 255' },
					{ en: '127 / 2', zh: '127 / 2' }
				],
				answer: 0,
				explain: { en: 'absmax maps the largest size to the largest code, 127: s = max|x| / 127 = 2 / 127, so q = round(x / s).', zh: 'absmax 把最大的绝对值映射到最大的编码 127：s = max|x| / 127 = 2 / 127，然后 q = round(x / s)。' }
			},
			{
				q: { en: 'Why do blocks of 32 weights, each with its own scale, help int4?', zh: '为什么每 32 个权重一块、各用一个缩放比例，对 int4 有帮助？' },
				options: [
					{ en: 'An outlier only coarsens the scale of its own block', zh: '离群值只会让它所在的那一块的刻度变粗' },
					{ en: 'Blocks use more bits per code', zh: '分块后每个编码用的位数更多' },
					{ en: 'Blocks remove the outliers', zh: '分块会去掉离群值' }
				],
				answer: 0,
				explain: { en: 'Each block’s scale follows its own largest value, so the other blocks keep fine steps. The codes stay 4 bits; the price is one extra scale per block.', zh: '每一块的缩放比例只跟随这一块里最大的值，所以其他块的刻度依然很细。编码仍然是 4 位；代价是每一块多存一个缩放比例。' }
			}
		],
		misconception: {
			claim: { en: 'Rounding a weight changes it so little that quantization is free.', zh: '舍入只会让权重变化一点点，所以量化没有代价。' },
			truth: {
				en: 'How much each weight moves depends on the scale its group shares, and the largest value of the group sets it. One outlier makes the steps coarse for every other value in the group.',
				zh: '每个权重变化多少，取决于它所在的组共用的缩放比例，而这个比例由组里最大的值决定。一个离群值会让组里其他所有值的刻度都变粗。'
			},
			step: 'outlier'
		},
		lessonNote: {
			en: 'Eleven steps: one number bit by bit, a rounding ledger for eight numbers, one outlier, then a real TinyGPT quantized in your browser.',
			zh: '十一步：逐位看一个数，八个数的舍入账本，一个离群值，再在你的浏览器里量化一个真实的 TinyGPT。'
		}
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'the evaluation text of the quantized TinyGPT: four validation stories in the browser, 1,000 validation windows in PyTorch (TinyGPT’s tokenized copy of the validation split)',
			attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)'
		},
		{
			name: 'Quantization from the ground up (ideas)',
			url: 'https://ngrok.com/blog/quantization',
			license: 'ideas only (no code or text used; no licence stated)',
			usedFor: 'the bit-level format widgets, an outlier that breaks the scale, and judging a quantized model by perplexity and KL divergence; code written fresh',
			attribution: 'Sam Rose, ngrok blog (2026)'
		},
		{
			name: 'llama.cpp block quantization (ideas)',
			url: 'https://github.com/ggml-org/llama.cpp',
			license: 'MIT (ideas only; no code used)',
			usedFor: 'blocks of 32 weights that each get their own scale; code written fresh',
			attribution: 'Georgi Gerganov and the ggml contributors'
		}
	],
	glossary: [
		{
			id: 'floating-point',
			term: { en: 'floating-point number (sign, exponent, fraction)', zh: '浮点数（符号、指数、尾数）' },
			def: {
				en: 'A number stored as a sign bit, an exponent that says how big it is (a power of 2) and a fraction that holds its leading digits. More exponent bits reach further; more fraction bits are more exact.',
				zh: '用一个符号位、一个表示大小（2 的几次方）的指数和一个保存有效数字的尾数来存储的数。指数位越多，能表示的范围越大；尾数位越多，越精确。'
			},
			related: ['precision']
		},
		{
			id: 'quant-scale',
			term: { en: 'scale (quantization)', zh: '缩放比例（量化）' },
			def: {
				en: 'The step between two neighbouring codes: a value x is stored as the code q = round(x / s) and read back as q × s. A group of values shares one scale.',
				zh: '相邻两个编码之间的步长：值 x 存成编码 q = round(x / s)，读回时为 q × s。一组值共用一个缩放比例。'
			},
			related: ['quantization', 'zero-point']
		},
		{
			id: 'zero-point',
			term: { en: 'zero point', zh: '零点' },
			def: {
				en: 'In affine quantization, the code that stands for 0, so that the codes can cover a range that is not centred on 0, such as only positive values.',
				zh: '仿射量化中代表 0 的那个编码，有了它，编码就能覆盖一个不以 0 为中心的范围，例如只有正数的范围。'
			},
			related: ['quant-scale']
		},
		{
			id: 'outlier',
			term: { en: 'outlier', zh: '离群值' },
			def: {
				en: 'A value far larger than the others in its group. In quantization it sets the shared scale, so the other values get coarse steps.',
				zh: '比同组其他值大得多的值。量化时它决定了共用的缩放比例，使其他值的刻度变粗。'
			},
			related: ['quant-scale']
		},
		{
			id: 'quant-granularity',
			term: { en: 'per-tensor, per-channel and block quantization', zh: '按张量、按通道与分块量化' },
			def: {
				en: 'How many values share one scale: a whole weight matrix (per tensor), one output row (per channel) or a short block of weights along a row, such as 32.',
				zh: '多少个值共用一个缩放比例：整个权重矩阵（按张量）、一行输出（按通道），或一行里一小段权重，比如 32 个（分块）。'
			},
			related: ['quant-scale', 'outlier']
		}
	]
};

export default manifest;
