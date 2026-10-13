// attention-variants: the guided lesson (eleven steps; the concept page's primary path). The cache numbers are computed
// live from the learner's configuration (process 'cache', data.ts) and from published configurations (data.ts PRESETS);
// the attention numbers come from TinyGPT run in the browser (process 'share'); the uptraining numbers from facts.json
// (PyTorch, pipeline/attention-variants/prepare.py).
//   cache -> mha -> predict (MQA) -> gqa -> shared (TinyGPT) -> task -> presets -> mla -> convert (breakIt) -> uptrain
//   -> limits
import type { At, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { bytesText, configOf, KIND, kindOf, MLA, perBlock, perToken, PRESETS, presetConfig, total, withKv, type Config } from './data.ts';
import { gqaKv } from './views/heads/meta.ts';
import { LAST, TOKENS } from './views/attn/meta.ts';

const NaNs = new Proxy({} as Record<string, number>, { get: () => NaN });
/** The step's frame metrics (NaN for anything missing, so lint reports it). */
const M = (at: At): Record<string, number> => ({ ...NaNs, ...(at.frame?.metrics ?? {}) }) as Record<string, number>;
/** Metrics of another frame of the step's tape, by phase and step. */
function F(at: At, phase: string, step = 0): Record<string, number> {
	const i = at.tape?.find({ phase, step }) ?? -1;
	return { ...NaNs, ...(i >= 0 ? (at.tape!.frame(i).metrics ?? {}) : {}) } as Record<string, number>;
}
/** The learner's configuration as the cache frame holds it (null before the frame exists). */
const cfg = (at: At): Config | null => (Number.isFinite(M(at).heads) ? configOf({ ...at.frame!.metrics, preset: 'custom' }) : null);
const B = (b: number | undefined) => (b === undefined || !Number.isFinite(b) ? NaN : bytesText(b));
const P = (id: string) => presetConfig(PRESETS.find((p) => p.id === id)!);
/** A probe position as text: “was” (position 6) / “was”（位置 6）. */
const tok = (j: number): I18n | number =>
	Number.isFinite(j) ? { en: `“${TOKENS[j].replace('␣', '')}” (position ${j})`, zh: `“${TOKENS[j].replace('␣', '')}”（位置 ${j}）` } : NaN;
/** "1 layer" / "4 layers" (zh has no plural). */
const pl = (n: number, one: string, many: string): I18n => ({ en: `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`, zh: n.toLocaleString('en-US') });
/** The GQA row's KV heads: the learner's own when they already use GQA, else a group of about four (views/heads). */
const gqaOf = (c: Config) => (kindOf(c) === 1 ? c.kvHeads : gqaKv(c.heads));

const lesson = defineLesson({
	title: { en: 'Attention variants, step by step', zh: '注意力变体，分步课程' },
	summary: {
		en: 'Count what the KV cache stores, see which heads share keys and values in MHA, GQA, MQA and MLA, follow one shared-key attention computation on TinyGPT, size the caches of published models, and measure what sharing costs a trained model.',
		zh: '数一数 KV 缓存存了什么，看清 MHA、GQA、MQA 和 MLA 中哪些头共用键和值，在 TinyGPT 上跟着做一次共用键的注意力计算，算出已发表模型的缓存大小，再测一测共享会让训练好的模型付出什么代价。'
	},
	steps: [
		{
			id: 'cache',
			title: { en: 'What the KV cache holds', zh: 'KV 缓存里存了什么' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mine'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			terms: ['kv-cache', 'query-key-value', 'attention-head'],
			narration: {
				en: 'To write the next token, each new query is compared with the key of every earlier token and mixes their values. Recomputing those keys and values at every step would repeat work, so the model keeps them: the KV cache. {formula} That is {per} per token at {b} per number, and {all} for {T}.',
				zh: '为了写出下一个词元，每个新的查询都要和之前每个词元的键比较，再混合它们的值。每一步都重算这些键和值是重复劳动，所以模型把它们存下来，这就是 KV 缓存。{formula}每个数 {b} 字节，每个词元就是 {per}，{T} 个词元共 {all}。',
				vars: (_rt, at) => {
					const c = cfg(at);
					if (!c) return { formula: NaN, per: NaN, b: NaN, all: NaN, T: NaN };
					const n = c.layers * perBlock(c);
					const formula: I18n =
						kindOf(c) === 3
							? { en: `In your configuration every token adds one latent of ${c.latent.toLocaleString('en-US')} numbers in each of ${pl(c.layers, 'layer', 'layers').en}: ${n.toLocaleString('en-US')} numbers.`, zh: `在你的配置里，每个词元在 ${c.layers} 层中每层加一个 ${c.latent} 个数的潜向量：共 ${n.toLocaleString('en-US')} 个数。` }
							: { en: `In your configuration every token adds a key and a value of ${pl(c.dHead, 'number', 'numbers').en} for each of ${pl(c.kvHeads, 'KV head', 'KV heads').en} in ${pl(c.layers, 'layer', 'layers').en}: 2 × ${c.layers} × ${c.kvHeads} × ${c.dHead} = ${n.toLocaleString('en-US')} numbers.`, zh: `在你的配置里，每个词元在 ${c.layers} 层中为 ${c.kvHeads} 个 KV 头各加一个键和一个值，每个有 ${c.dHead} 个数：2 × ${c.layers} × ${c.kvHeads} × ${c.dHead} = ${n.toLocaleString('en-US')} 个数。` }
					return { formula, per: B(perToken(c)), b: pl(c.bytes, 'byte', 'bytes'), all: B(total(c)), T: pl(c.tokens, 'token', 'tokens') };
				}
			},
			look: { en: 'In the stage, the top row is the query heads and the boxes under them are the key and value heads whose numbers are cached.', zh: '舞台里，上面一行是查询头，下面的方框是被缓存了数值的键头和值头。' }
		},
		{
			id: 'mha',
			title: { en: 'Multi-head attention: a key head for every query head', zh: '多头注意力：每个查询头都有自己的键头' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mha'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			terms: ['attention-head'],
			narration: {
				en: 'In multi-head attention (MHA), the design TinyGPT uses, each query head has its own key projection and value projection. Each head can then look for something different. The price is in the cache: with your {H}, MHA stores {keys} and {values} per token in every layer, {per} per token in all.',
				zh: '多头注意力（MHA）是 TinyGPT 采用的设计：每个查询头都有自己的键投影和值投影，所以每个头都可以去找不同的东西。代价在缓存上：你的配置有 {H} 个查询头，用 MHA 时每一层为每个词元存 {keys} 个键和 {values} 个值，每个词元一共 {per}。',
				vars: (_rt, at) => {
					const c = cfg(at);
					return c
						? { H: pl(c.heads, 'query head', 'query heads'), keys: pl(c.heads, 'key', 'keys'), values: pl(c.heads, 'value', 'values'), per: B(perToken(withKv(c, c.heads))) }
						: { H: NaN, keys: NaN, values: NaN, per: NaN };
				}
			},
			look: { en: 'Every query head has a line to a key-and-value box of its own.', zh: '每个查询头都连向一个只属于它自己的键值方框。' }
		},
		{
			id: 'predict',
			title: { en: 'Share one key head: guess the saving', zh: '共用一个键头：猜猜能省多少' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mha', 'mqa'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			terms: ['multi-query-attention'],
			predict: {
				id: 'attention-variants-mqa',
				question: {
					en: 'In multi-query attention (MQA), every query head reads one shared key head and one shared value head. How does the KV cache change?',
					zh: '在多查询注意力（MQA）中，所有查询头都读同一个键头和同一个值头。KV 缓存会怎样变化？'
				},
				options: [
					{ en: 'It shrinks by a factor equal to the number of query heads', zh: '它按查询头的个数成倍缩小' },
					{ en: 'It halves: only the keys are shared', zh: '它减半：只有键被共用' },
					{ en: 'It stays the same: every query head still needs keys', zh: '它不变：每个查询头仍然需要键' }
				],
				answer: 0,
				explain: {
					en: 'The cache stores keys and values per KV head, not per query head. MQA keeps one KV head instead of one per query head, so with H query heads the cache is H times smaller. The query heads still exist; they just read the same keys and values.',
					zh: '缓存是按 KV 头存键和值的，不是按查询头。MQA 只保留一个 KV 头，而不是每个查询头一个，所以有 H 个查询头时缓存缩小为原来的 H 分之一。查询头仍然都在，只是读同一份键和值。'
				}
			},
			narration: {
				en: 'Each query head needs keys to compare its query with and values to mix. Suppose your configuration, with {H}, keeps one shared key head and one shared value head instead of a pair for every query head. Make your guess before the stage shows the two caches.',
				zh: '每个查询头都需要键来和自己的查询比较，也需要值来混合。假设你的配置（{H} 个查询头）不再为每个查询头各留一个键头和一个值头，而是只留一对大家共用的。先做出猜测，舞台再显示两种缓存。',
				vars: (_rt, at) => {
					const h = M(at).heads;
					return { H: Number.isFinite(h) ? pl(h, 'query head', 'query heads') : NaN };
				}
			},
			look: { en: 'After you answer, compare the per-token sizes under the two rows.', zh: '回答之后，对比两行下面写的每个词元的大小。' }
		},
		{
			id: 'gqa',
			title: { en: 'Grouped-query attention: the middle ground', zh: '分组查询注意力：折中的办法' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mha', 'gqa', 'mqa'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			terms: ['grouped-query-attention'],
			narration: {
				en: 'MQA stores {mqa} per token instead of {mha}, but one key head for every query is a tight squeeze. Grouped-query attention (GQA) splits the query heads into groups, and each group shares one key head and one value head. {gq} MHA and MQA are its two ends: one query head per group, or a single group.',
				zh: 'MQA 每个词元只存 {mqa}，而不是 {mha}，可是所有查询只用一个键头，未免太挤。分组查询注意力（GQA）把查询头分成若干组，每组共用一个键头和一个值头。{gq}MHA 和 MQA 是它的两个极端：每组一个查询头，或者只有一组。',
				vars: (_rt, at) => {
					const c = cfg(at);
					if (!c) return { mqa: NaN, mha: NaN, gq: NaN };
					const k = gqaOf(c);
					const gq: I18n = k
						? { en: `With ${c.heads} query heads in ${k} groups of ${c.heads / k}, the cache is ${B(perToken(withKv(c, k)))} per token, ${k} / ${c.heads} of MHA’s.`, zh: `把 ${c.heads} 个查询头分成 ${k} 组、每组 ${c.heads / k} 个，缓存就是每个词元 ${B(perToken(withKv(c, k)))}，是 MHA 的 ${k} / ${c.heads}。` }
						: { en: `Your ${c.heads} query heads cannot be split into equal groups of more than one and fewer than all, so only MHA and MQA fit them.`, zh: `你的 ${c.heads} 个查询头无法均分成大小在 1 和全部之间的组，所以只能用 MHA 或 MQA。` };
					return { mqa: B(perToken(withKv(c, 1))), mha: B(perToken(withKv(c, c.heads))), gq };
				}
			},
			look: { en: 'The middle row: each key-and-value box now spans one group of query heads.', zh: '中间一行：每个键值方框现在横跨一组查询头。' }
		},
		{
			id: 'shared',
			title: { en: 'One attention computation with shared keys, on TinyGPT', zh: '在 TinyGPT 上做一次共用键的注意力计算' },
			source: 'share',
			at: { phase: 'shared', step: 0 },
			view: { key: 'attention-variants/attn', props: () => ({ head: 0 }) },
			provenance: 'live',
			terms: ['dot-product', 'softmax'],
			narration: {
				en: 'TinyGPT’s block {layer}, at the last token, “{last}” (blocks, heads and positions count from 0). As trained, head 0 puts {w0} on {t0} and head 1 puts {w1} on {t1}. Now heads 0 and 1 share one key head and one value head, the mean of their trained ones. Each head still scores every shared key k̄ with its own query q, as q · k̄ / √{d}, so head 0 now peaks at {s0} with {v0} and head 1 at {s1} with {v1}.',
				zh: 'TinyGPT 编号 {layer} 的块，最后一个词元“{last}”处（块、头和位置都从 0 数起）。训练好的模型里，头 0 把 {w0} 放在{t0}上，头 1 把 {w1} 放在{t1}上。现在头 0 和头 1 共用一个键头和一个值头，即它们训练出的键头、值头的平均。每个头仍用自己的查询 q 给每个共用的键 k̄ 打分，即 q · k̄ / √{d}，所以现在头 0 的峰值落在{s0}，占 {v0}；头 1 的落在{s1}，占 {v1}。',
				vars: (_rt, at) => {
					const m = M(at);
					const m0 = F(at, 'mha');
					const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : NaN);
					return { layer: m.layer, last: LAST, w0: pct(m0.top0), t0: tok(m0.arg0), w1: pct(m0.top1), t1: tok(m0.arg1), d: m.dHead, s0: tok(m.arg0), v0: pct(m.top0), s1: tok(m.arg1), v1: pct(m.top1) };
				}
			},
			look: { en: 'In the shared table, rows 0 and 1 read the same KV head, 0, yet their strongest-colored cells sit in different columns.', zh: '在共用的那张表里，行 0 和行 1 读的是同一个 KV 头 0，可它们颜色最强的格子落在不同的列。' }
		},
		{
			id: 'task',
			title: { en: 'Your turn: the cache of a big model', zh: '轮到你了：一个大模型的缓存' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mine'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			task: {
				prompt: { en: 'In the input widget, pick Llama 2 70B, then set its KV heads to 64, one per query head, as MHA would have it.', zh: '在输入控件里选 Llama 2 70B，再把它的 KV 头数设为 64，每个查询头一个，也就是 MHA 的做法。' },
				done: (rt) => {
					const c = configOf(rt.input);
					return c.layers === 80 && c.heads === 64 && c.kvHeads === 64 && c.latent === 0;
				},
				hint: { en: 'The KV heads field lists only the numbers that divide the query heads evenly.', zh: 'KV 头数一栏只列出能整除查询头数的数。' },
				waiting: { en: 'Not yet: the configuration is not Llama 2 70B with 64 KV heads.', zh: '还没有：当前配置不是 64 个 KV 头的 Llama 2 70B。' },
				doneText: { en: 'Done: the narration now sets this cache next to Llama 2 70B as published, with 8 KV heads.', zh: '完成：讲解里现在把这个缓存和按发表配置（8 个 KV 头）的 Llama 2 70B 放在一起。' }
			},
			narration: {
				en: 'Your configuration is {kind}: {H} and {K} per layer, in {L}. At {T} and {b} per number its cache holds {all}. {alt}',
				zh: '你的配置是{kind}：{L} 层，每层 {H} 个查询头、{K} 个 KV 头。{T} 个词元、每个数 {b} 字节时，它的缓存占 {all}。{alt}',
				vars: (_rt, at) => {
					const c = cfg(at);
					if (!c) return { kind: NaN, H: NaN, K: NaN, L: NaN, T: NaN, b: NaN, all: NaN, alt: NaN };
					const [mha, mqa] = [B(total(withKv(c, c.heads))), B(total(withKv(c, 1)))];
					const k = kindOf(c);
					const alt: I18n =
						k === 0
							? { en: `That is already one KV head per query head; with a single KV head it would hold ${mqa}.`, zh: `这已经是每个查询头一个 KV 头；只用一个 KV 头时会占 ${mqa}。` }
							: k === 2
								? { en: `With one KV head per query head it would hold ${mha}.`, zh: `每个查询头一个 KV 头时会占 ${mha}。` }
								: { en: `With one KV head per query head it would hold ${mha}; with a single one, ${mqa}.`, zh: `每个查询头一个 KV 头时会占 ${mha}；只有一个 KV 头时占 ${mqa}。` };
					// A published model of the same shape (layers, query heads, head width) that caches fewer KV heads: Llama 2 70B as
					// published once the learner has done the task, Llama 3 8B next to Llama 2 7B.
					const twin = c.latent === 0 ? PRESETS.find((p) => p.cfg.latent === 0 && p.cfg.layers === c.layers && p.cfg.heads === c.heads && p.cfg.dHead === c.dHead && p.cfg.kvHeads < c.kvHeads) : undefined;
					const tw: I18n = twin
						? { en: ` At the same length, ${twin.label.en} as published, with ${pl(twin.cfg.kvHeads, 'KV head', 'KV heads').en}, holds ${B(total(withKv(c, twin.cfg.kvHeads)))}.`, zh: `同样长度下，按发表的配置（${twin.cfg.kvHeads} 个 KV 头），${twin.label.zh} 占 ${B(total(withKv(c, twin.cfg.kvHeads)))}。` }
						: { en: '', zh: '' };
					return {
						kind: KIND[k],
						H: pl(c.heads, 'query head', 'query heads'),
						K: k === 3 ? { en: 'no KV heads', zh: '0' } : pl(c.kvHeads, 'KV head', 'KV heads'),
						L: pl(c.layers, 'layer', 'layers'),
						T: pl(c.tokens, 'token', 'tokens'),
						b: pl(c.bytes, 'byte', 'bytes'),
						all: B(total(c)),
						alt: { en: alt.en + tw.en, zh: (alt.zh ?? '') + (tw.zh ?? '') }
					};
				}
			},
			look: { en: 'The sizes under the diagram follow every change you make in the input widget.', zh: '示意图下面的大小会跟着你在输入控件里的每一次修改变化。' }
		},
		{
			id: 'presets',
			title: { en: 'Published models: what they cache per token', zh: '已发表的模型：每个词元缓存多少' },
			source: 'cache',
			at: 'first',
			view: 'attention-variants/presets',
			provenance: 'live',
			narration: {
				en: 'Published models, with their usual 2-byte numbers. Llama 2 7B uses MHA and caches {l2} per token; Llama 3 8B, about the same size, shares {l3k} KV heads among {l3h} query heads and caches {l3}. Llama 2 70B is far larger, yet with GQA it caches {l70}, less than Llama 2 7B. Falcon-7B’s single KV head gets down to {fal}.',
				zh: '这些已发表的模型都按常用的 2 字节数计算。Llama 2 7B 用 MHA，每个词元缓存 {l2}；规模相近的 Llama 3 8B 让 {l3h} 个查询头共用 {l3k} 个 KV 头，缓存 {l3}。Llama 2 70B 大得多，但用了 GQA，只缓存 {l70}，比 Llama 2 7B 还少。Falcon-7B 只有一个 KV 头，降到 {fal}。',
				vars: () => {
					const l3 = P('llama3-8b');
					return { l2: B(perToken(P('llama2-7b'))), l3k: l3.kvHeads, l3h: l3.heads, l3: B(perToken(l3)), l70: B(perToken(P('llama2-70b'))), fal: B(perToken(P('falcon-7b'))) };
				}
			},
			look: { en: 'Each bar is one model’s cache per token, on one axis from 0; the color is its variant.', zh: '每根柱是一个模型每个词元的缓存，画在同一根从 0 开始的坐标轴上；颜色表示它的注意力变体。' }
		},
		{
			id: 'mla',
			title: { en: 'Latent attention: cache a summary instead', zh: '潜在注意力：改为缓存一个摘要' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mha', 'mla'], preset: 'deepseek-v2' }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of DeepSeek-V2’s heads; the sizes are computed from its published configuration.', zh: 'DeepSeek-V2 各个头的示意图；大小按它发表的配置计算。' } },
			terms: ['latent-attention', 'positional-encoding'],
			narration: {
				en: 'DeepSeek-V2 goes further with multi-head latent attention (MLA). Its {H} heads of width {d} would cache {mha} numbers per token and layer with MHA. MLA caches one latent vector of {dc} numbers plus one position (RoPE) key of {dr} that every head shares: {mla} numbers, {ratio} times fewer, about what GQA with {eq} KV heads would store. Learned up-projections rebuild each head’s keys and values from the latent.',
				zh: 'DeepSeek-V2 更进一步，用了多头潜在注意力（MLA）。它有 {H} 个宽度为 {d} 的头，若用 MHA，每个词元每层要缓存 {mha} 个数。MLA 只缓存一个 {dc} 个数的潜向量，外加一个所有头共用的位置（RoPE）键，宽 {dr}：一共 {mla} 个数，只有 MHA 的 {ratio} 分之一，大约相当于 {eq} 个 KV 头的 GQA。每个头的键和值由学到的上投影矩阵从潜向量还原出来。',
				vars: () => {
					const c = P('deepseek-v2');
					const mha = perBlock(withKv(c, c.heads));
					return { H: c.heads, d: c.dHead, mha, dc: MLA.dc, dr: MLA.dr, mla: perBlock(c), ratio: mha / perBlock(c), eq: perBlock(c) / (2 * c.dHead) };
				}
			},
			look: { en: 'The lower row has no key or value heads at all: one latent box sits under all the query heads.', zh: '下面一行根本没有键头和值头：所有查询头下面只有一个潜向量方框。' }
		},
		{
			id: 'convert',
			title: { en: 'Sharing is not free: convert a trained TinyGPT', zh: '共享不是免费的：转换训练好的 TinyGPT' },
			source: 'share',
			at: { phase: 'eval', step: 2 },
			view: 'attention-variants/cost',
			provenance: 'live',
			kindOf: 'breakIt',
			terms: ['perplexity', 'kl-divergence', 'uptraining'],
			narration: {
				en: 'Now every block of TinyGPT shares its keys and values, made by averaging each group’s trained heads, with no further training. On {tokens} tokens of four stories, perplexity rises from {p4} to {p2} with 2 KV heads and to {p1} with 1; with MQA the top next token matches the trained model’s at only {a1|pct} of positions. Each head had learned its own keys, and their average fits none of them.',
				zh: '现在 TinyGPT 的每一块都共用键和值：把每组训练好的头取平均得到，不再训练。在四个故事的 {tokens} 个词元上，困惑度从 {p4} 升到 {p2}（2 个 KV 头），只有 1 个时升到 {p1}；用 MQA 时，只有 {a1|pct} 的位置上最可能的下一个词元与原模型相同。每个头原先学的是自己的一套键，它们的平均哪个头都不合适。',
				vars: (_rt, at) => {
					const [e4, e2, e1] = [F(at, 'eval', 0), F(at, 'eval', 1), F(at, 'eval', 2)];
					return { tokens: e1.tokens, p4: e4.ppl, p2: e2.ppl, p1: e1.ppl, a1: e1.agree };
				}
			},
			look: { en: 'Read the bar lengths and the KL line under each bar: both grow as the KV heads go from 4 to 1.', zh: '看柱子的长度和每根柱子下面的 KL：KV 头从 4 个减到 1 个时，两者都在变大。' }
		},
		{
			id: 'uptrain',
			title: { en: 'Uptraining repairs most of it', zh: '继续训练能修复大部分损失' },
			source: 'share',
			at: { phase: 'eval', step: 2 },
			view: { key: 'attention-variants/cost', props: () => ({ mode: 'uptrain' }) },
			provenance: 'precomputed',
			terms: ['uptraining', 'pre-training'],
			narration: {
				en: 'Ainslie et al. repair a converted model by training it a little longer, uptraining, with {share} of the original training steps. Measured in PyTorch on {fact:bigTokens} validation tokens (far more text than the four stories; there the unconverted TinyGPT scores {fact:pplMha0}), {fact:upSteps} such steps bring GQA from {fact:pplGqa0} to {fact:pplGqa1500} and MQA from {fact:pplMqa0} to {fact:pplMqa1500}, against {fact:pplMha1500} for MHA trained as long. GQA ends closer to MHA than MQA does.',
				zh: 'Ainslie 等人用多训练一小段来修复转换后的模型，这叫继续训练（uptraining），用的步数是原训练的 {share}。用 PyTorch 在 {fact:bigTokens} 个验证集词元上测量（文本比那四个故事多得多，未转换的 TinyGPT 在这里是 {fact:pplMha0}），{fact:upSteps} 步继续训练让 GQA 版的困惑度从 {fact:pplGqa0} 降到 {fact:pplGqa1500}，MQA 版从 {fact:pplMqa0} 降到 {fact:pplMqa1500}；同样多训练这些步的 MHA 版是 {fact:pplMha1500}。GQA 最终比 MQA 更接近 MHA。',
				vars: (rt) => {
					const s = rt.num('fact:upSteps') / rt.num('fact:pretrainSteps');
					return { share: Number.isFinite(s) ? `${Number((100 * s).toPrecision(3))}%` : NaN };
				}
			},
			look: { en: 'Follow the three lines from left to right: the steep early drop is the model relearning its keys.', zh: '从左往右看三条线：开头那段陡降，就是模型在重新学它的键。' }
		},
		{
			id: 'limits',
			title: { en: 'What sharing does not save', zh: '共享省不下什么' },
			source: 'cache',
			at: 'first',
			view: { key: 'attention-variants/heads', props: () => ({ rows: ['mine'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
			kindOf: 'limits',
			narration: {
				en: 'Sharing shrinks what is stored, and read again for every new token, not the arithmetic of the scores: each of your {H} still compares its query with every one of the {keys}, {macs} per layer for one new token, for MHA, GQA and MQA alike. And the cache still grows with every token: yours holds {all} at {T}, and each conversation served at the same time keeps a cache of its own.',
				zh: '共享缩小的是存下来、每生成一个新词元都要重新读一遍的数据，而不是打分的计算量：你的 {H} 个查询头仍然要各自和全部 {keys} 个缓存的键比较，每生成一个新词元，每层要做 {macs} 次乘加，MHA、GQA 和 MQA 都一样。缓存也仍然随每个词元增长：你的配置在 {T} 个词元时占 {all}，而同时服务的每一段对话都各有一份缓存。',
				vars: (_rt, at) => {
					const c = cfg(at);
					const macs = M(at).macs;
					return c && Number.isFinite(macs)
						? { H: pl(c.heads, 'query head', 'query heads'), keys: pl(c.tokens, 'cached key', 'cached keys'), T: pl(c.tokens, 'token', 'tokens'), macs: pl(macs, 'multiply-add', 'multiply-adds'), all: B(total(c)) }
						: { H: NaN, keys: NaN, T: NaN, macs: NaN, all: NaN };
				}
			},
			look: { en: 'However few the key and value boxes, the row of query heads stays as long.', zh: '不管键值方框有多少，上面那一排查询头的数量都没有变。' }
		}
	]
});

export default lesson;
