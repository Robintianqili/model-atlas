// kv-cache: the guided lesson (ten steps; the concept page's primary path). Every number is live: the gen source runs
// TinyGPT on the learner's prompt A without and with a KV cache (gpt.ts counts the multiply-adds as they run), the prefix
// source runs prompt B with and without A's cached prefix; the large models' sizes come from facts.json.
//   generate -> redo -> predict -> cache -> memory -> cost -> same -> prefix -> break (task, breakIt) -> limits
import type { At, ExplainerRuntime, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmtBytes, fmtNum } from '../../atlas/core/format.ts';
import { cellsOf } from './views/grid/meta.ts';
import { NEW, ONE_UPON, PRESETS, PROMPT_MAX, TASK_SAME, taskDone } from './data.ts';
import { VOCAB } from './vocab.ts';

const NaNs = new Proxy({} as Record<string, number>, { get: () => NaN });
/** The step's frame metrics (NaN for anything missing, so lint reports it). */
const M = (at: At): Record<string, number> => ({ ...NaNs, ...(at.frame?.metrics ?? {}) }) as Record<string, number>;
/** Metrics of another frame of the step's tape, by phase and step. */
function F(at: At, phase: string, step = 0): Record<string, number> {
	const i = at.tape?.find({ phase, step }) ?? -1;
	return { ...NaNs, ...(i >= 0 ? (at.tape!.frame(i).metrics ?? {}) : {}) } as Record<string, number>;
}
/** Every frame's metrics of the step's tape. */
const all = (at: At) => Array.from({ length: at.tape?.length ?? 0 }, (_, i) => at.tape!.frame(i).metrics ?? {});
/** The tokens of the state's ids from position a to b (exclusive), as text. */
function textOf(at: At, a: number, b: number): string {
	const ids = Array.from(((at.state as { ids?: { data: ArrayLike<number> } } | undefined)?.ids?.data ?? []) as ArrayLike<number>);
	const s = ids.slice(a, b).filter((v) => v >= 0);
	return s.length === b - a ? s.map((id) => VOCAB[id] ?? '').join('').replace(/\s+/g, ' ') : '';
}
/** The size of a cache with the dims of facts.json: layers × heads × tokens × head size × 2 × bytes. */
const factBytes = (rt: ExplainerRuntime, p: 'gpt2' | 'llama') => [`${p}Layers`, `${p}Heads`, `${p}Ctx`, `${p}DHead`, 'halfBytes'].reduce((a, k) => a * rt.num(`fact:${k}`), 2);
const bytes = (b: number) => (Number.isFinite(b) ? fmtBytes(b) : NaN);
/** A difference as words: exactly 0, or the number (en follows “is”, zh includes 是). */
const exact = (d: number): I18n | number => (d === 0 ? { en: 'exactly 0', zh: '正好是 0' } : Number.isFinite(d) ? { en: fmtNum(d), zh: `是 ${fmtNum(d)}` } : NaN);
const DEMO = 8; // the decode pass the steps 'redo', 'predict' and 'cache' look at

const lesson = defineLesson({
	title: { en: 'The KV cache, step by step', zh: 'KV 缓存，分步课程' },
	summary: {
		en: 'Watch TinyGPT redo all its work for every new token, keep the keys and values in a cache instead, count the memory and the multiply-adds, check that the answers stay the same, then share a cached prefix between two prompts and see where sharing stops.',
		zh: '看 TinyGPT 每生成一个新词元就把所有计算重做一遍，改用缓存保存键和值，算出内存和乘加次数，验证答案不变，再让两个提示词共享缓存的前缀，看共享在哪里停止。'
	},
	fixtures: [
		{ id: 'one-upon', label: { en: 'prompt B with its first word changed', zh: '提示词 B 改了第一个词' }, raw: ONE_UPON },
		{ id: 'same', label: { en: 'two identical prompts', zh: '两个相同的提示词' }, raw: { ...PRESETS[0].raw, text2: PRESETS[0].raw.text, ids2: PRESETS[0].raw.ids } },
		{ id: 'apart', label: { en: 'two prompts with nothing shared', zh: '两个完全不共享的提示词' }, raw: { ...PRESETS[2].raw, text2: PRESETS[1].raw.text2, ids2: PRESETS[1].raw.ids2 } }
	],
	steps: [
		{
			id: 'generate',
			title: { en: 'One new token per pass', zh: '每一遍只生成一个新词元' },
			source: 'gen',
			at: 'last',
			view: { key: 'kv-cache/grid', props: () => ({ mode: 'plain' }) },
			provenance: 'live',
			terms: ['autoregressive', 'forward-pass', 'greedy-decoding'],
			narration: {
				en: 'TinyGPT writes one token per pass through the model. Your prompt A is {P} tokens: pass 0 (passes count from 0) reads them all and picks the most likely next token. Each later pass appends the token just picked and runs the whole model again. After {N} passes the text goes on with “{text}”.',
				zh: 'TinyGPT 每完整算一遍模型，只写出一个词元。你的提示词 A 有 {P} 个词元：第 0 遍（遍数从 0 数起）把它们全部读入，选出最可能的下一个词元。之后每一遍都把刚选出的词元接到末尾，再把整个模型算一遍。{N} 遍之后，文字接着写成了“{text}”。',
				vars: (_rt, at) => {
					const m = M(at);
					const text = textOf(at, m.P, m.P + m.maxNew).trim();
					return { P: m.P, N: m.maxNew, text: text || NaN };
				}
			},
			look: { en: 'The dashed chips in the stage are TinyGPT’s own tokens; each row of the grid is one token looking back at the tokens up to it.', zh: '舞台里虚线框的小块是 TinyGPT 自己写出的词元；网格的每一行是一个词元回头查看它及之前的词元。' }
		},
		{
			id: 'redo',
			title: { en: 'Without a cache, every pass starts over', zh: '没有缓存时，每一遍都从头算起' },
			source: 'gen',
			at: { phase: 'decode', step: DEMO },
			view: { key: 'kv-cache/grid', props: () => ({ mode: 'full' }) },
			provenance: 'live',
			terms: ['attention', 'query-key-value', 'multiply-add'],
			narration: {
				en: 'Attention compares each token’s query with the keys of the tokens up to it, then mixes their values. Pass {k} reads {n} tokens and computes all of them again: of the {cells} cells in the stage’s grid, only {fresh} are new, the newest token’s row. That pass costs {macs} multiply-adds, against {macs0} for pass 0.',
				zh: '注意力把每个词元的查询与它及之前各词元的键作比较，再把它们的值混合起来。第 {k} 遍读入 {n} 个词元，把它们全部重算一遍：舞台网格里的 {cells} 个单元中，只有 {fresh} 个是新的，也就是最新词元那一行。这一遍要做 {macs} 次乘加，而第 0 遍是 {macs0} 次。',
				vars: (_rt, at) => {
					const m = M(at);
					const c = cellsOf(at.frame?.metrics, 'full');
					return { k: m.k, n: m.n, cells: c.fresh + c.redo, fresh: c.fresh, macs: m.macsFull, macs0: F(at, 'prefill').macsFull };
				}
			},
			look: { en: 'Orange cells in the stage belong to earlier tokens and are computed again in this pass; only the purple bottom row is new.', zh: '舞台里橙色的单元属于前面的词元，在这一遍里又算了一次；只有最下面紫色的一行是新的。' }
		},
		{
			id: 'predict',
			title: { en: 'Does a new token change the past?', zh: '新词元会改变前面的数吗？' },
			source: 'gen',
			at: { phase: 'decode', step: DEMO },
			view: { key: 'kv-cache/grid', props: () => ({ mode: 'full' }) },
			provenance: 'live',
			terms: ['causal-mask'],
			predict: {
				id: 'kv-cache-predict-past',
				question: {
					en: 'Pass 8 appended one token and recomputed the keys and values of every earlier token. Compared with pass 7, how did those earlier keys and values change?',
					zh: '第 8 遍在末尾加了一个词元，并重算了前面每个词元的键和值。和第 7 遍相比，这些键和值有什么变化？'
				},
				options: [
					{ en: 'All of them shifted a little: every token now has one more token next to it', zh: '全都变了一点：每个词元旁边都多了一个词元' },
					{ en: 'Not at all: they are exactly the same numbers', zh: '完全没变：数值一模一样' },
					{ en: 'Only the last few tokens’ keys and values changed', zh: '只有最后几个词元的键和值变了' }
				],
				answer: 1,
				explain: {
					en: 'A token never looks at the tokens after it: the causal mask hides them. So a token’s key and value depend only on the tokens up to it, and a new token at the end cannot change them. The orange work in the stage repeats the last pass number for number.',
					zh: '一个词元从不查看它后面的词元：因果掩码把它们挡住了。所以一个词元的键和值只取决于它自己及之前的词元，末尾新来的词元改变不了它们。舞台里橙色的那些计算，一个数不差地重复了上一遍。'
				}
			},
			narration: {
				en: 'Pass {k} starts by appending “{tok}”, the token pass {kp} picked, and then redoes the keys and values of the {prev} tokens before it, in all {L} layers. Make your guess first: the stage shows the pass once you answer.',
				zh: '第 {k} 遍先在末尾接上第 {kp} 遍选出的词元“{tok}”，再把它前面 {prev} 个词元在全部 {L} 层里的键和值重做一遍。请先猜一猜：回答之后，舞台才会显示这一遍。',
				vars: (_rt, at) => {
					const m = M(at);
					const t = textOf(at, m.n - 1, m.n);
					return { k: m.k, kp: m.k - 1, tok: t.trim() || NaN, prev: m.n - 1, L: m.L };
				}
			},
			look: { en: 'After you answer, compare the orange rows with the purple newest row in the stage.', zh: '回答之后，对比舞台里橙色的各行和紫色的最新一行。' }
		},
		{
			id: 'cache',
			title: { en: 'Keep the keys and values: the KV cache', zh: '把键和值存起来：KV 缓存' },
			source: 'gen',
			at: { phase: 'decode', step: DEMO },
			view: { key: 'kv-cache/grid', props: () => ({ mode: 'cache' }) },
			provenance: 'live',
			terms: ['kv-cache', 'prefill-decode'],
			narration: {
				en: 'Across all {N} passes, the earlier tokens’ keys and values never changed: the largest difference from one pass to the next is {dKV}. So TinyGPT keeps them in a KV cache. Pass {k} then computes only the newest token: its query against the {n} keys in the cache, its own included, {macsC} multiply-adds instead of {macsF}. Filling the cache from the prompt is the prefill; each one-token pass after it is a decode step.',
				zh: '在全部 {N} 遍里，前面各词元的键和值从未改变：相邻两遍之间的最大差异{dKV}。所以 TinyGPT 把它们存进 KV 缓存（键值缓存）。这样第 {k} 遍只需计算最新的词元：用它的查询去比较缓存里的 {n} 个键（包括它自己的），只要 {macsC} 次乘加，而不是 {macsF} 次。用提示词填满缓存叫预填充；之后每一遍只加一个词元，叫解码。',
				vars: (_rt, at) => {
					const m = M(at);
					return { N: m.maxNew, dKV: exact(Math.max(...all(at).map((f) => f.dKV ?? NaN))), k: m.k, n: m.n, macsC: m.macsCache, macsF: m.macsFull };
				}
			},
			look: { en: 'Gray cells in the stage are not computed at all; the blue K and V slots are read from the cache.', zh: '舞台里灰色的单元根本不用计算；蓝色的 K、V 格子是从缓存里读出来的。' }
		},
		{
			id: 'memory',
			title: { en: 'What the cache costs: memory', zh: '缓存的代价：内存' },
			source: 'gen',
			at: 'last',
			view: { key: 'kv-cache/cost', props: () => ({ memory: true }) },
			provenance: 'live',
			terms: ['attention-head', 'precision'],
			narration: {
				en: 'The cache holds a key and a value for every token in every head of every layer: {L} layers × {H} heads × {n} tokens × {dh} numbers × 2 (key and value) × 4 bytes (fp32) = {now} for TinyGPT in the last pass, {full} at its full window of {C} tokens. The same formula with {fact:halfBytes}-byte fp16 numbers gives GPT-2 small {gpt2} at {fact:gpt2Ctx} tokens and Llama 2 7B {llama} at {fact:llamaCtx}, for every single conversation.',
				zh: '缓存为每一层、每个头里的每个词元保存一个键和一个值。最后一遍时，TinyGPT 的缓存是 {L} 层 × {H} 个头 × {n} 个词元 × {dh} 个数 × 2（键和值）× 4 字节（fp32）= {now}；窗口满 {C} 个词元时为 {full}。同一个公式按每个数 {fact:halfBytes} 字节的 fp16 计算：GPT-2 small 在 {fact:gpt2Ctx} 个词元时是 {gpt2}，Llama 2 7B 在 {fact:llamaCtx} 个词元时是 {llama}，而且每一段对话都各要一份。',
				vars: (rt, at) => {
					const m = M(at);
					return { L: m.L, H: m.H, n: m.n, dh: m.dh, now: bytes(m.bytes), full: bytes(m.bytesFull), C: m.C, gpt2: bytes(factBytes(rt, 'gpt2')), llama: bytes(factBytes(rt, 'llama')) };
				}
			},
			look: { en: 'The table in the stage works the formula out row by row; the shaded rows are TinyGPT, live.', zh: '舞台里的表格逐行套用这个公式；带底色的两行是实时的 TinyGPT。' }
		},
		{
			id: 'cost',
			title: { en: 'Work per new token: growing or flat', zh: '每个新词元的计算量：一路增长还是基本持平' },
			source: 'gen',
			at: 'last',
			view: 'kv-cache/cost',
			provenance: 'live',
			narration: {
				en: 'Without the cache, a pass recomputes every token, so its work grows in step with the length: from {f1} multiply-adds in pass 1 to {fL} in pass {kL}. With the cache it stays nearly flat, {c1} to {cL}: only the newest token’s look back at the cached keys grows. Over all {N} passes that is {cumF} against {cumC}: without the cache TinyGPT does {ratio} times as much work, and the gap widens with every token.',
				zh: '没有缓存时，每一遍都要重算所有词元，所以计算量随长度同步增长：从第 1 遍的 {f1} 次乘加涨到第 {kL} 遍的 {fL} 次。有缓存时几乎持平，从 {c1} 次到 {cL} 次：只有最新词元回看缓存中各个键的那部分在增长。全部 {N} 遍合计是 {cumF} 次对 {cumC} 次，不用缓存的计算量是用缓存的 {ratio} 倍，而且每多一个词元，差距就更大。',
				vars: (_rt, at) => {
					const m = M(at);
					const f1 = F(at, 'decode', 1);
					return { f1: f1.macsFull, fL: m.macsFull, kL: m.k, c1: f1.macsCache, cL: m.macsCache, N: m.maxNew, cumF: m.cumFull, cumC: m.cumCache, ratio: m.cumFull / m.cumCache };
				}
			},
			look: { en: 'In the stage, switch to “All passes so far”: the orange total curves upward, the blue one is a straight line.', zh: '在舞台里切换到“到目前为止的总和”：橙色的总量向上弯，蓝色的是一条直线。' }
		},
		{
			id: 'same',
			title: { en: 'Same tokens, same numbers', zh: '同样的词元，同样的数' },
			source: 'gen',
			at: 'last',
			view: { key: 'kv-cache/grid', props: () => ({ check: true }) },
			provenance: 'live',
			terms: ['logit'],
			narration: {
				en: 'Did skipping all that work change the answer? Both runs picked the same token in {same} of {N} passes, and the largest difference between their logits is {dL}: the cache reuses numbers, it does not approximate them. {kit}',
				zh: '跳过这么多计算，答案变了吗？两次运行在 {N} 遍中有 {same} 遍选出相同的词元，logit 之间的最大差异{dL}：缓存是复用这些数，而不是近似它们。{kit}',
				vars: (_rt, at) => {
					const m = M(at);
					const kit: I18n =
						m.dKit === 0
							? { en: 'The kit’s own forward pass of TinyGPT, which is checked against PyTorch, gives the same logits to the last bit as well.', zh: '本站自带的 TinyGPT 前向计算（已与 PyTorch 核对过）给出的 logit 也一位不差。' }
							: { en: `The kit’s own forward pass of TinyGPT, which is checked against PyTorch, differs from them by at most ${fmtNum(m.dKit)}.`, zh: `本站自带的 TinyGPT 前向计算（已与 PyTorch 核对过）与它们最多相差 ${fmtNum(m.dKit)}。` };
					return { same: m.sameAll, N: m.maxNew, dL: exact(m.dLogitMax), kit: Number.isFinite(m.dKit) ? kit : NaN };
				}
			},
			look: { en: 'The table in the stage lists, pass by pass, the token each run picked.', zh: '舞台里的表格逐遍列出两次运行各自选出的词元。' }
		},
		{
			id: 'prefix',
			title: { en: 'Two prompts, one cached prefix', zh: '两个提示词，一个缓存的前缀' },
			source: 'prefix',
			at: 'last',
			view: 'kv-cache/prefix',
			provenance: 'live',
			terms: ['prefix-caching'],
			narration: {
				en: 'A second request, prompt B, arrives. {share} The largest difference between B’s next-token logits and those of a run from scratch is {dR}.',
				zh: '第二个请求，提示词 B，来了。{share}B 的下一词元 logit 与从头计算的结果之间的最大差异{dR}。',
				vars: (_rt, at) => {
					const m = M(at);
					const share: I18n =
						m.reused > 0
							? {
									en: `Its first ${m.s} tokens are the same as prompt A’s, so their keys and values are already in A’s cache: B’s prefill computes only ${m.computed} of its ${m.PB} tokens, ${fmtNum(m.macsReuse)} multiply-adds instead of ${fmtNum(m.macsB)}, and shares ${fmtBytes(m.bytesShared)} of cache.${m.reused < m.s ? ' B’s last token is computed anyway, because B’s next-token logits come from it.' : ''}`,
									zh: `它的前 ${m.s} 个词元与提示词 A 相同，所以它们的键和值已经在 A 的缓存里：B 的预填充只需计算 ${m.PB} 个词元中的 ${m.computed} 个，乘加从 ${fmtNum(m.macsB)} 次降到 ${fmtNum(m.macsReuse)} 次，并共享 ${fmtBytes(m.bytesShared)} 的缓存。${m.reused < m.s ? 'B 的最后一个词元仍要计算，因为 B 的下一词元 logit 要由它算出。' : ''}`
								}
							: {
									en: `Its first token already differs from prompt A’s, so nothing of A’s cache can be reused: B’s prefill computes all ${m.PB} tokens, ${fmtNum(m.macsB)} multiply-adds.`,
									zh: `它的第一个词元就与提示词 A 不同，所以 A 的缓存一点也不能复用：B 的预填充要计算全部 ${m.PB} 个词元，共 ${fmtNum(m.macsB)} 次乘加。`
								};
					return { share: Number.isFinite(m.s) ? share : NaN, dR: exact(m.dReuse) };
				}
			},
			look: { en: 'In the stage, B’s chips with a solid blue border come from A’s cache; dashed orange ones are computed.', zh: '舞台里，B 中实线蓝框的词元来自 A 的缓存；虚线橙框的是新算的。' }
		},
		{
			id: 'break',
			title: { en: 'Change one early word: the sharing stops', zh: '改掉开头的一个词：共享就停止了' },
			source: 'prefix',
			at: 'last',
			view: { key: 'kv-cache/prefix', props: () => ({ layers: true }) },
			provenance: 'live',
			kindOf: 'breakIt',
			task: {
				prompt: { en: 'In the input widget, change only the first word of prompt B (for example “Once” to “One”) and keep the rest of it.', zh: '在输入控件里只改提示词 B 的第一个词（比如把“Once”改成“One”），其余部分保持不变。' },
				done: (rt) => taskDone(rt.input as never),
				hint: { en: `Done when B’s first token differs from A’s while at least ${TASK_SAME} later tokens stay in the same places.`, zh: `当 B 的第一个词元与 A 不同、同时后面至少有 ${TASK_SAME} 个词元位置不变时，就算完成。` },
				waiting: { en: 'Not yet: the two prompts must differ in their first token and match in several later places.', zh: '还没有：两个提示词的第一个词元要不同，后面还要有几处位置相同。' },
				doneText: { en: 'Done: prompt B now differs from A only at the very start.', zh: '完成：现在提示词 B 只在最开头与 A 不同。' }
			},
			narration: {
				en: 'Prefix caching reuses keys and values only up to the first different token, even when much of the rest matches. {detail}',
				zh: '前缀缓存只复用到第一个不同的词元为止，哪怕后面大部分都一样。{detail}',
				vars: (_rt, at) => {
					const m = M(at);
					if (!Number.isFinite(m.nSame)) return { detail: NaN };
					const detail: I18n =
						m.nSame > 0
							? {
									en: `After the first difference, ${m.nSame} tokens of B are the same token in the same place as in A. In layer 0 their keys differ by ${m.dK0 === 0 ? 'exactly 0' : fmtNum(m.dK0)}: that layer sees only the token and its place. From layer 1 on they differ by up to ${fmtNum(m.dKdeep)}, because each layer mixes in the earlier tokens. Reusing them would move B’s next-token logits by up to ${fmtNum(m.dWrong)}.`,
									zh: `第一个不同之后，B 有 ${m.nSame} 个词元与 A 在同一位置、是同一个词元。在第 0 层，它们的键相差${m.dK0 === 0 ? '正好是 0' : ` ${fmtNum(m.dK0)}`}：这一层只看词元本身和它的位置。从第 1 层起最多相差 ${fmtNum(m.dKdeep)}，因为每一层都会混入前面的词元。如果复用它们，B 的下一词元 logit 最多会偏移 ${fmtNum(m.dWrong)}。`
								}
							: {
									en: `Your prompts share ${m.s} tokens and then have no same token in the same place, so there is nothing more to compare yet: do the task to see what the later layers do.`,
									zh: `你的两个提示词共享 ${m.s} 个词元，之后就没有同一位置的同一词元了，所以还没有可比较的：完成任务，看看后面几层会怎样。`
								};
					return { detail };
				}
			},
			look: { en: 'The table in the stage gives, layer by layer, how far A’s and B’s keys and values are apart for those same tokens.', zh: '舞台里的表格逐层列出，对于这些相同的词元，A 和 B 的键和值相差多少。' }
		},
		{
			id: 'limits',
			title: { en: 'What the cache does not fix', zh: '缓存解决不了的事' },
			source: 'gen',
			at: 'first',
			view: { key: 'kv-cache/grid', props: () => ({ mode: 'cache' }) },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['context-window'],
			narration: {
				en: 'Within one request, the cache only helps after the prompt. Pass 0, the prefill, costs {m0} multiply-adds either way: every prompt token is computed once, unless another request already cached the same prefix, as prompt A’s cache can for prompt B. The cache also grows with every token, to {full} at TinyGPT’s {C}-token window and {llama} for Llama 2 7B, and TinyGPT has no position beyond its window, so this page keeps the first {pmax} prompt tokens and stops after {N} new ones.',
				zh: '在一次请求之内，缓存只在提示词之后才起作用。第 0 遍也就是预填充，用不用缓存都要 {m0} 次乘加：每个提示词词元都得算一次，除非别的请求已经缓存了相同的前缀，就像提示词 A 的缓存可以供 B 复用那样。缓存还随每个词元增长，在 TinyGPT 的 {C} 词元窗口满时达到 {full}，Llama 2 7B 则是 {llama}；而且 TinyGPT 在窗口之外没有位置可用，所以本页只保留提示词的前 {pmax} 个词元，生成 {N} 个新词元后就停下。',
				vars: (rt, at) => {
					const m = M(at);
					return { m0: m.macsFull === m.macsCache ? m.macsFull : NaN, full: bytes(m.bytesFull), C: m.C, llama: bytes(factBytes(rt, 'llama')), pmax: PROMPT_MAX, N: NEW };
				}
			},
			look: { en: 'At pass 0 every cell in the stage is purple: with or without the cache, the prompt is computed in full.', zh: '在第 0 遍，舞台里每个单元都是紫色的：不论有没有缓存，提示词都要完整算一遍。' }
		}
	]
});

export default lesson;
