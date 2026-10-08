// tinybert: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (dims()), live values
// from the traces of the learner's text (TinyBERT 'fwd', TinyGPT 'gpt', the 'hide' process), measured ones from
// facts.json ({fact:...}). Claims that are not computed live are tested in claims.test.ts.
//   1 tokens → 2 hide one (task) → 3 three rows added → 4 attention both ways (predict) → 5 one score → 6 softmax, both
//   sides → 7 mix the values → 8 the guesses (predict) → 9 how training hides 15% → 10 what it is poor at (limits, task)
import type { LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtPct } from '../../atlas/core/format.ts';
import { MASK, type MaskedRaw } from './adapters/masked-text/adapter.ts';
import { attnRow, bigFeature, gap, gptGuess, rightHead, sides, T, type RT } from './bert.ts';

/** Structural numbers, read from the spec (never typed into the text). */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const s = (L('B1.scores')?.params ?? {}) as { nHead?: number; dHead?: number };
	return { d: Number(L('tok')?.outShape[1] ?? NaN), heads: s.nHead ?? NaN, dh: s.dHead ?? NaN, blocks: rt.spec?.layers.filter((l) => /^B\d+\.attn$/.test(l.id)).length ?? NaN, V: Number((rt.spec as { vocabSize?: number } | null)?.vocabSize ?? NaN), maskId: MASK };
}
/** the key the 'score' and 'softmax' steps work out: the token right after the gap */
export const after = (rt: RT) => Math.min(gap(rt).m + 1, gap(rt).len - 1);
const picked = (rt: RT) => (rt.input as MaskedRaw)?.picked === true;
/** a measured share as a whole percentage ("15%"), from facts.json */
const pctOf = (rt: RT, key: string) => {
	const v = rt.num(`fact:${key}`);
	return Number.isFinite(v) ? `${Math.round(v * 100)}%` : NaN;
};

/** Step 5: the [MASK] query against the key right after the gap, in the head that reads most after the gap. */
export function scoreOf(rt: RT) {
	const g = gap(rt);
	const D = dims(rt);
	const h = rightHead(rt);
	const s = attnRow(T(rt, 'B1.scores'), h, g.m)?.[after(rt)] ?? NaN;
	return { ...D, h, key: g.say(g.ids[after(rt)]), s, inv: 1 / Math.sqrt(D.dh), dot: s * Math.sqrt(D.dh) };
}

/** Step 6: softmax of the [MASK] row worked out for the key right after the gap (as zoom.softmax shows it). */
export function softmaxOf(rt: RT) {
	const g = gap(rt);
	const h = rightHead(rt);
	const sc = attnRow(T(rt, 'B1.scores'), h, g.m);
	const w = attnRow(T(rt, 'B1.attn'), h, g.m);
	const j = after(rt);
	let M = -Infinity;
	let sum = 0;
	if (sc) for (let i = 0; i < sc.length; i++) M = Math.max(M, sc[i]);
	if (sc) for (let i = 0; i < sc.length; i++) sum += Math.exp(sc[i] - M);
	const sA = sc ? sc[j] : NaN;
	const sd = sides(rt, 1, h);
	return {
		h,
		len: g.len,
		key: g.say(g.ids[j]),
		sA,
		M: sc ? M : NaN,
		ex: Math.exp(sA - M),
		sum: sc ? sum : NaN,
		w: w ? w[j] : NaN,
		...sd,
		afterN: plural(sd.nRight, { one: 'the token', many: 'the {n} tokens', zh: '{n} 个词元' }),
		beforeN: plural(sd.nLeft, { one: 'the token', many: 'the {n} tokens', zh: '{n} 个词元' })
	};
}

/** Step 7: the selected feature of the [MASK] row's mix and its largest part a × v. */
export function mixOf(rt: RT) {
	const g = gap(rt);
	const D = dims(rt);
	const h = rightHead(rt);
	const u = rt.sel.unit ?? [];
	const f = u[1] ?? NaN;
	const a = attnRow(T(rt, 'B1.attn'), h, g.m);
	const v = T(rt, 'B1.v');
	let j = 0;
	if (a && v) for (let i = 1; i < a.length; i++) if (Math.abs(a[i] * v.data[i * D.d + f]) > Math.abs(a[j] * v.data[j * D.d + f])) j = i;
	const aj = a ? a[j] : NaN;
	const vj = v ? v.data[j * D.d + f] : NaN;
	const where = j > g.m ? { en: 'after the gap', zh: '空缺后面' } : j < g.m ? { en: 'before the gap', zh: '空缺前面' } : { en: 'the gap itself', zh: '空缺本身' };
	return { ...D, h, lo: h * D.dh, hi: (h + 1) * D.dh - 1, f, y: T(rt, 'B1.mix')?.data[(u[0] ?? 0) * D.d + f] ?? NaN, j, a: aj, v: vj, av: aj * vj, src: g.say(g.ids[j]), where };
}

const lesson: LessonDef = {
	title: { en: 'TinyBERT: fill in a hidden word, reading both ways', zh: 'TinyBERT：双向阅读，补出被藏起来的词' },
	summary: {
		en: 'Hide one token of your text and follow the [MASK] row as it reads both sides of the gap, next to TinyGPT, which reads only to the left; see TinyBERT’s guesses, how training makes its own puzzles, and why it is a poor writer.',
		zh: '藏起你文字里的一个词元，跟着 [MASK] 行读空缺的两侧，并和只读左边的 TinyGPT 对照；看 TinyBERT 的猜测、训练如何自己出题，以及它为什么不擅长写作。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'Your text becomes tokens', zh: '你的文字变成词元' },
				view: { key: 'tinybert/chips', props: () => ({ mode: 'show' }) },
				narration: {
					en: 'TinyBERT cuts your text into {n} with the same tokenizer as TinyGPT. Then it adds special tokens, which stand for no text: [CLS] at the start and [SEP] at the end{two}. One token is swapped for [MASK], the gap the model must fill, so it reads {len} tokens in all.',
					zh: 'TinyBERT 用和 TinyGPT 相同的分词器把你的文字切成 {n}。然后加上特殊词元（不代表任何文字）：开头一个 [CLS]，结尾一个 [SEP]{two}。其中一个词元被换成 [MASK]，也就是模型要补的空缺，所以它一共读 {len} 个词元。',
					vars: (rt) => {
						const g = gap(rt);
						return {
							n: plural(g.n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }),
							len: g.len,
							two: pick(g.split >= 0, { en: '; your text has more than one sentence, so another [SEP] ends the first one (part A), and the rest is part B', zh: '；你的文字不止一句，所以第一句（A 段）后面还有一个 [SEP]，其余部分是 B 段' }, { en: '', zh: '' })
						};
					}
				},
				look: { en: 'The dashed chips are the special tokens. ␣ is a space: it belongs to the token after it, so “␣went” is “ went”.', zh: '虚线框的是特殊词元。␣ 表示空格：空格属于它后面的词元，所以“␣went”就是“ went”。' },
				terms: ['token', 'tokenizer', 'special-token']
			}
		},
		{
			id: 'hide',
			source: 'fwd',
			view: 'tinybert/chips',
			provenance: 'live',
			title: { en: 'Hide one token', zh: '藏起一个词元' },
			narration: {
				en: 'TinyBERT is trained for one job: fill in a hidden token from the tokens around it. Right now “{hidden}” is hidden (text token {k}, counting from 0). [MASK] has its own id, {maskId}, and its own learned row, so the model knows that something is missing, but not what.',
				zh: 'TinyBERT 只为一件事而训练：根据周围的词元补出一个被藏起来的词元。现在藏起来的是“{hidden}”（文字里的第 {k} 个词元，从 0 数起）。[MASK] 有自己的 id {maskId}，也有自己学来的一行数，所以模型知道这里缺了东西，但不知道缺的是什么。',
				vars: (rt) => {
					const g = gap(rt);
					return { hidden: g.say(g.hidden), k: g.k, maskId: String(MASK) };
				}
			},
			look: { en: 'Any token of your text can be hidden; [CLS] and [SEP] cannot. Every later step follows your choice.', zh: '你文字里的任何一个词元都可以藏；[CLS] 和 [SEP] 不行。后面的每一步都会跟着你的选择。' },
			task: {
				prompt: { en: 'Click another token in the row to hide it instead: pick a word you could guess from the words around it.', zh: '点这一行里的另一个词元，改为藏起它：挑一个你能从周围的词猜出来的词。' },
				done: (rt) => picked(rt),
				waiting: { en: 'Not yet: click one of the tokens.', zh: '还没有：点其中一个词元。' }
			},
			linger: 12000,
			terms: ['masked-language-model', 'special-token']
		},
		{
			id: 'rows',
			source: 'fwd',
			at: { layer: 'embed' },
			view: 'tinybert/rows',
			provenance: 'live',
			title: { en: 'Three learned rows per token', zh: '每个词元三行学来的数' },
			narration: {
				en: 'Each token becomes {d} numbers: the sum of three rows looked up in learned tables. The token’s row (its embedding) says which token it is, the position’s row says where it is, and the segment’s row says whether it is in part A or B; TinyGPT has only the first two. For the [MASK] at position {m}, feature {f} (counting from 0) is {t|n4} + {p|n4} + {s|n4} = {e}.',
				zh: '每个词元变成 {d} 个数：在三张学来的表里各查出一行，再加起来。词元行（它的嵌入）说明它是哪个词元，位置行说明它在哪里，段行说明它在 A 段还是 B 段；TinyGPT 只有前两种。位置 {m} 上的 [MASK]，特征 {f}（从 0 数起）是 {t|n4} + {p|n4} + {s|n4} = {e}。',
				vars: (rt) => {
					const g = gap(rt);
					const f = bigFeature(rt);
					const v = (id: string) => T(rt, id)?.data[g.m * dims(rt).d + f] ?? NaN;
					return { ...dims(rt), m: g.m, f, t: v('tok'), p: v('pos'), s: v('seg'), e: v('embed') };
				}
			},
			look: { en: 'Feature {f} is outlined in all four strips. The bottom strip is the sum of the three above it, cell by cell.', zh: '四条里都框出了特征 {f}。最下面一条是上面三条逐格相加的结果。', vars: (rt) => ({ f: bigFeature(rt) }) },
			terms: ['embedding', 'positional-encoding', 'segment-embedding']
		},
		{
			id: 'both',
			source: 'fwd',
			at: { layer: 'B1.attn' },
			view: 'tinybert/both-ways',
			provenance: 'live',
			title: { en: 'Attention can look both ways', zh: '注意力可以看两边' },
			narration: {
				en: 'Now {blocks} blocks let every token gather information from the others (attention), each with {heads} heads (separate attention calculations that run side by side). They are built like TinyGPT’s blocks, with one difference. The stage runs both models on your text: TinyBERT with [MASK], TinyGPT with the real word; the outlined row is the hidden token’s.',
				zh: '接下来 {blocks} 个块让每个词元从其他词元那里收集信息（注意力），每块有 {heads} 个头（并排进行的几组独立的注意力计算）。它们和 TinyGPT 的块构造相同，只有一处不同。舞台上两个模型读的都是你的文字：TinyBERT 读带 [MASK] 的，TinyGPT 读原来的词；加框的一行属于被藏的词元。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'Each row is one reader; the darker a cell, the more that reader takes from the token of its column. In TinyGPT the hatched cells are the ones its causal mask hides.', zh: '每一行是一个读者；格子越深，这个读者从那一列的词元拿得越多。TinyGPT 里打斜线的格子就是它的因果掩码遮住的地方。' },
			predict: {
				id: 'tinybert-both',
				question: { en: 'TinyGPT’s attention hides every token after the reader (a causal mask). In TinyBERT’s block 1, which tokens can the [MASK] row read?', zh: 'TinyGPT 的注意力把读者之后的每个词元都遮住（因果掩码）。在 TinyBERT 的第 1 块里，[MASK] 行能读到哪些词元？' },
				options: [
					{ en: 'Only the tokens before it', zh: '只有它前面的词元' },
					{ en: 'Only itself', zh: '只有它自己' },
					{ en: 'The tokens before and after it', zh: '它前面和后面的词元' }
				],
				answer: 2,
				explain: { en: 'TinyBERT has no causal mask, so every row reads every token, on both sides. In TinyGPT the upper-right triangle is exactly 0: no token reads a later one. Switch heads: each head spreads its weight its own way.', zh: 'TinyBERT 没有因果掩码，所以每一行都读所有词元，两边都读。TinyGPT 右上角的三角形正好全是 0：没有哪个词元能读到后面的词元。切换头看看：每个头分配权重的方式都不一样。' }
			},
			terms: ['attention', 'attention-head', 'causal-mask']
		},
		{
			id: 'score',
			source: 'fwd',
			at: { layer: 'B1.scores' },
			select: (rt) => ({ layer: 'B1.scores', unit: [rightHead(rt), gap(rt).m, after(rt)] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'One score, worked out', zh: '算出一个分数' },
			narration: {
				en: 'Block 1 rescales each row (layer normalization); learned weighted sums turn it into a query (what it looks for) and a key (what it offers), {dh} numbers per head. A score is a query times a key, number by number, added up and divided by √{dh}, which is × {inv|n4}. In head {h} (counting from 0), the [MASK] query and the key of “{key}”, right after the gap, give {dot|n4} × {inv|n4} = {s}. TinyGPT’s causal mask would turn this score into −∞ (minus infinity): a weight of exactly 0 in step 6.',
				zh: '第 1 块先把每一行重新缩放（层归一化），再用学来的加权和把它变成一个查询（它要找什么）和一个键（它能提供什么），每个头 {dh} 个数。分数就是查询和键逐个相乘、相加，再除以 √{dh}，也就是乘以 {inv|n4}。在头 {h}（从 0 数起）里，[MASK] 的查询和空缺后面紧挨着的“{key}”的键给出 {dot|n4} × {inv|n4} = {s}。TinyGPT 的因果掩码会把这个分数换成 −∞（负无穷），到第 6 步就成了正好为 0 的权重。',
				vars: (rt) => scoreOf(rt)
			},
			look: { en: 'The last column holds the {dh} products; they add up to {dot|n4}, and {dot|n4} × {inv|n4} is the score.', zh: '最后一列是 {dh} 个乘积，加起来是 {dot|n4}；{dot|n4} × {inv|n4} 就是分数。', vars: (rt) => scoreOf(rt) },
			terms: ['layer-norm', 'query-key-value', 'attention-head']
		},
		{
			id: 'softmax',
			source: 'fwd',
			at: { layer: 'B1.attn' },
			select: (rt) => ({ layer: 'B1.attn', unit: [rightHead(rt), gap(rt).m, after(rt)] }),
			view: 'zoom.softmax',
			provenance: 'live',
			title: { en: 'Scores become weights, on both sides', zh: '分数变成权重，两边都有' },
			narration: {
				en: 'Softmax turns the [MASK] row’s {len} scores into weights that add up to 1: from each score it subtracts the largest, raises e to that power and divides by the sum of all {len} results. For “{key}”: e to the power {sA|n4} − {M|n4} is {ex|n5}, and {ex|n5} ÷ {sum|n5} = {w|pct}. In head {h}, {right|pct} of the weight goes to {afterN} after the gap, {left|pct} to {beforeN} before it and {self|pct} to the [MASK] itself; in TinyGPT every weight after the gap is exactly 0.',
				zh: 'Softmax 把 [MASK] 行的 {len} 个分数变成加起来等于 1 的权重：每个分数先减去最大的分数，再求 e 的这个次方，最后除以全部 {len} 个结果的和。对“{key}”：e 的 {sA|n4} − {M|n4} 次方是 {ex|n5}，{ex|n5} ÷ {sum|n5} = {w|pct}。在头 {h} 里，{right|pct} 的权重给了空缺后面的 {afterN}，{left|pct} 给了空缺前面的 {beforeN}，{self|pct} 给了 [MASK] 自己；在 TinyGPT 里，空缺之后的每个权重都正好是 0。',
				vars: (rt) => softmaxOf(rt)
			},
			look: { en: 'Each bar is one token’s share of the [MASK] row’s attention; the shaded row is “{key}”, the one worked out here.', zh: '每一条是一个词元分到的 [MASK] 行注意力；底色加深的一行是“{key}”，也就是这里算的那一个。', vars: (rt) => ({ key: softmaxOf(rt).key }) },
			terms: ['softmax']
		},
		{
			id: 'mix',
			source: 'fwd',
			at: { layer: 'B1.mix' },
			select: (rt) => {
				const g = gap(rt);
				const D = dims(rt);
				const h = rightHead(rt);
				const m = T(rt, 'B1.mix');
				let f = h * D.dh;
				if (m) for (let c = h * D.dh; c < (h + 1) * D.dh; c++) if (Math.abs(m.data[g.m * D.d + c]) > Math.abs(m.data[g.m * D.d + f])) f = c;
				return { layer: 'B1.mix', unit: [g.m, f] };
			},
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'Mix what it reads', zh: '混合读到的内容' },
			narration: {
				en: 'Each token also has a value (what it passes on), {dh} numbers per head. The [MASK] row multiplies every token’s value by that token’s weight from step 6 (column a) and adds them up. Head {h} fills features {lo} to {hi}; here feature {f} (counting from 0) becomes {y|n4}, and the largest part, {a|n4} × {v|n4} = {av}, comes from “{src}”, {where}. A small MLP then works on each row alone, and blocks 2 to {blocks} repeat the recipe with their own weights.',
				zh: '每个词元还有一个值（它要传递的内容），每个头 {dh} 个数。[MASK] 行把每个词元的值乘以该词元在第 6 步的权重（a 列），再加起来。头 {h} 负责第 {lo} 到第 {hi} 个特征；这里特征 {f}（从 0 数起）变成了 {y|n4}，其中最大的一份 {a|n4} × {v|n4} = {av} 来自{where}的“{src}”。然后一个小的 MLP 对每一行单独加工，第 2 到第 {blocks} 块用各自的权重重复这个步骤。',
				vars: (rt) => mixOf(rt)
			},
			look: { en: 'Each value is scaled by its weight: tokens with a weight near 0 add almost nothing.', zh: '每个值都乘上它的权重：权重接近 0 的词元几乎没有贡献。' },
			terms: ['query-key-value', 'mlp', 'transformer']
		},
		{
			id: 'guess',
			source: 'fwd',
			at: { layer: 'probs' },
			view: 'tinybert/guesses',
			provenance: 'live',
			title: { en: 'TinyBERT fills the gap', zh: 'TinyBERT 补上空缺' },
			narration: {
				en: 'After block {blocks} and a last layer normalization, the [MASK] row’s {d} numbers are multiplied, number by number, with each of the {V} rows of the token table (the table the input came from) and added up, as for a score in step 5: one score per token. Softmax turns the scores into shares. You hid “{hidden}”.',
				zh: '经过第 {blocks} 块和最后一次层归一化，[MASK] 行的 {d} 个数和词元表（输入就是从这张表查出来的）的 {V} 行逐一相乘、相加，就像第 5 步算分数那样：每个词元得到一个分数。Softmax 把这些分数变成份额。你藏起来的是“{hidden}”。',
				vars: (rt) => ({ ...dims(rt), hidden: gap(rt).say(gap(rt).hidden) })
			},
			predict: {
				id: 'tinybert-guess',
				question: { en: 'Will TinyBERT’s top guess be exactly the token you hid?', zh: 'TinyBERT 的第一猜测会正好是你藏起来的词元吗？' },
				options: [
					{ en: 'Always: it has read thousands of stories', zh: '总是：它读过成千上万个故事' },
					{ en: 'Often, but not always: it can only rank the tokens by how well they fit there', zh: '经常是，但不总是：它只能按词元放在那里合不合适来排序' },
					{ en: 'Never: it cannot see the token', zh: '从来不会：它看不到这个词元' }
				],
				answer: 1,
				explain: { en: 'It never sees the hidden token, only its neighbours, so a word that fits just as well can come first. The bars show where your token landed; the try-this card “Hide a colour” shows such a miss.', zh: '它从来看不到被藏的词元，只能看到它的邻居，所以一个同样合适的词可能排在第一。条形图显示了你的词元排在哪里；“试一试”里的“藏起一个颜色”就是这样一个没猜中的例子。' }
			},
			look: { en: 'The longest bar is the top guess; the bar in the accent colour is the token you hid. A word with a space in front (␣) is a different token from the same word without one.', zh: '最长的条是第一猜测；强调色的条是你藏起来的词元。前面带空格（␣）的词和不带空格的同一个词是两个不同的词元。' },
			terms: ['logit', 'softmax', 'vocabulary']
		},
		{
			id: 'train',
			source: 'hide',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 450, unit: { en: 'token', zh: '词元' } },
			view: { key: 'tinybert/chips', props: () => ({ mode: 'train' }) },
			provenance: 'live',
			title: { en: 'Training makes its own puzzles', zh: '训练自己出题' },
			narration: {
				en: 'The right answers come from the text itself: training hides tokens and asks for them back, so nobody labels anything. Each token is chosen with chance {rate}; a chosen one becomes [MASK] {pm} of the time, a random token {pr}, and stays as it is {pk}, so no visible token can be trusted blindly. This draw ends with {hid} of your {ofN} hidden. After {fact:steps|int} steps of {fact:batch|int} puzzles, TinyBERT fills {fact:valAccMasked|pct} of [MASK] gaps in unseen stories correctly.',
				zh: '正确答案来自文字本身：训练时把词元藏起来，再要模型把它们找回来，所以不需要任何人标注。每个词元被选中的机会是 {rate}；被选中的词元有 {pm} 换成 [MASK]，{pr} 换成随机词元，{pk} 保持不变，所以看得见的词元也不能盲目相信。这一次抽签最后藏起了你 {ofN}中的 {hid} 个。经过 {fact:steps|int} 步、每步 {fact:batch|int} 道题的训练，TinyBERT 在没见过的故事里能正确补出 {fact:valAccMasked|pct} 的 [MASK] 空缺。',
				vars: (rt, at) => {
					const n = at.frame?.metrics?.n ?? NaN;
					return { rate: pctOf(rt, 'maskRate'), pm: pctOf(rt, 'maskShare'), pr: pctOf(rt, 'randomShare'), pk: pctOf(rt, 'keepShare'), hid: at.frame?.metrics?.hidden ?? NaN, ofN: plural(n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }) };
				}
			},
			look: { en: 'Play to watch each token draw a number u between 0 and 1: u below {u} chooses it, then a second number v decides what it becomes.', zh: '点播放，看每个词元依次抽一个 0 到 1 之间的数 u：u 小于 {u} 时它被选中，再由第二个数 v 决定它变成什么。', vars: (rt) => ({ u: String(rt.num('fact:maskRate')) }) },
			terms: ['self-supervised', 'masked-language-model', 'pre-training']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'probs' },
			view: { key: 'tinybert/guesses', props: () => ({ compare: true, pick: true }) },
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'What TinyBERT is poor at: writing', zh: 'TinyBERT 不擅长的事：写作' },
			narration: {
				en: 'TinyBERT was trained only to fill gaps with text on both sides, so it is a poor writer: writing means guessing the next word with nothing after it. TinyGPT is trained for exactly that; for your gap it reads only the left side{gpt}. On {fact:cmpN|int} hidden tokens in unseen stories, TinyBERT is right {fact:accBoth|pct} of the time with both sides, but only {fact:accLeft|pct} when the text stops right after the gap, below TinyGPT’s {fact:accGpt|pct}.',
				zh: 'TinyBERT 只被训练来填两侧都有文字的空，所以它不擅长写作：写作就是在后面什么都没有的时候猜下一个词。TinyGPT 正是为此而训练的；对你的空缺，它只读左边{gpt}。在没见过的故事里藏起 {fact:cmpN|int} 个词元：TinyBERT 读两侧时猜对 {fact:accBoth|pct}，文字在空缺处就结束时只有 {fact:accLeft|pct}，比 TinyGPT 的 {fact:accGpt|pct} 还低。',
				vars: (rt) => {
					const g = gap(rt);
					const q = gptGuess(rt);
					return { gpt: pick(q.top >= 0, { en: `, and guesses “${g.say(q.top)}” (${fmtPct(q.pTop)})`, zh: `，猜的是“${g.say(q.top)}”（${fmtPct(q.pTop)}）` }, { en: ', and for the first token it cannot guess at all: nothing comes before it', zh: '；如果藏的是第一个词元，它根本猜不了：前面什么都没有' }) };
				}
			},
			look: { en: 'This test hides one token per text; step 9’s {fact:valAccMasked|pct} comes from training-style puzzles with {rate} of the tokens chosen at once.', zh: '这个测试每段文字只藏一个词元；第 9 步的 {fact:valAccMasked|pct} 来自训练式的题目，一次选中 {rate} 的词元。', vars: (rt) => ({ rate: pctOf(rt, 'maskRate') }) },
			task: {
				prompt: { en: 'Hide the last word of your text: click it in the row above the bars. Then only a full stop or [SEP] follows the gap, so TinyBERT has hardly more to read than TinyGPT. Which model gives your word the bigger share?', zh: '藏起你文字的最后一个词：在条形图上方那一行里点它。这样空缺后面只剩句号或 [SEP]，TinyBERT 能读的并不比 TinyGPT 多多少。哪个模型给你的词的份额更大？' },
				done: (rt) => picked(rt) && gap(rt).k >= gap(rt).n - 2,
				hint: { en: 'The last word sits just before the final full stop. Under the bars, each model’s line gives your word’s share.', zh: '最后一个词就在最后的句号前面。条形图下面，每个模型的那一行写着你的词得到的份额。' },
				waiting: { en: 'Not yet: click the last word of your text in the row above the bars.', zh: '还没有：在条形图上方那一行里点你文字的最后一个词。' }
			},
			linger: 12000,
			terms: ['autoregressive', 'next-token-prediction', 'causal-mask']
		}
	],
	fixtures: [
		{ id: 'first', label: { en: 'the first token hidden', zh: '藏起第一个词元' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], mask: 0, picked: true } },
		{ id: 'last', label: { en: 'two sentences, the last word hidden', zh: '两句话，藏起最后一个词' }, raw: { text: 'Lily went to the park to play with her friends. She had a lot of fun.', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335, 410, 46, 338, 363, 258, 645, 369, 443, 46], mask: 16, picked: true } },
		{ id: 'colour', label: { en: 'a colour hidden (try-this)', zh: '藏起一个颜色（试一试）' }, raw: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], mask: 8, picked: true } },
		{ id: 'three', label: { en: 'three sentences', zh: '三句话' }, raw: { text: 'Ben has a red ball. He likes to throw it. His dog runs after it.', ids: [66, 300, 294, 115, 258, 673, 477, 46, 315, 514, 115, 265, 309, 114, 317, 308, 46, 804, 463, 731, 115, 893, 308, 46] } }
	]
};

export default lesson;
