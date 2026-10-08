<!--
  The TinyGPT guided lesson: twelve steps from the learner's text to the next token, one focused
  view at a time (kit Walkthrough). Each stage reuses the page's own views (the block player, the
  dot product, the logit lens, the sampling controls) or a small lesson view, all on the live run of
  the current text. The lesson has its own selection (LS: followed token, head, key, dimension), so
  it never moves the "Explore freely" views; the text itself is shared, so editing it here, or
  generating tokens in step 12, updates the whole page.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import type { TokenItem, WalkStep } from '$lib/kit/types';
	import type { TGState } from './state.svelte';
	import { D, H, V } from './state.svelte';
	import { argmax, pickToken, row, topIndices } from './analysis';
	import { argmaxAbs, decidedAt, lensPointName, meanSd, prevTokenHead, slices, type GenEntry } from './lesson';
	import { PRESETS } from './presets';
	import BlockView from './BlockView.svelte';
	import DotProduct from './DotProduct.svelte';
	import AttnText from './AttnText.svelte';
	import LogitLens from './LogitLens.svelte';

	type Fns = {
		samplingDistribution: (l: ArrayLike<number>, temp: number, k?: number, p?: number) => Float64Array;
		mulberry32: (seed: number) => () => number;
	};
	interface Props {
		/** the page state (the run of the current text) */
		S: TGState;
		/** the lesson's own selection, linked to S (see TGState) */
		LS: TGState;
		text: string;
		promptIds: number[];
		genIds: number[];
		/** a forward pass is running */
		busy: boolean;
		fns: Fns | null;
		setPrompt: (v: string) => void;
		/** append one token to the text and run the model on it */
		append: (token: number) => Promise<void>;
		/** wait until the run matches the current text */
		settle: () => Promise<void>;
		clearGenerated: () => void;
	}
	let { S, LS, text = $bindable(), promptIds, genIds, busy, fns, setPrompt, append, settle, clearGenerated }: Props = $props();

	let index = $state(0);
	// the lesson's own stage views are a separate chunk (the explore views above are already in the page)
	let st = $state.raw<typeof import('./stages') | null>(null);
	onMount(() => {
		import('./stages').then((m) => (st = m));
	});
	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const pos = $derived(LS.pos);
	const q = $derived(LS.label(pos));
	const kLab = $derived(LS.label(Math.min(LS.key, pos)));
	/** token labels quoted for the narration (a bare “.” or “,” is easy to miss in a sentence) */
	const qq = $derived(`“${q}”`);
	const kq = $derived(`“${kLab}”`);

	// ------------------------------------------------------------ head: the previous-token head of block 1, until the learner picks one
	let headTouched = $state(false);
	let keyTouched = $state(false);
	const auto = $derived(f ? prevTokenHead(f, 0, H) : { head: 0, avg: NaN });
	function enterAttention() {
		LS.block = 0;
		if (!headTouched) LS.head = auto.head;
	}
	function setHead(h: number) {
		headTouched = true;
		LS.head = h;
		LS.keySel = null;
		keyTouched = false;
	}
	// a new text: the most-attended key again
	let seenPromptForKey: number[] | null = null;
	$effect(() => {
		const p = promptIds;
		untrack(() => {
			if (p !== seenPromptForKey) {
				seenPromptForKey = p;
				keyTouched = false;
			}
		});
	});
	const ATTN = ['qkv', 'score', 'mask', 'softmax', 'mix'];
	const IN_BLOCK = ['ln', ...ATTN, 'mlp'];

	// ------------------------------------------------------------ live numbers for the narration
	const ln = $derived.by(() => {
		if (!f) return null;
		const x = row(f.t.embed.data, pos, D);
		return meanSd(x);
	});
	const score = $derived.by(() => {
		if (!f) return null;
		const k = Math.min(LS.key, pos);
		const o = (LS.head * T + pos) * T + k;
		return { s: f.t[`L${LS.block}.attn.scores`].data[o], a: f.t[`L${LS.block}.attn.weights`].data[o] };
	});
	const lensInfo = $derived.by(() => {
		const L = S.run?.lens;
		if (!L) return null;
		const p = decidedAt(L, pos);
		// the first point where the final answer is on top (it can lead early, lose the lead, and win again)
		let first = p;
		for (let k = 0; k < p; k++)
			if (L.topIds[(k * L.T + pos) * L.K] === L.finalTop[pos]) {
				first = k;
				break;
			}
		return { fin: S.idLabel(L.finalTop[pos]), at: p, where: lensPointName(L.points[p]), first, firstWhere: lensPointName(L.points[first]) };
	});
	const modelProbs = $derived(f ? row(f.t.probs.data, T - 1, V) : null);
	const lastLogits = $derived(f ? row(f.t.logits.data, T - 1, V) : null);
	const topNext = $derived(modelProbs ? argmax(modelProbs) : -1);
	const lastLensPoint = $derived((S.run?.lens.points.length ?? 1) - 1);

	// ------------------------------------------------------------ sampling and generation (the lesson's own settings)
	let temperature = $state(1);
	let topK = $state(0);
	const seed = 42;
	let rng: (() => number) | null = null;
	let draws = 0;
	let history = $state.raw<GenEntry[]>([]);
	let generating = $state(false);
	let lastMs = $state<number | null>(null);
	const dist = $derived.by(() => {
		if (!lastLogits || !fns) return null;
		if (temperature <= 0) {
			const d = new Float64Array(V);
			d[argmax(lastLogits)] = 1;
			return d;
		}
		return fns.samplingDistribution(lastLogits, temperature, topK, 1);
	});
	// a new prompt, or the generated tokens were removed (here or below): start the seeded generator again
	let seenPrompt: number[] | null = null;
	$effect(() => {
		const p = promptIds;
		const none = genIds.length === 0;
		untrack(() => {
			if (p !== seenPrompt || none) {
				seenPrompt = p;
				rng = null;
				draws = 0;
				history = [];
			}
		});
	});
	async function pickOne(): Promise<number> {
		await settle();
		const d = dist;
		const lg = lastLogits;
		const pm = modelProbs;
		if (!d || !lg || !pm || !fns) return -1;
		const after = S.label(S.T - 1);
		let token: number;
		let u: number | null = null;
		let draw: number | null = null;
		let lo = 0;
		let hi = 1;
		if (temperature <= 0) token = argmax(lg);
		else {
			if (!rng) {
				rng = fns.mulberry32(seed);
				draws = 0;
			}
			u = rng();
			draws += 1;
			draw = draws;
			({ token, lo, hi } = pickToken(d, u));
		}
		const entry: GenEntry = {
			n: history.length + 1,
			after,
			token,
			p: d[token],
			pModel: pm[token],
			u,
			draw,
			lo,
			hi,
			top: topIndices(d, 3).map((id) => ({ id, p: d[id] })),
			slices: u === null ? null : slices(d),
			temperature,
			topK,
			seed
		};
		history = [...history.map((h) => (h.slices ? { ...h, slices: null } : h)), entry];
		await append(token);
		lastMs = S.run?.ms ?? null;
		return token;
	}
	async function pickN(n: number) {
		generating = true;
		for (let k = 0; k < n && generating; k++) {
			const tok = await pickOne();
			if (tok < 0 || tok === 1023) break;
			if (!motion.reduced && k < n - 1) await new Promise((r) => setTimeout(r, 900));
		}
		generating = false;
	}
	async function pickSingle() {
		generating = true;
		await pickOne();
		generating = false;
	}

	// ------------------------------------------------------------ the twelve steps
	const steps: WalkStep[] = [
		{
			id: 'tokens',
			title: { en: 'Your text becomes tokens', zh: '你的文本变成词元' },
			narration: () => {
				const c = [...text].length;
				const n = promptIds.length;
				const g = genIds.length;
				return {
					en: `TinyGPT does not read letters. A tokenizer cuts your text into pieces from a vocabulary of 1,024 and replaces each piece by its number, its id. Your ${c} characters became ${n} tokens${g ? ` (plus ${g} written by the model)` : ''}. The model reads at most the last 64 tokens: its context window.`,
					zh: `TinyGPT 不直接读字母。分词器把你的文本切成词表（共 1024 个）里有的片段，再把每个片段换成它的编号，也就是 id。你的 ${c} 个字符变成了 ${n} 个词元${g ? `（另有 ${g} 个是模型写的）` : ''}。模型最多只读最后 64 个词元：这就是它的上下文窗口。`
				};
			},
			look: {
				en: 'Common words are one token, usually with their leading space (␣); rare words are split. The lesson follows the last token, because the next word is predicted there. Click another chip to follow that one instead.',
				zh: '常见的词是一个词元，通常带着前面的空格（␣）；少见的词会被拆开。课程默认跟踪最后一个词元，因为下一个词就在那里预测。点击别的色块，可以改为跟踪它。'
			}
		},
		{
			id: 'embed',
			title: { en: 'Each token becomes a vector', zh: '每个词元变成一个向量' },
			narration: () => {
				const id = f ? f.ids[pos] : 0;
				const full = promptIds.length + genIds.length > T;
				return {
					en: `The id of ${qq}, ${id}, picks row ${id} of a learned table that holds 76 numbers for every token: its token embedding. A second table adds where the token stands: row ${pos} of the position table (positions count from 0, so token ${pos + 1} ${full ? 'of the 64 in the window' : 'of your text'} is at position ${pos}). Their sum, 76 numbers, is what block 1 receives for this token.`,
					zh: `${qq}的 id 是 ${id}，它选出一张学到的表的第 ${id} 行，这张表为每个词元存了 76 个数：这就是词元嵌入。第二张表再加上这个词元所在的位置：位置表的第 ${pos} 行（位置从 0 开始数，所以${full ? '窗口中' : '你的文本'}的第 ${pos + 1} 个词元在位置 ${pos}）。两者相加得到 76 个数，就是第 1 块为这个词元收到的输入。`
				};
			},
			look: {
				en: 'The three rows of bars share one scale, so you can check the sum bar by bar. Each bar is one of the 76 numbers, called a dimension (numbered 0 to 75). Click a bar to work out that dimension below.',
				zh: '三行柱子使用同一个刻度，可以逐根核对加法。每根柱子是 76 个数中的一个，叫作一个维度（编号 0 到 75）。点击一根柱子，下方会算出那一维。'
			},
			onEnter: () => {
				if (f) LS.dim = argmaxAbs(row(f.t.embed.data, pos, D));
			}
		},
		{
			id: 'ln',
			title: { en: 'LayerNorm steadies each vector', zh: '层归一化让每个向量保持稳定' },
			narration: () => ({
				en: `Before attention, LayerNorm rescales each token’s vector on its own. The 76 numbers of ${qq} have mean ${fmt3(ln?.mean ?? 0)} and spread ${fmt3(ln?.sd ?? 0)} (the standard deviation: how far they typically lie from their mean). Subtract the mean and divide by the spread: now they have mean 0 and spread 1. A learned gain γ and shift β per dimension come last. However large or small a vector is, it leaves at a steady size.`,
				zh: `在注意力之前，层归一化把每个词元的向量单独重新缩放。${qq}的 76 个数均值为 ${fmt3(ln?.mean ?? 0)}，标准差为 ${fmt3(ln?.sd ?? 0)}（标准差衡量这些数通常离均值有多远）。减去均值、除以标准差之后，均值变成 0，标准差变成 1。最后每个维度再乘以学到的增益 γ、加上偏移 β。不管向量原来是大是小，出来时的大小都很稳定。`
			}),
			look: {
				en: 'The shaded band is mean ± spread. In block 1 the input numbers are small, so the first band is thin; after dividing by the spread the bars fill the scale.',
				zh: '色带表示均值 ± 标准差。第 1 块的输入数值很小，所以第一条色带很窄；除以标准差之后，柱子铺满了刻度。'
			},
			onEnter: () => (LS.block = 0)
		},
		{
			id: 'qkv',
			title: { en: 'Queries, keys and values', zh: '查询、键和值' },
			narration: () => {
				const h = LS.head + 1;
				const isAuto = !headTouched && Number.isFinite(auto.avg) && LS.head === auto.head;
				return {
					en: `Attention lets a token gather information from the tokens before it. It runs as 4 heads. Each head makes three vectors of 19 numbers for every token, each with its own learned weights: a query (what this token is looking for), a key (what it offers) and a value (what it passes on if chosen). This lesson uses head ${h} of block 1${isAuto ? `: on your text it puts ${fmtPct(auto.avg)} of its attention on the previous token, on average` : ''}.`,
					zh: `注意力让一个词元从它前面的词元那里收集信息。它分成 4 个头。每个头为每个词元算出三个各含 19 个数的向量，各用自己学到的权重：查询（这个词元在找什么）、键（它能提供什么）和值（被选中时传出去的内容）。本课程使用第 1 块的第 ${h} 个头${isAuto ? `：在你的文本上，它平均把 ${fmtPct(auto.avg)} 的注意力放在前一个词元上` : ''}。`
				};
			},
			look: {
				en: 'Top: the followed token’s normalized vector and its q, k and v in this head. Below: the same for every token, one row each, with one query number worked out. Switch heads above the stage.',
				zh: '上方：跟踪的词元归一化后的向量，以及它在这个头里的 q、k、v。下方：每个词元一行，并算出查询中的一个数。可以在舞台上方切换注意力头。'
			},
			onEnter: enterAttention
		},
		{
			id: 'score',
			title: { en: 'One attention score by hand', zh: '亲手算一个注意力分数' },
			narration: () => {
				const s = score?.s ?? 0;
				return {
					en: `How well does the query of ${qq} match the key of ${kq}? Multiply the two vectors entry by entry, add up the 19 products, and divide by √19 ≈ 4.36 so that scores do not grow with the vector length. The score is ${fmt3(s)}.${s < 0 ? ' Negative is fine: what counts is how it compares with this query’s other scores (step 7).' : ''} The model does this for every pair of tokens.`,
					zh: `${qq}的查询和${kq}的键有多匹配？把两个向量逐项相乘，19 个乘积相加，再除以 √19 ≈ 4.36，免得分数随向量长度变大。得到的分数是 ${fmt3(s)}。${s < 0 ? '分数为负没关系：重要的是它和这个查询的其他分数相比如何（第 7 步）。' : ''}模型对每一对词元都这样算一遍。`
				};
			},
			look: {
				en: 'Where q and k have the same sign the product is positive (orange) and raises the score. The lesson starts with the key this query attends to most; pick another key above the table.',
				zh: 'q 和 k 符号相同时乘积为正（橙色），会让分数变大。课程先选这个查询最关注的键；可以在表格上方换一个键。'
			},
			onEnter: () => {
				enterAttention();
				if (!keyTouched) LS.keySel = null;
			}
		},
		{
			id: 'mask',
			title: { en: 'The causal mask: no peeking ahead', zh: '因果掩码：不许偷看后面' },
			narration: {
				en: 'Scores are computed for every query and every key, including keys that come later in the text. But TinyGPT learns to predict the next token, so a token must never see the tokens after it. Those scores are replaced by −∞ (hatched). The first token may only look at itself; the last one sees every token.',
				zh: '分数是对每个查询和每个键都算的，也包括文本中更靠后的键。但 TinyGPT 学的是预测下一个词元，所以一个词元绝不能看到它后面的词元。这些分数被换成 −∞（阴影）。第一个词元只能看自己；最后一个词元能看到所有词元。'
			},
			look: {
				en: 'Watch the upper-right triangle turn into hatching. The framed cell is the score from step 5; click any cell to pick another query and key.',
				zh: '看右上三角变成阴影。带框的格子就是第 5 步算出的分数；点击任意格子可以换一个查询和键。'
			},
			waitForDone: true,
			linger: 6000,
			onEnter: enterAttention
		},
		{
			id: 'softmax',
			title: { en: 'Softmax turns scores into weights', zh: 'Softmax 把分数变成权重' },
			narration: () => ({
				en: `For one query, softmax turns its row of scores into weights: subtract the largest score, raise e to each, and divide by their sum. Only differences within the row matter, so negative scores are fine. Every weight is between 0 and 1, the row adds up to 1, and masked keys get exactly 0. Here ${qq} gives ${fmtPct(score?.a ?? 0)} of its attention to ${kq}.`,
				zh: `对一个查询，softmax 把它那一行分数变成权重：减去最大分数，对每个取 e 的幂，再除以它们的和。只有同一行内的差别才重要，所以分数为负也没关系。每个权重都在 0 和 1 之间，整行加起来等于 1，被掩码的键恰好得到 0。这里${qq}把 ${fmtPct(score?.a ?? 0)} 的注意力给了${kq}。`
			}),
			look: {
				en: 'Under the table, the text is shaded by these weights (darker purple = more attention; hover or focus a word to see where it looks), and the full map shows every row, each with its own softmax. Each row adds up to 1.',
				zh: '表格下方的文本按这些权重着色（紫色越深，注意力越多；把鼠标移到或聚焦一个词上，看它在看哪里），完整的图则显示每一行，每行各做一次 softmax，每行加起来都等于 1。'
			},
			waitForDone: true,
			linger: 6000,
			onEnter: enterAttention
		},
		{
			id: 'mix',
			title: { en: 'Mix the values, join the heads, add', zh: '混合值向量、拼接各头、相加' },
			narration: {
				en: 'With these weights, each head mixes the value vectors of the tokens it attends to: a weighted sum of 19 numbers. The 4 heads’ results are joined into 76 numbers, a 76 × 76 output projection mixes them, and the result is added to the vector that came in (x + attention). Attention adds to this residual stream; it never replaces it.',
				zh: '用这些权重，每个头把它关注的词元的值向量混合起来：19 个数的加权和。4 个头的结果拼成 76 个数，经过一个 76 × 76 的输出投影混合，再加到进来的向量上（x + 注意力）。注意力只往这条残差流里加东西，从不替换它。'
			},
			look: {
				en: 'The player walks through the four sub-steps; pause it or step through yourself. In the last one, compare the size of the update with the vector it is added to.',
				zh: '播放器会依次走过四个小步骤；可以暂停，或自己一步步看。在最后一步，比较这次更新和被加的向量各有多大。'
			},
			waitForDone: true,
			linger: 3500,
			onEnter: enterAttention
		},
		{
			id: 'mlp',
			title: { en: 'The MLP: 76 → 304 → GELU → 76', zh: 'MLP：76 → 304 → GELU → 76' },
			narration: {
				en: 'Now every token is processed on its own, with no information from other tokens. LayerNorm again, then a linear layer widens the 76 numbers to 304, GELU bends them (negative values are squashed toward 0), and a second linear layer brings them back to 76. The result is added to the stream again. Just over half of TinyGPT’s 364,040 parameters sit in these MLPs.',
				zh: '现在每个词元单独处理，不再从别的词元那里拿信息。先再做一次层归一化，然后一个线性层把 76 个数扩展到 304 个，GELU 把它们弯一下（负值被压向 0），第二个线性层再把它们变回 76 个。结果再次加回残差流。TinyGPT 的 364,040 个参数里，一半多一点在这些 MLP 中。'
			},
			look: {
				en: 'In the GELU sub-step every dot is one of the 304 numbers of your token: the blue ones (negative inputs) end up near 0.',
				zh: '在 GELU 这一小步里，每个点都是你的词元的 304 个数之一：蓝色的点（负的输入）最后都接近 0。'
			},
			waitForDone: true,
			linger: 3500,
			onEnter: () => (LS.block = 0)
		},
		{
			id: 'lens',
			title: { en: 'Four blocks in a row: the guess takes shape', zh: '4 个块依次处理：猜测逐渐成形' },
			narration: () => {
				const L = lensInfo;
				const intro = {
					en: 'Blocks 2, 3 and 4 repeat the same recipe with their own weights, each adding to the stream. The logit lens peeks at the stream after every step, as if the model had to answer right there.',
					zh: '第 2、3、4 块用各自的权重重复同样的步骤，每一步都往残差流里加东西。logit lens 在每一步之后偷看残差流，就好像模型必须当场回答。'
				};
				if (!L) return intro;
				const fin = `“${L.fin}”`;
				// the answer can lead early, lose the lead and win it back: say so instead of "only at the end"
				const early = L.first < L.at;
				const enEarly = early ? `is first on top at ${L.firstWhere.en}, then loses the lead, and ` : '';
				const zhEarly = early ? `先在${L.firstWhere.zh}排到第一，随后又被超过，` : '';
				let en: string;
				let zh: string;
				if (L.at === 0) {
					en = `After ${qq}, the final answer ${fin} is already the top guess right after the embedding.`;
					zh = `在${qq}之后，最终答案${fin}在嵌入之后就已经排第一了。`;
				} else if (L.at === lastLensPoint) {
					en = `After ${qq}, the final answer ${fin} ${enEarly}${early ? 'only settles as the top guess' : 'only becomes the top guess'} at the very last step, block 4 after the MLP: the model decides late here.`;
					zh = `在${qq}之后，最终答案${fin}${zhEarly}直到最后一步（第 4 块的 MLP 之后）才${early ? '稳定地' : ''}排到第一：这里模型是很晚才决定的。`;
				} else {
					en = early
						? `After ${qq}, the final answer ${fin} ${enEarly}stays the top guess from ${L.where.en} to the end.`
						: `After ${qq}, the final answer ${fin} becomes the top guess at ${L.where.en} and stays there.`;
					zh = early ? `在${qq}之后，最终答案${fin}${zhEarly}从${L.where.zh}开始才稳定地排在第一。` : `在${qq}之后，最终答案${fin}从${L.where.zh}开始排第一，并一直保持。`;
				}
				return { en: `${intro.en} ${en}`, zh: `${intro.zh}${zh}` };
			},
			look: {
				en: 'The thin bars: the probability of the final answer after each step, with its rank (#). Watch it climb, and the top guess change, as the blocks add to the stream. Pick another token in “Your text” to see its ladder.',
				zh: '细条形：每一步之后最终答案的概率，以及它的排名（#）。随着各个块往残差流里加东西，看它如何上升，第一猜测又如何变化。在“你的文本”里选另一个词元，可以看它的这张表。'
			}
		},
		{
			id: 'probs',
			title: { en: 'Logits become probabilities', zh: 'Logits 变成概率' },
			narration: () => ({
				en: `After block 4 and a final LayerNorm, the output head gives each of the 1,024 tokens a score, its logit. Softmax turns the logits into probabilities: at temperature 1, “${topNext >= 0 ? S.idLabel(topNext) : '…'}” gets ${fmtPct(modelProbs && topNext >= 0 ? modelProbs[topNext] : 0)}. Temperature divides the logits first (below 1 sharper, above 1 flatter), and top-k keeps only the k most likely tokens.`,
				zh: `经过第 4 块和最后的层归一化，输出头给 1024 个词元各打一个分数，即 logit。softmax 把 logits 变成概率：在温度 1 时，“${topNext >= 0 ? S.idLabel(topNext) : '…'}”得到 ${fmtPct(modelProbs && topNext >= 0 ? modelProbs[topNext] : 0)}。在此之前，logits 会先除以温度（温度小于 1 分布更尖锐，大于 1 更平坦），top-k 则只保留最可能的 k 个词元。`
			}),
			look: {
				en: 'Move the temperature slider and watch the bars of the “Sampling distribution”. The model’s own probabilities do not change; only the way we sample from them does.',
				zh: '拖动温度滑块，看“采样分布”那一组条形。模型自己的概率不会变；变的只是我们从中抽样的方式。'
			}
		},
		{
			id: 'generate',
			title: { en: 'Pick a token, append it, repeat', zh: '选一个词元，接上去，再来一遍' },
			narration: {
				en: 'To write, the model draws one token at random from these probabilities, appends it to the text, and runs everything again on the longer text, from step 1. Repeat, and a story appears one token at a time. The random numbers come from a seed (here 42), a starting value that fixes the whole sequence: the same text, seed and settings always give the same story.',
				zh: '写作时，模型按这些概率随机抽出一个词元，把它接到文本末尾，然后在更长的文本上从第 1 步开始把一切重新算一遍。不断重复，故事就一个词元一个词元地出现。随机数来自一个种子（这里是 42），种子固定了整串随机数：相同的文本、种子和设置，总会得到相同的故事。'
			},
			look: {
				en: 'Press “Pick the next token”. The new token appears in magenta here and in your input above, and every step of this lesson now shows the longer text.',
				zh: '点“选出下一个词元”。新词元会以品红色出现在这里和上方的输入中，课程的每一步现在都显示更长的文本。'
			}
		}
	];
	const stepIndex = (id: string) => steps.findIndex((s) => s.id === id);

	// ------------------------------------------------------------ the aside: your text, the followed token
	let editOpen = $state(false);
	const genInWindow = $derived(Math.min(genIds.length, T));
	const chips = $derived<TokenItem[]>(
		f && S.tok ? f.ids.map((id, i) => ({ text: S.tok!.tokenToDisplay(id), id, tone: i >= T - genInWindow ? 'generated' : undefined })) : []
	);
	let chipSel = $state<number | null>(null);
	$effect(() => {
		chipSel = T ? pos : null;
	});
	// keep the followed token visible in the small (scrolling) token box
	let miniEl: HTMLElement | undefined = $state();
	$effect(() => {
		void pos;
		void T;
		const box = miniEl;
		if (!box) return;
		requestAnimationFrame(() => {
			const c = box.querySelector('[aria-selected="true"]');
			if (!c) return;
			const d = c.getBoundingClientRect().top - box.getBoundingClientRect().top;
			if (d < 0 || d > box.clientHeight - 30) box.scrollTop += d - box.clientHeight / 2;
		});
	});
</script>

<Walkthrough {steps} bind:index ready={!!S.run} label={{ en: 'TinyGPT, step by step', zh: 'TinyGPT 分步课程' }} stageMinHeight={420} narrationFirst>
	{#snippet stage(step, ctx)}
		{#if !S.run || !st}
			<p class="muted">{t({ en: 'Loading the model… (type some text above if the input is empty)', zh: '正在加载模型……（如果输入为空，请先在上方输入一些文字）' })}</p>
		{:else}
			<div class="sh">
				<span class="ctx">
					{#if step.id === 'tokens'}
						{t({ en: 'Tokenizer', zh: '分词器' })}
					{:else if step.id === 'probs' || step.id === 'generate'}
						{t({ en: 'Output, after the last token', zh: '输出，在最后一个词元之后' })} <span class="tok">{S.label(T - 1)}</span>
					{:else}
						{t({ en: 'Following', zh: '跟踪' })} <span class="tok">{q}</span>
						{#if IN_BLOCK.includes(step.id)}<span class="muted">· {t({ en: 'block 1', zh: '第 1 块' })}</span>{/if}
					{/if}
					{#if ATTN.includes(step.id)}
						<span class="heads">
							<span class="muted">· {t({ en: 'head', zh: '头' })}</span>
							<span class="segmented" role="group" aria-label={t({ en: 'Attention head', zh: '注意力头' })}>
								{#each Array.from({ length: H }, (_, i) => i) as i (i)}
									<button type="button" aria-pressed={LS.head === i} onclick={() => setHead(i)}>{i + 1}</button>
								{/each}
							</span>
						</span>
					{/if}
				</span>
				<ProvenanceBadge mode="live" compact precision="fp32" />
			</div>
			{#if step.id === 'tokens'}
				<st.LessonTokens {LS} {text} {promptIds} {genIds} />
			{:else if step.id === 'embed'}
				<st.EmbedStage {LS} />
			{:else if step.id === 'ln'}
				<st.LnStage {LS} />
			{:else if step.id === 'qkv'}
				<BlockView S={LS} focus only={[1]} />
			{:else if step.id === 'score'}
				<div class="keypick">
					<span class="lbl">{t({ en: 'Key', zh: '键' })}</span>
					<TokenChips
						tokens={S.labels.slice(0, pos + 1).map((lab, j) => ({ text: lab, id: f?.ids[j] }))}
						selected={Math.min(LS.key, pos)}
						onselect={(j) => {
							keyTouched = true;
							LS.selectKey(j);
						}}
						label={{ en: 'Pick the key (only tokens up to the query)', zh: '选择键（只能是查询及之前的词元）' }}
					/>
				</div>
				<DotProduct S={LS} parts={['table', 'score', 'check']} />
			{:else if step.id === 'mask'}
				<BlockView S={LS} focus only={[2, 3]} interval={4000} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
			{:else if step.id === 'softmax'}
				<st.SoftmaxRow {LS} interval={3000} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
				<div class="attn">
					<AttnText S={LS} />
					{#if f}
						<div class="amap">
							<AttentionMatrix
								weights={f.t[`L${LS.block}.attn.weights`].data.subarray(LS.head * T * T, (LS.head + 1) * T * T)}
								queries={S.labels}
								causal
								row={pos}
								col={Math.min(LS.key, pos)}
								maxCell={22}
								label={{ en: `Every row gets its own softmax: all weights of block 1, head ${LS.head + 1}`, zh: `每一行各做一次 softmax：第 1 块第 ${LS.head + 1} 头的全部权重` }}
							/>
						</div>
					{/if}
				</div>
			{:else if step.id === 'mix'}
				<BlockView S={LS} focus only={[5, 6, 7, 8]} interval={3600} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
			{:else if step.id === 'mlp'}
				<BlockView S={LS} focus only={[9, 10, 11, 12, 13]} interval={3600} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
			{:else if step.id === 'lens'}
				<LogitLens S={LS} parts={['ladder']} />
			{:else if step.id === 'probs'}
				<st.LogitsStage {LS} {dist} bind:temperature bind:topK />
			{:else if step.id === 'generate'}
				<st.GenStage
					{LS}
					{dist}
					{temperature}
					{topK}
					{seed}
					promptText={text}
					{genIds}
					{history}
					{busy}
					{generating}
					{lastMs}
					onpick={() => void pickSingle()}
					onpickN={(n) => void pickN(n)}
					onstop={() => (generating = false)}
					onclear={() => {
						generating = false;
						clearGenerated();
					}}
					onsettings={() => (index = stepIndex('probs'))}
				/>
			{/if}
		{/if}
	{/snippet}

	{#snippet aside()}
		<div class="you">
			<div class="you-h">
				<h4>{t({ en: 'Your text', zh: '你的文本' })}</h4>
				{#if modelProbs && topNext >= 0}
					<span class="ans">{t({ en: 'next:', zh: '下一个：' })} <span class="tok">{S.idLabel(topNext)}</span> <span class="num">{fmtPct(modelProbs[topNext])}</span></span>
				{/if}
			</div>
			{#if chips.length}
				<div class="mini" bind:this={miniEl}>
					<TokenChips tokens={chips} bind:selected={chipSel} onselect={(i) => LS.selectPos(i)} label={{ en: 'Tokens in the context window (click one to follow it)', zh: '上下文窗口中的词元（点击一个来跟踪它）' }} />
				</div>
				<p class="hint">
					{t({ en: `Following token ${pos + 1} of ${T} (position ${pos}).`, zh: `正在跟踪第 ${pos + 1} 个词元（共 ${T} 个，位置 ${pos}）。` })}
					{#if LS.posSel !== null}
						<button type="button" class="link" onclick={() => LS.selectPos(T - 1)}>{t({ en: 'Follow the last token', zh: '改为跟踪最后一个词元' })}</button>
					{/if}
					{t({ en: 'Edit the text at any time: the step you are on updates at once.', zh: '随时可以修改文本：你所在的这一步会立刻更新。' })}
				</p>
			{/if}
			{#if editOpen}
				<TextInputBar
					bind:value={text}
					maxLength={400}
					multiline
					rows={2}
					presets={PRESETS}
					label={{ en: 'Lesson text', zh: '课程文本' }}
					placeholder={{ en: 'Type the start of a story…', zh: '输入一个故事的开头（英文）…' }}
					oninput={(v) => setPrompt(v)}
				/>
			{/if}
			<button type="button" class="btn small" aria-expanded={editOpen} onclick={() => (editOpen = !editOpen)}>
				<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5H3.5V10z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>
				{editOpen ? t({ en: 'Hide the text box', zh: '收起文本框' }) : t({ en: 'Edit the text here', zh: '在这里修改文本' })}
			</button>
		</div>
	{/snippet}

	{#snippet end()}
		<div class="fin">
			<p>
				<strong>{t({ en: 'That is the whole model.', zh: '这就是整个模型。' })}</strong>
				{t({
					en: 'Tokens → vectors → 4 blocks of attention and MLP → probabilities → one new token, again and again. Try another text and play the lesson again, or see every tensor at once below.',
					zh: '词元 → 向量 → 4 个由注意力和 MLP 组成的块 → 概率 → 一个新词元，如此反复。换一段文本再播放一遍课程，或者在下面同时查看所有张量。'
				})}
			</p>
			<a class="btn primary" href="#explore">{t({ en: 'Explore freely', zh: '自由探索' })} ↓</a>
		</div>
	{/snippet}
</Walkthrough>
<p class="foot muted">
	{t({
		en: 'Every number in the lesson is computed in your browser from your text and the trained weights. The lesson keeps its own choices (followed token, head, key, dimension, sampling settings), so the views below stay where you left them; the text is shared, so tokens generated in step 12 appear everywhere.',
		zh: '课程中的每个数都由你的文本和训练好的权重在浏览器中实时算出。课程有自己的选择（跟踪的词元、注意力头、键、维度、采样设置），所以下面的视图会停在你离开时的位置；文本是共用的，所以第 12 步生成的词元会出现在页面各处。'
	})}
</p>

<style>
	.sh {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 6px 12px;
		margin: -4px 0 10px;
		padding-bottom: 8px;
		border-bottom: 1px solid var(--rule);
	}
	.ctx {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 6px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.heads {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.heads .segmented button {
		min-width: 34px;
		min-height: 32px;
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
		color: var(--ink);
		box-shadow: inset 0 0 0 1.5px var(--q);
		font-variant-ligatures: none;
	}
	.keypick {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		margin-bottom: 10px;
	}
	.keypick .lbl {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.keypick :global(.chips) {
		flex: 1 1 200px;
	}
	.attn {
		display: grid;
		gap: 14px;
		margin-top: 12px;
	}
	.amap {
		max-width: 460px;
		min-width: 0;
	}
	.you {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
		min-width: 0;
	}
	.you-h {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 4px 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.ans {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		font-size: var(--fs-sm);
	}
	.ans .tok {
		box-shadow: none;
	}
	.mini {
		max-height: 150px;
		overflow-y: auto;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.link {
		border: 0;
		background: none;
		padding: 0;
		color: var(--accent-strong);
		text-decoration: underline;
		cursor: pointer;
		font: inherit;
	}
	.small {
		justify-self: start;
		gap: 6px;
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
	.you :global(.presets .btn),
	.you :global(.presets button) {
		min-height: 30px;
		font-size: var(--fs-xs);
	}
	.fin {
		display: grid;
		gap: 8px;
		justify-items: start;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--live-soft);
	}
	.fin p {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.foot {
		margin: 8px 0 0;
		font-size: var(--fs-xs);
	}
</style>
