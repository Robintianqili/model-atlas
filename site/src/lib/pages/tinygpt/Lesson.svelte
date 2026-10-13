<!--
  The TinyGPT guided lesson: twelve steps from the learner's text to the next token, one focused
  view at a time (kit Walkthrough). Each stage reuses the page's own views (the block player, the
  dot product, the logit lens, the sampling controls) or a small lesson view, all on the live run of
  the current text. The lesson has its own selection (LS: followed token, head, key, dimension), so
  it never moves the "Explore freely" views; the text itself is shared, so editing it here, or
  generating tokens in step 12, updates the whole page. Laid out as the lesson app (redesign spec §5): the input chip in
  the stage header opens the text sheet, the stage header's "Following" menu picks the token the lesson follows, every
  stage carries one provenance label, and the steps are grouped in four chapters (Tokens 2 · Attention 6 · Blocks 2 ·
  Next token 2). The cover (answer first) is Cover.svelte; the end card's recap names three real numbers of this run.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt, fmtPct } from '$lib/kit/format';
	import type { StageProv, WalkStep } from '$lib/kit/types';
	import type { TGPage } from './page.svelte';
	import { D, H, V } from './state.svelte';
	import { argmax, pickToken, row, topIndices } from './analysis';
	import { argmaxAbs, decidedAt, lensPointName, meanSd, prevTokenHead, slices, type GenEntry } from './lesson';
	import Cover from './Cover.svelte';

	let { P }: { P: TGPage } = $props();
	const S = untrack(() => P.S);
	const LS = untrack(() => P.LS);
	const text = $derived(P.text);
	const promptIds = $derived(P.promptIds);
	const genIds = $derived(P.genIds);
	const busy = $derived(P.busy);
	const fns = $derived(P.fns);
	const settle = () => P.settle();
	const append = (token: number) => P.appendToken(token);
	const clearGenerated = () => P.removeGenerated();

	/** the stage header's provenance label: every stage is computed live from the learner's text */
	const live: StageProv = { mode: 'live', precision: 'fp32', tolerance: 1e-4, against: 'PyTorch fp32', short: { en: '· computed on your text', zh: '· 用你的文本现场计算' } };
	const chapters: { title: Text; count: number }[] = [
		{ title: { en: 'Tokens', zh: '词元' }, count: 2 },
		{ title: { en: 'Attention', zh: '注意力' }, count: 6 },
		{ title: { en: 'Blocks', zh: '块' }, count: 2 },
		{ title: { en: 'Next token', zh: '下一个词元' }, count: 2 }
	];

	let index = $state(0);
	// the lesson's own stage views are a separate chunk (they render only once the model has run)
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
			prov: live,
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
			prov: live,
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
			prov: live,
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
			prov: live,
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
				en: 'Top: the followed token’s normalized vector and its q, k and v in this head. Below: the same for every token, one row each, with one query number worked out. Switch heads with the head buttons at the top of the stage.',
				zh: '上方：跟踪的词元归一化后的向量，以及它在这个头里的 q、k、v。下方：每个词元一行，并算出查询中的一个数。可以用演示区顶部的头按钮切换注意力头。'
			},
			onEnter: enterAttention
		},
		{
			id: 'score',
			prov: live,
			title: { en: 'One attention score by hand', zh: '亲手算一个注意力分数' },
			narration: () => {
				const s = score?.s ?? 0;
				return {
					en: `How well does the query of ${qq} match the key of ${kq}? Multiply the two vectors entry by entry, add up the 19 products, and divide by √19 ≈ 4.36 so that scores do not grow with the vector length. The score is ${fmt3(s)}.${s < 0 ? ' Negative is fine: what counts is how it compares with this query’s other scores (step 7).' : ''} The model does this for every pair of tokens.`,
					zh: `${qq}的查询和${kq}的键有多匹配？把两个向量逐项相乘，19 个乘积相加，再除以 √19 ≈ 4.36，免得分数随向量长度变大。得到的分数是 ${fmt3(s)}。${s < 0 ? '分数为负没关系：重要的是它和这个查询的其他分数相比如何（第 7 步）。' : ''}模型对每一对词元都这样算一遍。`
				};
			},
			look: {
				en: 'Where q and k have the same sign the product is positive (orange) and raises the score. The lesson starts with the key this query attends to most; pick another key with the Key chips in the stage.',
				zh: 'q 和 k 符号相同时乘积为正（橙色），会让分数变大。课程先选这个查询最关注的键；可以用演示区里的“键”色块换一个键。'
			},
			onEnter: () => {
				enterAttention();
				if (!keyTouched) LS.keySel = null;
			}
		},
		{
			id: 'mask',
			prov: live,
			title: { en: 'The causal mask: no peeking ahead', zh: '因果掩码：不许偷看后面' },
			narration: {
				en: 'Scores are computed for every query and every key, including keys that come later in the text. But TinyGPT learns to predict the next token, so a token must never see the tokens after it. Those scores are replaced by −∞ (hatched). The first token may only look at itself; the last one sees every token.',
				zh: '分数是对每个查询和每个键都算的，也包括文本中更靠后的键。但 TinyGPT 学的是预测下一个词元，所以一个词元绝不能看到它后面的词元。这些分数被换成 −∞（阴影）。第一个词元只能看自己；最后一个词元能看到所有词元。'
			},
			look: {
				en: 'Watch the upper-right triangle turn into hatching. The framed cell is the score from step 5; click any cell to pick another query and key.',
				zh: '看右上三角变成阴影。带框的格子就是第 5 步算出的分数；点击任意格子可以换一个查询和键。'
			},
			animated: true,
			waitForDone: true,
			linger: 6000,
			onEnter: enterAttention
		},
		{
			id: 'softmax',
			prov: live,
			title: { en: 'Softmax turns scores into weights', zh: 'Softmax 把分数变成权重' },
			narration: () => ({
				en: `For one query, softmax turns its row of scores into weights: subtract the largest score, raise e to each, and divide by their sum. Only differences within the row matter, so negative scores are fine. Every weight is between 0 and 1, the row adds up to 1, and masked keys get exactly 0. Here ${qq} gives ${fmtPct(score?.a ?? 0)} of its attention to ${kq}.`,
				zh: `对一个查询，softmax 把它那一行分数变成权重：减去最大分数，对每个取 e 的幂，再除以它们的和。只有同一行内的差别才重要，所以分数为负也没关系。每个权重都在 0 和 1 之间，整行加起来等于 1，被掩码的键恰好得到 0。这里${qq}把 ${fmtPct(score?.a ?? 0)} 的注意力给了${kq}。`
			}),
			look: {
				en: 'Under the table, the text is shaded by these weights (darker purple = more attention; hover or focus a word to see where it looks), and the full map shows every row, each with its own softmax. Each row adds up to 1.',
				zh: '表格下方的文本按这些权重着色（紫色越深，注意力越多；把鼠标移到或聚焦一个词上，看它在看哪里），完整的图则显示每一行，每行各做一次 softmax，每行加起来都等于 1。'
			},
			animated: true,
			waitForDone: true,
			linger: 6000,
			onEnter: enterAttention
		},
		{
			id: 'mix',
			prov: live,
			title: { en: 'Mix the values, join the heads, add', zh: '混合值向量、拼接各头、相加' },
			narration: {
				en: 'With these weights, each head mixes the value vectors of the tokens it attends to: a weighted sum of 19 numbers. The 4 heads’ results are joined into 76 numbers, a 76 × 76 output projection mixes them, and the result is added to the vector that came in (x + attention). Attention adds to this residual stream; it never replaces it.',
				zh: '用这些权重，每个头把它关注的词元的值向量混合起来：19 个数的加权和。4 个头的结果拼成 76 个数，经过一个 76 × 76 的输出投影混合，再加到进来的向量上（x + 注意力）。注意力只往这条残差流里加东西，从不替换它。'
			},
			look: {
				en: 'The player walks through the four sub-steps; pause it or step through yourself. In the last one, compare the size of the update with the vector it is added to.',
				zh: '播放器会依次走过四个小步骤；可以暂停，或自己一步步看。在最后一步，比较这次更新和被加的向量各有多大。'
			},
			animated: true,
			waitForDone: true,
			linger: 3500,
			onEnter: enterAttention
		},
		{
			id: 'mlp',
			prov: live,
			title: { en: 'The MLP: 76 → 304 → GELU → 76', zh: 'MLP：76 → 304 → GELU → 76' },
			narration: {
				en: 'Now every token is processed on its own, with no information from other tokens. LayerNorm again, then a linear layer widens the 76 numbers to 304, GELU bends them (negative values are squashed toward 0), and a second linear layer brings them back to 76. The result is added to the stream again. Just over half of TinyGPT’s 364,040 parameters sit in these MLPs.',
				zh: '现在每个词元单独处理，不再从别的词元那里拿信息。先再做一次层归一化，然后一个线性层把 76 个数扩展到 304 个，GELU 把它们弯一下（负值被压向 0），第二个线性层再把它们变回 76 个。结果再次加回残差流。TinyGPT 的 364,040 个参数里，一半多一点在这些 MLP 中。'
			},
			look: {
				en: 'In the GELU sub-step every dot is one of the 304 numbers of your token: the blue ones (negative inputs) end up near 0.',
				zh: '在 GELU 这一小步里，每个点都是你的词元的 304 个数之一：蓝色的点（负的输入）最后都接近 0。'
			},
			animated: true,
			waitForDone: true,
			linger: 3500,
			onEnter: () => (LS.block = 0)
		},
		{
			id: 'lens',
			prov: live,
			title: { en: 'Four blocks in a row: the guess takes shape', zh: '4 个块依次处理：猜测逐渐成形' },
			narration: () => {
				const intro = {
					en: 'Blocks 2, 3 and 4 repeat the same recipe with their own weights, each adding to the stream. The logit lens peeks at the stream after every step, as if the model had to answer right there.',
					zh: '第 2、3、4 块用各自的权重重复同样的步骤，每一步都往残差流里加东西。logit lens 在每一步之后偷看残差流，就好像模型必须当场回答。'
				};
				const L = lensInfo;
				if (!L) return intro;
				return {
					en: `${intro.en} After ${qq}, the model finally answers “${L.fin}”. Predict when that answer takes the lead.`,
					zh: `${intro.zh}在${qq}之后，模型最终的回答是“${L.fin}”。先预测：这个回答是从哪里开始领先的？`
				};
			},
			// a gated predict (spec §5.7): the ladder in the stage stays hidden until the learner commits; the answer and its
			// real numbers come from the logit lens of the current text
			predict: () => {
				const L = lensInfo;
				if (!L) return undefined;
				// lens points: 0 the embedding, then two per block (after attention, after the MLP)
				const block = L.at === 0 ? 0 : Math.ceil(L.at / 2) - 1;
				const fin = `“${L.fin}”`;
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
				return {
					question: { en: `From which block on is ${fin} the top guess for good?`, zh: `${fin}从哪一块开始稳定地排在第一？` },
					options: [
						{ en: 'Block 1, or already before it', zh: '第 1 块，或者更早' },
						{ en: 'Block 2', zh: '第 2 块' },
						{ en: 'Block 3', zh: '第 3 块' },
						{ en: 'Only in block 4', zh: '直到第 4 块' }
					],
					answer: Math.max(0, Math.min(3, block)),
					explain: { en, zh }
				};
			},
			look: {
				en: 'The thin bars: the probability of the final answer after each step, with its rank (#). Watch it climb, and the top guess change, as the blocks add to the stream. Pick another token with Following at the top of the stage to see its ladder.',
				zh: '细条形：每一步之后最终答案的概率，以及它的排名（#）。随着各个块往残差流里加东西，看它如何上升，第一猜测又如何变化。用演示区顶部的“跟踪”菜单选另一个词元，可以看它的这张表。'
			}
		},
		{
			id: 'probs',
			prov: live,
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
			prov: live,
			title: { en: 'Pick a token, append it, repeat', zh: '选一个词元，接上去，再来一遍' },
			narration: {
				en: 'To write, the model draws one token at random from these probabilities, appends it to the text, and runs everything again on the longer text, from step 1. Repeat, and a story appears one token at a time. The random numbers come from a seed (here 42), a starting value that fixes the whole sequence: the same text, seed and settings always give the same story.',
				zh: '写作时，模型按这些概率随机抽出一个词元，把它接到文本末尾，然后在更长的文本上从第 1 步开始把一切重新算一遍。不断重复，故事就一个词元一个词元地出现。随机数来自一个种子（这里是 42），种子固定了整串随机数：相同的文本、种子和设置，总会得到相同的故事。'
			},
			look: {
				en: 'Press “Pick the next token”. The new token appears in magenta here and in your text, and every step of this lesson now shows the longer text.',
				zh: '点“选出下一个词元”。新词元会以品红色出现在这里和你的文本中，课程的每一步现在都显示更长的文本。'
			}
		}
	];
	const stepIndex = (id: string) => steps.findIndex((s) => s.id === id);

	// ------------------------------------------------------------ the followed token (the stage header's menu)
	function follow(e: Event) {
		LS.selectPos(Number((e.currentTarget as HTMLSelectElement).value));
		keyTouched = false;
	}
	const FOLLOWS = ['embed', 'ln', 'qkv', 'score', 'mask', 'softmax', 'mix', 'mlp', 'lens'];
</script>

<Walkthrough {steps} bind:index ready={!!S.run} label={{ en: 'Follow your text through a real GPT', zh: '跟着你的文本走过一个真实的 GPT' }} {chapters} stageMinHeight={420}>
	{#snippet cover(ctx)}<Cover {P} compact={ctx.density === 'compact'} />{/snippet}

	{#snippet stage(step, ctx)}
		{#if !S.run || !st}
			<p class="muted">{t({ en: 'Loading the model… If your text is empty, press Edit in the stage header and type some.', zh: '正在加载模型……如果你的文本是空的，请点演示区顶部的“编辑”输入一些文字。' })}</p>
		{:else}
			<div class="tg">
				<div class="sh">
					<span class="ctx">
						{#if step.id === 'tokens'}
							{t({ en: 'Tokenizer', zh: '分词器' })}
						{:else if step.id === 'probs' || step.id === 'generate'}
							{t({ en: 'Output, after the last token', zh: '输出，在最后一个词元之后' })} <span class="tok">{S.label(T - 1)}</span>
						{:else}
							<label class="follow">
								<span>{t({ en: 'Following', zh: '跟踪' })}</span>
								<select class="tok" value={pos} onchange={follow} disabled={!FOLLOWS.includes(step.id)}>
									{#each S.labels as lab, i (i)}<option value={i}>{lab}{i === T - 1 ? t({ en: ' (last)', zh: '（最后一个）' }) : ''}</option>{/each}
								</select>
							</label>
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
				</div>
				{#if step.id === 'tokens'}
					<st.LessonTokens {LS} {text} {promptIds} {genIds} />
				{:else if step.id === 'embed'}
					<st.EmbedStage {LS} />
				{:else if step.id === 'ln'}
					<st.LnStage {LS} />
				{:else if step.id === 'qkv'}
					<st.BlockView S={LS} focus only={[1]} />
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
					<st.DotProduct S={LS} parts={['table', 'score', 'check']} />
				{:else if step.id === 'mask'}
					<st.BlockView S={LS} focus only={[2, 3]} interval={4000} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
				{:else if step.id === 'softmax'}
					<st.SoftmaxRow {LS} interval={3000} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
					<div class="attn">
						<st.AttnText S={LS} />
						{#if f}
							<div class="amap">
								<st.AttentionMatrix
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
					<st.BlockView S={LS} focus only={[5, 6, 7, 8]} interval={3600} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
				{:else if step.id === 'mlp'}
					<st.BlockView S={LS} focus only={[9, 10, 11, 12, 13]} interval={3600} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
				{:else if step.id === 'lens'}
					<st.LogitLens S={LS} parts={['ladder']} />
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
			</div>
		{/if}
	{/snippet}

	{#snippet recap()}
		<ul class="recap">
			<li>
				{t({
					en: `Your ${fmtInt([...text].length)} characters became ${fmtInt(promptIds.length)} tokens; the model read the last ${T}${genIds.length ? `, ${genIds.length} of them written by itself` : ''}.`,
					zh: `你的 ${fmtInt([...text].length)} 个字符变成了 ${fmtInt(promptIds.length)} 个词元；模型读的是最后 ${T} 个${genIds.length ? `，其中 ${genIds.length} 个是它自己写的` : ''}。`
				})}
			</li>
			{#if score}
				<li>
					{t({
						en: `In block 1, head ${LS.head + 1}, ${qq} gave ${fmtPct(score.a)} of its attention to ${kq} (score ${fmt3(score.s)}).`,
						zh: `在第 1 块第 ${LS.head + 1} 个头里，${qq}把 ${fmtPct(score.a)} 的注意力给了${kq}（分数 ${fmt3(score.s)}）。`
					})}
				</li>
			{/if}
			{#if topNext >= 0 && modelProbs}
				<li>
					{t({
						en: `After “${S.label(T - 1)}”, it gave “${S.idLabel(topNext)}” ${fmtPct(modelProbs[topNext])} of the probability, out of ${fmtInt(V)} tokens.`,
						zh: `在“${S.label(T - 1)}”之后，它在 ${fmtInt(V)} 个词元里给了“${S.idLabel(topNext)}” ${fmtPct(modelProbs[topNext])} 的概率。`
					})}
				</li>
			{/if}
		</ul>
	{/snippet}
</Walkthrough>

<style>
	/* token text: no programming ligatures ("<|" would turn into a triangle in JetBrains Mono) */
	.tg {
		font-variant-ligatures: none;
	}
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
		color: var(--ink-2);
		font-size: var(--t-sm);
	}
	.follow {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.follow select {
		max-width: 14em;
		min-height: 32px;
		padding: 0 6px;
		border: 1px solid var(--line-control);
		border-radius: var(--r2);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-sm);
	}
	.follow select:disabled {
		border-style: dashed;
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
		padding: 0 4px;
		border-radius: 3px;
		background: var(--surface-2);
		box-shadow: inset 0 0 0 1.5px var(--q);
		color: var(--ink);
		font-family: var(--font-mono);
		font-variant-ligatures: none;
		white-space: pre;
	}
	.keypick {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		margin-bottom: 10px;
	}
	.keypick .lbl {
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
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
	.recap {
		display: grid;
		gap: 6px;
		margin: 0;
		padding-left: 1.2em;
		color: var(--ink-2);
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-variant-ligatures: none;
	}
</style>
