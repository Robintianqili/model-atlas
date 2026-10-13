<!--
  The tokenization guided lesson: byte-pair encoding in seven steps on the learner's own text, one focused view at a
  time (kit Walkthrough, the page's Lesson view). Steps 1–5 use the training text (shared with Explore's Part 1), steps
  6–7 and the cover the sentence (shared with Explore's Parts 2–3); both are edited in the input sheet the stage
  header's chip opens. The lesson keeps its own merge counts and selections, so the Explore views never move. Every
  number is computed in the browser: training in the page's Web Worker (TokPage.train), encoding with bpe.ts, the real
  tokenizers in the worker; only TinyGPT's training pair counts (step 6) are precomputed.
-->
<script lang="ts">
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt, visibleToken } from '$lib/kit/format';
	import { BPETokenizer, bytesToDisplay } from '$lib/models/bpe';
	import type { StageProv, WalkStep } from '$lib/kit/types';
	import BytesStage from './lesson/BytesStage.svelte';
	import PretokStage from './lesson/PretokStage.svelte';
	import PairsStage from './lesson/PairsStage.svelte';
	import MergeStage from './lesson/MergeStage.svelte';
	import RepeatStage from './lesson/RepeatStage.svelte';
	import ApplyStage from './lesson/ApplyStage.svelte';
	import RealStage from './lesson/RealStage.svelte';
	import CoverStage from './CoverStage.svelte';
	import { segment } from './train';
	import { charSummary, firstSingleton, pairSources, plainToken, pretokenStats, tokensAt } from './lesson';
	import { loadAll } from './state.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { PARITY, type RowId } from './tokenizers';
	import type { TokPage } from './page.svelte';

	let { P }: { P: TokPage } = $props();
	const trainText = $derived(P.trainText);
	const sentence = $derived(P.sentence);
	const trace = $derived(P.trace);
	const trained = $derived(P.trained);
	const tiny = $derived(P.tiny);
	const mergeCounts = $derived(P.mergeCounts);

	// ------------------------------------------------------------ the lesson's own state
	let index = $state(0);
	let pairSel = $state(0);
	let mergeN = $state(1);
	let lk = $state(1);
	let src = $state<'yours' | 'tiny'>('yours');
	let counts = $state<Partial<Record<RowId, number>>>({});

	// a new training text starts the merge views from the beginning
	let lastTrained = '';
	$effect(() => {
		const tx = trained;
		if (tx !== lastTrained) {
			lastTrained = tx;
			pairSel = 0;
			mergeN = 1;
		}
	});

	const ready = $derived(!!tiny && !!trace);
	const K = $derived(trace?.steps.length ?? 0);
	const pattern = $derived(tiny?.pattern ?? '');
	/** the learner's finished tokenizer: all merges learned from the training text */
	const yours = $derived(trace && tiny ? BPETokenizer.fromMerges(trace.merges, tiny.pattern) : null);
	const yourCounts = $derived(trace ? trace.steps.map((s) => s.count) : []);
	const disp = (id: number) => (trace ? bytesToDisplay(trace.bytes[id]) : '');
	const cs = $derived(charSummary(trainText));
	/** the learner's first byte (or "a" for an empty text), to show how base 16 reads */
	const b0 = $derived(new TextEncoder().encode(trainText || 'a')[0]);
	const hx = (n: number) => n.toString(16).toUpperCase().padStart(2, '0');
	const ps = $derived(tiny ? pretokenStats(trainText, tiny.pattern) : null);
	const q = (s: string) => `“${s}”`;
	const LOADING = { en: 'Loading TinyGPT’s tokenizer and training on your text…', zh: '正在加载 TinyGPT 的分词器，并在你的文本上训练……' };

	const steps: WalkStep[] = [
		{
			id: 'bytes',
			title: { en: 'Your text becomes bytes', zh: '文本先变成字节' },
			narration: () => {
				const h = hx(b0);
				const hi = b0 >> 4;
				const lo = b0 & 15;
				return {
					en: `Computers store text as bytes, using a code called UTF-8. A byte is a whole number from 0 to 255, written here in base 16 (digits 0–9, then A–F): your first byte, ${h}, is ${hi} × 16 + ${lo} = ${b0}. English letters, digits, spaces and basic punctuation take 1 byte each; a Chinese character takes 3. Your text has ${fmtInt(cs.chars)} characters and ${fmtInt(cs.bytes)} bytes. BPE (byte-pair encoding) starts here: each byte is one token, a piece the model reads, so the first vocabulary (the list of every token the tokenizer knows) is just the 256 byte values, with IDs 0 to 255.`,
					zh: `计算机用一种叫 UTF-8 的编码把文本存成字节。一个字节是 0 到 255 之间的整数，这里用十六进制写出（数字 0–9，再接 A–F）：你的第一个字节 ${h} 就是 ${hi} × 16 + ${lo} = ${b0}。英文字母、数字、空格和半角标点各占 1 个字节；一个汉字占 3 个字节。你的文本有 ${fmtInt(cs.chars)} 个字符、${fmtInt(cs.bytes)} 个字节。BPE（字节对编码）就从这里出发：每个字节是一个词元，也就是模型读入的一个片段；所以最初的词表（分词器认识的全部词元）只有 256 个字节值，ID 是 0 到 255。`
				};
			},
			look: () =>
				cs.cjk > 0
					? { en: 'Characters on a grey card take more than one byte; each Chinese character has three. Hover over or tap a character to light up its bytes.', zh: '灰色底的字符占不止一个字节，每个汉字占三个。把鼠标移到字符上（或点按它），它的字节会亮起来。' }
					: { en: 'Every character here is one byte. Open “Your text” at the top of the stage, pick “Same story in Chinese” and watch each character become three bytes.', zh: '这里每个字符都是 1 个字节。打开演示区顶部的“你的文本”，选“同一个故事（中文）”，看看每个字符怎样变成 3 个字节。' },
			// longer narration than the kit's reading-time default allows (16 s)
			duration: 22000
		},
		{
			id: 'pretokens',
			title: { en: 'Split into pre-tokens', zh: '切成预分词片段' },
			narration: () => {
				const n = ps?.list.length ?? 0;
				const d = ps?.types.length ?? 0;
				const top = ps?.types.find((x) => x.bytes > 1) ?? ps?.types[0];
				const shown = top ? visibleToken(top.text) : '';
				const sp = shown.includes('␣');
				return {
					en: `Before counting anything, a fixed pattern rule (a “regular expression”) cuts the text into pre-tokens: words with their leading space, numbers, punctuation, and runs of spaces. The next steps glue bytes together only inside one pre-token, never across two, so a token never spans two words. Your text has ${fmtInt(n)} pre-tokens, ${fmtInt(d)} of them different${top && top.n > 1 ? `; ${q(shown)}${sp ? ' (␣ is a space)' : ''} alone occurs ${top.n} times` : ''}.`,
					zh: `在统计之前，先用一条固定的匹配规则（“正则表达式”）把文本切成预分词片段：单词连同它前面的空格、数字、标点，以及连续的空格。后面的步骤只在一个片段内部把字节拼起来，绝不跨越两个片段，所以一个词元不会横跨两个词。你的文本有 ${fmtInt(n)} 个预分词片段，其中 ${fmtInt(d)} 种不同${top && top.n > 1 ? `；光是${q(shown)}${sp ? '（␣ 表示空格）' : ''}就出现了 ${top.n} 次` : ''}。`
				};
			},
			look: {
				en: 'Spaces are drawn as ␣, so “␣the” (with its space) and “The” are different pre-tokens. Pick a row in the frequency list to find it in the text.',
				zh: '空格画成 ␣，所以“␣the”（带空格）和“The”是不同的片段。在频率表里选一行，就能在文本中找到它。'
			},
			duration: 18000
		},
		{
			id: 'pairs',
			title: { en: 'Count every neighbouring pair', zh: '统计每一对相邻的词元' },
			narration: () => {
				if (!trace) return LOADING;
				const top = trace.steps[0]?.top[0];
				if (!top)
					return {
						en: 'Now count every two neighbouring tokens inside each pre-token. Your text has no such pair: type a longer text.',
						zh: '现在统计每个预分词片段内部相邻的两个词元。你的文本里没有这样的对：请输入更长的文本。'
					};
				const pair = `${q(plainToken(disp(top.a)))} + ${q(plainToken(disp(top.b)))}`;
				const s0 = pairSources(segment(trained, pattern, trace.merges, 0), top.a, top.b)[0];
				const big = s0 && s0.f > 1;
				return {
					en: `Now look at every two neighbouring tokens inside each pre-token, and count each pair over the whole text. A pre-token counts as often as it occurs. In your text the most frequent pair is ${pair}: ${top.n} times, out of ${fmtInt(trace.steps[0].pairs)} different pairs.${big ? ` ${s0.per * s0.f} of them come from ${q(visibleToken(s0.text))}, which occurs ${s0.f} times.` : ''} This winner is what BPE joins into one new token in the next step.`,
					zh: `现在看每个预分词片段内部相邻的两个词元，并在整段文本里统计每一对出现了几次。片段出现几次，就计几次。在你的文本里，出现最多的是${pair}：${top.n} 次（共有 ${fmtInt(trace.steps[0].pairs)} 种不同的对）。${big ? `其中 ${s0.per * s0.f} 次来自出现了 ${s0.f} 次的${q(visibleToken(s0.text))}。` : ''}下一步，BPE 就把这个胜出的对拼成一个新词元。`
				};
			},
			duration: 20000,
			look: {
				en: 'The breakdown adds the count up pre-token by pre-token. Pick another pair in the list to see where its count comes from.',
				zh: '分解表一个片段一个片段地把计数加起来。在列表里选另一对，看看它的计数从哪里来。'
			},
			onEnter: () => (pairSel = 0)
		},
		{
			id: 'merge',
			title: { en: 'Merge the most frequent pair', zh: '合并出现最多的一对' },
			narration: () => {
				if (!trace) return LOADING;
				if (!K) return { en: 'Your text has no pair to merge. Type a longer text.', zh: '你的文本里没有可合并的对。请输入更长的文本。' };
				const m = Math.max(1, Math.min(mergeN, K));
				const st = trace.steps[m - 1];
				const before = tokensAt(trace, m - 1);
				const saved = before - st.tokenCount;
				const A = q(plainToken(disp(st.pair[0])));
				const B = q(plainToken(disp(st.pair[1])));
				const AB = q(plainToken(disp(st.newId)));
				const part = disp(st.newId).includes('<0x');
				// overlapping pairs exist only for a pair of two equal tokens (x, x): "xxx" holds 2 pairs but merges once
				const overlap = saved < st.count;
				const a0 = plainToken(disp(st.pair[0]));
				const sep = /^[0-9A-F]{2}( [0-9A-F]{2})*$/.test(a0) ? ' ' : '';
				const x3 = q([a0, a0, a0].join(sep));
				const x2 = q([a0, a0].join(sep));
				return {
					en: `The winning pair becomes a new token: ${A} + ${B} → ${AB}, with ID ${st.newId}${m === 1 ? ', the first ID after the 256 bytes (0 to 255)' : ''}. Every place where the pair occurs is rewritten as this one token, so the text gets ${fmtInt(saved)} tokens shorter: ${fmtInt(before)} − ${fmtInt(saved)} = ${fmtInt(st.tokenCount)}.${overlap ? ` It was counted ${st.count} times, but overlapping pairs merge only once (${x3} holds two pairs yet gives one ${x2}), so only ${fmtInt(saved)} places change.` : ''} The vocabulary grows by one, to ${256 + m}.${part ? ' This new token is not even a whole character: BPE sees bytes, not characters.' : ''}`,
					zh: `胜出的这一对成为一个新词元：${A} + ${B} → ${AB}，ID 是 ${st.newId}${m === 1 ? '，紧接在 256 个字节（0 到 255）之后' : ''}。文本中这一对出现的每个位置都改写成这一个词元，所以文本少了 ${fmtInt(saved)} 个词元：${fmtInt(before)} − ${fmtInt(saved)} = ${fmtInt(st.tokenCount)}。${overlap ? `它被计了 ${st.count} 次，但重叠的对只能合并一次（${x3}里有两对，却只能合出一个${x2}），所以只有 ${fmtInt(saved)} 处改变。` : ''}词表增加一个，变成 ${256 + m} 个。${part ? '这个新词元甚至不是一个完整的字符：BPE 只看字节，不看字符。' : ''}`
				};
			},
			look: () =>
				motion.reduced
					? {
							en: 'Press “Next step” under the text: first the purple pairs join up, then each pair becomes one token. “Next merge” does it again with the new counts.',
							zh: '点文本下方的“下一步”：先是紫色的成对词元靠拢，然后每一对变成一个词元。“下一次合并”会用新的计数再做一遍。'
						}
					: {
							en: 'Watch the purple pairs close up and turn into one token. “Next merge” does it again with the new counts.',
							zh: '看紫色的成对词元靠拢，变成一个词元。“下一次合并”会用新的计数再做一遍。'
						},
			waitForDone: true,
			// the animation takes about 5 s; leave time to read the narration under Play
			linger: 9000
		},
		{
			id: 'repeat',
			title: { en: 'Repeat: the vocabulary grows, the text shrinks', zh: '重复：词表变大，文本变短' },
			narration: () => {
				if (!trace) return LOADING;
				if (!K) return { en: 'Your text has no pair to merge. Type a longer text.', zh: '你的文本里没有可合并的对。请输入更长的文本。' };
				const s = firstSingleton(trace);
				const fin = tokensAt(trace, K);
				// training stops at 767 merges (BpeToy's cap, = TinyGPT's); a long text can still have pairs left then
				const capped = trace.finalPairs > 0;
				return {
					en: `Training is just this loop: count the pairs again, merge the new winner, repeat. Each merge adds one token to the vocabulary and makes the text shorter. ${capped ? `Here training stops after ${K} merges (the same number TinyGPT has), although pairs are still left: your text is down to ${fmtInt(fin)} tokens from ${fmtInt(trace.initialTokenCount)}.` : `Your text allows ${K} merges; after that every pre-token is a single token (${fmtInt(fin)} tokens instead of ${fmtInt(trace.initialTokenCount)}).`}${s ? ` From merge #${s} on, the winning pair occurs only once: the tokenizer is memorizing words of this text, not finding patterns.` : ''} Real tokenizers stop at a vocabulary size chosen in advance: TinyGPT, the small story-writing model on this site, stops after 767 merges.`,
					zh: `训练就是这个循环：重新统计相邻对，合并新的胜出者，再重复。每次合并给词表加一个词元，并让文本变短。${capped ? `这里合并 ${K} 次后就停止训练（和 TinyGPT 的合并次数一样），尽管还剩有相邻对：你的文本从 ${fmtInt(trace.initialTokenCount)} 个词元减到了 ${fmtInt(fin)} 个。` : `你的文本最多能合并 ${K} 次，之后每个预分词片段都成了一个词元（${fmtInt(fin)} 个，而不是 ${fmtInt(trace.initialTokenCount)} 个）。`}${s ? `从第 ${s} 次合并开始，胜出的对只出现 1 次：分词器是在背这段文本里的词，而不是在发现规律。` : ''}真实的分词器会在事先选好的词表大小处停下：本站写小故事的小模型 TinyGPT 合并 767 次后就停了。`
				};
			},
			look: {
				en: 'Solid line: your tokenizer on your text. Dashed line: TinyGPT’s first merges (learned from children’s stories) on the same text. Tap the chart to jump to any vocabulary size.',
				zh: '实线：你的分词器处理你的文本。虚线：在同一段文本上使用 TinyGPT 的前若干次合并（从儿童故事中学来）。点图表可以跳到任意词表大小。'
			},
			waitForDone: true,
			// the merge player runs about 9 s; leave time to read the narration under Play
			linger: 8000,
			onEnter: () => (lk = Math.max(0, Math.min(mergeN, K)))
		},
		{
			id: 'apply',
			title: { en: 'Use the merges on new text', zh: '用学到的合并切分新文本' },
			narration: () => {
				if (!trace || !tiny) return LOADING;
				const useTiny = src === 'tiny' || !yours;
				const es = (useTiny ? tiny : yours!).encodeWithSteps(sentence);
				const n = es.ids.length;
				const nb = new TextEncoder().encode(sentence).length;
				const longer = es.pretokens.filter((p) => !p.special && p.bytes.length > 1);
				const whole = longer.filter((p) => p.ids.length === 1).length;
				const which = useTiny ? { en: 'TinyGPT’s 767 merges', zh: ' TinyGPT 的 767 条合并规则' } : { en: `your ${K} merges`, zh: `你的 ${K} 条合并规则` };
				return {
					en: `A finished tokenizer is just its list of merges, in the order they were learned. To cut new text: split it into pre-tokens, start from the bytes, and inside each pre-token keep applying the earliest-learned merge that fits, until none fits. With ${which.en}, the ${fmtInt(nb)} bytes of your sentence become ${fmtInt(n)} tokens${longer.length ? (whole ? `, and ${whole} of its ${longer.length} pre-tokens longer than one byte come out whole, as one token` : `, and none of its ${longer.length} pre-tokens longer than one byte comes out whole`) : ''}. Words that were common in the training text end up whole; new words fall apart into pieces; characters never seen stay as byte pieces.`,
					zh: `训练好的分词器就是一张按学习顺序排列的合并列表。切分新文本时：先切成预分词片段，从字节开始，在每个片段内部不断应用能用上的、最早学到的那条合并，直到一条也用不上。用${which.zh}，你的句子的 ${fmtInt(nb)} 个字节变成 ${fmtInt(n)} 个词元${longer.length ? (whole ? `；它的 ${longer.length} 个多字节片段里，有 ${whole} 个合成了单个词元` : `；它的 ${longer.length} 个多字节片段，没有一个合成单个词元`) : ''}。训练文本里常见的词会变成完整的词元；新词会碎成几块；从没见过的字符只剩字节碎片。`
				};
			},
			look: {
				en: 'Each pair shows its merge number (— = never learned); the outlined one has the lowest number and is merged next. Switch to TinyGPT’s merges, or open “Your sentence” at the top of the stage and try one made of words from your training text.',
				zh: '每一对都标着合并编号（— 表示从没学过）；带框的编号最小，下一步就合并它。可以切换到 TinyGPT 的合并规则，或打开演示区顶部的“你的句子”，试一个由训练文本中的词组成的句子。'
			},
			waitForDone: true,
			// the player inside one pre-token takes a few seconds; leave time to read the narration under Play
			linger: 12000,
			duration: 25000
		},
		{
			id: 'real',
			title: { en: 'Real tokenizers: the same recipe, bigger', zh: '真实分词器：同样的方法，更大的规模' },
			narration: () => {
				const g = counts.gpt2;
				const qw = counts.qwen;
				const have = g !== undefined && qw !== undefined;
				return {
					en: `GPT-2, GPT-4 and Qwen2.5 were built the same way, only much bigger: GPT-2 learned 50,000 merges from English web text, GPT-4’s cl100k_base has about 100,000 tokens, and Qwen2.5 about 151,000, many of them Chinese words. ${have ? `Your sentence costs ${g} tokens in GPT-2 and ${qw} in Qwen2.5.` : 'Their files are loading now.'} BERT uses WordPiece, a close cousin: it cuts each word into the longest pieces it knows and has no bytes to fall back on, so a word it cannot spell becomes [UNK].`,
					zh: `GPT-2、GPT-4 和 Qwen2.5 也是这样做出来的，只是规模大得多：GPT-2 从英文网页文本中学到 50,000 次合并，GPT-4 的 cl100k_base 有约 100,000 个词元，Qwen2.5 有约 151,000 个，其中很多是中文词。${have ? `你的句子在 GPT-2 里要 ${g} 个词元，在 Qwen2.5 里要 ${qw} 个。` : '它们的文件正在加载。'}BERT 用的是 WordPiece，BPE 的近亲：它把每个词切成它认识的最长片段，没有字节可以退回，所以拼不出来的词会变成 [UNK]。`
				};
			},
			look: {
				en: 'Compare the bars: a tokenizer that learned many merges from text like yours needs fewer tokens. Hover or tap a token to see its ID and bytes; [UNK] is outlined in red.',
				zh: '比较条形：从类似你这种文本中学到大量合并的分词器，需要的词元更少。悬停或点按词元，查看它的 ID 和字节；[UNK] 用红框标出。'
			},
			onEnter: () => loadAll()
		}
	];

	/** the stage header's one provenance label: every step is live (step 7: the real tokenizer files, parity-checked);
	 *  step 6's TinyGPT pair counts are the exception and carry their own small label */
	const LIVE: StageProv = {
		mode: 'live',
		note: { en: 'Computed in your browser from your text right now (training runs in a Web Worker).', zh: '此刻在你的浏览器里由你的文本算出（训练在 Web Worker 中运行）。' },
		short: { en: '· computed on your text', zh: '· 用你的文本现场计算' }
	};
	const REAL_PROV: StageProv = { mode: 'live', note: PARITY.note, short: { en: '· real tokenizer files, on your sentence', zh: '· 真实分词器文件，处理你的句子' } };
	for (const s of steps) s.prov = s.id === 'real' ? REAL_PROV : LIVE;

	/** the end card's recap: three things the learner saw, with their numbers (spec §5.8) */
	const first = $derived(trace?.steps[0]);

	const chapters = [
		{ title: { en: 'Bytes', zh: '字节' }, count: 2 },
		{ title: { en: 'Training', zh: '训练' }, count: 3 },
		{ title: { en: 'Using it', zh: '投入使用' }, count: 2 }
	];
</script>

<Walkthrough {steps} bind:index {ready} {chapters} label={{ en: 'How text becomes the numbers a model reads', zh: '文本怎样变成模型读到的数' }}>
	{#snippet cover()}<CoverStage {P} />{/snippet}
	{#snippet stage(step, ctx)}
		{#if !tiny || (!trace && step.id !== 'bytes' && step.id !== 'pretokens')}
			<p class="muted">{P.loadError || P.trainError || t(LOADING)}</p>
		{:else if step.id === 'bytes'}
			<BytesStage text={trainText} />
		{:else if step.id === 'pretokens'}
			<PretokStage text={trainText} {pattern} />
		{:else if trace && step.id === 'pairs'}
			<PairsStage {trace} text={trained} {pattern} bind:sel={pairSel} />
		{:else if trace && step.id === 'merge'}
			<MergeStage {trace} text={trained} {pattern} bind:m={mergeN} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
		{:else if trace && step.id === 'repeat'}
			<RepeatStage {trace} text={trained} {pattern} bind:k={lk} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
		{:else if step.id === 'apply'}
			<ApplyStage {yours} {tiny} bind:src {sentence} {yourCounts} tinyCounts={mergeCounts} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />
		{:else if step.id === 'real'}
			<RealStage {sentence} {tiny} {yours} bind:counts />
		{/if}
	{/snippet}

	{#snippet recap()}
		<ul class="recap">
			<li>{t({ en: `Your text: ${fmtInt(cs.chars)} characters, ${fmtInt(cs.bytes)} UTF-8 bytes, one token per byte before any merge.`, zh: `你的文本：${fmtInt(cs.chars)} 个字符、${fmtInt(cs.bytes)} 个 UTF-8 字节，合并之前每个字节是一个词元。` })}</li>
			{#if trace && first}
				<li>{t({ en: `${K} merges, the first ${q(plainToken(disp(first.pair[0])))} + ${q(plainToken(disp(first.pair[1])))} (${first.count} times), took it from ${fmtInt(trace.initialTokenCount)} to ${fmtInt(tokensAt(trace, K))} tokens.`, zh: `${K} 次合并（第一次是${q(plainToken(disp(first.pair[0])))} + ${q(plainToken(disp(first.pair[1])))}，出现 ${first.count} 次）把它从 ${fmtInt(trace.initialTokenCount)} 个词元缩到 ${fmtInt(tokensAt(trace, K))} 个。` })}</li>
			{/if}
			{#if counts.gpt2 !== undefined && counts.qwen !== undefined}
				<li>{t({ en: `Your sentence costs ${counts.gpt2} tokens in GPT-2 and ${counts.qwen} in Qwen2.5.`, zh: `你的句子在 GPT-2 里要 ${counts.gpt2} 个词元，在 Qwen2.5 里要 ${counts.qwen} 个。` })}</li>
			{/if}
		</ul>
	{/snippet}

	{#snippet end()}
		<p class="fin">
			<strong>{t({ en: 'That is all of BPE:', zh: '这就是 BPE 的全部：' })}</strong>
			{t({
				en: 'bytes, pre-tokens, count, merge, repeat; then replay the merges in order on new text. Try another training text and run the lesson again, or open Explore to see every part at full size.',
				zh: '字节、预分词、统计、合并、重复；然后在新文本上按顺序重放这些合并。换一段训练文本再走一遍课程，或者打开“探索”，完整地看每个部分。'
			})}
		</p>
	{/snippet}
</Walkthrough>

<style>
	.recap {
		margin: 0;
		padding-left: 1.2em;
		display: grid;
		gap: 4px;
	}
	.fin {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
