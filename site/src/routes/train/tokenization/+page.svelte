<!--
  /train/tokenization/ — Tokenization concept page (Training Concepts, pre-training).
  Hook (TinyGPT ids of your sentence) · Toy (Part 1: train BPE on your text) · Real (Part 2: TinyGPT's
  tokenizer step by step; Part 3: four real tokenizers side by side) · Compare (ten languages) ·
  Pitfalls (two misconception checks) · Checkpoint · Links.
  Page components and helpers live in $lib/pages/tokenization/; tokenizer files in
  static/pages/tokenization/tok/ (made by pipeline/tokenizers/export.py, parity: parity.ts).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ConceptPageShell from '$lib/kit/ConceptPageShell.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { MODEL_BY_SLUG } from '$lib/catalog/models';
	import { CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
	import type { PredictQuestion } from '$lib/kit/types';
	import { BPETokenizer, type TokenizerJSON } from '$lib/models/bpe';

	import BpeToy from '$lib/pages/tokenization/BpeToy.svelte';
	import TinyGptPart from '$lib/pages/tokenization/TinyGptPart.svelte';
	import RealCompare from '$lib/pages/tokenization/RealCompare.svelte';
	import LangTable from '$lib/pages/tokenization/LangTable.svelte';
	import Pitfalls from '$lib/pages/tokenization/Pitfalls.svelte';
	import Lesson from '$lib/pages/tokenization/Lesson.svelte';
	import Quiz from '$lib/pages/tokenization/Quiz.svelte';
	import TryThis from '$lib/pages/tokenization/TryThis.svelte';
	import Chips from '$lib/pages/tokenization/Chips.svelte';
	import Hook from '$lib/pages/tokenization/content/Hook.svelte';
	import HookQuestions from '$lib/pages/tokenization/content/HookQuestions.svelte';
	import ToyIntro from '$lib/pages/tokenization/content/ToyIntro.svelte';
	import ToyAfter from '$lib/pages/tokenization/content/ToyAfter.svelte';
	import TinyIntro from '$lib/pages/tokenization/content/TinyIntro.svelte';
	import RealIntro from '$lib/pages/tokenization/content/RealIntro.svelte';
	import Compare from '$lib/pages/tokenization/content/Compare.svelte';
	import { PRESETS, DEFAULT_SENTENCE } from '$lib/pages/tokenization/presets';
	import { startClient, stopClient } from '$lib/pages/tokenization/state.svelte';
	import type { TrainTrace } from '$lib/pages/tokenization/train';
	import type { ChipItem, LangSummary, TryItem } from '$lib/pages/tokenization/types';

	const MAX_SENTENCE = 400;
	let sentence = $state(DEFAULT_SENTENCE);
	let trainText = $state(PRESETS.train[0].text);
	let k = $state(0);
	let trace = $state.raw<TrainTrace | null>(null);
	/** the text `trace` was trained on (set by BpeToy together with the trace) */
	let trained = $state('');
	let tiny = $state.raw<BPETokenizer | null>(null);
	let mergeCounts = $state.raw<number[]>([]);
	let loadError = $state('');
	let summary = $state<LangSummary[]>([]);

	onMount(() => {
		try {
			const q = new URLSearchParams(location.search).get('text');
			if (q) sentence = Array.from(q).slice(0, MAX_SENTENCE).join('');
		} catch {
			/* ignore */
		}
		const h = takeHandoff('text');
		if (h) sentence = Array.from(h.text).slice(0, MAX_SENTENCE).join('');
		startClient();
		(async () => {
			try {
				const res = await fetch(`${base}/models/tinygpt/tokenizer.json`);
				if (!res.ok) throw new Error(`tokenizer.json: HTTP ${res.status}`);
				const json = (await res.json()) as TokenizerJSON;
				tiny = new BPETokenizer(json);
				mergeCounts = json.mergeCounts ?? [];
			} catch (e) {
				loadError = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => stopClient();
	});

	// hook: TinyGPT ids of the sentence
	const hookIds = $derived(tiny ? tiny.encode(sentence) : []);
	const hookItems = $derived<ChipItem[]>(
		tiny
			? hookIds.map((id) => ({
					text: tiny!.tokenToDisplay(id),
					id,
					kind: tiny!.isSpecial(id) ? 'special' : tiny!.tokenToDisplay(id).includes('<0x') ? 'partial' : undefined,
					hex: Array.from(tiny!.tokenBytes(id), (b) => b.toString(16).padStart(2, '0')).join(' ')
				}))
			: []
	);
	const nChars = $derived(Array.from(sentence).length);
	const nBytes = $derived(new TextEncoder().encode(sentence).length);

	function scrollToId(id: string) {
		requestAnimationFrame(() => {
			const el = document.getElementById(id);
			if (!el) return;
			el.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'center' });
			el.classList.remove('flash');
			void el.offsetWidth;
			el.classList.add('flash');
		});
	}
	function setTrain(id: string, merges: number | 'end' = 0) {
		const p = PRESETS.train.find((x) => x.id === id);
		if (p) trainText = p.text;
		k = merges === 'end' ? 767 : merges;
		scrollToId('toy-player');
	}
	function setSentence(text: string, target = 'real-input') {
		sentence = text;
		scrollToId(target);
	}

	const toyTries: TryItem[] = [
		{
			q: { en: 'On the English story, which pair is merged first: “th”, “he” or “ t” (space + t)?', zh: '在英文故事上，第一次合并的是哪一对：“th”、“he”，还是“ t”（空格 + t）？' },
			go: { en: 'Start at 0 merges, then step', zh: '从 0 次合并开始，再逐步前进' },
			run: () => setTrain('story-en', 0)
		},
		{
			q: { en: 'Train on the same story in Chinese. Will the first merge make a whole character?', zh: '换成同一个故事的中文版来训练。第一次合并会拼出一个完整的字吗？' },
			go: { en: 'Train on the Chinese story', zh: '用中文故事训练' },
			run: () => setTrain('story-zh', 1)
		},
		{
			q: { en: 'The classic example from the BPE paper: low ×5, lower ×2, newest ×6, widest ×3. Which pair wins first?', zh: 'BPE 论文里的经典例子：low ×5、lower ×2、newest ×6、widest ×3。哪一对最先胜出？' },
			go: { en: 'Load the classic example', zh: '载入经典例子' },
			run: () => setTrain('classic', 1)
		},
		{
			q: { en: 'Keep merging until nothing is left. Where do the pair counts drop to 1, and what is the tokenizer doing then?', zh: '一直合并到没有可合并的为止。相邻对的计数从哪里开始降到 1？这时分词器在做什么？' },
			go: { en: 'Jump to the last merge', zh: '跳到最后一次合并' },
			run: () => {
				k = 767; // BpeToy clamps this to the last merge it learned; the learner's own text is kept
				scrollToId('toy-player');
			}
		}
	];
	const realTries: TryItem[] = [
		{
			q: { en: 'Are “the”, “ the” and “The” the same token?', zh: '“the”“ the”和“The”是同一个词元吗？' },
			go: { en: 'Try it', zh: '试一试' },
			run: () => setSentence('the The THE  the', 'real-rows')
		},
		{
			q: { en: 'How do GPT-2, GPT-4 and Qwen2.5 cut 12345678? Predict the number of tokens for each.', zh: 'GPT-2、GPT-4 和 Qwen2.5 怎么切 12345678？先预测各自的词元数。' },
			go: { en: 'Try numbers', zh: '试试数字' },
			run: () => setSentence('In 2024, 12345678 + 3.14159 = ?', 'real-rows')
		},
		{
			q: { en: 'Most emoji are 4 bytes, and a family emoji is several emoji joined together. Which tokenizer gives up and says [UNK]?', zh: '大多数表情符号占 4 个字节，“一家人”表情则是好几个表情连在一起。哪个分词器会放弃并输出 [UNK]？' },
			go: { en: 'Try emoji', zh: '试试表情符号' },
			run: () => setSentence('I ❤️ 🦙🦙 llamas! 👩‍👩‍👧', 'real-rows')
		}
	];

	const prequestion: PredictQuestion = {
		id: 'tokenization-pre',
		question: {
			en: 'In GPT-2’s tokenizer, which costs more tokens: “The weather is nice today” (25 characters) or “今天天气很好” (6 characters, the same meaning)?',
			zh: '在 GPT-2 的分词器里，哪个需要更多词元：“The weather is nice today”（25 个字符），还是“今天天气很好”（6 个字符，意思相同）？'
		},
		options: [
			{ en: 'The English one: it has four times as many characters', zh: '英文那句：它的字符是中文的四倍' },
			{ en: 'About the same: they mean the same thing', zh: '差不多：它们意思一样' },
			{ en: 'The Chinese one', zh: '中文那句' }
		],
		answer: 2,
		explain: {
			en: 'GPT-2 needs 5 tokens for the English and 10 for the Chinese. Each Chinese character is 3 UTF-8 bytes, and GPT-2 learned few merges for Chinese, so most characters become 2 byte pieces. Fewer characters can still mean more tokens.',
			zh: 'GPT-2 处理英文需要 5 个词元，处理中文需要 10 个。每个汉字占 3 个 UTF-8 字节，而 GPT-2 学到的中文合并很少，所以大多数汉字被切成 2 个字节碎片。字符少，词元不一定少。'
		}
	};

	const goals = [
		{ en: 'Train a byte-pair encoding (BPE) tokenizer on your own text and explain each merge: count neighbouring pairs, merge the most frequent one, repeat.', zh: '在你自己的文本上训练一个字节对编码（BPE）分词器，并解释每一次合并：统计相邻对，合并出现最多的一对，重复。' },
		{ en: 'Follow a real tokenizer at work: pre-tokens, merges by rank, token ids and the UTF-8 bytes underneath.', zh: '跟着一个真实分词器走一遍：预分词片段、按排名合并、词元 ID，以及底层的 UTF-8 字节。' },
		{ en: 'Explain why the same meaning costs different numbers of tokens in different languages and tokenizers, and what [UNK] and byte pieces are.', zh: '解释为什么同样的意思在不同语言、不同分词器下需要的词元数不同，以及 [UNK] 和字节碎片是什么。' }
	];

	const nextConcept = CONCEPT_BY_SLUG['next-token-prediction'];
	const tinyModel = MODEL_BY_SLUG['tinygpt'];
	const otherModels = ['tokenizers', 'bert-base', 'tinybert', 'whisper-tiny', 'clip'].map((s) => MODEL_BY_SLUG[s]).filter(Boolean);
</script>

<ConceptPageShell
	slug="tokenization"
	{goals}
	{prequestion}
	lede={{
		en: 'A language model never reads letters or words: it reads a list of token ids. Train your own byte-pair encoder, follow TinyGPT’s tokenizer byte by byte, and compare four real tokenizers on your own text.',
		zh: '语言模型从不读字母或词，它读的是一串词元 ID。训练你自己的字节对编码器，一个字节一个字节地跟踪 TinyGPT 的分词器，再用你自己的文本对比四个真实分词器。'
	}}
	titles={{
		hook: { en: 'What does a model read?', zh: '模型读到的是什么？' },
		toy: { en: 'Part 1 · Build a tokenizer from your own text', zh: '第 1 部分 · 用你自己的文本训练分词器' },
		real: { en: 'Parts 2 and 3 · Real tokenizers at work', zh: '第 2、3 部分 · 真实分词器' },
		compare: { en: 'Same meaning, different price', zh: '同样的意思，不同的价钱' },
		post: { en: 'Checkpoint', zh: '检查点' }
	}}
	badges={{ hook: 'live', lesson: 'live', toy: 'live', real: 'live', compare: 'live' }}
	lessonNote={{ en: '7 short steps: byte-pair encoding on your own text, with the real numbers', zh: '7 个小步骤：在你自己的文本上走一遍字节对编码，每一步都是真实的数' }}
	exploreIntro={{
		en: 'Everything from the lesson at full size: train merge by merge with every count visible, follow TinyGPT’s tokenizer byte by byte, and compare six tokenizers and ten languages on your own text.',
		zh: '课程里的一切，完整展开：逐次合并训练并看到每个计数，一个字节一个字节地跟踪 TinyGPT 的分词器，再用你自己的文本对比六个分词器和十种语言。'
	}}
	intros={{
		lesson: {
			en: 'Follow byte-pair encoding (BPE) one step at a time, on your own text. Press Next (or the → key), or Play to let the lesson run; every view uses your text and real tokenizers.',
			zh: '在你自己的文本上，一步一步地走一遍字节对编码（BPE）。点“下一步”（或按 → 键），或点“自动播放”让课程自己走；每个视图都用你的文本和真实的分词器。'
		}
	}}
>
	{#snippet hook()}
		<div class="sec">
			<Hook />
			<div class="panel inbox">
				<TextInputBar
					bind:value={sentence}
					maxLength={MAX_SENTENCE}
					label={{ en: 'Your sentence (used everywhere on this page)', zh: '你的句子（本页各处都会用到）' }}
					presets={PRESETS.sentence}
				/>
				{#if tiny}
					<p class="flow" aria-live="polite">
						<span><b class="num">{nChars}</b> {t({ en: 'characters', zh: '个字符' })}</span>
						<span class="arrow">→</span>
						<span><b class="num">{nBytes}</b> {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
						<span class="arrow">→</span>
						<span><b class="num big">{hookIds.length}</b> {t({ en: 'TinyGPT tokens', zh: '个 TinyGPT 词元' })}</span>
					</p>
					<Chips items={hookItems} showIds limit={600} label={{ en: 'TinyGPT tokens of your sentence', zh: '你的句子的 TinyGPT 词元' }} />
					<p class="ids num">[{hookIds.join(', ')}]</p>
				{:else if loadError}
					<p class="err">{loadError}</p>
				{:else}
					<p class="muted">{t({ en: 'Loading TinyGPT’s tokenizer (27 KB)…', zh: '正在加载 TinyGPT 的分词器（27 KB）…' })}</p>
				{/if}
			</div>
			<HookQuestions />
		</div>
	{/snippet}

	{#snippet lesson()}
		<Lesson bind:trainText bind:sentence {trace} {trained} {tiny} {mergeCounts} maxSentence={MAX_SENTENCE} />
	{/snippet}

	{#snippet toy()}
		<div class="sec">
			<ToyIntro />
			<TryThis items={toyTries} />
			<div id="toy-player" class="target">
				<BpeToy bind:text={trainText} bind:k bind:trace bind:trained {tiny} />
			</div>
			<ToyAfter />
		</div>
	{/snippet}

	{#snippet real()}
		<div class="sec">
			<h3 class="part">{t({ en: 'Part 2 · TinyGPT’s tokenizer, step by step', zh: '第 2 部分 · 一步一步看 TinyGPT 的分词器' })}</h3>
			<TinyIntro />
			<div id="real-input" class="panel inbox target">
				<TextInputBar bind:value={sentence} maxLength={MAX_SENTENCE} label={{ en: 'Your sentence', zh: '你的句子' }} presets={PRESETS.sentence} />
			</div>
			{#if tiny}
				<TinyGptPart tok={tiny} {sentence} counts={mergeCounts} />
			{/if}

			<h3 class="part">{t({ en: 'Part 3 · Real tokenizers side by side', zh: '第 3 部分 · 真实分词器并排对比' })}</h3>
			<RealIntro />
			<TryThis items={realTries} />
			<div id="real-rows" class="target">
				{#if tiny}
					<RealCompare {sentence} {tiny} {trace} {k} />
				{/if}
			</div>
		</div>
	{/snippet}

	{#snippet compare()}
		<div class="sec">
			<Compare part="intro" />
			{#if tiny}
				<LangTable {tiny} {trace} {k} bind:summary onuse={(s) => setSentence(s, 'real-rows')} />
			{/if}
			<Compare part="after" />
		</div>
	{/snippet}

	{#snippet pitfalls()}
		{#if tiny}<Pitfalls {tiny} {summary} />{/if}
	{/snippet}

	{#snippet post()}
		<Quiz
			question={{ en: 'With no merges at all (the vocabulary is just the 256 bytes), how many tokens is 我爱你?', zh: '如果一次合并都没有（词表只有 256 个字节），“我爱你”是多少个词元？' }}
			options={[{ en: '3', zh: '3' }, { en: '6', zh: '6' }, { en: '9', zh: '9' }]}
			answer={2}
			explain={{ en: 'Each Chinese character is 3 UTF-8 bytes, and with no merges every byte is its own token: 3 × 3 = 9. Check it in Part 1 (drag the vocabulary to 256) or in TinyGPT’s byte view in Part 2.', zh: '每个汉字占 3 个 UTF-8 字节；没有合并时，每个字节各是一个词元：3 × 3 = 9。可以在第 1 部分（把词表拖到 256）或第 2 部分 TinyGPT 的字节视图里验证。' }}
		/>
		<Quiz
			question={{ en: 'Why does BERT turn 爱 into [UNK], while GPT-2 never produces [UNK]?', zh: '为什么 BERT 会把“爱”变成 [UNK]，而 GPT-2 从不产生 [UNK]？' }}
			options={[
				{ en: 'BERT’s vocabulary has no piece for 爱 and nothing smaller to fall back on; GPT-2 can always fall back to single bytes.', zh: 'BERT 的词表里没有“爱”这个片段，也没有更小的单位可退；GPT-2 总能退回到单个字节。' },
				{ en: 'GPT-2 was trained on more Chinese text than BERT.', zh: 'GPT-2 训练时用的中文比 BERT 多。' },
				{ en: 'BERT lowercases the text, which removes Chinese characters.', zh: 'BERT 会把文本转成小写，这会删掉汉字。' }
			]}
			answer={0}
			explain={{ en: 'WordPiece can only use pieces that are in its vocabulary; when a word cannot be spelled, the whole word becomes [UNK]. Byte-level BPE starts from all 256 bytes, so any text can be written, at worst one byte per token (GPT-2 writes 爱 as 3 byte pieces, one per byte; GPT-4’s cl100k needs 2).', zh: 'WordPiece 只能用词表里有的片段；一个词拼不出来时，整个词就变成 [UNK]。字节级 BPE 从全部 256 个字节出发，所以任何文本都写得出来，最坏也就是一个字节一个词元（GPT-2 把“爱”写成 3 个字节碎片，每个字节一个；GPT-4 的 cl100k 需要 2 个）。' }}
		/>
	{/snippet}

	{#snippet links()}
		<h3 class="lh">{t({ en: 'Next', zh: '下一步' })}</h3>
		<div class="cards">
			{#if nextConcept}<ConceptCard concept={nextConcept} compact />{/if}
			{#if tinyModel}<ModelCard model={tinyModel} compact />{/if}
		</div>
		<p class="muted small">
			<Lang>
				{#snippet en()}The ids you made here are exactly what <a href="{base}/m/tinygpt/">TinyGPT</a> reads; <a href="{base}/train/next-token-prediction/">next-token prediction</a> is the game it plays with them.{/snippet}
				{#snippet zh()}你在这里得到的 ID 正是 <a href="{base}/m/tinygpt/">TinyGPT</a> 读入的内容；<a href="{base}/train/next-token-prediction/">下一个词元预测</a>就是它拿这些 ID 玩的游戏。{/snippet}
			</Lang>
		</p>
		{#if otherModels.length}
			<h3 class="lh">{t({ en: 'Models that depend on a tokenizer', zh: '依赖分词器的模型' })}</h3>
			<div class="cards">{#each otherModels as m (m.slug)}<ModelCard model={m} compact />{/each}</div>
		{/if}
	{/snippet}

	{#snippet further()}
		<h3>{t({ en: 'Further reading', zh: '延伸阅读' })}</h3>
		<ul>
			<li><a href="https://arxiv.org/abs/1508.07909" rel="external noopener">Sennrich, Haddow & Birch (2016), Neural Machine Translation of Rare Words with Subword Units</a> — {t({ en: 'the paper that brought BPE to neural networks (the low / lower / newest / widest example).', zh: '把 BPE 引入神经网络的论文（low / lower / newest / widest 的例子出自这里）。' })}</li>
			<li><a href="https://github.com/karpathy/minbpe" rel="external noopener">Andrej Karpathy, minbpe</a> — {t({ en: 'a short, readable BPE implementation, with a video lecture.', zh: '简短易读的 BPE 实现，附视频讲解。' })}</li>
			<li><a href="https://huggingface.co/learn/llm-course/chapter6/5" rel="external noopener">Hugging Face LLM course, chapter 6</a> — {t({ en: 'BPE, WordPiece and Unigram side by side.', zh: '并排讲解 BPE、WordPiece 和 Unigram。' })}</li>
			<li><a href="https://zh.d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html" rel="external noopener">{t({ en: 'Dive into Deep Learning: subword embedding (BPE), Chinese edition', zh: '《动手学深度学习》：子词嵌入（BPE）' })}</a> · <a href="https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html" rel="external noopener">English</a></li>
			<li><a href="https://arxiv.org/abs/2305.15425" rel="external noopener">Petrov et al. (2023), Language Model Tokenizers Introduce Unfairness Between Languages</a> — {t({ en: 'measures the “same meaning, different price” effect across many languages.', zh: '在许多语言上测量了“同样的意思、不同的价钱”这一现象。' })}</li>
			<li><a href="https://github.com/openai/tiktoken" rel="external noopener">OpenAI tiktoken</a> · <a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground" rel="external noopener">The Tokenizer Playground</a></li>
		</ul>
		<p class="muted small">
			{t({ en: `Tokenizer files on this page: GPT-2 (MIT), bert-base-uncased (Apache-2.0), cl100k_base (MIT, tiktoken), Qwen2.5 (Apache-2.0); ${fmtInt(578)} test strings checked against the reference implementations. See About for the full list of sources.`, zh: `本页用到的分词器文件：GPT-2（MIT）、bert-base-uncased（Apache-2.0）、cl100k_base（MIT，tiktoken）、Qwen2.5（Apache-2.0）；用 ${fmtInt(578)} 条测试文本与参考实现核对过。完整来源见“关于”页面。` })}
			<a href="{base}/about/">{t({ en: 'About', zh: '关于' })}</a>
		</p>
	{/snippet}
</ConceptPageShell>

<style>
	.sec {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.inbox {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
	}
	.flow {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		color: var(--ink-2);
	}
	.arrow {
		color: var(--muted);
	}
	.big {
		font-size: var(--fs-xl);
		color: var(--ink);
	}
	.ids {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		overflow-wrap: anywhere;
	}
	.err {
		color: var(--k-text);
	}
	.part {
		margin: var(--s5) 0 0;
		font-size: var(--fs-lg);
		padding-top: var(--s3);
		border-top: 1px dashed var(--rule);
	}
	.part:first-child {
		margin-top: 0;
		border-top: 0;
		padding-top: 0;
	}
	.lh {
		font-size: var(--fs-md);
		margin: var(--s3) 0 var(--s2);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
		gap: 10px;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.target {
		scroll-margin-top: calc(var(--header-h) + 16px);
		border-radius: var(--r3);
	}
	:global(.target.flash) {
		animation: flash 1.4s ease-out;
	}
	@keyframes flash {
		0% {
			box-shadow: 0 0 0 4px var(--accent);
		}
		100% {
			box-shadow: 0 0 0 4px transparent;
		}
	}
</style>
