<!--
  /train/tokenization/ — Tokenization concept page (Training methods, pre-training), as the lesson app (redesign spec §5):
  the page is the 7-step lesson (cover, then byte-pair encoding on the learner's own text); Explore (the hook, Part 1:
  train BPE on your text, Parts 2–3: TinyGPT's tokenizer and four real tokenizers, ten languages), Pitfalls and Sources
  are lazy panels one tab away (#explore, #pitfalls, #sources; the old #toy, #real, #compare, #post, #links still work).
  The input chip in the stage header opens the sheet with the two texts. Page code: $lib/pages/tokenization/ (TokPage is
  the shared state); tokenizer files in static/pages/tokenization/tok/ (pipeline/tokenizers/export.py, parity: parity.ts).
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import ConceptHost from '$lib/kit/lesson/ConceptHost.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import Lesson from '$lib/pages/tokenization/Lesson.svelte';
	import InputChip from '$lib/pages/tokenization/InputChip.svelte';
	import InputSheet from '$lib/pages/tokenization/InputSheet.svelte';
	import { TokPage, MAX_SENTENCE } from '$lib/pages/tokenization/page.svelte';
	import { conn, startClient, stopClient } from '$lib/pages/tokenization/state.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const P = new TokPage();

	onMount(() => {
		try {
			const q = new URLSearchParams(location.search).get('text');
			if (q) P.sentence = Array.from(q).slice(0, MAX_SENTENCE).join('');
		} catch {
			/* ignore */
		}
		const h = takeHandoff('text');
		if (h) P.sentence = Array.from(h.text).slice(0, MAX_SENTENCE).join('');
		startClient();
		void P.loadTiny();
		return () => stopClient();
	});

	// train BPE on the training text (debounced) whenever it changes or the worker becomes available. `trace` is read
	// untracked: tracking it made every new trace schedule another training run (a loop while the page sat idle).
	$effect(() => {
		const txt = P.trainText;
		if (!conn.client || !P.tiny) return;
		const id = setTimeout(() => P.train(txt), untrack(() => P.trace) ? 250 : 0);
		return () => clearTimeout(id);
	});

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

	/** the cover's reading time (spec §6.2, measured at the default inputs): 548 English narration words / 180 + 7 steps × 20 s, rounded */
	const MINUTES = 5;

	// Explore, Pitfalls and Sources are lazy chunks: the lesson shell carries none of them
	const views = {
		explore: () => import('$lib/pages/tokenization/panels/Explore.svelte'),
		pitfalls: () => import('$lib/pages/tokenization/panels/Pitfalls.svelte'),
		sources: () => import('$lib/pages/tokenization/panels/Sources.svelte')
	};
	// the panels get the page's state, the cover question (Pitfalls reveals it) and the related lessons (Sources)
	const viewProps = {
		P,
		prequestion,
		get related() {
			return data.related;
		}
	};
</script>

{#snippet inputChip(open: () => void)}<InputChip {P} {open} />{/snippet}
{#snippet inputSheet()}<InputSheet {P} />{/snippet}
{#snippet lesson()}<Lesson {P} />{/snippet}
{#snippet about()}
	<div class="about">
		<p>
			{t({
				en: 'Every number in the lesson is computed in your browser from your texts: BPE training in a Web Worker, TinyGPT’s tokenizer and the four real tokenizers (their files come from this site). Only TinyGPT’s training pair counts in step 6 were recorded when its tokenizer was trained, and are marked “precomputed”. The lesson keeps its own merge count, so Explore’s Part 1 stays where you left it.',
				zh: '课程中的每个数都由你的文本在浏览器中实时算出：BPE 训练在 Web Worker 中运行，TinyGPT 的分词器和四个真实分词器也在这里运行（它们的文件由本站提供）。只有第 6 步里 TinyGPT 的训练计数是在训练它的分词器时记录下来的，标为“预先计算”。课程有自己的合并次数，所以“探索”里的第 1 部分会停在你离开时的位置。'
			})}
		</p>
		<p class="muted">
			{t({
				en: 'Tokenizer files: GPT-2 (MIT), bert-base-uncased (Apache-2.0), cl100k_base (MIT, tiktoken), Qwen2.5 (Apache-2.0); token ids match the reference implementations on 578 of 578 test strings.',
				zh: '分词器文件：GPT-2（MIT）、bert-base-uncased（Apache-2.0）、cl100k_base（MIT，tiktoken）、Qwen2.5（Apache-2.0）；在 578 条测试文本上，词元 ID 与参考实现全部一致（578/578）。'
			})}
		</p>
	</div>
{/snippet}

<ConceptHost
	concept={data.conceptCatalog}
	{goals}
	{prequestion}
	summary={{
		en: 'A language model reads token ids, not letters. Train a byte-pair encoder on your own text, then watch TinyGPT, GPT-2, GPT-4 and Qwen2.5 cut your sentence.',
		zh: '语言模型读的是词元 ID，不是字母。在你自己的文本上训练一个字节对编码器，再看 TinyGPT、GPT-2、GPT-4 和 Qwen2.5 怎样切分你的句子。'
	}}
	minutes={MINUTES}
	exploreIntro={{
		en: 'Everything from the lesson at full size: train merge by merge with every count visible, follow TinyGPT’s tokenizer byte by byte, and compare six tokenizers and ten languages on your own text.',
		zh: '课程里的一切，完整展开：逐次合并训练并看到每个计数，一个字节一个字节地跟踪 TinyGPT 的分词器，再用你自己的文本对比六个分词器和十种语言。'
	}}
	{views}
	{viewProps}
	{inputChip}
	{inputSheet}
	{about}
	{lesson}
/>

<style>
	/* Kit issue (Walkthrough): the Look line's visually hidden "Look:" label is absolutely positioned with no positioned
	   ancestor inside the scrolling narration pane, so when the pane scrolls the label escapes it and makes the page a few
	   px taller than the window in app mode (measured: 688 px at 1280 x 680 on a long step). Anchor it to the pane. */
	:global(section.wt .narr) {
		position: relative;
	}
	.about {
		display: grid;
		gap: var(--s3);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.about p {
		margin: 0;
	}
</style>
