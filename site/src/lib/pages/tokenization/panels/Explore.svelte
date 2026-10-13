<!--
  Explore (redesign spec §5.9; a lazy panel of the tokenization page, #explore): everything from the lesson at full size.
    #explore-hook     what does a model read? (the sentence, its bytes and TinyGPT ids)
    #explore-toy      Part 1: train BPE on your own text, merge by merge (the old #toy)
    #explore-real     Part 2: TinyGPT's tokenizer step by step; Part 3: four real tokenizers side by side (#real)
    #explore-compare  ten languages, six tokenizers (#compare)
  It reads and writes the page's TokPage (vp.P), so the texts and the trained trace are the lesson's.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Section from '$lib/kit/Section.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import BpeToy from '../BpeToy.svelte';
	import TinyGptPart from '../TinyGptPart.svelte';
	import RealCompare from '../RealCompare.svelte';
	import LangTable from '../LangTable.svelte';
	import TryThis from '../TryThis.svelte';
	import ReadsStrip from '../ReadsStrip.svelte';
	import Hook from '../content/Hook.svelte';
	import HookQuestions from '../content/HookQuestions.svelte';
	import ToyIntro from '../content/ToyIntro.svelte';
	import ToyAfter from '../content/ToyAfter.svelte';
	import TinyIntro from '../content/TinyIntro.svelte';
	import RealIntro from '../content/RealIntro.svelte';
	import Compare from '../content/Compare.svelte';
	import { PRESETS } from '../presets';
	import { MAX_SENTENCE, type TokPage } from '../page.svelte';
	import type { TryItem } from '../types';

	let { vp }: { vp: { P: TokPage }; section?: string } = $props();
	const P = $derived(vp.P);

	let flashTid = 0;
	function scrollToId(id: string) {
		requestAnimationFrame(() => {
			const el = document.getElementById(id);
			if (!el) return;
			el.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'center' });
			el.classList.remove('flash');
			void el.offsetWidth;
			el.classList.add('flash');
			clearTimeout(flashTid);
			flashTid = window.setTimeout(() => el.classList.remove('flash'), 1500);
		});
	}
	onMount(() => () => clearTimeout(flashTid));
	function setTrain(id: string, merges: number | 'end' = 0) {
		const p = PRESETS.train.find((x) => x.id === id);
		if (p) P.trainText = p.text;
		P.k = merges === 'end' ? 767 : merges;
		scrollToId('toy-player');
	}
	function setSentence(text: string, target = 'real-input') {
		P.sentence = text;
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
				P.k = 767; // BpeToy clamps this to the last merge it learned; the learner's own text is kept
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
</script>

<div class="explore">
	<Section id="explore-hook" title={{ en: 'What does a model read?', zh: '模型读到的是什么？' }} provenance="live">
		<div class="sec">
			<Hook />
			<div class="panel inbox">
				<TextInputBar bind:value={P.sentence} maxLength={MAX_SENTENCE} label={{ en: 'Your sentence (the lesson uses it too)', zh: '你的句子（课程也用它）' }} presets={PRESETS.sentence} />
				<ReadsStrip {P} />
			</div>
			<HookQuestions />
		</div>
	</Section>

	<Section id="explore-toy" title={{ en: 'Part 1 · Build a tokenizer from your own text', zh: '第 1 部分 · 用你自己的文本训练分词器' }} provenance="live">
		<div class="sec">
			<ToyIntro />
			<TryThis items={toyTries} />
			<div id="toy-player" class="target">
				<BpeToy {P} />
			</div>
			<ToyAfter />
		</div>
	</Section>

	<Section id="explore-real" title={{ en: 'Parts 2 and 3 · Real tokenizers at work', zh: '第 2、3 部分 · 真实分词器' }} provenance="live">
		<div class="sec">
			<h3 class="part">{t({ en: 'Part 2 · TinyGPT’s tokenizer, step by step', zh: '第 2 部分 · 一步一步看 TinyGPT 的分词器' })}</h3>
			<TinyIntro />
			<div id="real-input" class="panel inbox target">
				<TextInputBar bind:value={P.sentence} maxLength={MAX_SENTENCE} label={{ en: 'Your sentence', zh: '你的句子' }} presets={PRESETS.sentence} />
			</div>
			{#if P.tiny}
				<TinyGptPart tok={P.tiny} sentence={P.sentence} counts={P.mergeCounts} />
			{/if}

			<h3 class="part">{t({ en: 'Part 3 · Real tokenizers side by side', zh: '第 3 部分 · 真实分词器并排对比' })}</h3>
			<RealIntro />
			<TryThis items={realTries} />
			<div id="real-rows" class="target">
				{#if P.tiny}
					<RealCompare sentence={P.sentence} tiny={P.tiny} trace={P.trace} k={P.k} />
				{/if}
			</div>
		</div>
	</Section>

	<Section id="explore-compare" title={{ en: 'Same meaning, different price', zh: '同样的意思，不同的价钱' }} provenance="live">
		<div class="sec">
			<Compare part="intro" />
			{#if P.tiny}
				<LangTable tiny={P.tiny} trace={P.trace} k={P.k} onuse={(s) => setSentence(s, 'real-rows')} />
			{/if}
			<Compare part="after" />
		</div>
	</Section>
</div>

<style>
	.explore {
		display: grid;
		gap: var(--s7);
	}
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
	.part {
		margin: var(--s5) 0 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		padding-top: var(--s3);
		border-top: 1px dashed var(--rule);
	}
	.part:first-child {
		margin-top: 0;
		border-top: 0;
		padding-top: 0;
	}
	.target {
		scroll-margin-top: calc(var(--ltop-h) + 16px);
		border-radius: var(--r3);
	}
	:global(.target.flash) {
		animation: flash 1.4s ease-out;
	}
	@keyframes flash {
		0% {
			box-shadow: 0 0 0 4px var(--mark);
		}
		100% {
			box-shadow: 0 0 0 4px transparent;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		:global(.target.flash) {
			animation: none;
		}
	}
</style>
