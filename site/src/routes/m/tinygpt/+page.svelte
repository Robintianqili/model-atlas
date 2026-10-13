<!--
  /m/tinygpt/ — TinyGPT, a 364,040-parameter GPT, explained on the learner's own text, as the lesson app (redesign
  spec §5): the lesson is the page. Every number is computed live in the browser (Web Worker) from the trained fp32
  weights; the forward pass matches PyTorch within 1e-4. The page state (the text, the runs, sampling and generation)
  is one TGPage (src/lib/pages/tinygpt/page.svelte.ts) shared by the lesson, the input sheet (the stage header's chip
  opens it) and the lazy views behind the top bar's tabs (#explore, #article, #code, #training; the old #overview,
  #inspector, #unit, #output, #tryThis, #checkpoint and #credits open them).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { PredictQuestion } from '$lib/kit/types';
	import { TGPage } from '$lib/pages/tinygpt/page.svelte';
	import { DEFAULT_TEXT } from '$lib/pages/tinygpt/presets';
	import Lesson from '$lib/pages/tinygpt/Lesson.svelte';
	import InputChip from '$lib/pages/tinygpt/InputChip.svelte';
	import Credits from '$lib/pages/tinygpt/Credits.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const P = new TGPage();
	// the About sheet's facts: the build-time spec facts (the engine brings the full spec later)
	const shellSpec = $derived({ name: data.catalog?.model.name ?? { en: 'TinyGPT', zh: 'TinyGPT' }, ...data.facts } as unknown as ModelSpec);

	onMount(() => {
		let start = DEFAULT_TEXT;
		try {
			const q = new URLSearchParams(location.search).get('text');
			const h = takeHandoff('text');
			if (q) start = q;
			else if (h?.text) start = h.text;
		} catch {
			/* ignore */
		}
		return P.load(new URL(`${base}/models/tinygpt/`, location.href).href, start);
	});

	const goals = [
		{ en: 'Follow one token from its id to a probability.', zh: '跟踪一个词元：从它的 id 一直到一个概率。' },
		{ en: 'Explain attention with real queries, keys and values.', zh: '用真实的查询、键和值解释注意力。' },
		{ en: 'See when the model decides, and how sampling picks.', zh: '看模型何时做出决定，以及采样如何挑选。' }
	];
	const summary = $derived({
		en: `TinyGPT has ${fmtInt(data.facts.params)} learned numbers. Step by step, you will follow your own text through it to the next token it writes.`,
		zh: `TinyGPT 有 ${fmtInt(data.facts.params)} 个学来的数。你将一步一步跟着自己的文本穿过它，直到它写出下一个词元。`
	});
	const prequestion: PredictQuestion = {
		id: 'tinygpt-where-decided',
		question: {
			en: 'TinyGPT reads “Once upon a time, there was a little” and must guess the next token. Where does that guess take shape?',
			zh: 'TinyGPT 读到 “Once upon a time, there was a little”，要猜下一个词元。这个猜测是在哪里成形的？'
		},
		options: [
			{ en: 'Only in the attention weights of the last block', zh: '只在最后一个块的注意力权重里' },
			{ en: 'Step by step: each attention and MLP step adds to the residual stream, and the guess sharpens over the blocks', zh: '一步步成形：每个注意力和 MLP 步骤都往残差流里加东西，猜测随着块的推进越来越清晰' },
			{ en: 'Only in the final softmax; the numbers before it mean nothing yet', zh: '只在最后的 softmax 里；之前的数字还没有任何意义' }
		],
		answer: 1,
		explain: {
			en: 'Step 10 of the lesson (and the logit lens in Explore) reads the residual stream after each step with the final LayerNorm and output head: the guess forms over several steps, and both attention and MLP steps move it.',
			zh: '课程第 10 步（以及“探索”里的 logit lens）用最终层归一化和输出头读取每一步之后的残差流：猜测在好几步中逐渐成形，注意力和 MLP 步骤都会推动它。'
		}
	};
</script>

<ModelPageShell
	slug="tinygpt"
	spec={shellSpec}
	downloadBytes={data.tinygptBytes}
	provenance="live"
	{goals}
	{prequestion}
	{summary}
	minutes={8}
	views={{
		explore: () => import('$lib/pages/tinygpt/views/Explore.svelte'),
		article: () => import('$lib/pages/tinygpt/views/Article.svelte'),
		code: () => import('$lib/pages/tinygpt/views/Code.svelte'),
		training: () => import('$lib/pages/tinygpt/views/Training.svelte')
	}}
	viewProps={{ P, prequestion }}
	exploreIntro={{
		en: 'The whole model at once: every tensor, all 16 attention heads, the logit lens for every position, sampling and experiments. Click anything.',
		zh: '整个模型一次摆开：每个张量、全部 16 个注意力头、每个位置的 logit lens、采样和各种实验。点哪里都可以。'
	}}
>
	{#snippet lesson()}<Lesson {P} />{/snippet}
	{#snippet inputChip(open: () => void)}<InputChip {P} {open} />{/snippet}
	{#snippet inputSheet()}
		{#if P.loadState === 'error'}
			<p class="err" role="alert">{t({ en: 'The model could not be loaded:', zh: '模型加载失败：' })} {P.loadError}</p>
		{/if}
		{#await import('$lib/pages/tinygpt/InputPanel.svelte')}
			<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
		{:then m}
			<m.default S={P.S} bind:text={P.text} promptIds={P.promptIds} genIds={P.genIds} oninput={(v: string) => void P.setPrompt(v)} />
		{/await}
	{/snippet}
	{#snippet facts()}
		<dl class="more">
			<div><dt>{t({ en: 'Shape', zh: '结构' })}</dt><dd class="num">{t({ en: '4 blocks · 4 heads · d 76 · context 64 · vocab 1,024', zh: '4 块 · 4 头 · d 76 · 上下文 64 · 词表 1024' })}</dd></div>
			<div>
				<dt>{t({ en: 'Where it runs', zh: '运行方式' })}</dt>
				<dd>
					{#if P.S.run}
						{P.S.run.where === 'worker'
							? t({ en: `own JS forward pass in a Web Worker · ${Math.round(P.S.run.ms)} ms`, zh: `自研 JS 前向计算（Web Worker）· ${Math.round(P.S.run.ms)} 毫秒` })
							: t({ en: `own JS forward pass on the main thread · ${Math.round(P.S.run.ms)} ms`, zh: `自研 JS 前向计算（主线程）· ${Math.round(P.S.run.ms)} 毫秒` })}
					{:else}
						{t({ en: 'own JS forward pass in a Web Worker', zh: '自研 JS 前向计算（Web Worker）' })}
					{/if}
				</dd>
			</div>
		</dl>
		<p class="pn">
			{t({
				en: 'Every one of the 66 traced tensors matches PyTorch within 1e-4 (the largest difference measured on fresh prompts was 4.2e-5, at a full 64-token context).',
				zh: '66 个被追踪的张量全部与 PyTorch 相差在 1e-4 以内（在新提示词上实测的最大差异为 4.2e-5，出现在满 64 个词元的上下文时）。'
			})}
		</p>
	{/snippet}
	{#snippet credits()}<Credits />{/snippet}
</ModelPageShell>

<style>
	.err {
		color: var(--no);
		font-weight: var(--w-ui);
	}
	.more {
		display: grid;
		gap: 10px;
		margin: 0;
	}
	.more dt {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.more dd {
		margin: 0;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.pn {
		margin: 10px 0 0;
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
