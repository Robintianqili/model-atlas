<!--
  /m/tinycnn/: the TinyCNN model page as the lesson app (redesign spec §5): the lesson is the page. Your drawing (or a
  preset or a photo) runs through the real model in this page; the lesson, the cover and every view read the same live
  trace (src/lib/pages/tinycnn/store). The input chip in the stage header opens the input sheet (InputPanel, loaded on
  first open); Explore, How it works, Code and Training are lazy views behind the top bar's tabs (#explore, #article,
  #code, #training; the old #overview, #inspector, #output, #tryThis, #checkpoint, #credits and #try-<id> open them).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import { fmtInt } from '$lib/kit/format';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { PredictQuestion } from '$lib/kit/types';
	import { t } from '$lib/i18n/i18n.svelte';
	import Lesson from '$lib/pages/tinycnn/Lesson.svelte';
	import InputChip from '$lib/pages/tinycnn/InputChip.svelte';
	import Credits from '$lib/pages/tinycnn/Credits.svelte';
	import { CnnStore } from '$lib/pages/tinycnn/store.svelte';
	import { HOUSE, cloneStrokes } from '$lib/pages/tinycnn/sketches';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	// the spec is fixed for this page (embedded at build time); the store keeps it
	const S = new CnnStore((() => data.spec)());
	const baseUrl = `${base}/models/tinycnn`;
	// the top bar names the model as the catalog does ("TinyCNN"), not the spec's long name
	const shellSpec = $derived({ ...data.spec, name: data.catalog?.model.name ?? data.spec.name } as unknown as ModelSpec);

	onMount(() => {
		const h = takeHandoff('drawing');
		S.init(baseUrl, h?.strokes?.length ? h.strokes : cloneStrokes(HOUSE));
	});

	const goals = [
		{ en: 'Compute one cell of a convolution by hand.', zh: '亲手算出卷积的一个格子。' },
		{ en: 'Explain why deeper cells see more of the drawing.', zh: '解释为什么越深的格子能看到画中越大的区域。' },
		{ en: 'Read the softmax answer critically.', zh: '批判地读 softmax 给出的答案。' }
	];
	const summary = $derived({
		en: `TinyCNN has ${fmtInt(data.spec.params)} learned numbers. Step by step, you will compute its answer for your own drawing, one layer at a time.`,
		zh: `TinyCNN 有 ${fmtInt(data.spec.params)} 个学来的数。你将一步一步、一层一层地算出它对你自己的画给出的答案。`
	});

	const prequestion: PredictQuestion = {
		id: 'tinycnn-rf',
		question: {
			en: 'A cell in the last conv layer (Conv 2-2) is computed from a 3×3 window. How much of your 28×28 drawing can that one cell see?',
			zh: '最后一个卷积层（卷积 2-2）的一个格子由一个 3×3 窗口算出。这一个格子能看到你 28×28 画面中的多大区域？'
		},
		options: [
			{ en: '3×3 pixels', zh: '3×3 像素' },
			{ en: '14×14 pixels', zh: '14×14 像素' },
			{ en: 'All 28×28', zh: '全部 28×28' }
		],
		answer: 1,
		explain: {
			en: 'Each 3×3 conv widens the view by 2 pixels and each 2×2 pool doubles the step between cells: 3 → 5 → 6 → 10 → 14. Step 7 of the lesson draws the nested boxes; in Explore, select any Conv 2-2 cell and the zoom view outlines its 14×14 patch. Only the dense layer combines the whole drawing.',
			zh: '每个 3×3 卷积让视野加宽 2 像素，每个 2×2 池化让格子之间的步长加倍：3 → 5 → 6 → 10 → 14。课程第 7 步画出了一层套一层的方框；在“探索”里选中卷积 2-2 的任意格子，放大视图会框出它的 14×14 区域。只有全连接层才把整幅画合在一起。'
		}
	};
</script>

<ModelPageShell
	slug="tinycnn"
	spec={shellSpec}
	downloadBytes={data.bytes.auto}
	provenance="live"
	{goals}
	{prequestion}
	{summary}
	minutes={7}
	views={{
		explore: () => import('$lib/pages/tinycnn/views/Explore.svelte'),
		article: () => import('$lib/pages/tinycnn/views/Article.svelte'),
		code: () => import('$lib/pages/tinycnn/views/Code.svelte'),
		training: () => import('$lib/pages/tinycnn/views/Training.svelte')
	}}
	viewProps={{ S, spec: data.spec, baseUrl, prequestion }}
	exploreIntro={{
		en: 'The whole network at once: every layer, every map, the arithmetic of any cell, and experiments. Click anything.',
		zh: '整个网络一次摆开：每一层、每张特征图、任意格子的计算过程，还有各种实验。点哪里都可以。'
	}}
>
	{#snippet lesson()}<Lesson {S} />{/snippet}
	{#snippet inputChip(open: () => void)}<InputChip {S} {open} />{/snippet}
	{#snippet inputSheet()}
		{#await import('$lib/pages/tinycnn/InputPanel.svelte')}
			<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
		{:then m}
			<m.default {S} />
		{/await}
	{/snippet}
	{#snippet credits()}<Credits />{/snippet}
</ModelPageShell>
