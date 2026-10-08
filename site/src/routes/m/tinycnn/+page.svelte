<!--
  /m/tinycnn/: the TinyCNN model page. Your drawing (or a preset or photo) runs through the real
  model in this page; every view below reads the same live trace (src/lib/pages/tinycnn/store).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { PredictQuestion } from '$lib/kit/types';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import InputPanel from '$lib/pages/tinycnn/InputPanel.svelte';
	import Overview from '$lib/pages/tinycnn/Overview.svelte';
	import Zoom from '$lib/pages/tinycnn/Zoom.svelte';
	import InspectorPanel from '$lib/pages/tinycnn/InspectorPanel.svelte';
	import OutputPanel from '$lib/pages/tinycnn/OutputPanel.svelte';
	import TryThis from '$lib/pages/tinycnn/TryThis.svelte';
	import TrainingPanel from '$lib/pages/tinycnn/TrainingPanel.svelte';
	import CodeSection from '$lib/pages/tinycnn/CodeSection.svelte';
	import Quiz from '$lib/pages/tinycnn/Quiz.svelte';
	import InputDock from '$lib/pages/tinycnn/InputDock.svelte';
	import Lesson from '$lib/pages/tinycnn/Lesson.svelte';
	import ArticleEn from '$lib/pages/tinycnn/article/En.svelte';
	import ArticleZh from '$lib/pages/tinycnn/article/Zh.svelte';
	import { CnnStore } from '$lib/pages/tinycnn/store.svelte';
	import { HOUSE, cloneStrokes } from '$lib/pages/tinycnn/sketches';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	// the spec is fixed for this page (embedded at build time); the store keeps it
	const S = new CnnStore((() => data.spec)());
	const baseUrl = `${base}/models/tinycnn`;

	let zoomEl = $state<HTMLElement>();
	function zoom() {
		zoomEl?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		zoomEl?.focus({ preventScroll: true });
	}

	onMount(() => {
		const h = takeHandoff('drawing');
		S.init(baseUrl, h?.strokes?.length ? h.strokes : cloneStrokes(HOUSE));
	});

	const goals = [
		{
			en: 'Compute one output cell of a convolution by hand: a 3×3 window in every input channel, times its kernel, summed, plus a bias.',
			zh: '亲手算出卷积的一个输出格：每个输入通道里的 3×3 窗口乘以对应的卷积核，求和，再加偏置。'
		},
		{
			en: 'Explain how ReLU, max pooling and stacking let deeper cells see larger parts of the drawing (receptive field 3 → 16 pixels).',
			zh: '解释 ReLU、最大池化和层层堆叠如何让更深的格子看到画中更大的区域（感受野从 3 到 16 像素）。'
		},
		{
			en: 'Read the softmax output critically: a high probability is a share among 10 known classes, not a guarantee of being right.',
			zh: '批判地读 softmax 输出：高概率只是在 10 个已知类别之间的份额，并不保证答对。'
		}
	];

	const prequestion: PredictQuestion = {
		id: 'tinycnn-rf',
		question: {
			en: 'A cell in the last conv layer (Conv 2-2) is computed from a 3×3 window. How much of your 28×28 drawing can that one cell “see”?',
			zh: '最后一个卷积层（卷积 2-2）的一个格子由一个 3×3 窗口算出。这一个格子能“看到”你 28×28 画面中的多大区域？'
		},
		options: [
			{ en: '3×3 pixels: the size of its kernel', zh: '3×3 像素：就是卷积核的大小' },
			{ en: '14×14 pixels: about a quarter of the drawing', zh: '14×14 像素：大约四分之一幅画' },
			{ en: 'All 28×28 pixels', zh: '全部 28×28 像素' }
		],
		answer: 1,
		explain: {
			en: 'Each 3×3 conv widens the view by 2 pixels and each 2×2 pool doubles the step between cells: 3 → 5 → 6 → 10 → 14. Select any Conv 2-2 cell: the zoom view outlines its 14×14 patch. Only the dense layer combines the whole drawing.',
			zh: '每个 3×3 卷积让视野加宽 2 像素，每个 2×2 池化让格子之间的步长加倍：3 → 5 → 6 → 10 → 14。选中卷积 2-2 的任意格子，放大视图会框出它的 14×14 区域。只有全连接层才把整幅画合在一起。'
		}
	};

	const quiz = [
		{
			id: 'tinycnn-q-products',
			q: { en: 'How many multiplications go into one cell of Conv 2-1 (before adding the bias)?', zh: '卷积 2-1 的一个格子（加偏置之前）需要多少次乘法？' },
			hint: { en: 'Click any Conv 2-1 map in the overview and look at the zoom view.', zh: '在总览中点击卷积 2-1 的任意一张图，看放大视图。' },
			options: [{ en: '9', zh: '9' }, { en: '72', zh: '72' }, { en: '144', zh: '144' }, { en: '256', zh: '256' }],
			answer: 1,
			explain: {
				en: 'Its input has 8 channels and each contributes a 3×3 window: 8 × 9 = 72 products, then one bias.',
				zh: '它的输入有 8 个通道，每个通道贡献一个 3×3 窗口：8 × 9 = 72 次乘法，再加一个偏置。'
			}
		},
		{
			id: 'tinycnn-q-params',
			q: { en: 'Which layer holds the most parameters?', zh: '哪一层的参数最多？' },
			hint: { en: 'See the parameter table in the Training section.', zh: '看“训练”一节里的参数表。' },
			options: [
				{ en: 'Conv 1-1, because it sees the full-size image', zh: '卷积 1-1，因为它看到的是全尺寸图像' },
				{ en: 'Conv 2-2', zh: '卷积 2-2' },
				{ en: 'Max pool 2', zh: '最大池化 2' },
				{ en: 'The dense layer', zh: '全连接层' }
			],
			answer: 3,
			explain: {
				en: 'Dense: 10 × 256 + 10 = 2,570 of the 6,722. Conv 1-1 has only 80 because its 9 weights per kernel are shared across all positions; pooling has none.',
				zh: '全连接层：10 × 256 + 10 = 2,570（共 6,722）。卷积 1-1 只有 80 个，因为每个卷积核的 9 个权重在所有位置共享；池化层一个也没有。'
			}
		}
	];
</script>

<ModelPageShell
	slug="tinycnn"
	spec={data.spec as unknown as ModelSpec}
	downloadBytes={data.bytes.auto}
	provenance="live"
	{goals}
	{prequestion}
	lessonNote={{ en: '10 short steps from your drawing to the answer, with the real numbers', zh: '10 个小步骤，从你的画一直到答案，每一步都是真实的数' }}
	exploreIntro={{
		en: 'The whole network at once: every layer, every map, the arithmetic of any cell, and experiments. Click anything.',
		zh: '整个网络一次摆开：每一层、每张特征图、任意格子的计算过程，还有各种实验。点哪里都可以。'
	}}
	titles={{
		overview: { en: 'The whole network, then zoom in', zh: '整个网络，再放大细看' },
		inspector: { en: 'Inside a layer', zh: '层的内部' }
	}}
	intros={{
		input: { en: 'Draw one of the 10 classes, pick a real drawing, or upload a photo of one. Everything below updates while you draw.', zh: '画出 10 个类别之一，选一幅真实的画，或者上传画的照片。你一边画，下面的内容就一边更新。' },
		lesson: {
			en: 'Follow your drawing through the network one step at a time. Press Next (or the → key), or Play to let the lesson run; each view uses your drawing and the real model.',
			zh: '一步一步跟着你的画穿过网络。点“下一步”（或按 → 键），或点“自动播放”让课程自己走；每个视图都用你的画和真实的模型。'
		},
		overview: {
			en: 'Every column is a layer, every tile a feature map, computed from your input just now. Lines show what feeds what. Click a map (or use the arrow keys) to zoom into its arithmetic below.',
			zh: '每一列是一层，每个小图是一张特征图，都由你的输入刚刚算出。连线表示数据从哪里来。点击一张图（或用方向键），在下方放大查看它的计算。'
		},
		inspector: { en: 'All channels of the selected layer, their statistics, the formula with live numbers, the code, and what each channel responds to.', zh: '选中层的全部通道、统计、带实时数字的公式、代码，以及每个通道对什么有反应。' },
		tryThis: { en: 'Each experiment sets up the page for you. Predict first, then look.', zh: '每个实验都会替你把页面设置好。先预测，再看结果。' },
		training: { en: 'How these 6,722 numbers were learned, and how well they work. These are measurements made once, offline.', zh: '这 6,722 个数是怎么学来的，效果如何。这些是离线测过一次的结果。' }
	}}
	badges={{ input: 'live', lesson: 'live', overview: 'live', inspector: 'live', output: 'live', tryThis: 'live', training: 'precomputed' }}
>
	{#snippet input()}<InputPanel {S} />{/snippet}
	{#snippet lesson()}<Lesson {S} />{/snippet}
	{#snippet overview()}
		<Overview {S} onzoom={zoom} />
		<Zoom {S} bind:el={zoomEl} />
	{/snippet}
	{#snippet inspector()}<InspectorPanel {S} />{/snippet}
	{#snippet output()}<OutputPanel {S} onzoom={zoom} />{/snippet}
	{#snippet tryThis()}<TryThis {S} {baseUrl} />{/snippet}
	{#snippet training()}<TrainingPanel spec={data.spec} />{/snippet}
	{#snippet article()}
		<div lang={i18n.lang === 'zh' ? 'zh-CN' : 'en'}>
			{#if i18n.lang === 'zh'}<ArticleZh {S} onshow={zoom} />{:else}<ArticleEn {S} onshow={zoom} />{/if}
		</div>
	{/snippet}
	{#snippet code()}<CodeSection {S} />{/snippet}
	{#snippet checkpoint()}<Quiz questions={quiz} />{/snippet}
	{#snippet credits()}
		<ul class="credits">
			<li>
				{t({ en: 'Training data and preset drawings', zh: '训练数据与预设画' })}:
				<a href="https://github.com/googlecreativelab/quickdraw-dataset" rel="noopener">Quick, Draw! dataset</a>, Google Creative Lab, CC BY 4.0.
			</li>
			<li>
				{t({ en: 'Layout idea (TinyVGG, per-cell convolution arithmetic)', zh: '布局思路（TinyVGG、逐格卷积计算）' })}:
				<a href="https://github.com/poloclub/cnn-explainer" rel="noopener">CNN Explainer</a>, Wang et al., Georgia Tech Polo Club (MIT). {t({ en: 'No code copied.', zh: '未复制代码。' })}
			</li>
			<li>{t({ en: 'Model, weights and this page: Model Atlas. Full list on the About page.', zh: '模型、权重和本页：Model Atlas。完整列表见“关于”页。' })} <a href="{base}/about/">{t({ en: 'Licenses', zh: '许可' })}</a></li>
		</ul>
	{/snippet}
</ModelPageShell>

<InputDock {S} />

<style>
	.credits {
		margin: 0;
		padding-left: 1.2em;
		font-size: var(--fs-sm);
	}
	.credits li + li {
		margin-top: 4px;
	}
</style>
