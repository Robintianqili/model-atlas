<!--
  /train/transfer-learning/ — Transfer learning and fine-tuning (concept page), as the lesson app (redesign spec §5):
    Lesson   the page: the guided 8-step lesson (kit Walkthrough; steps 1–6 live, 7–8 precomputed), with its cover;
             the drawing is picked or drawn with the stage header's chip and its sheet
    Explore  (lazy, #explore) hook: the 10-class TinyCNN must call a ladder one of its 10 classes; Part 1: freeze the
             body, train a new 2-way head on your drawings (live); Part 2: head only vs full fine-tune vs from scratch
             (precomputed, finetune.json); Part 3: our BERT reproduction (precomputed); the old #toy, #real, #compare
    Pitfalls (lazy, #pitfalls) two misconception checks + other pitfalls; the cover question revealed + checkpoint
    Sources  (lazy, #sources) the related lessons, our BERT reproduction, the glossary
  Page-specific code lives in src/lib/pages/transfer-learning/.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import ConceptHost from '$lib/kit/lesson/ConceptHost.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import Lesson from '$lib/pages/transfer-learning/Lesson.svelte';
	import InputChip from '$lib/pages/transfer-learning/InputChip.svelte';
	import InputSheet from '$lib/pages/transfer-learning/InputSheet.svelte';
	import { LessonState, LESSON_N } from '$lib/pages/transfer-learning/lessonState.svelte';
	import { getEngine, type Engine } from '$lib/pages/transfer-learning/engine';
	import { fetchJson, type HeldoutJson, type PcaJson } from '$lib/pages/transfer-learning/data';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const tolerance = $derived(page.data.liveSpecs?.tinycnn?.parity?.max_abs_diff);

	let engine = $state.raw<Engine | null>(null);
	let pca = $state.raw<PcaJson | null>(null);
	let heldout = $state.raw<HeldoutJson | null>(null);
	let loadError = $state<string | null>(null);
	// the guided lesson's own state (it never moves the Explore view)
	const L = new LessonState();

	onMount(async () => {
		const url = `${base}/models/tinycnn`;
		try {
			const [e, p, h] = await Promise.all([getEngine(url), fetchJson<PcaJson>(`${url}/pca.json`), fetchJson<HeldoutJson>(`${url}/heldout.json`)]);
			engine = e;
			pca = p;
			heldout = h;
			L.init(e, h, p, data.finetune);
		} catch (err) {
			loadError = err instanceof Error ? err.message : String(err);
		}
	});
	onDestroy(() => engine?.dispose());

	const goals = [
		{
			en: 'Explain what it means to freeze a pre-trained network and train a new head, and count the head’s trainable parameters (514 here).',
			zh: '说清楚“冻结预训练网络、训练新输出头”是什么意思，并数出输出头的可训练参数（这里是 514 个）。'
		},
		{
			en: 'Train a head on drawings of your own, and use a map of the frozen features to explain why a few examples can be enough.',
			zh: '用你自己的画训练一个输出头，并借助冻结特征的地图，解释为什么几个样本就可能够用。'
		},
		{
			en: 'Choose between a frozen head, full fine-tuning and training from scratch based on how much data you have, with evidence from TinyCNN and our BERT reproduction.',
			zh: '根据手头数据的多少，在“冻结 + 新输出头”、全量微调和从零训练之间做选择，并用 TinyCNN 与我们的 BERT 复现结果作为依据。'
		}
	];

	const prequestion: PredictQuestion = {
		id: 'transfer-learning-n1',
		question: {
			en: 'TinyCNN must learn two new classes from just 1 drawing of each. Which approach scores best on new drawings?',
			zh: 'TinyCNN 要从每类仅 1 张画学会两个新类别。哪种方法在新画上得分最高？'
		},
		options: [
			{ en: 'Freeze the network, train only a new head (514 parameters)', zh: '冻结网络，只训练新输出头（514 个参数）' },
			{ en: 'Fine-tune every layer (4,666 parameters)', zh: '微调所有层（4,666 个参数）' },
			{ en: 'Train the whole network from scratch (4,666 parameters)', zh: '从零训练整个网络（4,666 个参数）' },
			{ en: 'All about the same', zh: '差不多' }
		],
		answer: 0,
		explain: {
			en: 'Averaged over 5 random seeds on 3,000 test drawings: head only 93.1%, full fine-tune 85.9%, from scratch 78.7%. With almost no data, a few trainable parameters on top of good frozen features win. Head only and full fine-tuning meet at about 16 drawings per class; training from scratch catches up only with thousands (Explore, Part 2).',
			zh: '在 3,000 张测试图上、5 个随机种子的平均：只训练输出头 93.1%，全量微调 85.9%，从零训练 78.7%。数据几乎为零时，在好的冻结特征上只训练少量参数的方案胜出。到每类约 16 张时，只训练输出头和全量微调基本持平；从零训练要到每类几千张才追上（见“探索”第 2 部分）。'
		}
	};

	/** the cover's reading time (spec §6.2, measured at the default inputs): 712 English narration words / 180 + 8 steps × 20 s, rounded */
	const MINUTES = 7;

	// Explore, Pitfalls and Sources are lazy chunks: the lesson shell carries none of them
	const views = {
		explore: () => import('$lib/pages/transfer-learning/panels/Explore.svelte'),
		pitfalls: () => import('$lib/pages/transfer-learning/panels/Pitfalls.svelte'),
		sources: () => import('$lib/pages/transfer-learning/panels/Sources.svelte')
	};
	// the panels share the page's engine and data files, the cover question (Pitfalls) and the related lessons (Sources)
	const viewProps = {
		prequestion,
		get engine() {
			return engine;
		},
		get heldout() {
			return heldout;
		},
		get pca() {
			return pca;
		},
		get finetune() {
			return data.finetune;
		},
		get tolerance() {
			return tolerance;
		},
		get loadError() {
			return loadError;
		},
		get related() {
			return data.related;
		}
	};
</script>

{#snippet inputChip(open: () => void)}<InputChip {L} {open} />{/snippet}
{#snippet inputSheet()}<InputSheet {L} />{/snippet}
{#snippet lesson()}<Lesson {L} {tolerance} error={loadError} />{/snippet}
{#snippet about()}
	<p class="about">
		{t({
			en: `Steps 1–6 are computed in your browser: your drawing through the shipped TinyCNN, and a new head trained on ${LESSON_N} + ${LESSON_N} preset drawings (the map’s axes and gray points and the top-guess shares in step 1 were computed offline from the same model). Steps 7 and 8 are precomputed: TinyCNN fine-tuning runs (5 seeds) and our BERT reproduction. The lesson keeps its own state, so Explore stays where you left it.`,
			zh: `第 1–6 步都在你的浏览器里实时计算：你的画经过发布的 TinyCNN，新输出头用 ${LESSON_N} + ${LESSON_N} 张预置图训练（地图的坐标轴和灰色参照点，以及第 1 步的首选答案比例，是用同一个模型离线算好的）。第 7、8 步是预计算结果：TinyCNN 的微调实验（5 个种子）和我们的 BERT 复现。课程使用自己的状态，所以“探索”会停在你离开时的位置。`
		})}
	</p>
{/snippet}

<ConceptHost
	concept={data.conceptCatalog}
	{goals}
	{prequestion}
	summary={{
		en: 'Teach TinyCNN two new drawing classes with a handful of examples: freeze its trained layers and train only a new head, live in your browser.',
		zh: '只用几张画，教 TinyCNN 认识两个新类别：冻结它已经训练好的层，只在你的浏览器里实时训练一个新输出头。'
	}}
	minutes={MINUTES}
	exploreIntro={{
		en: 'Everything at once: train a head on your own two classes, mislabel examples, read every weight, compare all training-set sizes, and see the full BERT results.',
		zh: '所有内容一次摆开：用你自己的两个类别训练输出头、故意标错样本、查看每一个权重、比较各种训练集大小，以及完整的 BERT 结果。'
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
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
