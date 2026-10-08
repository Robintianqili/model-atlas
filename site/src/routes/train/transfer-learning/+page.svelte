<!--
  /train/transfer-learning/ — Transfer learning and fine-tuning (concept page, ConceptPageShell order).
    hook     the 10-class TinyCNN must call a ladder one of its 10 classes (live + precomputed)
    lesson   "Step by step": the guided 8-step lesson (kit Walkthrough; steps 1–6 live, 7–8 precomputed)
    — "Explore freely" —
    toy      Part 1: freeze the body, train a new 2-way head on your drawings (live)
    real     Part 2: head only vs full fine-tune vs from scratch (precomputed, finetune.json)
    compare  Part 3: at real scale, our BERT reproduction (precomputed, research/bert_facts.json,
             research/bert_improve.json)
    pitfalls two misconception checks + other pitfalls; post: predict-reveal + checkpoint; links
  Page-specific code lives in src/lib/pages/transfer-learning/.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import ConceptPageShell from '$lib/kit/ConceptPageShell.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import { CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import Article from '$lib/pages/transfer-learning/ArticlePart.svelte';
	import HookPanel from '$lib/pages/transfer-learning/HookPanel.svelte';
	import HeadLab from '$lib/pages/transfer-learning/HeadLab.svelte';
	import CurveChart from '$lib/pages/transfer-learning/CurveChart.svelte';
	import BertPanel from '$lib/pages/transfer-learning/BertPanel.svelte';
	import Misconception from '$lib/pages/transfer-learning/Misconception.svelte';
	import Checkpoint from '$lib/pages/transfer-learning/Checkpoint.svelte';
	import Lesson from '$lib/pages/transfer-learning/Lesson.svelte';
	import { LessonState } from '$lib/pages/transfer-learning/lessonState.svelte';
	import { getEngine, type Engine } from '$lib/pages/transfer-learning/engine';
	import { fetchJson, type HeldoutJson, type PcaJson } from '$lib/pages/transfer-learning/data';
	import { BERT_SLIDES } from '$lib/pages/transfer-learning/bert';
	import { lab, goTo } from '$lib/pages/transfer-learning/state.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const finetune = $derived(data.finetune);
	const tolerance = $derived(page.data.liveSpecs?.tinycnn?.parity?.max_abs_diff);

	let engine = $state.raw<Engine | null>(null);
	let pca = $state.raw<PcaJson | null>(null);
	let heldout = $state.raw<HeldoutJson | null>(null);
	let loadError = $state<string | null>(null);
	// the guided lesson's own state (it never moves the explore parts below)
	const L = new LessonState();
	let hookOpen = $state(false);

	onMount(async () => {
		const url = `${base}/models/tinycnn`;
		try {
			const [e, p, h] = await Promise.all([
				getEngine(url),
				fetchJson<PcaJson>(`${url}/pca.json`),
				fetchJson<HeldoutJson>(`${url}/heldout.json`)
			]);
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
			en: 'Averaged over 5 random seeds on 3,000 test drawings: head only 93.1%, full fine-tune 85.9%, from scratch 78.7%. With almost no data, a few trainable parameters on top of good frozen features win. Head only and full fine-tuning meet at about 16 drawings per class; training from scratch catches up only with thousands (Part 2).',
			zh: '在 3,000 张测试图上、5 个随机种子的平均：只训练输出头 93.1%，全量微调 85.9%，从零训练 78.7%。数据几乎为零时，在好的冻结特征上只训练少量参数的方案胜出。到每类约 16 张时，只训练输出头和全量微调基本持平；从零训练要到每类几千张才追上（见第 2 部分）。'
		}
	};

	const furtherSlugs = ['probe-vs-finetune', 'catastrophic-forgetting', 'lora', 'multitask-intermediate', 'vision-finetuning', 'lp-ft'];
	const furtherConcepts = $derived(furtherSlugs.map((s) => CONCEPT_BY_SLUG[s]).filter(Boolean));

	function showPart2(n: number, seeds = false) {
		lab.part2N = n;
		lab.part2Zoom = false;
		if (seeds) lab.part2Seeds = true;
		goTo('tl-curves');
	}
</script>

<ConceptPageShell
	slug="transfer-learning"
	{goals}
	{prequestion}
	lede={{
		en: 'Teach TinyCNN two new drawing classes with a handful of examples: freeze its trained layers and train only a new head, live in your browser. Then compare head-only, full fine-tuning and training from scratch, and see the same trade-off at BERT scale.',
		zh: '只用几张画，教 TinyCNN 认识两个新类别：冻结它已经训练好的层，只在你的浏览器里实时训练一个新输出头。然后比较只训练输出头、全量微调和从零训练，再看看同样的取舍在 BERT 规模上是什么样子。'
	}}
	lessonNote={{ en: '8 short steps from a forced guess to a trained head, with your drawing and the real numbers', zh: '8 个小步骤，从被迫的猜测到训练好的输出头，用你的画和真实的数' }}
	exploreIntro={{
		en: 'Everything at once: train a head on your own two classes, mislabel examples, read every weight, compare all training-set sizes, and see the full BERT results.',
		zh: '所有内容一次摆开：用你自己的两个类别训练输出头、故意标错样本、查看每一个权重、比较各种训练集大小，以及完整的 BERT 结果。'
	}}
	titles={{
		hook: { en: 'A model that knows only 10 things', zh: '只认识 10 样东西的模型' },
		toy: { en: 'Part 1 · Freeze the body, train a new head', zh: '第 1 部分 · 冻结主体，训练新输出头' },
		real: { en: 'Part 2 · Head only vs full fine-tune vs from scratch', zh: '第 2 部分 · 只训练输出头 vs 全量微调 vs 从零训练' },
		compare: { en: 'Part 3 · At real scale: our BERT reproduction', zh: '第 3 部分 · 真实规模：我们的 BERT 复现' },
		pitfalls: { en: 'Misconceptions and pitfalls', zh: '误区与陷阱' },
		post: { en: 'Checkpoint', zh: '检查点' }
	}}
	intros={{
		lesson: {
			en: 'Follow one drawing from a forced 10-way guess to a new head that learns ladder vs envelope. Press Next (or the → key), or Play to let the lesson run. Steps 1–6 compute live in your browser; steps 7–8 show precomputed results.',
			zh: '跟着一张画，从被迫的 10 选 1 猜测，走到一个学会区分梯子和信封的新输出头。点“下一步”（或按 → 键），或点“自动播放”让课程自己走。第 1–6 步在你的浏览器里实时计算；第 7–8 步展示预计算的结果。'
		}
	}}
	badges={{ lesson: 'live', toy: 'live', real: 'precomputed', compare: 'precomputed' }}
>
	{#snippet lesson()}<Lesson {L} {tolerance} />{/snippet}

	{#snippet hook()}
		<div class="flow">
			<Article part="hook-intro" />
			<!-- the same question is step 1 of the lesson right below, so the live hook panel is optional here -->
			<details class="hookpanel" bind:open={hookOpen}>
				<summary>{t({ en: 'Show it one here: a ladder, an envelope or your own drawing (step 1 of the lesson below does the same)', zh: '在这里给它看一张：梯子、信封或你自己的画（下面课程的第 1 步也是这样做的）' })}</summary>
				{#if hookOpen}<div class="hookin"><HookPanel {engine} {heldout} {finetune} {tolerance} /></div>{/if}
			</details>
			<Article part="hook-after" />
			{#if loadError}<p class="err" role="alert">{t({ en: `Could not load the model files: ${loadError}`, zh: `模型文件加载失败：${loadError}` })}</p>{/if}
		</div>
	{/snippet}

	{#snippet toy()}
		<div class="flow">
			<Article part="toy-intro" />
			<HeadLab {engine} {heldout} {pca} {tolerance} />
		</div>
	{/snippet}

	{#snippet real()}
		<div class="flow" id="tl-curves">
			<Article part="real-intro" />
			<CurveChart {finetune} />
			<div class="try">
				<p><strong>{t({ en: 'Try this:', zh: '试一试：' })}</strong> {t({ en: 'predict the winner, then jump to that size.', zh: '先预测谁赢，再跳到对应的数据量。' })}</p>
				<div class="cluster">
					<button type="button" class="btn" onclick={() => (lab.part2N = 1)}>{t({ en: '1 per class', zh: '每类 1 张' })}</button>
					<button type="button" class="btn" onclick={() => (lab.part2N = 16)}>{t({ en: '16 per class', zh: '每类 16 张' })}</button>
					<button type="button" class="btn" onclick={() => (lab.part2N = 2048)}>{t({ en: '2,048 per class', zh: '每类 2,048 张' })}</button>
				</div>
			</div>
			<Article part="real-after" />
		</div>
	{/snippet}

	{#snippet compare()}
		<BertPanel />
	{/snippet}

	{#snippet pitfalls()}
		<div class="flow">
			<Misconception
				statement={{ en: 'Fine-tuning everything is always better than freezing.', zh: '微调所有层总是比冻结更好。' }}
				truth={false}
				show={{ label: { en: 'Show me: 1 drawing per class in Part 2', zh: '带我看：第 2 部分中每类 1 张的情况' }, action: () => showPart2(1) }}
			>
				{#snippet evidence()}
					<p>
						{t({
							en: 'It depends on how much data you have. With 1 drawing per class, the frozen head scores 93.1% and full fine-tuning 85.9% (5 seeds). Full fine-tuning only wins at 2,048 per class (98.59% vs 98.41%). On SST-2, with tens of thousands of labeled sentences, fine-tuned BERT clearly beats the best probe on its frozen features (93.0% vs 86.8%). More data, or a task far from the pre-training, favors fine-tuning; little data favors freezing.',
							zh: '这取决于你有多少数据。每类 1 张图时，冻结方案得 93.1%，全量微调只有 85.9%（5 个种子）。要到每类 2,048 张，全量微调才胜出（98.59% 对 98.41%）。而在有几万条带标签句子的 SST-2 上，微调后的 BERT 明显胜过基于其冻结特征的最佳探针（93.0% 对 86.8%）。数据多、或新任务离预训练很远时，微调占优；数据少时，冻结占优。'
						})}
					</p>
				{/snippet}
			</Misconception>
			<Misconception
				statement={{ en: 'With little data, more trainable parameters is better: the model can learn more.', zh: '数据少的时候，可训练参数越多越好：模型能学到更多。' }}
				truth={false}
				show={{ label: { en: 'Show me the spread between seeds', zh: '带我看各种子之间的差异' }, action: () => showPart2(1, true) }}
			>
				{#snippet evidence()}
					<p>
						{t({
							en: 'With 1 drawing per class: 514 trainable parameters (frozen body) give 93.1% ± 2.5; 4,666 starting from pre-trained weights give 85.9% ± 7.1; 4,666 starting from random give 78.7% ± 19.7. More parameters can fit the two drawings in more wrong ways, and results swing more between seeds. At BERT scale, fine-tuning BERT-Large on RTE’s 2,490 pairs collapsed in 4 of 6 runs with the original optimizer.',
							zh: '每类 1 张图时：514 个可训练参数（冻结主体）得 93.1% ± 2.5；从预训练权重出发的 4,666 个参数得 85.9% ± 7.1；从随机权重出发的 4,666 个参数得 78.7% ± 19.7。参数越多，拟合这两张图的“错误方式”也越多，不同种子之间的结果波动也越大。在 BERT 规模上，用原版优化器在 RTE 的 2,490 对样本上微调 BERT-Large，6 次里有 4 次崩溃。'
						})}
					</p>
				{/snippet}
			</Misconception>
			<Article part="pitfalls" />
		</div>
	{/snippet}

	{#snippet post()}
		<div class="flow">
			<Checkpoint />
			<Article part="post" />
		</div>
	{/snippet}

	{#snippet further()}
		<h3>{t({ en: 'Go deeper', zh: '继续深入' })}</h3>
		<ul>
			<li>
				<a href="{base}/m/tinycnn/">{t({ en: 'TinyCNN, layer by layer', zh: 'TinyCNN 逐层详解' })}</a>
				— {t({ en: 'every convolution of the frozen body, on your own drawing.', zh: '在你自己的画上，看冻结主体里的每一次卷积。' })}
			</li>
			<li>
				<a href={BERT_SLIDES} rel="noopener">{t({ en: 'Our BERT reproduction (slides)', zh: '我们的 BERT 复现（幻灯片）' })}</a>
				— {t({ en: '218 fine-tuning runs: GLUE results, pre-training claims, fine-tuning claims.', zh: '218 次微调：GLUE 结果、预训练结论、微调结论。' })}
			</li>
			<li><a href="{base}/glossary/#transfer-learning">{t({ en: 'Glossary: transfer learning, fine-tuning, linear probe, frozen layer', zh: '术语表：迁移学习、微调、线性探针、冻结层' })}</a></li>
		</ul>
		<h3>{t({ en: 'Related concepts (planned)', zh: '相关概念（规划中）' })}</h3>
		<div class="cards">
			{#each furtherConcepts as c (c.slug)}<ConceptCard concept={c} compact />{/each}
		</div>
	{/snippet}
</ConceptPageShell>

<style>
	.flow {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		min-width: 0;
	}
	.hookpanel {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.hookpanel > summary {
		cursor: pointer;
		padding: 10px 14px;
		font-weight: 600;
	}
	.hookpanel[open] > summary {
		border-bottom: 1px solid var(--rule);
	}
	.hookin {
		padding: 12px 14px;
	}
	.try {
		padding: 12px 14px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.try p {
		margin: 0 0 8px;
	}
	.err {
		color: var(--illus);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
</style>
