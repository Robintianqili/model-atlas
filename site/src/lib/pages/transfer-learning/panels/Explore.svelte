<!--
  Explore (redesign spec §5.9; a lazy panel of the transfer-learning page, #explore): everything at once.
    #explore-hook     a model that knows only 10 things (live + precomputed)
    #explore-toy      Part 1: freeze the body, train a new head on your own drawings (live; the old #toy)
    #explore-real     Part 2: head only vs full fine-tune vs from scratch (precomputed, finetune.json; #real)
    #explore-compare  Part 3: at real scale, our BERT reproduction (precomputed; #compare)
  The engine and the data files come from the route (vp), so they load once for the lesson and Explore.
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import Article from '../ArticlePart.svelte';
	import HookPanel from '../HookPanel.svelte';
	import HeadLab from '../HeadLab.svelte';
	import CurveChart from '../CurveChart.svelte';
	import BertPanel from '../BertPanel.svelte';
	import { lab } from '../state.svelte';
	import type { Engine } from '../engine';
	import type { FinetuneJson, HeldoutJson, PcaJson } from '../data';

	interface VP {
		engine: Engine | null;
		heldout: HeldoutJson | null;
		pca: PcaJson | null;
		finetune: FinetuneJson;
		tolerance?: number;
		loadError: string | null;
	}
	let { vp }: { vp: VP; section?: string } = $props();
	let hookOpen = $state(false);
</script>

<div class="explore">
	<Section id="explore-hook" title={{ en: 'A model that knows only 10 things', zh: '只认识 10 样东西的模型' }}>
		<div class="flow">
			<Article part="hook-intro" />
			<details class="hookpanel" bind:open={hookOpen}>
				<summary>{t({ en: 'Show it one here: a ladder, an envelope or your own drawing (step 1 of the lesson does the same)', zh: '在这里给它看一张：梯子、信封或你自己的画（课程第 1 步也是这样做的）' })}</summary>
				{#if hookOpen}<div class="hookin"><HookPanel engine={vp.engine} heldout={vp.heldout} finetune={vp.finetune} tolerance={vp.tolerance} /></div>{/if}
			</details>
			<Article part="hook-after" />
			{#if vp.loadError}<p class="err" role="alert">{t({ en: `Could not load the model files: ${vp.loadError}`, zh: `模型文件加载失败：${vp.loadError}` })}</p>{/if}
		</div>
	</Section>

	<Section id="explore-toy" title={{ en: 'Part 1 · Freeze the body, train a new head', zh: '第 1 部分 · 冻结主体，训练新输出头' }} provenance="live">
		<div class="flow">
			<Article part="toy-intro" />
			<HeadLab engine={vp.engine} heldout={vp.heldout} pca={vp.pca} tolerance={vp.tolerance} />
		</div>
	</Section>

	<Section id="explore-real" title={{ en: 'Part 2 · Head only vs full fine-tune vs from scratch', zh: '第 2 部分 · 只训练输出头 vs 全量微调 vs 从零训练' }} provenance="precomputed">
		<div class="flow" id="tl-curves">
			<Article part="real-intro" />
			<CurveChart finetune={vp.finetune} />
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
	</Section>

	<Section id="explore-compare" title={{ en: 'Part 3 · At real scale: our BERT reproduction', zh: '第 3 部分 · 真实规模：我们的 BERT 复现' }} provenance="precomputed">
		<BertPanel />
	</Section>
</div>

<style>
	.explore {
		display: grid;
		gap: var(--s7);
	}
	.flow {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		min-width: 0;
	}
	.hookpanel {
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
	}
	.hookpanel > summary {
		cursor: pointer;
		padding: 10px 14px;
		font-weight: var(--w-ui);
	}
	.hookpanel[open] > summary {
		border-bottom: 1px solid var(--rule);
	}
	.hookin {
		padding: 12px 14px;
	}
	.try {
		padding: 12px 14px;
		border-radius: var(--r3);
		background: var(--well);
	}
	.try p {
		margin: 0 0 8px;
	}
	.cluster {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.err {
		color: var(--no);
	}
</style>
