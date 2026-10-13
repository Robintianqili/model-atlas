<!--
  Pitfalls (redesign spec §5.9; a lazy panel of the transfer-learning page, #pitfalls): two misconception checks with
  their evidence (each "Show me" opens Explore's Part 2 at that setting) and other pitfalls (#pitfalls-list), then the
  cover question revealed and the checkpoint (#pitfalls-post, the old #post and #checkpoint).
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import { t } from '$lib/i18n/i18n.svelte';
	import Article from '../ArticlePart.svelte';
	import Misconception from '../Misconception.svelte';
	import Checkpoint from '../Checkpoint.svelte';
	import { lab } from '../state.svelte';

	let { vp }: { vp: { prequestion: PredictQuestion }; section?: string } = $props();

	/** set Part 2's chart, then open Explore at Part 2 */
	function showPart2(n: number, seeds = false) {
		lab.part2N = n;
		lab.part2Zoom = false;
		if (seeds) lab.part2Seeds = true;
		location.hash = 'explore-real';
	}
</script>

<div class="pits">
	<Section id="pitfalls-list" title={{ en: 'Misconceptions and pitfalls', zh: '误区与陷阱' }}>
		<div class="flow">
			<Misconception
				statement={{ en: 'Fine-tuning everything is always better than freezing.', zh: '微调所有层总是比冻结更好。' }}
				truth={false}
				show={{ label: { en: 'Show me: 1 drawing per class in Explore, Part 2', zh: '带我看：“探索”第 2 部分中每类 1 张的情况' }, action: () => showPart2(1) }}
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
	</Section>
	<Section id="pitfalls-post" title={{ en: 'Checkpoint', zh: '检查点' }}>
		<div class="flow" id="checkpoint">
			<PredictReveal q={vp.prequestion} />
			<Checkpoint />
			<Article part="post" />
		</div>
	</Section>
</div>

<style>
	.pits {
		display: grid;
		gap: var(--s7);
	}
	.flow {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		min-width: 0;
	}
</style>
