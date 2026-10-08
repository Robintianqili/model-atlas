<!--
  Lesson step 8 (precomputed): the same choices at BERT scale, from our BERT reproduction.
  (a) a linear probe on frozen BERT features vs full fine-tuning (SST-2, MRPC; research/bert_facts.json)
  (b) MNLI first, then RTE vs RTE directly, 5 seeds (research/bert_improve.json)
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { t, i18n, colon } from '$lib/i18n/i18n.svelte';
	import MnliRte from './MnliRte.svelte';
	import { linear, markerPath } from './chart';
	import { FEATURES_VS_FT, GLUE_TRAIN } from './bert';

	const cFrozen = $derived(categorical(2, themeState.resolved));
	const cFull = $derived(categorical(6, themeState.resolved));
	const rows = FEATURES_VS_FT.rows.filter((r) => r.ours);
	const train: Record<string, number> = { sst2: GLUE_TRAIN.SST2, mrpc: GLUE_TRAIN.MRPC };
	let w = $state(480);
	const M = { l: 8, r: 46, t: 6, b: 22 };
	const rowH = 40;
	const H = rows.length * rowH + M.t + M.b;
	const x = $derived(linear([70, 100], [M.l, w - M.r]));
</script>

<div class="lb">
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'Frozen BERT features + linear probe vs full fine-tuning (dev accuracy %)', zh: '冻结的 BERT 特征 + 线性探针 vs 全量微调（验证集正确率 %）' })}</span>
			<ProvenanceBadge mode="precomputed" compact note={{ en: 'From our BERT reproduction (research/bert_facts.json).', zh: '来自我们的 BERT 复现（research/bert_facts.json）。' }} />
		</figcaption>
		<ul class="legend">
			<li><svg width="14" height="12" aria-hidden="true"><path d={markerPath('circle', 4.5)} transform="translate(7,6)" fill={cFrozen} /></svg>{t({ en: 'frozen features (best probe)', zh: '冻结特征（最佳探针）' })}</li>
			<li><svg width="14" height="12" aria-hidden="true"><path d={markerPath('square', 4.5)} transform="translate(7,6)" fill={cFull} /></svg>{t({ en: 'full fine-tuning', zh: '全量微调' })}</li>
		</ul>
		<div bind:clientWidth={w}>
			<svg viewBox="0 0 {w} {H}" width={w} height={H} role="img" aria-label={rows.map((r) => `${t(r.task)}${colon()}${i18n.lang === 'zh' ? '' : ' '}${r.frozen.toFixed(1)} → ${r.ft.toFixed(1)}`).join(i18n.lang === 'zh' ? '；' : '; ')}>
				{#each [70, 80, 90, 100] as v (v)}
					<line class="grid" x1={x(v)} x2={x(v)} y1={M.t} y2={H - M.b} />
					<text class="tick" x={x(v)} y={H - 6} text-anchor="middle">{v}</text>
				{/each}
				{#each rows as r, i (r.id)}
					{@const yy = M.t + i * rowH + 26}
					<text class="rl" x={M.l} y={yy - 11}>{r.id === 'sst2' ? 'SST-2' : 'MRPC'} <tspan class="muted-t">· {fmtInt(train[r.id])} {t({ en: 'training examples', zh: '个训练样本' })}</tspan></text>
					<line class="conn" x1={x(r.frozen)} x2={x(r.ft)} y1={yy} y2={yy} />
					<path class="mk" d={markerPath('circle', 5)} transform="translate({x(r.frozen)},{yy})" fill={cFrozen} />
					<path class="mk" d={markerPath('square', 5)} transform="translate({x(r.ft)},{yy})" fill={cFull} />
					<text class="val" x={x(r.frozen) - 8} y={yy + 4} text-anchor="end">{r.frozen.toFixed(1)}</text>
					<text class="val strong" x={x(r.ft) + 9} y={yy + 4}>{r.ft.toFixed(1)}</text>
				{/each}
			</svg>
		</div>
		<p class="cap">{t({ en: `Fine-tuning trains all 110 million parameters on every training example. The probe only trains a small head on frozen features: for SST-2 on a ${fmtInt(FEATURES_VS_FT.probeSubsample)}-sentence sample, for MRPC on all ${fmtInt(GLUE_TRAIN.MRPC)} pairs. Each probe score is the best of ${FEATURES_VS_FT.combos} layer and pooling choices, picked on the same dev set. SST-2: is a movie review positive? MRPC: do two sentences mean the same?`, zh: `微调用全部训练样本训练所有 1.1 亿个参数。探针只在冻结特征上训练一个小输出头：SST-2 用抽样的 ${fmtInt(FEATURES_VS_FT.probeSubsample)} 个句子，MRPC 用全部 ${fmtInt(GLUE_TRAIN.MRPC)} 对句子。每个探针分数都是 ${FEATURES_VS_FT.combos} 种层与池化组合中最好的一个，并且就是在同一个验证集上挑出来的。SST-2：一条影评是不是正面的？MRPC：两个句子意思是否相同？` })}</p>
	</figure>
	<MnliRte compact />
</div>

<style>
	.lb {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.fig {
		margin: 0;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 4px;
		min-width: 0;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 6px 12px;
		align-items: center;
		font-weight: 600;
		font-size: var(--fs-sm);
	}
	.legend {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	svg {
		display: block;
		max-width: 100%;
		height: auto;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 11px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.rl {
		font-size: 12px;
		fill: var(--ink);
		font-weight: 600;
	}
	.muted-t {
		fill: var(--muted);
		font-weight: 400;
	}
	.conn {
		stroke: var(--rule-strong);
		stroke-width: 3;
	}
	.mk {
		stroke: var(--surface);
		stroke-width: 2;
		paint-order: stroke;
	}
	.val {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.val.strong {
		fill: var(--ink);
		font-weight: 700;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
