<!--
  'lora/memory': three stacked bars on one MB axis from 0 (meta.ts). Segment colors are the categorical palette (cat-0
  weights, cat-1 gradients, cat-2 optimizer state), named in the legend; each bar's total is printed at its end.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { MB, memoryOf, rankAt } from '../../data.ts';
	import { PARTS, PART_LABEL, ROWS } from './meta.ts';

	let { rt, at }: StageProps = $props();
	const r = $derived(rankAt({ rt, at }));
	const m = $derived(memoryOf(r));
	const [W, ML, MR, RH] = [360, 4, 4, 44];
	const max = $derived(Math.ceil(MB(m.full.total) / 500) * 500);
	const x = (b: number) => ML + (MB(b) / max) * (W - ML - MR);
	const H = ROWS.length * RH + 30;
	const ticks = $derived(Array.from({ length: max / 500 + 1 }, (_, i) => i * 500));
	const starts = (id: 'full' | 'lora' | 'qlora') => {
		let acc = 0;
		return PARTS.map((p) => {
			const v = m[id][p];
			const s = acc;
			acc += v;
			return { p, s, v };
		});
	};
</script>

<div class="wrap">
	<figure>
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Training memory of GPT-2 small, in MB', zh: 'GPT-2 small 的训练内存（MB）' })}>
			{#each ticks as v (v)}
				<line class="grid" x1={x(v * 1e6)} x2={x(v * 1e6)} y1="0" y2={ROWS.length * RH} />
				<text class="tick" x={x(v * 1e6)} y={ROWS.length * RH + 12} text-anchor={v === 0 ? 'start' : v === max ? 'end' : 'middle'}>{fmtInt(v)}</text>
			{/each}
			{#each ROWS as row, i (row.id)}
				<text class="name" x={ML} y={i * RH + 12}>{t(row.label)} · <tspan class="num">{fmtInt(Math.round(MB(m[row.id].total)))} MB</tspan></text>
				{#each starts(row.id) as seg (seg.p)}
					{#if seg.v > 0}
						<rect class="seg {seg.p}" x={x(seg.s)} y={i * RH + 18} width={Math.max(1, x(seg.s + seg.v) - x(seg.s))} height="16" />
					{/if}
				{/each}
			{/each}
			<text class="axis" x={W / 2} y={H - 2} text-anchor="middle">{t({ en: 'memory (MB, millions of bytes)', zh: '内存（MB，百万字节）' })}</text>
		</svg>
		<figcaption>
			{#each PARTS as p (p)}<span class="sw {p}"></span>{t(PART_LABEL[p])}{/each}
		</figcaption>
	</figure>
	<p class="note">
		{t({
			en: `GPT-2 small at rank ${r}, every number 4 bytes, Adam; activations not counted. LoRA adapts the six matrices of each block; QLoRA keeps the frozen weights at 4 bits each (its small per-block scales left out).`,
			zh: `GPT-2 small，秩 ${r}，每个数 4 字节，优化器为 Adam；不计激活值。LoRA 适配每个块的六个矩阵；QLoRA 把冻结的权重存成每个 4 比特（不计它每块的小缩放常数）。`
		})}
	</p>
</div>

<style>
	.wrap {
		max-width: 560px;
	}
	figure {
		margin: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 10px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axis,
	.name {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.name .num {
		font-family: var(--font-mono);
	}
	.seg.weights,
	.sw.weights {
		fill: var(--cat-0);
		background: var(--cat-0);
	}
	.seg.grads,
	.sw.grads {
		fill: var(--cat-1);
		background: var(--cat-1);
	}
	.seg.optimizer,
	.sw.optimizer {
		fill: var(--cat-2);
		background: var(--cat-2);
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
	}
	.note {
		margin: 6px 0 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
