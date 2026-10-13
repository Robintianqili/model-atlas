<!--
  'lora/counter': the toy layer drawn to scale (one square per number: W0 frozen and gray, B and A trainable), then the
  counts for real layer sizes at the same rank (meta.ts).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { fmtInt } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { D, R_MAX, pctText, rankAt } from '../../data.ts';
	import { countRows } from './meta.ts';

	let { rt, at }: StageProps = $props();
	const r = $derived(rankAt({ rt, at }));
	const rows = $derived(countRows(r));
	const C = 7; // px per number
	const side = D * C;
	const bx = side + 34;
	const ax = $derived(bx + r * C + 16);
	const W = side + 34 + R_MAX * C + 16 + side;
	const H = side + 24;
	const cells = (rowsN: number, colsN: number) => Array.from({ length: rowsN * colsN }, (_, i) => [Math.floor(i / colsN), i % colsN]);
	const share = (v: unknown) => (typeof v === 'string' ? v : '…');
</script>

<figure class="pic">
	<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: `W0 has ${D * D} frozen numbers; B and A together have ${2 * D * r} trainable numbers`, zh: `W0 有 ${D * D} 个冻结的数；B 和 A 一共有 ${2 * D * r} 个可训练的数` })}>
		<text class="lbl" x="0" y="12">W0 · {D * D}</text>
		<g transform="translate(0 20)">
			{#each cells(D, D) as [i, j] (i * D + j)}<rect class="frozen" x={j * C} y={i * C} width={C - 1} height={C - 1} />{/each}
		</g>
		<text class="lbl" x={bx} y="12">B · {D * r}</text>
		<g transform="translate({bx} 20)">
			{#each cells(D, r) as [i, j] (i * r + j)}<rect class="train" x={j * C} y={i * C} width={C - 1} height={C - 1} />{/each}
		</g>
		<text class="lbl" x={ax} y={20 + r * C + 14}>A · {D * r}</text>
		<g transform="translate({ax} 20)">
			{#each cells(r, D) as [i, j] (i * D + j)}<rect class="train" x={j * C} y={i * C} width={C - 1} height={C - 1} />{/each}
		</g>
	</svg>
	<figcaption>
		<span class="sw frozen"></span>{t({ en: 'frozen', zh: '冻结' })}
		<span class="sw train"></span>{t({ en: 'trainable', zh: '可训练' })}
		<span class="dim">· {t({ en: 'one square per number', zh: '每个方块是一个数' })}</span>
	</figcaption>
</figure>

<div class="scroll" use:keyScroll>
	<table>
		<caption>{t({ en: `Trainable numbers at rank r = ${r}`, zh: `秩 r = ${r} 时可训练的数` })}</caption>
		<thead>
			<tr>
				<th scope="col">{t({ en: 'Layer (d_out × d_in)', zh: '层（d_out × d_in）' })}</th>
				<th scope="col" class="n">{t({ en: 'Full fine-tuning', zh: '全量微调' })}</th>
				<th scope="col" class="n">LoRA</th>
				<th scope="col" class="n">{t({ en: 'Share', zh: '占比' })}</th>
			</tr>
		</thead>
		<tbody>
			{#each rows as row (row.id)}
				<tr class:toy={row.id === 'toy'}>
					<th scope="row">{t(row.label)}{#if row.shape}<span class="dim num shape">{row.shape}</span>{/if}</th>
					<td class="n num">{fmtInt(row.full)}</td>
					<td class="n num">{fmtInt(row.lora)}</td>
					<td class="n num">
						{share(pctText(row.share))}
						<span class="bar" aria-hidden="true"><i style:width="{Math.min(1, row.share) * 100}%"></i></span>
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.pic {
		margin: 0 0 10px;
		max-width: 420px;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.lbl {
		font-size: 11px;
		font-family: var(--font-mono);
		fill: var(--ink-2);
	}
	rect.frozen {
		fill: var(--surface-3);
		stroke: var(--rule-strong);
		stroke-width: 0.5;
	}
	rect.train {
		fill: var(--param);
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
	.sw.frozen {
		background: var(--surface-3);
		border: 1px solid var(--rule-strong);
	}
	.sw.train {
		background: var(--param);
	}
	.dim {
		color: var(--muted);
	}
	.shape {
		display: block;
		font-size: var(--fs-xs);
	}
	.scroll {
		overflow-x: auto;
		max-width: 100%;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		min-width: 100%;
	}
	caption {
		text-align: left;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		padding-bottom: 4px;
	}
	th,
	td {
		padding: 4px 6px;
		border-bottom: 1px solid var(--rule);
		vertical-align: top;
	}
	thead th {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 600;
		text-align: left;
	}
	tbody th {
		font-weight: 400;
		text-align: left;
	}
	.n {
		text-align: right;
		white-space: nowrap;
	}
	tr.toy th {
		font-weight: 600;
	}
	.bar {
		display: block;
		height: 4px;
		margin-top: 3px;
		background: var(--surface-3);
		min-width: 3rem;
	}
	.bar i {
		display: block;
		height: 100%;
		min-width: 1px;
		background: var(--param);
	}
</style>
