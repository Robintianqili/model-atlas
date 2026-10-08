<!--
  Where the 364,040 parameters are: counted from the real tensor table in spec.json, next to the
  formula for each group (d = 76, V = 1024, C = 64, 4 blocks).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt, fmtPct } from '$lib/kit/format';
	import type { TinyGPTSpec } from '../../models/tinygpt';
	import { paramBreakdown } from './analysis';

	let { spec }: { spec: TinyGPTSpec } = $props();
	const pb = $derived(paramBreakdown(spec));
	const blockTotal = $derived(pb.rows.filter((r) => r.times > 1).reduce((a, r) => a + r.count, 0));
</script>

<div class="pt">
	<table>
		<caption>
			{t({ en: 'Parameters, counted from the shipped weights', zh: '参数量（按发布的权重逐个统计）' })}
			<span class="muted">d = {spec.dModel}, V = {spec.vocabSize}, C = {spec.context}, {spec.nLayer} {t({ en: 'blocks', zh: '个块' })}</span>
		</caption>
		<thead>
			<tr>
				<th scope="col">{t({ en: 'Part', zh: '部分' })}</th>
				<th scope="col">{t({ en: 'Formula', zh: '公式' })}</th>
				<th scope="col" class="r">{t({ en: 'Count', zh: '数量' })}</th>
				<th scope="col" class="r">× </th>
				<th scope="col" class="r">{t({ en: 'Total', zh: '合计' })}</th>
				<th scope="col" class="r">%</th>
			</tr>
		</thead>
		<tbody>
			{#each pb.rows as r (r.key)}
				<tr class:inblock={r.times > 1}>
					<td>{t(r.label)}</td>
					<td class="num f">{r.formula}</td>
					<td class="num r">{fmtInt(r.count)}</td>
					<td class="num r">{r.times > 1 ? `×${r.times}` : ''}</td>
					<td class="num r">{fmtInt(r.count * r.times)}</td>
					<td class="num r">{fmtPct((r.count * r.times) / pb.total)}</td>
				</tr>
			{/each}
		</tbody>
		<tfoot>
			<tr>
				<th scope="row">{t({ en: 'One block', zh: '一个块' })}</th>
				<td class="num f">12d² + 13d</td>
				<td class="num r">{fmtInt(blockTotal)}</td>
				<td class="num r">×{spec.nLayer}</td>
				<td class="num r">{fmtInt(blockTotal * spec.nLayer)}</td>
				<td class="num r">{fmtPct((blockTotal * spec.nLayer) / pb.total)}</td>
			</tr>
			<tr class="grand">
				<th scope="row">{t({ en: 'Total', zh: '总计' })}</th>
				<td class="num f">Vd + Cd + L(12d² + 13d) + 2d</td>
				<td></td>
				<td></td>
				<td class="num r">{fmtInt(pb.total)}</td>
				<td class="num r">100%</td>
			</tr>
		</tfoot>
	</table>
	<p class="note">
		{t({
			en: `The formula gives ${fmtInt(pb.formulaTotal)} and the shipped weights hold ${fmtInt(pb.total)} numbers${pb.formulaTotal === pb.total ? ' — they match' : ''}. At 4 bytes each (fp32) that is ${fmtInt(pb.total * 4)} bytes. About a fifth of the model is the token table E; because the output head reuses E ("tied"), predicting a token costs no extra parameters.`,
			zh: `公式算出 ${fmtInt(pb.formulaTotal)}，发布的权重里有 ${fmtInt(pb.total)} 个数${pb.formulaTotal === pb.total ? '——两者一致' : ''}。每个数 4 字节（fp32），共 ${fmtInt(pb.total * 4)} 字节。词元表 E 约占模型的五分之一；由于输出头复用 E（“权重共享”），预测词元不需要额外的参数。`
		})}
	</p>
</div>

<style>
	.pt {
		overflow-x: auto;
		max-width: 100%;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		min-width: 560px;
	}
	caption {
		text-align: left;
		font-weight: 650;
		margin-bottom: 6px;
	}
	caption .muted {
		font-weight: 400;
		font-size: var(--fs-xs);
		margin-left: 8px;
	}
	th,
	td {
		padding: 4px 10px;
		border-bottom: 1px solid var(--rule);
		text-align: left;
	}
	.r {
		text-align: right;
	}
	.f {
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	tr.inblock td:first-child {
		padding-left: 22px;
	}
	tfoot th,
	tfoot td {
		border-top: 1.5px solid var(--ink-2);
		font-weight: 650;
	}
	.grand th,
	.grand td {
		background: var(--accent-soft);
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>
