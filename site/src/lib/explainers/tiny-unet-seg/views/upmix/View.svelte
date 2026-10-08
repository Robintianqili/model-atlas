<!-- tiny-unet-seg/upmix (see meta.ts): the old map with the 2×2 window, the new map with the cell, and the 4 products.
     Hover or tap the new map to work out another cell. Nothing animates. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtSetting } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import SpatialMaps from '../../../../atlas/ui/views/kit.parts/SpatialMaps.svelte';
	import { mixOf, unitOf } from './meta.ts';

	let P: StageProps = $props();
	let picked = $state<number[] | null>(null);
	const u = $derived(picked ?? unitOf(P));
	const m = $derived(mixOf(P, u));
	$effect(() => {
		if (P.autoplay && m) P.onend();
	});
</script>

{#if !m}
	<Notice />
{:else}
	<div class="mix">
		<SpatialMaps input={{ data: m.inMap, rows: m.H, cols: m.H, domain: m.dom, label: { en: `Before: map ${m.c} of ${m.from.en}`, zh: `之前：${m.from.zh}的第 ${m.c} 张图` } }}
			output={{ data: m.outMap, rows: m.OH, cols: m.OH, domain: m.dom, label: { en: `After: map ${m.c}, twice the size`, zh: `之后：第 ${m.c} 张图，放大一倍` } }}
			box={m.box} cell={{ row: u[1], col: u[2] }} arrow="×2" onpick={(r, c) => (picked = [m.c, r, c])} />
		<table class="small">
			<thead><tr><th scope="col">{t({ en: 'old cell (row, column)', zh: '旧格子（行，列）' })}</th><th scope="col">{t({ en: 'value', zh: '值' })}</th><th scope="col">{t({ en: 'weight', zh: '权重' })}</th><th scope="col">{t({ en: 'value × weight', zh: '值 × 权重' })}</th></tr></thead>
			<tbody>
				{#each m.cells as q, i (i)}<tr><td class="num">({q.r}, {q.q})</td><td><Num value={q.v} /></td><td class="num">{fmtSetting(q.w)}</td><td><Num value={q.v * q.w} /></td></tr>{/each}
				<tr class="sum"><th scope="row" colspan="3">{t({ en: `new cell (${u[1]}, ${u[2]}) = the sum`, zh: `新格子（${u[1]}, ${u[2]}）= 总和` })}</th><td><Num value={m.y} /></td></tr>
			</tbody>
		</table>
		<p class="small muted">{t({ en: `The new cell’s centre falls between the old cells, ${m.ly} of the way down and ${m.lx} of the way across; each weight is a product of these distances, such as (1 − ${m.ly}) × (1 − ${m.lx}), so the 4 weights add up to 1.`, zh: `新格子的中心落在旧格子之间：向下 ${m.ly}，向右 ${m.lx}；每个权重都是这些距离的乘积，例如 (1 − ${m.ly}) × (1 − ${m.lx})，所以 4 个权重加起来是 1。` })}</p>
	</div>
{/if}

<style>
	.mix { display: grid; gap: 12px; min-width: 0; }
	table { border-collapse: collapse; max-width: 520px; }
	th { text-align: left; font-weight: normal; padding: 2px 12px 2px 0; }
	td { text-align: right; padding: 2px 0 2px 10px; }
	tr.sum th, tr.sum td { font-weight: 700; border-top: 1px solid var(--rule); }
	.small { font-size: var(--fs-sm); }
	p { margin: 0; }
</style>
