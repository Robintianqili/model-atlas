<!--
  'positional-encodings/table' (meta.ts): the kit's TensorGrid heat map of a position table (rows = positions, columns =
  dimensions; one diverging scale), row p outlined and row p + k dashed; for TinyGPT's learned table, a strip of the
  cosine similarity of row p with every row (−1 to 1).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { tableOf } from './meta.ts';

	let P: StageProps = $props();
	const o = $derived(tableOf(P));
	const hl = $derived(
		[
			{ row: o.p, col: 0, w: o.cols, style: 'solid' as const },
			{ row: o.p + o.k, col: 0, w: o.cols, style: 'dashed' as const }
		].filter((h) => h.row < o.rows)
	);
	$effect(() => void (P.autoplay && P.onend()));
</script>

{#if o.rows}
	<div class="tbl">
		<p class="cap">
			{t(o.learned ? { en: `TinyGPT’s table: ${o.rows} positions (rows) × ${o.cols} numbers`, zh: `TinyGPT 的位置表：${o.rows} 个位置（行）× ${o.cols} 个数` } : { en: `Sinusoidal table: ${o.rows} positions (rows) × ${o.cols} numbers`, zh: `正弦位置表：${o.rows} 个位置（行）× ${o.cols} 个数` })}
			· {t({ en: `row p = ${o.p} solid, row p + k = ${o.p + o.k} dashed`, zh: `第 p = ${o.p} 行实线框，第 p + k = ${o.p + o.k} 行虚线框` })}
		</p>
		<TensorGrid data={o.data} rows={o.rows} cols={o.cols} kind="diverging" domain={[-o.amax, o.amax]} maxCell={7} minCell={2} gridLines={false} labels={false} highlights={hl} label={o.learned ? { en: 'TinyGPT’s position table', zh: 'TinyGPT 的位置表' } : { en: 'Sinusoidal position table', zh: '正弦位置表' }} />
		{#if o.learned && o.sim.some(Number.isFinite)}
			<p class="cap">{t({ en: `Cosine similarity of row ${o.p} with rows 0 to ${o.rows - 1}`, zh: `第 ${o.p} 行与第 0 到 ${o.rows - 1} 行的余弦相似度` })}</p>
			<TensorGrid data={o.sim} rows={1} cols={o.sim.length} kind="diverging" domain={[-1, 1]} maxCell={9} minCell={3} labels={false} label={{ en: `cosine similarity with row ${o.p}`, zh: `与第 ${o.p} 行的余弦相似度` }} />
		{/if}
	</div>
{/if}

<style>
	.tbl { display: grid; gap: 6px; min-width: 0; }
	.cap { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
</style>
