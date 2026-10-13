<!--
  'attention-variants/attn' (meta.ts): heat tables of the last token's attention weights (rows: query heads with the KV
  head they read; columns: the probe's tokens), one sequential purple ramp on 0..1 for both tables; then one head's
  scores and weights as a table. Buttons pick the head. Both tables scroll sideways on a phone (keyScroll).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { attnOf, LAST, TOKENS } from './meta.ts';

	let P: StageProps = $props();
	const a = $derived(attnOf(P));
	let picked = $state<number | null>(null);
	const head = $derived(picked ?? Number(P.props?.head ?? 0));
	const scale = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const g = $derived(4 / a.kv);
	$effect(() => void (P.autoplay && P.onend()));
	const panels = $derived(
		a.shared
			? [
					{ id: 'mha', title: { en: 'As trained: 4 key/value heads (MHA)', zh: '训练好的：4 个键值头（MHA）' }, w: a.w0, kv: 4 },
					{ id: 'shared', title: { en: `Shared: ${a.kv} key/value head${a.kv === 1 ? '' : 's'}`, zh: `共用：${a.kv} 个键值头` }, w: a.w, kv: a.kv }
				]
			: [{ id: 'mha', title: { en: 'As trained: 4 key/value heads (MHA)', zh: '训练好的：4 个键值头（MHA）' }, w: a.w, kv: 4 }]
	);
</script>

<div class="attn">
	{#each panels as p (p.id)}
		<div class="panel">
			<h4>{t(p.title)}</h4>
			<div class="scroll" use:keyScroll>
				<div class="grid" role="table" aria-label={t(p.title)} style="grid-template-columns: 5.5rem repeat({TOKENS.length}, minmax(2.4rem, 1fr))">
					<div role="row" class="row">
						<span role="columnheader" class="corner">{t({ en: 'head → KV', zh: '头 → KV' })}</span>
						{#each TOKENS as tok, j (j)}<span role="columnheader" class="tok">{tok}</span>{/each}
					</div>
					{#each p.w as r, h (h)}
						<div role="row" class="row" class:sel={h === head}>
							<span role="rowheader" class="rh">{h} → {Math.floor(h / (4 / p.kv))}</span>
							{#each r as v, j (j)}<span role="cell" class="cell num" style="background:{scale.css(v)};color:{scale.ink(v)}">{Math.round(v * 100)}</span>{/each}
						</div>
					{/each}
				</div>
			</div>
		</div>
	{/each}
	<p class="note">{t({ en: `Cells: attention weight in % (0 to 100, stronger color = more) of the last token, “${LAST}”, in block ${a.layer} (counting from 0). Rows: query head → the key/value head it reads.`, zh: `格子：编号 ${a.layer} 的块（从 0 数起）中最后一个词元“${LAST}”的注意力权重，单位为 %（0 到 100，颜色越强越大）。行：查询头 → 它读的键值头。` })}</p>
	<div class="pick" role="group" aria-label={t({ en: 'Pick a head', zh: '选一个头' })}>
		{#each [0, 1, 2, 3] as h (h)}
			<button type="button" class="btn" aria-pressed={h === head} onclick={() => (picked = h)}>{t({ en: `Head ${h}`, zh: `头 ${h}` })}</button>
		{/each}
	</div>
	<div class="scroll" use:keyScroll>
		<div class="calc" style="grid-template-columns: repeat({a.shared ? 5 : 3}, auto)" role="table" aria-label={t({ en: `Head ${head}: score and weight per position`, zh: `头 ${head}：每个位置的分数和权重` })}>
			<div role="row" class="crow head">
				<span role="columnheader">{t({ en: 'position', zh: '位置' })}</span>
				<span role="columnheader">{t({ en: 'score q·k/√19', zh: '分数 q·k/√19' })}</span>
				<span role="columnheader">{t({ en: 'weight', zh: '权重' })}</span>
				{#if a.shared}
					<span role="columnheader">{t({ en: 'score, shared k̄', zh: '分数（共用 k̄）' })}</span>
					<span role="columnheader">{t({ en: 'weight, shared', zh: '权重（共用）' })}</span>
				{/if}
			</div>
			{#each TOKENS as tok, j (j)}
				<div role="row" class="crow">
					<span role="rowheader">{j} {tok}</span>
					<span role="cell" class="num">{fmt3((a.shared ? a.s0 : a.s)[head]?.[j])}</span>
					<span role="cell" class="num">{fmtPct((a.shared ? a.w0 : a.w)[head]?.[j] ?? NaN)}</span>
					{#if a.shared}
						<span role="cell" class="num">{fmt3(a.s[head]?.[j])}</span>
						<span role="cell" class="num">{fmtPct(a.w[head]?.[j] ?? NaN)}</span>
					{/if}
				</div>
			{/each}
		</div>
	</div>
	{#if a.shared}
		<p class="note">{t({ en: `k̄: the key of KV head ${Math.floor(head / g)}, the mean of the keys heads ${Math.floor(head / g) * g} to ${Math.floor(head / g) * g + g - 1} had learned. The query is still head ${head}’s own.`, zh: `k̄：KV 头 ${Math.floor(head / g)} 的键，即头 ${Math.floor(head / g) * g} 到 ${Math.floor(head / g) * g + g - 1} 原先学到的键的平均。查询仍是头 ${head} 自己的。` })}</p>
	{/if}
</div>

<style>
	.attn {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		max-width: 640px;
		min-width: 0;
	}
	h4 {
		margin: 0 0 4px;
		font-size: var(--fs-sm);
	}
	.scroll {
		overflow-x: auto;
		min-width: 0;
	}
	.grid {
		display: grid;
		gap: 2px;
		min-width: 26rem;
	}
	.row,
	.crow {
		display: contents;
	}
	.corner,
	.tok,
	.rh {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.tok {
		font-family: var(--font-mono);
		text-align: center;
		white-space: nowrap;
	}
	.sel .rh {
		font-weight: 700;
		color: var(--ink);
	}
	.cell {
		text-align: center;
		font-size: var(--fs-xs);
		padding: 3px 0;
		border-radius: 2px;
	}
	.sel .cell {
		outline: 1.5px solid var(--ink);
	}
	.calc {
		display: grid;
		gap: 2px 14px;
		font-size: var(--fs-xs);
		justify-content: start;
	}
	.crow.head span {
		color: var(--ink-2);
		font-weight: 600;
	}
	.num {
		font-family: var(--font-mono);
	}
	.pick {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.pick .btn {
		font-size: var(--fs-xs);
	}
	.pick .btn[aria-pressed='true'] {
		border-color: var(--accent);
		font-weight: 600;
	}
	.note {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
