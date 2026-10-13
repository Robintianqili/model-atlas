<!--
  'positional-encodings/order' (meta.ts): two rows of token chips, the text as typed and shuffled, each chip coloured by
  the last token's layer-0 attention weight on it (one sequential scale for both rows); the last chip, the query, is
  outlined. Under them, how much the last token's attention output changed (and, with props.final, its final hidden state).
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { tokText } from '../../vocab.ts';
	import { orderOf } from './meta.ts';

	let P: StageProps = $props();
	const o = $derived(orderOf(P));
	const scale = $derived(makeScale('sequential', [0, o.max], themeState.resolved));
	$effect(() => void (P.autoplay && P.onend()));
</script>

<div class="order">
	{#each o.rows as r (r.key)}
		<div class="row">
			<span class="lab">{t(r.label)}</span>
			<ol class="chips" aria-label={t(r.label)}>
				{#each r.ids as id, i (i)}
					<li class="chip" class:q={i === r.ids.length - 1} style="background:{scale.css(r.w[i])};color:{scale.ink(r.w[i])}">
						<span class="tok">{tokText(id).replace(/ /g, '·')}</span><span class="w num">{fmtPct(r.w[i], 0)}</span>
					</li>
				{/each}
			</ol>
		</div>
	{/each}
	<p class="note">
		{t({ en: 'Colour and %: the weight the last (outlined) token gives each token, mean of 4 heads, layer 0', zh: '颜色和百分比：最后一个（带框的）词元给每个词元的权重，第 0 层 4 个头的平均' })}{colon()}
		{t(o.pos ? { en: 'TinyGPT as trained', zh: '训练好的 TinyGPT' } : { en: 'position table set to zero', zh: '位置表设为零' })}.
	</p>
	<p class="res">
		{t({ en: 'Largest change of the last token’s attention output', zh: '最后一个词元的注意力输出的最大变化' })}{colon()} <strong class="num">{fmt3(o.d)}</strong>
	</p>
	{#if o.final}
		<p class="res">
			{t({ en: 'Largest change of the last token’s final hidden state (after the last layer)', zh: '最后一个词元的最终隐藏状态（最后一层之后）的最大变化' })}{colon()} <strong class="num">{fmt3(o.dF)}</strong>
		</p>
	{/if}
</div>

<style>
	.order { display: grid; gap: 10px; min-width: 0; }
	.row { display: grid; grid-template-columns: minmax(0, 5.5rem) minmax(0, 1fr); gap: 8px; align-items: start; }
	.lab { font-size: var(--fs-sm); color: var(--ink-2); padding-top: 4px; }
	.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 0; padding: 0; list-style: none; min-width: 0; }
	.chip { display: grid; justify-items: center; padding: 2px 6px; border-radius: 4px; border: 1px solid var(--rule); min-width: 2.2rem; }
	.chip.q { outline: 2px solid var(--ink); outline-offset: 1px; }
	.tok { font-family: var(--font-mono); font-size: var(--fs-sm); white-space: pre; }
	.w { font-size: var(--fs-xs); }
	.note { margin: 0; font-size: var(--fs-xs); color: var(--muted); }
	.res { margin: 0; font-size: var(--fs-sm); }
	@media (max-width: 480px) { .row { grid-template-columns: minmax(0, 1fr); } }
</style>
