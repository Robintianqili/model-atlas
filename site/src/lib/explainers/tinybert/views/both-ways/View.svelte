<!-- tinybert 'both-ways' (meta.ts): TinyBERT's and TinyGPT's block-1 attention on the same text, one head at a time. -->
<script lang="ts">
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { bothOf } from './meta.ts';

	let P: StageProps = $props();
	let pick = $state<number | null>(null);
	const B = $derived(bothOf(P, pick));
</script>

<div class="bw">
	<div class="heads" role="group" aria-label={t({ en: 'Attention head', zh: '注意力头' })}>
		{#each Array.from({ length: B.heads }, (_, h) => h) as h (h)}
			<button type="button" class="btn" aria-pressed={h === B.h} onclick={() => (pick = h)}>{t({ en: `head ${h}`, zh: `头 ${h}` })}</button>
		{/each}
	</div>
	<div class="two">
		{#if B.bert}<section><h4>{t({ en: 'TinyBERT: no mask', zh: 'TinyBERT：没有掩码' })}</h4><AttentionMatrix weights={B.bert} queries={B.bertTokens} row={B.bertRow} label={{ en: 'TinyBERT block 1', zh: 'TinyBERT 第 1 块' }} /></section>{/if}
		{#if B.gpt}<section><h4>{t({ en: 'TinyGPT: causal mask', zh: 'TinyGPT：因果掩码' })}</h4><AttentionMatrix weights={B.gpt} queries={B.gptTokens} causal row={B.gptRow} label={{ en: 'TinyGPT block 1', zh: 'TinyGPT 第 1 块' }} /></section>{/if}
	</div>
	<p class="small">{t(B.caption)}</p>
</div>

<style>
	.bw { display: grid; gap: 10px; min-width: 0; }
	.two { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }
	section { min-width: 0; }
	h4 { margin: 0 0 6px; font-size: var(--fs-sm); }
	.heads { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
