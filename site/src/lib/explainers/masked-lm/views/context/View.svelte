<!-- masked-lm 'context' (meta.ts): what TinyBERT and TinyGPT read for the same token, and their guesses. -->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { contextOf } from './meta.ts';

	let P: StageProps = $props();
	const C = $derived(contextOf(P));
</script>

{#if C}
	<div class="cx">
		{#each [{ s: C.bert, name: { en: 'Masked: TinyBERT', zh: '掩码：TinyBERT' } }, { s: C.gpt, name: { en: 'Causal: TinyGPT', zh: '因果：TinyGPT' } }] as side (side.name.en)}
			<section class="side">
				<h4>{t(side.name)}</h4>
				<TokenChips tokens={side.s.items} selected={side.s.selected} marked={side.s.marked} selectable={false} label={side.name} />
				<p class="small">{t(side.s.reads)}</p>
				{#if side.s.bars}<ProbBars items={side.s.bars} k={5} sort={false} tokens ids={false} label={{ en: 'Top guesses for the outlined token', zh: '对加框词元的前几名猜测' }} />{/if}
				<p class="small">{t(side.s.line)}</p>
			</section>
		{/each}
	</div>
{:else}
	<p class="small">{t({ en: 'No token is chosen yet: click one in the input to hide it.', zh: '还没有选中词元：在输入框里点一个词元把它藏起来。' })}</p>
{/if}

<style>
	.cx { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); min-width: 0; }
	.side { display: grid; gap: 6px; min-width: 0; align-content: start; }
	h4 { margin: 0; font-size: var(--fs-md); }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
