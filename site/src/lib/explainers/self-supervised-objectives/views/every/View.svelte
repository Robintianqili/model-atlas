<!-- self-supervised-objectives 'every' (meta.ts): TinyGPT's probability for every real next token of the text, in order. -->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { everyOf } from './meta.ts';

	let P: StageProps = $props();
	const E = $derived(everyOf(P));
</script>

<div class="ev">
	{#if E.n}
		<ProbBars items={E.items} k={E.items.length} sort={false} tokens ids={false} label={{ en: 'TinyGPT’s probability for the token that really comes next (token number: token)', zh: 'TinyGPT 给真正的下一个词元的概率（词元编号：词元）' }} />
	{/if}
	<p class="small">{t(E.caption)}</p>
</div>

<style>
	.ev { display: grid; gap: 10px; min-width: 0; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
