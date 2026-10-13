<!-- masked-lm 'fill' (meta.ts): TinyBERT's input with the chosen tokens; a click hides a token or gives it back; under
     it, TinyBERT's top guesses at each chosen position. -->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { clicked, crowded, fillOf } from './meta.ts';
	import type { MaskedRaw } from '../../adapters/masked-text/adapter.ts';

	let P: StageProps = $props();
	const F = $derived(fillOf(P));
	function set(raw: MaskedRaw | null) {
		if (!raw) return;
		if (P.setInput) P.setInput(raw);
		else P.rt.setInput(raw);
	}
</script>

<div class="fl">
	<TokenChips tokens={F.items} selected={null} marked={F.marked} showIds label={{ en: 'TinyBERT’s input: click a token to hide it or give it back', zh: 'TinyBERT 的输入：点一个词元把它藏起来，或把它还原' }} onselect={(q) => set(clicked(P, q))} />
	{#if F.crowd && F.n > 1}
		<button type="button" class="btn" onclick={() => set(crowded(P.rt.input as MaskedRaw))}>{t({ en: 'Hide two tokens of every three', zh: '每三个词元藏起两个' })}</button>
	{/if}
	<p class="small">{t(F.caption)}</p>
	{#if F.bars && F.gaps.length}
		<div class="gaps">
			{#each F.gaps as g (g.k)}
				<div class="gap">
					{#if g.bars}<ProbBars items={g.bars} k={3} sort={false} tokens ids={false} label={g.head} />{/if}
					<p class="small">{t(g.line)}</p>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.fl { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; min-width: 0; }
	.gaps { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); }
	.gap { display: grid; gap: 4px; min-width: 0; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	.btn { justify-self: start; min-height: 32px; padding: 2px 12px; font-size: var(--fs-sm); }
</style>
