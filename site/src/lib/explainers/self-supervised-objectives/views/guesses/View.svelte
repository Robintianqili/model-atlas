<!-- self-supervised-objectives 'guesses' (meta.ts): TinyGPT's guess for the hidden token from the left ('clm'), or
     TinyBERT's from both sides next to it ('mlm', with the clickable token row). -->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Chips from '../chips/View.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { guessesOf } from './meta.ts';

	let P: StageProps = $props();
	const G = $derived(guessesOf(P));
</script>

<div class="gs" class:two={G.mlm}>
	<div class="full"><Chips {...P} props={{ mode: G.mlm ? 'pick' : 'clm' }} /></div>
	{#if G.bert}
		<ProbBars items={G.bert} k={G.k} tokens ids={false} label={{ en: 'TinyBERT, reading both sides of the gap', zh: 'TinyBERT，读空缺的两侧' }} />
	{/if}
	{#if G.gpt}
		<ProbBars items={G.gpt} k={G.k} tokens ids={false} label={{ en: 'TinyGPT, reading only the tokens before the gap', zh: 'TinyGPT，只读空缺前面的词元' }} />
	{:else if G.hidden}
		<p class="small">{t(G.first ? { en: 'TinyGPT cannot guess the first token: nothing comes before it.', zh: 'TinyGPT 猜不了第一个词元：它前面什么都没有。' } : { en: 'TinyGPT: nothing to show yet.', zh: 'TinyGPT：还没有可显示的内容。' })}</p>
	{/if}
	{#if G.hidden}<p class="small full">{t({ en: `The bar in the accent colour is the hidden token, “${G.hidden}”, when it is among the top ${G.k}.`, zh: `强调色的条就是被藏的“${G.hidden}”（如果它排在前 ${G.k} 名）。` })} {G.lines.map((l) => t(l)).join(' ')}</p>{/if}
</div>

<style>
	.gs { display: grid; gap: 12px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.gs.two { grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); }
	.full { grid-column: 1 / -1; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
