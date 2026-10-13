<!-- masked-lm 'batch' (meta.ts): every row of the batch as the model would read it after the draw so far. -->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { batchOf } from './meta.ts';

	let P: StageProps = $props();
	const B = $derived(batchOf(P));
</script>

<div class="bt">
	{#each B.lines as l (l.r)}
		<div class="line">
			<span class="lab">{t(l.label)}</span>
			<TokenChips tokens={l.items} selected={l.selected} marked={l.marked} selectable={false} label={l.label} />
		</div>
	{/each}
	{#if B.now}<p class="small now">{t(B.now)}</p>{/if}
	<p class="small">{t(B.tally)} {t({ en: 'Underlined: chosen; dashed: [MASK]; coloured: a random token; outlined: the token drawing now.', zh: '下划线：被选中；虚线框：[MASK]；彩色：随机词元；加框：正在抽签的词元。' })}</p>
	{#if B.big}<p class="small big">{t(B.big)}</p>{/if}
</div>

<style>
	.bt { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; min-width: 0; }
	.line { display: grid; gap: 2px; min-width: 0; }
	.lab { font-size: var(--fs-xs); color: var(--ink-2); }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	.now { color: var(--ink); }
	.big { border-left: 3px solid var(--accent); padding-left: 8px; color: var(--ink); }
</style>
