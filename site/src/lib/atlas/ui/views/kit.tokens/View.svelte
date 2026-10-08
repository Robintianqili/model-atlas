<!--
  kit.tokens: the kit's TokenChips on the learner's tokens (props.tokens, else the adapter's rt.pre.tokens, else the
  ids of the model's token input), with ids shown. The stage's position (unit, sel.t, or props.selected) is selected,
  and the step's focus positions are underlined (e.g. the keys a query attends to). Picking a chip is local.
-->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { tokensOf } from './tokens.ts';

	let P: StageProps = $props();
	const K = $derived(tokensOf(P));
	let sel = $derived.by<number | null>(() => {
		void P.at;
		return K?.selected ?? null;
	});
	$effect(() => {
		if (P.autoplay && K) P.onend();
	});
</script>

{#if !K}
	<Notice text={{ en: 'No tokens to show: this model has no text input.', zh: '没有可显示的词元：这个模型没有文本输入。' }} />
{:else}
	<div class="tk">
		<TokenChips tokens={K.items} selected={sel} showIds={K.hasIds} marked={K.marked} label={{ en: 'Your text as tokens', zh: '你的文本被切成的词元' }} onselect={(i) => (sel = i)} />
		<p class="muted small num">
			{t({ en: `${K.items.length} tokens`, zh: `${K.items.length} 个词元` })}{#if sel !== null && K.items[sel]} · {t({ en: 'selected position', zh: '选中位置' })} {sel}: “{K.items[sel].text}”{#if K.items[sel].id !== undefined} = id {K.items[sel].id}{/if}{/if}
		</p>
	</div>
{/if}

<style>
	.tk {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.small {
		margin: 0;
		font-size: var(--fs-xs);
	}
</style>
