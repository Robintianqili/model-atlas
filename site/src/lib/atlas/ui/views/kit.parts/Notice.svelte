<!--
  What a view shows when it has nothing to draw: the input is empty, the source has not run yet, or the step points
  at something the trace does not have (the message names it, so a lesson author can fix the step).
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		error?: string;
		text?: Text;
	}
	let { error, text }: Props = $props();
</script>

<p class="nt">
	{t(text ?? { en: 'Nothing to show here yet: run the model on an input first.', zh: '这里还没有内容：请先在一个输入上运行模型。' })}
	{#if error}<span class="err num">({error})</span>{/if}
</p>

<style>
	.nt {
		margin: 0;
		padding: 12px 14px;
		border: 1px dashed var(--rule-strong);
		border-radius: var(--r2);
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.err {
		display: block;
		margin-top: 4px;
		font-size: var(--fs-xs);
		color: var(--muted);
		overflow-wrap: anywhere;
	}
</style>
