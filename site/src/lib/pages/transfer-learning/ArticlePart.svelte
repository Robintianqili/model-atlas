<!--
  One block of the page's article, in the reader's language. The article is split into parts
  that sit between the widgets, in ConceptPageShell order (hook → toy → real → compare → pitfalls → post).
  English: article/en.svelte (complete). 中文: article/zh.svelte (full translation).
-->
<script lang="ts">
	import { i18n } from '$lib/i18n/i18n.svelte';
	import En from './article/en.svelte';
	import Zh from './article/zh.svelte';

	let { part }: { part: string } = $props();
</script>

<div class="article prose" lang={i18n.lang === 'zh' ? 'zh-CN' : 'en'}>
	{#if i18n.lang === 'zh'}<Zh {part} />{:else}<En {part} />{/if}
</div>

<style>
	.article :global(p) {
		margin: 0 0 0.8em;
	}
	.article :global(p:last-child) {
		margin-bottom: 0;
	}
	.article :global(h3) {
		margin: var(--s4) 0 var(--s2);
		font-size: var(--fs-lg);
	}
	.article :global(ul) {
		margin: 0 0 0.8em;
		padding-left: 1.2em;
	}
</style>
