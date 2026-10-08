<!--
  Long article text per language. Write one plain Svelte component per language and pass both:

    src/lib/content/tinycnn/en.svelte   (English, complete)
    src/lib/content/tinycnn/zh.svelte   (中文; may be missing at first)

    <script> import En from '$lib/content/tinycnn/en.svelte'; import Zh from '$lib/content/tinycnn/zh.svelte'; </script>
    <Article en={En} zh={Zh} />

  If `zh` is missing, Chinese readers see a short notice and the English text.
  Article components can use any kit component (Term, Tex, Num, TensorGrid, …).
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import { i18n, ui } from '$lib/i18n/i18n.svelte';

	let { en: En, zh: Zh }: { en: Component; zh?: Component } = $props();
</script>

<div class="article prose" lang={i18n.lang === 'zh' && Zh ? 'zh-CN' : 'en'}>
	{#if i18n.lang === 'zh' && Zh}
		<Zh />
	{:else}
		{#if i18n.lang === 'zh'}<p class="nt" lang="zh-CN">{ui('notTranslatedYet')}</p>{/if}
		<En />
	{/if}
</div>

<style>
	.nt {
		font-size: var(--fs-sm);
		color: var(--muted);
		border-left: 3px solid var(--rule-strong);
		padding-left: 10px;
	}
</style>
