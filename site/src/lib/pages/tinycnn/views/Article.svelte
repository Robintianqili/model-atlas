<!--
  TinyCNN's How it works view (redesign spec §5.9), loaded only when the view opens: the article in the essay column, on
  the lesson's input. Its "show me" links select a layer and open Explore at the zoom view (#explore-zoom).
-->
<script lang="ts">
	import { i18n } from '$lib/i18n/i18n.svelte';
	import ArticleEn from '../article/En.svelte';
	import ArticleZh from '../article/Zh.svelte';
	import type { CnnStore } from '../store.svelte';

	let { vp }: { vp: { S: CnnStore }; section?: string } = $props();
</script>

<div class="article prose" lang={i18n.lang === 'zh' ? 'zh-CN' : 'en'}>
	{#if i18n.lang === 'zh'}<ArticleZh S={vp.S} />{:else}<ArticleEn S={vp.S} />{/if}
</div>
