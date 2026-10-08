<!--
  Inline bilingual block with markup (for short passages; use t({en, zh}) for plain strings
  and <Article> for long texts):
    <Lang>
      {#snippet en()}A <Term id="kernel" /> slides over the image.{/snippet}
      {#snippet zh()}<Term id="kernel" />在图像上滑动。{/snippet}
    </Lang>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { i18n } from '$lib/i18n/i18n.svelte';

	let { en, zh }: { en: Snippet; zh?: Snippet } = $props();
</script>

{#if i18n.lang === 'zh' && zh}{@render zh()}{:else}{@render en()}{/if}
