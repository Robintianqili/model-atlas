<!--
  Language switch (English ⇄ 中文). Persists the choice (try/catch) and updates <html lang>.
    variant 'seg'   an EN | 中文 segmented control (desktop bars)
    variant 'flip'  one button that names the other language (phones)
    variant 'auto'  both, the segment from 760 px up and the flip below (the default)
-->
<script lang="ts">
	import { i18n, setLang, toggleLang, ui } from '$lib/i18n/i18n.svelte';

	let { variant = 'auto' }: { variant?: 'auto' | 'seg' | 'flip' } = $props();
</script>

{#if variant !== 'flip'}
	<div class="seg lang-seg" class:auto={variant === 'auto'} role="group" aria-label={ui('language')}>
		<button type="button" lang="en" aria-pressed={i18n.lang === 'en'} onclick={() => setLang('en')}>EN</button>
		<button type="button" lang="zh-CN" aria-pressed={i18n.lang === 'zh'} onclick={() => setLang('zh')}>中文</button>
	</div>
{/if}
{#if variant !== 'seg'}
	<button
		type="button"
		class="icon-btn flip"
		class:auto={variant === 'auto'}
		onclick={toggleLang}
		aria-label={i18n.lang === 'en' ? '中文 · Switch to Chinese' : 'EN · 切换到英文'}
		lang={i18n.lang === 'en' ? 'zh-CN' : 'en'}
	>
		{i18n.lang === 'en' ? '中文' : 'EN'}
	</button>
{/if}

<style>
	.lang-seg {
		flex: none;
		flex-wrap: nowrap;
	}
	.lang-seg button {
		min-width: 44px;
	}
	.flip {
		width: auto;
		min-width: 44px;
		padding: 0 10px;
		font-size: var(--t-sm);
		font-weight: var(--w-title);
		white-space: nowrap;
		color: var(--ink);
	}
	.flip.auto {
		display: none;
	}
	@media (max-width: 759px) {
		.lang-seg.auto {
			display: none;
		}
		.flip.auto {
			display: inline-grid;
		}
	}
</style>
