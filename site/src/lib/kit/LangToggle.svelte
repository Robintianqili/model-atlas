<!-- Language toggle (English ⇄ 中文). Persists the choice (try/catch) and updates <html lang>. -->
<script lang="ts">
	import { i18n, toggleLang, ui } from '$lib/i18n/i18n.svelte';
</script>

<button
	type="button"
	class="btn ghost lang"
	onclick={toggleLang}
	aria-label={i18n.lang === 'en' ? 'Switch to Chinese (中文)' : '切换到英文 (English)'}
	lang={i18n.lang === 'en' ? 'zh-CN' : 'en'}
>
	<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6.3" /><path d="M1.8 8h12.4M8 1.7c2 2.2 2 10.4 0 12.6M8 1.7c-2 2.2-2 10.4 0 12.6" /></svg>
	<span>{i18n.lang === 'en' ? ui('switchToZh') : 'EN'}</span>
</button>

<style>
	.lang {
		white-space: nowrap;
		flex: 0 0 auto;
		padding: 0 10px;
		min-height: 36px;
		gap: 6px;
	}
</style>
