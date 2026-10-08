<!-- Theme toggle: cycles system → (opposite of the OS theme) → (OS theme) → system, so the first
     click always changes what you see. Persists (try/catch) via theme.svelte.ts. -->
<script lang="ts">
	import { themeState, cycleTheme } from './theme.svelte';
	import { ui } from '$lib/i18n/i18n.svelte';

	const mode = $derived(
		themeState.pref === 'light' ? ui('themeLight') : themeState.pref === 'dark' ? ui('themeDark') : ui('themeSystem')
	);
</script>

<button type="button" class="btn ghost icon" onclick={cycleTheme} aria-label={ui('themeToggleLabel', { mode })} title={ui('themeToggleLabel', { mode })}>
	<svg viewBox="0 0 16 16" width="17" height="17" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
		{#if themeState.pref === 'light'}
			<circle cx="8" cy="8" r="3" /><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" />
		{:else if themeState.pref === 'dark'}
			<path d="M13.2 10.2A5.6 5.6 0 0 1 5.8 2.8a5.6 5.6 0 1 0 7.4 7.4z" />
		{:else}
			<circle cx="8" cy="8" r="5.6" /><path d="M8 2.4v11.2" /><path d="M8 2.4a5.6 5.6 0 0 1 0 11.2z" fill="currentColor" />
		{/if}
	</svg>
</button>
