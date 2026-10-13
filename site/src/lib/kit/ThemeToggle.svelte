<!-- Theme toggle: cycles system → (opposite of the OS theme) → (OS theme) → system, so the first
     click always changes what you see. Persists (try/catch) via theme.svelte.ts. The icon shows the current choice. -->
<script lang="ts">
	import Icon from './Icon.svelte';
	import { contrast, moon, sun } from './icons';
	import { themeState, cycleTheme } from './theme.svelte';
	import { ui } from '$lib/i18n/i18n.svelte';

	const mode = $derived(
		themeState.pref === 'light' ? ui('themeLight') : themeState.pref === 'dark' ? ui('themeDark') : ui('themeSystem')
	);
	const label = $derived(ui('themeToggleLabel', { mode }));
</script>

<button type="button" class="icon-btn" onclick={cycleTheme} aria-label={label} title={label}>
	<Icon d={themeState.pref === 'light' ? sun : themeState.pref === 'dark' ? moon : contrast} />
</button>
