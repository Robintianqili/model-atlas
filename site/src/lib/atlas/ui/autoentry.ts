// The one door from /m/[slug]/+page.ts to the auto page (design §5.1). +page.ts imports THIS module lazily, and this module
// imports AutoModelPage lazily: two dynamic-import levels on purpose. SvelteKit links the CSS of a page node's first-level
// dynamic imports into the prerendered HTML of every page of the route (to avoid a flash of unstyled content), so a direct
// import('AutoModelPage.svelte') in +page.ts made all 101 Planned pages link 8 stylesheets they never use. This module
// imports nothing else and has no CSS. On the client, Vite's preload helper still loads AutoModelPage's CSS before the
// component renders; AutoModelPage renders its kit views only after mount, so the prerendered HTML has nothing unstyled.
import type { Component } from 'svelte';

export function loadAutoPage(): Promise<Component<any>> {
	return import('./AutoModelPage.svelte').then((m) => m.default as Component<any>);
}
