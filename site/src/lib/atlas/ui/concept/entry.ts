// The one door from /train/[slug]/+page.ts to the concept page (the concept twin of ../autoentry.ts). +page.ts imports
// THIS module lazily, and this module imports AutoConceptPage lazily: two dynamic-import levels on purpose. SvelteKit links
// the CSS of a page node's first-level dynamic imports into the prerendered HTML of every page of the route, so a direct
// import('AutoConceptPage.svelte') in +page.ts would make every Planned concept page link stylesheets it never uses. This
// module imports nothing else and has no CSS.
import type { Component } from 'svelte';

export function loadConceptPage(): Promise<Component<any>> {
	return import('../AutoConceptPage.svelte').then((m) => m.default as Component<any>);
}
