import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

/** Page-shell stability (DevKit integration 1b, LOG.md): the Svelte and SvelteKit client runtimes each go into one
 *  chunk. Rollup otherwise groups runtime helpers by the set of entries that use them, so every new lazy chunk (atlas
 *  views, adapters, packages) that uses a different mix of helpers re-split the chunks /m/tinycnn/ and /m/tinygpt/
 *  preload: 303.0 and 302.3 KB gzip at the 1b merge, over the 300 KB budget. Pinned, every page's shell is smaller
 *  (TinyCNN 296.7 KB, TinyGPT 297.2 KB, planned pages 105.8 KB). Package workers (?worker) are separate builds. */
function manualChunks(id: string): string | undefined {
	const p = id.replace(/\\/g, '/');
	if (/\/node_modules\/svelte\/src\/(internal|index-client|legacy|store|reactivity|attachments|events|motion|transition|easing|animate)/.test(p)) return 'svelte-runtime';
	if (/\/node_modules\/(@sveltejs\/kit|devalue)\//.test(p)) return 'kit-runtime';
	return undefined;
}

export default defineConfig({
	plugins: [sveltekit()],
	build: {
		// Report chunk sizes honestly; the page-shell budget is 300 KB gzipped.
		reportCompressedSize: true,
		chunkSizeWarningLimit: 400,
		rollupOptions: { output: { manualChunks } }
	},
	server: {
		fs: {
			// allow reading ../research etc. is NOT needed; keep defaults strict
			strict: true
		},
		// lane copies (.work/), atlas build output (.atlas/) and build/ hold tens of thousands of files: watching them
		// kept the main tree's dev server busy for minutes before it answered a request (integration 1b)
		watch: { ignored: ['**/.work/**', '**/.atlas/**', '**/build/**'] }
	}
});
