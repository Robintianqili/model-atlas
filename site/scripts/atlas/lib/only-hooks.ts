// The server hook of a package-only build (K2 speed pack; svelte.config.js installs it only when ATLAS_ONLY is set and
// the site has no src/hooks.server file of its own). Route-level `entries()` (every /m/<slug>/, /devkit/... page) are
// always queued by the prerenderer; this hook answers 204 for every page outside the ATLAS_ONLY list, so those pages are
// neither rendered nor written (SvelteKit saves only 200 responses, and a 2xx is no HTTP error). The pages on the list
// (and the 404.html fallback) render as usual.
import type { Handle } from '@sveltejs/kit';
import { onlyAllows, onlyEntries } from './only.ts';

const entries = onlyEntries() ?? [];
const base = (process.env.BASE_PATH ?? '').replace(/\/+$/, '');

export const handle: Handle = async ({ event, resolve }) => {
	if (!entries.length) return resolve(event);
	const path = base && event.url.pathname.startsWith(base) ? event.url.pathname.slice(base.length) || '/' : event.url.pathname;
	if (onlyAllows(path, entries)) return resolve(event);
	return new Response(null, { status: 204 });
};
