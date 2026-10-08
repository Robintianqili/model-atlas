import { existsSync } from 'node:fs';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// BASE_PATH is set for GitHub Pages project sites, e.g. BASE_PATH=/model-atlas npm run build
const base = process.env.BASE_PATH ?? '';

// ATLAS_BUILD_DIR moves the build output (DevKit: lanes and `atlas check` build into .atlas/build-<name>/, so parallel
// builds never touch build/ or each other). Default: build/. scripts/check-build.ts reads the same variable.
const buildDir = process.env.ATLAS_BUILD_DIR || 'build';

// Live pages are static route folders built by the model/concept streams. Until a folder exists,
// links to it are expected 404s during prerender: warn instead of failing the build.
// Live routes that are linked from the catalog before their page exists (none right now).
const PENDING_LIVE_ROUTES = [];

// Package-only build (DevKit K2 speed pack): ATLAS_ONLY=<slug>[,<slug>...] prerenders the home page and those packages'
// pages (/m/<slug>/, /train/<slug>/, /dev/atlas/lab/<slug>/; ATLAS_ONLY_PAGES adds more) instead of every page. The
// prerenderer does not crawl, and a server hook (scripts/atlas/lib/only-hooks.ts) answers the pages that route-level
// entries() still queue with an empty 204, which is neither rendered nor written. A candidate page a package does not
// have (a model has no /train/ page) answers 404 and is skipped quietly. The bundles are the same as a full build's;
// scripts/check-build.ts checks only these pages. Unset: the normal build (every page).
const onlyLib = process.env.ATLAS_ONLY?.trim() ? await import('./scripts/atlas/lib/only.ts') : null;
const onlyEntries = onlyLib?.onlyEntries() ?? null;
const ownHooks = ['ts', 'js'].some((x) => existsSync(`src/hooks.server.${x}`));

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	compilerOptions: {
		runes: true
	},
	kit: {
		adapter: adapter({
			pages: buildDir,
			assets: buildDir,
			// GitHub Pages serves 404.html for unknown paths: an SPA shell that renders +error.svelte
			fallback: '404.html',
			precompress: false,
			strict: true
		}),
		paths: {
			base,
			relative: true
		},
		...(onlyEntries && !ownHooks ? { files: { hooks: { server: 'scripts/atlas/lib/only-hooks' } } } : {}),
		prerender: {
			entries: onlyEntries ?? ['*'],
			crawl: !onlyEntries,
			handleHttpError: ({ path, referrer, message, status }) => {
				if (PENDING_LIVE_ROUTES.some((p) => path.endsWith(p))) {
					console.warn(`[prerender] live page not built yet: ${path} (linked from ${referrer})`);
					return;
				}
				if (onlyEntries && status === 404 && onlyLib?.isCandidate(path, onlyEntries)) return;
				throw new Error(message);
			},
			...(onlyEntries ? { handleUnseenRoutes: 'ignore' } : {}),
			handleMissingId: 'warn'
		}
	}
};

export default config;
