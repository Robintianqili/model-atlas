/**
 * Theme state (light / dark / system) and reduced-motion state.
 *
 *   import { themeState, motion } from '$lib/kit/theme.svelte';
 *   themeState.resolved   // 'light' | 'dark' — reactive; canvas components redraw on change
 *   themeState.pref       // 'system' | 'light' | 'dark'
 *   setTheme('dark')
 *   motion.reduced        // true when the OS asks for reduced motion
 *
 * `initThemeAndMotion()` is called once by the root layout on mount.
 */
import type { Theme } from './color';

export type ThemePref = 'system' | Theme;

const KEY = 'atlas.theme';

export const themeState = $state<{ pref: ThemePref; resolved: Theme; systemDark: boolean }>({
	pref: 'system',
	resolved: 'light',
	systemDark: false
});

export const motion = $state<{ reduced: boolean }>({ reduced: false });

function apply() {
	themeState.resolved =
		themeState.pref === 'system' ? (themeState.systemDark ? 'dark' : 'light') : themeState.pref;
	if (typeof document !== 'undefined') {
		const root = document.documentElement;
		if (themeState.pref === 'system') root.removeAttribute('data-theme');
		else root.setAttribute('data-theme', themeState.pref);
	}
}

export function setTheme(pref: ThemePref) {
	themeState.pref = pref;
	try {
		if (pref === 'system') localStorage.removeItem(KEY);
		else localStorage.setItem(KEY, pref);
	} catch {
		/* storage blocked: keep the choice for this page view only */
	}
	apply();
}

/**
 * Cycle through the three choices (used by the header toggle). The order starts with the
 * opposite of the OS theme, so the first click from "system" always changes what you see:
 * OS light: system -> dark -> light -> system; OS dark: system -> light -> dark -> system.
 */
export function cycleTheme() {
	const order: ThemePref[] = themeState.systemDark ? ['light', 'dark', 'system'] : ['dark', 'light', 'system'];
	setTheme(order[(order.indexOf(themeState.pref) + 1) % order.length]);
}

let initialized = false;

export function initThemeAndMotion() {
	if (initialized || typeof window === 'undefined') return;
	initialized = true;
	try {
		const saved = localStorage.getItem(KEY);
		if (saved === 'light' || saved === 'dark') themeState.pref = saved;
	} catch {
		/* ignore */
	}
	try {
		const mq = matchMedia('(prefers-color-scheme: dark)');
		themeState.systemDark = mq.matches;
		mq.addEventListener('change', (e) => {
			themeState.systemDark = e.matches;
			apply();
		});
	} catch {
		/* old browser */
	}
	try {
		const rm = matchMedia('(prefers-reduced-motion: reduce)');
		motion.reduced = rm.matches;
		rm.addEventListener('change', (e) => (motion.reduced = e.matches));
	} catch {
		/* ignore */
	}
	apply();
}
