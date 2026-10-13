/**
 * Language state and helpers.
 *
 *   import { i18n, t, ui, setLang } from '$lib/i18n/i18n.svelte';
 *   t({ en: 'Layer', zh: '层' })     // reactive: re-renders when the language changes
 *   ui('navAtlas')                   // from the UI dictionary (strings.ts)
 *   ui('stepOf', { i: 2, n: 9 })     // placeholders
 *   lui('nextStep', { k: 3 })        // the lesson app's strings (LESSON_UI); cui('allN', { n: 6 }) the catalog's
 *   setLang('zh')                    // persists (try/catch) and updates <html lang>
 *
 * Prerendered HTML is English. The saved language is applied after hydration by initLang()
 * (called once in the root layout). To avoid a flash of English, app.html marks the page
 * `data-lang-pending="zh"` before first paint (the content stays hidden until initLang has
 * rendered Chinese, with a short failsafe timeout); without JavaScript the English page shows.
 */
import { tick } from 'svelte';
import { pick, type I18n, type Lang, type Text } from './types.ts';
import { CATALOG_UI, LESSON_UI, UI, type CatalogUIKey, type LessonUIKey, type UIKey } from './strings.ts';

export type { I18n, Lang, Text, UIKey, LessonUIKey, CatalogUIKey };

const KEY = 'atlas.lang';

export const i18n = $state<{ lang: Lang }>({ lang: 'en' });

/** Translate a bilingual value for the current language (falls back to English). */
export function t(text: Text | undefined | null): string {
	return pick(text, i18n.lang);
}

/** A label colon in the current language: ':' in English, full-width '：' in Chinese. */
export function colon(): string {
	return i18n.lang === 'zh' ? '：' : ':';
}

/** Look up a UI string and fill `{placeholders}`. */
export function ui(key: UIKey, vars?: Record<string, string | number>): string {
	return fill(UI[key] as I18n, vars);
}

/** A lesson-app string (LESSON_UI: views, Next, Start, Skip, feedback, cover, end card). Bundled only where used. */
export function lui(key: LessonUIKey, vars?: Record<string, string | number>): string {
	return fill(LESSON_UI[key] as I18n, vars);
}

/** A catalog string (CATALOG_UI: cards, filters, paths, areas). Bundled only where used. */
export function cui(key: CatalogUIKey, vars?: Record<string, string | number>): string {
	return fill(CATALOG_UI[key] as I18n, vars);
}

function fill(text: I18n, vars?: Record<string, string | number>): string {
	let s = pick(text, i18n.lang);
	if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
	return s;
}

function applyHtmlLang() {
	if (typeof document === 'undefined') return;
	document.documentElement.lang = i18n.lang === 'zh' ? 'zh-CN' : 'en';
	document.documentElement.removeAttribute('data-lang-pending');
}

export function setLang(lang: Lang) {
	i18n.lang = lang;
	try {
		localStorage.setItem(KEY, lang);
	} catch {
		/* storage blocked */
	}
	applyHtmlLang();
}

export function toggleLang() {
	setLang(i18n.lang === 'en' ? 'zh' : 'en');
}

let initialized = false;

/** Read the saved language (or `?lang=zh` in the URL). Called once on mount by the root layout. */
export function initLang() {
	if (initialized || typeof window === 'undefined') return;
	initialized = true;
	let lang: Lang = 'en';
	try {
		const q = new URLSearchParams(location.search).get('lang');
		if (q === 'zh' || q === 'en') lang = q;
		else if (localStorage.getItem(KEY) === 'zh') lang = 'zh';
	} catch {
		/* ignore */
	}
	i18n.lang = lang;
	// reveal the page only after the Chinese text has been rendered (see app.html)
	tick().then(applyHtmlLang);
}
